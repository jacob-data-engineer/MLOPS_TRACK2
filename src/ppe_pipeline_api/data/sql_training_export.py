"""Explicit detector/classifier exports from the SQL page's reviewed rows."""
from __future__ import annotations

import hashlib
import io
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from botocore.exceptions import ClientError

import yaml
from PIL import Image

from . import sql_explorer as sx

DETECTOR_NAMES = list(sx.LABEL_NAMES.values())
CLASSIFIER_NAMES = ["helmet_worn", "no_helmet"]


def exists(bucket, key):
    try:
        sx.storage.s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as exc:
        if str(exc.response.get("Error", {}).get("Code")) in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def validate_names(mode, names):
    required = CLASSIFIER_NAMES if mode == "classification" else ["helmet", "person"]
    if mode not in ("detection", "classification"):
        raise ValueError("Choose detection or classification export.")
    if (not names or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in names)
            or not set(required).issubset(names)
            or (mode == "classification" and set(names) != set(required))):
        raise ValueError("Class names must be unique; retain " + ", ".join(required) + ".")
    return list(names)


def valid_box(raw, width, height):
    box = sx._box(raw)
    if box is None or not all(math.isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = box
    box = [max(0, round(x1)), max(0, round(y1)), min(width, round(x2)), min(height, round(y2))]
    return box if box[2] > box[0] and box[3] > box[1] else None


def samples(row, mode, names):
    """Return eligible samples and counts, without inventing per-person answers."""
    people = row.get("people") or []
    if isinstance(people, str):
        people = json.loads(people)
    width, height = int(row.get("width") or 0), int(row.get("height") or 0)
    if width <= 0 or height <= 0 or not people:
        return [], {"missing_frame_geometry": 1}
    if mode == "classification":
        kept, reasons = [], {}
        for p in people:
            verdict = p.get("verdict")
            box = valid_box(p.get("person_box"), width, height)
            reason = ("unresolved_person_verdict" if verdict not in ("violation", "no_violation")
                      else "invalid_person_box" if not box else "")
            if reason:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue
            label = "no_helmet" if verdict == "violation" else "helmet_worn"
            kept.append({"crop": box, "class_name": label, "class_id": names.index(label),
                         "candidate": p.get("candidate_id"), "verdict": verdict})
        return kept, reasons
    if any(not valid_box(p.get("person_box"), width, height) for p in people):
        return [], {"invalid_person_box": 1}
    wearers = [p for p in people if p.get("wearing")]
    for p in wearers:
        helmet = row.get("drawn_box") if len(wearers) == 1 and sx._box(row.get("drawn_box")) else p.get("auto")
        if not valid_box(helmet, width, height):
            return [], {"missing_usable_helmet_box": 1}
    labels = sx.frame_labels(row)
    if not labels["complete"]:
        return [], {"incomplete_labels": 1}
    labels = {**labels, "boxes": [{**b, "cls": names.index(b["name"])} for b in labels["boxes"]]}
    return [{"labels": sx.yolo_text(labels)}], {}


def export(key, sql, *, bucket="", folder="", dest_bucket, dest_folder,
           train_percent=80, mode="detection", names=None, progress=None):
    names = validate_names(mode, names if names is not None else (CLASSIFIER_NAMES if mode == "classification" else DETECTOR_NAMES))
    dest_folder = dest_folder.strip().strip("/")
    if not dest_bucket.strip() or not dest_folder or any(dest_folder == p or dest_folder.startswith(p + "/") for p in sx.PROTECTED):
        raise ValueError("Choose an unprotected destination bucket and folder.")
    if not 0 <= train_percent <= 100:
        raise ValueError("Train percentage must be between 0 and 100.")
    emit = progress or (lambda event: None)
    emit({"phase": "querying", "completed": 0, "total": 0})
    result = sx.run(key, sql, bucket=bucket, folder=folder)
    if result.get("truncated"):
        raise ValueError("Query results were truncated. Narrow the selection before export.")
    if mode == "classification" and "hazard_type" not in result["columns"]:
        raise ValueError("Reload the no-helmet selection before exporting person crops.")
    plan, reasons, excluded, seen, subjects = [], {}, 0, set(), {}
    for image in result["images"]:
        row = dict(zip(result["columns"], result["rows"][image["row"]]))
        if mode == "classification" and row.get("hazard_type") != "no_helmet":
            raise ValueError("Helmet classification requires the no-helmet class.")
        if not image.get("bucket"):
            raise ValueError("Every source image must have a bucket.")
        items, missing = samples(row, mode, names)
        if not items:
            excluded += 1
        for reason, count in missing.items():
            reasons[reason] = reasons.get(reason, 0) + count
        for item in items:
            identity = hashlib.sha256(json.dumps([image["uri"], item], sort_keys=True).encode()).hexdigest()
            subject = json.dumps([image["uri"], item.get("candidate") or item.get("crop") if mode == "classification" else "frame"], sort_keys=True)
            if subject in subjects and subjects[subject] != identity:
                raise ValueError("Conflicting labels for the same source sample; resolve the reviews before export.")
            subjects[subject] = identity
            if identity in seen:
                continue
            seen.add(identity)
            # All person crops from the same frame share its split.
            side = sx.split_for(image["uri"], train_percent)
            relative = (f"{side}/{item['class_id']:02d}_{item['class_name']}/{identity}.jpg" if mode == "classification"
                        else f"images/{side}/{identity}.jpg")
            plan.append((image, row, item, side, f"{dest_folder}/{relative}", identity))
    if not plan:
        raise ValueError("No eligible samples: " + json.dumps(reasons))
    if len(plan) > sx.PULL_LIMIT:
        raise ValueError(f"Export exceeds {sx.PULL_LIMIT} samples; narrow the selection.")
    s3 = sx.storage.s3
    contract = {"format_version": 1, "mode": mode, "names": names, "train_percent": train_percent}
    contract_key = f"{dest_folder}/export-contract.json"
    lock = f"{dest_folder}/.export-lock"
    try:
        s3.put_object(Bucket=dest_bucket, Key=lock, Body=b"SQL export", IfNoneMatch="*")
    except Exception as exc:
        raise ValueError("Could not lock destination; another export may be running. " + str(exc)) from exc
    try:
        if exists(dest_bucket, contract_key):
            prior = json.loads(s3.get_object(Bucket=dest_bucket, Key=contract_key)["Body"].read())
            if prior != contract:
                raise ValueError("Destination has a different class mapping, export type or split. Choose a new folder.")
        else:
            existing = s3.list_objects_v2(Bucket=dest_bucket, Prefix=dest_folder + "/", MaxKeys=3).get("Contents", [])
            if any(o["Key"] not in (lock, dest_folder + "/") for o in existing):
                raise ValueError("Choose a new empty folder for this export format; existing labels will not be renumbered in place.")
            s3.put_object(Bucket=dest_bucket, Key=contract_key, Body=json.dumps(contract).encode(), ContentType="application/json")
        index_key = f"{dest_folder}/sample-index.json"
        index = json.loads(s3.get_object(Bucket=dest_bucket, Key=index_key)["Body"].read()) if exists(dest_bucket, index_key) else {}
        if any(subject in index and index[subject] != fingerprint for subject, fingerprint in subjects.items()):
            raise ValueError("A previously exported sample has changed labels or crop geometry. Choose a new folder to avoid mixing old and new answers.")
        index.update(subjects)
        s3.put_object(Bucket=dest_bucket, Key=index_key, Body=json.dumps(index).encode(), ContentType="application/json")
        counters = {"completed": 0, "total": len(plan), "copied": 0, "skipped": 0, "failed": 0,
                    "left_out_incomplete": excluded}
        errors, records, sides = [], [], {"train": 0, "val": 0}
        def copy(entry):
            image, row, item, side, target, identity = entry
            try:
                already = exists(dest_bucket, target)
                if mode == "classification":
                    if not already:
                        raw = s3.get_object(Bucket=image["bucket"], Key=image["key"])["Body"].read()
                        with Image.open(io.BytesIO(raw)) as im:
                            if im.size != (int(row["width"]), int(row["height"])):
                                raise ValueError("Recorded frame dimensions do not match the image.")
                            out = io.BytesIO()
                            im.convert("RGB").crop(item["crop"]).save(out, format="JPEG", quality=95)
                        s3.put_object(Bucket=dest_bucket, Key=target, Body=out.getvalue(), ContentType="image/jpeg")
                else:
                    s3.put_object(Bucket=dest_bucket, Key=f"{dest_folder}/labels/{side}/{identity}.txt",
                                  Body=item["labels"].encode(), ContentType="text/plain")
                    if not already:
                        s3.copy_object(Bucket=dest_bucket, Key=target, CopySource={"Bucket": image["bucket"], "Key": image["key"]})
                return already, side, {"target": target, "source": image["uri"], "case_key": row.get("case_key"), **item}, None
            except Exception as exc:
                return False, side, None, f"{image['uri']}: {exc}"
        emit({"phase": "copying", **counters})
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(copy, entry) for entry in plan]
            for future in as_completed(futures):
                already, side, record, error = future.result()
                if error:
                    counters["failed"] += 1
                    errors.append(error)
                else:
                    counters["skipped" if already else "copied"] += 1
                    sides[side] += 1
                    records.append(record)
                counters["completed"] += 1
                emit({"phase": "copying", **counters})
        emit({"phase": "finalizing", **counters})
        config = {"path": ".", "train": "train" if mode == "classification" else "images/train",
                  "val": "val" if mode == "classification" else "images/val", "nc": len(names),
                  "names": dict(enumerate(names))}
        s3.put_object(Bucket=dest_bucket, Key=f"{dest_folder}/data.yaml", Body=yaml.safe_dump(config, sort_keys=False).encode(), ContentType="text/yaml")
        receipt = {"database": key, "sql": sql, "contract": contract, "samples": records,
                   "excluded_reasons": reasons, "errors": errors}
        # Immutable per-run provenance; repeated exports retain previous receipts.
        import uuid
        s3.put_object(Bucket=dest_bucket, Key=f"{dest_folder}/exports/{uuid.uuid4().hex}.json",
                      Body=json.dumps(receipt).encode(), ContentType="application/json")
        return {"bucket": dest_bucket, "folder": dest_folder, "found": len(plan), **counters, **sides,
                "with_label": len(records), "bare_paths_left_out": 0, "labels_from": "database",
                "errors": errors[:10], "excluded_reasons": reasons, "export_mode": mode}
    finally:
        s3.delete_object(Bucket=dest_bucket, Key=lock)
