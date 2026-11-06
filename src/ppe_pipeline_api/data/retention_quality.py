"""Read-only, explicitly requested coverage inspection; never changes sampling.

Class IDs are reported per source because different weeks may have different
taxonomies. Missing/unreadable labels are unknown, not zero-class examples.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import math

from botocore.exceptions import ClientError

from .. import pull_control as pc, storage
from . import retention as rt


def _classes(bucket: str, image: str) -> tuple[set, str]:
    key = storage.label_path_for(image)
    if not key:
        return set(), "missing"
    try:
        text = pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")
    except ClientError as exc:
        return set(), "missing" if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404") else "unreadable"
    except Exception:
        return set(), "unreadable"
    classes = set()
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            parts = line.split()
            class_id = int(parts[0])
            coords = [float(v) for v in parts[1:]]
            if len(coords) != 4 or class_id < 0 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in coords):
                return set(), "invalid"
            if coords[2] <= 0 or coords[3] <= 0:
                return set(), "invalid"
            classes.add(class_id)
        except (ValueError, IndexError):
            return set(), "invalid"
    return classes, "ok"


def inspect(*, root: str, config: dict) -> dict:
    inventory = rt.read_inventory(root, week_prefix=config.get("week_prefix") or "week_")
    planned = rt.plan_inventory(inventory, config)
    selected = {r["week"]: set(r["stems"]) for r in planned["weeks"]}
    bucket = storage.BUCKET or pc.destination_bucket()
    sources = [(w["name"], bucket, w["train"], selected[w["name"]]) for w in inventory
               if w["name"] not in set(config.get("excluded_weeks") or [])]
    golden_count = 0
    if config.get("use_golden") and config.get("golden_root"):
        gb = config.get("golden_bucket") or bucket
        base = config["golden_root"].strip("/") + "/images/"
        objects = list(pc._list_objects(base, gb))
        images = [o["Key"] for o in objects if o["Key"].lower().endswith(storage.IMAGE_EXTENSIONS)]
        split = any(k.startswith(base + "train/") or k.startswith(base + "val/") for k in images)
        keys = {k: k for k in images if not split or k.startswith(base + "train/")}
        golden_count = len(keys)
        sources.append(("golden", gb, keys, set(keys)))
    threshold = max(1, int(config.get("min_class_images") or 5))
    rows, issues = [], []
    complete = True
    for source, source_bucket, images, keep in sources:
        before, after = Counter(), Counter()
        missing, unreadable, invalid, empty = 0, 0, 0, 0
        identities = sorted(images)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = pool.map(lambda ident: _classes(source_bucket, images[ident]), identities)
            for ident, (classes, status) in zip(identities, results):
                missing += status == "missing"
                unreadable += status == "unreadable"
                invalid += status == "invalid"
                empty += status == "ok" and not classes
                if status != "ok":
                    continue
                before.update(classes)
                if ident in keep:
                    after.update(classes)
        classes = [{"class_id": c, "before": before[c], "after": after[c],
                    "low": after[c] < threshold, "lost": after[c] == 0} for c in sorted(before)]
        if missing or unreadable or invalid:
            complete = False
            issues.append(f"{source}: {missing} missing, {unreadable} unreadable and {invalid} invalid labels. Coverage is incomplete.")
        if any(c["lost"] for c in classes):
            issues.append(f"{source}: at least one observed class has no selected training images.")
        elif any(c["low"] for c in classes):
            issues.append(f"{source}: at least one observed class has fewer than {threshold} selected images.")
        rows.append({"source": source, "available": len(images), "selected": len(keep),
                     "missing_labels": missing, "unreadable_labels": unreadable,
                     "invalid_labels": invalid, "empty_labels": empty, "classes": classes})
    mix = rt.composition(planned["weeks"], config, golden_count)
    if mix["historical"] > mix["recent"] and mix["recent"]:
        issues.append("Historical selections outnumber recent selections. Decay does not guarantee that recent data dominates.")
    return {"root": root, "complete": complete, "min_class_images": threshold,
            "sources": rows, "composition": mix, "issues": issues,
            "note": "Counts are images containing each source class ID before remapping. "
                    "This does not check scene leakage, content duplicates or final detector filtering. "
                    "A fixed test benchmark must remain separate from checkpoint selection."}
