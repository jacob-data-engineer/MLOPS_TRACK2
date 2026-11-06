"""Split a curated folder into train and val, images and labels together.

The folder is one you have already decided on -- a golden set, a pile of
edge cases, whatever you have curated by hand. Nothing here decides what
belongs in it. All it does is divide it, copy it where you ask, and be
honest about what it did.

Two rules run through the whole module.

**An image and its label are one thing.** The split is chosen per STEM,
never per file, so `a.jpg` and `a.txt` always land on the same side.
Splitting them independently produces images whose labels are in the
other split -- which trains as an unannotated background example and
evaluates as a missing detection, both silently.

**The same input gives the same split.** Membership comes from a hash of
the stem, not from `random`, so re-running reproduces the division
exactly. A set that reshuffles itself on every build cannot be compared
with the run before it, which is the entire point of curating one.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import functools

from botocore.exceptions import ClientError

from .. import pull_control as pc

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
LABEL_EXTENSIONS = (".txt",)


def _rank(stem: str) -> str:
    """A stable position for a stem, independent of listing order.

    Sorting by this and taking the first N makes the split deterministic:
    the same folder at the same percentage always yields the same
    division, so a build can be repeated and an old one checked.
    """
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()


def read_folder(source: str, *, bucket: str = "",
                images_subpath: str = "images",
                labels_subpath: str = "labels",
                labels_source: str = "") -> dict:
    """What is in a curated folder, paired up by stem.

    Accepts both shapes: `<source>/images/` + `<source>/labels/`, and a
    flat folder holding images and labels side by side. Which one it
    found is reported, because guessing wrong about the layout is the
    difference between 400 pairs and none.

    `labels_source` is a THIRD shape: images under one prefix and labels
    under another, with no parent in common. Both other shapes work by
    listing one prefix and sorting what falls out by extension, which
    cannot reach a labels folder that is somebody else's sibling.

    Left empty it changes nothing -- the single listing below is still
    what runs, and the two original shapes behave exactly as before.
    """
    source = (source or "").strip().strip("/")
    if not source:
        raise ValueError("Pick the folder to split.")
    bucket = bucket or pc.destination_bucket()
    images_subpath = (images_subpath or "images").strip("/")
    labels_subpath = (labels_subpath or "labels").strip("/")

    labels_source = (labels_source or "").strip().strip("/")

    images: dict = {}
    labels: dict = {}

    def collect(prefix: str, *, want_images: bool, want_labels: bool) -> None:
        for obj in pc._list_objects(prefix + "/", bucket):
            key = obj["Key"]
            rest = key[len(prefix) + 1:]
            # Identity is the path within the images/labels tree, not
            # just its basename: camera_a/frame and camera_b/frame differ.
            relative = rest
            for sub in (images_subpath, labels_subpath):
                if not labels_source and relative.startswith(sub + "/"):
                    relative = relative[len(sub) + 1:]
                    break
            stem, _, ext = relative.rpartition(".")
            if not stem:
                continue
            ext = "." + ext.lower()
            if want_images and ext in IMAGE_EXTENSIONS:
                if stem in images:
                    raise ValueError(f"Ambiguous image pair {stem}: {images[stem]['key']} and {key}.")
                images[stem] = {"key": key, "size": obj.get("Size", 0), "rest": rest}
            elif want_labels and ext in LABEL_EXTENSIONS:
                if stem in labels:
                    raise ValueError(f"Ambiguous label pair {stem}: {labels[stem]['key']} and {key}.")
                labels[stem] = {"key": key, "size": obj.get("Size", 0), "rest": rest}
            else:
                continue
            entry = images.get(stem) if ext in IMAGE_EXTENSIONS else labels.get(stem)
            if entry is not None:
                entry["etag"] = str(obj.get("ETag") or "").strip('"')

    if labels_source:
        # Two listings, each taking only what it was asked for. Reading
        # labels out of the images prefix as well would pair a stray
        # .txt sitting beside the images against the one the labels
        # folder holds, and which won is whichever listed last.
        collect(source, want_images=True, want_labels=False)
        collect(labels_source, want_images=False, want_labels=True)
    else:
        collect(source, want_images=True, want_labels=True)

    nested = any(f"/{images_subpath}/" in f"/{v['rest']}" for v in images.values())
    paired = sorted(set(images) & set(labels), key=_rank)
    return {
        "source": source,
        "labels_source": labels_source,
        "bucket": bucket,
        "layout": ("two folders" if labels_source
                   else "images/ + labels/" if nested else "flat"),
        "images": len(images),
        "labels": len(labels),
        "paired": paired,
        # An image with no label is a legitimate background example; a
        # label with no image is always a mistake. Reported apart because
        # only one of them is a problem.
        "images_without_labels": sorted(set(images) - set(labels))[:50],
        "labels_without_images": sorted(set(labels) - set(images))[:50],
        "unpaired_images": len(set(images) - set(labels)),
        "orphan_labels": len(set(labels) - set(images)),
        "_image_keys": images,
        "_label_keys": labels,
    }


def split_stems(paired: list, train_percent: int) -> tuple:
    """`(train, val)` stems, deterministic and pair-safe.

    The percentage is of PAIRS, not of files -- an image and its label
    are one unit, and counting them separately would make a "20% val"
    mean ten percent of the pairs with their labels somewhere else.
    """
    train_percent = max(0, min(int(train_percent), 100))
    ordered = sorted(paired, key=_rank)
    cut = round(len(ordered) * train_percent / 100)
    return ordered[:cut], ordered[cut:]


def _previous(bucket: str, target: str) -> dict | None:
    """Read only when a manifest exists; access errors must not reset a split."""
    key = f"{target}/split.json"
    if not target or not any(o["Key"] == key for o in pc._list_objects(key, bucket)):
        return None
    manifest = json.loads(pc.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    if not isinstance(manifest, dict) or not all(isinstance(manifest.get(s), list) for s in ("train", "val")):
        raise ValueError("The existing golden split manifest is invalid; choose a new target folder.")
    if set(manifest["train"]) & set(manifest["val"]):
        raise ValueError("The existing golden split overlaps train and validation; choose a new target folder.")
    return manifest


def _selection(found: dict, target: str, bucket: str, split: bool,
               percent: int, images_subpath: str, labels_subpath: str) -> tuple:
    previous = _previous(bucket, target)
    paired = set(found["paired"])
    if previous is None:
        train, val = split_stems(list(paired), percent) if split else (sorted(paired), [])
        return train, val, None
    expected = {
        "source": f"s3://{found['bucket']}/{found['source']}",
        "split": split, "train_percent": percent if split else 100,
        "labels_source": found["labels_source"],
        "images_subpath": images_subpath.strip("/"),
        "labels_subpath": labels_subpath.strip("/"),
    }
    if any(previous.get(k, v) != v for k, v in expected.items()):
        raise ValueError("This target already has a frozen split with different settings. "
                         "Choose a new target folder to create a new benchmark version.")
    # The validation set is fixed, not merely deterministic for one input.
    # New examples go to training; deleted validation pairs must be resolved
    # explicitly rather than making successive scores describe different sets.
    val = set(previous["val"])
    if val - paired:
        raise ValueError(f"{len(val - paired)} frozen validation pair(s) are missing from the source. "
                         "Restore them or choose a new target folder for a new benchmark.")
    for stem, signatures in (previous.get("validation_objects") or {}).items():
        for kind, old in signatures.items():
            current = found[f"_{kind}_keys"].get(stem, {})
            if old.get("etag") and current.get("etag") != old["etag"]:
                raise ValueError(f"Frozen validation {kind} changed: {stem}. "
                                 "Choose a new target folder for a new benchmark version.")
    return sorted(paired - val, key=_rank), sorted(val, key=_rank), previous


def plan(*, source: str, target_folder: str, bucket: str = "",
         target_bucket: str = "", split: bool = True, train_percent: int = 80,
         images_subpath: str = "images", labels_subpath: str = "labels",
         labels_source: str = "", _found: dict | None = None) -> dict:
    """What the build would write. Copies nothing."""
    found = _found if _found is not None else read_folder(source, bucket=bucket, images_subpath=images_subpath,
                        labels_subpath=labels_subpath,
                        labels_source=labels_source)
    target_folder = (target_folder or "").strip().strip("/")
    target_bucket = (target_bucket or "").strip() or pc.destination_bucket()

    paired = found["paired"]
    train, val, previous = _selection(found, target_folder, target_bucket, split,
                                      train_percent, images_subpath, labels_subpath)

    def where(stem: str, side: str, kind: str) -> str:
        sub = images_subpath if kind == "image" else labels_subpath
        tail = f"/{side}" if split else ""
        extension = found[f"_{kind}_keys"][stem]["key"].rsplit(".", 1)[-1]
        name = f"{stem}.{extension}"
        return f"{target_folder}/{sub}{tail}/{name}"

    problems = []
    if not target_folder:
        problems.append({"level": "blocking", "text": "Give the target folder a name."})
    if not paired:
        problems.append({
            "level": "blocking",
            "text": f"No image in {source} has a label beside it, so there are no "
                    f"pairs to split. Found {found['images']} images and "
                    f"{found['labels']} labels.",
        })
    if found["orphan_labels"]:
        problems.append({
            "level": "warn",
            "text": f"{found['orphan_labels']} label(s) have no image and are ignored. "
                    f"A label without its image is always a mistake.",
        })
    if found["unpaired_images"]:
        problems.append({
            "level": "warn",
            "text": f"{found['unpaired_images']} image(s) have no label and are "
                    f"ignored. Copy them deliberately if they are meant as "
                    f"background examples.",
        })
    if split and (not train or not val):
        problems.append({
            "level": "warn",
            "text": f"The slider puts every pair on one side ({len(train)} train, "
                    f"{len(val)} val). One side being empty is the same as the "
                    f"split being off.",
        })
    if target_bucket == found["bucket"] and target_folder == source:
        problems.append({
            "level": "blocking",
            "text": "The target is the source. That would copy the folder over itself.",
        })

    if target_bucket == found["bucket"]:
        # An ancestor target's cleanup could delete the source. A nested
        # target would be ingested on the next source scan.
        for origin in (found["source"], found["labels_source"]):
            if origin and target_folder and (target_folder.startswith(origin + "/")
                                            or origin.startswith(target_folder + "/")):
                problems.append({"level": "blocking", "text":
                                 "Source and target folders must not contain one another."})

    sample = [
        {"stem": stem, "side": side,
         "image": where(stem, side, "image"), "label": where(stem, side, "label")}
        for side, stems in (("train", train[:3]), ("val", val[:3]))
        for stem in stems
    ]
    return {
        "source": source,
        "bucket": found["bucket"],
        "target_bucket": target_bucket,
        "target_folder": target_folder,
        "layout": found["layout"],
        "split": split,
        "train_percent": train_percent if split else 100,
        "pairs": len(paired),
        "train": len(train),
        "val": len(val),
        "images": found["images"],
        "labels": found["labels"],
        "unpaired_images": found["unpaired_images"],
        "orphan_labels": found["orphan_labels"],
        "files_to_copy": (len(train) + len(val)) * 2,
        "sample": sample,
        "problems": problems,
        "ready": not any(p["level"] == "blocking" for p in problems),
        "membership_locked": previous is not None,
        "benchmark_id": (previous or {}).get("benchmark_id", ""),
        "new_training_pairs": len(set(train) - set((previous or {}).get("train", []))) if previous else 0,
    }


def _exclusive_build(fn):
    """One writer per target across web workers/schedules, including Dynamo mode.

    A crashed writer leaves the lock deliberately: an operator must verify the
    writer has stopped before removing .split-build-lock.json and retrying.
    Automatically expiring a lock could let a slow copy race its replacement.
    """
    @functools.wraps(fn)
    def locked(**kwargs):
        target = (kwargs.get("target_folder") or "").strip().strip("/")
        if not target:
            raise ValueError("Give the target folder a name.")
        bucket = kwargs.get("target_bucket") or pc.destination_bucket()
        key = f"{target}/.split-build-lock.json"
        try:
            pc.s3.put_object(Bucket=bucket, Key=key, IfNoneMatch="*",
                             Body=json.dumps({"started_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}).encode(),
                             ContentType="application/json")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
                raise ValueError("Another golden build owns this target. If its process stopped, "
                                 "verify that before removing .split-build-lock.json and retrying.") from exc
            raise
        try:
            return fn(**kwargs)
        finally:
            response = pc.s3.delete_objects(Bucket=bucket, Delete={"Objects": [{"Key": key}], "Quiet": True})
            if response.get("Errors"):
                raise ValueError("The golden build lock could not be released; inspect .split-build-lock.json.")
    return locked


@_exclusive_build
def build(*, source: str, target_folder: str, bucket: str = "",
          target_bucket: str = "", split: bool = True, train_percent: int = 80,
          images_subpath: str = "images", labels_subpath: str = "labels",
          labels_source: str = "") -> dict:
    """Copies the split into the target. The one call that writes.

    A copy, never a move: the source is something you curated by hand,
    and a build that emptied it would destroy the only record of that
    work if the target turned out wrong.
    """
    found = read_folder(source, bucket=bucket, images_subpath=images_subpath,
                        labels_subpath=labels_subpath,
                        labels_source=labels_source)
    outline = plan(source=source, target_folder=target_folder, bucket=bucket,
                   labels_source=labels_source,
                   target_bucket=target_bucket, split=split,
                   train_percent=train_percent, images_subpath=images_subpath,
                   labels_subpath=labels_subpath, _found=found)
    if not outline["ready"]:
        blocking = "; ".join(p["text"] for p in outline["problems"]
                             if p["level"] == "blocking")
        raise ValueError(blocking)

    target_folder = outline["target_folder"]
    target_bucket = outline["target_bucket"]
    errors: list = []
    paired = found["paired"]
    train, val, previous = _selection(found, target_folder, target_bucket, split,
                                      train_percent, images_subpath, labels_subpath)

    # Persist the initial assignment before copying. An interrupted first
    # build must retry the same validation set even if the source grows.
    if previous is None:
        reservation = {"source": f"s3://{found['bucket']}/{found['source']}",
                       "split": split, "train_percent": outline["train_percent"],
                       "train": train, "val": val, "state": "building",
                       "labels_source": found["labels_source"],
                       "images_subpath": images_subpath.strip("/"),
                       "labels_subpath": labels_subpath.strip("/"),
                       "validation_objects": {s: {kind: {"etag": found[f"_{kind}_keys"][s].get("etag", "")}
                                                  for kind in ("image", "label")} for s in val}}
        pc.s3.put_object(Bucket=target_bucket, Key=f"{target_folder}/split.json",
                         Body=json.dumps(reservation).encode(), ContentType="application/json", IfNoneMatch="*")

    # EVERY key this build is about to write, worked out before anything
    # is copied, because it is also the list of what may stay.
    wanted: dict = {}
    for side, stems in (("train", train), ("val", val)):
        for stem in stems:
            for kind, sub in (("image", images_subpath), ("label", labels_subpath)):
                entry = found[f"_{kind}_keys"][stem]
                name = f"{stem}.{entry['key'].rsplit('.', 1)[-1]}"
                tail = f"/{side}" if split else ""
                wanted[f"{target_folder}/{sub.strip('/')}{tail}/{name}"] = entry

    # THE TARGET IS MADE TO MATCH THE PLAN, not added to.
    #
    # This used to copy and never clear, and the target is derived from a
    # split that changes: re-running an 80/20 build over a folder left by
    # a 5/95 one put 80 stems in `images/train` and left all 95 of the
    # old ones in `images/val`. Seventy-five images were then in BOTH --
    # trained on, and then scored on. The held-out number would have been
    # inflated and nothing on the run would have said so.
    #
    # The same clear-out removes stems deleted from the source since the
    # last build, which would otherwise keep being trained on.
    #
    # Scoped to the target's OWN images/ and labels/ subfolders and to
    # nothing else: the source is never touched -- it is the curated
    # thing and the whole reason this copies rather than moves -- and a
    # key outside those two subfolders is not ours to remove.
    removed = 0
    stale: list = []
    for sub in {images_subpath.strip("/"), labels_subpath.strip("/")}:
        prefix = f"{target_folder}/{sub}/"
        for obj in pc._list_objects(prefix, target_bucket):
            key = obj["Key"]
            if key.endswith("/") or key in wanted:
                continue
            stale.append({"Key": key})
    for start in range(0, len(stale), 1000):
        batch = stale[start:start + 1000]
        try:
            deleted = pc.s3.delete_objects(Bucket=target_bucket,
                                          Delete={"Objects": batch, "Quiet": True})
            errors.extend({"key": e.get("Key", ""), "error": e.get("Message", "Delete failed")}
                          for e in deleted.get("Errors", []))
            removed += len(batch)
        except Exception as exc:                               # noqa: BLE001
            errors.append({"key": batch[0]["Key"], "error": str(exc)})

    if errors:
        raise ValueError(f"Golden-set cleanup failed; build stopped: {errors[0]['error']}")
    copied = 0
    for destination, entry in wanted.items():
        try:
            pc.s3.copy_object(
                Bucket=target_bucket, Key=destination,
                CopySource={"Bucket": found["bucket"], "Key": entry["key"]},
                **({"CopySourceIfMatch": entry["etag"]} if entry.get("etag") else {}),
            )
            copied += 1
        except Exception as exc:                               # noqa: BLE001
            errors.append({"key": entry["key"], "error": str(exc)})

    # Beside the data, so the division is recoverable without re-deriving
    # it: the percentage alone does not say which stems fell where once
    # the source has grown.
    if errors:
        raise ValueError(f"Golden-set copying failed; no successful split recorded: {errors[0]['error']}")
    validation_objects = {stem: {kind: {"key": found[f"_{kind}_keys"][stem]["key"],
                                      "etag": found[f"_{kind}_keys"][stem].get("etag", "")}
                                for kind in ("image", "label")} for stem in val}
    benchmark_id = hashlib.sha256(json.dumps(
        [found["bucket"], found["source"], validation_objects], sort_keys=True).encode()).hexdigest()[:20]
    manifest = {
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": f"s3://{found['bucket']}/{found['source']}",
        "split": split,
        "train_percent": outline["train_percent"],
        "train": train,
        "val": val,
        "schema_version": 2,
        "state": "ready",
        "labels_source": labels_source.strip("/"),
        "images_subpath": images_subpath.strip("/"),
        "labels_subpath": labels_subpath.strip("/"),
        "benchmark_id": benchmark_id,
        "validation_objects": validation_objects,
    }
    # Keep an immutable record of each successful membership/content revision.
    revision = hashlib.sha256(json.dumps({k: v for k, v in manifest.items() if k != "built_at"},
                                        sort_keys=True).encode()).hexdigest()[:24]
    history_key = f"{target_folder}/split-versions/{revision}.json"
    try:
        pc.s3.put_object(Bucket=target_bucket, Key=history_key,
                         Body=json.dumps(manifest, indent=2).encode("utf-8"),
                         ContentType="application/json", IfNoneMatch="*")
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("PreconditionFailed", "412"):
            raise
    manifest_key = f"{target_folder}/split.json"
    try:
        pc.s3.put_object(Bucket=target_bucket, Key=manifest_key,
                         Body=json.dumps(manifest, indent=2).encode("utf-8"),
                         ContentType="application/json")
    except Exception as exc:
        raise ValueError("Golden split could not be recorded; retry before training.") from exc

    return {**outline, "copied": copied, "removed": removed, "errors": errors,
            "benchmark_id": benchmark_id,
            "split_version": f"s3://{target_bucket}/{history_key}",
            "manifest": f"s3://{target_bucket}/{manifest_key}" if manifest_key else "",
            "uri": f"s3://{target_bucket}/{target_folder}/"}
