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

from . import pull_control as pc

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
            name = key.rsplit("/", 1)[-1]
            stem, _, ext = name.rpartition(".")
            if not stem:
                continue
            ext = "." + ext.lower()
            if want_images and ext in IMAGE_EXTENSIONS:
                images[stem] = {"key": key, "size": obj.get("Size", 0), "rest": rest}
            elif want_labels and ext in LABEL_EXTENSIONS:
                labels[stem] = {"key": key, "size": obj.get("Size", 0), "rest": rest}

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


def plan(*, source: str, target_folder: str, bucket: str = "",
         target_bucket: str = "", split: bool = True, train_percent: int = 80,
         images_subpath: str = "images", labels_subpath: str = "labels",
         labels_source: str = "") -> dict:
    """What the build would write. Copies nothing."""
    found = read_folder(source, bucket=bucket, images_subpath=images_subpath,
                        labels_subpath=labels_subpath,
                        labels_source=labels_source)
    target_folder = (target_folder or "").strip().strip("/")
    target_bucket = (target_bucket or "").strip() or pc.destination_bucket()

    paired = found["paired"]
    if split:
        train, val = split_stems(paired, train_percent)
    else:
        # Off: the folder is copied whole, with no train/val level. What
        # you curated is what lands.
        train, val = paired, []

    def where(stem: str, side: str, kind: str) -> str:
        sub = images_subpath if kind == "image" else labels_subpath
        tail = f"/{side}" if split else ""
        name = found[f"_{kind}_keys"][stem]["key"].rsplit("/", 1)[-1]
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
    }


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
                   labels_subpath=labels_subpath)
    if not outline["ready"]:
        blocking = "; ".join(p["text"] for p in outline["problems"]
                             if p["level"] == "blocking")
        raise ValueError(blocking)

    target_folder = outline["target_folder"]
    target_bucket = outline["target_bucket"]
    errors: list = []
    paired = found["paired"]
    train, val = (split_stems(paired, train_percent) if split else (paired, []))

    # EVERY key this build is about to write, worked out before anything
    # is copied, because it is also the list of what may stay.
    wanted: dict = {}
    for side, stems in (("train", train), ("val", val)):
        for stem in stems:
            for kind, sub in (("image", images_subpath), ("label", labels_subpath)):
                entry = found[f"_{kind}_keys"][stem]
                name = entry["key"].rsplit("/", 1)[-1]
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
            pc.s3.delete_objects(Bucket=target_bucket,
                                 Delete={"Objects": batch, "Quiet": True})
            removed += len(batch)
        except Exception as exc:                               # noqa: BLE001
            errors.append({"key": batch[0]["Key"], "error": str(exc)})

    copied = 0
    for destination, entry in wanted.items():
        try:
            pc.s3.copy_object(
                Bucket=target_bucket, Key=destination,
                CopySource={"Bucket": found["bucket"], "Key": entry["key"]},
            )
            copied += 1
        except Exception as exc:                               # noqa: BLE001
            errors.append({"key": entry["key"], "error": str(exc)})

    # Beside the data, so the division is recoverable without re-deriving
    # it: the percentage alone does not say which stems fell where once
    # the source has grown.
    manifest = {
        "built_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "source": f"s3://{found['bucket']}/{source}",
        "split": split,
        "train_percent": outline["train_percent"],
        "train": train,
        "val": val,
    }
    manifest_key = f"{target_folder}/split.json"
    try:
        pc.s3.put_object(Bucket=target_bucket, Key=manifest_key,
                         Body=json.dumps(manifest, indent=2).encode("utf-8"),
                         ContentType="application/json")
    except Exception:                                          # noqa: BLE001
        manifest_key = ""

    return {**outline, "copied": copied, "removed": removed, "errors": errors,
            "manifest": f"s3://{target_bucket}/{manifest_key}" if manifest_key else "",
            "uri": f"s3://{target_bucket}/{target_folder}/"}
