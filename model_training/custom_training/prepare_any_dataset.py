"""Turn an ordinary YOLO folder into one this pipeline can evaluate.

The replay preparer (../replay_lambda/prepare_replay_dataset.py) exists to
remap a known six-class source down to the two classes the production
model detects. This one does no remapping at all -- the classes are
whatever the dataset declares -- and exists only to produce the
provenance the evaluation step reads.

What it deliberately does NOT do is invent that provenance. A replay
dataset's card records the snapshot it came from, a fingerprint of the
source, and which leakage checks ran. An arbitrary folder has no snapshot
and has had no leakage check, so the card written here says exactly that:
the source is the folder's own URI, and every check not performed is
recorded as not performed. A card that claimed otherwise would be worse
than no card, because the whole point of it is to let a metric be traced
back to data someone can trust.

Usage:

    python prepare_any_dataset.py s3://bucket/path/to/dataset/

It validates in place and writes dataset-card.json beside the data.yaml.
"""
import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

import boto3
import yaml

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
SPLITS = ("train", "val")


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _split_uri(uri):
    bucket, _, key = uri.removeprefix("s3://").partition("/")
    return bucket, key.strip("/")


def _read(client, bucket, key):
    return client.get_object(Bucket=bucket, Key=key)["Body"].read()


def _list(client, bucket, prefix):
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def validate_labels(text, class_ids, where):
    """Same box validation the replay preparer applies, minus the remap.

    An out-of-range class or a box outside the image trains silently and
    scores nonsense, so it is caught here rather than surfacing as a bad
    metric later.
    """
    for number, line in enumerate(text.splitlines(), start=1):
        values = line.split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"{where} line {number}: expected 'class cx cy w h'")
        try:
            class_id = int(values[0])
            x, y, w, h = (float(v) for v in values[1:5])
        except ValueError:
            raise ValueError(f"{where} line {number}: non-numeric field")
        if class_id not in class_ids:
            raise ValueError(f"{where} line {number}: class {class_id} is not in data.yaml")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"{where} line {number}: box outside 0-1")


def prepare(uri):
    client = boto3.client("s3")
    bucket, prefix = _split_uri(uri)
    root = f"{prefix}/" if prefix else ""

    manifest = yaml.safe_load(_read(client, bucket, root + "data.yaml"))
    names = {int(k): v for k, v in (manifest.get("names") or {}).items()}
    if not names:
        raise ValueError("data.yaml declares no class names")
    class_ids = set(names)

    records = []
    for split in SPLITS:
        image_prefix = f"{root}images/{split}/"
        label_prefix = f"{root}labels/{split}/"
        images = [k for k in _list(client, bucket, image_prefix) if k.lower().endswith(IMAGE_SUFFIXES)]
        if not images:
            raise ValueError(f"no images under {image_prefix}")
        for key in sorted(images):
            name = key[len(image_prefix):]
            body = _read(client, bucket, key)
            label_key = label_prefix + name.rsplit(".", 1)[0] + ".txt"
            try:
                label_text = _read(client, bucket, label_key).decode()
            except client.exceptions.NoSuchKey:
                # A missing label file means "no objects here", which is a
                # legitimate negative example, not an error.
                label_text = ""
            validate_labels(label_text, class_ids, label_key)
            records.append({
                "split": split,
                "image": name,
                "sha256": hashlib.sha256(body).hexdigest(),
                "label_sha256": hashlib.sha256(label_text.encode()).hexdigest(),
            })

    card = {
        "schema_version": "1.0.0",
        "adapter_version": "as-is-v1",
        "model": "custom",
        # No snapshot to point at: this is the folder itself, named
        # honestly rather than dressed up as a pull.
        "source_uri": f"s3://{bucket}/{root}",
        "snapshot_uri": None,
        "source_fingerprint": None,
        "source_classes": names,
        "classes": names,
        "samples": records,
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "validation": {
            "paired": True,
            "labels_within_image": True,
            "classes_within_manifest": True,
            # Recorded as not done, because they were not done. The replay
            # preparer can make stronger claims only because it controls
            # where its data came from.
            "exact_train_val_leakage": None,
            "scene_leakage_checked": False,
        },
    }
    card["dataset_version"] = "custom-" + digest(card)[:32]
    client.put_object(
        Bucket=bucket, Key=root + "dataset-card.json",
        Body=(json.dumps(card, indent=2) + "\n").encode(), ContentType="application/json",
    )
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("uri", help="s3:// URI of the dataset folder (the one holding data.yaml)")
    args = parser.parse_args()
    card = prepare(args.uri)
    print(json.dumps({"dataset_version": card["dataset_version"],
                      "classes": card["classes"],
                      "images": len(card["samples"])}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
