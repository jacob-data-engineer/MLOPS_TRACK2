"""Populates the pipeline bucket with a real-image, synthetic-label YOLO
dataset — proof-of-mechanics only, same precedent as Track2's own
DummyClassifier (see its train.py module docstring).

Why synthetic boxes: the 275 real hazard images in safety-model-dataset-dev
carry only filename category tags (Helmet/Gloves/Vehicles/FallZone +
TP/FP/test), no actual bounding-box annotations anywhere (checked S3 and
the local Label Studio install — neither has exported YOLO labels yet).
Real annotations should replace this once they exist; nothing downstream
(train.py, pipeline_definition.py) needs to change when that happens,
since they only care that images/<split>/*.jpg and labels/<split>/*.txt
exist in the shape ultralytics expects.

Box placement: a fixed centered region covering ~50% of the image,
labeled with the class the filename tag maps to. Plausible-looking, not
evidentially meaningful — this is stated here so nobody mistakes a
trained model's "accuracy" against this data for real detection quality.
"""

import io
import random
import re

import boto3
from PIL import Image

from config import CONFIG

s3 = boto3.client("s3", region_name=CONFIG.region)

NAME_RE = re.compile(r"^(?:[0-9a-f]+-)?(Helmet|Gloves|Vehicles|FallZone)_", re.IGNORECASE)
VAL_FRACTION = 0.2


def _list_source_images():
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=CONFIG.source_bucket, Prefix=CONFIG.source_images_prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].lower().endswith((".jpg", ".jpeg", ".png")):
                keys.append(obj["Key"])
    return keys


def _class_id_for(key: str) -> int | None:
    filename = key.rsplit("/", 1)[-1]
    match = NAME_RE.match(filename)
    if not match:
        return None
    category = match.group(1)
    for tag, class_id in CONFIG.class_map.items():
        if tag.lower() == category.lower():
            return class_id
    return None


def _synthetic_yolo_line(class_id: int) -> str:
    # centered box, ~50% width/height, matching the module docstring
    return f"{class_id} 0.5 0.5 0.5 0.5\n"


def seed(limit: int | None = None):
    keys = _list_source_images()
    labeled = [(k, _class_id_for(k)) for k in keys]
    labeled = [(k, c) for k, c in labeled if c is not None]
    if limit:
        labeled = labeled[:limit]

    random.seed(42)
    random.shuffle(labeled)
    n_val = max(1, int(len(labeled) * VAL_FRACTION))
    val_set, train_set = labeled[:n_val], labeled[n_val:]

    print(f"{len(labeled)} labeled images ({len(train_set)} train / {len(val_set)} val)")

    for split_name, split_items in (("train", train_set), ("val", val_set)):
        for key, class_id in split_items:
            filename = key.rsplit("/", 1)[-1]
            stem = filename.rsplit(".", 1)[0]

            body = s3.get_object(Bucket=CONFIG.source_bucket, Key=key)["Body"].read()
            # touch the image just to fail loudly now if it's not decodable,
            # rather than at training time inside the container
            Image.open(io.BytesIO(body)).verify()

            s3.put_object(
                Bucket=CONFIG.pipeline_bucket,
                Key=f"images/dataset/images/{split_name}/{filename}",
                Body=body,
            )
            s3.put_object(
                Bucket=CONFIG.pipeline_bucket,
                Key=f"images/dataset/labels/{split_name}/{stem}.txt",
                Body=_synthetic_yolo_line(class_id).encode(),
            )

    # No hardcoded `path:` — ultralytics defaults it to the yaml file's own
    # directory when omitted, so this resolves correctly whether mounted at
    # the TrainingStep's channel path (/opt/ml/input/data/train) or the
    # EvaluateStep's Processing job path (/opt/ml/processing/data). A
    # hardcoded absolute path here would work for training but silently
    # break evaluate.py's val() call in the Processing job context.
    data_yaml = (
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        + "\n".join(f"  {i}: {name}" for i, name in enumerate(__import__("config").CLASS_NAMES))
        + "\n"
    )
    s3.put_object(
        Bucket=CONFIG.pipeline_bucket,
        Key="images/dataset/data.yaml",
        Body=data_yaml.encode(),
    )
    print(f"Wrote data.yaml + dataset to s3://{CONFIG.pipeline_bucket}/images/dataset/")


if __name__ == "__main__":
    seed()
