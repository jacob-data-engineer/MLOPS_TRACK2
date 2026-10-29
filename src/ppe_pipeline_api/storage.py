"""The bucket, and the few S3 helpers BOTH sides use (2026-10-28).

Pulled out of `training/track2_shared.py` with the data/training split:
listing a folder or asking whether a key exists is not training and not
data preparation, so it belongs to neither. Kept tiny on purpose -- the
queue-runner Lambda imports it, and nothing here needs more than boto3.
"""

from __future__ import annotations

import os

import boto3
import botocore.exceptions

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
BUCKET = os.environ.get("PIPELINE_BUCKET", "")

# Kept in sync with model_training/training-container/train.py's own
# IMAGE_EXTENSIONS by convention (not imported -- separate deploy unit).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# boto3 clients are documented thread-safe -- one, shared.
s3 = boto3.client("s3", region_name=REGION)


def _yaml():
    """PyYAML, imported on use: the queue-runner Lambda ships boto3 and
    PyYAML's pure-Python parser only, and most of its paths never read a
    data.yaml. See track2_shared._yaml for the longer story."""
    import yaml
    return yaml


def list_s3_folders(prefix: str) -> list:
    """"Folders" directly under prefix (S3 has none really -- this is
    list_objects_v2's CommonPrefixes emulation via Delimiter)."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"][len(prefix):].rstrip("/") for p in response.get("CommonPrefixes", [])]


def _list_keys(prefix: str) -> list[str]:
    """Every object key under a prefix, paginated."""
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        keys.extend(item["Key"] for item in page.get("Contents", []))
    return keys


def _s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False
