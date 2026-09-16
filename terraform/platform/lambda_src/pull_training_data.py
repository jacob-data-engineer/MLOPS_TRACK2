"""On-demand Lambda: reads data/track2-datasets/track2_pull/latest.json and
copies the pull it points at into a fixed destination prefix --
sagemaker_training/ -- ready for the training pipeline to read.

Source and destination are both inside the same bucket (track2-s3), so
every copy is a server-side S3 copy_object call -- no download/upload
round-trip needed.

The source prefix is derived as "<latest.json's own directory>/<pull folder
name>/", NOT by trusting latest.json's own "snapshot" field's full absolute
URI literally. This external pipeline has changed shape twice already
while this Lambda was being built:
  1. It nested pull1/pull2 under a new track2_pull/ parent alongside
     latest.json, without updating "snapshot"'s absolute URI to match --
     that field pointed at the old, now-empty un-nested location.
  2. It later started naming pull folders "pull<N>_<timestamp>" instead of
     just "pull<N>", and added an explicit "pull_name" field to latest.json
     carrying that exact folder name.
Deriving the *parent path* from latest.json's own current location (not
from inside "snapshot") survives shape-change #1; preferring the explicit
"pull_name" field when present (falling back to "pull<N>" for older-shaped
latest.json files that predate it) survives shape-change #2.

sagemaker_training/ is fully replaced on every run (everything under it is
deleted first, then the current pull is copied in fresh) rather than
accumulated onto -- otherwise objects from a previous pull whose relative
path doesn't exist in the new pull would silently linger, polluting the
training set with stale data from a superseded pull.

Invoke manually whenever you want the training folder refreshed to the
latest completed pull:

    aws lambda invoke --function-name track2-sagemaker-pull-training --region ap-southeast-1 /dev/stdout

Not triggered automatically by anything (e.g. a new latest.json landing) --
that's a reasonable next step if this should run without a manual step,
but wasn't asked for here.
"""

import json
import posixpath

import boto3

BUCKET = "track2-s3"
LATEST_JSON_KEY = "data/track2-datasets/track2_pull/latest.json"
DEST_PREFIX = "data/track2-datasets/sagemaker_training/"

s3 = boto3.client("s3")


def _clear_destination() -> int:
    """Deletes everything currently under DEST_PREFIX, in batches of up to
    1000 (the delete_objects API limit). Returns the number deleted."""
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=DEST_PREFIX):
        keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if not keys:
            continue
        s3.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
        deleted += len(keys)
    return deleted


def handler(event, context):
    latest = json.loads(s3.get_object(Bucket=BUCKET, Key=LATEST_JSON_KEY)["Body"].read())
    if latest.get("status") != "complete":
        raise RuntimeError(
            f"Refusing to pull: latest.json status is {latest.get('status')!r}, not 'complete'."
        )

    pull_name = latest.get("pull_name") or f"pull{latest['pull']}"
    latest_json_dir = posixpath.dirname(LATEST_JSON_KEY)  # e.g. "data/track2-datasets/track2_pull"
    src_prefix = f"{latest_json_dir}/{pull_name}/"
    src_bucket = BUCKET
    print(f"Derived source prefix from latest.json's own location: s3://{src_bucket}/{src_prefix}")

    deleted = _clear_destination()
    print(f"Cleared {deleted} pre-existing object(s) under {DEST_PREFIX}")

    copied = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=src_bucket, Prefix=src_prefix):
        for obj in page.get("Contents", []):
            src_key = obj["Key"]
            if src_key == src_prefix:
                continue  # the zero-byte "folder marker" object itself
            rel_key = src_key[len(src_prefix) :]
            dest_key = f"{DEST_PREFIX}{rel_key}"
            s3.copy_object(
                Bucket=BUCKET,
                Key=dest_key,
                CopySource={"Bucket": src_bucket, "Key": src_key},
            )
            copied += 1

    result = {
        "pull": latest.get("pull"),
        "pull_name": pull_name,
        "source_snapshot": f"s3://{src_bucket}/{src_prefix}",
        "objects_deleted_from_destination": deleted,
        "objects_copied": copied,
        "destination": f"s3://{BUCKET}/{DEST_PREFIX}",
    }
    print(json.dumps(result))
    return result
