"""Mirrors the official Ultralytics COCO releases into this bucket.

The import card's third source reads `s3://<bucket>/track2-yolo-weights/`.
Those files are put there by this script, once, rather than fetched on
demand, for the same reason the base container images are mirrored:
a run that reaches the public internet is a run whose success depends on
GitHub being up and on whatever rate limit applies that day, and the one
thing that varied between a four-minute build and an eighteen-minute one
here was exactly that.

Mirrored, not vendored: these are published checkpoints and this only
copies them. Nothing is rebuilt, so the bytes are the ones Ultralytics
released.

**Run it on the review host, not a workstation.** 290MB over a home
connection, twice -- down and back up -- against seconds in-region. The
instance role already writes this bucket.

    python3 mirror_yolo_weights.py --bucket track2-s3            # what it would do
    python3 mirror_yolo_weights.py --bucket track2-s3 --apply

Detection weights only. `-seg`, `-pose`, `-cls` and `-obb` are different
tasks: this platform trains detectors, and a segmentation checkpoint
offered beside the others would train and register as if it were one.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import urllib.request

PREFIX = "track2-yolo-weights"
#: `v8.4.0`, not `v8.3.0`. The older tag carries yolo11 but NOT yolo26 --
#: a HEAD against it 404s for every yolo26 asset -- and `latest` is not a
#: resolvable tag on this repo at all. Checked, not assumed: half the
#: library would otherwise have mirrored and the other half "skipped"
#: with a 404 nobody would read as a wrong tag.
RELEASE_TAG = "v8.4.0"
RELEASE_URL = f"https://github.com/ultralytics/assets/releases/download/{RELEASE_TAG}/{{name}}"

#: s, m and l for both generations. Not `n`, which is too small to be a
#: serious starting point here, and not `x` -- asked and answered.
GENERATIONS = ("11", "26")
SIZES = ("s", "m", "l")


def wanted() -> list[str]:
    return [f"yolo{generation}{size}.pt"
            for generation in GENERATIONS for size in SIZES]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--apply", action="store_true",
                        help="without it, says what it would do and stops")
    args = parser.parse_args()

    import boto3

    s3 = boto3.client("s3")
    existing = set()
    token = ""
    while True:
        kwargs = {"Bucket": args.bucket, "Prefix": f"{PREFIX}/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            existing.add(obj["Key"].rsplit("/", 1)[-1])
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken") or ""
        if not token:
            break

    # Already there is already done. These are immutable published
    # releases, so re-uploading one would only cost time and produce the
    # same object -- and this is the kind of script somebody re-runs
    # because they are not sure whether it finished.
    todo = [name for name in wanted() if name not in existing]
    print(f"s3://{args.bucket}/{PREFIX}/")
    for name in wanted():
        print(f"  {name:14s} {'already there' if name in existing else 'to mirror'}")
    if not todo:
        print("Nothing to do.")
        return 0
    if not args.apply:
        print(f"\n{len(todo)} to mirror. Re-run with --apply.")
        return 0

    with tempfile.TemporaryDirectory() as work:
        for name in todo:
            local = os.path.join(work, name)
            url = RELEASE_URL.format(name=name)
            print(f"  {name}: downloading", flush=True)
            try:
                urllib.request.urlretrieve(url, local)
            except Exception as exc:                           # noqa: BLE001
                # One missing release must not stop the rest: the point
                # of the folder is what IS in it, and the card lists
                # what it finds rather than what it expects.
                print(f"  {name}: SKIPPED -- {exc}")
                continue
            size = os.path.getsize(local)
            if size < 1_000_000:
                print(f"  {name}: SKIPPED -- {size} bytes is not a checkpoint")
                continue
            s3.upload_file(local, args.bucket, f"{PREFIX}/{name}")
            print(f"  {name}: uploaded {size / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
