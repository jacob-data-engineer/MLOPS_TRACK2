"""Prove files land in week_1, then week_2, week_3, week_4 -- today.

Weeks advance on the clock, so showing four of them would normally take a
month. This walks the ANCHOR backwards instead: an anchor three intervals
in the past puts "now" inside the fourth interval, which is exactly the
arithmetic the sweep does and therefore exactly the thing being tested.

Each round starts from nothing -- weeks, sources and destinations all
cleared. An earlier version of this script kept them, and it lied twice:
stale source files re-swept into the counts, and every round began with no
weeks at all, which is the "first pull" case rather than the one being
tested.

It runs against the DEPLOYED Lambda, and checks exact destinations:

    <training root>/week_N/images/train/   <- the training images folders
    <training root>/week_N/labels/train/   <- the training labels folders
    <eval root>/week_N/images/val/         <- the validation images folders
    <eval root>/week_N/labels/val/         <- the validation labels folders

The live configuration is stashed and restored whether it passes or fails.

    .venv/Scripts/python.exe scripts/test_week_partitioning.py
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

os.environ.setdefault("PIPELINE_BUCKET", "track2-s3")
os.environ.setdefault("AWS_DEFAULT_REGION", "ap-southeast-1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import boto3

from ppe_pipeline_api import pull_control as pc

BUCKET = os.environ["PIPELINE_BUCKET"]
REGION = os.environ["AWS_DEFAULT_REGION"]
BASE = "test_delete/week_partitioning"
TRAIN_ROOT = f"{BASE}/retraining_training"
EVAL_ROOT = f"{BASE}/retraining_evaluation"
WEEK_DAYS = 7

s3 = boto3.client("s3", region_name=REGION)
lam = boto3.client("lambda", region_name=REGION)

failures: list = []


def check(label: str, got, want) -> None:
    if got == want:
        print(f"   PASS  {label}")
        return
    print(f"   FAIL  {label}")
    print(f"           got  {got}")
    print(f"           want {want}")
    failures.append(label)


def keys_under(prefix: str) -> list:
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        out += [o["Key"] for o in page.get("Contents", [])]
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated"):
            break
    return out


def wipe(prefix: str) -> int:
    """Delete everything under a prefix, and confirm it is gone.

    delete_objects reports per-key failures in its RESPONSE rather than
    raising, so a partial delete is silent. That bit once: a round began
    with the previous round's files still present, the sweep skipped them
    as already filed, and the test blamed the week numbering.
    """
    found = keys_under(prefix + "/")
    for batch in (found[i:i + 1000] for i in range(0, len(found), 1000)):
        reply = s3.delete_objects(
            Bucket=BUCKET, Delete={"Objects": [{"Key": k} for k in batch]})
        for problem in reply.get("Errors", []):
            raise RuntimeError(f"could not delete {problem['Key']}: {problem.get('Message')}")

    left = keys_under(prefix + "/")
    if left:
        raise RuntimeError(f"{len(left)} object(s) survived the wipe, e.g. {left[0]}")
    return len(found)


def clear_weeks() -> None:
    for week in pc.list_weeks():
        pc.delete_week(name=week["name"], starts_at=week["starts_at"])


def landed() -> set:
    return {k for root in (TRAIN_ROOT, EVAL_ROOT)
            for k in keys_under(root + "/") if not k.endswith("/")}


print("stashing the live configuration")
original = pc.read_config()
original_weeks = pc.list_weeks()
print(f"   {len(original.get('sources') or [])} source(s), mode "
      f"{original.get('mode')}, {len(original_weeks)} week(s)")

try:
    for target in (1, 2, 3, 4):
        wipe(BASE)
        clear_weeks()

        anchor = pc._now() - dt.timedelta(days=WEEK_DAYS * (target - 1))
        print(f"\nweek {target}: anchor {anchor.date()} "
              f"({WEEK_DAYS * (target - 1)} days back)")

        pc.write_config({
            "mode": "off",
            "week_by": "scan",
            "week_length_days": WEEK_DAYS,
            "week_prefix": "week_",
            "week_anchor": anchor.isoformat(),
            "auto_create_weeks": True,
            "first_week_takes_all": True,
            "source_bucket": "",
            "destination_bucket": "",
            "training_root": TRAIN_ROOT,
            "eval_root": EVAL_ROOT,
            "picked": {
                "train_images": [f"{BASE}/src/train_imgs"],
                "train_labels": [f"{BASE}/src/train_lbls"],
                "val_images": [f"{BASE}/src/val_imgs"],
                "val_labels": [f"{BASE}/src/val_lbls"],
            },
        })
        pc.write_config({"sources": pc.sources_from_picked(pc.read_config())})

        # week_1 is what the FIRST pull creates, at the anchor. For any
        # later week it already exists and the ladder runs from it, so
        # creating it here is what that first sweep did N intervals ago.
        if target > 1:
            pc.create_week(name="week_1", starts_at=anchor, make_folders=False)

        for sub, name, body in (
            ("train_imgs", "a.jpg", b"img"),
            ("train_lbls", "a.txt", b"0 .5 .5 .1 .1\n"),
            ("val_imgs", "b.jpg", b"img"),
            ("val_lbls", "b.txt", b"0 .4 .4 .2 .2\n"),
        ):
            s3.put_object(Bucket=BUCKET, Key=f"{BASE}/src/{sub}/{name}", Body=body)

        result = json.loads(lam.invoke(
            FunctionName="track2-app-pull-control-router",
            Payload=json.dumps({"detail-type": "Scheduled Event"}).encode(),
        )["Payload"].read())
        print(f"   lambda: moved {result.get('moved')}, "
              f"skipped {result.get('skipped')}, errors {result.get('errors')}")

        got = landed()
        week = f"week_{target}"
        check(f"week {target}: exactly 4 files filed", len(got), 4)
        check(f"week {target}: training image -> {week}/images/train",
              {k for k in got if k.endswith(".jpg") and TRAIN_ROOT in k},
              {f"{TRAIN_ROOT}/{week}/images/train/a.jpg"})
        check(f"week {target}: training label -> {week}/labels/train",
              {k for k in got if k.endswith(".txt") and TRAIN_ROOT in k},
              {f"{TRAIN_ROOT}/{week}/labels/train/a.txt"})
        check(f"week {target}: validation image -> {week}/images/val",
              {k for k in got if k.endswith(".jpg") and EVAL_ROOT in k},
              {f"{EVAL_ROOT}/{week}/images/val/b.jpg"})
        check(f"week {target}: validation label -> {week}/labels/val",
              {k for k in got if k.endswith(".txt") and EVAL_ROOT in k},
              {f"{EVAL_ROOT}/{week}/labels/val/b.txt"})
        # Neither root may hold the other's split.
        check(f"week {target}: no val data under the training root",
              {k for k in got if TRAIN_ROOT in k and "/val/" in k}, set())
        check(f"week {target}: no train data under the evaluation root",
              {k for k in got if EVAL_ROOT in k and "/train/" in k}, set())

        for key in sorted(got):
            print("           ", key.replace(BASE + "/", ""))

finally:
    print("\ncleaning up")
    removed = wipe(BASE)
    clear_weeks()
    pc.write_config(original)
    for week in original_weeks:
        pc.create_week(name=week["name"], starts_at=week["starts_at"], make_folders=False)
    back = pc.read_config()
    print(f"   removed {removed} objects")
    print(f"   restored {len(back.get('sources') or [])} source(s), mode "
          f"{back.get('mode')}, {len(pc.list_weeks())} week(s)")

print()
print("ALL CHECKS PASSED" if not failures else f"{len(failures)} CHECK(S) FAILED")
sys.exit(1 if failures else 0)
