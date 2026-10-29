"""A snapshot records BOTH downloads, not just the training one.

Training and evaluation are separate downloads -- the training job takes
TRAIN_DATA_S3_URI, the evaluation step takes EVAL_DATA_S3_URI and runs in
a different container -- so the held-out set can live anywhere in the
bucket.

`create` walked `root` and nothing else. When the evaluation set had a
prefix of its own, the manifest held the training files, the row's
counts described the training files, and the images the model was SCORED
on had no record at all. Every number a registered version is approved
on comes from that side.

Unchanged for the ordinary case: when both splits sit under one prefix
-- which is what "Assemble a dataset" produces from a single dataset
root -- one manifest already covers them and nothing extra is written.
"""
from __future__ import annotations

import inspect
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_snapshots as ds  # noqa: E402

TRAIN_YAML = "path: s3://b/train-set\ntrain: images/train\nval: images/train\nnames: {0: helmet}\n"
EVAL_YAML = "path: s3://b/holdout\ntrain: images/val\nval: images/val\nnames: {0: helmet}\n"

OBJECTS = {
    "train-set": ["train-set/images/train/a.jpg", "train-set/images/train/b.jpg",
                  "train-set/labels/train/a.txt", "train-set/labels/train/b.txt"],
    "holdout": ["holdout/images/val/c.jpg", "holdout/labels/val/c.txt"],
}


@pytest.fixture
def bucket(monkeypatch):
    """Two roots with no parent in common, and a recording S3."""
    put: dict = {}
    rows: list = []

    monkeypatch.setattr(ds.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(
        ds.pc, "_list_objects",
        lambda prefix, b: [{"Key": k}
                           for k in OBJECTS.get(prefix.strip("/").split("/")[0], [])
                           if k.startswith(prefix.strip("/"))])

    class _S3:
        def put_object(self, **kw):
            put[kw["Key"]] = kw["Body"]

        def copy_object(self, **kw):
            put[kw["Key"]] = b""

    monkeypatch.setattr(ds.pc, "s3", _S3())

    # What a dataset selects is the data side's since the split.
    from ppe_pipeline_api.data import selection
    monkeypatch.setattr(
        selection, "read_selection",
        lambda root, bucket="": {
            "yaml": TRAIN_YAML if root == "train-set" else EVAL_YAML,
            "lists": {}, "images": [],
        })

    class _Table:
        def put_item(self, Item):
            rows.append(Item)

    monkeypatch.setattr(ds, "_table", lambda: _Table())
    return {"put": put, "rows": rows}


# --- the second download is recorded ------------------------------------------------

def test_the_eval_prefix_gets_its_own_manifest(bucket):
    row = ds.create(root="train-set", eval_root="holdout")
    key = row["eval"]["manifest_key"]
    assert key in bucket["put"]
    written = json.loads(bucket["put"][key])
    assert written["files"] == ["images/val/c.jpg", "labels/val/c.txt"]
    assert written["root"] == "holdout"


def test_the_eval_yaml_is_stored_too(bucket):
    """So the held-out set can be rebuilt, not just named."""
    row = ds.create(root="train-set", eval_root="holdout")
    assert bucket["put"][row["eval"]["yaml_key"]].decode() == EVAL_YAML


def test_the_row_says_where_the_second_download_is(bucket):
    row = ds.create(root="train-set", eval_root="holdout")
    assert row["eval"]["root"] == "holdout"
    assert row["eval"]["download_prefix"] == "s3://b/holdout/"
    assert row["eval"]["file_count"] == 2
    assert row["eval"]["images"] == 1


def test_the_training_side_is_untouched_by_it(bucket):
    row = ds.create(root="train-set", eval_root="holdout")
    assert row["root"] == "train-set"
    assert row["file_count"] == 4
    assert row["train_images"] == 2


# --- the identity ---------------------------------------------------------------------

def test_a_different_held_out_set_is_a_different_snapshot(bucket):
    """Two runs over the same training images and different held-out
    sets are not the same dataset. One id would file two sets of metrics
    under one record."""
    with_eval = ds.create(root="train-set", eval_root="holdout")
    without = ds.create(root="train-set")
    assert with_eval["snapshot_id"] != without["snapshot_id"]


def test_the_same_pair_hashes_the_same_twice(bucket):
    a = ds.create(root="train-set", eval_root="holdout")
    b = ds.create(root="train-set", eval_root="holdout")
    assert a["snapshot_id"] == b["snapshot_id"]


# --- the ordinary case is unchanged ----------------------------------------------------

def test_an_eval_folder_under_the_root_writes_nothing_extra(bucket):
    """One prefix covers both splits -- the manifest already has them.
    This is what a single dataset root produces, and it is the common
    case."""
    row = ds.create(root="train-set", eval_root="train-set/images/train")
    assert row["eval"] == {}
    assert not any(k.endswith("eval-manifest.json") for k in bucket["put"])


def test_the_same_folder_on_both_sides_writes_nothing_extra(bucket):
    row = ds.create(root="train-set", eval_root="train-set")
    assert row["eval"] == {}


def test_no_eval_root_at_all_is_unchanged(bucket):
    row = ds.create(root="train-set")
    assert row["eval"] == {}
    assert row["manifest_key"].endswith("manifest.json")


def test_the_training_manifest_and_yaml_are_always_written(bucket):
    """What the card already relied on."""
    row = ds.create(root="train-set")
    assert bucket["put"][row["yaml_key"]].decode() == TRAIN_YAML
    assert json.loads(bucket["put"][row["manifest_key"]])["files"] == [
        "images/train/a.jpg", "images/train/b.jpg",
        "labels/train/a.txt", "labels/train/b.txt",
    ]


# --- both callers pass it ---------------------------------------------------------------

def test_the_queue_freeze_passes_the_entrys_eval_folder():
    from ppe_pipeline_api.routers import track2_models as m

    code = inspect.getsource(m._freeze_for_entry)
    assert 'eval_root=str(entry.get("eval_dataset_folder")' in code


def test_the_chain_passes_the_second_download():
    from ppe_pipeline_api import retrain_schedule as rs

    code = inspect.getsource(rs._run_chain_body)
    assert 'eval_root = (out.get("eval") or {}).get("folder") or ""' in code
    assert "eval_root=eval_root" in code


def test_the_week_folder_path_defines_it_too():
    """One freeze call for both branches -- a NameError on the ordinary
    scheduled retrain would be a worse bug than the one being fixed."""
    from ppe_pipeline_api import retrain_schedule as rs

    code = inspect.getsource(rs._run_chain_body)
    assert 'eval_root = ""' in code
