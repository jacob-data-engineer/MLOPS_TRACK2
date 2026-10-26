"""Freezing a dataset definition hard enough to run it again."""

from __future__ import annotations

import json

import pytest

from ppe_pipeline_api import dataset_snapshots as ds


class FakeS3:
    """Enough S3 to exercise the module: a listing, puts, gets, copies."""

    def __init__(self, objects: dict):
        self.objects = dict(objects)
        self.copies: list = []

    def get_paginator(self, _name):
        raise AssertionError("dataset_snapshots lists through pull_control")

    def put_object(self, *, Bucket, Key, Body, **_kw):
        self.objects[Key] = Body

    def get_object(self, *, Bucket, Key):
        class Body:
            def __init__(self, blob):
                self.blob = blob

            def read(self):
                return self.blob

        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": Body(self.objects[Key])}

    def copy_object(self, *, Bucket, Key, CopySource, **_kw):
        self.copies.append((CopySource["Key"], Key))
        self.objects[Key] = self.objects[CopySource["Key"]]


def fake_listing(objects):
    """`pull_control._list_objects` over a fixed key set."""

    def _list(prefix, bucket="", *, markers=False):
        for key in sorted(objects):
            if key.startswith(prefix):
                yield {"Key": key, "Size": 10}

    return _list


WEEK_KEYS = [
    "retraining/week_1/images/train/a.jpg",
    "retraining/week_1/images/train/b.jpg",
    "retraining/week_1/labels/train/a.txt",
    "retraining/week_1/labels/train/b.txt",
    "retraining/week_1/images/val/c.jpg",
    "retraining/week_1/labels/val/c.txt",
    "retraining/week_2/images/train/d.jpg",
    "retraining/week_2/labels/train/d.txt",
]


@pytest.fixture
def s3(monkeypatch):
    objects = {k: b"x" for k in WEEK_KEYS}
    fake = FakeS3(objects)
    monkeypatch.setattr(ds.pc, "s3", fake)
    monkeypatch.setattr(ds.pc, "_list_objects", fake_listing(objects))
    monkeypatch.setattr(ds.pc, "destination_bucket", lambda: "track2-s3")
    return fake


def test_the_manifest_is_every_image_and_label_relative_to_the_root(s3):
    """Relative, because `train.py` joins each key to a local directory
    and cannot check a manifest written in absolute terms."""
    out = ds.collect_files("retraining")
    assert out["files"] == sorted(k[len("retraining/"):] for k in WEEK_KEYS)
    assert all(not name.startswith("/") for name in out["files"])
    assert out["weeks"] == [
        {"week": "week_1", "images": 3, "labels": 3},
        {"week": "week_2", "images": 1, "labels": 1},
    ]


def test_the_manifest_is_sorted_so_an_unchanged_dataset_hashes_the_same(monkeypatch):
    """An unsorted listing would make every snapshot of an unchanged
    corpus look like a new one."""
    shuffled = list(reversed(WEEK_KEYS))
    objects = {k: b"x" for k in shuffled}
    monkeypatch.setattr(ds.pc, "_list_objects", fake_listing(objects))
    monkeypatch.setattr(ds.pc, "destination_bucket", lambda: "track2-s3")

    first = ds.collect_files("retraining")["files"]
    monkeypatch.setattr(ds.pc, "_list_objects", fake_listing({k: b"x" for k in WEEK_KEYS}))
    second = ds.collect_files("retraining")["files"]
    assert first == second
    assert ds._identity("yaml", first) == ds._identity("yaml", second)


def test_the_id_changes_when_either_the_definition_or_the_data_changes():
    """Both, because either alone lies: the same yaml over a grown corpus
    is a different dataset, and the same corpus under a reordered class
    list trains a different model."""
    files = ["week_1/images/train/a.jpg"]
    base = ds._identity("names: [helmet]", files)

    assert ds._identity("names: [helmet]", files) == base          # stable
    assert ds._identity("names: [gloves]", files) != base          # definition
    assert ds._identity("names: [helmet]", files + ["b.jpg"]) != base   # data


def test_the_id_is_not_fooled_by_names_running_together():
    """Hashing a concatenation without a separator makes ["ab"] and
    ["a", "b"] the same dataset."""
    assert ds._identity("y", ["ab"]) != ds._identity("y", ["a", "b"])


def test_drift_reports_per_week_not_just_a_total(s3, monkeypatch):
    """"The corpus grew by 400" does not say whether an OLD week was
    backfilled -- and a backfilled old week is the case where a re-run
    without the manifest silently trains on more than the original."""
    row = {
        "snapshot_id": "abc",
        "root": "retraining",
        "bucket": "track2-s3",
        "file_count": 8,
        "weeks": [
            {"week": "week_1", "images": 3, "labels": 3},
            {"week": "week_2", "images": 1, "labels": 1},
        ],
    }
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: dict(row))

    quiet = ds.drift("abc")
    assert quiet["unchanged"] is True
    assert quiet["changes"] == []

    # week_1 backfilled, and a week_3 appears.
    grown = dict.fromkeys(WEEK_KEYS, b"x")
    grown["retraining/week_1/images/train/z.jpg"] = b"x"
    grown["retraining/week_3/images/train/n.jpg"] = b"x"
    monkeypatch.setattr(ds.pc, "_list_objects", fake_listing(grown))

    moved = ds.drift("abc")
    assert moved["unchanged"] is False
    by_week = {c["week"]: c for c in moved["changes"]}
    assert by_week["week_1"] == {"week": "week_1", "was": 3, "now": 4, "new_week": False}
    assert by_week["week_3"]["new_week"] is True
    assert moved["file_count_now"] == 10


def test_restore_puts_the_yaml_in_the_one_slot_the_trainer_reads(s3, monkeypatch):
    """There is exactly one per root. `train.py` requires data.yaml at the
    root of the prefix it downloads, and only that parent holds every
    week folder the yaml names."""
    s3.objects["_dataset_snapshots/retraining/abc/data.yaml"] = b"names: [helmet]"
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: {
        "snapshot_id": "abc",
        "root": "retraining",
        "bucket": "track2-s3",
        "yaml_key": "_dataset_snapshots/retraining/abc/data.yaml",
        "live_key": "retraining/data.yaml",
        "download_prefix": "s3://track2-s3/retraining/",
        "manifest_uri": "s3://track2-s3/_dataset_snapshots/retraining/abc/manifest.json",
        "lists": {},
    })

    out = ds.restore("abc")
    assert s3.copies == [("_dataset_snapshots/retraining/abc/data.yaml",
                          "retraining/data.yaml")]
    assert out["written_to"] == "s3://track2-s3/retraining/data.yaml"
    # The run needs both: where to download from, and what to keep.
    assert out["download_prefix"] == "s3://track2-s3/retraining/"
    assert out["manifest_uri"].endswith("manifest.json")


def test_restore_puts_the_file_lists_back_before_the_yaml(s3, monkeypatch):
    """The other order leaves a window where the restored yaml names
    lists still holding the PREVIOUS selection, and a run started in it
    trains on the wrong images while looking entirely correct."""
    for key in ("data.yaml", "train.txt", "val.txt"):
        s3.objects[f"_dataset_snapshots/retraining/abc/{key}"] = b"x"
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: {
        "snapshot_id": "abc",
        "root": "retraining",
        "bucket": "track2-s3",
        "yaml_key": "_dataset_snapshots/retraining/abc/data.yaml",
        "live_key": "retraining/data.yaml",
        "download_prefix": "s3://track2-s3/retraining/",
        "manifest_uri": "s3://track2-s3/_dataset_snapshots/retraining/abc/manifest.json",
        "selected": 587,
        "names_from": "list",
        "lists": {
            "train": {"key": "_dataset_snapshots/retraining/abc/train.txt",
                      "live_key": "retraining/_retained/train.txt", "lines": 587},
            "val": {"key": "_dataset_snapshots/retraining/abc/val.txt",
                    "live_key": "retraining/_retained/val.txt", "lines": 149},
        },
    })

    out = ds.restore("abc")
    destinations = [to for _, to in s3.copies]
    assert destinations[-1] == "retraining/data.yaml"      # the yaml goes LAST
    assert "retraining/_retained/train.txt" in destinations[:-1]
    assert "retraining/_retained/val.txt" in destinations[:-1]
    assert out["selected"] == 587
    assert out["names_from"] == "list"


def test_snapshots_are_kept_outside_the_training_prefix():
    """Under it, every run would download the whole snapshot history
    before training -- `train.py` fetches every object beneath the prefix
    it is given."""
    assert not ds.PREFIX.startswith("retraining")
    folder = f"{ds.PREFIX}/retraining/abc"
    assert not folder.startswith("retraining/")


def test_the_manifest_json_names_the_prefix_its_paths_are_relative_to(s3, monkeypatch):
    """So the trainer can check a downloaded key without re-deriving it."""
    monkeypatch.setattr(ds, "_table", lambda: _NullTable())
    monkeypatch.setattr(ds.dataset_yaml, "build", lambda **_kw: {
        "yaml": "names: [helmet]\n",
        "download_prefix": "s3://track2-s3/retraining/",
        "write_to": "retraining/data.yaml",
        "train_images": 3, "val_images": 1,
        "classes": {"0": "helmet"},
        "problems": [], "ready": True,
    })

    row = ds.create(root="retraining")
    written = json.loads(s3.objects[row["manifest_key"]].decode("utf-8"))
    assert written["prefix"] == "retraining/"
    assert written["snapshot_id"] == row["snapshot_id"]
    assert written["files"][0].startswith("week_1/")
    # The row carries the counts so drift is readable without this file.
    assert row["file_count"] == len(written["files"])
    assert row["classes"] == ["helmet"]


class _NullTable:
    def put_item(self, **_kw):
        return {}


def test_readiness_is_probed_with_a_permission_the_app_actually_has(monkeypatch):
    """`table_status` is `dynamodb:DescribeTable`, which this app is not
    granted and does not need. Probing with it reported the table
    unreachable while every operation the feature performs was working --
    so the card told people to go and apply terraform that had already
    been applied.
    """
    calls = []

    class Table:
        def scan(self, **kw):
            calls.append(("scan", kw))
            return {"Items": []}

        @property
        def table_status(self):
            raise AssertionError("DescribeTable is not granted to this app")

    monkeypatch.setattr(ds, "_table", lambda: Table())
    assert ds.table_ready() is True
    assert calls == [("scan", {"Limit": 1})]


def test_readiness_is_false_when_the_table_cannot_be_read(monkeypatch):
    class Table:
        def scan(self, **_kw):
            raise RuntimeError("ResourceNotFoundException")

    monkeypatch.setattr(ds, "_table", lambda: Table())
    assert ds.table_ready() is False


def test_deleting_a_snapshot_leaves_the_images_alone(s3, monkeypatch):
    """A snapshot is a list of names beside a copy of a yaml, never a
    copy of the corpus. Deleting one throws away the record of a
    selection, not the data it selected."""
    deleted: list = []
    s3.delete_object = lambda *, Bucket, Key: deleted.append(Key)

    class Table:
        def __init__(self):
            self.gone: list = []

        def delete_item(self, *, Key):
            self.gone.append(Key["snapshot_id"])

    table = Table()
    monkeypatch.setattr(ds, "_table", lambda: table)
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: {
        "snapshot_id": "abc",
        "root": "retraining",
        "bucket": "track2-s3",
        "yaml_key": "_dataset_snapshots/retraining/abc/data.yaml",
        "manifest_key": "_dataset_snapshots/retraining/abc/manifest.json",
        "lists": {
            "train": {"key": "_dataset_snapshots/retraining/abc/train.txt",
                      "live_key": "retraining/_retained/train.txt", "lines": 3},
        },
    })

    out = ds.remove("abc")
    assert sorted(deleted) == sorted([
        "_dataset_snapshots/retraining/abc/data.yaml",
        "_dataset_snapshots/retraining/abc/manifest.json",
        "_dataset_snapshots/retraining/abc/train.txt",
    ])
    assert table.gone == ["abc"]
    assert out["root"] == "retraining"
    # The LIVE slots are not touched: deleting a record of a dataset must
    # not disturb the dataset that is currently configured.
    assert "retraining/_retained/train.txt" not in deleted
    assert "retraining/data.yaml" not in deleted


def test_a_missing_object_does_not_stop_the_delete(s3, monkeypatch):
    """A key already gone is in the state being asked for."""
    def boom(*, Bucket, Key):
        raise RuntimeError("NoSuchKey")

    s3.delete_object = boom
    rows: list = []
    monkeypatch.setattr(ds, "_table", lambda: type(
        "T", (), {"delete_item": lambda self, *, Key: rows.append(Key)})())
    monkeypatch.setattr(ds, "get", lambda sid, with_yaml=True: {
        "snapshot_id": "abc", "root": "r", "bucket": "b",
        "yaml_key": "k/data.yaml", "manifest_key": "k/manifest.json", "lists": {},
    })

    out = ds.remove("abc")
    assert out["removed"] == []
    assert len(rows) == 1          # the row still goes
