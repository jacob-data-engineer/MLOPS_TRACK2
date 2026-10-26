"""A freeze has to see the files, whichever layout they are in.

`collect_files` read `parts[0]` as the week and required the REST to
start with `images/`:

    retraining/week_3/images/f.jpg   ->  week=week_3  inside=images/f.jpg   OK
    Testing_Dataset/ds/images/train/f.jpg
                                     ->  week=images  inside=train/f.jpg    skipped

The second is the ordinary Ultralytics layout, and the one "Assemble a
dataset" names. Every file was skipped, so the snapshot froze ZERO
files -- and reported success. A freeze is the record that makes a run
repeatable, so an empty one is worse than no freeze at all: there is a
snapshot id, a row in the table, and nothing behind it.

Seen live: snapshot `138fc01d89915e02` of
`Testing_Dataset/object_detection_dataset`, file_count 0, taken for a
queued run on 15 Oct.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_snapshots as ds  # noqa: E402


def _bucket(monkeypatch, keys: list) -> None:
    monkeypatch.setattr(ds.pc, "_list_objects",
                        lambda prefix, b: [{"Key": k} for k in keys])
    monkeypatch.setattr(ds.pc, "destination_bucket", lambda: "track2-s3")


# --- the layout that froze nothing ------------------------------------------------

ULTRALYTICS = [
    "ds/data.yaml",
    "ds/images/train/a.jpg",
    "ds/images/train/b.jpg",
    "ds/images/val/c.jpg",
    "ds/labels/train/a.txt",
    "ds/labels/train/b.txt",
    "ds/labels/val/c.txt",
]


def test_the_assembled_layout_is_collected(monkeypatch):
    _bucket(monkeypatch, ULTRALYTICS)
    out = ds.collect_files("ds")
    assert len(out["files"]) == 6


def test_the_yaml_at_the_root_is_not_counted_as_data(monkeypatch):
    """`data.yaml` is the description, not the corpus."""
    _bucket(monkeypatch, ULTRALYTICS)
    assert "data.yaml" not in ds.collect_files("ds")["files"]


def test_it_groups_by_split_when_images_is_the_top_level(monkeypatch):
    """There is no week to group by, so the split is the only meaningful
    division -- and it is the one the card already reports."""
    _bucket(monkeypatch, ULTRALYTICS)
    rows = {r["week"]: r for r in ds.collect_files("ds")["weeks"]}
    assert rows["train"] == {"week": "train", "images": 2, "labels": 2}
    assert rows["val"] == {"week": "val", "images": 1, "labels": 1}


def test_the_paths_stay_relative_to_the_root(monkeypatch):
    """`train.py` downloads a prefix and joins each key to a local
    directory, so the manifest has to be relative to that prefix."""
    _bucket(monkeypatch, ULTRALYTICS)
    assert ds.collect_files("ds")["files"][0] == "images/train/a.jpg"


# --- the layout that already worked, unchanged ------------------------------------

WEEKS = [
    "retraining/train.txt",
    "retraining/week_1/images/a.jpg",
    "retraining/week_1/labels/a.txt",
    "retraining/week_2/images/b.jpg",
    "retraining/week_2/labels/b.txt",
    "retraining/week_2/notes/readme.md",
]


def test_week_folders_collect_exactly_as_before(monkeypatch):
    _bucket(monkeypatch, WEEKS)
    out = ds.collect_files("retraining")
    assert out["files"] == [
        "week_1/images/a.jpg", "week_1/labels/a.txt",
        "week_2/images/b.jpg", "week_2/labels/b.txt",
    ]


def test_week_grouping_is_unchanged(monkeypatch):
    _bucket(monkeypatch, WEEKS)
    rows = {r["week"]: r for r in ds.collect_files("retraining")["weeks"]}
    assert rows["week_1"] == {"week": "week_1", "images": 1, "labels": 1}
    assert rows["week_2"] == {"week": "week_2", "images": 1, "labels": 1}


def test_a_retention_list_at_the_root_is_still_ignored(monkeypatch):
    _bucket(monkeypatch, WEEKS)
    assert "train.txt" not in ds.collect_files("retraining")["files"]


def test_a_folder_that_is_neither_is_still_ignored(monkeypatch):
    _bucket(monkeypatch, WEEKS)
    assert not any("notes" in f for f in ds.collect_files("retraining")["files"])


# --- the rules that make a snapshot reproducible ----------------------------------

def test_the_listing_is_sorted(monkeypatch):
    """Unsorted, the same corpus would hash differently every time and
    every snapshot of an unchanged dataset would look new."""
    _bucket(monkeypatch, list(reversed(ULTRALYTICS)))
    files = ds.collect_files("ds")["files"]
    assert files == sorted(files)


def test_the_same_corpus_collects_the_same_bytes_twice(monkeypatch):
    _bucket(monkeypatch, ULTRALYTICS)
    assert ds.collect_files("ds")["files"] == ds.collect_files("ds")["files"]


def test_a_bare_images_folder_with_no_file_under_it_is_not_counted(monkeypatch):
    """A prefix marker, not a file. Counting it would put a name in the
    manifest that the trainer cannot download."""
    _bucket(monkeypatch, ["ds/images/", "ds/images/train/a.jpg"])
    assert ds.collect_files("ds")["files"] == ["images/train/a.jpg"]


def test_a_custom_subpath_is_honoured_in_both_layouts(monkeypatch):
    _bucket(monkeypatch, ["ds/pics/train/a.jpg", "ds/anns/train/a.txt",
                          "ds/images/train/ignored.jpg"])
    out = ds.collect_files("ds", images_subpath="pics", labels_subpath="anns")
    assert out["files"] == ["anns/train/a.txt", "pics/train/a.jpg"]


def test_an_empty_root_is_still_refused(monkeypatch):
    _bucket(monkeypatch, [])
    with pytest.raises(ValueError, match="Pick the folder"):
        ds.collect_files("")


def test_a_root_holding_nothing_usable_collects_nothing(monkeypatch):
    """Still zero -- but because there IS nothing, which is a different
    fact from the bug this file is about."""
    _bucket(monkeypatch, ["ds/readme.md", "ds/notes/x.txt"])
    assert ds.collect_files("ds")["files"] == []
