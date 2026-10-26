"""One data.yaml naming folders where they are, instead of copying them.

The constraint everything here exists to respect: Ultralytics cannot read
s3://. The trainer downloads ONE prefix to local disk and only then opens
the yaml, so every folder named has to sit under that one prefix -- and
that prefix is the common parent of whatever was picked.
"""
from __future__ import annotations

import pytest

from ppe_pipeline_api import dataset_yaml as dy

CLASSES = {"0": "person", "1": "helmet"}


def _splits(train, val=()):
    return {"train": [{"images": k} for k in train],
            "val": [{"images": k} for k in val]}


# --- where the labels have to be ----------------------------------------------

def test_the_labels_folder_is_derived_not_chosen():
    """Ultralytics swaps the LAST `/images/` for `/labels/`. A folder
    picked as "labels" that is not that one would be ignored in silence
    and every image would train as a background example -- a run that
    succeeds, reports plausible metrics and has learned nothing."""
    assert dy._labels_dir_for("jp/site-a/images/train") == "jp/site-a/labels/train"
    assert dy._labels_dir_for("jp/images") == "jp/labels"


def test_the_last_images_segment_wins():
    """`rsplit`, not `split` -- a path with `images` twice resolves on the
    one nearest the file, which is what Ultralytics does."""
    assert dy._labels_dir_for("images/jp/images/train") == "images/jp/labels/train"


def test_a_folder_with_no_images_segment_is_refused(monkeypatch):
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    with pytest.raises(ValueError, match="no 'images' folder"):
        dy.build_in_place(splits=_splits(["jp/site-a/pics"]), classes=CLASSES)


def test_a_mismatched_labels_folder_is_refused_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    splits = {"train": [{"images": "jp/a/images", "labels": "jp/somewhere-else"}],
              "val": []}
    with pytest.raises(ValueError, match="would be read from"):
        dy.build_in_place(splits=splits, classes=CLASSES)


# --- the one prefix the run downloads -----------------------------------------

def test_folders_sharing_a_parent_resolve_to_it():
    assert dy._parent_of_all(["jp/a/images", "jp/b/images"]) == "jp"


def test_one_folder_resolves_to_its_parent_not_itself():
    """The yaml has to sit at the root of what is downloaded, so it cannot
    sit inside the images directory it is describing."""
    assert dy._parent_of_all(["jp/site-a/images"]) == "jp/site-a"


def test_training_and_validation_may_live_in_unrelated_places(monkeypatch):
    """They are two separate downloads -- the training job takes
    TRAIN_DATA_S3_URI, the evaluation step takes EVAL_DATA_S3_URI and runs
    in a different container. An earlier version of this refused the case
    outright, which nothing in the system required."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})

    out = dy.build_in_place(
        splits=_splits(["jp/a/images"], ["ming/b/images"]), classes=CLASSES)
    # Since 2026-10-20 the TRAINER downloads both prefixes, so training
    # validates on the real val folder; the evaluation step still gets
    # its own one-prefix yaml at the val folder's root.
    assert out["many_roots"] is True
    assert sorted(out["download_prefixes"]) == ["s3://b/jp/a/", "s3://b/ming/b/"]
    assert out["eval"] is not None
    assert out["eval"]["folder"] == "ming/b"
    assert out["eval"]["key"] == "ming/b/data.yaml"


def test_the_training_yaml_validates_on_the_real_val_when_it_is_elsewhere(monkeypatch):
    """It used to validate on its own training images -- the real val was
    a download the trainer never saw, so best.pt was picked on training
    data. Both prefixes are fetched now and val is the val folder."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})

    out = dy.build_in_place(
        splits=_splits(["jp/a/images"], ["ming/b/images"]), classes=CLASSES)
    assert out["train"] == ["jp/a/images"]
    assert out["val"] == ["ming/b/images"]
    assert set(out["train"]).isdisjoint(out["val"])


def test_scattered_folders_within_one_side_are_now_several_downloads(monkeypatch):
    """They used to be REFUSED, because that side was one download and
    `train.py` fetched one prefix.

    It takes a comma-separated list now, so folders under unrelated
    roots are fetched as several prefixes and named by full key path --
    which is how a dataset, a golden set and week folders end up in one
    training set. The refusal was never Ultralytics': the yaml could
    always name folders anywhere.
    """
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda bucket, key: {})
    out = dy.build_in_place(
        splits=_splits(["jp/a/images", "ming/b/images"]), classes=CLASSES)
    assert out["many_roots"] is True
    assert out["download_prefixes"] == ["s3://b/jp/a/", "s3://b/ming/b/"]
    # Full key paths, matching what the downloader writes to disk when
    # it is given more than one prefix.
    assert out["train"] == ["jp/a/images", "ming/b/images"]


def test_folders_that_do_share_a_parent_stay_one_download(monkeypatch):
    """`jp/a` and `jp/b` share `jp`. Splitting them would turn an
    ordinary dataset into two prefixes for no reason."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    out = dy.build_in_place(
        splits=_splits(["jp/a/images", "jp/b/images"]), classes=CLASSES)
    assert not out.get("many_roots")
    assert out["folder"] == "jp"


def test_a_val_folder_inside_the_training_tree_stays_one_download(monkeypatch):
    """`jp/val` sits inside `jp`, so one prefix reaches both. Comparing the
    two bases for equality rather than containment split exactly this case
    into two downloads it never needed."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})

    out = dy.build_in_place(
        splits=_splits(["jp/a/images", "jp/b/images"], ["jp/val/images"]),
        classes=CLASSES)
    assert out["eval"] is None
    assert out["folder"] == "jp"
    assert out["val"] == ["val/images"]


# --- the document ---------------------------------------------------------------

def test_the_splits_are_relative_to_the_download_prefix(monkeypatch):
    """train.py sets `path` to the local download dir and only rewrites a
    split when it is a STRING -- a list stays relative and resolves
    against `path`. So relative is exactly right here."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})

    out = dy.build_in_place(
        splits=_splits(["jp/a/images", "jp/b/images"], ["jp/val/images"]),
        classes=CLASSES)
    assert out["folder"] == "jp"
    assert out["train"] == ["a/images", "b/images"]
    assert out["val"] == ["val/images"]
    assert out["key"] == "jp/data.yaml"


def test_labels_are_never_named_in_the_document(monkeypatch):
    """Naming them would make Ultralytics ignore them."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})

    out = dy.build_in_place(splits=_splits(["jp/a/images"]), classes=CLASSES)
    assert "labels" not in out["text"].split("names:")[0].replace("# ", "")


def test_no_classes_is_refused(monkeypatch):
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    with pytest.raises(ValueError, match="declare nothing"):
        dy.build_in_place(splits=_splits(["jp/a/images"]), classes={})


# --- what the run actually pays for --------------------------------------------

def test_everything_under_the_prefix_is_counted_not_just_what_was_picked(monkeypatch):
    """The gotcha of this mode: the common parent is the download prefix
    and the trainer takes ALL of it. Invisible until a GPU has been idle
    for twenty minutes, so it is counted and shown."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [
        {"Key": "jp/a/images/1.jpg", "Size": 100},
        {"Key": "jp/a/images/2.jpg", "Size": 100},
        {"Key": "jp/archive/huge.zip", "Size": 9000},
    ])

    out = dy.build_in_place(splits=_splits(["jp/a/images"]), classes=CLASSES)
    assert out["download"]["in_named_folders"] == 2
    assert out["download"]["extra_objects"] == 1
    assert out["download"]["extra_bytes"] == 9000


# --- not destroying somebody's file ---------------------------------------------

def test_a_handwritten_yaml_is_not_overwritten_without_being_told(monkeypatch):
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml",
                        lambda b, k: {"bytes": 400, "ours": False})

    out = dy.build_in_place(splits=_splits(["jp/a/images"]), classes=CLASSES)
    assert out["needs_overwrite"] is True
    with pytest.raises(ValueError, match="already exists"):
        dy.write_in_place(splits=_splits(["jp/a/images"]), classes=CLASSES)


def test_our_own_earlier_build_is_replaced_freely(monkeypatch):
    """Overwriting our own output is housekeeping; overwriting one
    somebody typed is destroying their work."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml",
                        lambda b, k: {"bytes": 400, "ours": True})

    out = dy.build_in_place(splits=_splits(["jp/a/images"]), classes=CLASSES)
    assert out["needs_overwrite"] is False


def test_the_marker_is_in_what_gets_written(monkeypatch):
    """It is the only way a later build can tell its own file apart."""
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})

    out = dy.build_in_place(splits=_splits(["jp/a/images"]), classes=CLASSES)
    assert dy.IN_PLACE_MARKER in out["text"]
