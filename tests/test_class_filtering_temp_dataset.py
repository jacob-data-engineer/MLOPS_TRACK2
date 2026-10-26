"""Training-time class filtering into a TEMPORARY dataset (2026-10-20 spec).

`build_filtered_dataset` runs in the training container after download and
before Ultralytics scans anything. It links each image into a fresh
`<download>_filtered/` tree, writes its label there filtered, remapped and
renumbered, writes a temporary data.yaml, and never touches the download.
Filtering only happens when "Filter undeclared classes" is ticked; with it
off, an undeclared class is an error, as on the card.

The twelve cases from the spec, in order.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_training" / "training-container"))
train = pytest.importorskip("train")


def _make(root, labels: dict, names="  0: helmet\n", extra_yaml="", train_=("ds/images/train",),
          val_=("ds/images/val",), images=None):
    """images/<name>.jpg for every label key (and any extra `images`)."""
    for rel, body in labels.items():
        img = rel.replace("/labels/", "/images/").rsplit(".", 1)[0] + ".jpg"
        os.makedirs(os.path.dirname(os.path.join(root, img)), exist_ok=True)
        open(os.path.join(root, img), "w").close()
        if body is not None:
            os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
            with open(os.path.join(root, rel), "w") as f:
                f.write(body)
    for img in images or []:
        os.makedirs(os.path.dirname(os.path.join(root, img)), exist_ok=True)
        open(os.path.join(root, img), "w").close()
    y = os.path.join(root, "data.yaml")
    with open(y, "w") as f:
        f.write(f"path: {root}\ntrain:\n" + "".join(f"- {t}\n" for t in train_)
                + "val:\n" + "".join(f"- {v}\n" for v in val_) + "names:\n" + names + extra_yaml)
    return y


def _run(root, y, drop=True):
    return train.build_filtered_dataset(root, y, drop_undeclared=drop,
                                        out_root=os.path.join(os.path.dirname(root), "filtered"))


def _read(path):
    with open(path) as f:
        return f.read()


@pytest.fixture
def ws():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "download")
        os.makedirs(root)
        yield root


MIXED = "0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.1 0.1\n0 0.7 0.6 0.2 0.3\n"


# 1
def test_only_allowed_classes_pass_through_unchanged(ws):
    y = _make(ws, {"ds/labels/train/a.txt": "0 0.5 0.5 0.2 0.2\n", "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    new, rep = _run(ws, y)
    out = os.path.dirname(new)
    assert _read(os.path.join(out, "ds/labels/train/a.txt")) == "0 0.5 0.5 0.2 0.2\n"
    assert rep["removed"] == 0


# 2
def test_only_excluded_classes_leave_a_background_image(ws):
    y = _make(ws, {"ds/labels/train/a.txt": "1 .5 .5 .1 .1\n", "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    new, rep = _run(ws, y)
    out = os.path.dirname(new)
    assert _read(os.path.join(out, "ds/labels/train/a.txt")) == ""
    assert os.path.exists(os.path.join(out, "ds/images/train/a.jpg")), "image kept as background"
    assert rep["now_empty"] == 1


# 3 -- the spec's own example
def test_mixed_labels_keep_the_allowed_boxes_exactly(ws):
    y = _make(ws, {"ds/labels/train/a.txt": MIXED, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    new, rep = _run(ws, y)
    assert _read(os.path.join(os.path.dirname(new), "ds/labels/train/a.txt")) == \
        "0 0.5 0.5 0.2 0.2\n0 0.7 0.6 0.2 0.3\n"
    assert rep["removed_by_class"] == {1: 1}


# 4
def test_an_empty_label_file_stays_empty(ws):
    y = _make(ws, {"ds/labels/train/a.txt": "", "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    new, rep = _run(ws, y)
    assert _read(os.path.join(os.path.dirname(new), "ds/labels/train/a.txt")) == ""


# 5
def test_an_image_without_a_label_file_stays_without_one(ws):
    y = _make(ws, {"ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"}, images=["ds/images/train/nolabel.jpg"])
    new, rep = _run(ws, y)
    out = os.path.dirname(new)
    assert os.path.exists(os.path.join(out, "ds/images/train/nolabel.jpg"))
    assert not os.path.exists(os.path.join(out, "ds/labels/train/nolabel.txt"))
    assert rep["no_label_file"] == 1


# 6
def test_sources_from_several_locations_are_all_filtered(ws):
    y = _make(ws, {
        "Testing_Dataset/ds/labels/train/a.txt": MIXED,
        "golden_set/labels/train/g.txt": "2 .5 .5 .1 .1\n0 .2 .2 .1 .1\n",
        "golden_set/labels/val/h.txt": "0 .5 .5 .1 .1\n",
    }, train_=("Testing_Dataset/ds/images/train", "golden_set/images/train"), val_=("golden_set/images/val",))
    new, rep = _run(ws, y)
    out = os.path.dirname(new)
    assert _read(os.path.join(out, "golden_set/labels/train/g.txt")) == "0 .2 .2 .1 .1\n"
    assert rep["removed_by_class"] == {1: 1, 2: 1}
    doc = yaml.safe_load(_read(new))
    assert doc["train"] == ["Testing_Dataset/ds/images/train", "golden_set/images/train"]


# 7
def test_a_conflicting_source_numbering_is_translated_by_its_map(ws):
    """The builder computes the map by name; here source 0='person',
    1='helmet' onto declared 0='helmet'."""
    y = _make(ws, {"site/labels/train/s.txt": "1 .5 .5 .1 .1\n0 .2 .2 .1 .1\n",
                   "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"},
              train_=("site/images/train",),
              extra_yaml="x_label_remap:\n  site/labels/train:\n    0: -1\n    1: 0\n")
    new, rep = _run(ws, y)
    assert _read(os.path.join(os.path.dirname(new), "site/labels/train/s.txt")) == "0 .5 .5 .1 .1\n"
    assert "x_label_remap" not in yaml.safe_load(_read(new))


# 8
def test_non_consecutive_class_ids_are_renumbered_and_the_yaml_agrees(ws):
    y = _make(ws, {"ds/labels/train/a.txt": "0 .5 .5 .1 .1\n3 .2 .2 .1 .1\n1 .3 .3 .1 .1\n",
                   "ds/labels/val/v.txt": "3 .5 .5 .1 .1\n"},
              names="  0: helmet\n  3: boots\n")
    new, rep = _run(ws, y)
    out = os.path.dirname(new)
    assert _read(os.path.join(out, "ds/labels/train/a.txt")) == "0 .5 .5 .1 .1\n1 .2 .2 .1 .1\n"
    doc = yaml.safe_load(_read(new))
    assert doc["names"] == {0: "helmet", 1: "boots"}
    assert doc["nc"] == 2


# 9
def test_repeated_runs_with_different_classes_start_fresh(ws):
    y = _make(ws, {"ds/labels/train/a.txt": MIXED, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"},
              names="  0: helmet\n  1: gloves\n")
    out_root = os.path.join(os.path.dirname(ws), "filtered")
    train.build_filtered_dataset(ws, y, drop_undeclared=True, out_root=out_root)
    open(os.path.join(out_root, "ds", "labels", "train.cache"), "w").close()   # a stale cache
    with open(y, "w") as f:
        f.write(f"path: {ws}\ntrain:\n- ds/images/train\nval:\n- ds/images/val\nnames:\n  0: helmet\n")
    new, rep = train.build_filtered_dataset(ws, y, drop_undeclared=True, out_root=out_root)
    assert not os.path.exists(os.path.join(out_root, "ds", "labels", "train.cache"))
    assert _read(os.path.join(out_root, "ds/labels/train/a.txt")).count("\n") == 2


# 10
def test_the_download_is_never_modified(ws):
    y = _make(ws, {"ds/labels/train/a.txt": MIXED, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    before = _read(os.path.join(ws, "ds/labels/train/a.txt"))
    before_yaml = _read(y)
    _run(ws, y)
    assert _read(os.path.join(ws, "ds/labels/train/a.txt")) == before
    assert _read(y) == before_yaml


# 11
def test_train_and_val_resolve_inside_the_temporary_dataset(ws):
    y = _make(ws, {"ds/labels/train/a.txt": MIXED, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    new, _ = _run(ws, y)
    doc = yaml.safe_load(_read(new))
    assert doc["path"] == os.path.dirname(new)
    for split in ("train", "val"):
        for entry in doc[split]:
            assert os.path.isdir(os.path.join(doc["path"], entry))
    # and the preflight agrees
    assert train.preflight_dataset(new)["train"]["images"] == 1


# 12
def test_the_model_is_sized_by_the_declared_classes_not_the_labels(ws):
    """Six classes in the labels, one declared: one class out."""
    y = _make(ws, {"ds/labels/train/a.txt": "0 .5 .5 .1 .1\n5 .2 .2 .1 .1\n4 .3 .3 .1 .1\n",
                   "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n2 .5 .5 .1 .1\n"})
    new, _ = _run(ws, y)
    doc = yaml.safe_load(_read(new))
    assert doc["nc"] == 1 and doc["names"] == {0: "helmet"}


# --- refusals: the checkbox, and malformed labels ----------------------------------

def test_with_the_box_unticked_an_undeclared_class_is_an_error(ws):
    y = _make(ws, {"ds/labels/train/a.txt": MIXED, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    with pytest.raises(SystemExit, match="labels use class\\(es\\) the yaml does not name: 1"):
        _run(ws, y, drop=False)


@pytest.mark.parametrize("line,why", [
    ("0 0.5 0.5 0.2\n", "fields"),
    ("0 a 0.5 0.2 0.2\n", "non-numeric"),
    ("0 1.4 0.5 0.2 0.2\n", "outside 0-1"),
    ("0.5 0.5 0.5 0.2 0.2\n", "not a whole number"),
])
def test_malformed_lines_are_errors_not_excluded_classes(ws, line, why):
    y = _make(ws, {"ds/labels/train/a.txt": line, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    with pytest.raises(SystemExit, match=why):
        _run(ws, y)


def test_nc_disagreeing_with_names_is_an_error(ws):
    y = _make(ws, {"ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"}, names="  0: helmet\nnc: 2\n")
    with pytest.raises(SystemExit, match="nc: 2"):
        _run(ws, y)


def test_a_failed_filter_leaves_no_temporary_dataset_behind(ws):
    y = _make(ws, {"ds/labels/train/a.txt": "0 a b c d\n", "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    with pytest.raises(SystemExit):
        _run(ws, y)
    assert not os.path.exists(os.path.join(os.path.dirname(ws), "filtered"))


def test_it_reports_what_it_did(ws, capsys):
    y = _make(ws, {"ds/labels/train/a.txt": MIXED, "ds/labels/val/v.txt": "0 .5 .5 .1 .1\n"})
    _run(ws, y)
    line = capsys.readouterr().out
    for piece in ("LABEL_FILTER:", "label files 2", "annotations kept 3", "removed 1",
                  "by source class {1: 1}", "originals unchanged"):
        assert piece in line, piece
