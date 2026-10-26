"""The multi-location dataset spec (2026-10-20), checked end to end.

Covers: several S3 prefixes in one yaml; train/val kept separate; no
validation folder refused rather than filled with training images;
overlap refused; a source whose own data.yaml numbers classes differently
refused; `nc` written; and the trainer's in-container preflight, run on a
real temporary filesystem laid out exactly as the download lays it out.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "model_training" / "training-container"))

from ppe_pipeline_api import dataset_yaml as dy      # noqa: E402
from ppe_pipeline_api import pull_control as pc      # noqa: E402

train = pytest.importorskip("train")


class _Body:
    def __init__(self, t):
        self._t = t.encode()

    def read(self):
        return self._t


class _S3:
    def __init__(self, files):
        self.files = files
        self.put = []

    def get_object(self, Bucket, Key):
        if Key not in self.files:
            raise KeyError(Key)
        return {"Body": _Body(self.files[Key])}

    def put_object(self, **kw):
        self.put.append(kw["Key"])


@pytest.fixture
def bucket(monkeypatch):
    files = {
        "Testing_Dataset/ds/labels/train/a.txt": "0 0.5 0.5 0.1 0.1\n",
        "Testing_Dataset/ds/labels/val/b.txt": "0 0.5 0.5 0.1 0.1\n",
        "golden_set/labels/train/g.txt": "0 0.5 0.5 0.1 0.1\n",
        "golden_set/labels/val/h.txt": "0 0.5 0.5 0.1 0.1\n",
    }
    s3 = _S3(files)
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: [
        {"Key": k, "Size": 1} for k in s3.files if k.startswith(prefix)])
    monkeypatch.setattr(pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    return s3


def S(*dirs):
    return [{"images": d, "labels": ""} for d in dirs]


# --- several prefixes, independent splits ------------------------------------

def test_two_prefixes_make_one_yaml_with_separate_train_and_val(bucket):
    out = dy.build_in_place(
        splits={"train": S("Testing_Dataset/ds/images/train", "golden_set/images/train"),
                "val": S("Testing_Dataset/ds/images/val", "golden_set/images/val")},
        classes={0: "helmet"}, bucket="b")
    assert out["many_roots"] is True
    assert out["train"] == ["Testing_Dataset/ds/images/train", "golden_set/images/train"]
    assert out["val"] == ["Testing_Dataset/ds/images/val", "golden_set/images/val"]
    assert set(out["train"]).isdisjoint(out["val"])
    assert sorted(out["download_prefixes"]) == ["s3://b/Testing_Dataset/ds/", "s3://b/golden_set/"]


def test_no_prefix_is_prepended_to_another(bucket):
    """Full keys, each its own -- not one dataset's prefix glued to another's."""
    out = dy.build_in_place(
        splits={"train": S("Testing_Dataset/ds/images/train", "golden_set/images/train"),
                "val": S("golden_set/images/val")},
        classes={0: "helmet"}, bucket="b")
    assert not any(e.startswith("golden_set/Testing_Dataset") for e in out["train"])
    assert "Testing_Dataset/ds/images/train" in out["text"]


def test_no_anchors_and_nc_is_written(bucket):
    out = dy.build_in_place(
        splits={"train": S("Testing_Dataset/ds/images/train", "golden_set/images/train"),
                "val": S("golden_set/images/val")},
        classes={0: "helmet"}, bucket="b")
    assert "&id" not in out["text"] and "*id" not in out["text"]
    assert "nc: 1" in out["text"]


# --- refusals before any GPU ---------------------------------------------------

def test_no_validation_folder_is_refused_on_write(bucket):
    with pytest.raises(ValueError, match="No validation folder is selected"):
        dy.write_in_place(splits={"train": S("Testing_Dataset/ds/images/train"), "val": []},
                          classes={0: "helmet"}, bucket="b")
    assert bucket.put == []


def test_the_preview_reports_it_rather_than_failing(bucket):
    out = dy.build_in_place(splits={"train": S("Testing_Dataset/ds/images/train"), "val": []},
                            classes={0: "helmet"}, bucket="b")
    assert out["no_val"] is True


def test_the_same_folder_on_both_sides_is_refused(bucket):
    with pytest.raises(ValueError, match="Training and validation overlap"):
        dy.build_in_place(splits={"train": S("golden_set/images/val"),
                                  "val": S("golden_set/images/val")},
                          classes={0: "helmet"}, bucket="b")


def test_a_val_folder_inside_a_train_folder_is_refused(bucket):
    with pytest.raises(ValueError, match="overlap"):
        dy.build_in_place(splits={"train": S("golden_set/images"),
                                  "val": S("golden_set/images/val")},
                          classes={0: "helmet"}, bucket="b")


def test_a_folder_named_val_may_be_trained_on_when_chosen(bucket):
    """The configured split decides, not the folder name."""
    out = dy.build_in_place(splits={"train": S("golden_set/images/val"),
                                    "val": S("Testing_Dataset/ds/images/val")},
                            classes={0: "helmet"}, bucket="b")
    assert out["train"] == ["golden_set/images/val"] or "golden_set/images/val" in out["text"]


def test_a_source_numbering_classes_differently_is_refused(bucket):
    bucket.files["Testing_Dataset/ds/data.yaml"] = "names:\n  0: person\n  1: helmet\n"
    with pytest.raises(ValueError, match="numbers its classes differently"):
        dy.build_in_place(splits={"train": S("Testing_Dataset/ds/images/train"),
                                  "val": S("golden_set/images/val")},
                          classes={0: "helmet"}, bucket="b")


def test_a_yaml_this_module_wrote_is_not_taken_as_evidence(bucket):
    bucket.files["Testing_Dataset/ds/data.yaml"] = dy.IN_PLACE_MARKER + "\nnames:\n  0: other\n"
    dy.build_in_place(splits={"train": S("Testing_Dataset/ds/images/train"),
                              "val": S("golden_set/images/val")},
                      classes={0: "helmet"}, bucket="b")


# --- the trainer's preflight, on a real filesystem ------------------------------

def _tree(root, files):
    for rel, body in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(body)


def _yaml(root, train_, val_, names="  0: helmet\n"):
    path = os.path.join(root, "data.yaml")
    with open(path, "w") as f:
        f.write(f"path: {root}\ntrain:\n" + "".join(f"- {t}\n" for t in train_)
                + "val:\n" + "".join(f"- {v}\n" for v in val_) + "names:\n" + names)
    return path


GOOD = {
    "Testing_Dataset/ds/images/train/a.jpg": "", "Testing_Dataset/ds/labels/train/a.txt": "0 .5 .5 .1 .1\n",
    "golden_set/images/train/g.jpg": "", "golden_set/labels/train/g.txt": "0 .5 .5 .1 .1\n",
    "golden_set/images/val/h.jpg": "", "golden_set/labels/val/h.txt": "0 .5 .5 .1 .1\n",
}


def test_preflight_passes_on_a_good_two_prefix_download():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, GOOD)
        rep = train.preflight_dataset(_yaml(root, ["Testing_Dataset/ds/images/train", "golden_set/images/train"],
                                            ["golden_set/images/val"]))
        assert rep["train"]["images"] == 2 and rep["val"]["images"] == 1


def test_preflight_names_a_missing_directory():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, GOOD)
        with pytest.raises(SystemExit, match="does not exist in the container"):
            train.preflight_dataset(_yaml(root, ["nowhere/images/train"], ["golden_set/images/val"]))


def test_preflight_names_labels_it_cannot_find():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, {"x/images/train/a.jpg": "", "golden_set/images/val/h.jpg": "",
                     "golden_set/labels/val/h.txt": "0 .5 .5 .1 .1\n"})
        with pytest.raises(SystemExit, match="none of its 1 images has a label file"):
            train.preflight_dataset(_yaml(root, ["x/images/train"], ["golden_set/images/val"]))


def test_preflight_names_undeclared_class_ids():
    with tempfile.TemporaryDirectory() as root:
        files = dict(GOOD)
        files["golden_set/labels/train/g.txt"] = "3 .5 .5 .1 .1\n"
        _tree(root, files)
        with pytest.raises(SystemExit, match="labels use class 3"):
            train.preflight_dataset(_yaml(root, ["golden_set/images/train"], ["golden_set/images/val"]))


def test_preflight_names_overlap():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, GOOD)
        with pytest.raises(SystemExit, match="in both train and val"):
            train.preflight_dataset(_yaml(root, ["golden_set/images/val"], ["golden_set/images/val"]))


def test_preflight_collects_every_problem_in_one_message():
    with tempfile.TemporaryDirectory() as root:
        _tree(root, GOOD)
        with pytest.raises(SystemExit) as caught:
            train.preflight_dataset(_yaml(root, ["nowhere/a", "golden_set/images/val"],
                                          ["golden_set/images/val"]))
        text = str(caught.value)
        assert "does not exist" in text and "in both train and val" in text


def test_preflight_runs_after_download_and_filter_before_training():
    src = (ROOT / "model_training" / "training-container" / "train.py").read_text(encoding="utf-8")
    assert src.index("data_yaml = download_dataset_with_progress(")         < src.index("data_yaml, _filtered = build_filtered_dataset(")         < src.index("preflight_dataset(data_yaml, classes_filter=args.classes)")         < src.index("results = model.train(")




# --- the card shows images, not objects ------------------------------------------

def test_image_counts_are_per_folder_and_per_split(bucket, monkeypatch):
    monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: [
        {"Key": prefix + n} for n in (["a.jpg", "b.jpg"] if "train" in prefix else ["c.jpg"])]
        + [{"Key": k, "Size": 1} for k in []])
    out = dy._image_counts(["Testing_Dataset/ds/images/train", "golden_set/images/train"],
                           ["golden_set/images/val"], "b")
    assert out["train"]["total"] == 4
    assert out["val"]["folders"] == [{"folder": "golden_set/images/val", "images": 1}]


def test_label_files_are_not_counted_as_images(monkeypatch):
    monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: [
        {"Key": prefix + "a.jpg"}, {"Key": prefix + "a.txt"}, {"Key": prefix + "x.cache"}])
    assert dy._image_counts(["d/images/val"], [], "b")["train"]["total"] == 1


def test_an_unlistable_folder_is_unknown_not_zero(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("denied")
    monkeypatch.setattr(pc, "_list_objects", boom)
    out = dy._image_counts(["d/images/val"], [], "b")
    assert out["train"]["folders"][0]["images"] is None and out["train"]["total"] is None


def test_the_card_shows_them_and_drops_the_misleading_lines():
    panel = " ".join((ROOT / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8").split())
    assert "{preview.image_counts && (" in panel
    assert "objects in the folders you picked;" not in panel
    assert "The training job above validates on its own images" not in panel
    assert "{!preview.many_roots && <p" in panel
