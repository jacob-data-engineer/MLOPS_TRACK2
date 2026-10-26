"""Class-ID remapping across sources, several buckets, and the yaml bug.

1. A source whose own data.yaml numbers its classes differently is
   REMAPPED by name, in the run's downloaded copy; S3 is never changed.
   A class the declared list lacks is dropped with the filter on and
   refused otherwise.
2. Folders may come from several buckets. Every entry is then
   <bucket>/<key> and the trainer lays files out the same way.
3. The training yaml returned for val folders under two roots was the
   EVALUATION yaml (`train:` = the val folders): the eval code reused
   `text`. Pinned here so it cannot come back.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest
import yaml

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

    def get_object(self, Bucket, Key):
        return {"Body": _Body(self.files[(Bucket, Key)])}


@pytest.fixture
def world(monkeypatch):
    files = {
        ("home", "ds/labels/train/a.txt"): "0 .5 .5 .1 .1\n",
        ("home", "ds/labels/val/b.txt"): "0 .5 .5 .1 .1\n",
        ("home", "golden_set/labels/val/g.txt"): "0 .5 .5 .1 .1\n",
        ("other", "site/labels/train/c.txt"): "1 .5 .5 .1 .1\n",
        ("other", "site/data.yaml"): "names:\n  0: person\n  1: helmet\n",
    }
    monkeypatch.setattr(pc, "s3", _S3(files))
    monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: [
        {"Key": k, "Size": 1} for (b, k) in files if b == (bucket or "home") and k.startswith(prefix)])
    monkeypatch.setattr(pc, "destination_bucket", lambda: "home")
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    return files


def S(*dirs):
    return [{"images": d, "labels": ""} for d in dirs]


def _doc(text):
    return yaml.safe_load("\n".join(l for l in text.splitlines() if not l.startswith("#")))


# --- 3. the bug ------------------------------------------------------------------

def test_the_training_yaml_is_the_training_yaml_when_val_spans_two_roots(world):
    out = dy.build_in_place(
        splits={"train": S("ds/images/train"), "val": S("ds/images/val", "golden_set/images/val")},
        classes={0: "helmet"}, bucket="home")
    doc = _doc(out["text"])
    assert doc["train"] == ["ds/images/train"], "training yaml must train on the train folders"
    assert doc["val"] == ["ds/images/val", "golden_set/images/val"]
    assert "EVALUATION ONLY" not in out["text"]
    assert "EVALUATION ONLY" in out["eval"]["text"]


# --- 1. remapping ----------------------------------------------------------------

def test_a_source_numbering_differently_is_remapped_by_name(world):
    out = dy.build_in_place(
        splits={"train": S("ds/images/train", "s3://other/site/images/train"),
                "val": S("ds/images/val")},
        classes={0: "helmet", 1: "person"}, bucket="home")
    doc = _doc(out["text"])
    # other's 0 'person' -> 1, 1 'helmet' -> 0
    assert doc["x_label_remap"] == {"other/site/labels/train": {0: 1, 1: 0}}


def test_a_class_the_list_lacks_is_refused_without_the_filter(world):
    with pytest.raises(ValueError, match="numbers its classes differently and has 0 'person'"):
        dy.build_in_place(
            splits={"train": S("s3://other/site/images/train"), "val": S("ds/images/val")},
            classes={0: "helmet"}, bucket="home")


def test_with_the_filter_it_is_mapped_to_drop(world):
    out = dy.build_in_place(
        splits={"train": S("s3://other/site/images/train"), "val": S("ds/images/val")},
        classes={0: "helmet"}, bucket="home", drop_undeclared=True)
    assert _doc(out["text"])["x_label_remap"] == {"other/site/labels/train": {0: -1, 1: 0}}


def test_remapped_sources_are_not_flagged_as_undeclared(world):
    """Their indices are their own; translated before Ultralytics reads."""
    out = dy.build_in_place(
        splits={"train": S("s3://other/site/images/train"), "val": S("ds/images/val")},
        classes={0: "helmet"}, bucket="home", drop_undeclared=True)
    assert out["undeclared"] == []


def test_the_trainer_applies_the_remap_to_its_copy_only():
    with tempfile.TemporaryDirectory() as root:
        lab = os.path.join(root, "other", "site", "labels", "train")
        os.makedirs(lab)
        with open(os.path.join(lab, "c.txt"), "w") as f:
            f.write("0 .1 .1 .1 .1\n1 .2 .2 .1 .1\n")
        y = os.path.join(root, "data.yaml")
        with open(y, "w") as f:
            f.write("names:\n  0: helmet\nx_label_remap:\n  other/site/labels/train:\n    0: -1\n    1: 0\n")
        got = train.apply_label_remap(root, y)
        assert got == {"files": 1, "sources": 1}
        assert open(os.path.join(lab, "c.txt")).read() == "0 .2 .2 .1 .1\n"


def test_the_remap_runs_before_the_filter_and_the_preflight():
    """One step now: build_filtered_dataset remaps, then filters, then
    renumbers, on a temporary copy -- before preflight and training."""
    src = (ROOT / "model_training" / "training-container" / "train.py").read_text(encoding="utf-8")
    assert src.index("data_yaml = download_dataset_with_progress(")         < src.index("data_yaml, _filtered = build_filtered_dataset(")         < src.index("preflight_dataset(data_yaml, classes_filter=args.classes)")         < src.index("results = model.train(")


def test_the_evaluator_applies_it_too():
    src = (ROOT / "model_training" / "custom_training" / "evaluate_any.py").read_text(encoding="utf-8")
    assert "def build_filtered_dataset(" in src
    assert 'build_filtered_dataset(str(data), str(data / "data.yaml"),' in src


def test_two_buckets_qualify_every_entry_with_its_bucket(world):
    out = dy.build_in_place(
        splits={"train": S("ds/images/train", "s3://other/site/images/train"),
                "val": S("ds/images/val")},
        classes={0: "helmet", 1: "person"}, bucket="home")
    assert out["multi_bucket"] is True
    doc = _doc(out["text"])
    assert doc["train"] == ["home/ds/images/train", "other/site/images/train"]
    assert doc["val"] == ["home/ds/images/val"]
    assert sorted(out["download_prefixes"]) == ["s3://home/ds/", "s3://other/site/"]


def test_one_bucket_is_unchanged(world):
    out = dy.build_in_place(
        splits={"train": S("ds/images/train"), "val": S("ds/images/val")},
        classes={0: "helmet"}, bucket="home")
    assert not out.get("multi_bucket")
    assert _doc(out["text"])["train"] == ["images/train"]


def test_the_home_bucket_spelled_as_a_uri_is_the_same_folder(world):
    out = dy.build_in_place(
        splits={"train": S("s3://home/ds/images/train"), "val": S("ds/images/val")},
        classes={0: "helmet"}, bucket="home")
    assert not out.get("multi_bucket")


def test_the_trainer_lays_several_buckets_out_as_bucket_key():
    src = (ROOT / "model_training" / "training-container" / "train.py").read_text(encoding="utf-8")
    assert "multi_bucket = len({_parse_s3_uri(u)[0] for u in uris}) > 1" in src
    assert "f\"{one_bucket}/{obj['Key']}\" if multi_bucket" in src


def test_an_evaluation_folder_in_another_bucket_travels_as_a_uri(world):
    out = dy.build_in_place(
        splits={"train": S("ds/images/train"), "val": S("s3://other/site/images/train")},
        classes={0: "helmet", 1: "person"}, bucket="home")
    assert out["eval"]["folder"] == "s3://other/site"
    assert out["eval"]["bucket"] == "other"


def test_each_yaml_is_written_to_its_own_bucket():
    import inspect
    code = inspect.getsource(dy.write_in_place)
    assert 'Bucket=side.get("bucket") or out["bucket"]' in code


def test_the_freeze_keeps_another_buckets_root_as_a_uri():
    import inspect
    from ppe_pipeline_api import retrain_schedule as rs, dataset_snapshots as ds
    assert 'u.split("/", 3)[2] == out["bucket"] else u.rstrip("/")' in inspect.getsource(rs._run_chain_body)
    assert 'if extra.startswith("s3://"):' in inspect.getsource(ds.create)


# --- the card ------------------------------------------------------------------

PANEL = " ".join((ROOT / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8").split())


def test_the_picker_offers_a_bucket_when_there_is_more_than_one():
    assert "{buckets.length > 1 && (" in PANEL
    assert '{b === homeBucket ? " (home)" : ""}' in PANEL


def test_other_buckets_are_listed_through_the_generic_listing():
    assert "listTrack2Folders(pickBucket, pickPrefix)" in PANEL


def test_a_folder_from_another_bucket_is_stored_as_a_uri():
    assert "`s3://${pickBucket}/" in PANEL
    assert "setSource(picking.side, picking.index, picking.field, qualified(p));" in PANEL
