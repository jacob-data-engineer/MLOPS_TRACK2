"""Code review 2026-10-26, findings F1, F4 and F5: the dataset contract.

F1 -- rebuilding a generated yaml must not lose the source's class-ID
      remap. A source's own data.yaml is the only record of what its label
      numbers mean; once Assemble replaced it, the next build saw "ours",
      ignored it, and silently dropped the remap -- valid indices, wrong
      classes, nothing erroring.
F4 -- train and val under SIBLING roots of one ancestor (jp/a, jp/b) must
      keep the real validation folders, not fall back to the training ones.
F5 -- every yaml entry must resolve against the layout the trainer
      actually downloads to: ONE prefix is stripped, several in one bucket
      keep full keys, several buckets keep <bucket>/<key>.

The F4/F5 checks run the generated yaml through the trainer's REAL
`download_dataset_with_progress` against a fake S3, then look for the
images and labels on disk -- not at strings in the yaml.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "model_training" / "training-container"))

from ppe_pipeline_api import dataset_yaml as dy      # noqa: E402
from ppe_pipeline_api import dataset_snapshots as ds  # noqa: E402
from ppe_pipeline_api import pull_control as pc      # noqa: E402

train = pytest.importorskip("train")

LABEL = "0 .5 .5 .1 .1\n"


class _Body:
    def __init__(self, b):
        self._b = b

    def read(self):
        return self._b


class _NoSuchKey(Exception):
    pass


class FakeS3:
    """A bucket world that answers get/put/list/download like boto3."""

    def __init__(self, files: dict):
        self.files = {k: (v.encode() if isinstance(v, str) else v) for k, v in files.items()}
        self.puts: list = []

    # app side
    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.files:
            raise _NoSuchKey(f"{Bucket}/{Key}")
        return {"Body": _Body(self.files[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, **kw):
        self.puts.append((Bucket, Key))
        self.files[(Bucket, Key)] = Body if isinstance(Body, bytes) else Body.encode()

    # trainer side
    def get_paginator(self, _name):
        world = self

        class _P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v)}
                                    for (b, k), v in sorted(world.files.items())
                                    if b == Bucket and k.startswith(Prefix)]}
        return _P()

    def download_file(self, Bucket, Key, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(self.files[(Bucket, Key)])


@pytest.fixture
def world(monkeypatch):
    def make(files):
        s3 = FakeS3(files)
        monkeypatch.setattr(pc, "s3", s3)
        monkeypatch.setattr(pc, "_list_objects", lambda prefix, bucket="", **kw: [
            {"Key": k, "Size": 1} for (b, k) in s3.files if b == (bucket or "home") and k.startswith(prefix)])
        monkeypatch.setattr(pc, "destination_bucket", lambda: "home")
        return s3
    return make


def S(*dirs):
    return [{"images": d, "labels": ""} for d in dirs]


def _doc(text):
    return yaml.safe_load("\n".join(l for l in text.splitlines() if not l.startswith("#")))


def _dataset(bucket, root, splits=("train", "val"), stem="x"):
    out = {}
    for s in splits:
        out[(bucket, f"{root}/images/{s}/{stem}_{s}.jpg")] = b"img"
        out[(bucket, f"{root}/labels/{s}/{stem}_{s}.txt")] = LABEL
    return out


SWAPPED = "names:\n  0: person\n  1: helmet\n"
WANTED = {0: "helmet", 1: "person"}


def _build(s3, *, train_dirs, val_dirs, classes=WANTED, write=True, overwrite=True):
    fn = dy.write_in_place if write else dy.build_in_place
    return fn(splits={"train": S(*train_dirs), "val": S(*val_dirs)}, classes=classes,
              bucket="home", overwrite=overwrite, label_scan=0)


def _remap_of(out):
    return _doc(out["text"]).get("x_label_remap") or {}


# --- F1: the remap survives the yaml being replaced ---------------------------

def test_the_remap_is_the_same_on_the_second_build(world):
    s3 = world({**_dataset("home", "ds"), ("home", "ds/data.yaml"): SWAPPED})
    first = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    assert _remap_of(first) == {"labels/train": {0: 1, 1: 0}, "labels/val": {0: 1, 1: 0}}
    # The source's own yaml was replaced...
    assert dy.IN_PLACE_MARKER in s3.files[("home", "ds/data.yaml")].decode()
    second = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    # ...and the second build still knows what the label numbers mean.
    assert _remap_of(second) == _remap_of(first)


def test_the_original_yaml_is_kept_verbatim_before_it_is_replaced(world):
    s3 = world({**_dataset("home", "ds"), ("home", "ds/data.yaml"): SWAPPED})
    _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    assert s3.files[("home", f"ds/{dy.SOURCE_TAXONOMY_FILE}")].decode() == SWAPPED
    # Never overwritten by a later build.
    _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    assert s3.files[("home", f"ds/{dy.SOURCE_TAXONOMY_FILE}")].decode() == SWAPPED


def test_a_new_output_order_translates_from_the_original_ids(world):
    s3 = world({**_dataset("home", "ds"), ("home", "ds/data.yaml"): SWAPPED})
    _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    # Third build, a THIRD order: person 0->2, helmet 1->0.
    third = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"],
                   classes={0: "helmet", 1: "vest", 2: "person"})
    assert _remap_of(third)["labels/train"] == {0: 2, 1: 0}
    # And back to the source's own order: no remap at all.
    same = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"],
                  classes={0: "person", 1: "helmet"})
    assert _remap_of(same) == {}


def test_a_legacy_generated_yaml_is_recovered_from_its_own_remap(world):
    # Written before the original was kept: ours, with the remap in it.
    legacy = "\n".join([dy.IN_PLACE_MARKER, yaml.safe_dump({
        "path": "s3://home/ds", "train": ["images/train"], "val": ["images/val"], "nc": 2,
        "names": {0: "helmet", 1: "person"},
        "x_label_remap": {"labels/train": {0: 1, 1: 0}, "labels/val": {0: 1, 1: 0}}})])
    s3 = world({**_dataset("home", "ds"), ("home", "ds/data.yaml"): legacy})
    again = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    assert _remap_of(again)["labels/train"] == {0: 1, 1: 0}


def test_an_unrecoverable_taxonomy_is_refused_not_guessed(world):
    # Ours, no remap, no kept original -- and the class order changed.
    legacy = "\n".join([dy.IN_PLACE_MARKER, yaml.safe_dump({
        "path": "s3://home/ds", "train": ["images/train"], "val": ["images/val"], "nc": 2,
        "names": {0: "person", 1: "helmet"}})])
    s3 = world({**_dataset("home", "ds"), ("home", "ds/data.yaml"): legacy})
    with pytest.raises(ValueError, match="cannot tell what"):
        _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"], write=False)
    # The same order as before needs no remap and still builds.
    ok = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"],
                classes={0: "person", 1: "helmet"}, write=False)
    assert _remap_of(ok) == {}


def test_the_separate_evaluation_yaml_keeps_its_remap_too(world):
    # Val spans two roots: the evaluation yaml is written AT golden_set/,
    # replacing golden_set's own (swapped) yaml.
    s3 = world({**_dataset("home", "ds"), **_dataset("home", "golden_set", ("val",), "g"),
                ("home", "golden_set/data.yaml"): SWAPPED})
    dirs = dict(train_dirs=["ds/images/train"], val_dirs=["ds/images/val", "golden_set/images/val"])
    first = _build(s3, **dirs)
    second = _build(s3, **dirs)
    assert first["eval"]["key"] == "golden_set/data.yaml"
    assert _remap_of(first["eval"]) and _remap_of(second["eval"]) == _remap_of(first["eval"])
    assert _remap_of(second) == _remap_of(first) != {}
    assert s3.files[("home", f"golden_set/{dy.SOURCE_TAXONOMY_FILE}")].decode() == SWAPPED


def test_the_trainer_applies_it_and_s3_labels_are_untouched(world, tmp_path, monkeypatch):
    s3 = world({**_dataset("home", "ds"), ("home", "ds/data.yaml"): SWAPPED})
    before = {k: v for k, v in s3.files.items() if "/labels/" in k[1]}
    _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    out = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"])
    assert {k: v for k, v in s3.files.items() if "/labels/" in k[1]} == before
    assert not [p for p in s3.puts if "/labels/" in p[1]]
    local = _download(monkeypatch, s3, tmp_path, out["download_prefix"], out["uri"])
    new_yaml, _ = train.build_filtered_dataset(str(tmp_path / "d"), local)
    root = pathlib.Path(new_yaml).parent
    # Source class 0 is "person"; in the declared list person is 1.
    assert (root / "labels/train/x_train.txt").read_text().split()[0] == "1"


# --- the trainer's download, for F4/F5 ----------------------------------------

def _download(monkeypatch, s3, tmp_path, uris, yaml_uri):
    monkeypatch.setattr(train.time, "sleep", lambda s: None)
    monkeypatch.setattr(train.boto3, "client", lambda *a, **k: s3)
    dest = tmp_path / "d"
    dest.mkdir(exist_ok=True)
    return train.download_dataset_with_progress(uris, "ap-southeast-1", str(dest),
                                                data_yaml_s3_uri=yaml_uri)


def _resolve(yaml_path, split):
    doc = yaml.safe_load(open(yaml_path))
    base = pathlib.Path(doc["path"])
    entries = doc[split] if isinstance(doc[split], list) else [doc[split]]
    return [p if p.is_absolute() else base / p for p in map(pathlib.Path, entries)]


def _images_under(dirs):
    return {f.name for d in dirs for f in d.rglob("*.jpg")}


def _assert_lands(monkeypatch, s3, tmp_path, out, train_names, val_names):
    # A preview writes nothing; put the yaml where the run would read it.
    s3.files[(out["bucket"], out["key"])] = out["text"].encode()
    local = _download(monkeypatch, s3, tmp_path, out["download_prefix"], out["uri"])
    tr, va = _resolve(local, "train"), _resolve(local, "val")
    for d in tr + va:
        assert d.is_dir(), f"yaml names {d}, which the download did not create"
        labels = pathlib.Path(str(d).replace(f"{os.sep}images", f"{os.sep}labels"))
        assert labels.is_dir(), f"no labels beside {d}"
    assert _images_under(tr) == set(train_names)
    assert _images_under(va) == set(val_names)
    assert not (_images_under(tr) & _images_under(va))


# --- F4: sibling roots keep the real validation folders -----------------------

def test_sibling_roots_validate_on_the_real_val_folders(world, tmp_path, monkeypatch):
    s3 = world({**_dataset("home", "jp/a", ("train",), "a"), **_dataset("home", "jp/b", ("val",), "b")})
    out = _build(s3, train_dirs=["jp/a/images/train"], val_dirs=["jp/b/images/val"], write=False)
    doc = _doc(out["text"])
    assert doc["val"] != doc["train"]
    _assert_lands(monkeypatch, s3, tmp_path, out, {"a_train.jpg"}, {"b_val.jpg"})


def test_sibling_roots_keep_their_remap(world):
    s3 = world({**_dataset("home", "jp/a", ("train",), "a"), **_dataset("home", "jp/b", ("val",), "b"),
                ("home", "jp/a/data.yaml"): SWAPPED, ("home", "jp/b/data.yaml"): SWAPPED})
    out = _build(s3, train_dirs=["jp/a/images/train"], val_dirs=["jp/b/images/val"], write=False)
    remap = _remap_of(out)
    assert remap and all(m == {0: 1, 1: 0} for m in remap.values())
    assert len(remap) == 2


def test_nested_and_ordinary_roots_still_land(world, tmp_path, monkeypatch):
    s3 = world({**_dataset("home", "jp", ("train",), "t"), **_dataset("home", "jp/sub", ("val",), "v")})
    out = _build(s3, train_dirs=["jp/images/train"], val_dirs=["jp/sub/images/val"], write=False)
    _assert_lands(monkeypatch, s3, tmp_path, out, {"t_train.jpg"}, {"v_val.jpg"})


def test_the_ordinary_same_root_case_is_unchanged(world, tmp_path, monkeypatch):
    s3 = world(_dataset("home", "ds"))
    out = _build(s3, train_dirs=["ds/images/train"], val_dirs=["ds/images/val"], write=False)
    doc = _doc(out["text"])
    assert (doc["train"], doc["val"]) == (["images/train"], ["images/val"])
    assert out["download_prefix"] == "s3://home/ds/"
    _assert_lands(monkeypatch, s3, tmp_path, out, {"x_train.jpg"}, {"x_val.jpg"})


def test_separate_top_level_roots_still_land(world, tmp_path, monkeypatch):
    s3 = world({**_dataset("home", "dsA", ("train",), "a"), **_dataset("home", "dsB", ("val",), "b")})
    out = _build(s3, train_dirs=["dsA/images/train"], val_dirs=["dsB/images/val"], write=False)
    assert len(out["download_prefixes"]) == 2
    _assert_lands(monkeypatch, s3, tmp_path, out, {"a_train.jpg"}, {"b_val.jpg"})


# --- F5: one foreign prefix, and the other download shapes --------------------

def test_one_prefix_in_a_foreign_bucket_lands(world, tmp_path, monkeypatch):
    s3 = world(_dataset("other", "site"))
    out = _build(s3, train_dirs=["s3://other/site/images/train"],
                 val_dirs=["s3://other/site/images/val"], write=False)
    assert out["download_prefix"] == "s3://other/site/"
    _assert_lands(monkeypatch, s3, tmp_path, out, {"x_train.jpg"}, {"x_val.jpg"})


def test_several_prefixes_in_one_foreign_bucket_land(world, tmp_path, monkeypatch):
    s3 = world({**_dataset("other", "s1", ("train",), "a"), **_dataset("other", "s2", ("val",), "b")})
    out = _build(s3, train_dirs=["s3://other/s1/images/train"],
                 val_dirs=["s3://other/s2/images/val"], write=False)
    assert len(out["download_prefixes"]) == 2
    _assert_lands(monkeypatch, s3, tmp_path, out, {"a_train.jpg"}, {"b_val.jpg"})


def test_home_plus_foreign_buckets_land(world, tmp_path, monkeypatch):
    s3 = world({**_dataset("home", "ds", ("train",), "a"), **_dataset("other", "site", ("val",), "b")})
    out = _build(s3, train_dirs=["ds/images/train"], val_dirs=["s3://other/site/images/val"],
                 write=False)
    _assert_lands(monkeypatch, s3, tmp_path, out, {"a_train.jpg"}, {"b_val.jpg"})


def test_several_home_folders_under_one_parent_land(world, tmp_path, monkeypatch):
    # Two TRAINING roots that share `jp`: one download prefix, so entries
    # must be relative to it -- the same shape as F5, in the home bucket.
    s3 = world({**_dataset("home", "jp/a", ("train",), "a"), **_dataset("home", "jp/b", ("train",), "b"),
                **_dataset("home", "jp/c", ("val",), "c")})
    out = _build(s3, train_dirs=["jp/a/images/train", "jp/b/images/train"],
                 val_dirs=["jp/c/images/val"], write=False)
    _assert_lands(monkeypatch, s3, tmp_path, out, {"a_train.jpg", "b_train.jpg"}, {"c_val.jpg"})


def test_the_snapshot_records_files_where_the_trainer_puts_them():
    one = ds.composed_local_paths(["s3://other/site"], "home", {"s3://other/site": ["images/train/x.jpg"]})
    assert one == ["images/train/x.jpg"]
    many = ds.composed_local_paths(["dsA", "dsB"], "home",
                                   {"dsA": ["images/train/a.jpg"], "dsB": ["images/val/b.jpg"]})
    assert many == ["dsA/images/train/a.jpg", "dsB/images/val/b.jpg"]
    mixed = ds.composed_local_paths(["ds", "s3://other/site"], "home",
                                    {"ds": ["images/train/a.jpg"], "s3://other/site": ["images/val/b.jpg"]})
    assert mixed == ["home/ds/images/train/a.jpg", "other/site/images/val/b.jpg"]
