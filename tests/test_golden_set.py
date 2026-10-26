"""Splitting a curated folder into train and val.

The rules worth testing are the silent ones. An image whose label went to
the other split trains as an unannotated background example and evaluates
as a missed detection, and nothing errors either time.
"""

from __future__ import annotations

import json

import pytest

from ppe_pipeline_api import golden_set as gs


class FakeS3:
    def __init__(self, objects: dict):
        self.objects = dict(objects)
        self.copied: list = []
        self.put: list = []

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        return {"Contents": [{"Key": k, "Size": v}
                             for k, v in sorted(self.objects.items())
                             if k.startswith(prefix)],
                "IsTruncated": False}

    def copy_object(self, **kwargs):
        self.copied.append((kwargs["CopySource"]["Key"], kwargs["Key"]))
        return {}

    def put_object(self, **kwargs):
        self.put.append((kwargs["Key"], kwargs["Body"]))
        return {}


def _world(monkeypatch, objects):
    s3 = FakeS3(objects)
    monkeypatch.setattr(gs.pc, "s3", s3)
    monkeypatch.setattr(gs.pc, "destination_bucket", lambda config=None: "track2-s3")
    return s3


def _curated(root, n, *, labels=True):
    out = {}
    for i in range(n):
        out[f"{root}/images/f{i:03d}.jpg"] = 100
        if labels:
            out[f"{root}/labels/f{i:03d}.txt"] = 10
    return out


def test_an_image_and_its_label_never_land_on_opposite_sides(monkeypatch):
    """The rule the whole module exists to keep.

    Split per file rather than per stem and roughly a fifth of the images
    lose their labels to the other split -- trained as backgrounds,
    evaluated as misses, silent both times.
    """
    _world(monkeypatch, _curated("golden", 50))
    out = gs.build(source="golden", target_folder="out", train_percent=70)

    sides: dict = {}
    for _, destination in gs.pc.s3.copied:
        parts = destination.split("/")
        kind, side, name = parts[1], parts[2], parts[3]
        stem = name.rsplit(".", 1)[0]
        sides.setdefault(stem, {})[kind] = side

    assert len(sides) == 50
    for stem, got in sides.items():
        assert got["images"] == got["labels"], f"{stem} split across sides"


def test_the_same_folder_always_splits_the_same_way(monkeypatch):
    """Reproducible, or a curated set cannot be compared with itself."""
    _world(monkeypatch, _curated("golden", 40))
    first = gs.plan(source="golden", target_folder="out", train_percent=75)
    _world(monkeypatch, _curated("golden", 40))
    again = gs.plan(source="golden", target_folder="out", train_percent=75)
    assert [s["stem"] for s in first["sample"]] == [s["stem"] for s in again["sample"]]

    train_a, val_a = gs.split_stems([f"f{i:03d}" for i in range(40)], 75)
    train_b, val_b = gs.split_stems(list(reversed([f"f{i:03d}" for i in range(40)])), 75)
    # Listing order must not change the division either.
    assert train_a == train_b and val_a == val_b


def test_the_slider_splits_pairs_not_files(monkeypatch):
    """80% of pairs, not 80% of the 2N files."""
    _world(monkeypatch, _curated("golden", 100))
    out = gs.plan(source="golden", target_folder="out", train_percent=80)
    assert out["pairs"] == 100
    assert out["train"] == 80
    assert out["val"] == 20
    assert out["files_to_copy"] == 200          # both halves of every pair


def test_the_split_can_be_turned_off(monkeypatch):
    """Off copies the folder whole, with no train/val level."""
    s3 = _world(monkeypatch, _curated("golden", 10))
    out = gs.build(source="golden", target_folder="out", split=False)

    assert out["train"] == 10 and out["val"] == 0
    destinations = [d for _, d in s3.copied]
    assert all("/train/" not in d and "/val/" not in d for d in destinations)
    assert any(d.startswith("out/images/") for d in destinations)
    assert any(d.startswith("out/labels/") for d in destinations)


def test_a_label_with_no_image_is_reported_and_ignored(monkeypatch):
    """Always a mistake, never a background example."""
    objects = _curated("golden", 5)
    objects["golden/labels/stray.txt"] = 10
    _world(monkeypatch, objects)

    out = gs.plan(source="golden", target_folder="out")
    assert out["pairs"] == 5
    assert out["orphan_labels"] == 1
    assert any("no image" in p["text"] for p in out["problems"])


def test_an_image_with_no_label_is_reported_separately(monkeypatch):
    """A legitimate background example, so it is a note and not an error."""
    objects = _curated("golden", 5)
    objects["golden/images/extra.jpg"] = 100
    _world(monkeypatch, objects)

    out = gs.plan(source="golden", target_folder="out")
    assert out["pairs"] == 5
    assert out["unpaired_images"] == 1
    note = [p for p in out["problems"] if "no label" in p["text"]]
    assert note and note[0]["level"] == "warn"


def test_a_folder_with_no_pairs_will_not_build(monkeypatch):
    _world(monkeypatch, _curated("golden", 6, labels=False))
    out = gs.plan(source="golden", target_folder="out")
    assert out["ready"] is False
    with pytest.raises(ValueError, match="no pairs to split"):
        gs.build(source="golden", target_folder="out")


def test_copying_a_folder_over_itself_is_refused(monkeypatch):
    _world(monkeypatch, _curated("golden", 4))
    out = gs.plan(source="golden", target_folder="golden")
    assert out["ready"] is False
    assert any("target is the source" in p["text"] for p in out["problems"])


def test_previewing_writes_nothing(monkeypatch):
    s3 = _world(monkeypatch, _curated("golden", 12))
    gs.plan(source="golden", target_folder="out", train_percent=50)
    assert s3.copied == [] and s3.put == []


def test_the_build_records_which_stems_went_where(monkeypatch):
    """The percentage alone does not say who fell where once the source grows."""
    s3 = _world(monkeypatch, _curated("golden", 20))
    out = gs.build(source="golden", target_folder="out", train_percent=60)

    written = {k: v for k, v in s3.put}
    assert "out/split.json" in written
    manifest = json.loads(written["out/split.json"].decode("utf-8"))
    assert len(manifest["train"]) == 12
    assert len(manifest["val"]) == 8
    assert set(manifest["train"]) & set(manifest["val"]) == set()
    assert out["copied"] == 40


def test_the_source_is_copied_never_moved(monkeypatch):
    """The curated folder is the record of somebody's judgement."""
    s3 = _world(monkeypatch, _curated("golden", 8))
    gs.build(source="golden", target_folder="out")
    assert not hasattr(s3, "deleted") or not getattr(s3, "deleted")
    assert all(src.startswith("golden/") for src, _ in s3.copied)
