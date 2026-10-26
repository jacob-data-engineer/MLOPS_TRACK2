"""A pull refuses to file images that have no labels.

Images with no label are BACKGROUND examples -- Ultralytics accepts them
deliberately and a few improve precision, so nothing downstream errors. A
whole side of them trains the model on nothing, quietly, and the first
sign is a metric that will not move.

Enforced in the sweep rather than only in the card, because the sweep
that matters is the one nobody is watching: a scheduled pull at 18:33
would otherwise report success and leave a retrain that can never improve.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402


def picks(**kw):
    base = {"train_images": [], "train_labels": [], "val_images": [], "val_labels": []}
    base.update(kw)
    return {"picked": base}


class TestWhichSidesAreUnpaired:
    def test_images_with_no_labels_is_reported(self):
        assert pc.unpaired_sides(picks(train_images=["a"])) == ["train"]

    def test_both_sides_can_be_reported(self):
        assert pc.unpaired_sides(
            picks(train_images=["a"], val_images=["b"])) == ["train", "val"]

    def test_a_paired_side_is_fine(self):
        assert pc.unpaired_sides(picks(train_images=["a"], train_labels=["b"])) == []

    def test_labels_with_no_images_is_not_this_problem(self):
        # A label with no image is a mistake, but a different one: it
        # describes nothing and the trainer never reads it. It does not
        # fill a week with unlabelled pictures, so it does not block.
        assert pc.unpaired_sides(picks(train_labels=["b"])) == []

    def test_nothing_picked_is_not_unpaired(self):
        # "Add at least one folder" is a separate check; this must not
        # fire on an empty config and mask it.
        assert pc.unpaired_sides(picks()) == []

    def test_whitespace_is_not_a_folder(self):
        assert pc.unpaired_sides(picks(train_images=["a"], train_labels=["   "])) == ["train"]

    def test_the_val_side_alone(self):
        assert pc.unpaired_sides(
            picks(train_images=["a"], train_labels=["b"], val_images=["c"])) == ["val"]


class TestTheSweepRefuses:
    def test_it_raises_before_moving_anything(self, monkeypatch):
        monkeypatch.setattr(pc, "read_config", lambda: picks(train_images=["a"]))
        # If it got as far as listing weeks the guard came too late.
        monkeypatch.setattr(pc, "list_weeks", lambda: pytest.fail("got past the guard"))
        with pytest.raises(ValueError) as caught:
            pc.route_pending()
        assert "train side has image folders picked and no label folder" in str(caught.value)

    def test_the_message_says_what_to_do(self, monkeypatch):
        monkeypatch.setattr(pc, "read_config", lambda: picks(val_images=["c"]))
        monkeypatch.setattr(pc, "list_weeks", lambda: [])
        with pytest.raises(ValueError) as caught:
            pc.route_pending()
        detail = str(caught.value)
        assert "background examples" in detail
        assert "Pick the val label folder" in detail

    def test_a_paired_config_is_not_refused(self, monkeypatch):
        monkeypatch.setattr(
            pc, "read_config",
            lambda: picks(train_images=["a"], train_labels=["b"]))
        monkeypatch.setattr(pc, "list_weeks", lambda: [])
        monkeypatch.setattr(pc, "count_pending", lambda *a, **k: 0)
        monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
        monkeypatch.setattr(pc, "_notify_pull_finished", lambda *a, **k: None)
        monkeypatch.setattr(pc, "splits_per_root", lambda *a, **k: {})
        monkeypatch.setattr(pc, "_list_objects", lambda *a, **k: iter(()))
        # Must not raise. What it returns is route_pending's business.
        pc.route_pending()
