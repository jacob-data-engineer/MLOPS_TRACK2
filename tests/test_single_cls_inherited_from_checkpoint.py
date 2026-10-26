"""A checkpoint that calls its one class "item" (2026-10-26).

Ultralytics re-applies `single_cls` from the STARTING checkpoint's saved
args (Model._reset_ckpt_args keeps imgsz, data, task, single_cls). v14 of
yolo11s-helmet-detector was trained single-class, so every continuation
trained single-class even with SingleCls off -- and the evaluator refused
the result: "candidate checkpoint classes {0: 'item'} do not match the
dataset's {0: 'helmet'}". Two fixes: the trainer always states
single_cls; the evaluator accepts a one-class "item" checkpoint against a
one-class dataset.
"""
import inspect
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model_training" / "custom_training"))
sys.path.insert(0, str(ROOT / "model_training" / "training-container"))

ev = pytest.importorskip("evaluate_any")
train = pytest.importorskip("train")


class _Box:
    mp, mr, map50, map = 0.5, 0.4, 0.45, 0.3
    ap_class_index = [0]

    def class_result(self, i):
        return (0.5, 0.4, 0.45, 0.3)


def _fake_yolo(names):
    class Y:
        def __init__(self, path):
            self.names = names

        def val(self, **kw):
            return types.SimpleNamespace(box=_Box())
    return Y


def test_an_item_checkpoint_is_scored_against_a_one_class_dataset(monkeypatch):
    import ultralytics
    monkeypatch.setattr(ultralytics, "YOLO", _fake_yolo({0: "item"}))
    out = ev.detection("m.pt", "d.yaml", "candidate", {0: "helmet"})
    assert out["classes"] == {0: "helmet"}
    assert set(out["per_class"]) == {"helmet"}


def test_a_real_mismatch_is_still_refused(monkeypatch):
    import ultralytics
    monkeypatch.setattr(ultralytics, "YOLO", _fake_yolo({0: "person"}))
    with pytest.raises(ValueError, match="do not match"):
        ev.detection("m.pt", "d.yaml", "candidate", {0: "helmet"})
    monkeypatch.setattr(ultralytics, "YOLO", _fake_yolo({0: "item"}))
    with pytest.raises(ValueError, match="do not match"):
        ev.detection("m.pt", "d.yaml", "candidate", {0: "helmet", 1: "vest"})


def test_the_trainer_always_states_single_cls():
    code = inspect.getsource(train.main)
    assert 'extra["single_cls"] = bool(args.single_cls)' in code
