"""The evaluator must be told how the trainer narrowed the dataset.

A one-class detector trained from a six-class corpus uses Ultralytics'
load-time filtering: `classes=[0]` drops every other box, `single_cls=True`
renumbers the survivors to 0 and builds a one-output head named "item".

The evaluation step was never given those two values, so it read six
names from data.yaml, loaded a one-class checkpoint, and raised its own
consistency check -- AFTER a successful seven-minute training step, every
single time. Nothing could ever reach RegisterYoloModel.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "model_training" / "custom_training" / "evaluate_any.py"
PIPELINE = ROOT / "model_training" / "custom_training" / "pipeline_open.json"


@pytest.fixture(scope="module")
def ev():
    spec = importlib.util.spec_from_file_location("evaluate_any_filter_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"evaluate_any.py needs deps not installed here: {exc}")
    return module


class FakeBox:
    mp = 0.8
    mr = 0.6
    map50 = 0.7
    map = 0.5
    ap_class_index = [0]

    @staticmethod
    def class_result(index):
        return (0.8, 0.6, 0.7, 0.5)


class FakeResult:
    box = FakeBox()


class FakeModel:
    """Records what val() was handed, which is the whole point."""
    last_val_kwargs: dict = {}

    def __init__(self, names):
        self.names = names

    def val(self, **kwargs):
        FakeModel.last_val_kwargs = kwargs
        return FakeResult()


def _patch_yolo(monkeypatch, ev, names):
    import types
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda path: FakeModel(names)
    monkeypatch.setitem(sys.modules, "ultralytics", fake)


def test_a_single_class_checkpoint_is_accepted(monkeypatch, ev, tmp_path):
    """The regression. {0: 'item'} against a six-class yaml used to raise."""
    _patch_yolo(monkeypatch, ev, {0: "item"})
    dataset = {0: "helmet", 1: "gloves", 2: "person"}
    out = ev.detection(tmp_path / "model.pt", tmp_path / "data.yaml", "candidate",
                       dataset, classes=[0], single_cls=True, filtered_name="helmet")
    assert out["metrics"]["mAP50"] == 0.7


def test_the_same_narrowing_is_applied_at_validation(monkeypatch, ev, tmp_path):
    """Or the model is scored against boxes it was never taught to
    predict: every gloves and person box would count as a miss."""
    _patch_yolo(monkeypatch, ev, {0: "item"})
    ev.detection(tmp_path / "m.pt", tmp_path / "d.yaml", "candidate",
                 {0: "helmet", 1: "gloves"}, classes=[0], single_cls=True)
    assert FakeModel.last_val_kwargs["classes"] == [0]
    assert FakeModel.last_val_kwargs["single_cls"] is True
    assert FakeModel.last_val_kwargs["split"] == "val"


def test_an_unfiltered_run_passes_nothing_extra(monkeypatch, ev, tmp_path):
    """The path every run took before filtering existed must not change."""
    names = {0: "helmet", 1: "gloves"}
    _patch_yolo(monkeypatch, ev, names)
    ev.detection(tmp_path / "m.pt", tmp_path / "d.yaml", "candidate", names)
    assert "classes" not in FakeModel.last_val_kwargs
    assert "single_cls" not in FakeModel.last_val_kwargs


def test_the_strict_check_still_catches_a_real_mismatch(monkeypatch, ev, tmp_path):
    """Unfiltered, a checkpoint whose class ORDER differs produces numbers
    attributed to the wrong labels. That check must survive."""
    _patch_yolo(monkeypatch, ev, {0: "gloves", 1: "helmet"})
    with pytest.raises(ValueError, match="do not match"):
        ev.detection(tmp_path / "m.pt", tmp_path / "d.yaml", "candidate",
                     {0: "helmet", 1: "gloves"})


def test_a_single_cls_run_whose_checkpoint_has_many_classes_is_refused(monkeypatch, ev, tmp_path):
    """Says the run was single-class and the artifact is not: something
    other than what was asked for got trained."""
    _patch_yolo(monkeypatch, ev, {0: "helmet", 1: "gloves"})
    with pytest.raises(ValueError, match="single-class"):
        ev.detection(tmp_path / "m.pt", tmp_path / "d.yaml", "candidate",
                     {0: "helmet", 1: "gloves"}, classes=[0], single_cls=True)


def test_the_class_is_reported_by_its_real_name(monkeypatch, ev, tmp_path):
    """Ultralytics calls it 'item'. A metrics table saying 'item' tells
    nobody which detector this was."""
    _patch_yolo(monkeypatch, ev, {0: "item"})
    out = ev.detection(tmp_path / "m.pt", tmp_path / "d.yaml", "candidate",
                       {0: "helmet"}, classes=[0], single_cls=True,
                       filtered_name="helmet")
    assert "helmet" in out["per_class"]
    assert "item" not in out["per_class"]


def test_the_pipeline_passes_the_filter_to_the_evaluation_step():
    """The file and the live definition are two different things, and
    this only covers the file -- scripts/deploy_pipelines.py is what
    reconciles them. Without this, the fix above is unreachable."""
    definition = json.loads(PIPELINE.read_text(encoding="utf-8"))
    step = next(s for s in definition["Steps"] if s["Name"] == "EvaluateYolo")
    env = step["Arguments"]["Environment"]
    assert env["CLASSES"] == {"Get": "Parameters.Classes"}
    assert env["SINGLE_CLS"] == {"Get": "Parameters.SingleCls"}
    assert env["EVAL_IS_SEPARATE"] == {"Get": "Parameters.EvaluationDatasetIsSeparate"}
    names = {p["Name"] for p in definition["Parameters"]}
    assert {"Classes", "SingleCls", "EvaluationDatasetIsSeparate"} <= names


def test_the_submitter_says_whether_the_eval_set_is_separate():
    """So the report can state that the metrics are selection-biased
    rather than leaving a reader to assume the generous reading."""
    import inspect
    from ppe_pipeline_api import track2_shared as t2
    source = inspect.getsource(t2.start_custom_training_run)
    assert '"EvaluationDatasetIsSeparate"' in source
