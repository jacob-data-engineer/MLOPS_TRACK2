"""Tests for the person detector offered as an alternative to vendor boxes.

No weights are loaded: `find` is exercised through a fake model, because the
committed checkpoint is 40MB and ~5 seconds a frame on a CPU.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from ppe_pipeline import person_detector
from ppe_pipeline.config import PipelineConfig


def _frame(width=1920, height=1080) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (90, 90, 90)).save(buffer, format="JPEG")
    return buffer.getvalue()


class _Box:
    def __init__(self, xyxy):
        self.xyxy = [xyxy]


class _Result:
    def __init__(self, boxes):
        self.boxes = [_Box(b) for b in boxes]


class _FakeModel:
    def __init__(self, boxes):
        self._boxes = boxes
        self.calls = []

    def predict(self, _frame, **kwargs):
        self.calls.append(kwargs)
        return [_Result(self._boxes)]


def _detector(boxes, imgsz=1536):
    det = person_detector.PersonDetector(
        weights_path=None, min_confidence=0.25, device="cpu", imgsz=imgsz
    )
    det._model = _FakeModel(boxes)
    return det


def test_boxes_come_back_in_reading_order():
    """The same order `snapshot_boxes.extract_boxes` returns, so
    `person_index` means the same thing whichever source produced them."""
    det = _detector([(500, 300, 560, 420), (100, 20, 160, 140), (900, 25, 960, 145)])

    boxes = det.find(_frame())

    assert boxes == [(100, 20, 160, 140), (900, 25, 960, 145), (500, 300, 560, 420)]


def test_the_configured_inference_size_is_passed():
    """The model was TRAINED at 1536, and ultralytics' default of 640 would
    scale this corpus's ~83px workers to ~35px."""
    det = _detector([(1, 1, 2, 2)])

    det.find(_frame())

    assert det._model.calls[0]["imgsz"] == 1536
    assert det._model.calls[0]["conf"] == 0.25


def test_no_inference_size_is_passed_when_unset():
    det = _detector([(1, 1, 2, 2)], imgsz=None)

    det.find(_frame())

    assert "imgsz" not in det._model.calls[0]


def test_a_degenerate_box_is_dropped_not_returned():
    det = _detector([(10, 10, 10, 200), (20, 20, 80, 200)])

    assert det.find(_frame()) == [(20, 20, 80, 200)]


def test_a_failed_pass_yields_no_boxes_rather_than_raising():
    """The caller already handles an empty result: the frame settles as no
    violation with its own reason code and stays in the audit."""
    det = _detector([])

    class _Exploding:
        @staticmethod
        def predict(*_a, **_k):
            raise RuntimeError("cuda is on fire")

    det._model = _Exploding()

    assert det.find(_frame()) == []
    assert det.find(b"not an image") == []


def test_missing_weights_report_rather_than_raise(monkeypatch, tmp_path):
    """Whether a checkpoint exists is a fact about the deployment, so the
    caller refuses to START rather than failing per alarm."""
    config = PipelineConfig()
    config.person_detector.weights = str(tmp_path / "nope.pt")
    person_detector.reset_person_detector_cache()

    assert person_detector.get_person_detector(config) is None


def test_the_detector_is_loaded_once_per_configuration(monkeypatch):
    config = PipelineConfig()
    person_detector.reset_person_detector_cache()

    monkeypatch.setattr(person_detector, "resolve_weights_path", lambda w: _Existing())

    first = person_detector.get_person_detector(config)
    second = person_detector.get_person_detector(config)

    assert first is second


class _Existing:
    """A weights path that claims to exist, without a 40MB file behind it."""

    @staticmethod
    def is_file():
        return True

    @staticmethod
    def as_posix():
        return "weights/fake.pt"


# --------------------------------------------------------------------------
# WARMING, WHICH IS ALSO THE ONLY HEALTH CHECK THIS PATH HAS.
# --------------------------------------------------------------------------


class _BrokenAtInference:
    """Imports and builds, then dies on the first call. The common shape of a
    bad torch/ultralytics install, and the reason constructing the model
    proves nothing."""

    def predict(self, *_args, **_kwargs):
        raise RuntimeError(
            "Could not run 'torchvision::nms' with arguments from the 'CPU' backend"
        )


def _detector_for(model, imgsz=1536):
    """A detector whose model is injected through `_load`, so `warm` exercises
    the real load-then-predict path rather than a pre-set `_model`."""
    detector = person_detector.PersonDetector(Path("weights/best.pt"), 0.25, "cpu", imgsz)
    detector._load = lambda: model
    return detector


def test_warming_runs_a_real_inference_not_just_a_load(monkeypatch):
    """Ultralytics fuses and specialises on the FIRST predict, which is most
    of the 6.49s-vs-0.89s gap this exists to move off the first alarm. A
    warm-up that only constructed the model would leave that on it."""
    predicted = []

    class _Model:
        def predict(self, frame, **kwargs):
            predicted.append((frame.size, kwargs))
            return []

    _detector_for(_Model()).warm()

    assert len(predicted) == 1, "warm() never asked the model to infer"
    # The same settings `find` uses, or the warm-up proves the wrong thing:
    # `imgsz` in particular selects different code paths.
    assert predicted[0][1] == {"conf": 0.25, "device": "cpu", "verbose": False, "imgsz": 1536}


def test_warming_raises_where_find_would_swallow():
    """`find` returns no boxes on any failure, and no boxes is not an error
    downstream: the frame settles as "no person, no violation" with a reason
    code. So a broken install judged EVERY alarm that way, silently, forever.
    Warming is the one place that can say so."""
    detector = _detector_for(_BrokenAtInference())

    with pytest.raises(RuntimeError, match="torchvision::nms"):
        detector.warm()

    # The contrast this exists for: the same failure, through the path the
    # feed actually uses, is invisible.
    assert detector.find(_frame()) == []


def test_warming_needs_no_alarm_to_have_arrived():
    """It runs before the poll loop, so there is no frame yet -- and it must
    not wait for one to find out the model is broken."""
    sizes = []

    class _Model:
        def predict(self, frame, **_kwargs):
            sizes.append(frame.size)
            return []

    _detector_for(_Model()).warm()

    assert sizes and all(size == (64, 64) for size in sizes)


def test_the_warm_up_and_the_real_pass_cannot_drift(monkeypatch):
    """One source for the inference settings, so a change to `find` cannot
    leave `warm` proving something about a different code path."""
    seen = []

    class _Model:
        def predict(self, _frame, **kwargs):
            seen.append(kwargs)
            return []

    detector = _detector_for(_Model(), imgsz=None)
    detector.warm()
    detector.find(_frame())

    assert seen[0] == seen[1]
    # And `imgsz` is omitted rather than passed as None when unconfigured.
    assert "imgsz" not in seen[0]
