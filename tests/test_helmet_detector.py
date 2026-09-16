"""Tests for the stage-2 helmet detector.

No real weights are loaded anywhere here: `HelmetDetector._load` is bypassed
by setting `_model` directly, so nothing imports torch or reads the 22MB
checkpoint. What's under test is this module's own logic — pick the single
best box, normalize it, and never raise — not ultralytics'.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from PIL import Image

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.helmet_detector import (
    HelmetDetector,
    HelmetLocation,
    get_helmet_detector,
    reset_detector_cache,
    resolve_weights_path,
)


def _image_bytes(width: int = 200, height: int = 100) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color="white").save(buffer, format="JPEG")
    return buffer.getvalue()


class _FakeBox:
    def __init__(self, xyxy, conf):
        self.xyxy = [xyxy]
        self.conf = [conf]


class _FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class _FakeModel:
    """Shaped like the slice of `ultralytics.YOLO` this module actually uses."""

    def __init__(self, results, raises: Exception | None = None):
        self._results = results
        self._raises = raises
        self.predict_kwargs = None

    def predict(self, _frame, **kwargs):
        self.predict_kwargs = kwargs
        if self._raises is not None:
            raise self._raises
        return self._results


def _detector(results, raises=None, min_confidence=0.25):
    detector = HelmetDetector(Path("unused.pt"), min_confidence=min_confidence, device="cpu")
    detector._model = _FakeModel(results, raises)
    return detector


# --- picking the box ----------------------------------------------------


def test_returns_the_single_highest_confidence_box():
    """A cleared helmet case is one person's head crop, so competing boxes
    are rival guesses at the same helmet — only the best one is drawn."""
    detector = _detector(
        [
            _FakeResult([
                _FakeBox([20.0, 10.0, 60.0, 30.0], 0.41),
                _FakeBox([100.0, 20.0, 140.0, 40.0], 0.88),
                _FakeBox([10.0, 5.0, 30.0, 15.0], 0.62),
            ])
        ]
    )

    located = detector.locate(_image_bytes())

    assert located is not None
    assert located.confidence == pytest.approx(0.88)
    # the 0.88 box: x 100-140 of 200 wide, y 20-40 of 100 tall
    assert located.box == pytest.approx([0.5, 0.2, 0.7, 0.4])


def test_picks_the_best_box_across_several_results():
    detector = _detector(
        [
            _FakeResult([_FakeBox([0.0, 0.0, 20.0, 20.0], 0.30)]),
            _FakeResult([_FakeBox([40.0, 20.0, 80.0, 60.0], 0.75)]),
        ]
    )

    located = detector.locate(_image_bytes())

    assert located.confidence == pytest.approx(0.75)


def test_no_detections_returns_none():
    assert _detector([_FakeResult([])]).locate(_image_bytes()) is None


def test_min_confidence_is_handed_to_the_model():
    """Filtering is the detector's job, not a second pass here — assert the
    threshold actually reaches `predict`."""
    detector = _detector([_FakeResult([])], min_confidence=0.4)
    detector.locate(_image_bytes())

    assert detector._model.predict_kwargs["conf"] == 0.4
    assert detector._model.predict_kwargs["device"] == "cpu"


# --- coordinates --------------------------------------------------------


def test_box_is_normalized_to_the_crop():
    detector = _detector([_FakeResult([_FakeBox([50.0, 25.0, 150.0, 75.0], 0.9)])])

    located = detector.locate(_image_bytes(width=200, height=100))

    assert located.box == pytest.approx([0.25, 0.25, 0.75, 0.75])


def test_out_of_frame_coordinates_are_clamped():
    """Ultralytics can return a box slightly outside the image; a negative or
    >1 coordinate would draw off-crop and corrupts judge_box_to_original."""
    detector = _detector([_FakeResult([_FakeBox([-10.0, -5.0, 260.0, 130.0], 0.9)])])

    located = detector.locate(_image_bytes(width=200, height=100))

    assert located.box == pytest.approx([0.0, 0.0, 1.0, 1.0])


@pytest.mark.parametrize(
    "xyxy",
    [
        [50.0, 25.0, 50.0, 75.0],   # zero width
        [50.0, 25.0, 150.0, 25.0],  # zero height
        [150.0, 25.0, 50.0, 75.0],  # inverted in x
    ],
)
def test_degenerate_boxes_are_dropped(xyxy):
    assert _detector([_FakeResult([_FakeBox(xyxy, 0.9)])]).locate(_image_bytes()) is None


# --- never raising ------------------------------------------------------


def test_a_model_failure_returns_none_rather_than_raising():
    """The verdict is already decided by the time this runs. A failed box
    costs a rectangle; raising would cost the whole run."""
    detector = _detector([], raises=RuntimeError("inference blew up"))

    assert detector.locate(_image_bytes()) is None


def test_unreadable_image_bytes_return_none():
    assert _detector([_FakeResult([])]).locate(b"not an image") is None


# --- wiring -------------------------------------------------------------


def test_disabled_config_yields_no_detector():
    config = PipelineConfig()
    config.helmet_box.enabled = False

    assert get_helmet_detector(config) is None


def test_missing_weights_yield_no_detector_rather_than_raising(tmp_path):
    """A deployment without the checkpoint should still judge cases
    correctly, just without a tighter box."""
    config = PipelineConfig()
    config.helmet_box.weights = str(tmp_path / "does_not_exist.pt")
    reset_detector_cache()

    assert get_helmet_detector(config) is None


def test_present_weights_yield_a_detector(tmp_path):
    weights = tmp_path / "fake.pt"
    weights.write_bytes(b"not really a checkpoint")
    config = PipelineConfig()
    config.helmet_box.weights = str(weights)
    reset_detector_cache()

    detector = get_helmet_detector(config)

    # Constructed but not loaded -- `_load` is lazy, so a bogus file is fine
    # until something actually calls locate().
    assert isinstance(detector, HelmetDetector)
    assert detector._model is None


def test_relative_weights_resolve_against_the_repo_root():
    """The default is a relative path, and the app is started from several
    different working directories (uvicorn, pytest, the CLI)."""
    config = PipelineConfig()
    resolved = resolve_weights_path(config.helmet_box.weights)

    assert resolved.is_absolute()
    assert resolved.parts[-2:] == ("weights", Path(config.helmet_box.weights).name)


def test_any_configured_weights_string_resolves_the_same_way():
    """It takes the string, not the config block. `helmet_cv` used to hold a
    person checkpoint of its own; that was removed on 2 Sep and this stage-2
    helmet detector is the only one left."""
    resolved = resolve_weights_path("weights/anything.pt")

    assert resolved.is_absolute()
    assert resolved.parts[-2:] == ("weights", "anything.pt")


def test_absolute_weights_are_left_alone(tmp_path):
    assert resolve_weights_path(str(tmp_path / "elsewhere.pt")) == tmp_path / "elsewhere.pt"


def test_the_shipped_default_weights_exist():
    """The checkpoint is committed, so a fresh clone localizes helmets with
    no extra setup. Guards against it being dropped by a stray gitignore.

    Reads the path out of config rather than naming a file, which is what
    let the YOLOv8s -> YOLO11s swap be a two-line config change: a hardcoded
    filename here would have had to move with it, and would have been the
    kind of thing that fails one release after the swap rather than during
    it."""
    assert resolve_weights_path(PipelineConfig().helmet_box.weights).is_file()


def test_config_toml_and_the_built_in_default_name_the_same_checkpoint():
    """config.toml's header promises every value matches the built-in default
    in config.py. The weights path is the one most likely to break that
    promise, because a model swap is exactly the change someone makes in one
    file and forgets in the other — and the two disagreeing is silent: the
    app loads whichever the file says and the tests load whichever the
    default says."""
    from ppe_pipeline.config import load_config

    assert load_config().helmet_box.weights == PipelineConfig().helmet_box.weights


def test_helmet_location_reports_confidence_not_agreement():
    """The VLM path scored a box by IoU between two independent asks because
    no single answer carried confidence. The detector reports a real one."""
    located = HelmetLocation(box=[0.1, 0.2, 0.3, 0.4], confidence=0.77)

    assert located.confidence == 0.77
    assert not hasattr(located, "agreement")
