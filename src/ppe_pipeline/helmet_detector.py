"""Stage-2 helmet localization: *where* is the helmet, on a crop a verdict
has already cleared as `helmet_worn`.

A single-class (`helmet`) YOLO11s detector, run locally on the CPU. This
replaces the VLM grounding call the judge client used to make for the same
job — asking a vision-language model to return coordinates, twice, and
scoring the two answers against each other because neither was trustworthy
alone. A purpose-trained detector answers the question directly, so the box
comes with a real confidence score instead of a self-agreement proxy.

Scope: localization only. The verdict is still the judge's, and nothing here
can change it — see `locate`'s docstring.
"""

from __future__ import annotations

import io
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image

from .config import PipelineConfig, get_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HelmetLocation:
    """One helmet box, normalized to 0-1 and relative to the crop it was
    found in (the same coordinate space the judge's inline `helmet_box`
    used, so `judge_box_to_original` still applies unchanged).

    `confidence` is the detector's own score for this box. It replaces the
    `agreement` (IoU-between-two-asks) figure the VLM path reported: that
    existed only because a single VLM answer carried no usable confidence.
    """

    box: list[float]
    confidence: float


class HelmetDetector:
    """Wraps one loaded YOLO model. Construct via `get_helmet_detector`
    rather than directly, so the weights are loaded once per process.
    """

    def __init__(self, weights_path: Path, min_confidence: float, device: str):
        self._weights_path = weights_path
        self._min_confidence = min_confidence
        self._device = device
        self._model = None
        # Ultralytics predict() is not documented as thread-safe, and both
        # the eval runner's background jobs and /track1/run-job can be in
        # flight at once (see session_store.py) — one lock around inference
        # is cheap next to the model call itself.
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            # Imported here, not at module scope: ultralytics pulls in torch,
            # which is slow to import and needed only when a helmet case is
            # actually localized. Nothing else in the pipeline should pay for
            # it at startup.
            from ultralytics import YOLO

            logger.info("loading helmet detector weights from %s", self._weights_path)
            self._model = YOLO(str(self._weights_path))
        return self._model

    def locate(self, image_bytes: bytes, media_type: str | None = None) -> HelmetLocation | None:
        """The single highest-confidence helmet box in `image_bytes`, or
        `None` if the detector found nothing at/above `min_confidence`.

        Only the best box is returned by design: a cleared helmet case is one
        person's head crop, so competing boxes are alternative guesses at the
        same helmet, not separate helmets worth drawing.

        Never raises. A missing box costs a rectangle in the UI, while
        raising would take down a run whose verdicts are all perfectly good
        — the caller's verdict is already decided before this is called.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                frame = image.convert("RGB")
                width, height = frame.size
                if not width or not height:
                    return None

                with self._lock:
                    results = self._load().predict(
                        frame,
                        conf=self._min_confidence,
                        device=self._device,
                        verbose=False,
                    )

            best = None
            for result in results:
                for candidate in result.boxes:
                    score = float(candidate.conf[0])
                    if best is None or score > best[0]:
                        best = (score, candidate)
            if best is None:
                return None

            score, candidate = best
            x1, y1, x2, y2 = (float(v) for v in candidate.xyxy[0])
            box = [
                max(0.0, min(1.0, x1 / width)),
                max(0.0, min(1.0, y1 / height)),
                max(0.0, min(1.0, x2 / width)),
                max(0.0, min(1.0, y2 / height)),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            return HelmetLocation(box=box, confidence=score)
        except Exception:  # noqa: BLE001 - a decorative box must never fail a run
            logger.exception("helmet localization failed; falling back to no box")
            return None


def resolve_weights_path(weights: str) -> Path:
    """A configured weights path, resolved against the repo root when it is
    relative — the same convention `config.DEFAULT_CONFIG_PATH` uses, so the
    default works whatever directory the app is started from.

    Takes the string rather than the config block because there is now more
    than one block that holds one: `helmet_box.weights`, and the detector and
    pose checkpoints under `helmet_cv`.
    """
    path = Path(weights)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[2] / path


@lru_cache(maxsize=4)
def _detector_for(weights_path: Path, min_confidence: float, device: str) -> HelmetDetector:
    return HelmetDetector(weights_path, min_confidence, device)


def get_helmet_detector(config: PipelineConfig | None = None) -> HelmetDetector | None:
    """The process-wide detector for the current config, or `None` when
    localization is disabled or the weights file isn't there.

    Returning `None` rather than raising on missing weights is deliberate:
    the box is additive, and a deployment without the file should still
    judge cases correctly, just without a rectangle.
    """
    config = config or get_config()
    if not config.helmet_box.enabled:
        return None

    weights_path = resolve_weights_path(config.helmet_box.weights)
    if not weights_path.is_file():
        logger.warning(
            "helmet detector weights not found at %s — helmet boxes will fall back "
            "to the judge's inline field",
            weights_path,
        )
        return None
    return _detector_for(weights_path, config.helmet_box.min_confidence, config.helmet_box.device)


def reset_detector_cache() -> None:
    """Drop cached detectors — for tests that swap weights or config."""
    _detector_for.cache_clear()
