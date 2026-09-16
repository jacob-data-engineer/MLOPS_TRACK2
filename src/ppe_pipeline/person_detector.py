"""Finding people in a clean alarm frame, as an ALTERNATIVE to the vendor's boxes.

WHY THIS EXISTS AGAIN

A person detector was removed from this repo on 2 Sep, and the reasons are
recorded in `[helmet_cv]` in `config.toml`: the checkpoint committed then had
been fine-tuned until it could no longer find a person (six boxes over six
alarms, none on a person), and stock COCO found one worker in a ship's hold
where the vendor marked seven.

This is a different model and a different question. `weights/best.pt` is a
SINGLE-CLASS person detector -- yolo11m, 60 epochs, trained at imgsz 1536 --
so it was built for this footage's small workers rather than adapted to them.
Whether it beats the vendor's own boxes is a measurement nobody has taken, and
taking it is the point: the run screen offers both, every row records which
produced it, and the two never pool.

WHAT IS DELIBERATELY THE SAME AS THE VENDOR PATH

The feed still walks `snapshot/`, so both options judge THE SAME ALARMS. The
old detector option walked `raw_video/` instead, which meant the two arms saw
different populations and their numbers could not be compared at all. Only the
source of the boxes changes here.

Boxes are found on the CLEAN frame from the clip, never on the annotated
snapshot: a detector run over a picture with red rectangles drawn on it is
being handed the answer.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path

from PIL import Image

from .config import PipelineConfig, get_config
from .helmet_detector import resolve_weights_path

logger = logging.getLogger(__name__)


class PersonDetector:
    """Wraps one loaded YOLO model. Construct via `get_person_detector`, so
    the weights are read once per process rather than per alarm."""

    def __init__(self, weights_path: Path, min_confidence: float, device: str, imgsz: int | None):
        self._weights_path = weights_path
        self._min_confidence = min_confidence
        self._device = device
        self._imgsz = imgsz
        self._model = None
        # Ultralytics `predict()` is not documented as thread-safe, and the
        # feed judges the crops of one frame concurrently -- one lock around
        # inference is cheap next to the call itself.
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            # Imported here, not at module scope: ultralytics pulls in torch,
            # which is slow and needed only when this option is actually
            # chosen. A vendor-box run must not pay for it at startup.
            from ultralytics import YOLO

            logger.info("loading person detector weights from %s", self._weights_path)
            self._model = YOLO(str(self._weights_path))
        return self._model

    def _predict_kwargs(self) -> dict:
        """The inference settings, in ONE place so `warm` cannot drift from
        `find`. A warm-up that predicted with different settings would prove
        the wrong thing -- `imgsz` in particular selects different code paths.
        """
        kwargs = {
            "conf": self._min_confidence,
            "device": self._device,
            "verbose": False,
        }
        # Only when configured. Ultralytics' default is 640, which on a
        # 1920x1080 frame scales this corpus's ~83px workers to ~28px; this
        # model was TRAINED at 1536, and inferring at a different size than
        # training is its own silent penalty.
        if self._imgsz:
            kwargs["imgsz"] = self._imgsz
        return kwargs

    def warm(self) -> None:
        """Load the weights AND RUN ONE INFERENCE, so no alarm pays for either.

        IT MUST PREDICT, not merely construct. Building the model is the cheap
        half: `YOLO(path)` imports torch and reads the checkpoint, but
        ultralytics fuses and specialises on the FIRST `predict`, which is most
        of the 6.49s-vs-0.89s gap this exists to move off the first alarm.

        And construction proves almost nothing about health. The failure worth
        catching is an install that imports and builds and then dies at
        inference -- the torchvision operator mismatch
        ("Could not run 'torchvision::nms' with arguments from the 'CPU'
        backend") is the common shape. `find` swallows exactly that and returns
        no boxes, which is not an error downstream: the frame settles as "no
        person, no violation" with a reason code, so a broken deployment judges
        every alarm that way, silently, forever. A warm-up that only
        constructed the model would have let that through unchanged.

        RAISES, unlike `find`. Called once from the worker before the poll
        loop, so the failure surfaces where the run can be marked `error`.

        Under the inference lock, so two threads cannot both build the model.
        """
        # A synthetic frame, because warming must not depend on an alarm having
        # arrived. Small: ultralytics resizes to `imgsz` regardless, so the
        # size here buys nothing and a full 1920x1080 buffer would just be
        # slower to allocate.
        probe = Image.new("RGB", (64, 64), (0, 0, 0))
        with self._lock:
            self._load().predict(probe, **self._predict_kwargs())

    def find(self, image_bytes: bytes) -> list[tuple[float, float, float, float]]:
        """Every person in the frame, as `(x1, y1, x2, y2)` in its own pixels.

        The same shape `snapshot_boxes.extract_boxes` returns, so the caller
        does not care which produced them.

        Never raises. An empty list means "found nobody", which the caller
        already handles: the frame settles as no violation with its own reason
        code and stays in the audit for a human to overturn.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                frame = image.convert("RGB")
                if not frame.width or not frame.height:
                    return []
                with self._lock:
                    results = self._load().predict(frame, **self._predict_kwargs())
        except Exception:  # noqa: BLE001 - a failed pass must not end a run
            logger.exception("person detection failed; falling back to no boxes")
            return []

        boxes: list[tuple[float, float, float, float]] = []
        for result in results:
            for candidate in getattr(result, "boxes", ()):
                try:
                    x1, y1, x2, y2 = (float(v) for v in candidate.xyxy[0])
                except Exception:  # noqa: BLE001 - one bad box, not the frame
                    continue
                if x2 > x1 and y2 > y1:
                    boxes.append((x1, y1, x2, y2))
        # Reading order, matching `extract_boxes`, so `person_index` means the
        # same thing whichever source produced the boxes.
        boxes.sort(key=lambda b: (round(b[1] / 50), b[0]))
        return boxes


_DETECTORS: dict[tuple[str, float, str, int | None], PersonDetector] = {}
_LOCK = threading.Lock()


def get_person_detector(config: PipelineConfig | None = None) -> PersonDetector | None:
    """The configured detector, or `None` when its weights are not on disk.

    `None` rather than an exception, and the caller refuses to START rather
    than failing per alarm: whether a checkpoint exists is a fact about the
    deployment, not about any one alarm, and checked per alarm it reads as a
    stream of durable failures instead of one refusal at the button.
    """
    config = config or get_config()
    settings = config.person_detector
    weights_path = resolve_weights_path(settings.weights)
    if not weights_path.is_file():
        logger.warning("person detector weights not found at %s", weights_path)
        return None
    key = (weights_path.as_posix(), settings.confidence, settings.device, settings.imgsz)
    with _LOCK:
        detector = _DETECTORS.get(key)
        if detector is None:
            detector = PersonDetector(
                weights_path, settings.confidence, settings.device, settings.imgsz
            )
            _DETECTORS[key] = detector
    return detector


def reset_person_detector_cache() -> None:
    """For tests that change the configured weights within one process."""
    with _LOCK:
        _DETECTORS.clear()
