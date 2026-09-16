"""Load real per-image detections (bounding boxes) from a JSON manifest —
the counterpart to mock_data.py's synthetic layout, for when you actually
know where the violations are in a specific image (e.g. testing the UI
against a real photo, or a one-off manual annotation).
"""

from __future__ import annotations

import json
from pathlib import Path

from .models import BoundingBox, Detection, ViolationType


def _box(coords: list[float] | None) -> BoundingBox | None:
    return BoundingBox(*coords) if coords else None


def parse_detections(raw_json: str) -> list[Detection]:
    """Parse a detections manifest shaped like:

        [
          {
            "violation_type": "helmet",
            "person_box": [x1, y1, x2, y2],
            "region_box": [x1, y1, x2, y2]
          },
          {
            "violation_type": "too_near_forklift",
            "person_box": [x1, y1, x2, y2],
            "forklift_box": [x1, y1, x2, y2]
          }
        ]

    `region_box` only applies to glove/helmet; `forklift_box` only to
    too_near_forklift. `source`/`confidence` are optional (default
    "manual" / 1.0) since these are typically hand-specified boxes.
    """
    raw = json.loads(raw_json)
    return [
        Detection(
            violation_type=ViolationType(entry["violation_type"]),
            person_box=BoundingBox(*entry["person_box"]),
            source=entry.get("source", "manual"),
            confidence=entry.get("confidence", 1.0),
            region_box=_box(entry.get("region_box")),
            forklift_box=_box(entry.get("forklift_box")),
        )
        for entry in raw
    ]


def load_detections(manifest_path: Path) -> list[Detection]:
    return parse_detections(manifest_path.read_text())
