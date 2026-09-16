"""Core data model: bounding boxes, per-detector detections, and merged persons."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ViolationType(str, Enum):
    GLOVE = "glove"
    HELMET = "helmet"
    DANGEROUS_PLANE = "dangerous_plane"
    TOO_NEAR_FORKLIFT = "too_near_forklift"
    SUSPENDED_LOAD = "suspended_load"


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(f"degenerate box: {self}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def intersection_area(self, other: BoundingBox) -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    def iou(self, other: BoundingBox) -> float:
        inter = self.intersection_area(other)
        if inter == 0.0:
            return 0.0
        union = self.area + other.area - inter
        return inter / union

    def containment_ratio(self, other: BoundingBox) -> float:
        """Fraction of the smaller box's area covered by the intersection.

        Unlike IoU, this stays high when one box (e.g. a hand or head crop)
        sits entirely inside a much larger one (e.g. a full-body person box)
        — the case IoU underrates because the union is dominated by the
        larger box.
        """
        inter = self.intersection_area(other)
        smaller_area = min(self.area, other.area)
        return inter / smaller_area if smaller_area else 0.0

    def union_box(self, other: BoundingBox) -> BoundingBox:
        return BoundingBox(
            x1=min(self.x1, other.x1),
            y1=min(self.y1, other.y1),
            x2=max(self.x2, other.x2),
            y2=max(self.y2, other.y2),
        )

    def expand(self, pct: float, clip_to: tuple[float, float] | None = None) -> BoundingBox:
        """Expand symmetrically by `pct` of this box's own width/height.

        `clip_to` is an optional (image_width, image_height) to clamp against.
        """
        dw = self.width * pct
        dh = self.height * pct
        x1, y1 = self.x1 - dw, self.y1 - dh
        x2, y2 = self.x2 + dw, self.y2 + dh
        if clip_to is not None:
            max_w, max_h = clip_to
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(max_w, x2), min(max_h, y2)
        return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


@dataclass
class Detection:
    """A single raw detection from one detector/rule, before merging.

    `person_box` is what every detector produces per section 2 of the
    architecture doc ("a classifier producing a person box + label") and is
    what merging clusters on. `region_box` is the tighter hand/head box that
    Glove and Helmet crops key off (section 5's "CV output" column) — it is
    a sub-region of `person_box`, not a replacement for it. `forklift_box`
    only applies to TooNearForklift. `frame_path` identifies which image
    this detection came from — set automatically by `ImageFrame` for
    multi-frame batches (see batching.py); left `None` for the plain
    single-image pipeline, which falls back to the caller's image path.
    """

    violation_type: ViolationType
    person_box: BoundingBox
    source: str
    confidence: float = 1.0
    region_box: BoundingBox | None = None
    forklift_box: BoundingBox | None = None
    frame_path: Path | None = None
    forklift_is_ambiguous_operator: bool = False
    forklift_containment_ratio: float | None = None


@dataclass
class PersonCluster:
    """One canonical person after merging, with the union of violation labels."""

    person_id: str
    box: BoundingBox
    detections: list[Detection] = field(default_factory=list)

    @property
    def violations(self) -> set[ViolationType]:
        return {d.violation_type for d in self.detections}

    def detection_for(self, violation_type: ViolationType) -> Detection:
        for d in self.detections:
            if d.violation_type == violation_type:
                return d
        raise KeyError(f"{violation_type} not present on person {self.person_id}")

    def detection_for_forklift_context(self) -> Detection:
        """Like `detection_for(TOO_NEAR_FORKLIFT)`, but for building the
        judge's operator-context sentence specifically: when a person has
        more than one TOO_NEAR_FORKLIFT detection in the same cluster (e.g.
        flagged against two different forklifts), a detection whose
        containment is genuinely ambiguous-operator must not silently hide a
        different, genuine near-violation detection's forklift info just
        because it happens to come first in the list. Prefers a non-ambiguous
        detection if one exists; falls back to the first TOO_NEAR_FORKLIFT
        detection (today's list-order behavior) if every one of them is
        flagged ambiguous-operator.
        """
        candidates = [d for d in self.detections if d.violation_type == ViolationType.TOO_NEAR_FORKLIFT]
        if not candidates:
            raise KeyError(f"{ViolationType.TOO_NEAR_FORKLIFT} not present on person {self.person_id}")
        genuine = [d for d in candidates if not d.forklift_is_ambiguous_operator]
        return genuine[0] if genuine else candidates[0]
