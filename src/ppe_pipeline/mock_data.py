"""Hand-written mock YOLOv8-style detections for exercising the pipeline
before the real annotated dataset (YOLOv8 labels + metadata JSON) arrives.

Lays out four people side by side, matching the architecture doc's example
dictionary: person 1 (glove + helmet), person 2 (too-near-forklift),
person 3 (dangerous plane), person 4 (helmet + too-near-forklift).
Swap this module out for a real YOLOv8-output loader once annotations land
— everything downstream only depends on the `Detection` list it returns.
"""

from __future__ import annotations

from .models import BoundingBox, Detection, ViolationType


def _lane_box(lane: int, lane_width: float, height: float, margin_frac: float = 0.1) -> BoundingBox:
    margin = lane_width * margin_frac
    x1 = lane * lane_width + margin
    x2 = (lane + 1) * lane_width - margin
    y1 = height * 0.05
    y2 = height * 0.9
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def generate_mock_detections(image_size: tuple[int, int]) -> list[Detection]:
    width, height = image_size
    lane_width = width / 4

    detections: list[Detection] = []

    # Person 1: gloves + helmet
    p1 = _lane_box(0, lane_width, height)
    head1 = BoundingBox(
        p1.x1 + p1.width * 0.2, p1.y1, p1.x2 - p1.width * 0.2, p1.y1 + p1.height * 0.18
    )
    hand1 = BoundingBox(
        p1.x1, p1.y1 + p1.height * 0.75, p1.x1 + p1.width * 0.35, p1.y1 + p1.height * 0.9
    )
    detections.append(
        Detection(ViolationType.HELMET, person_box=p1, region_box=head1, source="helmet_classifier", confidence=0.91)
    )
    detections.append(
        Detection(ViolationType.GLOVE, person_box=p1, region_box=hand1, source="glove_classifier", confidence=0.87)
    )

    # Person 2: too near forklift
    p2 = _lane_box(1, lane_width, height)
    forklift2 = BoundingBox(
        p2.x2 - p2.width * 0.3, p2.y1, min(p2.x2 + lane_width * 0.6, width), p2.y2
    )
    detections.append(
        Detection(
            ViolationType.TOO_NEAR_FORKLIFT,
            person_box=p2,
            forklift_box=forklift2,
            source="forklift_proximity_rule",
            confidence=0.95,
        )
    )

    # Person 3: dangerous plane
    p3 = _lane_box(2, lane_width, height)
    detections.append(
        Detection(ViolationType.DANGEROUS_PLANE, person_box=p3, source="dangerous_plane_classifier", confidence=0.78)
    )

    # Person 4: helmet + too near forklift
    p4 = _lane_box(3, lane_width, height)
    head4 = BoundingBox(
        p4.x1 + p4.width * 0.2, p4.y1, p4.x2 - p4.width * 0.2, p4.y1 + p4.height * 0.18
    )
    forklift4 = BoundingBox(
        p4.x2 - p4.width * 0.3, p4.y1, min(p4.x2 + lane_width * 0.6, width), p4.y2
    )
    detections.append(
        Detection(ViolationType.HELMET, person_box=p4, region_box=head4, source="helmet_classifier", confidence=0.85)
    )
    detections.append(
        Detection(
            ViolationType.TOO_NEAR_FORKLIFT,
            person_box=p4,
            forklift_box=forklift4,
            source="forklift_proximity_rule",
            confidence=0.9,
        )
    )

    return detections
