"""Too-near-forklift: a distance proxy without per-camera calibration.

Real distance estimation needs per-camera calibration and is fragile across
sites with different camera models, so instead of estimating distance we
estimate violation directly: expand the person's and forklift's boxes by a
tuned percentage each, and flag when the expanded boxes overlap enough.
Percentage (not fixed-pixel) expansion partially self-corrects for
perspective, since closer objects already have bigger boxes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .config import PipelineConfig, get_config
from .models import BoundingBox, Detection, ViolationType


@dataclass
class ForkliftProximityResult:
    is_violation: bool
    is_operator: bool
    coverage_ratio: float
    containment_ratio: float


def evaluate_too_near_forklift(
    person_box: BoundingBox,
    forklift_box: BoundingBox,
    expand_pct: float | None = None,
    overlap_threshold: float | None = None,
    operator_containment_threshold: float | None = None,
    config: PipelineConfig | None = None,
) -> ForkliftProximityResult:
    """Geometric proximity check, run before a candidate ever reaches the judge.

    Flags candidates whose person box is heavily contained within the
    forklift's box as an ambiguous "possible operator" case: this pattern
    usually means the forklift's own driver, but can also mean a person
    standing or hanging off the vehicle in a hazardous position — geometry
    alone can't tell those apart, so the caller sends it on to the judge
    with the containment ratio as extra context instead of auto-rejecting.

    Any of `expand_pct` / `overlap_threshold` / `operator_containment_threshold`
    left as `None` fall back to `config.forklift` (config.toml / built-in
    defaults — see config.py).
    """
    config = config or get_config()
    expand_pct = expand_pct if expand_pct is not None else config.forklift.expand_pct
    overlap_threshold = (
        overlap_threshold if overlap_threshold is not None else config.forklift.overlap_threshold
    )
    operator_containment_threshold = (
        operator_containment_threshold
        if operator_containment_threshold is not None
        else config.forklift.operator_containment_threshold
    )

    containment_ratio = person_box.containment_ratio(forklift_box)
    if containment_ratio >= operator_containment_threshold:
        return ForkliftProximityResult(
            is_violation=False,
            is_operator=True,
            coverage_ratio=0.0,
            containment_ratio=containment_ratio,
        )

    expanded_person = person_box.expand(expand_pct)
    expanded_forklift = forklift_box.expand(expand_pct)

    # Intersection-over-person-area, not IoU: the forklift's box is much
    # larger, so IoU reads low even when the person's danger zone is fully
    # covered by the forklift's expanded footprint.
    intersection = expanded_person.intersection_area(expanded_forklift)
    coverage_ratio = intersection / expanded_person.area if expanded_person.area else 0.0

    return ForkliftProximityResult(
        is_violation=coverage_ratio >= overlap_threshold,
        is_operator=False,
        coverage_ratio=coverage_ratio,
        containment_ratio=containment_ratio,
    )


def filter_too_near_forklift_detections(
    detections: list[Detection],
    config: PipelineConfig | None = None,
) -> list[Detection]:
    """Drop too_near_forklift detections that don't survive the geometric
    proximity check — person/forklift boxes too far apart. Candidates whose
    person box is heavily contained within the forklift's box (possible
    operator) are kept but flagged via `forklift_is_ambiguous_operator` /
    `forklift_containment_ratio`, so the judge can decide TP vs FP from the
    image instead of the geometry auto-rejecting them. Every other
    violation type passes through untouched.

    Call this on the raw detection list before `merge_detections`, so a
    candidate that fails the far-apart check never reaches cropping or the
    judge — the crop's own person+forklift union box is wide enough that
    the judge can't reliably tell "near" from "same frame, far apart" on
    image content alone.
    """
    config = config or get_config()
    kept = []
    for det in detections:
        if det.violation_type is not ViolationType.TOO_NEAR_FORKLIFT:
            kept.append(det)
            continue
        if det.forklift_box is None:
            raise ValueError("too_near_forklift detection missing forklift_box")
        result = evaluate_too_near_forklift(det.person_box, det.forklift_box, config=config)
        if result.is_operator:
            kept.append(
                replace(
                    det,
                    forklift_is_ambiguous_operator=True,
                    forklift_containment_ratio=result.containment_ratio,
                )
            )
        elif result.is_violation:
            kept.append(det)
    return kept
