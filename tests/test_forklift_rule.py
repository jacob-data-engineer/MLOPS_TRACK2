from ppe_pipeline.forklift_rule import evaluate_too_near_forklift, filter_too_near_forklift_detections
from ppe_pipeline.models import BoundingBox, Detection, ViolationType

FORKLIFT = BoundingBox(0, 0, 300, 200)


def test_operator_inside_forklift_box_is_excluded():
    operator = BoundingBox(50, 20, 150, 180)  # almost entirely inside the forklift box
    result = evaluate_too_near_forklift(operator, FORKLIFT)
    assert result.is_operator is True
    assert result.is_violation is False
    assert result.containment_ratio >= 0.8


def test_person_overlapping_forklift_danger_zone_is_a_violation():
    nearby_person = BoundingBox(280, 20, 340, 180)  # mostly outside, edge overlaps forklift
    result = evaluate_too_near_forklift(nearby_person, FORKLIFT)
    assert result.is_operator is False
    assert result.is_violation is True


def test_person_far_from_forklift_is_safe():
    far_person = BoundingBox(1000, 1000, 1100, 1200)
    result = evaluate_too_near_forklift(far_person, FORKLIFT)
    assert result.is_operator is False
    assert result.is_violation is False
    assert result.coverage_ratio == 0.0


def test_filter_drops_too_near_forklift_detections_that_are_actually_far_away():
    far_person = BoundingBox(1000, 1000, 1100, 1200)
    nearby_person = BoundingBox(280, 20, 340, 180)
    detections = [
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=far_person, forklift_box=FORKLIFT, source="t"),
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=nearby_person, forklift_box=FORKLIFT, source="t"),
        Detection(ViolationType.HELMET, person_box=nearby_person, source="t"),
    ]

    kept = filter_too_near_forklift_detections(detections)

    assert len(kept) == 2
    assert kept[0].person_box == nearby_person
    assert kept[1].violation_type is ViolationType.HELMET


def test_filter_keeps_the_forklifts_own_operator_flagged_as_ambiguous():
    operator = BoundingBox(50, 20, 150, 180)  # almost entirely inside the forklift box
    detections = [
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=operator, forklift_box=FORKLIFT, source="t"),
    ]

    kept = filter_too_near_forklift_detections(detections)
    expected_ratio = evaluate_too_near_forklift(operator, FORKLIFT).containment_ratio

    assert len(kept) == 1
    assert kept[0].forklift_is_ambiguous_operator is True
    assert kept[0].forklift_containment_ratio == expected_ratio


def test_filter_leaves_new_fields_at_default_on_a_normal_kept_detection():
    nearby_person = BoundingBox(280, 20, 340, 180)
    detections = [
        Detection(ViolationType.TOO_NEAR_FORKLIFT, person_box=nearby_person, forklift_box=FORKLIFT, source="t"),
    ]

    kept = filter_too_near_forklift_detections(detections)

    assert len(kept) == 1
    assert kept[0].forklift_is_ambiguous_operator is False
    assert kept[0].forklift_containment_ratio is None
