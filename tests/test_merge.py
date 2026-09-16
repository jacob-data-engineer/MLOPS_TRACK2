from ppe_pipeline.merge import merge_detections
from ppe_pipeline.models import BoundingBox, Detection, ViolationType

PERSON_A = BoundingBox(0, 0, 200, 400)
HEAD_A = BoundingBox(50, 0, 150, 60)
HAND_A = BoundingBox(10, 300, 60, 360)

PERSON_B = BoundingBox(500, 0, 700, 400)
FORKLIFT_B = BoundingBox(650, 50, 900, 350)


def test_overlapping_detections_merge_into_one_person():
    detections = [
        Detection(ViolationType.HELMET, person_box=PERSON_A, region_box=HEAD_A, source="helmet_clf"),
        Detection(ViolationType.GLOVE, person_box=PERSON_A, region_box=HAND_A, source="glove_clf"),
    ]
    clusters = merge_detections(detections)
    assert len(clusters) == 1
    assert clusters[0].violations == {ViolationType.HELMET, ViolationType.GLOVE}


def test_non_overlapping_detections_stay_separate():
    detections = [
        Detection(ViolationType.HELMET, person_box=PERSON_A, region_box=HEAD_A, source="helmet_clf"),
        Detection(
            ViolationType.TOO_NEAR_FORKLIFT,
            person_box=PERSON_B,
            forklift_box=FORKLIFT_B,
            source="forklift_rule",
        ),
    ]
    clusters = merge_detections(detections)
    assert len(clusters) == 2


def test_merge_does_not_drop_labels_like_plain_nms_would():
    # Three overlapping detections on the same person, each a different
    # violation type and a different confidence — plain NMS would keep only
    # the highest-confidence box and drop the other two labels.
    detections = [
        Detection(ViolationType.HELMET, person_box=PERSON_A, region_box=HEAD_A, source="helmet_clf", confidence=0.99),
        Detection(ViolationType.GLOVE, person_box=PERSON_A, region_box=HAND_A, source="glove_clf", confidence=0.5),
        Detection(ViolationType.DANGEROUS_PLANE, person_box=PERSON_A, source="plane_clf", confidence=0.6),
    ]
    clusters = merge_detections(detections)
    assert len(clusters) == 1
    assert clusters[0].violations == {
        ViolationType.HELMET,
        ViolationType.GLOVE,
        ViolationType.DANGEROUS_PLANE,
    }
    assert len(clusters[0].detections) == 3


def test_representative_box_is_the_largest_in_the_cluster():
    small_head = BoundingBox(50, 0, 150, 60)
    detections = [
        Detection(ViolationType.HELMET, person_box=small_head, region_box=small_head, source="helmet_clf"),
        Detection(ViolationType.DANGEROUS_PLANE, person_box=PERSON_A, source="plane_clf"),
    ]
    clusters = merge_detections(detections)
    assert len(clusters) == 1
    assert clusters[0].box == PERSON_A
