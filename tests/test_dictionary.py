from ppe_pipeline.dictionary import build_person_violation_dict, invert_violations
from ppe_pipeline.models import BoundingBox, PersonCluster, Detection, ViolationType

BOX = BoundingBox(0, 0, 100, 100)


def _cluster(person_id: str, *violations: ViolationType) -> PersonCluster:
    detections = [Detection(v, person_box=BOX, source="test") for v in violations]
    return PersonCluster(person_id=person_id, box=BOX, detections=detections)


def test_build_person_violation_dict_matches_doc_example_shape():
    clusters = [
        _cluster("1", ViolationType.GLOVE, ViolationType.HELMET),
        _cluster("2", ViolationType.TOO_NEAR_FORKLIFT),
        _cluster("3", ViolationType.DANGEROUS_PLANE),
        _cluster("4", ViolationType.HELMET, ViolationType.TOO_NEAR_FORKLIFT),
    ]
    result = build_person_violation_dict(clusters)
    assert result == {
        "1": ["glove", "helmet"],
        "2": ["too_near_forklift"],
        "3": ["dangerous_plane"],
        "4": ["helmet", "too_near_forklift"],
    }


def test_invert_violations_groups_people_by_violation_type():
    person_to_violations = {
        "1": ["glove", "helmet"],
        "2": ["too_near_forklift"],
        "3": ["dangerous_plane"],
        "4": ["helmet", "too_near_forklift"],
    }
    inverted = invert_violations(person_to_violations)
    assert inverted == {
        "glove": ["1"],
        "helmet": ["1", "4"],
        "too_near_forklift": ["2", "4"],
        "dangerous_plane": ["3"],
    }
