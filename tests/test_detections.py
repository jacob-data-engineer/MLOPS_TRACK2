import json

from ppe_pipeline.detections import load_detections, parse_detections
from ppe_pipeline.models import ViolationType


def test_parse_detections_builds_boxes_and_defaults():
    raw = json.dumps(
        [
            {
                "violation_type": "helmet",
                "person_box": [0, 0, 100, 200],
                "region_box": [10, 0, 60, 40],
            },
            {
                "violation_type": "too_near_forklift",
                "person_box": [200, 0, 300, 200],
                "forklift_box": [280, 0, 500, 200],
                "source": "yolo_v8",
                "confidence": 0.76,
            },
        ]
    )
    detections = parse_detections(raw)

    assert len(detections) == 2
    assert detections[0].violation_type is ViolationType.HELMET
    assert detections[0].region_box is not None
    assert detections[0].forklift_box is None
    assert detections[0].source == "manual"  # default
    assert detections[0].confidence == 1.0  # default

    assert detections[1].violation_type is ViolationType.TOO_NEAR_FORKLIFT
    assert detections[1].forklift_box is not None
    assert detections[1].source == "yolo_v8"
    assert detections[1].confidence == 0.76


def test_load_detections_reads_from_file(tmp_path):
    manifest = tmp_path / "detections.json"
    manifest.write_text(
        json.dumps([{"violation_type": "dangerous_plane", "person_box": [1, 2, 3, 4]}])
    )
    detections = load_detections(manifest)
    assert len(detections) == 1
    assert detections[0].violation_type is ViolationType.DANGEROUS_PLANE
