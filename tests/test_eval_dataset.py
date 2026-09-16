import io
import json
from pathlib import Path

from botocore.exceptions import ClientError
from PIL import Image

from ppe_pipeline.eval import dataset
from ppe_pipeline.eval.dataset import (
    EvalCase,
    filter_by_violation_type,
    list_valid_eval_dataset_paths,
    load_eval_dataset,
    remove_cases_by_id,
)
from ppe_pipeline.models import BoundingBox, ViolationType
from ppe_pipeline.storage import S3Storage


def test_load_eval_dataset_resolves_paths_and_boxes(tmp_path):
    Image.new("RGB", (200, 200)).save(tmp_path / "glove_001.jpg")

    manifest = [
        {
            "case_id": "glove_001",
            "image": "glove_001.jpg",
            "violation_type": "glove",
            "person_box": [0, 0, 100, 200],
            "region_box": [10, 150, 60, 190],
            "ground_truth": "TP",
            "cv_confidence": 0.9,
        },
        {
            "case_id": "forklift_002",
            "image": "glove_001.jpg",
            "violation_type": "too_near_forklift",
            "person_box": [0, 0, 50, 50],
            "forklift_box": [40, 0, 150, 50],
            "ground_truth": "FP",
        },
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    cases = load_eval_dataset(manifest_path)

    assert len(cases) == 2
    assert cases[0].violation_type is ViolationType.GLOVE
    assert cases[0].ground_truth_is_violation is True
    assert cases[0].image_path == tmp_path / "glove_001.jpg"
    assert cases[0].region_box is not None
    assert cases[0].cv_confidence == 0.9

    assert cases[1].violation_type is ViolationType.TOO_NEAR_FORKLIFT
    assert cases[1].ground_truth_is_violation is False
    assert cases[1].forklift_box is not None
    assert cases[1].cv_confidence is None


def test_load_eval_dataset_resolves_resources_prefixed_images_as_a_top_level_key(tmp_path):
    # "resources/..." is already a top-level storage key (no cwd, no ".."
    # walk against the manifest's own directory) — every shipped eval
    # manifest uses this convention today.
    manifest = [
        {
            "case_id": "glove_001",
            "image": "resources/hazard_images/Annotated/testing_set/images/glove_001.jpg",
            "violation_type": "glove",
            "person_box": [0, 0, 100, 200],
            "ground_truth": "TP",
        }
    ]
    manifest_path = tmp_path / "eval" / "manifest.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(json.dumps(manifest))

    cases = load_eval_dataset(manifest_path)

    assert len(cases) == 1
    assert cases[0].image_path == Path(
        "resources/hazard_images/Annotated/testing_set/images/glove_001.jpg"
    )


class _FakeS3ClientForManifest:
    """Minimal fake just for `get_object`, keyed by exact full key, to prove
    `load_eval_dataset` reads the manifest through `storage.get_storage()`
    rather than `Path.read_text()`."""

    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def get_object(self, Bucket, Key):
        if Key not in self._objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")
        return {"Body": io.BytesIO(self._objects[Key]), "ETag": '"etag"'}


def test_load_eval_dataset_reads_manifest_through_storage(tmp_path, monkeypatch):
    Image.new("RGB", (200, 200)).save(tmp_path / "glove_001.jpg")
    manifest = [
        {
            "case_id": "glove_001",
            "image": "glove_001.jpg",
            "violation_type": "glove",
            "person_box": [0, 0, 100, 200],
            "ground_truth": "TP",
        }
    ]
    manifest_path = tmp_path / "manifest.json"
    # Deliberately never written to local disk — if load_eval_dataset still
    # did a raw Path.read_text() under the hood, this would raise
    # FileNotFoundError instead of reading from the fake S3 backend.
    fake_client = _FakeS3ClientForManifest({manifest_path.as_posix(): json.dumps(manifest).encode("utf-8")})
    fake_backend = S3Storage(bucket="fake-bucket", client=fake_client)
    monkeypatch.setattr(dataset.storage, "get_storage", lambda: fake_backend)

    cases = load_eval_dataset(manifest_path)

    assert len(cases) == 1
    assert cases[0].case_id == "glove_001"
    assert cases[0].image_path == tmp_path / "glove_001.jpg"


def _case(case_id: str, violation_type: ViolationType) -> EvalCase:
    return EvalCase(
        case_id=case_id,
        image_path=Path(f"{case_id}.jpg"),
        violation_type=violation_type,
        person_box=BoundingBox(0, 0, 10, 10),
        ground_truth_is_violation=True,
    )


def test_filter_by_violation_type_keeps_only_requested_types():
    cases = [
        _case("g1", ViolationType.GLOVE),
        _case("h1", ViolationType.HELMET),
        _case("h2", ViolationType.HELMET),
    ]

    filtered = filter_by_violation_type(cases, [ViolationType.HELMET])

    assert [c.case_id for c in filtered] == ["h1", "h2"]


def test_filter_by_violation_type_empty_list_means_no_filtering():
    cases = [_case("g1", ViolationType.GLOVE), _case("h1", ViolationType.HELMET)]

    assert filter_by_violation_type(cases, []) == cases


def test_filter_by_violation_type_none_means_no_filtering():
    cases = [_case("g1", ViolationType.GLOVE), _case("h1", ViolationType.HELMET)]

    assert filter_by_violation_type(cases, None) == cases


def test_remove_cases_by_id_drops_matching_cases(tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps([
        {"case_id": "keep", "violation_type": "glove"},
        {"case_id": "drop", "violation_type": "helmet"},
    ]))

    changed = remove_cases_by_id(path, {"drop"})

    assert changed is True
    remaining = json.loads(path.read_text())
    assert [c["case_id"] for c in remaining] == ["keep"]


def test_remove_cases_by_id_no_match_reports_unchanged(tmp_path):
    path = tmp_path / "cases.json"
    original = [{"case_id": "keep", "violation_type": "glove"}]
    path.write_text(json.dumps(original))

    changed = remove_cases_by_id(path, {"not_present"})

    assert changed is False
    assert json.loads(path.read_text()) == original


def test_remove_cases_by_id_empty_set_is_a_no_op(tmp_path):
    path = tmp_path / "cases.json"
    original = [{"case_id": "keep", "violation_type": "glove"}]
    path.write_text(json.dumps(original))

    changed = remove_cases_by_id(path, set())

    assert changed is False
    assert json.loads(path.read_text()) == original


# --- list_valid_eval_dataset_paths ---------------------------------------


def _write_case_list(path: Path) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "case_id": "case-1", "image": "scene.jpg", "violation_type": "helmet",
                    "person_box": [0, 0, 10, 10], "ground_truth": "TP",
                }
            ]
        )
    )


def test_list_valid_eval_dataset_paths_excludes_gold_annotations_shaped_file(tmp_path):
    _write_case_list(tmp_path / "real_dataset.json")
    # Shaped like eval/gold_annotations.json: a dict keyed by case_id, not a
    # list of case entries.
    (tmp_path / "gold_annotations.json").write_text(
        json.dumps({"case-1": {"final_label": "violation"}})
    )

    valid = list_valid_eval_dataset_paths(tmp_path)

    assert (tmp_path / "real_dataset.json").as_posix() in valid
    assert (tmp_path / "gold_annotations.json").as_posix() not in valid


def test_list_valid_eval_dataset_paths_excludes_track1_audit_log_shaped_file(tmp_path):
    _write_case_list(tmp_path / "real_dataset.json")
    # Shaped like eval/track1_audit_log.json: a list, but of audit rows
    # with none of the required case-list keys.
    (tmp_path / "track1_audit_log.json").write_text(
        json.dumps(
            [
                {
                    "alarm_id": "a1", "hazard_type": "no_helmet", "outcome": "CONFIRM_TP",
                    "reason_code": "bare_head", "confidence": 0.9, "reasoning": "x",
                    "timestamp": "2026-08-14T00:00:00+00:00", "source_detection_id": "d1",
                    "image_key": "a1.jpg", "person_box": {"x1": 0, "y1": 0, "x2": 10, "y2": 10},
                }
            ]
        )
    )

    valid = list_valid_eval_dataset_paths(tmp_path)

    assert (tmp_path / "real_dataset.json").as_posix() in valid
    assert (tmp_path / "track1_audit_log.json").as_posix() not in valid


def test_list_valid_eval_dataset_paths_excludes_corrupt_and_empty_list_files(tmp_path):
    _write_case_list(tmp_path / "real_dataset.json")
    (tmp_path / "corrupt.json").write_text("not valid json {{{")
    (tmp_path / "empty_list.json").write_text("[]")

    valid = list_valid_eval_dataset_paths(tmp_path)

    assert valid == [(tmp_path / "real_dataset.json").as_posix()]


def test_list_valid_eval_dataset_paths_caches_within_ttl(tmp_path, monkeypatch):
    # Each of the other tests above uses its own fresh tmp_path, so they
    # never observe this cache -- this one deliberately calls twice with
    # the *same* directory to pin down that the second call is served
    # from cache (a file added between the two calls stays invisible)
    # rather than re-scanning S3/disk, and that expiring the cache entry
    # picks the change back up.
    dataset._dataset_list_cache.clear()
    _write_case_list(tmp_path / "real_dataset.json")

    first = list_valid_eval_dataset_paths(tmp_path)
    assert first == [(tmp_path / "real_dataset.json").as_posix()]

    _write_case_list(tmp_path / "added_later.json")
    second = list_valid_eval_dataset_paths(tmp_path)
    assert second == first, "second call within the TTL should be served from cache, not rescanned"

    # Force the cached entry to look expired, then confirm the new file
    # is picked up.
    directory_key = tmp_path.as_posix()
    stale_time, cached_value = dataset._dataset_list_cache[directory_key]
    monkeypatch.setattr(
        dataset,
        "_dataset_list_cache",
        {directory_key: (stale_time - dataset._DATASET_LIST_CACHE_TTL_SECONDS - 1, cached_value)},
    )
    third = list_valid_eval_dataset_paths(tmp_path)
    assert (tmp_path / "added_later.json").as_posix() in third
