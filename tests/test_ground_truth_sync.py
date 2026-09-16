import json
from pathlib import Path

from botocore.exceptions import ClientError

from ppe_pipeline.eval import ground_truth_sync
from ppe_pipeline.eval.ground_truth_sync import sync_ground_truth
from ppe_pipeline.storage import S3Storage


def _write(path: Path, cases: list) -> None:
    path.write_text(json.dumps(cases))


def test_sync_updates_every_file_containing_the_case(tmp_path):
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "FP"}],
    )
    _write(
        tmp_path / "combined_eval_dataset.json",
        [
            {"case_id": "abc", "violation_type": "glove", "ground_truth": "FP"},
            {"case_id": "xyz", "violation_type": "helmet", "ground_truth": "TP"},
        ],
    )

    updated = sync_ground_truth("abc", "TP", tmp_path)

    updated_names = {p.name for p in updated}
    assert updated_names == {"known_cases.json", "combined_eval_dataset.json"}

    known = json.loads((tmp_path / "known_cases.json").read_text())
    assert known[0]["ground_truth"] == "TP"

    combined = json.loads((tmp_path / "combined_eval_dataset.json").read_text())
    abc = next(c for c in combined if c["case_id"] == "abc")
    xyz = next(c for c in combined if c["case_id"] == "xyz")
    assert abc["ground_truth"] == "TP"
    assert xyz["ground_truth"] == "TP"  # untouched — was already TP, different case anyway


def test_sync_is_a_noop_when_nothing_changes(tmp_path):
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"}],
    )

    updated = sync_ground_truth("abc", "TP", tmp_path)

    assert updated == []


def test_sync_leaves_unlabeled_pending_entries_alone(tmp_path):
    _write(
        tmp_path / "pending_validation.json",
        [{"case_id": "abc", "violation_type": "glove"}],  # no ground_truth key yet
    )
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "FP"}],
    )

    updated = sync_ground_truth("abc", "TP", tmp_path)

    assert {p.name for p in updated} == {"known_cases.json"}
    pending = json.loads((tmp_path / "pending_validation.json").read_text())
    assert "ground_truth" not in pending[0]


def test_sync_ignores_files_without_a_matching_case_id(tmp_path):
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "other", "violation_type": "glove", "ground_truth": "FP"}],
    )

    updated = sync_ground_truth("abc", "TP", tmp_path)

    assert updated == []
    known = json.loads((tmp_path / "known_cases.json").read_text())
    assert known[0]["ground_truth"] == "FP"


def test_sync_skips_non_case_list_json_files(tmp_path):
    (tmp_path / "few_shot_manifest.json").write_text(json.dumps({"glove": []}))
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "FP"}],
    )

    updated = sync_ground_truth("abc", "TP", tmp_path)

    assert {p.name for p in updated} == {"known_cases.json"}


class _AlwaysConflictingFakeS3Client:
    """Minimal fake S3 client (just enough of `get_object`/`put_object`/
    `get_paginator` for `S3Storage`) whose every `put_object` loses the CAS
    race, simulating permanent contention on a file."""

    def __init__(self, objects: dict):
        import io

        self._io = io
        self._objects = objects

    def get_object(self, Bucket, Key):
        if Key not in self._objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")
        body = json.dumps(self._objects[Key]).encode("utf-8")
        return {"Body": self._io.BytesIO(body), "ETag": '"etag"'}

    def put_object(self, **kwargs):
        raise ClientError({"Error": {"Code": "PreconditionFailed", "Message": "at least one condition failed"}}, "PutObject")

    def get_paginator(self, operation_name):
        objects = self._objects

        class _Paginator:
            def paginate(self, Bucket, Prefix=""):
                yield {"Contents": [{"Key": key} for key in sorted(objects) if key.startswith(Prefix)]}

        return _Paginator()


def test_sync_skips_file_that_always_conflicts_instead_of_aborting(monkeypatch):
    objects = {
        "eval/known_cases.json": [{"case_id": "abc", "violation_type": "glove", "ground_truth": "FP"}],
        "eval/combined_eval_dataset.json": [{"case_id": "abc", "violation_type": "glove", "ground_truth": "FP"}],
    }
    fake_backend = S3Storage(bucket="fake-bucket", prefix="", client=_AlwaysConflictingFakeS3Client(objects))
    monkeypatch.setattr(ground_truth_sync.storage, "get_storage", lambda: fake_backend)

    updated = sync_ground_truth("abc", "TP", Path("eval"))

    assert updated == []
