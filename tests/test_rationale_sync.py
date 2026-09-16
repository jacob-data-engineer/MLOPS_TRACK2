import json
from pathlib import Path

from botocore.exceptions import ClientError

from ppe_pipeline.eval import rationale_sync
from ppe_pipeline.eval.rationale_sync import sync_ground_truth_rationale
from ppe_pipeline.storage import S3Storage


def _write(path: Path, data) -> None:
    path.write_text(json.dumps(data))


def test_sync_propagates_resolved_rationale(tmp_path):
    _write(
        tmp_path / "gold_annotations.json",
        {"abc": {"final_label": "TP", "final_rationale": "Bare hands clearly visible."}},
    )
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"}],
    )

    updated = sync_ground_truth_rationale(tmp_path / "gold_annotations.json", tmp_path)

    assert {p.name for p in updated} == {"known_cases.json"}
    known = json.loads((tmp_path / "known_cases.json").read_text())
    assert known[0]["ground_truth_rationale"] == "Bare hands clearly visible."


def test_sync_skips_whitespace_only_rationale(tmp_path):
    _write(
        tmp_path / "gold_annotations.json",
        {"abc": {"final_label": "TP", "final_rationale": "   "}},
    )
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"}],
    )

    updated = sync_ground_truth_rationale(tmp_path / "gold_annotations.json", tmp_path)

    assert updated == []
    known = json.loads((tmp_path / "known_cases.json").read_text())
    assert "ground_truth_rationale" not in known[0]


def test_sync_skips_empty_rationale(tmp_path):
    _write(
        tmp_path / "gold_annotations.json",
        {"abc": {"final_label": "TP", "final_rationale": ""}},
    )
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"}],
    )

    updated = sync_ground_truth_rationale(tmp_path / "gold_annotations.json", tmp_path)

    assert updated == []
    known = json.loads((tmp_path / "known_cases.json").read_text())
    assert "ground_truth_rationale" not in known[0]


def test_sync_skips_cannot_determine_and_missing_label(tmp_path):
    _write(
        tmp_path / "gold_annotations.json",
        {
            "abc": {"final_label": "cannot_determine", "final_rationale": "Ambiguous."},
            "xyz": {"final_rationale": "No label yet."},
        },
    )
    _write(
        tmp_path / "known_cases.json",
        [
            {"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"},
            {"case_id": "xyz", "violation_type": "helmet", "ground_truth": "FP"},
        ],
    )

    updated = sync_ground_truth_rationale(tmp_path / "gold_annotations.json", tmp_path)

    assert updated == []
    known = json.loads((tmp_path / "known_cases.json").read_text())
    assert "ground_truth_rationale" not in known[0]
    assert "ground_truth_rationale" not in known[1]


def test_sync_skips_non_case_list_json_files(tmp_path):
    _write(
        tmp_path / "gold_annotations.json",
        {"abc": {"final_label": "TP", "final_rationale": "Bare hands."}},
    )
    (tmp_path / "few_shot_manifest.json").write_text(json.dumps({"glove": []}))
    _write(
        tmp_path / "known_cases.json",
        [{"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"}],
    )

    updated = sync_ground_truth_rationale(tmp_path / "gold_annotations.json", tmp_path)

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
        "gold_annotations.json": {"abc": {"final_label": "TP", "final_rationale": "Bare hands clearly visible."}},
        "eval/known_cases.json": [{"case_id": "abc", "violation_type": "glove", "ground_truth": "TP"}],
    }
    fake_backend = S3Storage(bucket="fake-bucket", prefix="", client=_AlwaysConflictingFakeS3Client(objects))
    monkeypatch.setattr(rationale_sync.storage, "get_storage", lambda: fake_backend)

    updated = sync_ground_truth_rationale(Path("gold_annotations.json"), Path("eval"))

    assert updated == []
