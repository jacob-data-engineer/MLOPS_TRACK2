"""Tests for `scripts/sync_s3_state.py`'s diff logic and pull/push flows.

The pure shape-classification/diff functions are tested directly with
plain Python values. The pull/push flows are exercised end-to-end against
a `LocalStorage` pointed at a `tmp_path` standing in for "S3" (an
injected, already-constructed `Storage`, per `run_pull`/`run_push`'s
signature) — this follows the hand-rolled-fake convention used in
`tests/test_storage.py`/`tests/test_sync_onedrive_to_s3.py` rather than
`moto`/real S3 access.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ppe_pipeline.storage import LocalStorage

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sync_s3_state.py"
_spec = importlib.util.spec_from_file_location("sync_s3_state", _SCRIPT_PATH)
sync_s3_state = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sync_s3_state
_spec.loader.exec_module(sync_s3_state)

classify_json_shape = sync_s3_state.classify_json_shape
diff_json = sync_s3_state.diff_json
format_diff_lines = sync_s3_state.format_diff_lines
run_pull = sync_s3_state.run_pull
run_push = sync_s3_state.run_push


# --- classify_json_shape ---------------------------------------------------


def test_classify_json_shape_detects_case_list():
    value = [{"case_id": "a", "ground_truth": "TP"}, {"case_id": "b", "ground_truth": "FP"}]
    assert classify_json_shape(value) == "case_list"


def test_classify_json_shape_treats_empty_list_as_case_list():
    assert classify_json_shape([]) == "case_list"


def test_classify_json_shape_detects_dict_keys():
    value = {"glove": [{"image": "a.jpg"}], "helmet": [{"image": "b.jpg"}]}
    assert classify_json_shape(value) == "dict_keys"


def test_classify_json_shape_falls_back_to_opaque_for_list_without_case_id():
    assert classify_json_shape([{"image": "a.jpg"}, {"image": "b.jpg"}]) == "opaque"


def test_classify_json_shape_falls_back_to_opaque_for_scalar():
    assert classify_json_shape("just a string") == "opaque"


# --- diff_json: case_list shape --------------------------------------------


def test_diff_case_list_detects_added_case():
    old = [{"case_id": "a", "ground_truth": "TP"}]
    new = [{"case_id": "a", "ground_truth": "TP"}, {"case_id": "b", "ground_truth": "FP"}]

    diff = diff_json(old, new)

    assert diff.kind == "case_list"
    assert diff.added == ["b"]
    assert diff.removed == []
    assert diff.changed == []
    assert not diff.is_identical


def test_diff_case_list_detects_removed_case():
    old = [{"case_id": "a", "ground_truth": "TP"}, {"case_id": "b", "ground_truth": "FP"}]
    new = [{"case_id": "a", "ground_truth": "TP"}]

    diff = diff_json(old, new)

    assert diff.added == []
    assert diff.removed == ["b"]
    assert diff.changed == []


def test_diff_case_list_detects_changed_case():
    old = [{"case_id": "a", "ground_truth": "TP"}]
    new = [{"case_id": "a", "ground_truth": "FP"}]

    diff = diff_json(old, new)

    assert diff.added == []
    assert diff.removed == []
    assert diff.changed == ["a"]


def test_diff_case_list_identical_content_reports_no_changes():
    old = [{"case_id": "a", "ground_truth": "TP"}]
    new = [{"case_id": "a", "ground_truth": "TP"}]

    diff = diff_json(old, new)

    assert diff.is_identical


def test_diff_case_list_guards_the_real_clobber_scenario():
    """The scenario the whole tool exists for: a reviewer has already
    promoted case 'b' out of pending_validation.json (it's gone from the
    S3 copy), but a freshly regenerated local pending_validation.json
    still has it — this must show up as a 'removed' case_id on S3's side,
    i.e. an 'added' case_id from the push's perspective, not be silently
    treated as identical."""
    s3_pending = [{"case_id": "a", "ground_truth": None}]
    freshly_regenerated_local = [{"case_id": "a", "ground_truth": None}, {"case_id": "b", "ground_truth": None}]

    diff = diff_json(s3_pending, freshly_regenerated_local)

    assert diff.added == ["b"]
    assert not diff.is_identical


# --- diff_json: dict_keys shape ---------------------------------------------


def test_diff_dict_keys_detects_added_removed_changed():
    old = {"a.jpg": {"reason": "x"}, "b.jpg": {"reason": "y"}}
    new = {"a.jpg": {"reason": "x"}, "c.jpg": {"reason": "z"}}

    diff = diff_json(old, new)

    assert diff.kind == "dict_keys"
    assert diff.added == ["c.jpg"]
    assert diff.removed == ["b.jpg"]
    assert diff.changed == []


def test_diff_dict_keys_detects_changed_value():
    old = {"glove": [{"image": "a.jpg", "verdict": "TP"}]}
    new = {"glove": [{"image": "a.jpg", "verdict": "FP"}]}

    diff = diff_json(old, new)

    assert diff.changed == ["glove"]


def test_diff_dict_keys_identical_content_reports_no_changes():
    value = {"a.jpg": {"reason": "x"}}

    diff = diff_json(value, json.loads(json.dumps(value)))

    assert diff.is_identical


# --- diff_json: opaque fallback --------------------------------------------


def test_diff_opaque_reports_identical_for_equal_scalars():
    assert diff_json("same", "same").is_identical


def test_diff_opaque_reports_not_identical_for_different_scalars():
    diff = diff_json("old-value", "new-value")
    assert diff.kind == "opaque"
    assert not diff.is_identical


def test_diff_falls_back_to_opaque_when_shapes_disagree():
    old = [{"case_id": "a"}]
    new = {"a": {"case_id": "a"}}

    diff = diff_json(old, new)

    assert diff.kind == "opaque"
    assert not diff.is_identical


# --- format_diff_lines -------------------------------------------------------


def test_format_diff_lines_mentions_case_id_counts_and_names():
    diff = diff_json(
        [{"case_id": "a"}],
        [{"case_id": "a"}, {"case_id": "b"}],
    )

    lines = format_diff_lines("eval/pending_validation.json", diff)

    joined = "\n".join(lines)
    assert "1 case_id(s) added" in joined
    assert "b" in joined


# --- run_pull ----------------------------------------------------------------


def test_run_pull_writes_json_from_s3_store_to_local_path(tmp_path):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.write_json_cas("eval/known_cases.json", [{"case_id": "a"}], expected_token=None)

    run_pull("eval/known_cases.json", repo_root, fake_s3)

    local_path = repo_root / "eval" / "known_cases.json"
    assert json.loads(local_path.read_text()) == [{"case_id": "a"}]


def test_run_pull_dry_run_does_not_write(tmp_path):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.write_json_cas("eval/known_cases.json", [{"case_id": "a"}], expected_token=None)

    run_pull("eval/known_cases.json", repo_root, fake_s3, dry_run=True)

    assert not (repo_root / "eval" / "known_cases.json").exists()


def test_run_pull_handles_binary_keys_via_get_bytes(tmp_path):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.put_bytes("few_shot/crops/glove_01.jpg", b"fake-jpeg-bytes")

    run_pull("few_shot/crops/glove_01.jpg", repo_root, fake_s3)

    assert (repo_root / "few_shot" / "crops" / "glove_01.jpg").read_bytes() == b"fake-jpeg-bytes"


# --- run_push ----------------------------------------------------------------


def _write_local_json(repo_root: Path, key: str, value) -> None:
    path = repo_root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def test_run_push_uploads_directly_when_key_is_new(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    _write_local_json(repo_root, "eval/known_cases.json", [{"case_id": "a"}])

    run_push("eval/known_cases.json", repo_root, fake_s3)

    value, _ = fake_s3.read_json("eval/known_cases.json")
    assert value == [{"case_id": "a"}]
    assert "does not exist in S3 yet" in capsys.readouterr().out


def test_run_push_new_key_dry_run_does_not_write(tmp_path):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    _write_local_json(repo_root, "eval/known_cases.json", [{"case_id": "a"}])

    run_push("eval/known_cases.json", repo_root, fake_s3, dry_run=True)

    with pytest.raises(FileNotFoundError):
        fake_s3.read_json("eval/known_cases.json")


def test_run_push_identical_content_is_a_noop(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.write_json_cas("eval/known_cases.json", [{"case_id": "a"}], expected_token=None)
    _write_local_json(repo_root, "eval/known_cases.json", [{"case_id": "a"}])

    run_push("eval/known_cases.json", repo_root, fake_s3, yes=True)

    assert "identical to S3" in capsys.readouterr().out


def test_run_push_refuses_to_write_a_diff_without_yes(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.write_json_cas("eval/pending_validation.json", [{"case_id": "a"}], expected_token=None)
    _write_local_json(repo_root, "eval/pending_validation.json", [{"case_id": "a"}, {"case_id": "b"}])

    run_push("eval/pending_validation.json", repo_root, fake_s3)

    value, _ = fake_s3.read_json("eval/pending_validation.json")
    assert value == [{"case_id": "a"}]  # unchanged — the already-reviewed case wasn't clobbered
    out = capsys.readouterr().out
    assert "added: b" in out
    assert "Refusing to overwrite" in out


def test_run_push_dry_run_never_writes_even_with_yes(tmp_path):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.write_json_cas("eval/pending_validation.json", [{"case_id": "a"}], expected_token=None)
    _write_local_json(repo_root, "eval/pending_validation.json", [{"case_id": "a"}, {"case_id": "b"}])

    run_push("eval/pending_validation.json", repo_root, fake_s3, dry_run=True, yes=True)

    value, _ = fake_s3.read_json("eval/pending_validation.json")
    assert value == [{"case_id": "a"}]


def test_run_push_writes_when_diff_shown_and_yes_given(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.write_json_cas("eval/known_cases.json", [{"case_id": "a"}], expected_token=None)
    _write_local_json(repo_root, "eval/known_cases.json", [{"case_id": "a"}, {"case_id": "b"}])

    run_push("eval/known_cases.json", repo_root, fake_s3, yes=True)

    value, _ = fake_s3.read_json("eval/known_cases.json")
    assert value == [{"case_id": "a"}, {"case_id": "b"}]
    assert "Pushed eval/known_cases.json" in capsys.readouterr().out


def test_run_push_raises_when_local_file_missing(tmp_path):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")

    with pytest.raises(SystemExit):
        run_push("eval/known_cases.json", repo_root, fake_s3)


def test_run_push_binary_key_diffs_by_hash_and_refuses_without_yes(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.put_bytes("few_shot/crops/glove_01.jpg", b"old-bytes")
    (repo_root / "few_shot" / "crops").mkdir(parents=True)
    (repo_root / "few_shot" / "crops" / "glove_01.jpg").write_bytes(b"new-bytes")

    run_push("few_shot/crops/glove_01.jpg", repo_root, fake_s3)

    assert fake_s3.get_bytes("few_shot/crops/glove_01.jpg") == b"old-bytes"
    assert "content differs" in capsys.readouterr().out


def test_run_push_binary_key_identical_is_a_noop(tmp_path, capsys):
    repo_root = tmp_path / "repo"
    fake_s3 = LocalStorage(root=tmp_path / "fake_s3")
    fake_s3.put_bytes("few_shot/crops/glove_01.jpg", b"same-bytes")
    (repo_root / "few_shot" / "crops").mkdir(parents=True)
    (repo_root / "few_shot" / "crops" / "glove_01.jpg").write_bytes(b"same-bytes")

    run_push("few_shot/crops/glove_01.jpg", repo_root, fake_s3)

    assert "identical to S3" in capsys.readouterr().out
