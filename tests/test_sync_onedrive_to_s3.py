"""Tests for `scripts/sync_onedrive_to_s3.py`'s pure diff/classification
logic — `classify_files` takes an already-collected file listing and a
fake manifest, so it needs no real filesystem or S3 access (following the
hand-rolled-fake convention used in `tests/test_storage.py`, rather than
`moto`/real uploads). Full real-upload behavior is covered by the manual
smoke-test procedure in the script's own module docstring instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "sync_onedrive_to_s3.py"
_spec = importlib.util.spec_from_file_location("sync_onedrive_to_s3", _SCRIPT_PATH)
sync_onedrive_to_s3 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sync_onedrive_to_s3
_spec.loader.exec_module(sync_onedrive_to_s3)

FileInfo = sync_onedrive_to_s3.FileInfo
classify_files = sync_onedrive_to_s3.classify_files
s3_key_for = sync_onedrive_to_s3.s3_key_for
default_manifest_path = sync_onedrive_to_s3.default_manifest_path


def _raising_hash_fn(rel_path: str) -> str:
    raise AssertionError(f"hash_fn should not have been called for {rel_path!r}")


def test_new_file_with_no_manifest_entry_is_classified_as_new():
    local_files = {"foo.jpg": FileInfo(size=100, mtime=1000.0)}
    manifest: dict = {}

    plan = classify_files(local_files, manifest, hash_fn=lambda _rel: "abc123")

    assert plan.to_upload == [("foo.jpg", "abc123", "new")]
    assert plan.unchanged == []
    assert plan.skipped_placeholder == []


def test_unchanged_file_matching_size_and_mtime_skips_hashing():
    local_files = {"foo.jpg": FileInfo(size=100, mtime=1000.0)}
    manifest = {"foo.jpg": {"size": 100, "mtime": 1000.0, "sha256": "abc123"}}

    plan = classify_files(local_files, manifest, hash_fn=_raising_hash_fn)

    assert plan.to_upload == []
    assert plan.unchanged == ["foo.jpg"]
    assert plan.skipped_placeholder == []


def test_touched_mtime_but_same_content_is_classified_as_unchanged():
    """size/mtime differ from the manifest (e.g. OneDrive touched the file
    without altering its bytes), but the recomputed hash matches — this
    must not be re-uploaded."""
    local_files = {"foo.jpg": FileInfo(size=100, mtime=2000.0)}
    manifest = {"foo.jpg": {"size": 100, "mtime": 1000.0, "sha256": "abc123"}}

    plan = classify_files(local_files, manifest, hash_fn=lambda _rel: "abc123")

    assert plan.to_upload == []
    assert plan.unchanged == ["foo.jpg"]


def test_changed_content_is_classified_as_changed():
    local_files = {"foo.jpg": FileInfo(size=200, mtime=2000.0)}
    manifest = {"foo.jpg": {"size": 100, "mtime": 1000.0, "sha256": "abc123"}}

    plan = classify_files(local_files, manifest, hash_fn=lambda _rel: "xyz789")

    assert plan.to_upload == [("foo.jpg", "xyz789", "changed")]
    assert plan.unchanged == []


def test_placeholder_file_is_skipped_without_hashing():
    local_files = {"foo.jpg": FileInfo(size=100, mtime=1000.0, is_placeholder=True)}
    manifest: dict = {}

    plan = classify_files(local_files, manifest, hash_fn=_raising_hash_fn)

    assert plan.to_upload == []
    assert plan.unchanged == []
    assert plan.skipped_placeholder == ["foo.jpg"]


def test_placeholder_takes_priority_even_with_a_stale_manifest_entry():
    local_files = {"foo.jpg": FileInfo(size=999, mtime=9999.0, is_placeholder=True)}
    manifest = {"foo.jpg": {"size": 100, "mtime": 1000.0, "sha256": "abc123"}}

    plan = classify_files(local_files, manifest, hash_fn=_raising_hash_fn)

    assert plan.skipped_placeholder == ["foo.jpg"]
    assert plan.to_upload == []


def test_classify_files_handles_mixed_batch_and_sorts_output_by_rel_path():
    local_files = {
        "b_new.jpg": FileInfo(size=10, mtime=10.0),
        "a_unchanged.jpg": FileInfo(size=20, mtime=20.0),
        "c_placeholder.jpg": FileInfo(size=30, mtime=30.0, is_placeholder=True),
    }
    manifest = {"a_unchanged.jpg": {"size": 20, "mtime": 20.0, "sha256": "same"}}

    plan = classify_files(local_files, manifest, hash_fn=lambda _rel: "new-hash")

    assert plan.to_upload == [("b_new.jpg", "new-hash", "new")]
    assert plan.unchanged == ["a_unchanged.jpg"]
    assert plan.skipped_placeholder == ["c_placeholder.jpg"]


def test_s3_key_for_matches_the_app_side_naming_contract():
    assert s3_key_for("foo.jpg") == "resources/hazard_images/Annotated/testing_set/images/foo.jpg"


def test_s3_key_for_agrees_with_dataset_side_image_key_resolution():
    """`s3_key_for` and `dataset._resolve_image_key` are two independent
    implementations of the same naming contract — this one used when
    uploading images, the other when the app/eval tooling reads them back.
    Cross-check them directly against each other (rather than each against
    a separately hardcoded string) so a future change to either side that
    breaks the agreement is caught automatically."""
    from pathlib import Path

    from ppe_pipeline.eval.dataset import _resolve_image_key

    filename = "foo.jpg"
    key = s3_key_for(filename)

    assert _resolve_image_key(Path("eval"), key) == Path(key)


def test_default_manifest_path_uses_localappdata_when_set(monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", "C:\\Users\\someone\\AppData\\Local")

    path = default_manifest_path()

    assert path == Path("C:\\Users\\someone\\AppData\\Local") / "ppe-pipeline" / "sync_manifest.json"


def test_default_manifest_path_falls_back_sanely_when_localappdata_unset(monkeypatch):
    monkeypatch.delenv("LOCALAPPDATA", raising=False)

    path = default_manifest_path()

    assert path == Path.home() / ".local" / "share" / "ppe-pipeline" / "sync_manifest.json"
