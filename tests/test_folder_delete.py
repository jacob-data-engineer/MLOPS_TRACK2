"""Deleting a folder from the S3 folder browser.

Asked twice: a count first, then the delete. The bucket root, other
people's buckets and the pipeline's own code are refused outright; a
folder a setting still uses is refused until the dialog sends `force`.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "components"


class S3:
    def __init__(self, keys):
        self.keys = dict.fromkeys(keys, 10)
        self.deleted_batches = []

    def list_objects_v2(self, Bucket, Prefix, ContinuationToken=None):
        keys = sorted(k for k in self.keys if k.startswith(Prefix))
        start = int(ContinuationToken or 0)
        page = keys[start:start + 1000]
        out = {"Contents": [{"Key": k, "Size": self.keys[k]} for k in page],
               "IsTruncated": start + 1000 < len(keys)}
        if out["IsTruncated"]:
            out["NextContinuationToken"] = str(start + 1000)
        return out

    def delete_objects(self, Bucket, Delete):
        batch = [o["Key"] for o in Delete["Objects"]]
        self.deleted_batches.append(len(batch))
        for k in batch:
            self.keys.pop(k, None)
        return {}


@pytest.fixture
def live(monkeypatch):
    s3 = S3([f"retraining/_retained/{i}.txt" for i in range(1500)]
            + ["retraining/helmet_test/week_2/images/train/a.jpg",
               "retraining/week_1/images/train/b.jpg"])
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "BUCKET", "track2-s3")
    monkeypatch.setattr(pc, "read_config", lambda: {
        "root": "retraining/helmet_test",
        "sources": [{"destination": "retraining/helmet_test"}]})
    return s3


def test_the_check_counts_and_names_nothing_for_an_unused_folder(live):
    out = pc.folder_delete_check(bucket="", prefix="retraining/_retained")
    assert out["objects"] == 1500 and out["bytes"] == 15000
    assert out["in_use"] == []


def test_every_page_is_deleted_not_just_the_first(live):
    out = pc.delete_folder(bucket="track2-s3", prefix="retraining/_retained")
    assert out["deleted"] == 1500 and live.deleted_batches == [1000, 500]
    assert not any(k.startswith("retraining/_retained/") for k in live.keys)
    assert "retraining/week_1/images/train/b.jpg" in live.keys


def test_a_folder_in_use_needs_force(live):
    check = pc.folder_delete_check(bucket="", prefix="retraining/helmet_test")
    assert any("Pull Control's root" in u for u in check["in_use"])
    with pytest.raises(PermissionError, match="Pull Control's root"):
        pc.delete_folder(bucket="", prefix="retraining/helmet_test")
    assert pc.delete_folder(bucket="", prefix="retraining/helmet_test", force=True)["deleted"] == 1


def test_a_parent_of_something_in_use_counts_as_in_use(live):
    assert pc.folder_delete_check(bucket="", prefix="retraining")["in_use"]


@pytest.mark.parametrize("prefix", ["", "/", "custom", "custom/code", "codebuild-source"])
def test_root_and_pipeline_code_are_refused(live, prefix):
    with pytest.raises(ValueError):
        pc.folder_delete_check(bucket="", prefix=prefix)


def test_another_bucket_is_refused(live):
    with pytest.raises(ValueError, match="somebody else's"):
        pc.delete_folder(bucket="safety-model-dataset-dev", prefix="track2", force=True)


def test_the_field_asks_in_a_dialog_and_refreshes():
    field = (WEB / "S3FolderField.tsx").read_text(encoding="utf-8")
    assert "<ConfirmDialog" in field and "onDelete={openDelete}" in field
    assert "this cannot be undone" in field
    assert "[prefix, bucket, disabled, reload]" in field
    browser = (WEB / "Track2S3FolderBrowser.tsx").read_text(encoding="utf-8")
    assert "onClick={() => onDelete(fullPath)}" in browser
