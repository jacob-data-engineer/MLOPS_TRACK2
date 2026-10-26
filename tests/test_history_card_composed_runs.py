"""The History card describes a composed run by its real folders.

It said "0 images", "the pipeline did not run what was submitted" and
"same as the training set -- not held out" for a run that trained on
6,807 images with a genuinely separate validation set, because the audit
was built from `_generated/<hash>` -- the yaml's home, holding no image.
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ppe_pipeline_api import track2_shared as t2  # noqa: E402


class _Body:
    def __init__(self, t):
        self._t = t.encode()

    def read(self):
        return self._t


class _Paginator:
    def __init__(self, keys):
        self.keys = keys

    def paginate(self, Bucket, Prefix):
        yield {"Contents": [{"Key": k} for k in self.keys if k.startswith(Prefix)]}


class _S3:
    keys = ["ds/images/train/a.jpg", "ds/images/train/b.jpg", "ds/labels/train/a.txt",
            "golden_set/images/train/g.jpg", "ds/images/val/v.jpg", "golden_set/images/val/h.jpg"]

    def get_object(self, Bucket, Key):
        return {"Body": _Body("train:\n- ds/images/train\n- golden_set/images/train\n"
                              "val:\n- ds/images/val\n- golden_set/images/val\nnames:\n  0: helmet\n")}

    def get_paginator(self, _):
        return _Paginator(self.keys)


def _fields(monkeypatch, uri="s3://b/golden_set/,s3://b/ds/"):
    monkeypatch.setattr(t2, "s3", _S3())
    monkeypatch.setattr(t2, "BUCKET", "b")
    return t2.composed_audit_fields(train_data_uris=uri,
                                    data_yaml_uri="s3://b/_generated/x/data.yaml",
                                    eval_uri="s3://b/golden_set/")


def test_composed_fields_count_the_real_folders(monkeypatch):
    f = _fields(monkeypatch)
    assert f["train_dataset"]["file_count"] == 3
    assert f["eval_dataset"]["file_count"] == 2
    assert f["train_dataset"]["source_folders"] == ["ds/images/train", "golden_set/images/train"]


def test_the_requested_uri_is_what_the_job_received(monkeypatch):
    """The card compares this with the job's own training URI; equal means
    no "did not run what was submitted"."""
    uri = "s3://b/golden_set/,s3://b/ds/"
    assert _fields(monkeypatch, uri)["train_dataset"]["s3_uri"] == uri


def test_separate_val_folders_are_held_out(monkeypatch):
    assert _fields(monkeypatch)["eval_is_train_dataset"] is False


def test_the_audit_uses_them_for_a_composed_run():
    code = " ".join(inspect.getsource(t2.write_training_audit).split())
    assert "if train_data_uris and data_yaml_uri:" in code
    assert "record.update(composed_audit_fields(" in code
    run = " ".join(inspect.getsource(t2.start_custom_training_run).split())
    assert 'train_data_uris=train_uri if "," in train_uri else "",' in run


def test_older_rows_can_be_refreshed():
    assert 'if "," not in uri:' in inspect.getsource(t2.refresh_composed_audits)
