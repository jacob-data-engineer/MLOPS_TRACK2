import io
import json

import pytest

from ppe_pipeline_api.audit_files import version_file_rows
from ppe_pipeline_api.data import repair_run_files as repair
from ppe_pipeline_api.training import track2_shared as t2


MANIFEST = {"version_id": "dv-test", "train": ["bucket-one/set/images/train/a.jpg"],
            "val": ["s3://bucket-two/set/images/val/b.png"]}


def test_manifest_records_full_addresses_labels_and_splits():
    rows = version_file_rows("run", MANIFEST, "bucket-one")
    assert len(rows) == 4
    assert rows[0]["s3_uri"] == "s3://bucket-one/set/images/train/a.jpg"
    assert rows[1]["s3_uri"] == "s3://bucket-one/set/labels/train/a.txt"
    assert rows[2]["s3_uri"] == "s3://bucket-two/set/images/val/b.png"
    assert rows[3]["split"] == "val" and rows[3]["role"] == "eval"
    assert rows[0]["name"] == "set/images/train/a.jpg"
    assert rows[2]["name"] == "bucket-two/set/images/val/b.png"
    assert len({r["file_key"] for r in rows}) == 4


def test_version_recorder_writes_real_annotation_rows(monkeypatch):
    written = []
    class Table:
        def batch_writer(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def put_item(self, Item): written.append(Item)
    class S3:
        def get_object(self, **kwargs): return {"Body": io.BytesIO(json.dumps(MANIFEST).encode())}
    monkeypatch.setattr(t2, "s3", S3())
    monkeypatch.setattr(t2, "BUCKET", "bucket-one")
    monkeypatch.setattr(t2, "training_files_table", lambda: Table())
    result = t2._version_audit_fields("s3://bucket-one/_datasets/dv-test/data.yaml",
                                    train_data_uris="", eval_uri="", eval_extra_uris="", audit_id="run")
    assert len(written) == 4
    assert all(r["s3_uri"].startswith("s3://") for r in written)
    assert all(r["file_name"] not in ("train.txt", "val.txt") for r in written)
    assert result["train_dataset"]["label_count"] == 1
    assert result["eval_dataset"]["file_rows"] == 2


def test_repair_refuses_manifest_count_mismatch(monkeypatch):
    class S3:
        def get_object(self, **kwargs): return {"Body": io.BytesIO(json.dumps(MANIFEST).encode())}
    monkeypatch.setattr(repair.storage, "s3", S3())
    monkeypatch.setattr(repair.catalog_view, "_rows", lambda *args: [{"role": "train", "n": 999}])
    with pytest.raises(ValueError, match="counts do not match"):
        repair.plan({"run_id": "run", "run_number": 57, "version_id": "dv-test",
                     "version_record": {"manifest_uri": "s3://bucket-one/manifest.json"}})


def test_manifest_rejects_unresolvable_annotation():
    with pytest.raises(ValueError, match="annotation"):
        version_file_rows("run", {"train": ["bucket/unknown/a.jpg"]}, "bucket")


def test_composed_recorder_keeps_cross_bucket_addresses(monkeypatch):
    written = []
    class Table:
        def batch_writer(self): return self
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def put_item(self, Item): written.append(Item)
    monkeypatch.setattr(t2, "BUCKET", "home")
    monkeypatch.setattr(t2, "training_files_table", lambda: Table())
    monkeypatch.setattr(t2, "_list_all", lambda bucket, prefix: [{"Key": prefix + ("a.jpg" if "images" in prefix else "a.txt")}])
    t2._composed_audit_block(["other/set/images/val"], uri="s3://other/set/", multi_bucket=True, audit_id="run", role="eval")
    assert {r["s3_uri"] for r in written} == {"s3://other/set/images/val/a.jpg", "s3://other/set/labels/val/a.txt"}
    assert all(r["split"] == "val" for r in written)


def test_repair_understands_old_composed_snapshots(monkeypatch):
    snapshot = {"snapshot_id": "snapshot-1", "prefix": "", "root": "_generated/test",
                "files": ["set/images/train/a.jpg", "set/labels/train/a.txt", "set/images/val/b.jpg", "set/labels/val/b.txt"]}
    class S3:
        def get_object(self, **kwargs): return {"Body": io.BytesIO(json.dumps(snapshot).encode())}
    monkeypatch.setattr(repair.storage, "s3", S3())
    monkeypatch.setattr(repair.catalog_view, "_rows", lambda *args: [{"role": "train", "n": 1}, {"role": "eval", "n": 1}])
    rows, summary = repair.plan({"run_id": "run", "run_number": 43, "version_id": "snapshot-1",
        "version_record": {"bucket": "bucket", "manifest_uri": "s3://bucket/snapshot/manifest.json"}})
    assert summary["images"] == 2 and summary["labels"] == 2
    assert rows[0]["s3_uri"] == "s3://bucket/set/images/train/a.jpg"
