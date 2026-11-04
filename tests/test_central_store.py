"""UAT-01 (2026-11-04): every file a dataset version uses is kept in the
central data store at freeze, so the version can always be rebuilt."""
from __future__ import annotations

import io
import json

import pytest

from ppe_pipeline_api.data import central_store as cs


class _S3:
    def __init__(self, objects):
        self.objects = dict(objects)            # (bucket, key) -> bytes
        self.copies = []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {"ETag": '"%s"' % abs(hash(self.objects[(Bucket, Key)]))}

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.copies.append(Key)

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=""):
        self.objects[(Bucket, Key)] = Body


@pytest.fixture
def store(monkeypatch):
    manifest = {"version_id": "dv-1", "train": ["src/set/images/train/a.jpg"],
                "val": ["src/set/images/val/b.jpg", "src/set/images/val/gone.jpg"]}
    s3 = _S3({("src", "set/images/train/a.jpg"): b"A", ("src", "set/labels/train/a.txt"): b"0 .5 .5 .1 .1",
              ("src", "set/images/val/b.jpg"): b"B",
              ("home", "_datasets/dv-1/manifest.json"): json.dumps(manifest).encode()})
    monkeypatch.setattr(cs.storage, "s3", s3)
    monkeypatch.setattr(cs, "location", lambda: ("home", "store"))
    from ppe_pipeline_api.data import catalog
    monkeypatch.setattr(catalog, "configured", lambda: False)
    return s3


VERSION = {"version_id": "dv-1", "manifest_uri": "s3://home/_datasets/dv-1/manifest.json"}


def test_every_file_is_copied_server_side_and_recorded_beside_the_version(store):
    out = cs.preserve(VERSION)
    assert out["files"] == 3 and out["kept"] == 2 and out["missing"] == 1
    kept = [k for (b, k) in store.objects if b == "home" and k.startswith("store/objects/")]
    assert len(kept) == 3                         # two images + one label
    record = json.loads(store.objects[("home", "_datasets/dv-1/preserved.json")])
    a = next(m for m in record["members"] if m["image"].endswith("a.jpg"))
    assert a["stored_image"].startswith("s3://home/store/objects/") and a["stored_label"]
    gone = next(m for m in record["members"] if m["image"].endswith("gone.jpg"))
    assert gone["stored_image"] == ""             # missing at freeze: said, not invented


def test_preserving_again_copies_nothing_twice(store):
    cs.preserve(VERSION)
    first = len(store.copies)
    cs.preserve(VERSION)
    assert len(store.copies) == first


def test_freezing_starts_preservation_and_export_falls_back_to_the_copy():
    import inspect
    from ppe_pipeline_api.data import catalog_view, dataset_versions
    assert "central_store.preserve_in_background(version)" in inspect.getsource(dataset_versions.create)
    src = inspect.getsource(catalog_view)
    assert "stored_image_uri, stored_label_uri" in src


def test_only_new_versions_are_caught_up():
    assert cs.SINCE == "2026-11-04"


def test_the_schema_step_adds_and_drops_nothing():
    from ppe_pipeline_api.data import catalog
    text = " ".join(catalog.MIGRATIONS[8]).upper()
    assert "STORED_IMAGE_URI" in text and "PRESERVED_AT" in text
    assert "DROP" not in text and "DELETE" not in text


# --- R1 (independent UAT, 2026-11-04): an overwritten original must not
# change what a frozen version exports. Driven through the real builder.

class _ExportS3:
    """get/put with ETags: an object's ETag is fixed when it is written."""
    def __init__(self, objects):
        self.objects = {k: (v, f"tag-{v.decode()}") for k, v in objects.items()}

    def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body, tag = self.objects[(Bucket, Key)]
        return {"Body": io.BytesIO(body), "ETag": f'"{tag}"'}

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[(Bucket, Key)] = (Body if isinstance(Body, bytes) else Body.encode(), "")


def _export(monkeypatch, objects):
    import threading
    from ppe_pipeline_api.data import catalog_view as cv
    s3 = _ExportS3(objects)
    monkeypatch.setattr(cv.storage, "s3", s3)
    monkeypatch.setattr(cv, "_rows", lambda *a: [{"state": "building"}])
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, "_building", slots)
    finished = []
    monkeypatch.setattr(cv, "_finish", lambda *a: finished.append(a))
    entries = [("images/train/a.jpg", "s3://src/images/a.jpg",
                {"path": "images/train/a.jpg", "stored": "s3://home/store/objects/tag-ORIGINAL-IMG.jpg"}),
               ("labels/train/a.txt", "s3://src/labels/a.txt",
                {"path": "labels/train/a.txt", "stored": "s3://home/store/objects/tag-ORIGINAL-LBL.txt"})]
    cv._build("x", "version", "dv-1", "version-1", entries, ["helmet"], "s3", "out", "exp", "folder")
    files = {k.split("version-1-x/", 1)[1]: v[0] for (b, k), v in s3.objects.items() if b == "out"}
    return files, finished[-1][2]


def test_an_overwritten_original_does_not_change_a_frozen_export(monkeypatch):
    files, params = _export(monkeypatch, {
        ("src", "images/a.jpg"): b"CHANGED-IMG", ("src", "labels/a.txt"): b"CHANGED-LBL",
        ("home", "store/objects/tag-ORIGINAL-IMG.jpg"): b"ORIGINAL-IMG",
        ("home", "store/objects/tag-ORIGINAL-LBL.txt"): b"ORIGINAL-LBL"})
    assert files["images/train/a.jpg"] == b"ORIGINAL-IMG"
    assert files["labels/train/a.txt"] == b"ORIGINAL-LBL"
    assert params[0] == "ready"


def test_a_changed_original_is_reported_not_packed_when_the_copy_is_gone(monkeypatch):
    files, params = _export(monkeypatch, {
        ("src", "images/a.jpg"): b"CHANGED-IMG", ("src", "labels/a.txt"): b"ORIGINAL-LBL"})
    assert "images/train/a.jpg" not in files
    assert files["labels/train/a.txt"] == b"ORIGINAL-LBL"   # unchanged original is still good
    assert params[0] == "partial" and params[3:5] == (1, 1)
    assert b"original changed since freeze" in files["manifest.csv"]


# --- R2 (independent UAT, 2026-11-04): a run export uses the copies kept for
# the run's version, keeping the run's own role/split layout.

def test_a_run_export_reads_the_kept_copy_of_a_gone_original(monkeypatch):
    from ppe_pipeline_api.data import catalog, catalog_view as cv
    monkeypatch.setattr(catalog, "ensure_schema_once", lambda: None)

    def rows(sql, params=()):
        if "FROM training_runs" in sql:
            return [{"run_number": 7, "version_id": "dv-1"}]
        if "class_names FROM dataset_versions" in sql:
            return [{"class_names": ["helmet"]}]
        if "FROM dataset_members" in sql:
            return [{"image_uri": "s3://src/images/a.jpg", "label_uri": "s3://src/labels/a.txt",
                     "stored_image_uri": "s3://home/store/objects/tag-ORIGINAL-IMG.jpg",
                     "stored_label_uri": "s3://home/store/objects/tag-ORIGINAL-LBL.txt"}]
        if "FROM run_files" in sql:
            return [{"s3_uri": "s3://src/images/a.jpg", "role": "train", "split": "val", "kind": "image"},
                    {"s3_uri": "s3://src/labels/a.txt", "role": "train", "split": "val", "kind": "label"},
                    {"s3_uri": "s3://src/eval/e.jpg", "role": "eval", "split": "val", "kind": "image"}]
        return [{"state": "building"}]

    monkeypatch.setattr(cv, "_rows", rows)
    entries, names, label = cv._plan("run", "r-1")
    assert label == "run-7" and names == ["helmet"]
    by_path = {p: row for p, _, row in entries}
    assert by_path["images/val/a.jpg"]["stored"].endswith("tag-ORIGINAL-IMG.jpg")
    assert by_path["labels/val/a.txt"]["stored"].endswith("tag-ORIGINAL-LBL.txt")
    assert by_path["eval/images/val/e.jpg"]["stored"] == ""    # not a member: no copy assumed

    import threading
    s3 = _ExportS3({("home", "store/objects/tag-ORIGINAL-IMG.jpg"): b"ORIGINAL-IMG",
                    ("home", "store/objects/tag-ORIGINAL-LBL.txt"): b"ORIGINAL-LBL",
                    ("src", "eval/e.jpg"): b"EVAL"})                     # originals of a.* gone
    monkeypatch.setattr(cv.storage, "s3", s3)
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(cv, "_building", slots)
    finished = []
    monkeypatch.setattr(cv, "_finish", lambda *a: finished.append(a))
    cv._build("y", "run", "r-1", label, entries, names, "s3", "out", "exp", "folder")
    files = {k.split("run-7-y/", 1)[1]: v[0] for (b, k), v in s3.objects.items() if b == "out"}
    assert files["images/val/a.jpg"] == b"ORIGINAL-IMG"
    assert files["labels/val/a.txt"] == b"ORIGINAL-LBL"
    assert files["eval/images/val/e.jpg"] == b"EVAL"
    assert finished[-1][2][0] == "ready" and finished[-1][2][3:5] == (3, 0)
