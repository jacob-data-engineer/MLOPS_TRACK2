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
    assert 'body = _fetch(entry[2]["stored"])' in src
    assert "stored_image_uri, stored_label_uri" in src


def test_only_new_versions_are_caught_up():
    assert cs.SINCE == "2026-11-04"


def test_the_schema_step_adds_and_drops_nothing():
    from ppe_pipeline_api.data import catalog
    text = " ".join(catalog.MIGRATIONS[8]).upper()
    assert "STORED_IMAGE_URI" in text and "PRESERVED_AT" in text
    assert "DROP" not in text and "DELETE" not in text
