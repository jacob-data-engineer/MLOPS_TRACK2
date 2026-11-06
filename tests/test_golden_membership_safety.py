"""Behavioral tests of rebuilds against a local S3 double; no AWS calls."""
import hashlib
import io
import json

import pytest
from botocore.exceptions import ClientError

from ppe_pipeline_api import golden_set as gs


class Store:
    def __init__(self):
        self.objects = {}
        self.fail_copy = False

    def add(self, name, body=b"image"):
        self.objects[name] = body

    def list(self, prefix, bucket):
        return [{"Key": k, "Size": len(v), "ETag": hashlib.md5(v).hexdigest()}
                for k, v in sorted(self.objects.items()) if k.startswith(prefix)]

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}

    def put_object(self, **kwargs):
        if kwargs.get("IfNoneMatch") and kwargs["Key"] in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {}

    def copy_object(self, **kwargs):
        if self.fail_copy:
            raise RuntimeError("copy failed")
        self.objects[kwargs["Key"]] = self.objects[kwargs["CopySource"]["Key"]]
        return {}

    def delete_objects(self, **kwargs):
        for obj in kwargs["Delete"]["Objects"]:
            self.objects.pop(obj["Key"], None)
        return {}


@pytest.fixture
def store(monkeypatch):
    s = Store()
    monkeypatch.setattr(gs.pc, "s3", s)
    monkeypatch.setattr(gs.pc, "_list_objects", s.list)
    monkeypatch.setattr(gs.pc, "destination_bucket", lambda: "test")
    for i in range(100):
        s.add(f"source/images/img{i:04}.jpg")
        s.add(f"source/labels/img{i:04}.txt", b"0 0.5 0.5 0.2 0.2")
    return s


def manifest(store):
    return json.loads(store.objects["target/split.json"])


def test_growth_never_moves_validation_and_preserves_benchmark(store):
    first = gs.build(source="source", target_folder="target")
    before = manifest(store)
    store.add("source/images/img0101.jpg")
    store.add("source/labels/img0101.txt", b"")
    preview = gs.plan(source="source", target_folder="target")
    assert preview["membership_locked"] and preview["new_training_pairs"] == 1
    second = gs.build(source="source", target_folder="target")
    after = manifest(store)
    assert set(before["val"]) == set(after["val"])
    assert set(before["train"]) < set(after["train"])
    assert first["benchmark_id"] == second["benchmark_id"]
    assert first["split_version"] != second["split_version"]
    assert len([k for k in store.objects if k.startswith("target/split-versions/")]) == 2


def test_changed_split_requires_new_target(store):
    gs.build(source="source", target_folder="target")
    with pytest.raises(ValueError, match="new target"):
        gs.build(source="source", target_folder="target", train_percent=50)
    gs.build(source="source", target_folder="new-target", train_percent=50)


@pytest.mark.parametrize("kind", ["image", "label"])
def test_changed_validation_bytes_require_new_benchmark(store, kind):
    gs.build(source="source", target_folder="target")
    old = manifest(store)
    name = old["validation_objects"][old["val"][0]][kind]["key"]
    store.add(name, b"changed")
    with pytest.raises(ValueError, match="Frozen validation .* changed"):
        gs.build(source="source", target_folder="target")


def test_deleted_validation_pair_is_not_silently_dropped(store):
    gs.build(source="source", target_folder="target")
    name = manifest(store)["val"][0]
    store.objects.pop(f"source/images/{name}.jpg")
    with pytest.raises(ValueError, match="frozen validation pair"):
        gs.build(source="source", target_folder="target")


def test_legacy_manifest_assignments_are_adopted(store):
    train, val = gs.split_stems([f"img{i:04}" for i in range(100)], 80)
    store.add("target/split.json", json.dumps({"source": "s3://test/source", "split": True,
               "train_percent": 80, "train": train, "val": val}).encode())
    store.add("source/images/img0101.jpg")
    store.add("source/labels/img0101.txt", b"")
    gs.build(source="source", target_folder="target")
    assert set(manifest(store)["val"]) == set(val)


def test_nested_duplicate_basenames_keep_their_own_labels_and_destinations(store):
    store.objects.clear()
    for camera in ("a", "b"):
        store.add(f"source/images/{camera}/frame.jpg", camera.encode())
        store.add(f"source/labels/{camera}/frame.txt", camera.encode())
    got = gs.build(source="source", target_folder="target", split=False)
    assert got["pairs"] == 2
    for camera in ("a", "b"):
        assert store.objects[f"target/images/{camera}/frame.jpg"] == camera.encode()
        assert store.objects[f"target/labels/{camera}/frame.txt"] == camera.encode()


def test_ambiguous_extensions_fail_before_copying(store):
    store.add("source/images/img0000.png")
    with pytest.raises(ValueError, match="Ambiguous image pair"):
        gs.build(source="source", target_folder="target")
    assert not any(k.startswith("target/") for k in store.objects)


def test_failed_copy_does_not_publish_success_manifest(store):
    store.fail_copy = True
    with pytest.raises(ValueError, match="copying failed"):
        gs.build(source="source", target_folder="target")
    assert manifest(store)["state"] == "building"
    val = manifest(store)["val"]
    store.fail_copy = False
    store.add("source/images/img0101.jpg")
    store.add("source/labels/img0101.txt", b"")
    gs.build(source="source", target_folder="target")
    assert manifest(store)["state"] == "ready"
    assert manifest(store)["val"] == val


def test_an_existing_writer_blocks_the_build(store):
    store.add("target/.split-build-lock.json", b"{}")
    with pytest.raises(ValueError, match="Another golden build"):
        gs.build(source="source", target_folder="target")
    assert "target/split.json" not in store.objects


def test_nested_target_cannot_reingest_itself(store):
    with pytest.raises(ValueError, match="must not contain"):
        gs.build(source="source", target_folder="source/output")
