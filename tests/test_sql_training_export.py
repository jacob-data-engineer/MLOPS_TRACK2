import io
import json

import pytest
import yaml
from PIL import Image
from botocore.exceptions import ClientError

from ppe_pipeline_api.data import sql_training_export as ex


def row():
    return {"width": 100, "height": 100, "case_key": "case", "hazard_type": "no_helmet",
            "people": [{"candidate_id": "p1", "person_box": [10, 10, 60, 90],
                        "wearing": True, "verdict": "no_violation", "auto": [20, 10, 40, 30]},
                       {"candidate_id": "p2", "person_box": [60, 10, 90, 90],
                        "wearing": False, "verdict": "violation", "auto": None}]}


class S3:
    def __init__(self):
        body = io.BytesIO()
        Image.new("RGB", (100, 100), "red").save(body, format="JPEG")
        self.objects = {("src", "frame.jpg"): body.getvalue()}

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        return {}

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Bucket, Key])}

    def put_object(self, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and (Bucket, Key) in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Bucket, Key] = Body

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[Bucket, Key] = self.objects[CopySource["Bucket"], CopySource["Key"]]

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):
        return {"Contents": [{"Key": k} for b, k in self.objects if b == Bucket and k.startswith(Prefix)][:MaxKeys]}

    def delete_object(self, Bucket, Key):
        self.objects.pop((Bucket, Key), None)


@pytest.fixture
def setup(monkeypatch):
    data, s3 = row(), S3()
    def run(*args, **kwargs):
        return {"columns": list(data), "rows": [list(data.values())], "images": [
            {"bucket": "src", "key": "frame.jpg", "uri": "s3://src/frame.jpg", "row": 0}]}
    monkeypatch.setattr(ex.sx, "run", run)
    monkeypatch.setattr(ex.sx.storage, "s3", s3)
    return data, s3


def export(**kwargs):
    return ex.export("db", "select", dest_bucket="dest", dest_folder="new", train_percent=80, **kwargs)


def test_mapping_changes_labels_and_yaml_together_and_retry_skips(setup):
    _, s3 = setup
    first = export(names=["person", "helmet"])
    labels = [v.decode() for (b, k), v in s3.objects.items() if k.endswith(".txt")]
    assert len(labels) == 1
    assert [line.split()[0] for line in labels[0].splitlines()] == ["0", "1", "0"]
    assert yaml.safe_load(s3.objects["dest", "new/data.yaml"])["names"] == {0: "person", 1: "helmet"}
    assert first["copied"] == 1
    assert export(names=["person", "helmet"])["skipped"] == 1
    with pytest.raises(ValueError, match="different class mapping"):
        export(names=["helmet", "person"])
    assert ("dest", "new/.export-lock") not in s3.objects


def test_original_pull_remaps_only_person_and_helmet(setup, monkeypatch):
    data, s3 = setup
    labels = ex.sx.frame_labels(data)
    monkeypatch.setattr(ex.sx, "run", lambda *a, **kw: {"images": [
        {"uri": "s3://src/frame.jpg", "bucket": "src", "key": "frame.jpg", "labels": labels}]})
    def pull(ids):
        return ex.sx.pull("db", "q", dest_bucket="dest", dest_folder="simple", class_ids=ids)
    result = pull({"helmet": 1, "person": 0})
    assert result["copied"] == 1
    config = yaml.safe_load(s3.objects["dest", "simple/data.yaml"])
    assert config["names"] == {0: "person", 1: "helmet"}
    text = next(v.decode() for (b, k), v in s3.objects.items() if k.endswith(".txt"))
    assert [line.split()[0] for line in text.splitlines()] == ["0", "1", "0"]
    assert pull({"helmet": 1, "person": 0})["skipped"] == 1
    with pytest.raises(ValueError, match="different class IDs"):
        pull({"helmet": 0, "person": 1})
    assert yaml.safe_load(s3.objects["dest", "simple/data.yaml"])["names"] == config["names"]


def test_classifier_recovers_crops_from_missing_helmet_frames_and_respects_ids(setup):
    data, s3 = setup
    data["people"][0]["auto"] = None
    assert ex.samples(data, "detection", ex.DETECTOR_NAMES)[0] == []
    events = []
    result = export(mode="classification", names=["no_helmet", "helmet_worn"], progress=events.append)
    assert result["copied"] == 2 and result["left_out_incomplete"] == 0
    targets = [k for b, k in s3.objects if b == "dest" and k.endswith(".jpg")]
    assert any("/00_no_helmet/" in k for k in targets)
    assert any("/01_helmet_worn/" in k for k in targets)
    assert len({k.split("/")[1] for k in targets}) == 1
    assert sorted(Image.open(io.BytesIO(s3.objects["dest", k])).size for k in targets) == [(30, 80), (50, 80)]
    assert events[-1]["completed"] == events[-1]["total"] == 2
    assert events[-1]["phase"] == "finalizing"


def test_classifier_does_not_infer_answers_from_wearing_fallback(setup):
    data, _ = setup
    data["people"][0]["verdict"] = None
    data["people"][1]["verdict"] = "cannot_determine"
    with pytest.raises(ValueError, match="unresolved_person_verdict"):
        export(mode="classification")


@pytest.mark.parametrize("box", [[0, 0, 0, 0], [120, 0, 130, 10], [0, 0, float('nan'), 10]])
def test_invalid_person_boxes_are_not_trainable(box):
    data = row()
    data["people"][0]["person_box"] = box
    assert not ex.samples(data, "detection", ex.DETECTOR_NAMES)[0]
    assert len(ex.samples(data, "classification", ex.CLASSIFIER_NAMES)[0]) == 1


def test_nonempty_legacy_destination_is_not_relabelled(setup):
    _, s3 = setup
    s3.objects["dest", "new/data.yaml"] = b"old"
    with pytest.raises(ValueError, match="new empty folder"):
        export()
    assert s3.objects["dest", "new/data.yaml"] == b"old"


def test_changed_human_answer_cannot_append_contradictory_sample(setup):
    data, _ = setup
    export(mode="classification")
    data["people"][0]["verdict"] = "violation"
    with pytest.raises(ValueError, match="changed labels"):
        export(mode="classification")


def test_label_write_failure_does_not_publish_unlabelled_image(setup, monkeypatch):
    _, s3 = setup
    put = s3.put_object
    def fail_label(**kwargs):
        if kwargs["Key"].endswith(".txt"):
            raise RuntimeError("label unavailable")
        return put(**kwargs)
    monkeypatch.setattr(s3, "put_object", fail_label)
    result = export()
    assert result["failed"] == 1 and result["copied"] == 0
    assert not any(b == "dest" and k.endswith(".jpg") for b, k in s3.objects)


def test_lock_collision_preserves_other_export_lock(setup):
    _, s3 = setup
    s3.objects["dest", "new/.export-lock"] = b"other"
    with pytest.raises(ValueError, match="lock"):
        export()
    assert s3.objects["dest", "new/.export-lock"] == b"other"


def test_permission_error_is_not_treated_as_absent(monkeypatch):
    def denied(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
    monkeypatch.setattr(ex.sx.storage.s3, "head_object", denied)
    with pytest.raises(ClientError):
        ex.exists("bucket", "key")


def test_classifier_rejects_other_hazards(setup):
    data, _ = setup
    data["hazard_type"] = "worker_on_edge"
    with pytest.raises(ValueError, match="no-helmet"):
        export(mode="classification")


def test_route_passes_selected_mapping_to_export(monkeypatch):
    from ppe_pipeline_api.routers import track2_sql as routes
    seen = {}
    def capture(*args, **kwargs):
        seen.update(kwargs)
        return {"ok": True}
    monkeypatch.setattr(ex, "export", capture)
    request = routes.PullRequest(database="db", sql="select", dest_bucket="b", dest_folder="new",
                                 export_mode="classification", class_names=["no_helmet", "helmet_worn"])
    assert routes.run_pull(request) == {"ok": True}
    assert seen["mode"] == "classification" and seen["names"] == ["no_helmet", "helmet_worn"]


def test_image_only_export_cannot_pollute_managed_folder(monkeypatch):
    from ppe_pipeline_api.routers import track2_sql as routes
    monkeypatch.setattr(ex, "exists", lambda *args: True)
    request = routes.PullRequest(database="db", sql="select", dest_bucket="b", dest_folder="new", with_labels=False)
    with pytest.raises(ValueError, match="Keep labels enabled"):
        routes.run_pull(request)


@pytest.mark.parametrize("names", [[], ["helmet", "helmet"], ["person"], ["helmet", "person", "../bad"]])
def test_invalid_class_maps_are_rejected(names):
    with pytest.raises(ValueError):
        ex.validate_names("detection", names)
