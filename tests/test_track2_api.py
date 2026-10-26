"""Tests for the `track2_*` routers -- the YOLO/Track2 training-pipeline
control surface ported from `terraform/review-app/review_app.py`. Fakes
stand in for DynamoDB/SageMaker/EventBridge/SNS, same hand-rolled-fake
pattern as `test_track2_status.py`."""

from __future__ import annotations

import pytest
from botocore.exceptions import ClientError
from fastapi.testclient import TestClient

from ppe_pipeline_api import track2_shared as t2
from ppe_pipeline_api.main import app

_PASSWORD = "test-suite-password"


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("PPE_APP_PASSWORD", _PASSWORD)
    c = TestClient(app)
    token = c.post("/auth/login", json={"password": _PASSWORD}).json()["token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


def test_track2_endpoints_require_auth():
    c = TestClient(app)
    assert c.get("/track2/lineage").status_code == 401


# --- fakes --------------------------------------------------------------------

class FakeTable:
    def __init__(self):
        self.items: dict[tuple, dict] = {}
        self.key_name = "id"

    def scan(self, **kwargs):
        return {"Items": list(self.items.values())}

    def get_item(self, Key):
        item = self.items.get(tuple(Key.values()))
        return {"Item": item} if item else {}

    def put_item(self, Item):
        self.items[tuple(Item[k] for k in self._key_of(Item))] = Item

    def update_item(self, Key, UpdateExpression, ExpressionAttributeNames, ExpressionAttributeValues):
        item = self.items.setdefault(tuple(Key.values()), dict(Key))
        item["status"] = ExpressionAttributeValues[":status"]
        item["reviewed_at"] = ExpressionAttributeValues[":ts"]

    def delete_item(self, Key):
        self.items.pop(tuple(Key.values()), None)

    def _key_of(self, item):
        return ["id"] if "id" in item else list(item.keys())[:1]


class FakeSnsClient:
    def __init__(self):
        self.subscriptions: list[dict] = []

    def list_subscriptions_by_topic(self, TopicArn):
        return {"Subscriptions": self.subscriptions}

    def subscribe(self, TopicArn, Protocol, Endpoint, Attributes=None):
        self.subscriptions.append({
            "SubscriptionArn": "PendingConfirmation", "Endpoint": Endpoint, "Protocol": Protocol,
        })

    class exceptions:
        class InvalidParameterException(Exception):
            pass

    def unsubscribe(self, SubscriptionArn):
        # Real SNS rejects anything that is not an ARN. The old mock was a
        # plain list filter, so it happily "removed" the literal
        # "PendingConfirmation" -- and the test asserting Remove works for
        # an unconfirmed subscriber passed against a path that raises in
        # production. A mock that is more permissive than the service it
        # stands in for certifies code that cannot run.
        if not SubscriptionArn.startswith("arn:"):
            raise self.exceptions.InvalidParameterException(
                f"Invalid parameter: SubscriptionArn Reason: {SubscriptionArn} "
                "is not a valid subscription ARN"
            )
        self.subscriptions = [s for s in self.subscriptions if s["SubscriptionArn"] != SubscriptionArn]


class FakeEventsClient:
    def __init__(self):
        self.rules: dict[str, dict] = {}

    class exceptions:
        class ResourceNotFoundException(Exception):
            pass

    def describe_rule(self, Name):
        if Name not in self.rules:
            raise self.exceptions.ResourceNotFoundException()
        return self.rules[Name]

    def put_rule(self, Name, ScheduleExpression, State):
        self.rules[Name] = {"State": State, "ScheduleExpression": ScheduleExpression}

    def disable_rule(self, Name):
        self.rules.setdefault(Name, {})["State"] = "DISABLED"

    def put_targets(self, Rule, Targets):
        pass

    def remove_targets(self, Rule, Ids):
        pass

    def delete_rule(self, Name):
        self.rules.pop(Name, None)


# --- lineage / notifications (simplest) ---------------------------------------

def test_lineage_empty(client, monkeypatch):
    fake = FakeTable()
    monkeypatch.setattr(t2, "lineage_table", lambda: fake)
    assert client.get("/track2/lineage").json() == {"records": []}


def test_lineage_sorted_newest_first(client, monkeypatch):
    fake = FakeTable()
    fake.items[("a",)] = {"id": "a", "timestamp_utc": "2026-01-01T00:00:00Z"}
    fake.items[("b",)] = {"id": "b", "timestamp_utc": "2026-06-01T00:00:00Z"}
    monkeypatch.setattr(t2, "lineage_table", lambda: fake)
    records = client.get("/track2/lineage").json()["records"]
    assert [r["id"] for r in records] == ["b", "a"]


def test_notifications_add_list_remove(client, monkeypatch):
    fake_sns = FakeSnsClient()
    monkeypatch.setattr(t2, "sns", fake_sns)
    monkeypatch.setattr(t2, "reviewer_notifications_topic_arn", lambda: "arn:aws:sns:x:1:topic")

    assert client.get("/track2/notifications").json()["subscriptions"] == []

    add = client.post("/track2/notifications/add", json={"email": "a@example.com"})
    assert add.status_code == 200
    listing = client.get("/track2/notifications").json()["subscriptions"]
    # `events` is None: an unconfirmed subscriber's choices cannot be
    # read back until it has an ARN.
    assert listing == [{"subscription_arn": "PendingConfirmation", "email": "a@example.com",
                        "confirmed": False, "events": None}]

    # An unconfirmed subscriber has no ARN to unsubscribe. This used to
    # assert {"removed": True}, and passed, because the mock was a list
    # filter rather than SNS -- in production the same call raised
    # InvalidParameterException and the page said "Remove failed." forever.
    remove = client.post("/track2/notifications/remove", json={"subscription_arn": "PendingConfirmation"})
    assert remove.status_code == 422
    assert "has not confirmed" in remove.json()["detail"]
    assert client.get("/track2/notifications").json()["subscriptions"] != []

    # ...and a confirmed one still removes.
    fake_sns.subscriptions = [{
        "SubscriptionArn": "arn:aws:sns:x:1:topic:abc", "Endpoint": "b@example.com", "Protocol": "email",
    }]
    remove = client.post("/track2/notifications/remove",
                         json={"subscription_arn": "arn:aws:sns:x:1:topic:abc"})
    assert remove.json() == {"removed": True}
    assert client.get("/track2/notifications").json()["subscriptions"] == []


def test_the_sns_mock_rejects_a_non_arn_the_way_sns_does(client, monkeypatch):
    """Pins the mock itself.

    The bug this file carried was not in the router -- it was that the
    double was more permissive than the service, so a test could certify a
    call that cannot succeed. If this ever passes a bare string again, the
    Remove test goes back to proving nothing.
    """
    import pytest

    fake_sns = FakeSnsClient()
    with pytest.raises(FakeSnsClient.exceptions.InvalidParameterException):
        fake_sns.unsubscribe(SubscriptionArn="PendingConfirmation")


def test_notifications_add_rejects_empty_email(client, monkeypatch):
    monkeypatch.setattr(t2, "reviewer_notifications_topic_arn", lambda: "arn:aws:sns:x:1:topic")
    response = client.post("/track2/notifications/add", json={"email": "   "})
    assert response.status_code == 422


# --- review queue / model registry ---------------------------------------------

def test_review_counts_and_empty_registries(client, monkeypatch):
    fake_table = FakeTable()
    fake_table.items[("r1",)] = {"review_id": "r1", "status": "pending"}
    fake_table.items[("r2",)] = {"review_id": "r2", "status": "approved"}
    monkeypatch.setattr(t2, "table", lambda: fake_table)

    class FakeSm:
        def list_model_packages(self, **kwargs):
            raise ClientError({"Error": {"Code": "ValidationException", "Message": "does not exist"}}, "ListModelPackages")

    monkeypatch.setattr(t2, "sm", FakeSm())

    body = client.get("/track2/review").json()
    assert body["counts"] == {"pending": 1, "approved": 1, "rejected": 0}
    assert all(v["total"] == 0 for v in body["registries"].values())


def test_review_approve_updates_table(client, monkeypatch):
    fake_table = FakeTable()
    monkeypatch.setattr(t2, "table", lambda: fake_table)
    response = client.post("/track2/review/r1/approve")
    assert response.json() == {"review_id": "r1", "status": "approved"}
    assert fake_table.items[("r1",)]["status"] == "approved"


def test_review_model_delete_failure_returns_400(client, monkeypatch):
    class FakeSm:
        def delete_model_package(self, ModelPackageName):
            raise ClientError({"Error": {"Code": "ValidationException", "Message": "still referenced"}}, "DeleteModelPackage")

    monkeypatch.setattr(t2, "sm", FakeSm())
    response = client.post("/track2/review/model/arn:aws:sagemaker:x/g/1/delete")
    assert response.status_code == 400


# --- schedule ------------------------------------------------------------------

def test_schedule_manual_by_default(client, monkeypatch):
    fake_events = FakeEventsClient()
    monkeypatch.setattr(t2, "events", fake_events)
    monkeypatch.setattr(t2, "get_training_progress", lambda pipeline_name="": None)

    body = client.get("/track2/schedule").json()
    assert body["schedule"] == {"enabled": False, "expression": ""}


def test_schedule_set_daily_and_disable(client, monkeypatch):
    fake_events = FakeEventsClient()
    monkeypatch.setattr(t2, "events", fake_events)

    response = client.post("/track2/schedule", json={"mode": "daily", "daily_hour": 9})
    assert response.status_code == 200
    assert fake_events.rules[t2.RULE_NAME]["ScheduleExpression"] == "cron(0 1 * * ? *)"

    disable = client.post("/track2/schedule", json={"mode": "manual"})
    assert disable.json() == {"message": "Schedule disabled - manual trigger only"}
    assert fake_events.rules[t2.RULE_NAME]["State"] == "DISABLED"


def test_train_now_conflicts_when_already_running(client, monkeypatch):
    monkeypatch.setattr(t2, "training_already_running_reason", lambda progress: "busy")
    monkeypatch.setattr(t2, "get_training_progress", lambda pipeline_name="": None)
    response = client.post("/track2/schedule/train-now")
    assert response.status_code == 409


# --- ad-hoc queue ----------------------------------------------------------------

def test_queue_add_remove_start_stop(client, monkeypatch):
    monkeypatch.setattr(t2, "_training_queue", [], raising=False)
    monkeypatch.setattr(t2, "_queue_running", None, raising=False)
    monkeypatch.setattr(t2, "_queue_started", False, raising=False)
    monkeypatch.setattr(t2, "training_already_running_reason", lambda progress: "busy, stay queued")
    monkeypatch.setattr(t2, "get_training_progress", lambda pipeline_name="": None)
    # GET /track2/models also fetches registries, the S3 browsers and GPU
    # usage -- none of that is what this test is about, so it is all
    # stubbed out rather than hitting real AWS.
    monkeypatch.setattr(t2, "all_model_registries", lambda: {})
    monkeypatch.setattr(t2, "list_s3_folders", lambda prefix: [])
    monkeypatch.setattr(t2, "list_s3_folders_and_weight_files", lambda prefix: ([], []))
    # The dataset browser lists yaml files beside the folders now, so the
    # picker can offer the data.yaml itself rather than a folder tick box.
    monkeypatch.setattr(t2, "list_s3_folders_and_yaml_files", lambda prefix: ([], []))
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    # "data/x" is a stand-in, not a real dataset: without this the entry is
    # rejected for having no data.yaml, which is the pre-flight check doing
    # its job rather than anything this test is about.
    monkeypatch.setattr(t2, "validate_dataset_folder",
                        lambda folder, *, used_for_evaluation, custom_classes=False, detector_slug="": None)

    add = client.post("/track2/models/queue/add", json={
        "generation": "11", "size": "s", "dataset_folder": "data/x",
    })
    assert add.status_code == 200, add.json()
    entry_id = add.json()["entry"]["id"]

    snapshot = client.get("/track2/models").json()["queue"]
    assert len(snapshot["pending"]) == 1

    started = client.post("/track2/models/queue/start")
    assert started.status_code == 200

    removed = client.post(f"/track2/models/queue/remove/{entry_id}")
    # Already popped by queue/start's advance attempt in some runs, or still
    # present -- either a clean 200 or a 404 "already started/removed" is
    # correct, never a 500.
    assert removed.status_code in (200, 404)

    stopped = client.post("/track2/models/queue/stop")
    assert stopped.status_code == 200


def test_queued_custom_run_uses_the_custom_pipeline(client, monkeypatch):
    """A queued custom-class entry must reach start_custom_training_run.

    Before custom_classes was carried on the entry, the runner sent every
    queued entry to start_training_run -- so ticking "train on this
    dataset's own classes" and queueing it silently ran a PRODUCTION run,
    with no error anywhere.
    """
    monkeypatch.setattr(t2, "_training_queue", [], raising=False)
    monkeypatch.setattr(t2, "_queue_running", None, raising=False)
    monkeypatch.setattr(t2, "_queue_started", False, raising=False)
    monkeypatch.setattr(t2, "get_training_progress", lambda pipeline_name="": None)
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    monkeypatch.setattr(t2, "validate_dataset_folder",
                        lambda folder, *, used_for_evaluation, custom_classes=False, detector_slug="": None)

    monkeypatch.setattr(t2, "resolve_base_model_s3_uri", lambda folder: f"s3://b/{folder}/model.pt")
    calls = {"custom": [], "production": []}
    monkeypatch.setattr(t2, "start_custom_training_run",
                        lambda **kw: calls["custom"].append(kw) or
                        {"run_id": "custom-1", "execution_arn": "arn:exec/custom-1", "audit_id": "a1"})
    monkeypatch.setattr(t2, "start_training_run",
                        lambda *a, **kw: calls["production"].append(a) or "build-1")

    add = client.post("/track2/models/queue/add", json={
        "generation": "11", "size": "s", "dataset_folder": "data/x",
        "custom_classes": True, "base_model_folder": "weights/mine",
        "architecture": "yolov8n",
    })
    assert add.status_code == 200, add.json()
    assert add.json()["entry"]["custom_classes"] is True

    # Nothing else holding the GPU, so the advance actually starts it.
    monkeypatch.setattr(t2, "training_already_running_reason", lambda progress: None)
    assert client.post("/track2/models/queue/start").status_code == 200

    assert calls["production"] == [], "queued custom run leaked into the production pipeline"
    assert len(calls["custom"]) == 1
    assert calls["custom"][0]["architecture"] == "yolov8n"
    # The runner forwards the selection; start_custom_training_run's
    # dispatcher is what turns it into a URI.
    assert calls["custom"][0]["base_model_folder"] == "weights/mine"
    assert calls["custom"][0]["base_model_scratch"] is False


def test_queue_add_rejects_custom_run_without_exactly_one_base_model(client, monkeypatch):
    monkeypatch.setattr(t2, "_training_queue", [], raising=False)
    monkeypatch.setattr(t2, "validate_dataset_folder",
                        lambda folder, *, used_for_evaluation, custom_classes=False, detector_slug="": None)
    # Neither base model source given.
    neither = client.post("/track2/models/queue/add", json={
        "generation": "11", "size": "s", "dataset_folder": "data/x", "custom_classes": True,
    })
    assert neither.status_code == 400
    # Both given.
    both = client.post("/track2/models/queue/add", json={
        "generation": "11", "size": "s", "dataset_folder": "data/x", "custom_classes": True,
        "base_model_folder": "weights/mine", "base_model_package_arn": "arn:pkg/1",
    })
    assert both.status_code == 400


def test_base_model_upload_goes_through_the_api_not_a_presigned_put(client, monkeypatch):
    """The upload must be same-origin.

    A presigned browser->S3 PUT is blocked by track2-s3's CORS policy,
    which allows only *.sagemaker.aws origins, so this route streams the
    file through the app instead and verifies it server-side.
    """
    captured = {}

    def fake_upload(fileobj, filename, architecture):
        captured["bytes"] = fileobj.read()
        captured["filename"] = filename
        captured["architecture"] = architecture
        return {"key": f"external_models/uploaded/{architecture}/abc123/model.pt",
                "folder": f"external_models/uploaded/{architecture}/abc123",
                "s3_uri": "s3://track2-s3/external_models/uploaded/x/abc123/model.pt",
                "declared_architecture": architecture}

    monkeypatch.setattr(t2, "upload_base_model_fileobj", fake_upload)
    monkeypatch.setattr(t2, "inspect_checkpoint", lambda key, arch: {
        "key": key, "declared_architecture": arch, "detected_architecture": arch,
        "architecture_matches": True, "architecture_known": True,
        "classes": {"0": "person"}, "size_bytes": 4,
    })

    response = client.post(
        "/track2/models/base-models/upload",
        data={"architecture": "yolo11s"},
        files={"file": ("best.pt", b"WEIGHTS", "application/octet-stream")},
    )
    assert response.status_code == 200, response.text
    assert captured["bytes"] == b"WEIGHTS"
    assert captured["filename"] == "best.pt"
    assert captured["architecture"] == "yolo11s"
    body = response.json()
    assert body["architecture_matches"] is True
    assert body["key"].endswith("/model.pt")


def test_base_model_upload_rejects_a_bad_architecture(client, monkeypatch):
    def refuse(fileobj, filename, architecture):
        raise ValueError(f"'{architecture}' is not a known architecture.")

    monkeypatch.setattr(t2, "upload_base_model_fileobj", refuse)
    response = client.post(
        "/track2/models/base-models/upload",
        data={"architecture": "not-a-model"},
        files={"file": ("best.pt", b"W", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert "not a known architecture" in response.json()["detail"]


@pytest.mark.parametrize("arch", ["yolo11s", "yolo11m", "yolo11l", "yolo11x"])
def test_scratch_run_resolves_the_seed_for_every_size(monkeypatch, arch):
    """Executes start_custom_training_run for real, for all four sizes.

    Every earlier test monkeypatched this function, so its body never ran
    and a leftover reference to a deleted parameter reached production as
    "cannot access local variable 'base_model_uri'". These call it.
    """
    started = {}

    class FakeSm:
        def start_pipeline_execution(self, **kwargs):
            started.update(kwargs)
            return {"PipelineExecutionArn": "arn:aws:sagemaker:x:1:pipeline/p/execution/e1"}

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "validate_training_params", lambda *a, **k: (1, 640, 1, 42))
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    monkeypatch.setattr(t2, "write_training_audit", lambda **kw: {"id": "audit1", **kw})
    monkeypatch.setattr(t2, "training_audit_table", lambda: FakeTable())

    result = t2.start_custom_training_run(
        dataset_folder="data/x", eval_dataset_folder="", epochs=1, img_size=640,
        batch=1, seed=42, notes="", architecture=arch, base_model_scratch=True,
    )

    params = {p["Name"]: p["Value"] for p in started["PipelineParameters"]}
    assert params["BaseModelUri"] == f"s3://{t2.BUCKET}/external_models/scratch/{arch}/model.pt"
    assert params["Architecture"] == arch
    # 32 is the documented ClientRequestToken minimum.
    assert len(started["ClientRequestToken"]) >= 32
    assert result["run_id"].startswith("custom-")


def test_scratch_run_without_an_architecture_says_so(monkeypatch):
    monkeypatch.setattr(t2, "validate_training_params", lambda *a, **k: (1, 640, 1, 42))
    with pytest.raises(ValueError, match="architecture"):
        t2.start_custom_training_run(
            dataset_folder="data/x", eval_dataset_folder="", epochs=1, img_size=640,
            batch=1, seed=42, notes="", architecture="", base_model_scratch=True,
        )


@pytest.mark.parametrize("arch,group", [
    ("yolo11s", "yolo11s-open-detector"), ("yolo11m", "yolo11m-open-detector"),
    ("yolo11l", "yolo11l-open-detector"), ("yolo11x", "yolo11x-open-detector"),
])
def test_each_size_registers_into_its_own_group(monkeypatch, arch, group):
    """Per-size groups, so version numbers mean something per size.

    The names must NOT be the production yolo11<size>-detector groups: those
    are an exact-match list on the EventBridge rule that fires
    dump_approved_weights, which swaps the live inference component and
    overwrites production weights. These runs register as Approved.
    """
    started = {}

    class FakeSm:
        def start_pipeline_execution(self, **kwargs):
            started.update(kwargs)
            return {"PipelineExecutionArn": "arn:aws:sagemaker:x:1:pipeline/p/execution/e1"}

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "validate_training_params", lambda *a, **k: (1, 640, 1, 42))
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    monkeypatch.setattr(t2, "write_training_audit", lambda **kw: {"id": "a1", **kw})
    monkeypatch.setattr(t2, "training_audit_table", lambda: FakeTable())

    t2.start_custom_training_run(
        dataset_folder="data/x", eval_dataset_folder="", epochs=1, img_size=640,
        batch=1, seed=42, notes="", architecture=arch, base_model_scratch=True,
    )
    params = {p["Name"]: p["Value"] for p in started["PipelineParameters"]}
    assert params["RegistryGroup"] == group
    assert params["Architecture"] == arch
    assert params["BaseModelSource"] == "scratch"
    # The four names the approval rule watches must never be produced here.
    assert group not in {f"yolo11{s}-detector" for s in ("s", "m", "l", "x")}


def test_a_run_without_an_architecture_cannot_pick_a_group(monkeypatch):
    monkeypatch.setattr(t2, "validate_training_params", lambda *a, **k: (1, 640, 1, 42))
    with pytest.raises(ValueError, match="not a known architecture"):
        t2.start_custom_training_run(
            dataset_folder="data/x", eval_dataset_folder="", epochs=1, img_size=640,
            batch=1, seed=42, notes="", architecture="", base_model_scratch=True,
        )


def test_versioned_weights_are_written_once_and_reused(monkeypatch):
    """The versioned key is immutable, so a second call must not re-upload."""
    arn = "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11m-open-detector/7"
    exists = {"value": False}
    uploaded = []

    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: exists["value"])

    class FakeSm:
        def describe_model_package(self, ModelPackageName):
            return {"InferenceSpecification": {"Containers": [
                {"ModelDataUrl": f"s3://{t2.BUCKET}/training-output/run/output/model.pt"}]}}

    class FakeS3:
        def download_file(self, bucket, key, local):
            open(local, "wb").write(b"WEIGHTS")

        def upload_file(self, local, bucket, key):
            uploaded.append(key)

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "s3", FakeS3())

    uri = t2.publish_versioned_weights(arn)
    assert uri == f"s3://{t2.BUCKET}/models/yolo11m-open-detector/v7/model.pt"
    assert uploaded == ["models/yolo11m-open-detector/v7/model.pt"]

    # Second call: the object now exists, so nothing is re-uploaded.
    exists["value"] = True
    assert t2.publish_versioned_weights(arn) == uri
    assert len(uploaded) == 1


def test_a_matching_taxonomy_needs_no_filtered_copy(monkeypatch):
    """Deriving copies a dataset to strip classes the detector does not
    want. When the dataset ALREADY declares exactly the detector's
    classes at the same indices there is nothing to strip, and the copy
    would change nothing.

    It also unblocks the shape a generated data.yaml produces. A derive
    reads `<source>/images/train` directly, which only exists for a
    dataset assembled into one folder; a root of week folders has no such
    path, and this failed with "no images under images/train" on a
    dataset that needed nothing done to it.
    """
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "dataset_classes", lambda folder: {0: "helmet"})
    called = []
    monkeypatch.setattr(t2, "_list_keys", lambda prefix: called.append(prefix) or [])

    out = t2.derive_filtered_dataset(
        "retraining", {"slug": "helmet", "classes": {0: "helmet"}})
    assert out == "retraining"
    assert called == []            # nothing was listed, nothing was copied


def test_a_different_taxonomy_still_derives(monkeypatch):
    """The skip is only for an exact match. A dataset labelled with more
    than the detector wants must still be filtered, or the single output
    class learns from several kinds of box."""
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "dataset_classes",
                        lambda folder: {0: "person", 1: "helmet"})
    monkeypatch.setattr(t2, "_list_keys", lambda prefix: [])
    monkeypatch.setattr(t2, "list_s3_folders", lambda prefix: [])
    monkeypatch.setattr(t2.s3, "get_object",
                        lambda **kw: (_ for _ in ()).throw(KeyError("no marker")))

    # Reaches the copy, finds nothing, and says so -- rather than
    # returning the source unfiltered, which would train helmet on
    # person boxes.
    with pytest.raises(ValueError, match="nothing to filter"):
        t2.derive_filtered_dataset(
            "assembled", {"slug": "helmet", "classes": {0: "helmet"}})


def test_a_one_class_detector_is_isolated_without_copying_anything(monkeypatch):
    """Six-class site dataset, one-class detector.

    `classes=[n]` drops every other box at load time and `single_cls`
    renumbers what survives to 0 and sizes the head to one output. So the
    week folders stay the record and no filtered copy is written -- which
    also means it works on a dataset whose yaml names week folders in
    place, where a copy cannot even be read.
    """
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "dataset_classes", lambda folder: {
        0: "helmet", 1: "gloves", 2: "person",
        3: "working_boots", 4: "man_cage", 5: "safe_area",
    })
    out = t2.class_filter_for("retraining", {"slug": "helmet", "classes": {0: "helmet"}})
    assert out == {"classes": [0], "single_cls": True}

    # And the index is the DATASET's, not the detector's.
    out = t2.class_filter_for("retraining", {"slug": "boots",
                                             "classes": {0: "working_boots"}})
    assert out == {"classes": [3], "single_cls": True}


def test_an_exact_match_needs_no_filtering_at_all(monkeypatch):
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "dataset_classes", lambda folder: {0: "helmet"})
    assert t2.class_filter_for("f", {"slug": "helmet", "classes": {0: "helmet"}}) == {}


def test_several_wanted_classes_still_need_a_rewritten_copy(monkeypatch):
    """`classes=` keeps their ORIGINAL indices and `single_cls` would
    collapse them all into one, which is a different model from the one
    asked for. Neither is a two-class detector, so the copy is the only
    way to renumber them to 0 and 1."""
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "dataset_classes", lambda folder: {
        0: "helmet", 1: "gloves", 2: "person"})
    out = t2.class_filter_for("f", {"slug": "ppe",
                                    "classes": {0: "helmet", 1: "gloves"}})
    assert out == {"derive": True}


def test_a_class_the_dataset_does_not_have_falls_through_to_the_copy(monkeypatch):
    """Which then reports what is missing, rather than this quietly
    filtering on an index that means something else."""
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "dataset_classes", lambda folder: {0: "helmet"})
    out = t2.class_filter_for("f", {"slug": "cage", "classes": {0: "man_cage"}})
    assert out == {"derive": True}


# --- a folder that is not a folder --------------------------------------------
#
# The UAT sweep found this by asking for `?output_folder=../../etc/passwd`.
# S3 answers a malformed key with a 400, which is not one of the codes the
# NoSuchKey check looks for, so the ClientError escaped and the browser got
# a 500 with a traceback behind it. Nothing was reachable through it -- `..`
# means nothing to S3 -- but a 500 says the service is broken when the true
# answer is that the argument is wrong.

def test_a_traversal_folder_is_refused_before_it_reaches_s3():
    from ppe_pipeline_api import track2_shared as t2

    with pytest.raises(ValueError, match="not a folder"):
        t2.safe_key_part("../../etc/passwd", field="output_folder")


def test_an_empty_folder_is_refused_too():
    from ppe_pipeline_api import track2_shared as t2

    with pytest.raises(ValueError, match="empty"):
        t2.safe_key_part("   ", field="output_folder")


def test_an_ordinary_folder_survives_with_its_slashes_trimmed():
    from ppe_pipeline_api import track2_shared as t2

    assert t2.safe_key_part("/runs/2026-10-13/", field="f") == "runs/2026-10-13"


def test_the_lineage_route_answers_400_not_500(client):
    """The route, not just the helper: a ValueError with no global handler
    is a 500, so the catch in the router is the half that matters."""
    res = client.get("/track2/deploy/lineage", params={"output_folder": "../../etc/passwd"})
    assert res.status_code == 400
    assert "not a folder" in res.json()["detail"]


# --- the metrics scan must not trade slow for broken --------------------------
#
# Describing every registered version one at a time put Review & Metrics at
# 12-15s before a single row appeared. Running the GROUP LISTING in parallel
# alongside it was tried and SageMaker throttled it --
# "ListModelPackages ... (reached max retries: 4): Rate exceeded" -- which
# botocore raises and which 500s the whole page. So the listing is serial
# again, and the part that IS parallel falls back instead of failing.

def _throttle(operation):
    return ClientError(
        {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
        operation,
    )


def test_a_throttled_parallel_scan_falls_back_instead_of_500ing(monkeypatch):
    import threading

    monkeypatch.setattr(t2, "exportable_groups", lambda: [
        {"group_name": "yolo11s-demo-detector", "size": "s", "registry": "open"},
    ])
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"packages": [
        {"ModelPackageArn": f"arn:aws:sagemaker:::model-package/{g}/{n}",
         "ModelApprovalStatus": "Approved"}
        for n in (1, 2, 3)
    ]})

    main = threading.current_thread()

    def only_on_the_main_thread(arn):
        # Exactly the real failure: the pooled calls are the ones that get
        # rate-limited, the serial retry on the main thread goes through.
        if threading.current_thread() is not main:
            raise _throttle("DescribeModelPackage")
        return ({"mAP50": 0.5}, "", None)

    monkeypatch.setattr(t2, "fetch_metrics_and_lineage", only_on_the_main_thread)

    rows = t2.scan_all_model_metrics()
    assert [r["version"] for r in rows] == ["1", "2", "3"]
    assert all(r["metrics"] == {"mAP50": 0.5} for r in rows)


def test_the_rows_are_the_same_whichever_path_ran(monkeypatch):
    """The fallback is only worth having if it produces the same answer --
    a slower page is fine, a differently-ordered one is not."""
    monkeypatch.setattr(t2, "exportable_groups", lambda: [
        {"group_name": "yolo11s-demo-detector", "size": "s", "registry": "open"},
    ])
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"packages": [
        {"ModelPackageArn": f"arn:aws:sagemaker:::model-package/{g}/{n}",
         "ModelApprovalStatus": "Approved"}
        for n in (1, 2, 3, 4, 5)
    ]})
    monkeypatch.setattr(t2, "fetch_metrics_and_lineage",
                        lambda arn: ({"mAP50": 0.1}, arn.rsplit("/", 1)[-1], None))

    parallel = t2.scan_all_model_metrics()

    import threading
    main = threading.current_thread()

    def throttle_the_pool(arn):
        if threading.current_thread() is not main:
            raise _throttle("DescribeModelPackage")
        return ({"mAP50": 0.1}, arn.rsplit("/", 1)[-1], None)

    monkeypatch.setattr(t2, "fetch_metrics_and_lineage", throttle_the_pool)
    serial = t2.scan_all_model_metrics()

    assert parallel == serial


def test_a_group_that_will_not_list_does_not_take_the_page_with_it(monkeypatch):
    """53 groups means 53 ListModelPackages calls per page load, and the
    live account throttled them. One group raising used to come straight
    out of the route as a 500, discarding every other group's rows."""
    monkeypatch.setattr(t2, "exportable_groups", lambda: [
        {"group_name": "yolo11s-good-detector", "size": "s", "registry": "open"},
        {"group_name": "yolo11m-throttled-detector", "size": "m", "registry": "open"},
    ])

    def list_or_throttle(group):
        if "throttled" in group:
            raise _throttle("ListModelPackages")
        return {"packages": [{
            "ModelPackageArn": f"arn:aws:sagemaker:::model-package/{group}/1",
            "ModelApprovalStatus": "Approved",
        }]}

    monkeypatch.setattr(t2, "model_registry_summary", list_or_throttle)
    monkeypatch.setattr(t2, "fetch_metrics_and_lineage", lambda arn: (None, "", None))

    rows = t2.scan_all_model_metrics()
    assert [r["group_name"] for r in rows] == ["yolo11s-good-detector"]


def test_a_version_is_described_once_and_then_remembered(monkeypatch):
    """Metrics and lineage are written at registration and never change,
    so re-describing them on every page load was pure API traffic -- and
    that traffic is what tripped the rate limiter."""
    t2._metrics_cache.clear()
    calls = []

    def describe(**kwargs):
        calls.append(kwargs["ModelPackageName"])
        return {"ModelPackageDescription": "first look"}

    monkeypatch.setattr(t2.sm, "describe_model_package", describe)

    arn = "arn:aws:sagemaker:::model-package/yolo11s-demo-detector/7"
    assert t2.fetch_metrics_and_lineage(arn)[1] == "first look"
    assert t2.fetch_metrics_and_lineage(arn)[1] == "first look"
    assert calls == [arn]
    t2._metrics_cache.clear()


def test_a_failed_describe_is_not_remembered(monkeypatch):
    """Caching a throttle would turn one bad moment into a row that stays
    blank until the container restarts."""
    t2._metrics_cache.clear()
    calls = []

    def describe(**kwargs):
        calls.append(kwargs["ModelPackageName"])
        if len(calls) == 1:
            raise _throttle("DescribeModelPackage")
        return {"ModelPackageDescription": "second time lucky"}

    monkeypatch.setattr(t2.sm, "describe_model_package", describe)

    arn = "arn:aws:sagemaker:::model-package/yolo11s-demo-detector/8"
    assert t2.fetch_metrics_and_lineage(arn) == (None, "", None)
    assert t2.fetch_metrics_and_lineage(arn)[1] == "second time lucky"
    assert len(calls) == 2
    t2._metrics_cache.clear()


def test_the_sagemaker_client_retries_adaptively():
    """The default `legacy` mode gives up after 4 attempts and raises,
    which is exactly how 53 serial list calls came to 500 the page."""
    config = t2.sm.meta.config.retries
    assert config["mode"] == "adaptive"
    # botocore normalises `max_attempts` (retries) to `total_max_attempts`
    # (tries), so 10 retries reads back as 11.
    assert config["total_max_attempts"] >= 11


def test_a_group_listing_is_cached_then_dropped_on_a_decision(monkeypatch):
    """53 groups per page load is 53 ListModelPackages calls to find 20
    rows -- 10 of the page's 19 seconds, and enough on its own to trip the
    rate limiter. Cached, but never across a change the operator just made.
    """
    t2.forget_registry_listing()
    t2._metrics_cache.clear()
    calls = []

    def list_packages(**kwargs):
        calls.append(kwargs["ModelPackageGroupName"])
        return {"ModelPackageSummaryList": [], "NextToken": ""}

    monkeypatch.setattr(t2.sm, "list_model_packages", list_packages)

    t2.model_registry_summary("yolo11s-demo-detector")
    t2.model_registry_summary("yolo11s-demo-detector")
    assert calls == ["yolo11s-demo-detector"], "the second read should be cached"

    t2.forget_registry_listing()
    t2.model_registry_summary("yolo11s-demo-detector")
    assert len(calls) == 2, "an approval must drop the cache"
    t2.forget_registry_listing()


def test_max_age_zero_always_asks_the_registry(monkeypatch):
    """So anything that must not read a cached answer can say so."""
    t2.forget_registry_listing()
    calls = []
    monkeypatch.setattr(t2.sm, "list_model_packages", lambda **k: (
        calls.append(k["ModelPackageGroupName"]),
        {"ModelPackageSummaryList": [], "NextToken": ""},
    )[1])

    t2.model_registry_summary("yolo11s-demo-detector")
    t2.model_registry_summary("yolo11s-demo-detector", max_age=0)
    assert len(calls) == 2
    t2.forget_registry_listing()


# --- the one destructive write that acted on an empty body --------------------
#
# The UAT sweep listed 60 write routes whose body has no required field.
# Checking them one by one, nearly all are fine: `weeks/purge-all` requires
# the exact object count and re-counts server-side, `export/weights`
# answers 400 on an empty group, the settings routes fall back to the
# default file name. `pull-control/run` was the exception -- an empty POST
# started a sweep, and a sweep MOVES files out of the drop folders with no
# second copy.

def test_starting_a_pull_needs_a_limit(client):
    """`POST {}` used to start a real sweep. The page always sends the
    field, so requiring it costs the caller nothing."""
    res = client.post("/track2/pull-control/run", json={})
    assert res.status_code == 422


def test_a_pull_request_that_names_its_limit_still_validates():
    """The guard must not have broken the ordinary path.

    The model, not the route: reaching the handler starts a sweep, and
    the suite's AWS guard rightly refuses to let a test do that. What
    needs proving here is the request CONTRACT, and that is the model.
    """
    import pydantic

    from ppe_pipeline_api.routers.track2_pull_control import RunRequest

    assert RunRequest(limit=10).limit == 10
    with pytest.raises(pydantic.ValidationError):
        RunRequest()


# --- the first load after a deploy ---------------------------------------------
#
# Review & Metrics measured 9-12s cold and effectively instant warm. Every
# deploy ends every session, so without a warm-up the first person back in
# always pays the cold price -- and twelve seconds reads as broken, not as
# busy.

def test_the_warm_up_fills_the_caches(monkeypatch):
    t2.forget_registry_listing()
    t2._metrics_cache.clear()
    monkeypatch.setattr(t2, "exportable_groups", lambda: [
        {"group_name": "yolo11s-demo-detector", "size": "s", "registry": "open"},
    ])
    monkeypatch.setattr(t2.sm, "list_model_packages", lambda **k: {
        "ModelPackageSummaryList": [], "NextToken": "",
    })

    out = t2.warm_registry_cache()
    assert out["warmed"] is True
    assert "yolo11s-demo-detector" in t2._listing_cache
    t2.forget_registry_listing()


def test_the_warm_up_never_takes_the_app_down(monkeypatch):
    """It runs at start-up. A start-up that fell over because a cache
    could not be pre-filled would be a worse trade than a slow page."""
    monkeypatch.setattr(t2, "scan_all_model_metrics", lambda: (_ for _ in ()).throw(
        _throttle("ListModelPackages")))

    out = t2.warm_registry_cache()
    assert out["warmed"] is False
    assert out["rows"] == 0


def test_the_cache_outlives_the_gap_between_startup_and_the_first_visitor():
    """The warm-up filled the cache and it aged out 30 seconds later, so
    the first person in still paid 7.6 seconds -- measured over HTTP
    against the warmed process. The TTL has to outlast that gap, and the
    refresher has to run inside it."""
    assert t2._LISTING_TTL >= 300
    assert t2._LISTING_REFRESH_SECONDS < t2._LISTING_TTL


def test_the_refresher_survives_a_failure(monkeypatch):
    """A cache that stops being refreshed degrades silently back into the
    slow page this exists to prevent, so the loop must not die on one bad
    pass."""
    import asyncio

    calls = []

    def flaky():
        calls.append(1)
        if len(calls) == 1:
            raise _throttle("ListModelPackages")
        return {"warmed": True, "rows": 0, "seconds": 0.0}

    monkeypatch.setattr(t2, "warm_registry_cache", flaky)
    monkeypatch.setattr(t2, "_LISTING_REFRESH_SECONDS", 0.01)

    async def run_briefly():
        task = asyncio.create_task(t2.registry_warm_loop())
        await asyncio.sleep(0.15)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(run_briefly())
    assert len(calls) >= 2, "it should have carried on after the throttle"


def test_approving_a_version_the_registry_refuses_is_a_400_not_a_500(client, monkeypatch):
    """`delete_model` always caught ClientError; approve and reject did
    not, so any of them -- AccessDenied, a throttle, a version deleted in
    another tab -- came out as a 500 with a traceback behind it."""
    from ppe_pipeline_api.routers import track2_review as router

    def refuse(**kwargs):
        raise ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "nope"}},
            "UpdateModelPackage",
        )

    monkeypatch.setattr(router.t2.sm, "update_model_package", refuse)
    arn = "arn:aws:sagemaker:::model-package/yolo11s-demo-detector/1"
    for verb in ("approve", "reject"):
        res = client.post(f"/track2/review/model/{arn}/{verb}")
        assert res.status_code == 400, verb
        assert "Could not mark it" in res.json()["detail"]
