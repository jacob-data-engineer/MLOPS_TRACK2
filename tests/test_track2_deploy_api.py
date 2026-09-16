"""Integration tests for /track2/deploy -- a single shared active
Inference Component (not one per size). Deploying only happens through
the inference queue (deploy -> wait -> run_batch_inference -> next),
no separate one-shot deploy endpoint. Mocks deploy_inference_component/
describe_deployed_inference_component/run_batch_inference directly
rather than exercising real boto3 calls."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient

from ppe_pipeline_api import track2_shared as t2
from ppe_pipeline_api.main import app

_PASSWORD = "test-suite-password"

# Captured before the autouse fixture below stubs the name out, so the
# permission-failure test can exercise the real implementation.
_REAL_ENDPOINT_STATUS = t2.endpoint_status


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("PPE_APP_PASSWORD", _PASSWORD)
    c = TestClient(app)
    token = c.post("/auth/login", json={"password": _PASSWORD}).json()["token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


@pytest.fixture(autouse=True)
def _stub_registries_and_folders(monkeypatch):
    monkeypatch.setattr(t2, "all_model_registries", lambda: {})
    monkeypatch.setattr(t2, "list_s3_folders", lambda prefix: [])
    monkeypatch.setattr(t2, "list_s3_weight_files", lambda prefix: [])
    # Tests that care about a specific deployed-component shape override
    # this themselves -- this default just keeps every other GET (e.g.
    # after an add/remove) from hitting real AWS with no credentials.
    monkeypatch.setattr(t2, "describe_deployed_inference_component", lambda: None)
    # Same reasoning: GET /track2/deploy reports endpoint state on every
    # call, and the endpoint is created/deleted on demand now, so "not
    # there" is the ordinary resting state to default to here.
    monkeypatch.setattr(t2, "endpoint_status", lambda: None)
    # Endpoint teardown refuses while EITHER queue still has work, so
    # delete_endpoint reads the eval queue too -- a real DynamoDB scan that
    # this file otherwise never stubs (only the inference queue table is
    # faked below). It passed locally purely because a developer machine
    # has AWS credentials, and failed in CI, which has none.
    monkeypatch.setattr(t2, "list_eval_runs", lambda: [])


class FakeInferenceQueueTable:
    """Generic-enough SET-only update_item to support whatever attributes
    track2_shared.py's inference-queue functions happen to write -- same
    shape as test_track2_eval_api.py's FakeEvalTable, since
    _update_inference_queue_item uses the identical dynamic
    UpdateExpression pattern."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def scan(self, **kwargs):
        return {"Items": list(self.items.values())}

    def get_item(self, Key):
        item = self.items.get(Key["id"])
        return {"Item": item} if item else {}

    def put_item(self, Item):
        self.items[Item["id"]] = Item

    def update_item(self, Key, UpdateExpression, ExpressionAttributeNames=None, ExpressionAttributeValues=None):
        item = self.items.setdefault(Key["id"], dict(Key))
        names = ExpressionAttributeNames or {}
        values = ExpressionAttributeValues or {}
        assignments = UpdateExpression.removeprefix("SET").strip()
        for assignment in assignments.split(","):
            attr, _, value_placeholder = assignment.strip().partition("=")
            attr = names.get(attr.strip(), attr.strip())
            item[attr] = values[value_placeholder.strip()]

    def delete_item(self, Key):
        self.items.pop(Key["id"], None)


@pytest.fixture(autouse=True)
def _reset_inference_queue(monkeypatch):
    fake = FakeInferenceQueueTable()
    monkeypatch.setattr(t2, "inference_queue_table", lambda: fake)
    monkeypatch.setattr(t2, "_inference_queue_started", False, raising=False)
    monkeypatch.setattr(t2, "_inference_worker_thread", None, raising=False)
    return fake


def test_deploy_endpoints_require_auth():
    c = TestClient(app)
    assert c.get("/track2/deploy").status_code == 401


def test_get_deploy_nothing_deployed_yet(client, monkeypatch):
    monkeypatch.setattr(t2, "describe_deployed_inference_component", lambda: None)
    body = client.get("/track2/deploy").json()
    assert body["deployed"] is None
    assert body["folders"] == []
    assert body["weight_files"] == []
    assert body["inference_queue"] == []
    assert body["inference_queue_status"] == {"started": False, "worker_alive": False}


def test_get_deploy_shows_active_model(client, monkeypatch):
    monkeypatch.setattr(t2, "describe_deployed_inference_component", lambda: {
        "active_size": "s", "status": "InService", "model_name": "track2-yolo-s-model-x",
        "last_modified": "2026-09-10T00:00:00+00:00", "source": "YOLO11-S v3", "version": "3",
        "current_copy_count": 1, "desired_copy_count": 1, "running_inference": False,
    })
    body = client.get("/track2/deploy").json()
    assert body["deployed"]["active_size"] == "s"
    assert body["deployed"]["source"] == "YOLO11-S v3"
    assert body["deployed"]["version"] == "3"


def test_get_deploy_reports_endpoint_absent(client):
    """A torn-down endpoint is the resting state between runs, not an
    error -- the UI reads this to say "not running, not billing"."""
    assert client.get("/track2/deploy").json()["endpoint"] is None


def test_get_deploy_reports_endpoint_running(client, monkeypatch):
    monkeypatch.setattr(t2, "endpoint_status", lambda: {
        "status": "InService", "instance_type": "ml.m5.2xlarge",
        "instance_count": 1, "created_at": "2026-09-15T00:00:00+00:00",
    })
    endpoint = client.get("/track2/deploy").json()["endpoint"]
    assert endpoint["status"] == "InService"
    assert endpoint["instance_type"] == "ml.m5.2xlarge"


def test_endpoint_status_reports_unknown_when_permission_is_missing(monkeypatch):
    """endpoint_status runs on every GET /track2/deploy, so an unhandled
    AccessDenied would 500 the whole page. It must also not report None,
    which the UI renders as "not running, nothing being billed" -- the one
    answer that must never be guessed about an hourly-billed resource."""
    import botocore.exceptions

    def _denied(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "not authorized: sagemaker:DescribeEndpoint"}},
            "DescribeEndpoint",
        )

    monkeypatch.setattr(t2.sm, "describe_endpoint", _denied)
    status = _REAL_ENDPOINT_STATUS()
    assert status is not None, "must not read as 'torn down, costing nothing'"
    assert status["status"] == "Unknown"
    assert status["instance_count"] is None


def test_endpoint_status_reports_none_when_genuinely_absent(monkeypatch):
    import botocore.exceptions

    def _not_found(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find endpoint"}},
            "DescribeEndpoint",
        )

    monkeypatch.setattr(t2.sm, "describe_endpoint", _not_found)
    assert _REAL_ENDPOINT_STATUS() is None


def test_delete_endpoint_requires_auth():
    c = TestClient(app)
    assert c.post("/track2/deploy/endpoint/delete").status_code == 401


def test_delete_endpoint_when_idle(client, monkeypatch):
    """The component MUST be deleted before the endpoint: SageMaker rejects
    DeleteEndpoint outright while one is still attached, so getting this
    order wrong fails every teardown -- the button and the automatic
    on-drain one alike."""
    calls = []
    monkeypatch.setattr(t2, "endpoint_status", lambda: {
        "status": "InService", "instance_type": "ml.m5.2xlarge", "instance_count": 1, "created_at": None,
    })
    monkeypatch.setattr(t2, "delete_active_inference_component", lambda: calls.append("component") or True)
    monkeypatch.setattr(t2.sm, "delete_endpoint", lambda **kw: calls.append(f"endpoint:{kw['EndpointName']}"))

    body = client.post("/track2/deploy/endpoint/delete").json()
    assert body == {"deleted": True, "reason": None}
    assert calls == ["component", f"endpoint:{t2.MULTI_SIZE_ENDPOINT_NAME}"]


def test_delete_active_inference_component_waits_for_it_to_go(monkeypatch):
    """Deletion is async. Returning as soon as the API accepts the call
    loses the race with DeleteEndpoint, which is the same failure as not
    deleting the component at all."""
    states = iter([{"status": "Deleting"}, {"status": "Deleting"}, None])
    monkeypatch.setattr(t2.sm, "delete_inference_component", lambda **kw: None)
    monkeypatch.setattr(t2, "describe_deployed_inference_component", lambda: next(states))
    monkeypatch.setattr(t2.time, "sleep", lambda _s: None)

    assert t2.delete_active_inference_component() is True


def test_delete_active_inference_component_tolerates_a_missing_one(monkeypatch):
    import botocore.exceptions

    def _gone(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "ValidationException", "Message": "Could not find inference component"}},
            "DeleteInferenceComponent",
        )

    monkeypatch.setattr(t2.sm, "delete_inference_component", _gone)
    assert t2.delete_active_inference_component() is False


def test_delete_endpoint_refuses_while_a_queue_runs(client, monkeypatch):
    """Deleting mid-run kills every in-flight invocation, so a running
    queue has to block teardown -- both queues share this one endpoint."""
    monkeypatch.setattr(t2, "endpoint_status", lambda: {
        "status": "InService", "instance_type": "ml.m5.2xlarge", "instance_count": 1, "created_at": None,
    })
    monkeypatch.setattr(t2, "_inference_queue_started", True, raising=False)

    def _must_not_delete(**kw):
        raise AssertionError("delete_endpoint called while a queue was running")

    monkeypatch.setattr(t2.sm, "delete_endpoint", _must_not_delete)

    body = client.post("/track2/deploy/endpoint/delete").json()
    assert body["deleted"] is False
    assert "queue" in body["reason"]


def test_delete_endpoint_refuses_when_already_gone(client, monkeypatch):
    monkeypatch.setattr(t2, "endpoint_status", lambda: None)
    body = client.post("/track2/deploy/endpoint/delete").json()
    assert body["deleted"] is False
    assert "doesn't exist" in body["reason"]


def test_ensure_endpoint_exists_creates_when_absent(monkeypatch):
    """A deploy against a torn-down endpoint has to bring it back first --
    an Inference Component can't be created on an endpoint that isn't up."""
    calls = []
    states = iter([None, {"status": "Creating"}, {"status": "InService"}])
    monkeypatch.setattr(t2, "endpoint_status", lambda: next(states))
    # Nothing kept from a previous teardown -- the genuinely-first-ever run.
    monkeypatch.setattr(t2, "endpoint_config_exists", lambda: False)
    monkeypatch.setattr(t2.sm, "create_endpoint_config", lambda **kw: calls.append(("config", kw)))
    monkeypatch.setattr(t2.sm, "create_endpoint", lambda **kw: calls.append(("endpoint", kw)))
    monkeypatch.setattr(t2.time, "sleep", lambda _s: None)

    assert t2.ensure_endpoint_exists() is True
    assert [name for name, _ in calls] == ["config", "endpoint"]
    assert calls[1][1]["EndpointName"] == t2.MULTI_SIZE_ENDPOINT_NAME
    assert calls[0][1]["ProductionVariants"][0]["InstanceType"] == t2.MULTI_SIZE_INSTANCE_TYPE


def test_ready_wait_uses_one_describe_per_poll(monkeypatch):
    """The readiness poll must not drag the model-tag lookups along: that
    turns one call per poll into three, doubled again because a background
    thread waits on the same component, all at a 5s cadence."""
    calls = []
    monkeypatch.setattr(t2.sm, "describe_inference_component", lambda **kw: calls.append("ic") or {
        "InferenceComponentStatus": "InService", "RuntimeConfig": {"CurrentCopyCount": 1},
    })

    def _must_not_be_called(**kwargs):
        raise AssertionError("readiness poll fetched model tags")

    monkeypatch.setattr(t2.sm, "describe_model", _must_not_be_called)
    monkeypatch.setattr(t2.sm, "list_tags", _must_not_be_called)

    t2.wait_for_inference_component_ready(timeout_seconds=30, poll_seconds=1)
    assert calls == ["ic"]


def test_ready_wait_timeout_says_what_it_last_saw(monkeypatch):
    """The bare "Timed out waiting..." message gave no way to tell a
    component stuck at zero copies from one that never appeared, which is
    exactly the distinction needed when this fires."""
    monkeypatch.setattr(t2.sm, "describe_inference_component", lambda **kw: {
        "InferenceComponentStatus": "Creating", "RuntimeConfig": {"CurrentCopyCount": 0},
    })
    monkeypatch.setattr(t2.time, "sleep", lambda _s: None)

    with pytest.raises(RuntimeError) as err:
        t2.wait_for_inference_component_ready(timeout_seconds=0.01, poll_seconds=0)
    assert "Creating" in str(err.value)
    assert "running copies=0" in str(err.value)


def test_ensure_scale_to_zero_skips_a_missing_component(monkeypatch):
    """The scalable target is registered AGAINST the component, so once
    teardown deletes the component there is nothing to register. This runs
    from a background thread minutes after a deploy, by which time a
    teardown may well have happened -- it must not raise."""
    import botocore.exceptions

    def _no_component(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "ValidationException", "Message": (
                "Validation failed for resource: inference-component/track2-yolo-active, "
                'scalable dimension: sagemaker:inference-component:DesiredCopyCount. '
                'Reason: Could not find inference component '
                '"arn:aws:sagemaker:ap-southeast-1:125192671932:inference-component/track2-yolo-active".'
            )}},
            "RegisterScalableTarget",
        )

    monkeypatch.setattr(t2.autoscaling, "register_scalable_target", _no_component)

    def _must_not_reach(**kwargs):
        raise AssertionError("kept going after the component was found missing")

    monkeypatch.setattr(t2.autoscaling, "put_scaling_policy", _must_not_reach)
    t2.ensure_scale_to_zero()  # must not raise


def test_ensure_endpoint_exists_reuses_the_kept_config(monkeypatch):
    """Teardown keeps the endpoint config on purpose, so the spin-up after
    it must NOT try to recreate it. Getting this wrong failed the first
    queued job after every teardown -- the exact path the whole on-demand
    design depends on."""
    calls = []
    states = iter([None, {"status": "InService"}])
    monkeypatch.setattr(t2, "endpoint_status", lambda: next(states))
    monkeypatch.setattr(t2.sm, "describe_endpoint_config", lambda **kw: calls.append("described") or {})

    def _must_not_create(**kwargs):
        raise AssertionError("tried to recreate an endpoint config that already exists")

    monkeypatch.setattr(t2.sm, "create_endpoint_config", _must_not_create)
    monkeypatch.setattr(t2.sm, "create_endpoint", lambda **kw: calls.append("endpoint"))
    monkeypatch.setattr(t2.time, "sleep", lambda _s: None)

    assert t2.ensure_endpoint_exists() is True
    assert calls == ["described", "endpoint"]


def test_ensure_endpoint_exists_tolerates_the_real_already_exists_message(monkeypatch):
    """AWS says "Cannot create already EXISTING endpoint configuration".
    A guard testing for "already exists" does not match that string, which
    is how this surfaced: a ValidationException escaped and failed the job."""
    import botocore.exceptions

    states = iter([None, {"status": "InService"}])
    monkeypatch.setattr(t2, "endpoint_status", lambda: next(states))
    # Config reads as absent, then creation reports it exists -- the race
    # the except clause is there to absorb.
    monkeypatch.setattr(t2, "endpoint_config_exists", lambda: False)

    def _already_existing(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "ValidationException", "Message": (
                "Cannot create already existing endpoint configuration "
                '"arn:aws:sagemaker:ap-southeast-1:125192671932:endpoint-config/track2-yolo-multi-size-config".'
            )}},
            "CreateEndpointConfig",
        )

    monkeypatch.setattr(t2.sm, "create_endpoint_config", _already_existing)
    created = []
    monkeypatch.setattr(t2.sm, "create_endpoint", lambda **kw: created.append(kw["EndpointName"]))
    monkeypatch.setattr(t2.time, "sleep", lambda _s: None)

    assert t2.ensure_endpoint_exists() is True
    assert created == [t2.MULTI_SIZE_ENDPOINT_NAME]


def test_ensure_endpoint_exists_is_a_noop_when_already_in_service(monkeypatch):
    monkeypatch.setattr(t2, "endpoint_status", lambda: {"status": "InService"})

    def _must_not_create(**kw):
        raise AssertionError("created an endpoint that was already InService")

    monkeypatch.setattr(t2.sm, "create_endpoint", _must_not_create)
    assert t2.ensure_endpoint_exists() is False


def test_lineage_requires_auth():
    c = TestClient(app)
    assert c.get("/track2/deploy/lineage", params={"output_folder": "results/x"}).status_code == 401


def test_lineage_returns_what_produced_the_output(client, monkeypatch):
    record = {
        "bucket": "track2-s3",
        "images_folder": "inference_test/images",
        "output_folder": "results/2026-09-15",
        "files": ["a.jpg", "b.jpg"],
        "model": {"size": "s", "weights_kind": "registry", "version": "12"},
        "inference_started_at": "2026-09-15T07:44:26+00:00",
        "inference_completed_at": "2026-09-15T07:50:29+00:00",
        "generated_at": "2026-09-15T07:50:29+00:00",
    }
    monkeypatch.setattr(t2, "read_run_lineage", lambda run_id, folder: record if folder == "results/2026-09-15" else None)

    body = client.get("/track2/deploy/lineage", params={"output_folder": "results/2026-09-15"}).json()
    assert body["files"] == ["a.jpg", "b.jpg"]
    assert body["images_folder"] == "inference_test/images"
    assert body["model"]["size"] == "s"
    assert body["model"]["version"] == "12"


def test_lineage_is_404_when_the_run_predates_it(client, monkeypatch):
    """Runs from before lineage was written, and jobs that died before
    inference started, have no lineage.json. That is a plain absence, not
    an error worth a 500."""
    monkeypatch.setattr(t2, "read_run_lineage", lambda run_id, folder: None)
    assert client.get("/track2/deploy/lineage", params={"output_folder": "results/old"}).status_code == 404


class _FakeLineageTable:
    def __init__(self, item=None, raises=None):
        self.item, self.raises, self.keys = item, raises, []

    def get_item(self, Key):
        self.keys.append(Key)
        if self.raises:
            raise self.raises
        return {"Item": self.item} if self.item else {}

    def put_item(self, Item):
        self.item = Item


def test_read_run_lineage_prefers_the_table(monkeypatch):
    """Keyed by run id, so two runs writing to the same output folder stay
    distinguishable -- which keying on the folder would silently collapse."""
    fake = _FakeLineageTable(item={"id": "run-1", "files": ["a.jpg"], "output_folder": "results/x"})
    monkeypatch.setattr(t2, "inference_lineage_table", lambda: fake)

    def _s3_must_not_be_used(**kwargs):
        raise AssertionError("went to S3 despite the record being in DynamoDB")

    monkeypatch.setattr(t2.s3, "get_object", _s3_must_not_be_used)
    assert t2.read_run_lineage("run-1", "results/x")["files"] == ["a.jpg"]
    assert fake.keys == [{"id": "run-1"}]


def test_read_run_lineage_falls_back_to_s3(monkeypatch):
    """Runs predating the table have no row, and output produced outside
    the queue has a folder but no run id at all."""
    monkeypatch.setattr(t2, "inference_lineage_table", lambda: _FakeLineageTable())
    monkeypatch.setattr(t2.s3, "get_object", lambda **kw: {
        "Body": io.BytesIO(json.dumps({"files": ["from-s3.jpg"]}).encode()),
    })
    assert t2.read_run_lineage("run-missing", "results/x")["files"] == ["from-s3.jpg"]


def test_read_run_lineage_falls_back_when_the_table_errors(monkeypatch):
    """A table problem should degrade to the slower answer, not to none:
    the folder's own copy is just as authoritative."""
    monkeypatch.setattr(t2, "inference_lineage_table", lambda: _FakeLineageTable(raises=RuntimeError("ddb down")))
    monkeypatch.setattr(t2.s3, "get_object", lambda **kw: {
        "Body": io.BytesIO(json.dumps({"files": ["from-s3.jpg"]}).encode()),
    })
    assert t2.read_run_lineage("run-1", "results/x")["files"] == ["from-s3.jpg"]


def test_read_run_lineage_none_when_key_absent(monkeypatch):
    import botocore.exceptions

    monkeypatch.setattr(t2, "inference_lineage_table", lambda: _FakeLineageTable())

    def _missing(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "NoSuchKey", "Message": "The specified key does not exist."}}, "GetObject",
        )

    monkeypatch.setattr(t2.s3, "get_object", _missing)
    assert t2.read_run_lineage(None, "results/whatever") is None


def test_read_run_lineage_does_not_swallow_access_denied(monkeypatch):
    """A permissions problem answered as "no lineage" would read as a
    settled fact about the run rather than a fault to fix."""
    import botocore.exceptions

    monkeypatch.setattr(t2, "inference_lineage_table", lambda: _FakeLineageTable())

    def _denied(**kwargs):
        raise botocore.exceptions.ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}}, "GetObject",
        )

    monkeypatch.setattr(t2.s3, "get_object", _denied)
    with pytest.raises(botocore.exceptions.ClientError):
        t2.read_run_lineage(None, "results/whatever")


def test_browse_requires_auth():
    c = TestClient(app)
    assert c.get("/track2/deploy/browse").status_code == 401


def test_browse_returns_folders_and_weight_files(client, monkeypatch):
    seen = {}

    def _fake(prefix):
        seen["prefix"] = prefix
        return (["datasets", "runs"], ["model.pt"])

    monkeypatch.setattr(t2, "list_s3_folders_and_weight_files", _fake)
    body = client.get("/track2/deploy/browse", params={"prefix": "weights/yolo11s"}).json()
    assert body == {"folders": ["datasets", "runs"], "weight_files": ["model.pt"]}
    assert seen["prefix"] == "weights/yolo11s"


def test_browse_does_no_registry_or_endpoint_work(client, monkeypatch):
    """The whole reason this route exists separately from GET
    /track2/deploy: a folder click must not pay for the registry summary
    or the live-component describe (~30 serial AWS calls between them),
    which is what made navigating take over a second per click."""
    def _boom(*args, **kwargs):
        raise AssertionError("browse must not touch registry/endpoint state")

    monkeypatch.setattr(t2, "serialize_registries_brief", _boom)
    monkeypatch.setattr(t2, "all_model_registries", _boom)
    monkeypatch.setattr(t2, "describe_deployed_inference_component", _boom)
    monkeypatch.setattr(t2, "list_s3_folders_and_weight_files", lambda prefix: ([], []))

    assert client.get("/track2/deploy/browse").status_code == 200


def test_list_s3_folders_and_weight_files_makes_one_s3_call(monkeypatch):
    calls = []

    def _fake_list(**kwargs):
        calls.append(kwargs)
        return {
            "CommonPrefixes": [{"Prefix": "weights/run1/"}, {"Prefix": "weights/run2/"}],
            "Contents": [
                {"Key": "weights/"},
                {"Key": "weights/model.pt"},
                {"Key": "weights/model.tar.gz"},
                {"Key": "weights/notes.txt"},
            ],
        }

    monkeypatch.setattr(t2.s3, "list_objects_v2", _fake_list)
    folders, weight_files = t2.list_s3_folders_and_weight_files("weights")

    assert len(calls) == 1
    assert calls[0]["Prefix"] == "weights/"
    assert calls[0]["Delimiter"] == "/"
    assert folders == ["run1", "run2"]
    assert weight_files == ["model.pt", "model.tar.gz"]


def test_add_inference_queue_item_registry(client):
    payload = {
        "size": "s", "weights_kind": "registry",
        "model_package_arn": "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-detector/3",
        "images_folder": "eval-datasets/x/images", "output_folder": "results/2026-09-10",
    }
    response = client.post("/track2/deploy/inference-queue/add", json=payload)
    assert response.status_code == 200
    item = response.json()["item"]
    assert item["status"] == "queued"
    assert item["weights_label"] == "YOLO11-S v3"


def test_add_while_running_keeps_the_worker_alive(client, monkeypatch):
    """A job added behind a running one must be reached without anyone
    pressing Start. The worker loop exits the moment it finds nothing
    queued, which it can do between the running job finishing and this add
    landing -- and that drain also tears the endpoint down, so the orphan
    would pay a full cold start for a queue that never went idle."""
    started = []
    monkeypatch.setattr(t2, "inference_queue_snapshot", lambda: {"started": True, "worker_alive": False})
    monkeypatch.setattr(t2, "start_inference_queue", lambda: started.append(True) or True)

    payload = {
        "size": "s", "weights_kind": "registry",
        "model_package_arn": "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-detector/3",
        "images_folder": "images/x", "output_folder": "results/x",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 200
    assert started == [True]


def test_add_while_idle_does_not_start_the_queue(client, monkeypatch):
    """Adding while nothing is running still just queues: several jobs get
    lined up and started deliberately, rather than the first add
    committing the endpoint."""
    monkeypatch.setattr(t2, "inference_queue_snapshot", lambda: {"started": False, "worker_alive": False})

    def _must_not_start():
        raise AssertionError("an add on an idle queue started it")

    monkeypatch.setattr(t2, "start_inference_queue", _must_not_start)

    payload = {
        "size": "s", "weights_kind": "registry",
        "model_package_arn": "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-detector/3",
        "images_folder": "images/x", "output_folder": "results/x",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 200


def test_add_inference_queue_item_s3_file(client):
    payload = {
        "size": "m", "weights_kind": "s3", "weights_key": "candidates/x/my_model.pt",
        "images_folder": "images/x", "output_folder": "results/x",
    }
    response = client.post("/track2/deploy/inference-queue/add", json=payload)
    assert response.status_code == 200
    assert response.json()["item"]["weights_label"] == "S3: candidates/x/my_model.pt"


def test_add_inference_queue_item_missing_images_folder_is_422(client):
    payload = {
        "size": "s", "weights_kind": "registry", "model_package_arn": "x",
        "images_folder": "", "output_folder": "results/x",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 422


def test_add_inference_queue_item_missing_weights_source_is_422(client):
    payload = {
        "size": "s", "weights_kind": "registry",
        "images_folder": "images/x", "output_folder": "results/x",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 422


def test_add_inference_queue_item_s3_neither_folder_nor_key_is_422(client):
    payload = {
        "size": "s", "weights_kind": "s3",
        "images_folder": "images/x", "output_folder": "results/x",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 422


def test_add_inference_queue_item_unknown_size_is_422(client):
    payload = {
        "size": "xl", "weights_kind": "s3", "weights_key": "x/model.pt",
        "images_folder": "images/x", "output_folder": "results/x",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 422


def test_inference_queue_caps_at_ten_pending(client):
    for i in range(10):
        payload = {
            "size": "s", "weights_kind": "s3", "weights_key": f"x/model{i}.pt",
            "images_folder": "images/x", "output_folder": f"results/{i}",
        }
        assert client.post("/track2/deploy/inference-queue/add", json=payload).status_code == 200
    overflow = {
        "size": "s", "weights_kind": "s3", "weights_key": "x/model10.pt",
        "images_folder": "images/x", "output_folder": "results/10",
    }
    assert client.post("/track2/deploy/inference-queue/add", json=overflow).status_code == 422


def test_remove_queued_inference_item(client):
    added = client.post("/track2/deploy/inference-queue/add", json={
        "size": "s", "weights_kind": "s3", "weights_key": "x/model.pt",
        "images_folder": "images/x", "output_folder": "results/x",
    }).json()["item"]
    response = client.post(f"/track2/deploy/inference-queue/remove/{added['id']}")
    assert response.json() == {"removed": True}
    assert client.get("/track2/deploy").json()["inference_queue"] == []


def test_remove_nonexistent_inference_item_is_404(client):
    assert client.post("/track2/deploy/inference-queue/remove/does-not-exist").status_code == 404


def test_start_inference_queue_with_nothing_queued_is_400(client):
    assert client.post("/track2/deploy/inference-queue/start").status_code == 400


def test_stop_inference_queue_always_succeeds(client):
    assert client.post("/track2/deploy/inference-queue/stop").status_code == 200
