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

    def subscribe(self, TopicArn, Protocol, Endpoint):
        self.subscriptions.append({
            "SubscriptionArn": "PendingConfirmation", "Endpoint": Endpoint, "Protocol": Protocol,
        })

    def unsubscribe(self, SubscriptionArn):
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

    assert client.get("/track2/notifications").json() == {"subscriptions": []}

    add = client.post("/track2/notifications/add", json={"email": "a@example.com"})
    assert add.status_code == 200
    listing = client.get("/track2/notifications").json()["subscriptions"]
    assert listing == [{"subscription_arn": "PendingConfirmation", "email": "a@example.com", "confirmed": False}]

    remove = client.post("/track2/notifications/remove", json={"subscription_arn": "PendingConfirmation"})
    assert remove.json() == {"removed": True}
    assert client.get("/track2/notifications").json() == {"subscriptions": []}


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
    monkeypatch.setattr(t2, "scan_automated_queues", lambda: [])
    monkeypatch.setattr(t2, "get_training_progress", lambda: None)

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
    monkeypatch.setattr(t2, "get_training_progress", lambda: None)
    response = client.post("/track2/schedule/train-now")
    assert response.status_code == 409


# --- ad-hoc queue ----------------------------------------------------------------

def test_queue_add_remove_start_stop(client, monkeypatch):
    monkeypatch.setattr(t2, "_training_queue", [], raising=False)
    monkeypatch.setattr(t2, "_queue_running", None, raising=False)
    monkeypatch.setattr(t2, "_queue_started", False, raising=False)
    monkeypatch.setattr(t2, "training_already_running_reason", lambda progress: "busy, stay queued")
    monkeypatch.setattr(t2, "get_training_progress", lambda: None)
    # GET /track2/models also fetches registries/S3 browse/GPU usage/
    # automated queues -- none of that is what this test is about, so it's
    # all stubbed out rather than hitting real AWS.
    monkeypatch.setattr(t2, "all_model_registries", lambda: {})
    monkeypatch.setattr(t2, "list_s3_folders", lambda prefix: [])
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    monkeypatch.setattr(t2, "scan_automated_queues", lambda: [])

    add = client.post("/track2/models/queue/add", json={
        "generation": "11", "size": "s", "dataset_folder": "data/x",
    })
    assert add.status_code == 200
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
