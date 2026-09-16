"""Integration tests for the /track2/eval router -- queue add/remove/
start/stop against a fake DynamoDB table. The IoU/AP math itself is
covered separately in test_track2_eval_metrics.py; this file is about the
queue/HTTP plumbing around it."""

from __future__ import annotations

import pytest
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


class FakeEvalTable:
    """Generic-enough SET-only update_item to support whatever attributes
    track2_shared.py's eval-run functions happen to write, unlike the
    single-purpose FakeTable in test_track2_api.py (hardcoded to the
    review-decision status/reviewed_at pair only)."""

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
            attr = attr.strip()
            attr = names.get(attr, attr)
            item[attr] = values[value_placeholder.strip()]

    def delete_item(self, Key):
        self.items.pop(Key["id"], None)


@pytest.fixture
def fake_eval_table(monkeypatch):
    fake = FakeEvalTable()
    monkeypatch.setattr(t2, "eval_runs_table", lambda: fake)
    return fake


def test_eval_endpoints_require_auth():
    c = TestClient(app)
    assert c.get("/track2/eval").status_code == 401


def test_get_eval_empty(client, fake_eval_table, monkeypatch):
    monkeypatch.setattr(t2, "all_model_registries", lambda: {})
    body = client.get("/track2/eval").json()
    assert body["runs"] == []
    assert body["queue"] == {"started": False, "worker_alive": False}


def test_add_registry_run(client, fake_eval_table):
    payload = {
        "size": "s",
        "data_folder": "eval-datasets/my-set",
        "weights_kind": "registry",
        "model_package_arn": "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-detector/3",
    }
    response = client.post("/track2/eval/add", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["run"]["status"] == "queued"
    assert body["run"]["weights_label"] == "YOLO11-S v3"
    assert body["run"]["data_folder"] == "eval-datasets/my-set"


def test_add_s3_run(client, fake_eval_table):
    payload = {
        "size": "m",
        "data_folder": "eval-datasets/my-set",
        "weights_kind": "s3",
        "s3_folder": "weights/m/2026-09-09",
    }
    response = client.post("/track2/eval/add", json=payload)
    assert response.status_code == 200
    assert response.json()["run"]["weights_label"] == "S3: weights/m/2026-09-09"


def test_add_run_missing_weights_source_is_422(client, fake_eval_table):
    payload = {"size": "s", "data_folder": "eval-datasets/my-set", "weights_kind": "registry"}
    response = client.post("/track2/eval/add", json=payload)
    assert response.status_code == 422


def test_add_run_unknown_size_is_404(client, fake_eval_table):
    payload = {
        "size": "xl", "data_folder": "eval-datasets/my-set", "weights_kind": "s3", "s3_folder": "x",
    }
    assert client.post("/track2/eval/add", json=payload).status_code == 404


def test_remove_queued_run(client, fake_eval_table, monkeypatch):
    monkeypatch.setattr(t2, "all_model_registries", lambda: {})
    added = client.post("/track2/eval/add", json={
        "size": "s", "data_folder": "data", "weights_kind": "s3", "s3_folder": "x",
    }).json()["run"]
    response = client.post(f"/track2/eval/remove/{added['id']}")
    assert response.json() == {"removed": True}
    assert client.get("/track2/eval").json()["runs"] == []


def test_remove_nonexistent_run_is_404(client, fake_eval_table):
    assert client.post("/track2/eval/remove/does-not-exist").status_code == 404


def test_start_with_nothing_queued_is_400(client, fake_eval_table):
    assert client.post("/track2/eval/start").status_code == 400


def test_stop_always_succeeds(client, fake_eval_table):
    assert client.post("/track2/eval/stop").status_code == 200
