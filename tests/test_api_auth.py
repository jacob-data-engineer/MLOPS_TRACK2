"""Tests for `ppe_pipeline_api.auth` — the HTTP login gate and its
`_password_ok`: fails closed, `hmac`-compared, and every other route
depends on `require_session` validating the issued token."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from ppe_pipeline_api import auth as auth_module
from ppe_pipeline_api.main import app


@pytest.fixture(autouse=True)
def _reset_failed_attempts():
    """`_failed_attempts` is module-level global state keyed by client
    host — every `TestClient()` instance in this file shares the same
    host, so each test must start from a clean slate rather than
    inheriting an earlier test's failed-attempt count."""
    auth_module._failed_attempts.clear()
    yield
    auth_module._failed_attempts.clear()


def _client() -> TestClient:
    return TestClient(app)


def test_login_fails_closed_when_password_unset(monkeypatch):
    monkeypatch.delenv("PPE_APP_PASSWORD", raising=False)
    response = _client().post("/auth/login", json={"password": "anything"})
    assert response.status_code == 503


def test_login_rejects_wrong_password(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().post("/auth/login", json={"password": "wrong"})
    assert response.status_code == 401


def test_login_accepts_correct_password_and_issues_token(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().post("/auth/login", json={"password": "correct-horse"})
    assert response.status_code == 200
    token = response.json()["token"]
    assert isinstance(token, str) and token


def test_login_non_ascii_password_round_trips(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "café123")
    response = _client().post("/auth/login", json={"password": "café123"})
    assert response.status_code == 200


def test_login_locks_out_after_max_attempts(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    client = _client()
    for _ in range(auth_module.MAX_LOGIN_ATTEMPTS):
        response = client.post("/auth/login", json={"password": "wrong"})
        assert response.status_code == 401

    locked_out = client.post("/auth/login", json={"password": "correct-horse"})
    assert locked_out.status_code == 429


def test_protected_route_rejects_missing_authorization_header(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().get("/track1/hazard-types")
    assert response.status_code == 401


def test_protected_route_rejects_invalid_token(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    response = _client().get("/track1/hazard-types", headers={"Authorization": "Bearer not-a-real-token"})
    assert response.status_code == 401


def test_protected_route_accepts_valid_token(monkeypatch):
    monkeypatch.setenv("PPE_APP_PASSWORD", "correct-horse")
    client = _client()
    token = client.post("/auth/login", json={"password": "correct-horse"}).json()["token"]
    response = client.get("/track1/hazard-types", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200


def test_health_check_does_not_require_auth():
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
