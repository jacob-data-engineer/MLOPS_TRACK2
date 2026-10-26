"""A blank cut-off must not silently disarm an armed schedule.

The schedule is counted FROM the week cut-off -- `set_mode` refuses
schedule mode without one. So a config write that clears it while a
schedule is armed leaves an ENABLED rule at AWS with nothing to count
from: the next run simply never happens, and nothing says so.

Every "my settings cleared themselves" report on this page has been a
card writing a field it had not read yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402
from ppe_pipeline_api.main import app  # noqa: E402

_PASSWORD = "test-password"


@pytest.fixture
def client(monkeypatch):
    """Logs in for real, the way the rest of the suite does.

    An earlier version monkeypatched `auth.require_session` and THEN keyed
    `dependency_overrides` on it -- which keys the override on the patched
    object, not the one the route closed over, so every request still got
    a 401.
    """
    monkeypatch.setenv("PPE_APP_PASSWORD", _PASSWORD)
    c = TestClient(app)
    token = c.post("/auth/login", json={"password": _PASSWORD}).json()["token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


@pytest.fixture
def armed(monkeypatch):
    stored = {"mode": "schedule", "week_cutoff": "2026-03-25T00:00:00.000Z"}
    monkeypatch.setattr(pc, "read_config", lambda: dict(stored))
    monkeypatch.setattr(pc, "write_config", lambda patch: stored.update(patch) or dict(stored))
    monkeypatch.setattr(pc, "sources_from_picked", lambda cfg: [])
    return stored


class TestClearingTheCutOff:
    def test_it_is_refused_while_a_schedule_is_armed(self, client, armed):
        res = client.post("/track2/pull-control/config", json={"week_cutoff": ""})
        assert res.status_code == 409
        assert "counted from it" in res.json()["detail"]
        # And the stored value is untouched.
        assert armed["week_cutoff"] == "2026-03-25T00:00:00.000Z"

    def test_whitespace_counts_as_clearing(self, client, armed):
        assert client.post("/track2/pull-control/config",
                           json={"week_cutoff": "   "}).status_code == 409

    def test_a_real_date_is_accepted(self, client, armed):
        res = client.post("/track2/pull-control/config",
                          json={"week_cutoff": "2026-04-01T00:00:00.000Z"})
        assert res.status_code == 200
        assert armed["week_cutoff"] == "2026-04-01T00:00:00.000Z"

    def test_clearing_is_allowed_when_nothing_is_scheduled(self, client, monkeypatch):
        stored = {"mode": "off", "week_cutoff": "2026-03-25T00:00:00.000Z"}
        monkeypatch.setattr(pc, "read_config", lambda: dict(stored))
        monkeypatch.setattr(pc, "write_config", lambda p: stored.update(p) or dict(stored))
        monkeypatch.setattr(pc, "sources_from_picked", lambda cfg: [])
        res = client.post("/track2/pull-control/config", json={"week_cutoff": ""})
        assert res.status_code == 200
        assert stored["week_cutoff"] == ""

    def test_a_patch_that_does_not_mention_it_is_untouched(self, client, armed):
        res = client.post("/track2/pull-control/config", json={"root": "retraining"})
        assert res.status_code == 200
        assert armed["week_cutoff"] == "2026-03-25T00:00:00.000Z"
