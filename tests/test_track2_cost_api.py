"""Integration tests for /track2/cost -- per-24h SageMaker spend read from
Cost Explorer. The boto3 call is stubbed throughout: Cost Explorer bills
$0.01 per request, so a test suite must never actually call it."""

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


def _response(*days):
    return {"ResultsByTime": list(days)}


def _day(date: str, groups: dict[str, str]):
    return {
        "TimePeriod": {"Start": date, "End": date},
        "Groups": [
            {"Keys": [usage_type], "Metrics": {"UnblendedCost": {"Amount": amount, "Unit": "USD"}}}
            for usage_type, amount in groups.items()
        ],
    }


@pytest.fixture(autouse=True)
def _clear_cost_cache():
    """The cache is module-level and outlives a test. Without this, the
    first test to hit a given day-range fixes the answer every later test
    sees, however they stub Cost Explorer."""
    t2._cost_cache.clear()
    yield
    t2._cost_cache.clear()


@pytest.fixture(autouse=True)
def _stub_cost_explorer(monkeypatch):
    monkeypatch.setattr(
        t2.cost_explorer, "get_cost_and_usage",
        lambda **kw: _response(
            _day("2026-09-13", {"APS1-Host:ml.m5.2xlarge": "13.824", "APS1-Train:ml.g4dn.xlarge": "1.10"}),
            _day("2026-09-14", {"APS1-Host:ml.m5.2xlarge": "8.064"}),
        ),
    )


def test_cost_requires_auth():
    c = TestClient(app)
    assert c.get("/track2/cost").status_code == 401


def test_cost_returns_days_newest_first(client):
    body = client.get("/track2/cost").json()
    assert [d["date"] for d in body["days"]] == ["2026-09-14", "2026-09-13"]
    assert body["days"][0]["total"] == pytest.approx(8.064)
    assert body["days"][1]["total"] == pytest.approx(14.924)
    assert body["total"] == pytest.approx(22.988)
    assert body["currency"] == "USD"


def test_cost_items_sorted_by_amount_within_a_day(client):
    body = client.get("/track2/cost").json()
    day = next(d for d in body["days"] if d["date"] == "2026-09-13")
    assert [i["usage_type"] for i in day["items"]] == [
        "APS1-Host:ml.m5.2xlarge",
        "APS1-Train:ml.g4dn.xlarge",
    ]


def test_cost_aggregates_usage_types_across_the_period(client):
    body = client.get("/track2/cost").json()
    hosting = next(i for i in body["by_usage_type"] if i["usage_type"] == "APS1-Host:ml.m5.2xlarge")
    assert hosting["amount"] == pytest.approx(21.888)
    # Sorted biggest first, so the standing hosting cost leads.
    assert body["by_usage_type"][0]["usage_type"] == "APS1-Host:ml.m5.2xlarge"


def test_cost_drops_zero_amount_groups(client, monkeypatch):
    """Cost Explorer returns a group per usage type that existed in the
    window, including ones that cost nothing that day -- those are noise
    in a breakdown, not information."""
    monkeypatch.setattr(
        t2.cost_explorer, "get_cost_and_usage",
        lambda **kw: _response(_day("2026-09-14", {"APS1-Host:ml.m5.2xlarge": "5.00", "APS1-Studio:Idle": "0"})),
    )
    body = client.get("/track2/cost").json()
    assert [i["usage_type"] for i in body["days"][0]["items"]] == ["APS1-Host:ml.m5.2xlarge"]


def test_cost_days_param_is_clamped(client, monkeypatch):
    """A hand-edited ?days=99999 must not turn into an enormous (and
    billable) Cost Explorer query."""
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return _response()

    monkeypatch.setattr(t2.cost_explorer, "get_cost_and_usage", _capture)
    client.get("/track2/cost", params={"days": 99999})

    start = seen["TimePeriod"]["Start"]
    end = seen["TimePeriod"]["End"]
    from datetime import date
    span = (date.fromisoformat(end) - date.fromisoformat(start)).days
    assert span <= t2.COST_MAX_DAYS + 1


def test_cost_is_cached_between_calls(client, monkeypatch):
    """Cost Explorer bills per request and the Deploy page reads today's
    endpoint cost on every visit, so a second read must not spend again."""
    calls = []
    monkeypatch.setattr(t2.cost_explorer, "get_cost_and_usage", lambda **kw: calls.append(1) or _response())

    client.get("/track2/cost", params={"days": 7})
    client.get("/track2/cost", params={"days": 7})
    assert len(calls) == 1


def test_cost_refresh_bypasses_the_cache(client, monkeypatch):
    """The Cost tab's Refresh button has to mean it -- a cached answer
    behind a button labelled Refresh is a lie about how current it is."""
    calls = []
    monkeypatch.setattr(t2.cost_explorer, "get_cost_and_usage", lambda **kw: calls.append(1) or _response())

    client.get("/track2/cost", params={"days": 7})
    client.get("/track2/cost", params={"days": 7, "refresh": "true"})
    assert len(calls) == 2


def test_cost_cache_is_per_day_range(client, monkeypatch):
    """1d and 14d are different questions; one must not answer the other."""
    calls = []
    monkeypatch.setattr(t2.cost_explorer, "get_cost_and_usage", lambda **kw: calls.append(1) or _response())

    client.get("/track2/cost", params={"days": 1})
    client.get("/track2/cost", params={"days": 14})
    assert len(calls) == 2


def test_cost_filters_to_sagemaker_and_groups_by_usage_type(client, monkeypatch):
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return _response()

    monkeypatch.setattr(t2.cost_explorer, "get_cost_and_usage", _capture)
    client.get("/track2/cost")

    assert seen["Granularity"] == "DAILY"
    assert seen["Filter"] == {"Dimensions": {"Key": "SERVICE", "Values": ["Amazon SageMaker"]}}
    assert seen["GroupBy"] == [{"Type": "DIMENSION", "Key": "USAGE_TYPE"}]
