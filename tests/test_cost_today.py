"""Today's training spend has to answer today.

Cost Explorer is the authority and lags roughly a day, so on the
afternoon of 6 October -- after seven training runs -- the card read
$0.00. A cost figure that says "free" while the GPU is warm is the one
answer it must never give.

So the figure is MEASURED: SageMaker reports `BillableTimeInSeconds` per
training job, which is the quantity it bills on, and the published rate
turns it into money.
"""
from __future__ import annotations

import datetime as dt
import importlib

import pytest

t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")


@pytest.fixture
def rates(monkeypatch):
    """Stubbed rates. NOT autouse: the last two cases inspect the real
    `sagemaker_hourly_rate`, and a fixture that had replaced it would be
    reading its own lambda."""
    monkeypatch.setattr(t2, "_RATE_CACHE", {})
    monkeypatch.setattr(t2, "sagemaker_hourly_rate",
                        lambda instance, component:
                        1.03 if component == "Training" else 0.288)
    return None


def _sm(monkeypatch, training=(), processing=(), described=None):
    described = described or {}
    monkeypatch.setattr(t2.sm, "list_training_jobs",
                        lambda **kw: {"TrainingJobSummaries": list(training)})
    monkeypatch.setattr(t2.sm, "list_processing_jobs",
                        lambda **kw: {"ProcessingJobSummaries": list(processing)})
    monkeypatch.setattr(t2.sm, "describe_training_job",
                        lambda TrainingJobName: described[TrainingJobName])
    monkeypatch.setattr(t2.sm, "describe_processing_job",
                        lambda ProcessingJobName: described[ProcessingJobName])


def test_a_finished_training_job_is_priced_on_billable_seconds(rates, monkeypatch):
    """Not wall clock. SageMaker bills `BillableTimeInSeconds`, which
    excludes the time spent waiting for capacity."""
    _sm(monkeypatch,
        training=[{"TrainingJobName": "j1"}],
        described={"j1": {"BillableTimeInSeconds": 3600,
                          "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
                          "TrainingJobStatus": "Completed"}})
    got = t2.training_cost_today()
    assert got["estimated"] == pytest.approx(1.03)
    assert got["training_seconds"] == 3600
    assert got["complete"] is True


def test_several_jobs_add_up(rates, monkeypatch):
    """Seven runs at about six minutes each was the real day."""
    names = [f"j{i}" for i in range(7)]
    _sm(monkeypatch,
        training=[{"TrainingJobName": n} for n in names],
        described={n: {"BillableTimeInSeconds": 380,
                       "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
                       "TrainingJobStatus": "Completed"} for n in names})
    got = t2.training_cost_today()
    assert got["estimated"] == pytest.approx(7 * 380 / 3600 * 1.03, abs=1e-3)
    assert len(got["jobs"]) == 7


def test_a_running_job_is_counted_from_its_start(rates, monkeypatch):
    """`BillableTimeInSeconds` is absent until a job finishes. Treating
    that as zero would make the figure drop while the GPU is working."""
    began = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=30)
    _sm(monkeypatch,
        training=[{"TrainingJobName": "live"}],
        described={"live": {"TrainingStartTime": began,
                            "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
                            "TrainingJobStatus": "InProgress"}})
    got = t2.training_cost_today()
    assert got["training_seconds"] >= 1790
    assert got["estimated"] > 0.5


def test_evaluation_is_counted_at_the_processing_rate(rates, monkeypatch):
    """A run is two machines in turn, and the second one is not free."""
    began = dt.datetime(2026, 10, 6, 1, 0, tzinfo=dt.timezone.utc)
    _sm(monkeypatch,
        processing=[{"ProcessingJobName": "eval"}],
        described={"eval": {
            "ProcessingStartTime": began,
            "ProcessingEndTime": began + dt.timedelta(hours=1),
            "ProcessingResources": {"ClusterConfig": {"InstanceType": "ml.m5.xlarge"}},
            "ProcessingJobStatus": "Completed"}})
    got = t2.training_cost_today()
    assert got["estimated"] == pytest.approx(0.288)
    assert got["processing_seconds"] == 3600
    assert got["training_seconds"] == 0


def test_an_unpriced_instance_marks_the_total_incomplete(rates, monkeypatch):
    """So the UI can say "at least this" rather than presenting a short
    total as final."""
    monkeypatch.setattr(t2, "sagemaker_hourly_rate", lambda instance, component: 0.0)
    _sm(monkeypatch,
        training=[{"TrainingJobName": "j1"}],
        described={"j1": {"BillableTimeInSeconds": 3600,
                          "ResourceConfig": {"InstanceType": "ml.unknown"},
                          "TrainingJobStatus": "Completed"}})
    got = t2.training_cost_today()
    assert got["complete"] is False
    assert got["estimated"] == 0


def test_a_job_that_cannot_be_described_does_not_stop_the_others(rates, monkeypatch):
    described = {"good": {"BillableTimeInSeconds": 3600,
                          "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
                          "TrainingJobStatus": "Completed"}}

    def describe(TrainingJobName):
        if TrainingJobName not in described:
            raise RuntimeError("no access")
        return described[TrainingJobName]

    monkeypatch.setattr(t2.sm, "list_training_jobs",
                        lambda **kw: {"TrainingJobSummaries": [
                            {"TrainingJobName": "bad"}, {"TrainingJobName": "good"}]})
    monkeypatch.setattr(t2.sm, "list_processing_jobs",
                        lambda **kw: {"ProcessingJobSummaries": []})
    monkeypatch.setattr(t2.sm, "describe_training_job", describe)
    got = t2.training_cost_today()
    assert got["estimated"] == pytest.approx(1.03)
    assert got["complete"] is False


def test_a_quiet_day_is_zero_and_says_so(rates, monkeypatch):
    _sm(monkeypatch)
    got = t2.training_cost_today()
    assert got["estimated"] == 0
    assert got["jobs"] == []
    assert got["complete"] is True


def test_a_listing_that_fails_does_not_raise(rates, monkeypatch):
    def boom(**kw):
        raise RuntimeError("throttled")
    monkeypatch.setattr(t2.sm, "list_training_jobs", boom)
    monkeypatch.setattr(t2.sm, "list_processing_jobs", boom)
    got = t2.training_cost_today()
    assert got["estimated"] == 0


def test_the_jobs_come_back_dearest_first(rates, monkeypatch):
    _sm(monkeypatch,
        training=[{"TrainingJobName": "small"}, {"TrainingJobName": "big"}],
        described={
            "small": {"BillableTimeInSeconds": 60,
                      "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
                      "TrainingJobStatus": "Completed"},
            "big": {"BillableTimeInSeconds": 6000,
                    "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
                    "TrainingJobStatus": "Completed"}})
    got = t2.training_cost_today()
    assert [j["name"] for j in got["jobs"]] == ["big", "small"]


def test_only_todays_jobs_are_asked_for(rates, monkeypatch):
    """`CreationTimeAfter` is midnight UTC. Without it the figure would
    sweep in every job the account has ever run."""
    seen = {}
    monkeypatch.setattr(t2.sm, "list_training_jobs",
                        lambda **kw: seen.update(kw) or {"TrainingJobSummaries": []})
    monkeypatch.setattr(t2.sm, "list_processing_jobs",
                        lambda **kw: {"ProcessingJobSummaries": []})
    t2.training_cost_today()
    after = seen["CreationTimeAfter"]
    assert (after.hour, after.minute, after.second) == (0, 0, 0)
    assert after.date() == dt.datetime.now(dt.timezone.utc).date()


def test_the_rate_lookup_asks_for_the_right_component():
    """The same instance costs different amounts for training and for
    processing, and the usage types Cost Explorer reports are built from
    exactly that pair -- so the two sources agree by construction."""
    import inspect
    source = inspect.getsource(t2.sagemaker_hourly_rate)
    assert '"Field": "component"' in source
    assert '"Field": "regionCode"' in source
    # Pricing lives only in us-east-1 regardless of where the work runs.
    assert 'region_name="us-east-1"' in source


def test_a_rate_that_cannot_be_read_is_zero_not_a_guess(monkeypatch):
    """A hard-coded fallback price would be wrong the first time AWS
    changed one, and nothing would say so."""
    monkeypatch.setattr(t2, "_RATE_CACHE", {})

    def boom(*a, **k):
        raise RuntimeError("no pricing access")
    monkeypatch.setattr(t2.boto3, "client", boom)
    assert t2.sagemaker_hourly_rate("ml.g4dn.xlarge", "Training") == 0.0
