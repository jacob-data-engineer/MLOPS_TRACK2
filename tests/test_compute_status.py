"""GPU and CPU on one indicator, because a run uses both in turn.

A pipeline execution is not one machine. `TrainYolo` holds the account's
single GPU; `EvaluateYolo` then runs on an ml.m5.xlarge for seven or
eight minutes, and `RegisterYoloModel` is an API call. So between the end
of training and the end of the run, "the queue says a run is going" and
"the GPU is free" are BOTH true -- which read as a contradiction to
anybody who had not memorised the step layout, and was asked about twice
in one morning.
"""
from __future__ import annotations

import importlib

import pytest

t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")
retention = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")


def _jobs(monkeypatch, summaries, instance="ml.m5.xlarge"):
    monkeypatch.setattr(t2.sm, "list_processing_jobs",
                        lambda **kw: {"ProcessingJobSummaries": summaries})
    monkeypatch.setattr(t2.sm, "describe_processing_job",
                        lambda **kw: {"ProcessingResources": {
                            "ClusterConfig": {"InstanceType": instance}}})


def test_nothing_processing_is_empty(monkeypatch):
    _jobs(monkeypatch, [])
    assert t2.get_cpu_resource_usage() == []


def test_an_evaluation_job_is_reported_with_its_step(monkeypatch):
    """The step is read off SageMaker's own naming,
    `pipelines-<execution>-<StepName>-<suffix>`, so the bar can say
    "evaluating" rather than just "busy"."""
    _jobs(monkeypatch, [{"ProcessingJobName": "pipelines-abc123-EvaluateYolo-XYZ"}])
    got = t2.get_cpu_resource_usage()
    assert len(got) == 1
    assert got[0]["step"] == "EvaluateYolo"
    assert got[0]["own_pipeline"] is True
    assert got[0]["instance_type"] == "ml.m5.xlarge"


def test_a_job_from_elsewhere_is_not_ours(monkeypatch):
    _jobs(monkeypatch, [{"ProcessingJobName": "somebody-elses-job"}])
    got = t2.get_cpu_resource_usage()
    assert got[0]["own_pipeline"] is False
    assert got[0]["step"] == ""


def test_a_job_that_cannot_be_described_is_still_reported_busy(monkeypatch):
    """Busy is the safe answer. Dropping it would report the machine free
    while it is working, which is the failure this whole thing exists to
    stop."""
    monkeypatch.setattr(t2.sm, "list_processing_jobs",
                        lambda **kw: {"ProcessingJobSummaries": [
                            {"ProcessingJobName": "pipelines-a-EvaluateYolo-b"}]})

    def boom(**kw):
        raise RuntimeError("no access")
    monkeypatch.setattr(t2.sm, "describe_processing_job", boom)
    got = t2.get_cpu_resource_usage()
    assert len(got) == 1
    assert got[0]["instance_type"] == ""


def test_a_listing_that_raises_does_not_break_the_bar(monkeypatch):
    def boom(**kw):
        raise RuntimeError("throttled")
    monkeypatch.setattr(t2.sm, "list_processing_jobs", boom)
    assert t2.get_cpu_resource_usage() == []


def test_the_chain_status_cpu_block(monkeypatch):
    monkeypatch.setattr(retention.t2, "get_cpu_resource_usage", lambda: [
        {"name": "pipelines-a-EvaluateYolo-b", "step": "EvaluateYolo",
         "instance_type": "ml.m5.xlarge", "own_pipeline": True}])
    got = retention._cpu_status()
    assert got["busy"] is True
    assert got["ours"] is True
    assert got["step"] == "EvaluateYolo"


def test_the_cpu_block_when_idle(monkeypatch):
    monkeypatch.setattr(retention.t2, "get_cpu_resource_usage", lambda: [])
    got = retention._cpu_status()
    assert got["busy"] is False
    assert got["step"] == ""
    assert got["known"] is True


def test_the_cpu_block_survives_a_failure(monkeypatch):
    def boom():
        raise RuntimeError("no")
    monkeypatch.setattr(retention.t2, "get_cpu_resource_usage", boom)
    got = retention._cpu_status()
    assert got["known"] is False
    assert got["busy"] is False
    assert got["error"]


def test_both_chain_status_return_paths_carry_the_cpu():
    """The GPU call failing is exactly when somebody is watching the bar,
    so the early return must not be the one that drops it."""
    import inspect
    source = inspect.getsource(retention.chain_status)
    assert source.count("_cpu_status()") == 2, source


def test_the_models_live_endpoint_carries_cpu_jobs():
    import inspect
    models = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
    body = inspect.getsource(models.get_models_live).split('"""', 2)[-1]
    assert "cpu_jobs" in body


@pytest.mark.parametrize("name,step", [
    ("pipelines-xyz-EvaluateYolo-abc", "EvaluateYolo"),
    ("pipelines-xyz-SomethingElse-abc", "SomethingElse"),
    ("pipelines-short", ""),
    ("", ""),
])
def test_step_parsing(monkeypatch, name, step):
    _jobs(monkeypatch, [{"ProcessingJobName": name}] if name else [])
    got = t2.get_cpu_resource_usage()
    assert (got[0]["step"] if got else "") == step
