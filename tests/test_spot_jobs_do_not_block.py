"""A managed-spot job must not block an on-demand run.

They are SEPARATE quotas. Read off the account, not assumed:

    aws service-quotas list-service-quotas --service-code sagemaker \\
      --region ap-southeast-1 \\
      --query "Quotas[?contains(QuotaName,'g4dn.xlarge')].[QuotaName,Value]"

    ml.g4dn.xlarge for training job usage        1.0
    ml.g4dn.xlarge for spot training job usage   1.0

Our own TrainYolo step is on-demand -- `EnableManagedSpotTraining` is not
set on it, which `describe-pipeline` shows -- so a spot job elsewhere in
the account cannot exhaust the quota our runs need.

The guard counted every InProgress g4dn.xlarge job regardless. On
2026-10-15 a spot smoke test sat in a capacity retry loop for its whole
45-minute wait -- `TrainingStartTime` null, no instance, $0.00 billed --
and for all of it the bar read "GPU in use -- another job" and the queue
refused to pop. The GPU was idle the entire time.

That was OUR refusal, not AWS's.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402


def _job(name, *, spot=False, own=False, started=True):
    return {"name": name, "start_time": "2026-10-15T08:00:00+00:00" if started else None,
            "own_pipeline": own, "spot": spot, "started": started}


@pytest.fixture
def quiet(monkeypatch):
    """Everything else the guard looks at says "free"."""
    monkeypatch.setattr(t2, "custom_run_in_progress", lambda: None)
    monkeypatch.setattr(t2, "recent_training_submission", lambda: None)
    monkeypatch.setattr(t2, "_custom_submission_at", None, raising=False)
    monkeypatch.setattr(t2, "_pending_submission", None, raising=False)


# --- the refusal that was wrong ----------------------------------------------------

def test_a_spot_job_does_not_block_a_run(quiet, monkeypatch):
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: [_job("t2-rt1-smoke", spot=True, started=False)])
    assert t2.training_already_running_reason(None) is None


def test_a_spot_job_that_IS_running_still_does_not_block(quiet, monkeypatch):
    """Different quota is different quota, allocated instance or not."""
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: [_job("t2-rt1-smoke", spot=True, started=True)])
    assert t2.training_already_running_reason(None) is None


# --- the refusals that are right ---------------------------------------------------

def test_an_on_demand_job_outside_this_app_still_blocks(quiet, monkeypatch):
    """One on-demand slot, and this would take it."""
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: [_job("somebody-elses-job")])
    reason = t2.training_already_running_reason(None)
    assert reason and "somebody-elses-job" in reason


def test_a_mix_blocks_on_the_on_demand_one(quiet, monkeypatch):
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [
        _job("spot-one", spot=True),
        _job("on-demand-one"),
    ])
    reason = t2.training_already_running_reason(None)
    assert reason and "on-demand-one" in reason


def test_our_own_pipeline_is_still_filtered_as_before(quiet, monkeypatch):
    """`own_pipeline` means the run this app started -- it is reported by
    the progress check, not as somebody else holding the slot."""
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: [_job("pipelines-abc-TrainYolo-xyz", own=True)])
    assert t2.training_already_running_reason(None) is None


def test_an_executing_run_still_blocks(quiet, monkeypatch):
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    reason = t2.training_already_running_reason(
        {"status": "Executing", "execution_arn": "arn:aws:x/exec-123"})
    assert reason and "exec-123" in reason


def test_a_custom_run_still_blocks(quiet, monkeypatch):
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    monkeypatch.setattr(t2, "custom_run_in_progress", lambda: "custom-abc")
    reason = t2.training_already_running_reason(None)
    assert reason and "custom-abc" in reason


# --- what the usage list now carries -------------------------------------------------

def test_the_usage_list_reports_spot(monkeypatch):
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "j1"}]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": t2.GPU_INSTANCE_TYPE},
        "EnableManagedSpotTraining": True})
    jobs = t2.get_gpu_resource_usage()
    assert jobs[0]["spot"] is True


def test_the_usage_list_says_whether_an_instance_was_allocated(monkeypatch):
    """`TrainingStartTime` is null until one is. A job in a capacity
    retry loop is billed nothing and holds no hardware, and "InProgress"
    on its own reads as "running"."""
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "j1"}]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": t2.GPU_INSTANCE_TYPE}})
    jobs = t2.get_gpu_resource_usage()
    assert jobs[0]["started"] is False
    assert jobs[0]["spot"] is False


def test_only_one_describe_call_per_job(monkeypatch):
    """The spot flag rides on the describe that was already being made."""
    calls = []
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "j1"}]})

    def describe(TrainingJobName):
        calls.append(TrainingJobName)
        return {"ResourceConfig": {"InstanceType": t2.GPU_INSTANCE_TYPE}}

    monkeypatch.setattr(t2.sm, "describe_training_job", describe)
    t2.get_gpu_resource_usage()
    assert calls == ["j1"]


def test_a_non_gpu_job_is_still_ignored(monkeypatch):
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "cpu"}]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": "ml.m5.xlarge"}})
    assert t2.get_gpu_resource_usage() == []


# --- the bar says what it means --------------------------------------------------------

def test_busy_counts_only_what_would_block_a_run():
    import inspect
    from ppe_pipeline_api.routers import track2_retention as r

    code = inspect.getsource(r)
    assert '"busy": any(not j.get("spot") for j in gpu),' in code


def test_the_quota_finding_is_written_down_where_the_guard_is():
    """So the next person does not re-derive it from scratch, or worse,
    assume the opposite."""
    assert t2.SPOT_USES_A_DIFFERENT_QUOTA is True
    import inspect
    assert "spot training job usage" in inspect.getsource(t2.get_gpu_resource_usage) \
        or "spot training job usage" in (t2.__doc__ or "") \
        or "spot training job usage" in pathlib.Path(
            t2.__file__).read_text(encoding="utf-8")
