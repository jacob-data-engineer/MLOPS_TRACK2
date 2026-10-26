"""Stop Training can end a GPU job this app did not start.

The account has ONE on-demand ml.g4dn.xlarge slot and the guard refuses
to start a run while anything holds it -- including a job from another
system. The app could SEE that job, name it in the status bar, and do
nothing about it, so the slot could stay blocked for hours with no
control anywhere on the page. Stop Training was greyed out in exactly
that state.

TWO LIMITS, and they are different kinds:

  * This repo's code -- the route stopped only pipeline executions.
    Fixed here.
  * The ACCOUNT -- the host role has no `sagemaker:StopTrainingJob`:

        AccessDeniedException ... User: .../track2-yolo-review-host
        is not authorized to perform: sagemaker:StopTrainingJob

    That is a policy change. The grant is written into
    terraform/app/audit.tf, and `deploy.sh` does not apply terraform, so
    the code ships before the permission exists and must say so clearly
    until it does.
"""
from __future__ import annotations

import pathlib
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
AUDIT_TF = (pathlib.Path(__file__).resolve().parents[1]
            / "terraform" / "app" / "audit.tf")


def _jobs(*specs):
    return [{"name": n, "own_pipeline": own, "spot": False,
             "started": True, "start_time": None}
            for n, own in specs]


# --- what it stops ------------------------------------------------------------------

def test_a_foreign_job_is_stopped(monkeypatch):
    stopped = []
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: _jobs(("someone-elses", False)))
    monkeypatch.setattr(t2.sm, "stop_training_job",
                        lambda TrainingJobName: stopped.append(TrainingJobName))
    out = t2.stop_foreign_gpu_jobs()
    assert stopped == ["someone-elses"]
    assert out["stopped"] == ["someone-elses"]
    assert out["problems"] == []


def test_our_own_pipeline_job_is_left_alone(monkeypatch):
    """Stopping the JOB under a running execution leaves the execution to
    fail its step rather than stop cleanly. `stop_pipeline_execution` is
    the right call for those, and the route makes it first."""
    stopped = []
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: _jobs(("pipelines-abc-TrainYolo-x", True)))
    monkeypatch.setattr(t2.sm, "stop_training_job",
                        lambda TrainingJobName: stopped.append(TrainingJobName))
    out = t2.stop_foreign_gpu_jobs()
    assert stopped == []
    assert out["stopped"] == []


def test_several_foreign_jobs_all_get_stopped(monkeypatch):
    stopped = []
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: _jobs(("a", False), ("mine", True), ("b", False)))
    monkeypatch.setattr(t2.sm, "stop_training_job",
                        lambda TrainingJobName: stopped.append(TrainingJobName))
    assert t2.stop_foreign_gpu_jobs()["stopped"] == ["a", "b"]
    assert "mine" not in stopped


def test_one_refusal_does_not_hide_the_others(monkeypatch):
    """A job that will not stop must not take the ones that did with
    it."""
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: _jobs(("bad", False), ("good", False)))

    def stop(TrainingJobName):
        if TrainingJobName == "bad":
            raise RuntimeError("boom")

    monkeypatch.setattr(t2.sm, "stop_training_job", stop)
    out = t2.stop_foreign_gpu_jobs()
    assert out["stopped"] == ["good"]
    assert out["problems"] and "bad" in out["problems"][0]


# --- the permission it does not have yet ----------------------------------------------

def test_access_denied_is_named_as_a_policy_change(monkeypatch):
    """Not "stop failed". The host role genuinely cannot do this yet,
    and a message that does not say so sends somebody looking at the
    code."""
    monkeypatch.setattr(t2, "get_gpu_resource_usage",
                        lambda: _jobs(("someone-elses", False)))

    def denied(TrainingJobName):
        raise RuntimeError(
            "An error occurred (AccessDeniedException) when calling the "
            "StopTrainingJob operation: User: ... is not authorized")

    monkeypatch.setattr(t2.sm, "stop_training_job", denied)
    problem = t2.stop_foreign_gpu_jobs()["problems"][0]
    assert "not allowed to stop training jobs" in problem
    assert "sagemaker:StopTrainingJob" in problem
    assert "policy change" in problem
    assert "terraform/app/audit.tf" in problem


def test_the_grant_is_written_into_the_stack():
    """`deploy.sh` reads terraform outputs and never applies, so the
    code can ship before the permission exists -- but the grant has to
    be recorded, or a future apply silently removes nothing and fixes
    nothing."""
    tf = AUDIT_TF.read_text(encoding="utf-8")
    assert "sagemaker:StopTrainingJob" in tf
    assert "StopAnyGpuTrainingJob" in tf


def test_the_stack_says_why_the_resource_is_a_wildcard():
    """Every other statement in that policy is scoped. This one cannot
    be, and that is worth knowing before applying it."""
    tf = AUDIT_TF.read_text(encoding="utf-8")
    assert "WHY THE RESOURCE IS A WILDCARD" in tf


# --- the route --------------------------------------------------------------------------

def test_the_flag_is_off_by_default():
    from ppe_pipeline_api.routers.track2_models import StopTrainingRequest

    assert StopTrainingRequest().include_foreign is False


def test_the_route_only_reaches_foreign_jobs_when_asked():
    import inspect
    from ppe_pipeline_api.routers import track2_models as m

    code = inspect.getsource(m.stop_training)
    assert "if payload.include_foreign:" in code
    assert "t2.stop_foreign_gpu_jobs()" in code


def test_our_run_is_stopped_first():
    """A pipeline execution stops cleanly; its training job stopped from
    underneath does not."""
    import inspect
    from ppe_pipeline_api.routers import track2_models as m

    code = inspect.getsource(m.stop_training)
    assert code.index("stop_pipeline_execution") < code.index("stop_foreign_gpu_jobs")


def test_nothing_executing_but_something_foreign_says_so():
    """"No training run is currently executing" while the guard refuses
    every run is the card contradicting itself."""
    import inspect
    from ppe_pipeline_api.routers import track2_models as m

    code = inspect.getsource(m.stop_training)
    assert "No run started here is executing." in code
    assert "was not started here" in code


# --- the dialog ---------------------------------------------------------------------------

def _page() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_the_button_asks_before_it_stops_anything():
    code = _page()
    assert "onClick={() => { setAlsoForeign(false); setStopOpen(true); }}" in code
    assert "<ConfirmDialog" in code


def test_the_dialog_names_each_foreign_job():
    code = " ".join(_page().split())
    assert "{foreignGpuJobs.map((j) => (" in code
    assert "{j.name}" in code


def test_the_harsher_choice_is_in_the_same_dialog():
    """Not a second control somewhere else."""
    code = " ".join(_page().split())
    assert "checked={alsoForeign}" in code
    assert "another system." in code


def test_the_escalation_is_off_every_time_it_opens():
    """Reopening with the previous answer still ticked is how somebody
    stops a stranger's job by accident."""
    code = _page()
    assert "setAlsoForeign(false); setStopOpen(true);" in code


def test_the_dialog_says_what_is_not_touched():
    code = " ".join(_page().split())
    assert "Nothing else is touched." in code
    assert "the training queue keeps its entries" in code


def test_the_button_is_reachable_when_only_a_foreign_job_holds_the_gpu():
    """It was greyed out in exactly the state where stopping matters
    most."""
    code = " ".join(_page().split())
    assert "!data.gpu_jobs.some((j) => !j.own_pipeline)" in code
