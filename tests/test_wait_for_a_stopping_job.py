"""A job that is still stopping holds the GPU; the queue waits for it.

Twice on 2026-10-20 a run was queued straight after Stop Training and
failed in one second with SAGEMAKER_RESOURCE_LIMIT: the gate listed only
InProgress jobs, and the stopped one was in `Stopping`, still holding the
1-instance quota.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ppe_pipeline_api import track2_shared as t2  # noqa: E402


class _SM:
    def __init__(self, stopping):
        self.stopping = stopping

    def list_training_jobs(self, StatusEquals, **kw):
        assert StatusEquals == "Stopping"
        return {"TrainingJobSummaries": [{"TrainingJobName": n} for n in self.stopping]}


def _clear(monkeypatch):
    monkeypatch.setattr(t2, "custom_run_in_progress", lambda: None)
    monkeypatch.setattr(t2, "recent_training_submission", lambda: None)
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    monkeypatch.setattr(t2, "_custom_submission_at", None)


def test_a_stopping_job_blocks_the_next_start(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(t2, "sm", _SM(["pipelines-t985-TrainYolo-x"]))
    reason = t2.training_already_running_reason(None)
    assert reason and "still stopping and holds the GPU" in reason


def test_no_stopping_job_lets_it_start(monkeypatch):
    _clear(monkeypatch)
    monkeypatch.setattr(t2, "sm", _SM([]))
    assert t2.training_already_running_reason(None) is None


def test_a_failed_listing_does_not_hold_the_queue_forever(monkeypatch):
    class _Broken:
        def list_training_jobs(self, **kw):
            raise RuntimeError("throttled")
    monkeypatch.setattr(t2, "sm", _Broken())
    assert t2.stopping_training_jobs() == []
