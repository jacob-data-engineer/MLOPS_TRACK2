"""The guard that stops two processes submitting onto one GPU slot.

`_pending_submission` and `_custom_submission_at` are module globals, so
they were invisible to anything but the process that set them. That was
survivable while one container was the only thing starting runs. It is
not survivable now, for two reasons that both already existed:

  * the dev and stable containers share these tables and both run the
    training-queue poll loop;
  * the queue-runner Lambda is a brand new process on every tick.

The window is real. CodeBuild accepts the build and returns; SageMaker
does not report the pipeline as Executing for minutes. A second process
looking in that window sees an idle GPU, pops the next entry and submits
it -- and the account has exactly one training instance, so that run
fails outright with SAGEMAKER_RESOURCE_LIMIT and no retry.
"""
from __future__ import annotations

import datetime as dt

import pytest

from ppe_pipeline_api import track2_shared as t2


@pytest.fixture
def marker_table(monkeypatch):
    """Stands in for the audit table, holding just the marker row."""
    rows: dict = {}

    class Fake:
        def put_item(self, Item):                                  # noqa: N803
            rows[Item["id"]] = Item
            return {}

        def get_item(self, Key):                                   # noqa: N803
            item = rows.get(Key["id"])
            return {"Item": item} if item else {}

        def delete_item(self, Key):                                # noqa: N803
            rows.pop(Key["id"], None)
            return {}

    monkeypatch.setattr(t2, "training_audit_table", lambda: Fake())
    return rows


def test_a_submission_is_recorded_where_other_processes_can_see_it(marker_table):
    t2.mark_training_submitted("build-123")
    got = t2.recent_training_submission()
    assert got is not None
    assert got["reference"] == "build-123"


def test_nothing_recorded_means_nothing_blocking(marker_table):
    assert t2.recent_training_submission() is None


def test_the_marker_stops_blocking_once_it_is_old(marker_table, monkeypatch):
    """It is a race guard, not a record. A run that genuinely never
    started must not wedge the queue for ever."""
    t2.mark_training_submitted("build-123")
    row = marker_table[t2.TRAINING_SUBMISSION_MARKER_ID]
    old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        seconds=t2.SUBMISSION_GUARD_SECONDS + 60)
    row["submitted_at"] = old.isoformat()
    assert t2.recent_training_submission() is None


def test_clearing_it_releases_the_slot(marker_table):
    t2.mark_training_submitted("build-123")
    t2.clear_training_submitted()
    assert t2.recent_training_submission() is None


def test_the_guard_reports_the_marker_a_different_process_wrote(marker_table, monkeypatch):
    """The whole point: this process never called mark_training_submitted,
    and must still refuse."""
    marker_table[t2.TRAINING_SUBMISSION_MARKER_ID] = {
        "id": t2.TRAINING_SUBMISSION_MARKER_ID,
        "record_type": "training_submission",
        "reference": "build-from-the-lambda",
        "submitted_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    # Everything else says the GPU is free.
    monkeypatch.setattr(t2, "custom_run_in_progress", lambda: None)
    monkeypatch.setattr(t2, "get_gpu_resource_usage", lambda: [])
    monkeypatch.setattr(t2, "_pending_submission", None, raising=False)
    monkeypatch.setattr(t2, "_custom_submission_at", None, raising=False)

    reason = t2.training_already_running_reason(None)
    assert reason is not None
    assert "has not been reported as running yet" in reason


def test_an_unreadable_marker_leaves_the_other_checks_in_force(monkeypatch):
    """Inventing a block nobody can clear would be worse than the race.
    The SageMaker and in-memory checks still apply either way."""
    class Broken:
        def get_item(self, Key):                                   # noqa: N803
            raise RuntimeError("table unavailable")

    monkeypatch.setattr(t2, "training_audit_table", lambda: Broken())
    assert t2.recent_training_submission() is None


def test_the_marker_is_not_mistaken_for_a_run_or_a_queue_entry():
    """It lives in the same table as the audit history and the training
    queue, both of which filter on record_type. A marker showing up as
    either would be a run that never happened."""
    import inspect

    audits = inspect.getsource(t2.scan_training_audits)
    assert 'record_type", "training_audit") == "training_audit"' in audits

    # The filter lives in the paginated scan `restore_queue` calls, not in
    # `restore_queue` itself, since the scan had to learn to read every
    # page rather than only the first.
    restore = inspect.getsource(t2.restore_queue) + inspect.getsource(t2._scan_queue_rows)
    assert "QUEUE_RECORD_TYPE" in restore
