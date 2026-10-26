"""The training history shows how long each run took, on the row.

It was captured all along -- `started_at` and `ended_at` are written by
`_reconcile_one_audit` from `describe_pipeline_execution` -- and
rendered only inside the expanded detail. "How long did it take" is the
question the list is read to answer, and it was one click away on every
row.

And a RUNNING row read "--" for its whole twenty minutes, which is
exactly when somebody wants the number.

WHAT IT MEASURES. The pipeline execution's wall time: waiting for GPU
capacity, training, evaluation and registration. NOT billable GPU time.
The gap is real -- on 2026-10-15 execution custom-dc730990 started at
18:41:07 and its training job got no instance until 18:44:04, three
minutes nobody was charged for -- so the field says "whole run" rather
than leaving it to be assumed.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2TrainingAuditPanel.tsx")


def _code() -> str:
    return PANEL.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


def _helper() -> str:
    code = _code()
    start = code.index("function formatDuration(")
    return code[start:code.index("\n}", start)]


# --- it is captured server-side ---------------------------------------------------

def test_the_audit_row_carries_both_timestamps():
    """Nothing to add server-side -- this is what made it a display
    change rather than a data one."""
    import inspect
    from ppe_pipeline_api import track2_shared as t2

    code = inspect.getsource(t2._reconcile_one_audit)
    assert 'updates["started_at"]' in code
    assert 'updates["ended_at"]' in code


def test_they_come_from_the_pipeline_execution():
    """Which is why the label says "whole run"."""
    import inspect
    from ppe_pipeline_api import track2_shared as t2

    code = inspect.getsource(t2._reconcile_one_audit)
    assert "describe_pipeline_execution" in code


# --- and now shown on the row ------------------------------------------------------

def test_the_collapsed_row_shows_the_duration():
    code = _flat()
    row = code.split("{formatTime(audit.submitted_at)}</span>")[1]
    row = row[:row.index("</button>")]
    assert "formatDuration(audit.started_at, audit.ended_at," in row


def test_a_row_with_no_start_shows_nothing_rather_than_a_dash():
    """A queued-but-never-started run has no clock to show."""
    code = _flat()
    assert "{audit.started_at && (" in code


def test_the_expanded_field_says_what_it_measures():
    code = _flat()
    assert 'label="Duration (whole run)"' in code


# --- a running row counts up --------------------------------------------------------

def test_an_unfinished_run_counts_up_instead_of_reading_a_dash():
    helper = _helper()
    assert "running ? Date.now() : NaN" in helper


def test_the_running_statuses_are_named():
    code = _code()
    assert 'const STILL_GOING = new Set(["Executing", "Stopping", "InProgress"]);' in code


def test_both_places_use_the_same_running_test():
    """The row and the expanded field describe the same run; two tests
    for "is it still going" would eventually disagree."""
    assert _flat().count("STILL_GOING.has(audit.status)") == 2


def test_a_finished_run_ignores_the_running_flag():
    """`ended_at` wins wherever it exists, so a stale status cannot make
    a finished run tick forever."""
    helper = _helper()
    assert "ended ? new Date(ended).getTime() : (running ? Date.now() : NaN)" in helper


# --- the formatting that was already right -------------------------------------------

def test_under_an_hour_is_minutes_and_seconds():
    helper = _helper()
    assert "if (minutes < 60) return `${minutes}m ${seconds}s`;" in helper


def test_over_an_hour_is_hours_and_minutes():
    helper = _helper()
    assert "${Math.floor(minutes / 60)}h ${minutes % 60}m" in helper


def test_a_negative_or_unparseable_span_is_a_dash():
    """Clock skew between the server and the browser would otherwise
    read as "-3m 0s"."""
    helper = _helper()
    assert "if (!Number.isFinite(ms) || ms < 0) return \"--\";" in helper


# --- what must not have broken ---------------------------------------------------------

def test_the_row_still_shows_status_model_and_time():
    code = _flat()
    assert "<Badge tone={statusTone(audit.status)} label={audit.status} />" in code
    assert "{formatTime(audit.submitted_at)}" in code
    # Now the FALLBACK branch, for runs with no detector -- and a
    # template literal rather than JSX interpolation. The row leads with
    # the Model Name where there is one; see
    # tests/test_history_names_the_model.py.
    assert "`YOLO${audit.generation}-${audit.size.toUpperCase()}`" in code
    assert "{audit.detector_slug}" in code


def test_started_and_ended_are_still_in_the_detail():
    code = _flat()
    assert 'label="Started"' in code
    assert 'label="Ended"' in code
