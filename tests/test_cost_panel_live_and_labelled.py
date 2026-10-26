"""Today's cost reads live, and says whose jobs it is counting.

The panel listed every GPU job on the account -- which is right, the
bill charges for all of them -- but said nothing about which were this
app's. On 15 October it showed five jobs totalling $0.25, three of them
from another system entirely, and nothing on the row distinguished them.

It also read once on mount and never again, so a figure that had moved
on looked exactly like one that had not.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2TrainingCostToday.tsx")


def _code() -> str:
    return PANEL.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


# --- whose job is it ------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "pipelines-q5ypp1t6zuvz-TrainYolo-o4ZUgXQ08A",
    "pipelines-dkvgoit79m6z-TrainYolo-MF5KoR0QDf",
])
def test_a_pipeline_training_job_is_ours(name):
    assert t2.is_own_pipeline_job(name) is True


@pytest.mark.parametrize("name", [
    "t2-rt1-smoke-20261015-092009",
    "t2-rt1-new-only-20261015-092751",
    "someone-elses-job",
])
def test_a_job_from_elsewhere_is_not(name):
    assert t2.is_own_pipeline_job(name) is False


def test_a_pipeline_job_from_another_step_is_not_a_training_job_of_ours():
    """The rule is the pair -- `pipelines-` AND `-TrainYolo-`. A
    different step is a different job."""
    assert t2.is_own_pipeline_job("pipelines-abc-SomeOtherStep-xyz") is False


def test_our_evaluation_processing_job_is_still_ours():
    """It runs under the same execution and is named the same way, but
    its step is not TrainYolo -- the training rule would call this app's
    own evaluation foreign."""
    assert t2.is_own_processing_job("pipelines-abc-Evaluate-xyz") is True
    assert t2.is_own_processing_job("someone-elses-processing") is False


def test_the_gpu_guard_uses_the_same_rule():
    """Two copies would eventually disagree, and the disagreement would
    be a job the guard treats as foreign and the cost panel calls ours."""
    import inspect
    assert "is_own_pipeline_job(name)" in inspect.getsource(t2.get_gpu_resource_usage)


# --- the rows and the totals carry it ---------------------------------------------------

def test_every_cost_row_says_whose_it_is(monkeypatch):
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [
            {"TrainingJobName": "pipelines-abc-TrainYolo-x"},
            {"TrainingJobName": "t2-rt1-smoke-1"},
        ]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
        "BillableTimeInSeconds": 3600,
        "TrainingJobStatus": "Completed"})
    monkeypatch.setattr(t2.sm, "list_processing_jobs", lambda **kw: {
        "ProcessingJobSummaries": []})
    monkeypatch.setattr(t2, "sagemaker_hourly_rate", lambda i, c: 1.0)

    out = t2.training_cost_today()
    by = {j["name"]: j for j in out["jobs"]}
    assert by["pipelines-abc-TrainYolo-x"]["own_pipeline"] is True
    assert by["t2-rt1-smoke-1"]["own_pipeline"] is False


def test_the_total_is_split_by_whose_it_is(monkeypatch):
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [
            {"TrainingJobName": "pipelines-abc-TrainYolo-x"},
            {"TrainingJobName": "t2-rt1-smoke-1"},
        ]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
        "BillableTimeInSeconds": 3600,
        "TrainingJobStatus": "Completed"})
    monkeypatch.setattr(t2.sm, "list_processing_jobs", lambda **kw: {
        "ProcessingJobSummaries": []})
    monkeypatch.setattr(t2, "sagemaker_hourly_rate", lambda i, c: 2.0)

    out = t2.training_cost_today()
    assert out["estimated_ours"] == 2.0
    assert out["estimated_other"] == 2.0
    assert out["estimated"] == 4.0


# --- it reads live ------------------------------------------------------------------------

def test_it_polls_every_five_seconds():
    assert "window.setInterval(tick, 5000)" in _flat()


def test_it_stops_polling_when_the_tab_is_hidden():
    """A background tab polling an AWS API forever is throttling
    somebody else's call for numbers nobody is looking at."""
    code = _flat()
    assert "if (!document.hidden) void reload();" in code
    assert 'addEventListener("visibilitychange", tick)' in code


def test_the_interval_is_torn_down():
    code = _flat()
    assert "window.clearInterval(timer);" in code
    assert 'removeEventListener("visibilitychange", tick)' in code


def test_cost_explorer_is_not_polled():
    """It charges $0.01 a request and lags about a day. Re-reading it
    every five seconds would bill real money for a figure that cannot
    have changed."""
    code = _code()
    reload_fn = code.split("const reload = useCallback")[1].split("}, []);")[0]
    assert "getTrack2Cost(" not in reload_fn


def test_there_is_a_refresh_button():
    code = _flat()
    assert '{busy ? "Refreshing..." : "Refresh"}' in code
    assert "onClick={() => void reload()}" in code


def test_it_says_when_it_last_read():
    """A figure that quietly stopped refreshing looks exactly like one
    that has not changed."""
    code = _flat()
    assert "read ${at.toLocaleTimeString()}" in code


def test_a_failed_tick_does_not_stop_the_next_one():
    code = _code()
    reload_fn = code.split("const reload = useCallback")[1].split("}, []);")[0]
    assert "catch" in reload_fn
    assert "finally" in reload_fn


# --- and labels the rows --------------------------------------------------------------------

def test_each_row_is_labelled():
    code = _flat()
    assert '{job.own_pipeline ? "this app" : "another system"}' in code


def test_the_label_is_coloured_by_whose_it_is():
    code = _flat()
    assert 'job.own_pipeline ? "text-accent" : "text-critical"' in code


def test_the_caption_says_how_much_was_not_ours():
    code = _flat()
    assert "is from jobs this app did not start." in code


def test_it_says_nothing_when_everything_was_ours():
    """A line reading "$0.00 is from jobs this app did not start" is
    noise on the ordinary day."""
    code = _flat()
    assert "measured.estimated_other > 0 && (" in code


def test_the_job_list_is_three_rows_then_scrolls():
    """A busy day lists a dozen jobs; the list scrolls instead of pushing
    the page down. Rows must not shrink, or a capped flex column squashes
    the truncating rows rather than scrolling them."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
           / "components" / "Track2TrainingCostToday.tsx").read_text(encoding="utf-8")
    assert "max-h-[3.25rem]" in src and "overflow-y-auto" in src
    assert 'className="shrink-0 truncate" title={job.name}' in src


def test_the_job_list_is_in_start_order():
    """Earliest first, so the list reads as the day happened; a job with
    no start yet goes last."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
           / "components" / "Track2TrainingCostToday.tsx").read_text(encoding="utf-8")
    assert '.sort((a, b) => (a.started_at ?? "\uffff").localeCompare(b.started_at ?? "\uffff")' in src
    # ...and either way round, chosen on the card (2026-10-26).
    assert '* (newestFirst ? -1 : 1))' in src
    assert 'useCardOpen("cost_today_newest_first", false)' in src


def test_the_history_buttons_say_what_they_export():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
           / "components" / "Track2TrainingAuditPanel.tsx").read_text(encoding="utf-8")
    assert "Export training record as CSV" in src
    assert '"Export .pt to S3"' in src
