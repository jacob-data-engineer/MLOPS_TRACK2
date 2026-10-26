"""The GPU badge says how long the slot has been held.

"GPU in use -- this pipeline" answers "can I start a run". It does not
answer "how much longer", which is the next question every time, and the
bar had the timestamp needed to answer it sitting unused in the payload.

Read from `TrainingStartTime`, which is also when SageMaker starts
BILLING -- so the figure is elapsed paid time, not time since submit.
Those differ by a lot: on 2026-10-15 a job sat 45 minutes between
submission and its first billable second, and never got an instance at
all.
"""
from __future__ import annotations

import pathlib

BAR = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "components" / "Track2ChainStatusBar.tsx")
API = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "lib" / "api.ts")


def _code() -> str:
    return BAR.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


def _helper() -> str:
    code = _code()
    start = code.index("function gpuHeldFor(")
    return code[start:code.index("\n}", start)]


# --- the timer ---------------------------------------------------------------------

def test_the_badge_has_a_held_for_reading():
    assert "function gpuHeldFor(" in _code()


def test_it_is_minutes_and_a_clock_time():
    """"13m since 18:44" -- the elapsed figure answers "how much
    longer", the clock answers "since when", and neither replaces the
    other."""
    assert "${minutes}m since ${clock}" in _flat()


def test_it_is_measured_from_the_billing_start():
    """Not from submission. A job can sit for 45 minutes before its
    first billable second, and did."""
    assert "j.start_time" in _helper()


def test_it_is_rendered_beside_the_label_only_when_busy():
    code = _flat()
    assert "{gpu.busy && gpuHeld && (" in code


def test_it_does_not_replace_the_label():
    """The label answers "can I start a run" and is read at a different
    moment."""
    code = _flat()
    assert "{gpuLabel}</span>" in code
    assert code.index("{gpuLabel}</span>") < code.index("{gpuHeld}</span>")


# --- what it refuses to time ----------------------------------------------------------

def test_a_job_with_no_instance_yet_is_not_given_a_timer():
    """`TrainingStartTime` is null until an instance is allocated. A
    timer counting from a moment that has not happened is a number
    somebody would act on."""
    helper = _helper()
    assert 'return "waiting for capacity -- no instance yet";' in helper


def test_spot_jobs_are_not_timed():
    """They draw on a different quota, so one is not in the way -- and
    timing it would describe something that is not blocking anything."""
    helper = _helper()
    assert "filter((j) => !j.spot)" in helper


def test_no_blocking_job_means_no_reading():
    helper = _helper()
    assert "if (!blocking.length) return \"\";" in helper


def test_it_times_from_the_earliest_of_several():
    """The slot has been held since the FIRST of them, not the latest."""
    helper = _helper()
    assert "Math.min(...started)" in helper


def test_a_broken_timestamp_is_dropped_rather_than_shown_as_nan():
    helper = _helper()
    assert "Number.isNaN" in helper


def test_elapsed_never_goes_negative():
    """A clock skew between the server and the browser would otherwise
    read as "-3m since"."""
    helper = _helper()
    assert "Math.max(0," in helper


# --- the type it reads ------------------------------------------------------------------

def test_the_chain_status_carries_the_real_job_shape():
    """It declared a narrower copy without `spot`, so nothing reading
    that list could tell a spot job from one holding the slot."""
    api = API.read_text(encoding="utf-8")
    block = api.split("export interface Track2ChainStatus {")[1]
    block = block[:block.index("\n}")]
    assert "jobs: Track2GpuJob[];" in block
    assert "{ name: string; start_time: string | null; own_pipeline: boolean }[]" not in block


def test_the_server_sends_the_fields_it_needs():
    import inspect
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from ppe_pipeline_api import track2_shared as t2

    code = inspect.getsource(t2.get_gpu_resource_usage)
    assert '"start_time"' in code
    assert '"spot"' in code


# --- what must not have broken ------------------------------------------------------------

def test_the_three_badge_states_survive():
    code = _flat()
    for label in ("GPU unknown", "GPU free", "GPU in use -- this pipeline",
                  "GPU in use -- another job"):
        assert label in code, label


def test_the_bar_still_polls_faster_while_busy():
    """A minute counter that updates every twenty seconds is a minute
    counter that is wrong most of the time."""
    code = _flat()
    assert "live ? 3000 : 20000" in code


# --- where start_time comes from --------------------------------------------------------

def test_the_start_time_is_read_from_the_describe_not_the_summary():
    """`TrainingJobSummary` carries CreationTime, TrainingEndTime,
    LastModifiedTime and TrainingJobStatus -- and NO TrainingStartTime.
    Reading it from there returned None for every job, always.

    Verified against the live account: a job training since 18:44:04 had
    `start_time: None` in this list, so the new timer reported "waiting
    for capacity" for a run seventeen minutes into training.
    """
    import inspect
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from ppe_pipeline_api import track2_shared as t2

    code = inspect.getsource(t2.get_gpu_resource_usage)
    assert 'start_time = details.get("TrainingStartTime")' in code
    assert 'start_time = s.get("TrainingStartTime")' not in code


def test_a_started_job_reports_started(monkeypatch):
    import datetime as _dt
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from ppe_pipeline_api import track2_shared as t2

    began = _dt.datetime(2026, 10, 15, 10, 44, 4, tzinfo=_dt.timezone.utc)
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        # Exactly the shape AWS returns: no TrainingStartTime.
        "TrainingJobSummaries": [{
            "TrainingJobName": "j1",
            "CreationTime": began,
            "TrainingJobStatus": "InProgress",
        }]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": t2.GPU_INSTANCE_TYPE},
        "TrainingStartTime": began})
    job = t2.get_gpu_resource_usage()[0]
    assert job["started"] is True
    assert job["start_time"] == began.isoformat()


def test_a_job_with_no_instance_still_reports_none(monkeypatch):
    """The case the field is actually for."""
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "j1"}]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": t2.GPU_INSTANCE_TYPE}})
    job = t2.get_gpu_resource_usage()[0]
    assert job["started"] is False
    assert job["start_time"] is None
