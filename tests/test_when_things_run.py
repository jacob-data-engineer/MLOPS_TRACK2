""""When does it next run" and "when did these run" both have answers.

The strip said

    next run: rate(1 day)

which is a sentence about what it does forever, not when it does it
next -- and that is the only question somebody who just pressed Apply is
asking. The EventBridge rule's own `next_run` is often empty, so the
fallback was all anybody saw. The TRAINING clock always computes one
from the cadence, and that is the thing this strip now arms.

The cost panel listed a duration and a cost for every job and no clock
time at all, so "what was the GPU doing at three o'clock" could not be
answered from the one card that lists every job.
"""
from __future__ import annotations

import datetime as dt
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402

BAR = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "components" / "Track2PinnedBar.tsx")
PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2TrainingCostToday.tsx")


def _flat(path: pathlib.Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


# --- the next run ---------------------------------------------------------------------

def test_the_strip_reads_the_training_clock():
    code = _flat(BAR)
    assert "getTrack2Retrain().catch(() => null)" in code
    # Its own clock only (2026-10-29): following the pull, the pull's
    # date is the answer.
    assert 'retrain.trigger === "clock" ? retrain.next_runs?.[0] || "" : ""' in code


def test_the_training_clock_wins_over_the_rule():
    """It is always computable; the rule's own next_run often is not."""
    code = _flat(BAR)
    assert "setTrainNext(" in code


def test_the_rate_expression_is_still_the_last_resort():
    """Better than nothing when neither clock has a date -- but it is
    the fallback now, not the usual answer."""
    code = _flat(BAR)
    assert "rule.next_run || rule.schedule" in code


def test_an_unarmed_clock_contributes_no_date():
    code = _flat(BAR)
    assert 'retrain.trigger === "clock" ? retrain.next_runs?.[0] || "" : ""' in code


# --- when the jobs ran -----------------------------------------------------------------

def test_the_cost_rows_carry_their_clock_times():
    code = inspect.getsource(t2.training_cost_today)
    assert '"started_at": began.isoformat() if began else None' in code
    assert '"ended_at": ended.isoformat() if ended else None' in code


def test_training_times_come_from_the_describe():
    code = inspect.getsource(t2.training_cost_today)
    assert 'began=described.get("TrainingStartTime")' in code
    assert 'ended=described.get("TrainingEndTime")' in code


def test_processing_times_do_too():
    code = inspect.getsource(t2.training_cost_today)
    assert 'ended=described.get("ProcessingEndTime")' in code


def test_the_times_are_recorded_for_real_jobs(monkeypatch):
    began = dt.datetime(2026, 10, 16, 6, 2, tzinfo=dt.timezone.utc)
    done = dt.datetime(2026, 10, 16, 6, 34, tzinfo=dt.timezone.utc)
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "j1"}]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
        "BillableTimeInSeconds": 1920,
        "TrainingJobStatus": "Completed",
        "TrainingStartTime": began, "TrainingEndTime": done})
    monkeypatch.setattr(t2.sm, "list_processing_jobs", lambda **kw: {
        "ProcessingJobSummaries": []})
    monkeypatch.setattr(t2, "sagemaker_hourly_rate", lambda i, c: 1.0)

    row = t2.training_cost_today()["jobs"][0]
    assert row["started_at"] == began.isoformat()
    assert row["ended_at"] == done.isoformat()


def test_a_job_with_no_instance_yet_has_no_times(monkeypatch):
    """Billing starts at `started_at`; a job waiting on capacity has
    neither, and inventing one would be a number somebody reads."""
    monkeypatch.setattr(t2.sm, "list_training_jobs", lambda **kw: {
        "TrainingJobSummaries": [{"TrainingJobName": "j1"}]})
    monkeypatch.setattr(t2.sm, "describe_training_job", lambda TrainingJobName: {
        "ResourceConfig": {"InstanceType": "ml.g4dn.xlarge"},
        "TrainingJobStatus": "InProgress"})
    monkeypatch.setattr(t2.sm, "list_processing_jobs", lambda **kw: {
        "ProcessingJobSummaries": []})
    monkeypatch.setattr(t2, "sagemaker_hourly_rate", lambda i, c: 1.0)

    row = t2.training_cost_today()["jobs"][0]
    assert row["started_at"] is None
    assert row["ended_at"] is None


# --- and the panel shows them -------------------------------------------------------------

def test_the_panel_renders_a_clock_range():
    code = _flat(PANEL)
    assert "function clockRange(job:" in code
    assert "{clockRange(job)}" in code


def test_it_is_rendered_in_the_readers_own_timezone():
    """ISO with an offset, formatted by the browser. A server that
    formatted it would be guessing which zone to use."""
    code = _flat(PANEL)
    assert "new Date(iso).toLocaleTimeString(undefined," in code


def test_a_running_job_reads_as_still_going():
    code = _flat(PANEL)
    assert "`${from}-now`" in code


def test_a_job_waiting_on_capacity_says_so():
    code = _flat(PANEL)
    assert 'if (!job.started_at) return "waiting";' in code


def test_the_time_comes_before_the_duration():
    """"When" is what the list is scanned for; "how long" is detail."""
    code = _flat(PANEL)
    assert code.index("{clockRange(job)}") < code.index("{Math.round(job.seconds / 60)}m")


def test_the_client_type_declares_the_fields():
    api = (pathlib.Path(__file__).resolve().parents[1]
           / "web" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "started_at?: string | null;" in api
    assert "ended_at?: string | null;" in api


# --- where the cost panel lives -----------------------------------------------------

def test_the_cost_panel_sits_inside_the_pinned_bar():
    """"What is the GPU doing" and "what has it cost today" are one
    glance, so the panel travels with the always-visible strip rather
    than sitting below it as a separate block."""
    bar = BAR.read_text(encoding="utf-8")
    assert "<Track2TrainingCostToday />" in bar
    assert 'import { Track2TrainingCostToday } from "./Track2TrainingCostToday";' in bar


def test_it_is_directly_under_the_gpu_line():
    bar = " ".join(BAR.read_text(encoding="utf-8").split())
    assert bar.index("<Track2ChainStatusBar />") < bar.index("<Track2TrainingCostToday />")


def test_the_page_no_longer_renders_it_separately():
    """Two copies would poll the same endpoint twice and disagree about
    the read time."""
    page = (pathlib.Path(__file__).resolve().parents[1]
            / "web" / "src" / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    assert "<Track2TrainingCostToday />" not in page
    assert "Track2TrainingCostToday" not in page.split("export function Track2ModelsPage")[0]


# --- the zone the reader is shown -------------------------------------------------

def test_the_next_run_is_rendered_in_singapore_time():
    """It used to be built from `getHours()` -- the viewer's own clock --
    and then have a literal " SGT" appended. Right on a laptop already
    set to Singapore, quietly hours out on any other, and the label made
    it uncheckable either way."""
    code = _flat(BAR)
    assert 'import { formatDayTime } from "../lib/format";' in code
    assert "const savedWhen = formatDayTime;" in code
    # The prose above the helper still NAMES getHours, which is the
    # point of it -- what must be gone is the call.
    assert "pad(d.getHours())" not in code
    assert "d.getMinutes()" not in code


def test_the_helper_carries_its_own_zone_so_it_cannot_be_mislabelled():
    code = _flat(BAR)
    assert "? savedWhen(trainNext)" in code and "? savedWhen(pullNext)" in code
    assert "SGT`" not in code


def test_the_input_is_still_labelled_sgt():
    """The field IS interpreted as Singapore time by the backend, so
    saying so beside the box is correct and has to stay."""
    assert "SGT" in BAR.read_text(encoding="utf-8")


def test_the_models_page_shows_server_instants_in_the_same_zone():
    """Snapshot frozen, dataset built, last pull, next run -- all
    computed by the server. Two people reading one page side by side
    must not see different times for the same event."""
    page = " ".join((pathlib.Path(__file__).resolve().parents[1]
                     / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
                    .read_text(encoding="utf-8").split())
    assert "function localDate(iso: string): string { return iso ? formatDayTime(iso) : \"\"; }" in page
    assert "at.toLocaleString(undefined," not in page


def test_the_formatter_is_pinned_to_singapore_not_the_browser():
    fmt = " ".join((pathlib.Path(__file__).resolve().parents[1]
                    / "web" / "src" / "lib" / "format.ts")
                   .read_text(encoding="utf-8").split())
    assert "export function formatDayTime" in fmt
    assert 'const SGT_SHORT = new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Singapore"' in fmt
    assert "return `${p.day} ${p.month} ${p.hour}:${p.minute} SGT`;" in fmt


def test_no_screen_prints_a_raw_utc_timestamp_as_a_time():
    """The snapshot list showed 05:05 for a snapshot the line above it
    called 13:05 SGT -- the raw UTC string, sliced. Every server instant
    goes through formatDayTime / localDate, which name the zone."""
    web = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
    offenders = []
    for f in list((web / "components").glob("*.tsx")) + list((web / "routes").glob("*.tsx")):
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if '.replace("T", " ").slice(0, 1' in line and "SGT_OFFSET_MS" not in line:
                offenders.append(f"{f.name}:{n}")
    assert not offenders, offenders


def test_each_snapshot_has_a_red_delete_with_a_confirm_dialog():
    page = " ".join((pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "routes"
                     / "Track2ModelsPage.tsx").read_text(encoding="utf-8").split())
    assert 'className="rounded bg-red-600 px-2 py-0.5 text-xs font-medium text-white' in page
    assert "setSnapToDelete(snap)" in page
    assert "await deleteTrack2Snapshot(snapToDelete.snapshot_id);" in page
    assert "The images and labels are not touched" in page
    assert "This cannot be undone." in page
