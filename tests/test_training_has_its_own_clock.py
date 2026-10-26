"""Training on a schedule, without a pull.

"Use these folders only" switches Pull Control, Auto Training and
Retention OFF -- the Assemble card names its folders and no sweep runs.
Hanging the retrain off a pull in that mode meant:

  * a daily no-op pull existing only as a clock;
  * `even_if_nothing_moved` having to be ticked, because every pull
    moves nothing, forever;
  * `min_files` and `cooldown_minutes` unable to do anything, since both
    pace training against upload bursts that cannot happen;
  * enabling it marking the newest pull handled -- right for a backlog
    of sweeps, meaningless for a clock -- which silently cost a cycle.

`follow_pull` and its `every_days`/`at` fields were already in DEFAULTS,
documented as "the only mode wired today". This wires the other one.

NO NEW AWS RESOURCES. The two-minute `track2-app-queue-tick` already
runs and already calls into `track2_shared`; the check rides on it, and
`claim_once` -- a conditional put -- is what stops the dev container,
the stable one and the Lambda starting three runs for one slot.
"""
from __future__ import annotations

import datetime as dt
import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import queue_after_pull as qap  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402

SGT = dt.timezone(dt.timedelta(hours=8))
DAILY = {"enabled": True, "follow_pull": False, "every_days": 1, "at": "09:00"}
BY_HAND = {"retrain_schedule": DAILY, "dataset_assemble": {"only": True}}


def _at(day: int, hour: int, minute: int = 0) -> dt.datetime:
    return dt.datetime(2026, 10, day, hour, minute, tzinfo=SGT)


# --- who owns the clock ---------------------------------------------------------------

def test_building_by_hand_takes_the_pull_out_of_it():
    assert rs.follows_pull(BY_HAND) is False


def test_the_pull_keeps_the_clock_otherwise():
    assert rs.follows_pull({
        "dataset_assemble": {"only": False},
        "retrain_schedule": {"follow_pull": True},
    }) is True


def test_the_setting_still_decides_when_folders_are_not_hand_picked():
    assert rs.follows_pull({
        "dataset_assemble": {"only": False},
        "retrain_schedule": {"follow_pull": False},
    }) is False


def test_the_after_pull_chain_stands_down(monkeypatch):
    """Two triggers on one GPU slot is two ways to start a run nobody
    expected."""
    decision = qap.pending({**BY_HAND, "queue_after_pull": {"enabled": True}})
    assert decision["act"] is False
    assert "the pull is not the trigger" in decision["reason"]


def test_it_still_chains_on_a_pull_when_that_is_the_mode():
    """Removing the pull path would break anyone who uses Pull Control."""
    decision = qap.pending({
        "dataset_assemble": {"only": False},
        "retrain_schedule": {"follow_pull": True},
        "queue_after_pull": {"enabled": True},
    })
    assert "the pull is not the trigger" not in decision["reason"]


# --- when it fires ----------------------------------------------------------------------

def test_a_slot_that_has_passed_is_due():
    assert rs.due_training_run(BY_HAND, now=_at(16, 9, 30)) is not None


def test_a_slot_that_has_not_arrived_is_not_due_for_today():
    """09:00 has not happened at 08:30, and yesterday's 09:00 is 23 and a
    half hours gone -- far outside the grace. Nothing is due.

    This used to return YESTERDAY'S slot and start a run for it, which is
    the whole defect: a schedule set for a time of day could fire at any
    hour of the following day."""
    assert rs.due_training_run(BY_HAND, now=_at(16, 8, 30)) is None


def test_the_slot_it_let_go_is_named_rather_than_swallowed():
    """A schedule that quietly does nothing reads exactly like one that
    is working."""
    missed = rs.overdue_training_slot(BY_HAND, now=_at(16, 8, 30))
    assert missed is not None
    assert missed["slot"].startswith("2026-10-15")
    assert missed["minutes_late"] > 60


def test_it_catches_up_once_rather_than_once_per_missed_day():
    """A container down for a week starts one run, not seven."""
    slot = rs.due_training_run(BY_HAND, now=_at(16, 9, 30))
    assert slot == "2026-10-16T01:00:00+00:00"


def test_a_slot_older_than_one_interval_is_let_go():
    """The NEXT one is closer than the one that was missed, so firing it
    would spend GPU on a slot nobody is waiting for."""
    # Anchored on a day the grid lands on the 16th. Unanchored, a weekly
    # schedule used to be due on EVERY day at this time (F2).
    weekly = {"retrain_schedule": {**DAILY, "every_days": 7, "anchor": "2026-10-09"},
              "dataset_assemble": {"only": True}}
    assert rs.due_training_run(weekly, now=_at(16, 9, 30)) is not None


def test_every_caller_computes_the_same_slot_string():
    """Which is what makes `claim_once` able to pick one winner between
    dev, stable and the Lambda. Both readings are inside the grace, so
    both see the same slot -- a tick two minutes behind another must not
    compute a different string and start a second run."""
    a = rs.due_training_run(BY_HAND, now=_at(16, 9, 2))
    b = rs.due_training_run(BY_HAND, now=_at(16, 9, 28))
    assert a == b is not None


# --- it runs at the time it was set, not some time that day ----------------------

def test_it_is_due_at_its_own_time():
    assert rs.due_training_run(BY_HAND, now=_at(16, 9, 0)) is not None


def test_a_tick_or_two_late_still_counts():
    """What starts a run is a two-minute tick, and the container it runs
    in is replaced by deploys. Zero grace would miss on a good day."""
    assert rs.due_training_run(BY_HAND, now=_at(16, 9, 29)) is not None


def test_past_the_grace_it_is_let_go():
    assert rs.due_training_run(BY_HAND, now=_at(16, 9, 31)) is None


def test_the_grace_is_minutes_not_a_whole_interval():
    """The incident: a 15:34 slot on 18 Oct started at 10:58 the next
    morning, 11h24m late, because `every_days: 1` kept it due for a full
    24 hours. It looked like a timezone fault and was not."""
    assert rs.CATCH_UP_MINUTES <= 60


def test_the_real_incident_cannot_happen_again():
    sgt = dt.timezone(dt.timedelta(hours=8))
    cfg = {"retrain_schedule": {**DAILY, "at": "15:34"},
           "dataset_assemble": {"only": True}}
    on_time = dt.datetime(2026, 10, 18, 15, 34, tzinfo=sgt)
    next_morning = dt.datetime(2026, 10, 19, 10, 58, tzinfo=sgt)
    assert rs.due_training_run(cfg, now=on_time) == "2026-10-18T07:34:00+00:00"
    assert rs.due_training_run(cfg, now=next_morning) is None


# --- the time of day is Singapore, and the slot it produces is UTC ---------------

def test_the_time_is_read_as_singapore_and_stored_as_utc():
    """The field says SGT and the claim string is UTC, so +08:00 has to
    be applied exactly once. 15:34 SGT is 07:34 UTC."""
    cfg = {"retrain_schedule": {**DAILY, "at": "15:34"},
           "dataset_assemble": {"only": True}}
    sgt = dt.timezone(dt.timedelta(hours=8))
    slot = rs.due_training_run(
        cfg, now=dt.datetime(2026, 10, 18, 15, 34, tzinfo=sgt))
    assert slot == "2026-10-18T07:34:00+00:00"


def test_midnight_does_not_wrap_into_the_wrong_day():
    """00:30 SGT is 16:30 UTC the PREVIOUS day -- the case a fixed
    offset gets wrong if it is applied to the date as well."""
    cfg = {"retrain_schedule": {**DAILY, "at": "00:30"},
           "dataset_assemble": {"only": True}}
    sgt = dt.timezone(dt.timedelta(hours=8))
    slot = rs.due_training_run(
        cfg, now=dt.datetime(2026, 10, 18, 0, 31, tzinfo=sgt))
    assert slot == "2026-10-17T16:30:00+00:00"


def test_the_tick_reports_a_missed_run(monkeypatch):
    """`check_training_schedule` is what the two-minute tick calls, and
    "Nothing due" would be true and useless for a slot that passed."""
    import inspect as _inspect
    code = " ".join(_inspect.getsource(rs.check_training_schedule).split())
    assert "overdue_training_slot(config)" in code
    assert "was missed by" in code


# --- when it does not fire ----------------------------------------------------------------

def test_a_schedule_that_is_off_is_never_due():
    assert rs.due_training_run({
        "retrain_schedule": {**DAILY, "enabled": False},
        "dataset_assemble": {"only": True},
    }) is None


def test_a_pull_driven_schedule_is_never_due_on_this_clock():
    assert rs.due_training_run({
        "retrain_schedule": {**DAILY, "follow_pull": True},
        "dataset_assemble": {"only": False},
    }) is None


@pytest.mark.parametrize("bad", ["", "nonsense", "25:00", "9"])
def test_an_unusable_time_is_never_due(bad):
    assert rs.due_training_run({
        "retrain_schedule": {**DAILY, "at": bad},
        "dataset_assemble": {"only": True},
    }) is None


# --- one winner ---------------------------------------------------------------------------

def test_the_slot_is_claimed_before_anything_starts():
    code = inspect.getsource(rs.check_training_schedule)
    assert 'pc.claim_once("training_schedule", slot)' in code
    assert code.index("claim_once") < code.index("run_chain(")


def test_a_losing_caller_does_nothing():
    code = inspect.getsource(rs.check_training_schedule)
    assert "Another instance has this slot." in code


def test_it_never_raises_into_the_tick():
    """It runs inside the loop that also advances the queue."""
    code = inspect.getsource(rs.check_training_schedule)
    assert "except Exception as exc:" in code


def test_it_runs_the_same_chain_as_everything_else():
    code = inspect.getsource(rs.check_training_schedule)
    assert 'run_chain(config=config, trigger="schedule", submit="queue")' in code


# --- wired to tickers that already exist ------------------------------------------------------

def test_the_apps_poll_loop_calls_it():
    from ppe_pipeline_api import track2_shared as t2

    code = inspect.getsource(t2.queue_poll_loop)
    assert "retrain_schedule.check_training_schedule" in code


def test_the_two_minute_lambda_calls_it_too():
    """A schedule that only fires while somebody has the page open is
    not a schedule."""
    from ppe_pipeline_api import queue_runner

    code = inspect.getsource(queue_runner.lambda_handler)
    assert "retrain_schedule.check_training_schedule()" in code


def test_no_new_aws_resource_was_needed():
    """The tick it rides on is already ENABLED and already proven by the
    queue advancing on its own."""
    tf = pathlib.Path(__file__).resolve().parents[1] / "terraform" / "app"
    text = "\n".join(f.read_text(encoding="utf-8") for f in tf.glob("*.tf"))
    assert "training_schedule" not in text


# --- enabling does not start a run ---------------------------------------------------------------

def test_switching_it_on_claims_the_slot_that_already_passed():
    """Enable a 09:00 daily schedule at 09:30 and the look-back finds
    09:00 today. The operator pressed Save, not Run."""
    from ppe_pipeline_api.routers import track2_retention as r

    code = inspect.getsource(r.save_retrain)
    assert 'pc.claim_once("training_schedule", slot)' in code
    assert 'if patch.get("enabled") and not current.get("enabled"):' in code
