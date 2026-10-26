"""Queueing a retrain once a pull has finished."""

from __future__ import annotations

import pytest

from ppe_pipeline_api import queue_after_pull as qap


# THE REAL SHAPE of a `run_state` row, as `write_run_state` writes it at
# the end of every sweep. It DOES carry `finished_at` and `status` --
# unlike the `event` rows, which do not.
#
# Which row type this reads is the whole story of two bugs:
#   1. it first read `event` rows but checked `finished_at`, a field only
#      `run_state` carries, so nothing ever fired;
#   2. corrected to `event` rows without that check, it then could not
#      see a scheduled sweep that moved nothing, because `_record_run`
#      deliberately does not write one -- making `even_if_nothing_moved`
#      unreachable for the scheduled trigger.
# Reading `run_state` fixes both: written for every sweep, and carrying
# the completion fields.
#
# `trigger` is part of the shape and not decoration: only a SCHEDULED
# sweep chains. `follow_pull` always said so -- "a manual Pull now and
# the file-arrival trigger do not, because those fire unpredictably often
# and each run of this costs GPU hours" -- but nothing enforced it, and
# on 2026-10-06 an ARRIVAL pull that moved ONE file started a full GPU
# training run.
DONE = {"run_id": "r2", "finished_at": "2026-09-30T01:00:00+00:00",
        "status": "done", "trigger": "schedule", "moved": 40, "skipped": 0,
        "errors": 0, "aborted": False}
EMPTY = {"run_id": "r3", "finished_at": "2026-09-30T02:00:00+00:00",
         "status": "done", "trigger": "schedule", "moved": 0, "skipped": 0,
         "errors": 0, "aborted": False}
RUNNING = {"run_id": "r4", "status": None, "trigger": "schedule", "moved": 5}
ABORTED = {"run_id": "r5", "finished_at": "2026-09-30T03:00:00+00:00",
           "status": "failed", "trigger": "schedule", "moved": 12, "errors": 3,
           "aborted": True}
ARRIVED = {"run_id": "r6", "finished_at": "2026-09-30T04:00:00+00:00",
           "status": "done", "trigger": "arrival", "moved": 1, "skipped": 0,
           "errors": 0, "aborted": False}
BY_HAND = {"run_id": "r7", "finished_at": "2026-09-30T05:00:00+00:00",
           "status": "done", "trigger": "manual", "moved": 9, "skipped": 0,
           "errors": 0, "aborted": False}


@pytest.fixture
def runs(monkeypatch):
    """The sweep history, newest first, as the card would see it."""
    history: list = []
    monkeypatch.setattr(qap.pc, "recent_run_states", lambda limit=20: history[:limit])
    # `check` now claims the run before chaining, so dev and stable cannot
    # both queue it. Granted here; the claim has its own tests, and the
    # alternative is every test in this file reaching DynamoDB.
    monkeypatch.setattr(qap.pc, "claim_once", lambda what, key, **kw: True)
    # The durable marker. `pending` asks it rather than trusting
    # `last_run_id` on the config blob, which any concurrent writer can
    # revert -- and did, live, for four consecutive sweeps.
    monkeypatch.setattr(qap.pc, "claim_exists", lambda what, key: False)
    # The defaults, not the LIVE config. `_finished_pulls()` reads the
    # chaining triggers when no plan is passed, and this test used to
    # pass or fail on whatever the real Pull Control card was set to:
    # it broke the day the live triggers were changed to arrival-only.
    monkeypatch.setattr(qap.pc, "read_config", lambda *a, **kw: {})
    return history


def test_an_arrival_pull_does_NOT_chain_by_default(runs):
    """One uploaded file is not a reason to spend GPU hours.

    Which triggers may chain is a setting now, but the DEFAULT is still
    schedule-only -- so a settings file written before that setting
    existed, which has no `triggers` key at all, keeps behaving exactly
    as it did.
    """
    runs.append(ARRIVED)
    decision = qap.pending({qap.KEY: {"enabled": True, "even_if_nothing_moved": True}})
    assert decision["act"] is False
    assert "trigger that may chain" in decision["reason"]
    assert "schedule" in decision["reason"]


def test_an_arrival_pull_chains_once_arrival_is_ticked(runs):
    """What the setting is for: arrival can now be chosen deliberately."""
    runs.append(ARRIVED)
    decision = qap.pending({qap.KEY: {
        "enabled": True, "triggers": ["schedule", "arrival"]}})
    assert decision["act"] is True, decision["reason"]
    assert decision["run"]["run_id"] == ARRIVED["run_id"]


def test_a_small_arrival_is_held_back_by_min_files(runs):
    """The guard that makes arrival safe to tick. An arrival pull fires
    per upload burst, and on 2026-10-06 one that moved a SINGLE file
    started a full GPU run."""
    runs.append({**ARRIVED, "moved": 1})
    decision = qap.pending({qap.KEY: {
        "enabled": True, "triggers": ["arrival"], "min_files": 25}})
    assert decision["act"] is False
    assert "below the 25" in decision["reason"]


def test_a_real_batch_passes_min_files(runs):
    runs.append({**ARRIVED, "moved": 400})
    decision = qap.pending({qap.KEY: {
        "enabled": True, "triggers": ["arrival"], "min_files": 25}})
    assert decision["act"] is True, decision["reason"]


def test_the_cooldown_caps_a_burst(runs):
    """Twenty uploads in ten minutes must not become twenty runs, however
    big each individual batch is."""
    import datetime as _dt

    runs.append({**ARRIVED, "moved": 400})
    just_now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    decision = qap.pending({qap.KEY: {
        "enabled": True, "triggers": ["arrival"], "cooldown_minutes": 60,
        "last_result": {"at": just_now},
    }})
    assert decision["act"] is False
    assert "waits 60" in decision["reason"]


def test_the_cooldown_lets_the_next_one_through_once_it_has_passed(runs):
    import datetime as _dt

    runs.append({**ARRIVED, "moved": 400})
    long_ago = (_dt.datetime.now(_dt.timezone.utc)
                - _dt.timedelta(hours=4)).isoformat(timespec="seconds")
    decision = qap.pending({qap.KEY: {
        "enabled": True, "triggers": ["arrival"], "cooldown_minutes": 60,
        "last_result": {"at": long_ago},
    }})
    assert decision["act"] is True, decision["reason"]


def test_a_scheduled_pull_is_unaffected_by_the_arrival_guards(runs):
    """The guards exist for arrival. A schedule fires on a timer somebody
    chose, so a quiet week must still retrain."""
    runs.append(DONE)
    decision = qap.pending({qap.KEY: {"enabled": True, "min_files": 1}})
    assert decision["act"] is True, decision["reason"]


def test_a_manual_pull_does_NOT_chain(runs):
    """Pressing Pull now is somebody looking at the screen; they can press
    Run it now too if that is what they meant."""
    runs.append(BY_HAND)
    assert qap.pending({qap.KEY: {"enabled": True}})["act"] is False


def test_an_arrival_pull_does_not_mask_the_scheduled_one_behind_it(runs):
    """`pending` reads the NEWEST eligible sweep. If arrival runs were
    merely ignored at the last moment instead of filtered out, a file
    landing after a scheduled pull would hide it and the chain would
    never fire that day."""
    runs.append(DONE)        # the scheduled sweep
    runs.insert(0, ARRIVED)  # a file arrived afterwards -- newer
    decision = qap.pending({qap.KEY: {"enabled": True}})
    assert decision["act"] is True
    assert decision["run"]["run_id"] == "r2"


def test_the_durable_marker_stops_a_repeat(runs, monkeypatch):
    """The config blob is read-modify-write with no condition, so
    `last_run_id` can silently revert -- live, it sat four sweeps behind
    while `pending` answered "act" every tick. The claim row cannot."""
    runs.append(DONE)
    monkeypatch.setattr(qap.pc, "claim_exists", lambda what, key: key == "r2")
    decision = qap.pending({qap.KEY: {"enabled": True, "last_run_id": ""}})
    assert decision["act"] is False
    assert decision["reason"] == "Already queued for this pull."


def test_a_lost_claim_stops_the_chain(runs, monkeypatch):
    """The other container got there first. No GPU work, and a reason
    that says so rather than a silent False."""
    runs.append(DONE)
    monkeypatch.setattr(qap.pc, "read_config", lambda: {qap.KEY: {"enabled": True}})
    monkeypatch.setattr(qap.pc, "claim_once", lambda what, key, **kw: False)
    started: list = []
    from ppe_pipeline_api import retrain_schedule as rs
    monkeypatch.setattr(rs, "run_chain", lambda **kw: started.append(kw) or {"ok": True})
    out = qap.check()
    assert out["queued"] is False
    assert "Another instance" in out["reason"]
    assert started == [], "the chain must not run when the claim was lost"


def test_off_by_default(runs):
    runs.append(DONE)
    assert qap.settings({})["enabled"] is False
    assert qap.pending({})["act"] is False


def test_a_finished_pull_that_moved_files_is_acted_on(runs):
    runs.append(DONE)
    decision = qap.pending({qap.KEY: {"enabled": True}})
    assert decision["act"] is True
    assert decision["run"]["run_id"] == "r2"


def test_the_same_pull_is_never_queued_twice(runs):
    """The whole of the de-duplication. The watcher runs every 25
    seconds, so without this one sweep would queue a run per tick until
    somebody noticed the queue filling up."""
    runs.append(DONE)
    config = {qap.KEY: {"enabled": True, "last_run_id": "r2"}}
    decision = qap.pending(config)
    assert decision["act"] is False
    assert "Already queued" in decision["reason"]


def test_a_sweep_still_running_is_left_alone(runs):
    """Week folders still filling are not something to train on -- the
    dataset would be built half way through the pull."""
    runs.insert(0, RUNNING)
    runs.append(DONE)
    assert qap.pending({qap.KEY: {"enabled": True}})["run"]["run_id"] == "r2"


def test_an_empty_scheduled_sweep_is_visible_at_all(runs):
    """The reason this reads `run_state` and not `event`. `_record_run`
    skips a scheduled sweep that moved nothing, so on the event rows
    `even_if_nothing_moved` could never take effect for the trigger it
    was written for."""
    runs.append(EMPTY)
    assert qap._finished_pulls() == [EMPTY]
    forced = qap.pending({qap.KEY: {"enabled": True, "even_if_nothing_moved": True}})
    assert forced["act"] is True and forced["run"]["run_id"] == "r3"


def test_an_aborted_sweep_does_not_trigger_one(runs):
    """What landed is by definition not what was meant to. `aborted` IS
    on the summary, unlike the two fields this used to check."""
    runs.append(ABORTED)
    assert qap.pending({qap.KEY: {"enabled": True}})["act"] is False


def test_a_pull_that_moved_nothing_is_skipped_by_default(runs):
    """The corpus is unchanged, so the run would reproduce the model
    that already exists and bill for it."""
    runs.append(EMPTY)
    decision = qap.pending({qap.KEY: {"enabled": True}})
    assert decision["act"] is False
    assert "moved no files" in decision["reason"]

    # ...but it is a choice, not a rule.
    forced = qap.pending({qap.KEY: {"enabled": True, "even_if_nothing_moved": True}})
    assert forced["act"] is True


def test_check_records_the_run_id_even_when_the_chain_failed(runs, monkeypatch):
    """Otherwise one misconfiguration retries every 25 seconds forever,
    and the log fills with the same failure until somebody looks."""
    runs.append(DONE)
    written: dict = {}
    monkeypatch.setattr(qap.pc, "read_config",
                        lambda: {qap.KEY: {"enabled": True}})
    monkeypatch.setattr(qap.pc, "write_config", lambda patch: written.update(patch))

    from ppe_pipeline_api import retrain_schedule as rs
    monkeypatch.setattr(rs, "run_chain", lambda **kw: {
        "ok": False, "steps": [{"step": "preflight", "ok": False,
                                "detail": "No detector chosen."}]})

    out = qap.check()
    assert out["queued"] is False
    assert written[qap.KEY]["last_run_id"] == "r2"
    assert "No detector chosen" in written[qap.KEY]["last_result"]["detail"]


def test_check_queues_through_the_chain_not_a_second_implementation(runs, monkeypatch):
    """A queued entry must be built by the same code a manual run uses,
    or the two train on datasets assembled differently."""
    runs.append(DONE)
    calls: list = []
    monkeypatch.setattr(qap.pc, "read_config",
                        lambda: {qap.KEY: {"enabled": True}})
    monkeypatch.setattr(qap.pc, "write_config", lambda patch: None)

    from ppe_pipeline_api import retrain_schedule as rs

    def fake_chain(**kw):
        calls.append(kw)
        return {"ok": True, "steps": [{"step": "queue", "ok": True, "detail": "queued abc"}]}

    monkeypatch.setattr(rs, "run_chain", fake_chain)
    out = qap.check()
    assert out["queued"] is True
    assert calls[0]["submit"] == "queue"
    assert calls[0]["trigger"] == "after-pull"


def test_it_reads_the_row_type_that_covers_every_sweep():
    """The guard for the bug itself.

    `recent_runs` queries `kind = "event"`; `finished_at` and `status`
    are written by `write_run_state` under `kind = "run_state"`. Reading
    either here silently rejects every real sweep, and the failure looks
    exactly like "no pull has happened yet" -- which is why it went
    unnoticed with eight passing tests.
    """
    import inspect
    from ppe_pipeline_api import pull_control as pc

    source = inspect.getsource(qap)
    assert "recent_run_states" in source, (
        "must read the run_state rows: the event rows omit every "
        "scheduled sweep that moved nothing."
    )
    assert "pc.recent_runs(" not in source, (
        "recent_runs reads the event rows, which skip empty scheduled "
        "sweeps and carry neither finished_at nor status."
    )
    # ...and the two really are different row kinds.
    assert 'ExpressionAttributeValues={":k": {"S": "event"}}' in inspect.getsource(pc.recent_runs)
    assert '{"S": "run_state"}' in inspect.getsource(pc.recent_run_states)
    # The event writer really does drop empty scheduled sweeps.
    assert 'summary["trigger"] == "schedule"' in inspect.getsource(pc._record_run)
