"""An "every N days" training clock stays on ONE grid, whenever it is asked.

The defect (F2): with no `from_date`, `next_runs` started counting from
whatever moment it was handed, and `_passed_slots` handed it "now minus
one interval". The grid therefore slid forward with the clock, and every
day shortly after the configured time produced a NEW slot string -- so
`claim_once`, which dedups one string, let a weekly schedule queue a GPU
run every day. Evidence: every_days=7, at=16:35, ticks at 20/21/22 Oct
16:36 SGT each reported that same day's 16:35 as due.

Now every slot is counted from a fixed anchor:

  * `from_date`, when the operator picked one, wins;
  * else the stored `anchor` (written when the schedule is switched on or
    its cadence is changed -- the date of the first occurrence still
    ahead, so switching on never fires on the spot);
  * else (a schedule saved before anchors existed) a fixed epoch,
    1970-01-01 SGT -- stable immediately, with no write.
"""
from __future__ import annotations

import copy
import datetime as dt
import importlib
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402

SGT = dt.timezone(dt.timedelta(hours=8))
UTC = dt.timezone.utc
EPOCH = dt.date(1970, 1, 1)


def _cfg(**plan) -> dict:
    """Training on its own clock: folders only, so the pull is not it."""
    base = {"enabled": True, "follow_pull": False, "every_days": 7, "at": "16:35"}
    return {"retrain_schedule": {**base, **plan}, "dataset_assemble": {"only": True}}


def _sgt(day: dt.date, hour: int, minute: int) -> dt.datetime:
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=SGT)


def _slot(day: dt.date, at: str) -> str:
    hour, minute = (int(p) for p in at.split(":"))
    return _sgt(day, hour, minute).astimezone(UTC).isoformat(timespec="seconds")


def _tick_all(config: dict, first: dt.date, days: int, every_minutes: int = 2):
    """Every two-minute tick over `days` days: {slot: [ticks it was due on]}."""
    due: dict = {}
    moment = _sgt(first, 0, 0)
    end = moment + dt.timedelta(days=days)
    while moment < end:
        slot = rs.due_training_run(config, now=moment)
        if slot is not None:
            due.setdefault(slot, []).append(moment)
        moment += dt.timedelta(minutes=every_minutes)
    return due


# --- the reported case ----------------------------------------------------------------

def test_the_reported_case_is_not_due_three_days_running():
    """every 7 days at 16:35, no start date: 20, 21 and 22 Oct at 16:36
    used to each report that day's 16:35 as due."""
    config = _cfg()
    slots = [rs.due_training_run(config, now=_sgt(dt.date(2026, 10, d), 16, 36))
             for d in (20, 21, 22)]
    assert sum(s is not None for s in slots) <= 1


# --- 15+ days of ticks, anchored ------------------------------------------------------

@pytest.mark.parametrize("every", [1, 2, 7])
def test_only_the_anchored_dates_are_due(every):
    anchor = dt.date(2026, 10, 20)
    config = _cfg(every_days=every, anchor=anchor.isoformat())
    first, days = dt.date(2026, 10, 18), 20
    due = _tick_all(config, first, days)

    expected = {_slot(anchor + dt.timedelta(days=every * k), "16:35")
                for k in range(0, days)
                if anchor + dt.timedelta(days=every * k) < first + dt.timedelta(days=days)}
    assert set(due) == expected
    # ...and each only inside its catch-up window.
    for slot, ticks in due.items():
        start = dt.datetime.fromisoformat(slot)
        assert all(dt.timedelta(0) <= t - start <= dt.timedelta(minutes=rs.CATCH_UP_MINUTES)
                   for t in ticks)


@pytest.mark.parametrize("every", [1, 2, 7])
def test_the_card_shows_the_slot_that_will_fire(every):
    """Display and due come from one grid: the 'next run' shown the minute
    before a slot is the slot that then fires."""
    config = _cfg(every_days=every, anchor="2026-10-20")
    plan = rs.settings(config)
    for slot in _tick_all(config, dt.date(2026, 10, 18), 16):
        before = dt.datetime.fromisoformat(slot) - dt.timedelta(minutes=1)
        assert rs.next_runs(plan, start=before, count=1)[0].isoformat(timespec="seconds") == slot


def test_nothing_is_due_before_the_anchor():
    config = _cfg(every_days=1, anchor="2026-10-25")
    due = _tick_all(config, dt.date(2026, 10, 18), 9)
    assert min(due) == _slot(dt.date(2026, 10, 25), "16:35")


# --- legacy: no anchor stored ---------------------------------------------------------

@pytest.mark.parametrize("every", [1, 2, 7])
def test_a_legacy_schedule_is_stable_without_a_write(every):
    config = _cfg(every_days=every)
    due = _tick_all(config, dt.date(2026, 10, 18), 16)
    days = sorted(dt.datetime.fromisoformat(s).astimezone(SGT).date() for s in due)
    assert days, "a 16-day window holds at least one slot for every <= 7"
    assert all((d - EPOCH).days % every == 0 for d in days)
    assert all((b - a).days == every for a, b in zip(days, days[1:]))


def test_a_legacy_weekly_schedule_does_not_fire_on_its_first_tick():
    """Upgrading must not start a run: 16:36 on a day off the grid is
    nothing, where the old look-back said 'due'."""
    config = _cfg(every_days=7)
    off_grid = next(dt.date(2026, 10, 20) + dt.timedelta(days=i) for i in range(7)
                    if (dt.date(2026, 10, 20) + dt.timedelta(days=i) - EPOCH).days % 7)
    assert rs.due_training_run(config, now=_sgt(off_grid, 16, 36)) is None


def test_a_legacy_schedule_shows_the_same_next_run_all_week():
    plan = rs.settings(_cfg(every_days=7))
    nexts = {rs.next_runs(plan, start=_sgt(dt.date(2026, 10, 15), 17, 0)
                          + dt.timedelta(hours=h), count=1)[0]
             for h in range(0, 24 * 6, 5)}
    assert len(nexts) == 1


# --- restart and two callers ----------------------------------------------------------

def test_a_restart_computes_the_same_slot_strings():
    config = _cfg(every_days=2, anchor="2026-10-20")
    before = _tick_all(config, dt.date(2026, 10, 18), 15, every_minutes=10)
    fresh = importlib.reload(rs)
    after = {}
    moment = _sgt(dt.date(2026, 10, 18), 0, 0)
    while moment < _sgt(dt.date(2026, 11, 2), 0, 0):
        slot = fresh.due_training_run(config, now=moment)
        if slot:
            after.setdefault(slot, []).append(moment)
        moment += dt.timedelta(minutes=10)
    assert set(before) == set(after)


def test_a_fresh_process_computes_the_same_slot_string():
    config = _cfg(every_days=7)   # legacy, the case most likely to drift
    moment = _sgt(dt.date(2026, 10, 22), 16, 40)
    here = rs.due_training_run(config, now=moment)
    code = (
        "import sys, datetime as dt; sys.path.insert(0, r'%s');"
        "from ppe_pipeline_api import retrain_schedule as rs;"
        "print(rs.due_training_run(%r, now=dt.datetime.fromisoformat(%r)))"
        % (ROOT / "src", config, moment.isoformat()))
    there = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
    assert there == str(here)


def test_the_app_and_the_lambda_agree_at_the_same_instant(monkeypatch):
    """Two callers, one slot string, one winner, one run."""
    config = _cfg(every_days=7, anchor="2026-10-20")
    moment = _sgt(dt.date(2026, 10, 27), 16, 37)
    app = rs.due_training_run(copy.deepcopy(config), now=moment)
    lam = rs.due_training_run(copy.deepcopy(config), now=moment)
    assert app == lam == _slot(dt.date(2026, 10, 27), "16:35")

    claimed: set = set()
    started: list = []
    monkeypatch.setattr(rs, "_now", lambda: moment.astimezone(UTC))
    monkeypatch.setattr(pc, "claim_once",
                        lambda what, key, **kw: not (key in claimed or claimed.add(key)))
    monkeypatch.setattr(rs, "run_chain",
                        lambda **kw: started.append(1) or {"ok": True, "steps": []})
    first = rs.check_training_schedule(copy.deepcopy(config))
    second = rs.check_training_schedule(copy.deepcopy(config))
    assert first["started"] is True and first["slot"] == app
    assert second == {"started": False, "reason": "Another instance has this slot."}
    assert started == [1]


# --- midnight, from_date, missed ------------------------------------------------------

def test_crossing_midnight_sgt_keeps_the_sgt_date():
    """00:10 SGT is 16:10 UTC the day before; the grid counts SGT days."""
    config = _cfg(every_days=2, at="00:10", anchor="2026-10-21")
    due = _tick_all(config, dt.date(2026, 10, 18), 16)
    days = sorted(dt.datetime.fromisoformat(s).astimezone(SGT).date() for s in due)
    assert days[0] == dt.date(2026, 10, 21)
    assert all((d - dt.date(2026, 10, 21)).days % 2 == 0 for d in days)
    assert min(due) == "2026-10-20T16:10:00+00:00"


def test_an_explicit_start_date_wins_over_the_anchor():
    config = _cfg(every_days=7, from_date="2026-10-01", anchor="2026-10-20")
    due = _tick_all(config, dt.date(2026, 10, 10), 20)
    assert set(due) == {_slot(dt.date(2026, 10, d), "16:35") for d in (15, 22, 29)}


def test_a_slot_missed_past_the_window_is_let_go_not_run():
    config = _cfg(every_days=2, anchor="2026-10-20")
    late = _sgt(dt.date(2026, 10, 22), 16, 35) + dt.timedelta(minutes=rs.CATCH_UP_MINUTES + 1)
    assert rs.due_training_run(config, now=late) is None
    missed = rs.overdue_training_slot(config, now=late)
    assert missed["slot"] == _slot(dt.date(2026, 10, 22), "16:35")
    # The next one is the next grid slot, not a deferred copy of this one.
    assert rs.due_training_run(config, now=_sgt(dt.date(2026, 10, 23), 16, 40)) is None
    assert rs.due_training_run(config, now=_sgt(dt.date(2026, 10, 24), 16, 40)) \
        == _slot(dt.date(2026, 10, 24), "16:35")


# --- writing the anchor ---------------------------------------------------------------

def test_a_fresh_anchor_is_the_next_occurrence():
    before = _sgt(dt.date(2026, 10, 20), 16, 0)
    after = _sgt(dt.date(2026, 10, 20), 16, 36)
    assert rs.fresh_anchor({"at": "16:35"}, now=before) == "2026-10-20"
    assert rs.fresh_anchor({"at": "16:35"}, now=after) == "2026-10-21"


def test_saving_other_fields_keeps_the_grid():
    now = _sgt(dt.date(2026, 10, 22), 9, 0)
    old = {**rs.DEFAULTS, "enabled": True, "every_days": 7, "at": "16:35", "anchor": "2026-10-20"}
    assert rs.with_anchor(old, {**old, "epochs": 80}, now=now)["anchor"] == "2026-10-20"


def test_changing_the_cadence_resets_the_grid():
    now = _sgt(dt.date(2026, 10, 22), 17, 0)
    old = {**rs.DEFAULTS, "enabled": True, "every_days": 7, "at": "16:35", "anchor": "2026-10-20"}
    assert rs.with_anchor(old, {**old, "every_days": 3}, now=now)["anchor"] == "2026-10-23"
    assert rs.with_anchor(old, {**old, "at": "18:00"}, now=now)["anchor"] == "2026-10-22"


def test_a_legacy_save_pins_the_grid_it_was_already_showing():
    """No visible jump: the anchor written is the legacy grid's own next date."""
    now = _sgt(dt.date(2026, 10, 22), 9, 0)
    old = {**rs.DEFAULTS, "enabled": True, "every_days": 7, "at": "16:35"}
    new = rs.with_anchor(old, {**old, "epochs": 80}, now=now)
    assert new["anchor"]
    assert rs.next_runs(new, start=now, count=3) == rs.next_runs(old, start=now, count=3)


def _enable_through_the_route(monkeypatch, store: dict, now: dt.datetime):
    from ppe_pipeline_api.routers import track2_retention as r

    monkeypatch.setattr(pc, "read_config", lambda: copy.deepcopy(store))
    monkeypatch.setattr(pc, "write_config", lambda patch: store.update(copy.deepcopy(patch)))
    claims: list = []
    monkeypatch.setattr(pc, "claim_once", lambda what, key, **kw: claims.append(key) or True)
    monkeypatch.setattr(rs, "_now", lambda: now.astimezone(UTC))
    monkeypatch.setattr(rs, "preflight", lambda config=None: {"settings": rs.settings(config)})
    out = r.save_retrain(r.RetrainScheduleRequest(enabled=True), session=None)
    return out, claims


def test_enabling_just_after_its_time_today_does_not_fire(monkeypatch):
    store = {"dataset_assemble": {"only": True},
             "retrain_schedule": {"enabled": False, "follow_pull": False,
                                  "every_days": 7, "at": "16:35"}}
    now = _sgt(dt.date(2026, 10, 22), 16, 36)
    out, _ = _enable_through_the_route(monkeypatch, store, now)
    assert store["retrain_schedule"]["anchor"] == "2026-10-23"
    for minutes in range(0, 60, 2):
        assert rs.due_training_run(store, now=now + dt.timedelta(minutes=minutes)) is None
    assert out["next_runs"][0] == _slot(dt.date(2026, 10, 23), "16:35")


def test_enabling_on_a_start_date_grid_claims_the_slot_just_passed(monkeypatch):
    """from_date wins over the anchor, so the route's claim guard is what
    keeps Save from being Run there."""
    store = {"dataset_assemble": {"only": True},
             "retrain_schedule": {"enabled": False, "follow_pull": False, "every_days": 7,
                                  "at": "16:35", "from_date": "2026-10-01"}}
    now = _sgt(dt.date(2026, 10, 22), 16, 36)
    _, claims = _enable_through_the_route(monkeypatch, store, now)
    assert claims == [_slot(dt.date(2026, 10, 22), "16:35")]


class _NoAws:
    def __getattr__(self, name):
        from botocore.exceptions import ClientError

        def refuse(**kw):
            raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, name)
        return refuse


def _set_mode(monkeypatch, store: dict, now: dt.datetime, **kw):
    monkeypatch.setattr(pc, "read_config", lambda: copy.deepcopy(store))
    monkeypatch.setattr(pc, "write_config", lambda patch: store.update(copy.deepcopy(patch)))
    claims: list = []
    monkeypatch.setattr(pc, "claim_once", lambda what, key, **k: claims.append(key) or True)
    monkeypatch.setattr(pc, "events", lambda: _NoAws())
    monkeypatch.setattr(pc, "scheduler", lambda: _NoAws())
    monkeypatch.setattr(pc, "trigger_status", lambda: {})
    monkeypatch.setattr(pc, "schedule_status", lambda: {})
    monkeypatch.setattr(pc, "schedule_plan", lambda config=None, now=None: {})
    monkeypatch.setattr(rs, "_now", lambda: now.astimezone(UTC))
    pc.set_mode(**kw)
    return claims


def test_the_pull_strip_writes_the_anchor_when_it_arms_the_clock(monkeypatch):
    store = {"dataset_assemble": {"only": True}, "mode": "off",
             "retrain_schedule": {"enabled": False, "follow_pull": False}}
    now = _sgt(dt.date(2026, 10, 22), 16, 36)
    _set_mode(monkeypatch, store, now, mode="schedule",
              schedule_every_days=7, schedule_time="16:35")
    assert store["retrain_schedule"]["anchor"] == "2026-10-23"
    assert rs.due_training_run(store, now=now) is None

    # Pressing Apply again the next day, unchanged, keeps the grid.
    _set_mode(monkeypatch, store, now + dt.timedelta(days=1, hours=1), mode="schedule",
              schedule_every_days=7, schedule_time="16:35")
    assert store["retrain_schedule"]["anchor"] == "2026-10-23"

    # A new interval is a deliberate reset.
    _set_mode(monkeypatch, store, now + dt.timedelta(days=2), mode="schedule",
              schedule_every_days=3, schedule_time="16:35")
    assert store["retrain_schedule"]["anchor"] == "2026-10-25"


def test_the_pull_strip_claims_a_passed_start_date_slot(monkeypatch):
    store = {"dataset_assemble": {"only": True}, "mode": "off",
             "retrain_schedule": {"enabled": False, "follow_pull": False}}
    now = _sgt(dt.date(2026, 10, 22), 16, 36)
    claims = _set_mode(monkeypatch, store, now, mode="schedule", schedule_every_days=7,
                       schedule_time="16:35", schedule_from="date",
                       schedule_start_date="2026-10-01")
    assert claims == [_slot(dt.date(2026, 10, 22), "16:35")]
