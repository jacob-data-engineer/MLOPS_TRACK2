"""Setting a time and pressing Apply in that same minute must not cost a day.

Live on 2026-10-06: the time was set to 12:37, Apply reached Scheduler at
12:37:00.027, and the StartDate came back as the SEVENTH at 12:37. The run
was deferred twenty-four hours because it was twenty-seven milliseconds
late. From outside, the two times are the same minute, so nothing in the
UI could have explained it -- the question was simply "I set it at 12:37,
why did it not fire?".
"""
from __future__ import annotations

import datetime as _dt
import importlib

import pytest

pc = importlib.import_module("src.ppe_pipeline_api.pull_control")

CONFIG = {"week_cutoff": "2026-03-25T00:00:00+00:00", "schedule_time": "12:37",
          "schedule_every_days": 1}


def _sgt(day, hour, minute, second=0, micro=0):
    return _dt.datetime(2026, 10, day, hour, minute, second, micro,
                        tzinfo=pc.SGT)


def _first(now):
    got = pc.next_pull_times(CONFIG, count=1, now=now)
    return _dt.datetime.fromisoformat(got[0]).astimezone(pc.SGT)


def test_the_exact_millisecond_case_that_cost_a_day():
    """27ms late. Must be today, not the seventh."""
    got = _first(_sgt(6, 12, 37, 0, 27_000))
    assert got.date() == _dt.date(2026, 10, 6), got


def test_a_time_still_ahead_is_taken_as_given():
    got = _first(_sgt(6, 12, 30))
    assert (got.date(), got.hour, got.minute) == (_dt.date(2026, 10, 6), 12, 37)


def test_just_inside_the_grace_runs_today():
    got = _first(_sgt(6, 12, 41))                 # 4 minutes late
    assert got.date() == _dt.date(2026, 10, 6)


def test_the_rescheduled_time_is_in_the_FUTURE():
    """Scheduler rejects a StartDate in the past, and by the time the call
    lands `now` already is. Rounding up to the next minute is what makes
    the grace usable rather than a different error."""
    now = _sgt(6, 12, 37, 0, 27_000)
    assert _first(now) > now


def test_past_the_grace_it_is_tomorrow():
    """Setting 12:37 at 13:00 means tomorrow. The grace forgives a person
    pressing Apply; it must not start guessing at intent."""
    got = _first(_sgt(6, 13, 0))
    assert (got.date(), got.hour, got.minute) == (_dt.date(2026, 10, 7), 12, 37)


def test_a_morning_time_set_in_the_afternoon_is_tomorrow():
    config = {**CONFIG, "schedule_time": "09:00"}
    got = _dt.datetime.fromisoformat(
        pc.next_pull_times(config, count=1, now=_sgt(6, 16, 0))[0]).astimezone(pc.SGT)
    assert (got.date(), got.hour) == (_dt.date(2026, 10, 7), 9)


def test_the_series_keeps_its_interval_after_a_graced_first():
    """The grace moves the FIRST run only. Everything after it is still
    one interval apart, or a late Apply would shift the whole schedule."""
    got = pc.next_pull_times({**CONFIG, "schedule_every_days": 2}, count=3,
                             now=_sgt(6, 12, 37, 0, 27_000))
    times = [_dt.datetime.fromisoformat(t) for t in got]
    assert (times[1] - times[0]).days == 2
    assert (times[2] - times[1]).days == 2


def test_no_cutoff_still_returns_nothing():
    """The grace must not resurrect a schedule that cannot be armed."""
    assert pc.next_pull_times({"schedule_time": "12:37"}, count=1,
                              now=_sgt(6, 12, 30)) == []


def test_a_blank_time_still_means_the_cutoffs_own_hour():
    """One clock, not two -- the part of the old design worth keeping."""
    config = {"week_cutoff": "2026-03-25T02:15:00+00:00", "schedule_every_days": 1}
    got = _dt.datetime.fromisoformat(
        pc.next_pull_times(config, count=1, now=_sgt(6, 4, 0))[0]).astimezone(pc.SGT)
    cutoff_local = _dt.datetime.fromisoformat("2026-03-25T02:15:00+00:00").astimezone(pc.SGT)
    assert (got.hour, got.minute) == (cutoff_local.hour, cutoff_local.minute)


@pytest.mark.parametrize("late_seconds", [0, 1, 59, 299])
def test_everything_inside_the_window_is_today(late_seconds):
    now = _sgt(6, 12, 37) + _dt.timedelta(seconds=late_seconds)
    assert _first(now).date() == _dt.date(2026, 10, 6)


def test_the_boundary_is_not_off_by_one():
    """301 seconds late is tomorrow; the window is exactly what it says."""
    now = _sgt(6, 12, 37) + _dt.timedelta(seconds=301)
    assert _first(now).date() == _dt.date(2026, 10, 7)
