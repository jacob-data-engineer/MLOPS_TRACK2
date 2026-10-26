"""The pull cadence counts from NOW, not from the week cut-off.

It used to anchor the series to `week_cutoff + every` and roll forward by
whole intervals, so the pull times sat on a grid fixed by a date chosen
for a different purpose. Set "every 7 days" on 5 October with a cut-off in
March and the next pull was the 7th -- the grid happened not to include
today, and nothing on the page said why.

The cut-off still decides which WEEK a file is filed into. It has no
business deciding when the collecting happens; those are two questions
that were sharing one answer.

NOTE ON THE CLOCK USED HERE. These cases run at 17:30 against a 16:25
slot -- an hour past, not five minutes. Five was the original figure and
it is now exactly SCHEDULE_GRACE_SECONDS, where a slot that has only just
gone is treated as still wanted rather than deferred a day. Sitting a
suite about "the slot has passed" precisely on that boundary would make
every case here a test of the grace instead, and the first adjustment to
either number would silently change what the other is checking. The grace
has its own file: test_schedule_grace.py.
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402

SGT = dt.timezone(dt.timedelta(hours=8))
MARCH = "2026-03-25T00:00:00.000Z"


def cfg(every, time="16:25", cutoff=MARCH):
    return {"schedule_every_days": every, "schedule_time": time, "week_cutoff": cutoff}


def at(y, m, d, hh, mm):
    return dt.datetime(y, m, d, hh, mm, tzinfo=SGT)


class TestItCountsFromNow:
    def test_a_weekly_pull_starts_at_the_next_slot_not_a_march_grid(self):
        # The reported bug: "stuck at 07 October".
        got = pc.next_pull_times(cfg(7), count=1, now=at(2026, 10, 5, 17, 30))
        assert got[0].startswith("2026-10-06")

    def test_todays_slot_is_used_when_it_has_not_passed(self):
        got = pc.next_pull_times(cfg(1, "23:00"), count=1, now=at(2026, 10, 5, 17, 30))
        assert got[0].startswith("2026-10-05")

    def test_a_passed_slot_rolls_to_tomorrow(self):
        got = pc.next_pull_times(cfg(1, "16:25"), count=1, now=at(2026, 10, 5, 17, 30))
        assert got[0].startswith("2026-10-06")

    @pytest.mark.parametrize("every,expected", [
        (1, ["2026-10-06", "2026-10-07", "2026-10-08"]),
        (2, ["2026-10-06", "2026-10-08", "2026-10-10"]),
        (7, ["2026-10-06", "2026-10-13", "2026-10-20"]),
    ])
    def test_the_interval_is_honoured(self, every, expected):
        got = pc.next_pull_times(cfg(every), count=3, now=at(2026, 10, 5, 17, 30))
        assert [g[:10] for g in got] == expected


class TestTheTimeOfDayIsKept:
    def test_every_run_sits_at_the_chosen_time(self):
        got = pc.next_pull_times(cfg(2, "09:15"), count=4, now=at(2026, 10, 5, 17, 30))
        # 09:15 SGT is 01:15 UTC; whole-day intervals and no daylight
        # saving mean it cannot drift.
        assert all(g[11:16] == "01:15" for g in got), got


class TestTheCutOffNoLongerMovesTheCadence:
    @pytest.mark.parametrize("cutoff", [
        "2026-03-25T00:00:00.000Z",
        "2020-01-01T00:00:00.000Z",
        "2026-09-30T00:00:00.000Z",
    ])
    def test_the_series_is_the_same_whatever_the_cut_off(self, cutoff):
        got = pc.next_pull_times(cfg(7, cutoff=cutoff), count=2,
                                 now=at(2026, 10, 5, 17, 30))
        assert [g[:10] for g in got] == ["2026-10-06", "2026-10-13"]

    def test_no_cut_off_still_arms_nothing(self):
        # The cadence no longer COUNTS from the cut-off, but one is still
        # required: `set_mode` refuses a schedule without it, because the
        # week folders are measured from it and a pull would have nowhere
        # to file. Offering times for a config that cannot be armed would
        # be a schedule nobody can have.
        assert pc.next_pull_times(cfg(3, cutoff=""), count=3,
                                  now=at(2026, 10, 5, 17, 30)) == []


class TestItDoesNotFightTheGrace:
    """One boundary, two files -- pinned here so neither drifts.

    `test_schedule_grace.py` owns the behaviour; this owns the fact that
    the cadence cases are deliberately clear of it.
    """

    def test_the_cases_above_are_outside_the_grace(self):
        slot = dt.datetime(2026, 10, 5, 16, 25, tzinfo=SGT)
        used = dt.datetime(2026, 10, 5, 17, 30, tzinfo=SGT)
        assert (used - slot).total_seconds() > pc.SCHEDULE_GRACE_SECONDS

    def test_inside_the_grace_the_slot_is_still_today(self):
        got = pc.next_pull_times(cfg(1, "16:25"), count=1,
                                 now=at(2026, 10, 5, 16, 27))
        assert got[0].startswith("2026-10-05"), got
