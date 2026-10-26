"""'Every N days' or 'every N days from this date'.

From a date, the pulls (and the training clock mirrored from them) sit
on the grid that starts at that date, whenever Apply is pressed.
"""
import datetime as dt
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest, ModeRequest  # noqa: E402

SGT = dt.timezone(dt.timedelta(hours=8))
NOW = dt.datetime(2026, 10, 20, 12, 0, tzinfo=SGT).astimezone(dt.timezone.utc)
BASE = {"week_cutoff": "2026-09-01T02:00:00+08:00", "schedule_every_days": 7,
        "schedule_time": "16:35"}


def _sgt(iso):
    return dt.datetime.fromisoformat(iso).astimezone(SGT)


def test_today_counts_from_the_next_time_of_day():
    times = [_sgt(t) for t in pc.next_pull_times({**BASE, "schedule_from": "today"}, count=2, now=NOW)]
    assert times[0] == dt.datetime(2026, 10, 20, 16, 35, tzinfo=SGT)
    assert times[1] - times[0] == dt.timedelta(days=7)


def test_from_a_past_date_keeps_that_dates_grid():
    cfg = {**BASE, "schedule_from": "date", "schedule_start_date": "2026-10-01"}
    times = [_sgt(t) for t in pc.next_pull_times(cfg, count=3, now=NOW)]
    # 1 Oct + 7k: 1, 8, 15, 22 Oct -- the first still ahead is the 22nd.
    assert [t.day for t in times] == [22, 29, 5]
    assert all((t.hour, t.minute) == (16, 35) for t in times)


def test_from_a_future_date_starts_on_it():
    cfg = {**BASE, "schedule_from": "date", "schedule_start_date": "2026-11-02"}
    first = _sgt(pc.next_pull_times(cfg, count=1, now=NOW)[0])
    assert first == dt.datetime(2026, 11, 2, 16, 35, tzinfo=SGT)


def test_a_bad_date_falls_back_to_today():
    cfg = {**BASE, "schedule_from": "date", "schedule_start_date": "not-a-date"}
    assert _sgt(pc.next_pull_times(cfg, count=1, now=NOW)[0]).day == 20


def test_the_training_clock_keeps_the_same_grid():
    plan = {"every_days": 7, "at": "16:35", "from_date": "2026-10-01"}
    runs = [t.astimezone(SGT) for t in rs.next_runs(plan, start=NOW, count=2)]
    assert [r.day for r in runs] == [22, 29]
    # ...and the due-slot logic sees the 15th as the latest passed slot.
    passed = rs._passed_slots(plan, NOW)
    assert passed and passed[-1].astimezone(SGT).day == 15


def test_set_mode_refuses_from_date_without_a_date(monkeypatch):
    monkeypatch.setattr(pc, "read_config", lambda: dict(BASE))
    with pytest.raises(ValueError, match="Pick the date"):
        pc.set_mode(mode="schedule", schedule_from="date", schedule_start_date="")
    with pytest.raises(ValueError, match="not a date"):
        pc.set_mode(mode="schedule", schedule_from="date", schedule_start_date="31/12/2026")


def test_it_is_a_saved_setting():
    assert pc.DEFAULTS["schedule_from"] == "today" and pc.DEFAULTS["schedule_start_date"] == ""
    assert ConfigRequest(schedule_from="date", schedule_start_date="2026-10-01").model_dump(
        exclude_none=True) == {"schedule_from": "date", "schedule_start_date": "2026-10-01"}
    assert ModeRequest(mode="schedule", schedule_from="date").schedule_from == "date"


def test_the_bar_has_the_radio_and_the_date():
    bar = (ROOT / "web" / "src" / "components" / "Track2PinnedBar.tsx").read_text(encoding="utf-8")
    assert 'name="pull-count-from"' in bar and 'type="date"' in bar
    assert "from: countFrom, startDate," in bar
