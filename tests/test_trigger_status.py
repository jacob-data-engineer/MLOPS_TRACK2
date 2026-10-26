"""The card must report the trigger that is actually armed.

Two mechanisms exist for the schedule: the classic EventBridge RULE and
EventBridge SCHEDULER. `set_mode` has armed the Scheduler for a long
time, because `rate()` on a rule is anchored to whenever the rule was
last written and its cron is UTC with no start date -- neither of which
can express "first pull at this moment, then every N days" in SGT.

`trigger_status` still read the RULE. So a schedule could be enabled and
counting down while the one line on the page whose job is to say so
reported "schedule: disabled".
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402

rs_pc = pc  # the module under test, aliased for the newer cases


@pytest.fixture
def armed(monkeypatch):
    """Scheduler enabled, legacy rule disabled -- the real situation."""
    monkeypatch.setattr(pc, "schedule_status", lambda: {
        "exists": True, "enabled": True, "expression": "rate(7 days)",
        "timezone": "Asia/Singapore", "start": "", "error": "",
    })
    monkeypatch.setattr(pc, "rule_state", lambda name: {
        "name": name, "exists": True, "enabled": False, "schedule": "rate(1 day)",
    })


class TestItReportsTheScheduler:
    def test_an_enabled_scheduler_reads_as_enabled(self, armed):
        assert pc.trigger_status()["schedule"]["enabled"] is True

    def test_it_carries_the_schedulers_expression(self, armed):
        # Not the rule's stale rate(1 day).
        assert pc.trigger_status()["schedule"]["schedule"] == "rate(7 days)"

    def test_the_shape_the_card_reads_is_unchanged(self, armed):
        got = pc.trigger_status()["schedule"]
        for key in ("name", "exists", "enabled", "schedule", "error"):
            assert key in got, key

    def test_arrival_still_comes_from_its_rule(self, armed):
        # Arrival genuinely is a rule; only the schedule moved.
        assert pc.trigger_status()["arrival"]["name"] == pc.ARRIVAL_RULE

    def test_a_missing_scheduler_reads_as_not_armed(self, monkeypatch):
        monkeypatch.setattr(pc, "schedule_status", lambda: {
            "exists": False, "enabled": False, "error": "ResourceNotFoundException",
        })
        monkeypatch.setattr(pc, "rule_state", lambda name: {
            "name": name, "exists": True, "enabled": True, "schedule": "rate(1 day)",
        })
        got = pc.trigger_status()["schedule"]
        # The rule being enabled must NOT make this look armed -- that is
        # the inversion of the bug, and just as wrong.
        assert got["exists"] is False
        assert got["enabled"] is False
        assert "ResourceNotFound" in got["error"]


class TestItSaysWhenItNextFires:
    """"armed: rate(7 days)" does not say WHICH day.

    The cadence grid is counted from the week cut-off, so a time of 15:43
    set on the 5th can arm the 7th at 15:43. The card said only "armed",
    which reads as "today" to anyone who has just pressed Apply -- and
    then nothing happens for two days and the schedule looks broken.
    """

    def test_the_schedule_carries_its_next_run(self, monkeypatch):
        monkeypatch.setattr(rs_pc, "schedule_status", lambda: {
            "exists": True, "enabled": True, "expression": "rate(7 days)", "error": "",
        })
        monkeypatch.setattr(rs_pc, "rule_state", lambda name: {
            "name": name, "exists": True, "enabled": False, "schedule": "",
        })
        monkeypatch.setattr(rs_pc, "next_pull_times",
                            lambda *a, **k: ["2026-10-07T07:43:00+00:00"])
        got = rs_pc.trigger_status()["schedule"]
        assert got["next_run"] == "2026-10-07T07:43:00+00:00"

    def test_no_upcoming_time_is_empty_not_a_guess(self, monkeypatch):
        monkeypatch.setattr(rs_pc, "schedule_status", lambda: {
            "exists": True, "enabled": True, "expression": "rate(7 days)", "error": "",
        })
        monkeypatch.setattr(rs_pc, "rule_state", lambda name: {
            "name": name, "exists": True, "enabled": False, "schedule": "",
        })
        monkeypatch.setattr(rs_pc, "next_pull_times", lambda *a, **k: [])
        assert rs_pc.trigger_status()["schedule"]["next_run"] == ""
