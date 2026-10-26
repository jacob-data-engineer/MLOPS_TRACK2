"""The chain publishes its stages as it runs, not only at the end.

A retrain takes twenty minutes end to end. A progress display that only
reports once it is finished answers the one question nobody is asking.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402


@pytest.fixture
def store(monkeypatch):
    kept: dict = {}
    monkeypatch.setattr(rs.pc, "read_config", lambda: dict(kept))
    monkeypatch.setattr(rs.pc, "write_config", lambda patch: kept.update(patch) or dict(kept))
    return kept


class TestTheStageList:
    def test_every_stage_is_listed_before_anything_runs(self, store):
        stages = rs.chain_progress()["stages"]
        assert [s["step"] for s in stages] == [k for k, _ in rs.CHAIN_STAGES]

    def test_they_all_start_pending(self, store):
        assert {s["state"] for s in rs.chain_progress()["stages"]} == {"pending"}

    def test_each_carries_a_human_label(self, store):
        assert all(s["label"] for s in rs.chain_progress()["stages"])


class TestProgressWhileRunning:
    def test_a_finished_stage_reads_ok(self, store):
        rs._publish_progress(running=True, steps=[{"step": "lists", "ok": True,
                                                   "detail": "wrote lists"}])
        by = {s["step"]: s for s in rs.chain_progress()["stages"]}
        assert by["lists"]["state"] == "ok"
        assert by["lists"]["detail"] == "wrote lists"

    def test_a_failed_stage_reads_failed(self, store):
        rs._publish_progress(running=False, steps=[{"step": "freeze", "ok": False,
                                                    "detail": "no snapshot"}])
        by = {s["step"]: s for s in rs.chain_progress()["stages"]}
        assert by["freeze"]["state"] == "failed"

    def test_the_next_stage_shows_as_running(self, store):
        rs._publish_progress(running=True, steps=[{"step": "preflight", "ok": True,
                                                   "detail": ""}])
        states = {s["step"]: s["state"] for s in rs.chain_progress()["stages"]}
        assert states["preflight"] == "ok"
        # Exactly one in flight, and it is the next one in order.
        # "golden" now sits between preflight and lists: the Golden Set
        # card is one a scheduled run acts on, and its val side becomes
        # the evaluation set, so it has to exist before anything names
        # it. See tests/test_schedule_runs_the_live_cards.py.
        running = [k for k, v in states.items() if v == "running"]
        assert running == ["golden"]

    def test_nothing_is_running_once_it_has_finished(self, store):
        rs._publish_progress(running=False, ok=True,
                             steps=[{"step": "preflight", "ok": True, "detail": ""}])
        states = [s["state"] for s in rs.chain_progress()["stages"]]
        assert "running" not in states


class TestItAlwaysCloses:
    def test_a_raising_chain_still_clears_running(self, store, monkeypatch):
        # The failure a progress bar must never have: stuck on "running"
        # forever because a rarely-taken exit forgot to clear it.
        def boom(**kw):
            raise RuntimeError("nope")

        monkeypatch.setattr(rs, "_run_chain_body", boom)
        with pytest.raises(RuntimeError):
            rs.run_chain(trigger="manual")
        assert rs.chain_progress()["running"] is False

    def test_a_normal_chain_clears_running_and_records_the_outcome(self, store, monkeypatch):
        monkeypatch.setattr(rs, "_run_chain_body", lambda **kw: {"ok": True, "steps": []})
        rs.run_chain(trigger="schedule")
        got = rs.chain_progress()
        assert got["running"] is False
        assert got["ok"] is True

    def test_progress_failures_cannot_break_the_chain(self, monkeypatch):
        # A chain that fell over because its display was unreachable would
        # be a worse failure than no display.
        monkeypatch.setattr(rs.pc, "read_config", lambda: (_ for _ in ()).throw(OSError("down")))
        rs._publish_progress(running=True)  # must not raise
