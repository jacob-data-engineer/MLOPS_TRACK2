"""The status bar has to show a pull that runs with no chain behind it.

A scheduled sweep is the one nobody is watching. Before this it appeared
nowhere except inside the Pull Control card, whose run-discovery poll only
runs while that card is expanded -- so a collapsed card meant 1,600 files
could move start to finish with nothing on screen saying so, and the bar
read "No retrain yet" the whole time.
"""
from __future__ import annotations

import importlib

import pytest

mod = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")


@pytest.fixture
def quiet(monkeypatch):
    """No live pull, no finished pull, queue-after-pull off."""
    monkeypatch.setattr(mod.pc, "active_run", lambda: None)
    monkeypatch.setattr(mod.pc, "recent_run_states", lambda limit=20: [])
    qap = importlib.import_module("src.ppe_pipeline_api.queue_after_pull")
    monkeypatch.setattr(qap, "settings", lambda config=None: {"enabled": False})
    return monkeypatch


def test_no_pull_at_all_is_still_a_full_shape(quiet):
    """Every field present. A bar that has to test for undefined before
    each read is a bar that renders blank on the first poll."""
    got = mod._pull_status()
    for key in ("run_id", "status", "trigger", "done", "total", "moved",
                "skipped", "errors", "started_at", "finished_at", "chains"):
        assert key in got, key
    assert got["status"] == ""
    assert got["total"] == 0


def test_a_running_sweep_is_reported(quiet):
    quiet.setattr(mod.pc, "active_run", lambda: {
        "run_id": "20261006T004215-bba9d723", "status": "running",
        "trigger": "schedule", "total": 1614, "done": 1375, "moved": 1374,
        "skipped": 0, "errors": 0, "started_at": "2026-10-06T00:42:16+00:00",
    })
    got = mod._pull_status()
    assert got["status"] == "running"
    assert (got["done"], got["total"]) == (1375, 1614)
    assert got["trigger"] == "schedule"
    # Still running, so there is no finish time to show.
    assert got["finished_at"] == ""


def test_a_running_sweep_wins_over_a_finished_one(quiet):
    """What is happening now, not what happened last."""
    quiet.setattr(mod.pc, "active_run", lambda: {"run_id": "now", "status": "running"})
    quiet.setattr(mod.pc, "recent_run_states", lambda limit=20: [
        {"run_id": "before", "status": "done", "finished_at": "2026-10-05T00:00:00+00:00"},
    ])
    assert mod._pull_status()["run_id"] == "now"


def test_falls_back_to_the_last_FINISHED_sweep(quiet):
    """Skipping the unfinished rows matters: a run that was abandoned has
    no `finished_at`, and reporting it as the last pull would show a
    stale in-flight count forever."""
    quiet.setattr(mod.pc, "recent_run_states", lambda limit=20: [
        {"run_id": "abandoned", "status": "running"},
        {"run_id": "real", "status": "done", "moved": 1614,
         "finished_at": "2026-10-06T00:44:19+00:00"},
    ])
    got = mod._pull_status()
    assert got["run_id"] == "real"
    assert got["moved"] == 1614


def test_chains_reports_whether_anything_follows_the_pull(quiet):
    """The silent one. A pull that moved thousands of files and is
    followed by nothing looks identical to one about to retrain."""
    assert mod._pull_status()["chains"] is False
    qap = importlib.import_module("src.ppe_pipeline_api.queue_after_pull")
    quiet.setattr(qap, "settings", lambda config=None: {"enabled": True})
    assert mod._pull_status()["chains"] is True


def test_a_broken_read_does_not_break_the_bar(quiet):
    """One status bar is not worth erroring every page over."""
    def boom(*a, **k):
        raise RuntimeError("ddb is having a day")
    quiet.setattr(mod.pc, "active_run", boom)
    quiet.setattr(mod.pc, "recent_run_states", boom)
    got = mod._pull_status()
    assert got["status"] == ""
    assert got["chains"] is False


def test_both_return_paths_of_chain_status_carry_the_pull():
    """The GPU call failing is exactly when somebody is staring at this
    bar, so the early return must not be the one that drops the pull."""
    import inspect
    source = inspect.getsource(mod.chain_status)
    assert source.count("_pull_status()") == 2, source
