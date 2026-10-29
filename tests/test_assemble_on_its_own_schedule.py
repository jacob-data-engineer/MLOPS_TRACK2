"""Assemble's Manual / Scheduled radio (2026-10-29).

Scheduled rebuilds the dataset after each SCHEDULED Data Preparation pull,
training nothing. These pin when it does and does not fire.
"""
from __future__ import annotations

import pathlib

import pytest

from ppe_pipeline_api.chain import assemble_after_pull as aap
from ppe_pipeline_api.chain import retrain_schedule as rs

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"

SCHEDULED = {"mode": "schedule", "dataset_assemble": {"run": "scheduled"},
             "retrain_schedule": rs.trigger_fields("clock")}


@pytest.fixture
def pulls(monkeypatch):
    runs = [{"run_id": "r1", "moved": 12, "trigger": "schedule", "finished_at": "x"}]
    claimed: set = set()
    monkeypatch.setattr(aap.qap, "_finished_pulls", lambda **kw: runs)
    monkeypatch.setattr(aap.pc, "claim_exists", lambda what, key: (what, key) in claimed)

    def claim(what, key, days=7):
        if (what, key) in claimed:
            return False
        claimed.add((what, key))
        return True

    monkeypatch.setattr(aap.pc, "claim_once", claim)
    return runs


def test_manual_never_builds(pulls):
    assert aap.pending({**SCHEDULED, "dataset_assemble": {"run": "manual"}})["act"] is False
    assert aap.mode({}) == "manual"


def test_scheduled_builds_once_per_pull_and_trains_nothing(pulls, monkeypatch):
    calls = []
    monkeypatch.setattr(aap.rs, "run_chain",
                        lambda **kw: calls.append(kw) or {"ok": True, "folder": "f", "steps": []})
    monkeypatch.setattr(aap.pc, "write_config", lambda c: c)
    assert aap.check(SCHEDULED)["built"] is True
    assert calls[0]["submit"] == "build" and calls[0]["config"]["data_source"] == "assembled"
    assert aap.check(SCHEDULED)["built"] is False          # same pull: once
    assert len(calls) == 1


def test_it_waits_for_the_pull_schedule(pulls):
    decision = aap.pending({**SCHEDULED, "mode": "arrival"})
    assert decision["act"] is False and "PULL" in decision["reason"].upper()


def test_it_stands_down_when_training_follows_the_pull(pulls):
    """That chain builds the same dataset itself; two writers would race."""
    config = {**SCHEDULED, "retrain_schedule": rs.trigger_fields("pull")}
    assert aap.pending(config)["act"] is False


def test_an_empty_pull_changes_nothing(pulls):
    pulls[0]["moved"] = 0
    assert aap.pending(SCHEDULED)["act"] is False


def test_the_chain_can_stop_before_training():
    import inspect
    body = inspect.getsource(rs._run_chain_body)
    stop = body.index('if submit == "build":')
    assert stop < body.index("# 2c. What it continues FROM.")


def test_the_radio_is_on_the_card_and_saved():
    panel = (WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")
    assert 'name="assemble-run"' in panel
    assert "run: runMode," in panel


def test_the_tick_watches_for_it():
    ticker = (pathlib.Path(rs.__file__).parent / "ticker.py").read_text(encoding="utf-8")
    assert "assemble_after_pull.check" in ticker
