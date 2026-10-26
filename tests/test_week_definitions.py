"""Week definitions that will route files somewhere nothing reads."""
from __future__ import annotations

import importlib

import pytest

pc = importlib.import_module("src.ppe_pipeline_api.pull_control")

CONFIG = {"week_prefix": "week_", "week_length_days": 7}


def _weeks(monkeypatch, rows):
    monkeypatch.setattr(pc, "list_weeks", lambda: rows)


def test_a_contiguous_set_is_clean(monkeypatch):
    _weeks(monkeypatch, [
        {"name": "week_1", "starts_at": "2026-03-18T00:00:00+00:00"},
        {"name": "week_2", "starts_at": "2026-03-25T00:00:00+00:00"},
    ])
    assert pc.week_definition_problems(CONFIG) == []


def test_a_misnamed_week_is_reported(monkeypatch):
    """`day_1` survived an experiment with a different prefix. It still
    covers a date range, so the newest captures route into a folder whose
    name contradicts every other folder."""
    _weeks(monkeypatch, [
        {"name": "week_1", "starts_at": "2026-03-18T00:00:00+00:00"},
        {"name": "day_1", "starts_at": "2026-03-25T00:00:00+00:00"},
    ])
    got = pc.week_definition_problems(CONFIG)
    assert [p["kind"] for p in got] == ["prefix"]
    assert "day_1" in got[0]["detail"]


def test_a_gap_is_reported(monkeypatch):
    _weeks(monkeypatch, [
        {"name": "week_3", "starts_at": "2026-04-01T00:00:00+00:00"},
        {"name": "week_6", "starts_at": "2026-04-22T00:00:00+00:00"},
    ])
    got = pc.week_definition_problems(CONFIG)
    assert [p["kind"] for p in got] == ["gap"]
    assert "2026-04-08" in got[0]["detail"]


def test_touching_weeks_are_not_a_gap(monkeypatch):
    """week_1 ends exactly where week_2 starts. Off-by-one here would
    report a gap on every correctly-defined pair."""
    _weeks(monkeypatch, [
        {"name": "week_1", "starts_at": "2026-03-18T00:00:00+00:00"},
        {"name": "week_2", "starts_at": "2026-03-25T00:00:00+00:00"},
        {"name": "week_3", "starts_at": "2026-04-01T00:00:00+00:00"},
    ])
    assert pc.week_definition_problems(CONFIG) == []


def test_an_overlap_is_not_reported_as_a_gap(monkeypatch):
    """Overlapping weeks are a different problem; this check must not
    invent a gap for them."""
    _weeks(monkeypatch, [
        {"name": "week_1", "starts_at": "2026-03-18T00:00:00+00:00"},
        {"name": "week_2", "starts_at": "2026-03-20T00:00:00+00:00"},
    ])
    assert [p["kind"] for p in pc.week_definition_problems(CONFIG)] == []


def test_a_custom_prefix_is_honoured(monkeypatch):
    """Somebody who set `day_` meant it: then `week_1` is the odd one."""
    _weeks(monkeypatch, [
        {"name": "day_1", "starts_at": "2026-03-18T00:00:00+00:00"},
        {"name": "week_1", "starts_at": "2026-03-25T00:00:00+00:00"},
    ])
    got = pc.week_definition_problems({"week_prefix": "day_", "week_length_days": 7})
    assert [p["week"] for p in got if p["kind"] == "prefix"] == ["week_1"]


def test_an_unparseable_start_is_skipped_not_fatal(monkeypatch):
    _weeks(monkeypatch, [
        {"name": "week_1", "starts_at": ""},
        {"name": "week_2", "starts_at": "2026-03-25T00:00:00+00:00"},
    ])
    assert pc.week_definition_problems(CONFIG) == []


def test_no_weeks_at_all(monkeypatch):
    _weeks(monkeypatch, [])
    assert pc.week_definition_problems(CONFIG) == []
