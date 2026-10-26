"""Training on a schedule has to be reachable from both configurations.

Scanned on 2026-10-15, the live config could not get there from either:

    week_cutoff        ""          <- blocked Apply outright
    schedule_mode      null        <- so the schedule was never armed
    queue_after_pull   {}          <- and the chain was off
    dataset_assemble   only=True, scheduled=False
    picked             all four lists empty

TWO FAULTS, both ours.

1. Applying a schedule REFUSED without a week cut-off date. The chain
   that trains is triggered by the pull schedule, so scheduled training
   was unreachable without a date for week folders -- including when
   "Use these folders only" had explicitly switched week folders off.

2. The combination that can never fire. "Use these folders only" means
   no pull moves anything, so every pull reports `moved: 0`, and
   `even_if_nothing_moved` is off by default, which skips exactly those.
   The schedule runs on time, the pull succeeds, the chain declines every
   one. Nothing is broken, which is what makes it invisible.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402


def _blocking(problems: list) -> str:
    return " ".join(p["text"] for p in problems if p["level"] == "blocking")


def _pull_source() -> str:
    """pull_control.py as ONE line.

    Both the comment and the refusal wrap across lines, so a contiguous
    match against the raw source tests the line breaks rather than the
    words.
    """
    import inspect
    return " ".join(inspect.getsource(pc).split())


# --- 1. the cut-off no longer blocks a schedule nobody files into ----------------------

def test_the_cutoff_refusal_is_in_the_apply_path():
    """Kept as a test rather than a comment: this is the sentence that
    made scheduled training unreachable, and it should not come back
    unconditionally."""
    code = _pull_source()
    assert "Set a cut-off date first" in code
    assert 'assemble_only = bool((merged.get("dataset_assemble") or {}).get("only"))' in code


def test_it_is_waived_only_for_assemble_only():
    code = _pull_source()
    at = code.index("Set a cut-off date first")
    assert "if not assemble_only:" in code[max(0, at - 400):at]


def test_the_waiver_is_explained_where_it_is_made():
    # A fragment that sits on ONE line. Flattening joins lines with a
    # space, so a phrase spanning two of them picks up the comment's `#`
    # or the gap between two string literals and never matches.
    assert "Auto Training and Retention OFF" in _pull_source()


def test_the_hole_the_waiver_leaves_is_reported():
    """Without a cut-off every file falls into week 1. Harmless while
    nothing is swept; a corpus collapsed into one week the moment
    something is."""
    assert "every file this pull moves will land in week 1" in _pull_source()


# --- 2. the combination that can never fire -------------------------------------------

@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(rs.pc, "orphan_week_folders", lambda config: [])


def test_building_by_hand_takes_the_pull_out_of_the_trigger(quiet):
    """The combination that "could never fire" is gone, because the pull
    is no longer the trigger at all when "Use these folders only" is
    ticked -- `even_if_nothing_moved` has nothing left to decide."""
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
        "dataset_assemble": {"only": True},
        "queue_after_pull": {"enabled": True, "even_if_nothing_moved": False},
        "retrain_schedule": {"enabled": True, "every_days": 1, "at": "09:00"},
    })
    assert "This cannot fire." not in _blocking(checks["problems"])
    assert checks["ready"] is True


def test_its_own_clock_with_no_time_set_is_blocking(quiet):
    """What replaced it: a schedule that owns the clock and has no time
    of day would never fire, and nothing else would say so."""
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
        "dataset_assemble": {"only": True},
        "retrain_schedule": {"enabled": True, "every_days": 1, "at": ""},
    })
    assert "no time of day is set" in _blocking(checks["problems"])
    assert checks["ready"] is False


def test_ticking_even_if_nothing_moved_clears_it(quiet):
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
        "dataset_assemble": {"only": True},
        "queue_after_pull": {"enabled": True, "even_if_nothing_moved": True},
    })
    assert "This cannot fire." not in _blocking(checks["problems"])


def test_unticking_use_these_folders_only_clears_it_too(quiet):
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
        "dataset_assemble": {"only": False},
        "queue_after_pull": {"enabled": True, "even_if_nothing_moved": False},
    })
    assert "This cannot fire." not in _blocking(checks["problems"])


def test_a_chain_that_is_off_is_not_reported_as_unfireable(quiet):
    """It is not configured to fire at all -- a different fact, and the
    card already says it."""
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
        "dataset_assemble": {"only": True},
        "queue_after_pull": {"enabled": False},
    })
    assert "This cannot fire." not in _blocking(checks["problems"])


def test_the_message_says_which_clock_is_in_charge(quiet):
    """A blocking problem that does not say what to change is a dead
    end."""
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
        "dataset_assemble": {"only": True},
        "retrain_schedule": {"enabled": True, "every_days": 1, "at": ""},
    })
    text = _blocking(checks["problems"])
    assert "its own clock" in text
    assert "Use these folders only" in text


def test_the_defaults_are_imported_not_restated():
    """A second copy would answer "is this on?" differently from the
    module that owns the setting."""
    from ppe_pipeline_api.queue_after_pull import DEFAULTS
    assert rs.DEFAULTS_AFTER_PULL is DEFAULTS


# --- what must not have broken ----------------------------------------------------------

def test_the_original_configuration_still_demands_a_cutoff(quiet):
    """Week folders are measured from it. Nothing about the waiver makes
    that less true where a sweep actually files."""
    assert "it is what the week folders are" in _pull_source()


def test_a_missing_detector_still_blocks(quiet):
    checks = rs.preflight({"root": "retraining"})
    assert "No detector chosen" in _blocking(checks["problems"])


def test_a_ready_configuration_is_still_ready(quiet):
    checks = rs.preflight({
        "root": "retraining",
        "default_detector": {"slug": "fakecad-it"},
    })
    assert checks["ready"] is True
