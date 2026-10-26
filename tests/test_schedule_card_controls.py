"""The clock has controls, and they arm the mechanism actually in charge.

`every_days` and `at` were in DEFAULTS and read by the chain, with no
way to set them: a schedule whose cadence can only be changed by editing
a settings file by hand is not a schedule anybody will use.

WHICH ENABLE. "Enabled for scheduling" has to arm whichever mechanism
owns the clock, and that depends on "Use these folders only":

    only ticked   ->  retrain_schedule's own clock
    only unticked ->  the pull chain

A tick that armed one of them would do nothing in half the
configurations -- silently, because the other is what the schedule
actually reads. So it writes BOTH to the same value.

WHICH CLOCK IS IN CHARGE is computed server-side and returned as
`follows_pull`, rather than the rule being restated in TypeScript where
it could drift.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
API = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "lib" / "api.ts")


def _flat() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())


# --- the effective answer comes from the server --------------------------------------

def test_both_retrain_routes_report_which_clock_is_in_charge():
    from ppe_pipeline_api.routers import track2_retention as r

    for fn in (r.get_retrain, r.save_retrain):
        assert 'out["follows_pull"] = rs.follows_pull(config)' in inspect.getsource(fn), fn


def test_the_client_type_declares_it():
    api = API.read_text(encoding="utf-8")
    block = api.split("export interface Track2RetrainStatus {")[1]
    block = block[:block.index("\n}")]
    assert "follows_pull?: boolean;" in block


def test_the_rule_is_not_restated_in_typescript():
    """`dataset_assemble.only` deciding the clock lives in ONE place."""
    code = _flat()
    assert "follows_pull === false" in code
    assert "dataset_assemble" not in code.split("Enabled for scheduling")[1][:2000]


# --- the tick arms both --------------------------------------------------------------

def test_the_tick_writes_both_enables():
    code = _flat()
    block = code.split("const on = e.target.checked;")[1][:700]
    assert "saveTrack2QueueAfterPull({ enabled: on })" in block
    assert "saveTrack2Retrain({ enabled: on })" in block


def test_it_says_why_both():
    code = _flat()
    # A fragment that sits on ONE line: flattening joins lines with a
    # space, so a phrase spanning two picks up the comment's `//`.
    assert "would be a switch that does nothing" in code


# --- the cadence controls ---------------------------------------------------------------

def test_the_cadence_is_not_set_twice():
    """ONE interval on the page. The PULL, BUILD, QUEUE, TRAIN strip
    owns it; a second pair of fields here would be two controls for one
    cadence, one of them deciding nothing depending on the mode."""
    code = _flat()
    assert "Train every" not in code
    assert "saveClock" not in code


def test_they_are_shown_only_when_this_schedule_owns_the_clock():
    """Two interval fields, one of them inert, is how somebody sets the
    wrong one."""
    code = _flat()
    assert "{retrain && retrain.follows_pull === false && (" in code


def test_the_strip_mirrors_its_cadence_onto_the_training_clock():
    """So the one control on screen means the same thing whichever
    mechanism is in charge."""
    from ppe_pipeline_api import pull_control as pc

    code = inspect.getsource(pc.set_mode)
    assert 'patch["retrain_schedule"] = retrain' in code
    assert 'retrain["every_days"] = max(1, every_days)' in code
    assert 'retrain["at"] = at_time' in code


def test_off_disarms_the_training_clock_too():
    """Leaving it armed while the strip reads "Off" is the card saying
    one thing and the system doing another."""
    from ppe_pipeline_api import pull_control as pc

    assert 'retrain["enabled"] = mode == "schedule"' in inspect.getsource(pc.set_mode)


def test_the_interval_is_still_floored_at_one_day():
    """A retrain takes hours of GPU, so anything shorter overlaps
    itself."""
    from ppe_pipeline_api import pull_control as pc

    assert "max(1, every_days)" in inspect.getsource(pc.set_mode)


def test_the_save_is_a_patch():
    """Turning the schedule on must not reset a cadence set last week."""
    api = API.read_text(encoding="utf-8")
    assert "A PATCH -- only the fields sent are changed" in api
    assert "export function saveTrack2Retrain(" in api


# --- it says when it will actually run ------------------------------------------------------

def test_it_shows_the_next_run_as_a_date():
    """"every 7 days at 02:00" is not something anybody can turn into a
    date in their head."""
    code = _flat()
    assert "Next run ${localDate(retrain.next_runs[0])}" in code


def test_an_unarmed_schedule_says_where_to_arm_it():
    """Pointing at the one control that sets it, rather than leaving
    somebody to find it."""
    code = _flat()
    assert "Not armed" in code
    assert "PULL, BUILD, QUEUE, TRAIN at the top." in code


def test_a_blocking_problem_is_shown_with_the_control():
    """Not at the top of the page, where the answer to a control down
    here would not be read."""
    code = _flat()
    assert 'retrain.problems?.some((p) => p.level === "blocking")' in code


# --- loaded, so the fields are not empty on arrival --------------------------------------------

def test_the_schedule_is_still_read_so_the_card_can_report_it():
    code = _flat()
    assert "getTrack2Retrain()" in code
    assert "setRetrain(res)" in code


def test_a_failed_read_does_not_block_the_page():
    code = _flat()
    block = code.split("getTrack2Retrain()")[1][:400]
    assert ".catch(() => undefined)" in block
