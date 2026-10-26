"""The page keeps looking while a CUSTOM run is going.

History showed "Executing · YOLOcustom-YOLO11S · 5:54:50 PM" while the
execution was Stopped and the audit row already said so. The backend was
right; nothing on the page asked it again.

`shouldPoll` decided whether to re-read every 15 seconds, and none of
its three conditions can be true during a custom run:

    training_progress.status === "Executing"
        reports the PRODUCTION pipeline. A custom-class run never moves
        it.
    queue.started
        set FALSE in memory the moment the queue empties -- which is the
        moment the last entry has just STARTED.
    pending_message
        covers the submit window, not the run.

So from the instant a custom run began, the page stopped polling. The
10-second live poll kept the status bar current but only bumped History
on an `Executing -> other` transition of the production pipeline, which
also never happens for a custom run.

Two holes, same cause: `custom_run_id` was in the live payload and read
by nothing.
"""
from __future__ import annotations

import pathlib

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")


def _code() -> str:
    return PAGE.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


def _should_poll() -> str:
    code = _code()
    start = code.index("const shouldPoll = useMemo(")
    return code[start:code.index("[data],", start)]


# --- it polls while a custom run is going ---------------------------------------------

def test_a_custom_run_keeps_the_page_polling():
    assert "data?.custom_run_id" in _should_poll()


def test_the_other_three_conditions_are_still_there():
    """This adds a fourth reason to poll; it removes none."""
    block = _should_poll()
    assert 'data?.training_progress?.status === "Executing"' in block
    assert "data?.queue.started" in block
    assert "data?.pending_message" in block


def test_polling_is_still_bounded():
    """An idle page must still make no extra requests -- the 15s tick is
    the heavy payload."""
    code = _flat()
    assert "if (!shouldPoll) return;" in code


# --- and reloads the moment one ends -----------------------------------------------------

def test_the_end_of_a_custom_run_reloads_history():
    code = _flat()
    assert "if (customBefore && !customNow) {" in code


def test_it_tracks_the_custom_id_across_ticks():
    code = _flat()
    assert "const lastCustomRun = useRef<string | null>(null);" in code
    assert "lastCustomRun.current = customNow;" in code


def test_it_fires_on_the_transition_not_on_every_tick():
    """`!customNow` alone would reload the page's heaviest endpoint
    every ten seconds forever on an idle page."""
    code = _flat()
    assert "customBefore && !customNow" in code


def test_the_production_transition_still_works():
    """The existing trigger, untouched."""
    code = _flat()
    assert 'before === "Executing" && now && now !== "Executing"' in code


def test_both_triggers_reload_the_same_two_things():
    code = _flat()
    assert code.count("setHistoryKey((k) => k + 1); load();") >= 2


# --- what it reads ----------------------------------------------------------------------

def test_the_live_payload_already_carried_it():
    """No new endpoint and no new field -- `custom_run_id` was there and
    read by nothing."""
    api = (pathlib.Path(__file__).resolve().parents[1]
           / "web" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    live = api.split("export interface Track2ModelsLive {")[1]
    live = live[:live.index("\n}")]
    assert "custom_run_id: string | null;" in live


def test_the_server_still_sends_it():
    import inspect
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from ppe_pipeline_api.routers import track2_models as m

    assert '"custom_run_id": custom_run_id,' in inspect.getsource(m)
