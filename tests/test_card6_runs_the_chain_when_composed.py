"""The training card's button runs the CHAIN when the run is composed.

The radio on card 6 said "Use the assembled dataset + the other ticked
sources ... composed at run time". The button under it called
`addTrack2Queue` with one `dataset_folder`, and one folder cannot carry
a golden set that lives under another root -- only the chain builds
that yaml. So the label promised the golden set and the run trained
without it, with the assembled folder's own val as the "held-out" set.

Worse, the only button that DID run the chain -- "Run it now" on the
Retention card -- sits inside Auto Training, which is switched off and
inert in exactly the configuration that composes: folders + golden,
Pull Control unticked. There was no reachable way to trigger the
composed run by hand; only the schedule could.

Now: when the radio says composed, the button runs the chain with
`submit="queue"`, the same hand-over the schedule uses, and says so.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api.routers import track2_retention as router  # noqa: E402

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PAGE = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
API = (WEB / "lib" / "api.ts").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- the route -----------------------------------------------------------------

def test_the_manual_route_takes_submit():
    assert "submit" in router.RetrainRunRequest.model_fields
    assert router.RetrainRunRequest().submit == "start"


def test_it_is_the_same_two_values_the_chain_takes():
    code = inspect.getsource(router.run_retrain)
    # `enqueue` added 2026-10-26: Add to Queue for the assembled dataset.
    assert 'if submit not in ("start", "queue", "enqueue"):' in code
    assert 'rs.run_chain(trigger="manual", submit=submit)' in code


def test_an_unknown_submit_is_a_400_not_a_default():
    """Silently falling back to `start` would run a job the caller
    asked to queue."""
    code = inspect.getsource(router.run_retrain)
    assert "status_code=400" in code


def test_the_body_is_optional_so_run_it_now_is_unchanged():
    """The Retention card posts `{}` and expects `start`."""
    code = inspect.getsource(router.run_retrain)
    assert "body: RetrainRunRequest | None = None" in code
    assert 'submit = (body.submit if body else "start") or "start"' in code


# --- the client ----------------------------------------------------------------

def test_the_client_can_ask_for_either():
    assert 'runTrack2Retrain(submit: "start" | "queue" | "enqueue" = "start")' in API
    assert 'apiPost("/track2/retention/retrain/run", { submit })' in API


# --- card 6 --------------------------------------------------------------------

def test_composed_is_read_from_the_radio_and_the_ticks():
    code = _flat(PAGE)
    assert 'const composed = dataSource === "assembled" && (sources.golden || sources.pull);' in code


def test_a_composed_run_goes_through_the_chain_as_queue():
    """`queue`, not `start`: the entry lands in the queue card with every
    other run and the queue releases it -- the schedule's hand-over."""
    code = _flat(PAGE)
    # Every assembled run goes through the chain now (2026-10-26); a
    # composed one still as `queue`, which also releases the queue.
    assert 'if (dataSource === "assembled") {' in code
    assert 'const got = await runTrack2Retrain(composed ? "queue" : "enqueue");' in code


def test_an_uncomposed_run_still_queues_the_old_way():
    """One folder, one entry, nothing rebuilt -- every run before this
    existed."""
    code = _flat(PAGE)
    assert "await addTrack2Queue(params()); onSubmitted();" in code


def test_the_chain_path_comes_before_the_old_one():
    code = _flat(PAGE)
    assert code.index('runTrack2Retrain(composed ? "queue" : "enqueue")') < code.index("await addTrack2Queue(params());")


def test_the_failing_step_is_shown_in_its_own_words():
    """"Add to queue failed" would hide the one sentence that says what
    to fix -- the chain already wrote it."""
    code = _flat(PAGE)
    assert 'setError(`Stopped at "${failed?.step ?? "?"}": ${failed?.detail ?? "no detail"}`);' in code


def test_the_button_says_what_it_will_do():
    code = _flat(PAGE)
    assert '(composed ? "Build, Queue and Start" : "Add to Queue")' in code
    assert '(dataSource === "assembled" ? "Generating the yaml and queueing..." : "Working...")' in code


def test_the_change_of_behaviour_is_said_under_the_button():
    code = _flat(PAGE)
    assert "{composed && (" in code
    assert "runs the full chain: build the golden set" in code
    assert "Exactly what the schedule does when it fires, done now." in code


def test_no_time_of_day_is_hard_coded_into_the_note():
    """The schedule's time is a setting. A literal in the UI would read
    right today and lie the moment somebody changed it."""
    import re

    code = _flat(PAGE)
    note = code[code.index("runs the full chain"):code.index("done now.")]
    assert "SGT" not in note
    # A clock time, specifically -- the sentence has an ordinary colon.
    assert not re.search(r"\b\d{1,2}:\d{2}\b", note)


def test_the_label_that_promised_composition_is_now_true():
    """The radio has said 'composed at run time' since the sources
    could be combined. Until this, nothing under it composed."""
    code = _flat(PAGE)
    assert "after the retention policy when Pull Control is ticked." in code
    assert 'runTrack2Retrain(composed ? "queue" : "enqueue")' in code


def test_the_client_import_is_present():
    assert "runTrack2Retrain," in PAGE
