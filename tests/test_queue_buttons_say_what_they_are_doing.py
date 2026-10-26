"""Start Queue, Stop Queue and Stop Training say they were pressed.

Each is a round trip -- Stop Training calls SageMaker -- and all three
left the button looking exactly as it did before. Pressing again was the
only way to find out whether the first press had registered.

Worse, `handleQueueStart` awaited the reply and threw it away. So when
the queue was empty the server answered

    400  "Queue is empty -- add at least one run first."

and the page said nothing at all. It was pressed eleven times in a row
that way, which is in the access log.

And the reply belongs IN the queue card: the page's own banners sit
nearly three hundred lines above these buttons, off the top of the
screen.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")


def _code() -> str:
    return PAGE.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


def _handler(name: str) -> str:
    code = _code()
    start = code.index(f"async function {name}(")
    return code[start:code.index("\n  }", start)]


# --- the labels ---------------------------------------------------------------------

def test_each_button_says_what_it_is_doing():
    code = _flat()
    assert '{queueBusy === "start" ? "Starting queue..." : "Start Queue"}' in code
    assert '{queueBusy === "stop" ? "Stopping queue..." : "Stop Queue"}' in code
    assert ('{queueBusy === "stop-training" ? "Stopping training..." '
            ': "Stop Training"}') in code


def test_every_button_goes_down_while_any_one_is_in_flight():
    """They all act on the same queue, so a second press during the
    first is never the next thing to do."""
    assert _flat().count("!!queueBusy") == 3


def test_a_second_press_is_refused_at_the_handler_too():
    """The disabled attribute is the UI's answer; this is the one that
    holds if a click lands first."""
    for name in ("handleQueueStart", "handleQueueStop", "handleStopTraining"):
        assert "if (queueBusy) return;" in _handler(name), name


# --- the flag always clears -----------------------------------------------------------

def test_the_busy_flag_clears_in_finally():
    """A failure is exactly when the button must stop saying
    "Stopping training..." forever."""
    for name in ("handleQueueStart", "handleQueueStop", "handleStopTraining"):
        body = _handler(name)
        assert "} finally {" in body, name
        assert body.index("} finally {") < body.index('setQueueBusy("");'), name


def test_the_list_is_reloaded_whatever_happened():
    for name in ("handleQueueStart", "handleQueueStop", "handleStopTraining"):
        body = _handler(name)
        assert body.index("} finally {") < body.index("load();"), name


# --- the reply is shown, and where it was asked for -------------------------------------

def test_start_no_longer_throws_its_reply_away():
    body = _handler("handleQueueStart")
    assert "setQueueNote({ text: (await startTrack2Queue()).message" in body


def test_each_handler_reports_its_failure():
    for name, fallback in (("handleQueueStart", "Could not start the queue."),
                           ("handleQueueStop", "Could not stop the queue."),
                           ("handleStopTraining", "Stop failed.")):
        body = _handler(name)
        assert "catch (err)" in body, name
        assert fallback in body, name
        assert "err instanceof ApiError ? err.detail" in body, name


def test_the_note_renders_inside_the_queue_card():
    """Not in the page banner three hundred lines up."""
    code = _code()
    note = code.index("{queueNote && (")
    # The BUTTON, not the several prose mentions of "press Start Queue"
    # earlier in the page -- anchoring on the bare label found one of
    # those and measured the wrong distance entirely.
    buttons = code.index(': "Start Queue"}')
    # Within the same card: the buttons are in its header, the note at
    # the top of its body, a few lines apart.
    assert 0 < note - buttons < 3000, (note - buttons)


def test_a_refusal_reads_as_one():
    code = _flat()
    assert 'queueNote.bad ? "text-critical" : "text-ink-secondary"' in code


def test_the_note_is_cleared_before_each_press():
    """Otherwise the previous answer sits under the new press looking
    like the new answer."""
    for name in ("handleQueueStart", "handleQueueStop", "handleStopTraining"):
        assert "setQueueNote(null);" in _handler(name), name


# --- the dead banner went with it --------------------------------------------------------

def test_the_page_level_message_banner_is_gone():
    """Its only writers were these three handlers. Left behind it would
    be state nothing can ever set -- and `noUnusedLocals` would fail the
    build on the setter."""
    code = _code()
    assert "const [message, setMessage]" not in code
    assert "setMessage(" not in code


def test_the_page_level_error_banner_stays():
    """The load failure still uses it."""
    code = _code()
    assert "const [error, setError] = useState<string | null>(null);" in code
    assert "Couldn't load training status." in code


# --- what must not have broken --------------------------------------------------------------

def test_the_original_disabled_rules_survive():
    code = _flat()
    assert "data.queue.started" in code
    assert "data.queue.pending.length === 0 && !data.queue.running" in code
    assert "!data.custom_run_id" in code, (
        "the custom-run guard decides whether Stop Training is reachable "
        "during a custom run"
    )


def test_stop_training_is_still_a_different_thing_from_stop_queue():
    """/queue/stop only halts the loop and lets the in-flight run
    finish."""
    # It takes an argument now -- `includeForeign`, so Stop Training can
    # also end a GPU job this app did not start. Still a different call
    # from stopping the queue, which is what this test is about.
    assert "stopTrack2Training(includeForeign)" in _code()
    assert "stopTrack2Queue()" in _code()


def test_the_server_still_explains_an_empty_queue():
    from ppe_pipeline_api.routers import track2_models as m

    code = inspect.getsource(m.start_queue)
    assert "Queue is empty" in code
