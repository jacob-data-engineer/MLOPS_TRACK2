"""Destructive actions ask in a dialog, and the dialog carries it all.

From a real complaint. Deleting a Model Name used to be: a
`window.confirm`, then a red line at the TOP of the card, then a
separate "Delete it and its versions" button up there too -- the answer
to a question asked near the bottom, several hundred pixels and a scroll
away from the row that was clicked.

See CLAUDE.md, "Make the screen do the work, not the person".
"""
from __future__ import annotations

import pathlib
import re

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
COMPONENTS = WEB / "components"
DIALOG = COMPONENTS / "ConfirmDialog.tsx"


def _code(path: pathlib.Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


# --- the dialog itself ----------------------------------------------------------

def test_the_dialog_exists():
    assert DIALOG.exists(), "ConfirmDialog is where destructive asks live"


def test_cancel_takes_focus_not_confirm():
    """Enter on a freshly opened dialog must not delete anything."""
    code = _code(DIALOG)
    assert "cancelRef.current?.focus()" in code, (
        "the cancel button must take focus, so the default answer is 'no'"
    )


def test_escape_and_the_backdrop_both_cancel():
    """Every reflex that means 'not this' should do the safe thing."""
    code = _code(DIALOG)
    assert '"Escape"' in code and "onCancel()" in code
    assert "onClick={() => { if (!busy) onCancel(); }}" in code, (
        "clicking the backdrop should cancel"
    )


def test_a_request_in_flight_cannot_be_sent_twice():
    code = _code(DIALOG)
    assert "disabled={busy}" in code


def test_the_dialog_can_say_more_than_a_browser_confirm_can():
    """The reason it exists: a body, a destructive styling, and a
    confirm label the caller chooses -- so a refusal can turn the same
    dialog into the escalation."""
    code = _code(DIALOG)
    for prop in ("body", "destructive", "confirmLabel", "cancelLabel"):
        assert prop in code, f"ConfirmDialog needs a {prop}"


# --- the flow that caused the complaint -----------------------------------------

DETECTORS = COMPONENTS / "Track2DetectorsPanel.tsx"


def test_deleting_a_model_name_no_longer_uses_a_browser_confirm():
    assert "window.confirm" not in _code(DETECTORS)


def test_the_escalation_button_is_gone_from_the_top_of_the_card():
    """"Delete it and its versions" used to be rendered at the top of
    the card, nowhere near the Delete that produced it. It is a label on
    the dialog's own confirm button now."""
    code = _code(DETECTORS)
    assert "forceTarget" not in code, "the old top-of-card escalation is still there"
    assert "ConfirmDialog" in code
    assert 'confirmLabel={deleteBlocked ? "Delete it and its versions" : "Delete"}' in code


def test_the_servers_refusal_is_shown_inside_the_dialog():
    """It names the registries and how many versions each holds, which
    is the part that decides whether this is safe."""
    code = _code(DETECTORS)
    assert "setDeleteBlocked(" in code
    assert "{deleteBlocked}" in code, "the refusal must render in the dialog body"


def test_a_refusal_keeps_the_dialog_open():
    """Closing on refusal would send the operator back to the row to
    press Delete again, to be told the same thing."""
    code = _code(DETECTORS)
    body = code.split("async function handleDelete")[1][:900]
    assert "setDeleteTarget(null)" not in body.split("catch")[1], (
        "the dialog should stay open on a refusal and offer the escalation"
    )


def test_the_dialog_says_what_survives():
    """"Datasets and trained weights are untouched" is the sentence that
    lets somebody press the button."""
    assert "untouched" in DETECTORS.read_text(encoding="utf-8")


# --- everywhere else ------------------------------------------------------------

#: `window.confirm` call sites not yet moved to the dialog. Listed so a
#: NEW one fails this test rather than joining a silent backlog. Shrink
#: it; never add to it.
STILL_ON_BROWSER_CONFIRM = {
    # A navigation guard: it has to answer synchronously to stop the
    # route changing, which a dialog cannot do without reworking
    # NavGuardContext. Deliberately last.
    "Layout.tsx",
    # On the Metrics screen, not Multi-Auto Training. Next in line.
    "Track2MetricsExportCard.tsx",
}


@pytest.mark.parametrize(
    "panel", sorted(p.name for p in COMPONENTS.glob("*.tsx")))
def test_no_new_browser_confirms(panel: str):
    code = _code(COMPONENTS / panel)
    if "window.confirm" not in code or panel in STILL_ON_BROWSER_CONFIRM:
        return
    pytest.fail(
        f"{panel} uses window.confirm. Use ConfirmDialog -- see CLAUDE.md, "
        "'Destructive actions get a dialog, and it carries everything'."
    )


def test_the_backlog_is_honest():
    """A name left in the list after its confirm is gone makes the list
    lie, and the next person trusts it."""
    stale = []
    for name in STILL_ON_BROWSER_CONFIRM:
        path = COMPONENTS / name
        if not path.exists() or "window.confirm" not in _code(path):
            stale.append(name)
    assert not stale, (
        f"these no longer use window.confirm -- remove them from "
        f"STILL_ON_BROWSER_CONFIRM: {sorted(stale)}"
    )
