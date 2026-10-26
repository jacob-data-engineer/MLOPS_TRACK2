"""The card that picks training data by hand, and what it switches off.

Static reads of the TSX, like the other UI tests here: there is no JS
runtime in this environment (CLAUDE.md keeps builds in CodeBuild and Node
is not installed), so what can be checked is the wiring.

What matters and is easy to get wrong:

  * the card sits ABOVE the golden set / edge cases card, which is where
    it was asked for and also where it belongs -- it answers "where does
    this run's data come from", and everything below it is the other
    answer to that question;
  * ticking it switches Pull Control and Auto Training off rather than
    leaving two cards deciding the same thing;
  * they are DIMMED, not hidden, so the page still says they exist;
  * the state behind it is declared above the cards that read it. A const
    reading a `useState` declared below it throws on every render and
    takes the whole page with it -- that has happened here, and `tsc`
    does not catch it when the read is inside a callback.
"""
from __future__ import annotations

import pathlib

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PAGE = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
PANEL = (WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")
OFF = (WEB / "components" / "SwitchedOff.tsx").read_text(encoding="utf-8")


def test_the_card_is_above_the_edge_cases_card():
    """Asked for above it, and it reads in the right order there: pick the
    data, then curate a golden set out of what you have."""
    assemble = PAGE.index("only={pickFoldersByHand}")
    golden = PAGE.index("<Track2GoldenSetPanel />")
    assert assemble < golden


def test_auto_training_is_switched_off_by_the_wrapper():
    before = PAGE[:PAGE.index("<Track2AutoTrainingPanel")]
    wrapper = before.rindex("<SwitchedOff")
    # The wrapper has to be the one immediately around it, not an
    # earlier one that happens to appear above.
    assert "</SwitchedOff>" not in before[wrapper:]
    assert "off={pickFoldersByHand}" in before[wrapper:]


def test_pull_control_switches_its_own_body_off_instead():
    """It cannot use the wrapper any more: the wrapper makes the whole
    subtree `inert`, and the tick that turns this card back ON now sits
    in its own heading. Unticking it would be a one-way door -- the
    switch would be inside the part that stopped taking clicks.

    So the card dims its body and leaves its header alone."""
    flat = " ".join(PAGE.split())
    assert "<Track2PullControlPanel only={pickFoldersByHand}" in flat
    # Nothing above it may have an UNCLOSED wrapper -- which includes
    # there being no wrapper above it at all, as there now is not.
    before = PAGE[:PAGE.index("<Track2PullControlPanel")]
    assert before.count("<SwitchedOff") == before.count("</SwitchedOff>"), \
        "pull control must not be inside a SwitchedOff any more"


def test_it_is_still_switched_off_though():
    pull = (WEB / "components" / "Track2PullControlPanel.tsx").read_text(encoding="utf-8")
    assert "pointer-events-none select-none opacity-45 saturate-50" in pull
    assert "inert={only && onOnlyChange ? true : undefined}" in pull


def test_the_tick_sits_in_the_title_row():
    """Asked for on the same row as the card's name."""
    pull = (WEB / "components" / "Track2PullControlPanel.tsx").read_text(encoding="utf-8")
    flat = " ".join(pull.split())
    assert "titleAside={onOnlyChange ? (" in flat
    assert "Use Pull Control" in flat
    assert flat.index("titleAside=") < flat.index("Use Pull Control") < flat.index("inert={only")


def test_they_are_dimmed_and_left_on_the_page_not_hidden():
    """Hiding them would shorten the page and cost the operator the one
    thing they need to know: that these exist and what turned them off."""
    assert "opacity-45" in OFF
    assert "inert" in OFF, "dimming alone leaves every control tabbable"
    assert "return <>{children}</>" in OFF, "untouched when the tick is off"


def test_the_reason_names_the_tick_that_did_it():
    """Auto Training still says it from the wrapper; Pull Control says it
    itself, next to the tick that turns it back on."""
    assert "reason={" in PAGE
    assert "Untick" in PAGE or "Tick" in PAGE
    pull = (WEB / "components" / "Track2PullControlPanel.tsx").read_text(encoding="utf-8")
    flat = " ".join(pull.split())
    assert "The run is using the folders picked on the Assemble card." in flat
    assert "Use Pull Control</span> to collect data week by week again." in flat


def test_the_tick_is_only_offered_where_it_turns_something_off():
    """The standard tab has none of those three cards, so a tick there
    would claim to switch off nothing."""
    assert "onOnlyChange ?" in PANEL
    standard = PAGE[PAGE.index("{!multi && ("):PAGE.index("only={pickFoldersByHand}")]
    assert "onOnlyChange" not in standard


def test_the_state_is_declared_before_the_cards_that_read_it():
    """The temporal dead zone that blanked this page once already."""
    declared = PAGE.index("const [pickFoldersByHand, setPickFoldersByHand]")
    assert declared < PAGE.index("only={pickFoldersByHand}")
    assert declared < PAGE.index("off={pickFoldersByHand}")


def test_the_card_explains_the_consequence_on_itself():
    """The tick is on this card, so the consequence has to be too --
    otherwise the only explanation lives on cards that are now greyed
    out and easy to scroll past."""
    assert "Pull Control, Auto Training" in PANEL
    assert "switched off" in PANEL
