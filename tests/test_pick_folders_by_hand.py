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
# The week-by-week cards and the tick that switches them off live on the
# Data Preparation tab since the data/training split (2026-10-28).
PAGE = (WEB / "routes" / "Track2DataPrepPage.tsx").read_text(encoding="utf-8")
MODELS = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
PANEL = (WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")
OFF = (WEB / "components" / "SwitchedOff.tsx").read_text(encoding="utf-8")


def test_one_definition_replaces_the_competing_folder_cards():
    assert "<Track2DataPreparationWorkflow" in PAGE
    assert "Track2PullControlPanel" not in PAGE
    assert "Track2AutoTrainingPanel" not in PAGE
    workflow=(WEB / "components" / "Track2DataPreparationWorkflow.tsx").read_text(encoding="utf-8")
    assert workflow.index('title="Data sources"') < workflow.index('title="Split and golden-set policy"')
    assert 'kind: "collected"' in workflow and '<option value="golden">' in workflow


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


def test_the_tick_is_only_offered_where_it_turns_something_off():
    """The standard tab has none of those three cards, so a tick there
    would claim to switch off nothing."""
    assert "onOnlyChange ?" in PANEL
    # The standard tab is still on the training page, and still has no tick.
    at = MODELS.index("{!multi && (")
    standard = MODELS[at:MODELS.index(")}", at)]
    assert "onOnlyChange" not in standard


def test_the_card_explains_the_consequence_on_itself():
    """The tick is on this card, so the consequence has to be too --
    otherwise the only explanation lives on cards that are now greyed
    out and easy to scroll past."""
    assert "Pull Control, Auto Training" in PANEL
    assert "switched off" in PANEL
