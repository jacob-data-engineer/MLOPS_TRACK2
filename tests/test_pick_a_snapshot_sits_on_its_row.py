""""Pick a snapshot" sits on the row it acts on.

It was rendered after all three radios and the dataset lines beneath
them, so it read as belonging to whichever option happened to be ticked.
It only ever acts on one: "Use a frozen snapshot".

It is a SIBLING of that label, not a child. A button inside a label is
also that label's control, so clicking it would tick the radio as well
-- the picker would select the source as a side effect of opening the
list.
"""
from __future__ import annotations

import pathlib

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")


def _code() -> str:
    return PAGE.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


def _options_block() -> str:
    """The map that renders the three radios."""
    code = _code()
    start = code.index('const ready = option.key === "snapshot"')
    return code[start:code.index("\n          })}", start)]


# --- it is on the row ----------------------------------------------------------------

def test_the_button_is_rendered_inside_the_options_map():
    assert '{snapOpen ? "Hide snapshots" : "Pick a snapshot"}' in _options_block()


def test_only_the_snapshot_row_gets_it():
    block = " ".join(_options_block().split())
    assert 'if (option.key !== "snapshot") return row;' in block


def test_it_shares_a_flex_row_with_the_label():
    block = " ".join(_options_block().split())
    assert '<div key={option.key} className="flex items-start gap-2"> {row}' in block


def test_it_is_a_sibling_of_the_label_not_a_child():
    """A button inside a label is the label's control too, so clicking
    it would tick the radio as a side effect."""
    block = " ".join(_options_block().split())
    row_end = block.index("</label> );")
    button_at = block.index('{snapOpen ? "Hide snapshots"')
    assert button_at > row_end, "the button is nested inside the label"


def test_it_does_not_grow_or_shrink_with_the_row():
    block = _options_block()
    assert "shrink-0" in block


def test_the_old_standalone_margins_are_gone():
    """`ml-2 mt-1.5` positioned it as its own line under everything."""
    assert "ml-2 mt-1.5 rounded-md" not in _code()


# --- both return paths are valid list items --------------------------------------------

def test_the_plain_row_still_carries_a_key():
    """Whichever of the two is returned IS the list item."""
    block = _options_block()
    assert "<label key={option.key}" in block


def test_the_wrapped_row_carries_one_too():
    block = _options_block()
    assert "<div key={option.key}" in block


# --- it still does what it did ----------------------------------------------------------

def test_it_still_toggles_and_loads_the_list():
    block = " ".join(_options_block().split())
    assert "const next = !snapOpen; setSnapOpen(next);" in block
    assert "listTrack2Snapshots(root, 25)" in block


def test_it_still_reads_the_root_from_the_live_slot():
    block = " ".join(_options_block().split())
    assert 'const root = (autoDataset?.folder || datasetFolder || "").trim();' in block


def test_it_is_still_disabled_unless_that_source_is_chosen():
    block = " ".join(_options_block().split())
    assert 'disabled={dataSource !== "snapshot"}' in block


def test_it_still_says_why_when_disabled():
    code = _flat()
    assert "Switch to “Use a frozen snapshot” to pick one." in code


def test_a_failed_read_still_reports():
    block = " ".join(_options_block().split())
    assert 'setSnapNote("Could not read the snapshots.")' in block


# --- what must not have broken ------------------------------------------------------------

def test_all_three_options_still_render():
    code = _flat()
    for label in ("Use the assembled dataset", "Use a frozen snapshot",
                  "Use snapshot generated during auto pull"):
        assert label in code, label


def test_the_snapshot_list_and_note_still_follow_below():
    code = _flat()
    assert "{snapNote && (" in code
    assert "{snapOpen && (" in code


def test_the_snapshot_option_is_still_always_pickable():
    block = " ".join(_options_block().split())
    assert 'const ready = option.key === "snapshot" || (option.key === "assembled"' in block
