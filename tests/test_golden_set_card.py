"""The Golden Set card: switching it on shows what it then needs.

Two small changes, and the regressions they could cause are what most of
this file checks -- the card has a lot of state that has nothing to do
with either.
"""
from __future__ import annotations

import pathlib

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2GoldenSetPanel.tsx")


def _code() -> str:
    return "\n".join(
        line for line in PANEL.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


def _grid_after(marker: str) -> str:
    """The first grid div following a section marker.

    Sliced to the element rather than a character count: the comment
    explaining the layout sits between the two, and a fixed window
    stopped short of the very line being checked.
    """
    code = PANEL.read_text(encoding="utf-8")
    after = code.split(marker)[1]
    start = after.index('<div className="grid')
    return after[start:after.index(">", start) + 1]


# --- the change ------------------------------------------------------------------

def test_switching_it_on_opens_the_card():
    """The card remembers being shut, so "use a golden set" left the one
    thing it then wants -- which folder -- behind a second click."""
    code = _code()
    block = code.split("async function setUse(")[1][:900]
    assert "if (next) setOpen(true);" in block


def test_switching_it_off_does_not_close_the_card():
    """The save is a PATCH precisely so the folder and split settings
    survive being switched off. Folding the card away would hide what it
    just took care to keep."""
    code = _code()
    block = code.split("async function setUse(")[1][:900]
    assert "setOpen(false)" not in block


def test_the_bucket_spans_the_full_width_above_the_pickers():
    """One bucket holds both folders, so it is asked for once at the top.
    In a half-width column beside a folder list it read as belonging to
    that one picker."""
    code = PANEL.read_text(encoding="utf-8")
    section = code.split("the source ----")[1]
    section = section[:section.index("the split ----")]
    assert section.index("Source bucket") < section.index("sm:grid-cols-2"), (
        "the bucket must sit outside, and above, the two-column grid"
    )


def test_the_outer_source_container_is_a_single_column():
    """So the bucket row and the picker row stack."""
    grid = _grid_after("the source ----")
    assert grid == '<div className="grid gap-2">', grid


def test_the_pair_sits_in_a_bordered_box_like_the_assemble_card():
    code = _code()
    assert 'rounded-md border border-hairline bg-page p-2.5' in code


def test_each_folder_is_a_collapsed_field_not_an_open_browser():
    """The Assemble card's shape. Two always-open lists side by side
    were two scrolling things to read before either question had been
    answered."""
    code = _code()
    assert '"Choose a folder..."' in code
    assert code.count("Remove this folder") >= 1


def test_one_browser_opens_under_whichever_field_was_clicked():
    code = _code()
    assert "const [picking, setPicking]" in code
    assert "<Track2S3FolderBrowser" in code
    assert code.count("<Track2S3FolderBrowser") == 1, (
        "one browser, not one per field"
    )


def test_the_pairing_check_is_the_shared_endpoint():
    """A second implementation of "do these describe each other" would be
    a second answer."""
    code = _code()
    assert "checkTrack2Pairing(" in code


def test_the_pairing_check_separates_the_two_kinds_of_mismatch():
    """An image with no label is a background example and fine; a label
    with no image never is."""
    # Collapsed to one line first: the warnings wrap across several in
    # the JSX, so a contiguous-phrase match would be testing the line
    # breaks rather than the words.
    code = " ".join(_code().split())
    assert "Nothing pairs." in code
    assert "labels have no image." in code
    assert "No image has a label." in code
    # And the two kinds are read off different fields, not one count.
    assert "labels_without_images" in code
    assert "images_without_labels" in code


def test_the_target_folder_keeps_its_own_field():
    """One folder, not a pair -- and the only picker here that may create
    a folder that does not exist yet."""
    code = _code()
    assert "<S3FolderField" in code
    assert 'label="Target folder"' in code
    assert "allowCreate" in code


def test_the_two_pickers_are_side_by_side():
    """The same shape the Assemble card uses for its Images and Labels,
    which is what makes the pair legible as a pair.

    They are rendered from a two-entry array now rather than two
    hand-written fields, so this looks for the entries.
    """
    code = PANEL.read_text(encoding="utf-8")
    section = code.split("the source ----")[1]
    section = section[:section.index("the split ----")]
    two_col = section.index("sm:grid-cols-2")
    assert section.index('["images", "Images"') > two_col
    assert section.index('["labels", "Labels"') > two_col


def test_they_stack_on_a_narrow_screen():
    """`sm:` and not a bare `grid-cols-2`: two scrolling lists at phone
    width are each too narrow to read a folder name in."""
    code = PANEL.read_text(encoding="utf-8")
    section = code.split("the source ----")[1]
    section = section[:section.index("the split ----")]
    assert "sm:grid-cols-2" in section
    assert 'className="grid grid-cols-2' not in section


def test_the_target_pair_is_still_two_up():
    """Two short fields. Only the source pair had the picker in it."""
    grid = _grid_after("the target ----")
    assert grid == '<div className="grid gap-2 sm:grid-cols-2">', grid


# --- what must not have broken ---------------------------------------------------

def test_the_switch_is_one_tick_now():
    """It was a pair of buttons -- "Use a golden set" / "Not using one" --
    which read as a choice between two modes. It stopped being that the
    moment the three sources could be combined: this is one of them, and
    all three are asked the same way, on the Assemble card's row and
    here."""
    code = _code()
    assert "Use Pick Golden Set / Edge Cases" in code
    assert 'type="checkbox"' in code
    assert "Not using one" not in code
    assert "Use a golden set" not in code


def test_the_not_in_use_state_still_dims_rather_than_hides():
    code = _code()
    assert "pointer-events-none opacity-40" in code
    assert 'inUse === false ? "pointer-events-none opacity-40" : ""' in code


def test_the_shared_picker_is_still_used():
    """This card used to roll its own -- an "up" link and a flat list --
    so the same job was done three ways depending on the card.

    The first picker was labelled "Folder to split" until a second one
    was added beside it; the pair now reads "Images folder" and "Labels
    folder", matching the Assemble card. Same component, same field,
    new wording.
    """
    code = _code()
    assert "Track2S3FolderBrowser" in code
    assert ">Images" in code or '"Images"' in code


def test_every_field_is_still_there():
    code = _code()
    for field in ("Source bucket", "Target bucket", "Target folder",
                  "Split into train and val"):
        assert field in code, field


def test_the_card_still_saves_its_settings():
    """It is one of the cards the settings file carries; see
    tests/test_card_settings_are_saved.py."""
    code = _code()
    assert "saveTrack2PullConfig" in code
    assert "golden_set:" in code


def test_the_open_state_is_still_the_shared_record():
    code = _code()
    assert 'useCardOpen("golden_set", false)' in code
