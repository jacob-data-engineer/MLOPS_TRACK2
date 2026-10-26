"""Once a Model Name is chosen, the two cards that MAKE one go out of play.

Defining a Model Name and importing a checkpoint are the two ways to
create the thing that has already been chosen. Left live they invite a
second one, which is how a page ends up with four Model Names and a
checkpoint filed under whichever was clicked last.

Dimmed, not hidden: a card that vanishes teaches nobody that it exists
or what brings it back.
"""
from __future__ import annotations

import pathlib

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
DETECTORS = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "src" / "components" / "Track2DetectorsPanel.tsx")
IMPORT = (pathlib.Path(__file__).resolve().parents[1]
          / "web" / "src" / "components" / "Track2ImportCheckpointPanel.tsx")
SWITCHED_OFF = (pathlib.Path(__file__).resolve().parents[1]
                / "web" / "src" / "components" / "SwitchedOff.tsx")


def _code(path: pathlib.Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


def test_the_lock_is_a_model_being_chosen():
    """And nothing else. It was `&& !defineNew` while a button could
    override it; with the dropdown as the way back there is one fact
    deciding this rather than two that can disagree."""
    code = _code(PAGE)
    assert "const modelLocked = Boolean(modelChoice.slug);" in code


def test_the_define_card_is_locked():
    code = _code(PAGE)
    assert "lockedByPick={modelLocked}" in code


def test_the_import_card_is_locked():
    """Dimmed from the inside, not wrapped in `SwitchedOff`.

    The wrapper put a banner above the card repeating what the card
    already said, and faded everything it wrapped -- including the
    reason, which is the one line that has to stay readable.
    """
    page, imp = _code(PAGE), _code(IMPORT)
    block = page.split("aboveDefine={")[1][:900]
    assert "lockedByPick={modelLocked}" in block
    assert "<SwitchedOff" not in block
    assert 'lockedByPick ? "pointer-events-none opacity-40" : ""' in imp


def test_the_import_card_keeps_a_dead_button_and_a_live_reason():
    """Same shape as the define card: the button says what it would have
    done, the reason says why it will not, and neither fades."""
    imp = _code(IMPORT)
    after = imp.split("{lockedByPick && (")[1]
    block = after[:after.index("<ConfirmDialog")]
    assert "disabled" in block
    assert "Register" in block
    assert "A model has already been selected." in block
    assert "text-info" in block


def test_the_way_back_is_the_dropdown_itself():
    """No button on either card. Clearing the Model Name is what unlocks
    them, so one control decides this rather than two that can disagree.

    The blank option used to be permanently disabled, so a Model Name
    could be picked and never unpicked -- and with both cards that MAKE
    one greyed out behind that choice, there was no way back at all.
    """
    page, detectors = _code(PAGE), _code(DETECTORS)
    assert "Define a new model" not in detectors
    assert "Define a new model" not in page
    assert 'disabled={!chosenSlug}' in detectors, (
        "the blank option has to be selectable once something is chosen"
    )
    assert "Clear the selection" in detectors
    assert "const modelLocked = Boolean(modelChoice.slug);" in page


def test_the_dropdown_is_not_inside_a_dimmed_subtree():
    """It is the only way back now, so it has to stay usable. It lives
    under "Your Model Names", below the card that dims."""
    code = _code(DETECTORS)
    dropdown = code.index('disabled={!chosenSlug}')
    card_ends = code.index("</Card>")
    assert dropdown > card_ends, (
        "the Model Name dropdown must be outside the card that greys out"
    )


def test_the_message_is_the_blue_one():
    code = _code(DETECTORS)
    # To the end of the block rather than a character window: a fixed
    # count stopped short of the sentence the moment the greyed button
    # was added above it.
    after = code.split("{lockedByPick && (")[1]
    block = after[:after.index("</Card>")]
    assert "A model has already been selected." in block
    assert "text-info" in block


def test_the_register_button_is_kept_and_greyed():
    """Hidden, it left nothing to explain why the card does nothing. Dead
    with the reason beside it says both at once."""
    code = _code(DETECTORS)
    after = code.split("{lockedByPick && (")[1]
    block = after[:after.index("</Card>")]
    assert "Register New Model Name" in block
    assert "disabled" in block, "faded is not disabled"


def test_the_lock_is_not_saved_to_the_settings_file():
    """It says what somebody is doing right now, not how runs are
    configured. Saving it would let a loaded file decide which cards are
    usable."""
    code = _code(PAGE)
    assert "defineNew" not in code.split("saveTrack2PullConfig(")[1][:400] \
        if "saveTrack2PullConfig(" in code else True


def test_switched_off_really_disables_rather_than_fading():
    """`pointer-events-none` alone leaves every control reachable by
    keyboard -- looking disabled without being disabled."""
    code = _code(SWITCHED_OFF)
    assert "inert" in code
