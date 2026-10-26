"""The Assemble card checks that images and labels describe each other.

An Ultralytics label is matched to its image by STEM -- `frame_07.jpg`
belongs to `frame_07.txt`, wherever each lives. So the two folders can be
picked independently, and nothing else in the pipeline checks that they
line up.

Reported from the page: `images/train` paired with `labels/val`. Nothing
matches, the dataset assembles cleanly, the run trains, and the model
learns nothing. No error anywhere.
"""
from __future__ import annotations

import pathlib
import re

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx")


def _code() -> str:
    return "\n".join(
        line for line in PANEL.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


# --- clearing one field at a time -----------------------------------------------

def test_a_single_field_can_be_cleared():
    """"Remove this pair" throws both away, which is the wrong tool when
    the images are right and only the labels point at the wrong split."""
    code = _code()
    assert "function clearField(" in code
    assert "Remove this folder" in code, (
        "the control says what it removes -- one folder, not the pair"
    )
    # Sits under its own field, so which of the two it empties is not a
    # guess. The row-level "Remove this pair" is gone: emptying both
    # fields leaves a blank row, and `currentFolders` drops any pair
    # with no images, so a blank one is never assembled.
    assert "Remove this pair" not in code


def test_clearing_applies_to_both_sides():
    """Training and evaluation rows are rendered by the same loop, so the
    clear has to be keyed on the side rather than assuming train."""
    code = _code()
    assert "clearField(side, index, field)" in code


def test_clearing_drops_the_stale_pairing_result():
    """Otherwise the warning for the old folder stays on screen next to
    an empty field."""
    code = _code()
    block = code.split("function clearField(")[1][:400]
    assert "setPairing(" in block


# --- the check ------------------------------------------------------------------

def test_the_existing_pairing_endpoint_is_reused():
    """It already answers exactly this, including samples of each kind of
    mismatch. A second implementation would be a second answer."""
    code = _code()
    assert "checkTrack2Pairing(" in code


def test_it_checks_every_filled_pair_on_both_sides():
    code = _code()
    block = code.split("const pairs:")[1][:600]
    assert '["train", trainSources], ["eval", evalSources]' in block
    assert "source.images.trim() && source.labels.trim()" in block, (
        "a half-filled pair has nothing to compare yet"
    )


def test_it_is_debounced():
    """Picking a folder changes a field per click; un-debounced this
    would list two folders on every one."""
    code = _code()
    block = code.split("const pairs:")[1][:1200]
    assert "setTimeout(" in block and "clearTimeout" in block


def test_the_worst_case_is_called_out_loudest():
    """Nothing pairing at all is the images/train + labels/val mistake,
    and it is the one that looks completely fine otherwise."""
    code = _code()
    assert "Nothing pairs." in code
    assert "check.paired === 0" in code


def test_a_label_with_no_image_is_treated_as_an_error():
    """It describes nothing and never reaches the trainer, so whatever it
    annotated is silently absent from the run."""
    code = _code()
    assert "labels_without_images" in code
    assert re.search(r"labels have no image", code)


def test_images_without_labels_are_not_treated_as_an_error_unless_total():
    """Ultralytics takes an unlabelled image as a BACKGROUND example and
    they improve precision. A few are healthy; all of them means the
    labels folder is wrong."""
    code = _code()
    assert "background examples" in code, "a few unlabelled images are fine"
    assert "bare === check.images" in code, "all of them is not"
    assert "No image has a label." in code
