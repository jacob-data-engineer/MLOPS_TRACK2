"""Two complaints about the Assemble card, and what they came from.

1. "Write it is not working." It was disabled until "Preview the yaml"
   had been pressed, and said nothing about why -- so a card with every
   field filled showed a dead button, and the control that would have
   woken it was a separate press with nothing pointing at it.

2. The Classes dropdown asked for a Model Name that had already been
   chosen once, in "Register a model and Classes". Asking twice means
   two answers can disagree, and only one of them reaches the yaml.
"""
from __future__ import annotations

import pathlib

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx")
API = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "lib" / "api.ts")


def _code() -> str:
    return PANEL.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


# --- 1. the button works -----------------------------------------------------------

def test_write_it_is_gated_on_the_same_things_the_preview_is():
    """Complete inputs and a Model Name -- the two questions the card
    actually asks."""
    code = _flat()
    assert "if (!chosen || !complete) {" in code


def test_the_card_says_why_the_button_is_off():
    """A disabled control with no reason is the complaint that started
    this. Each missing answer names itself."""
    code = _flat()
    assert "Pick a training folder." in code
    assert "Pick an evaluation folder" in code
    # The classes are no longer asked for here -- they follow the Model
    # Name card -- so the missing answer points there rather than at a
    # control on this card that no longer exists.
    assert "Pick a Model Name above" in code


def test_the_inline_overwrite_checkbox_is_gone():
    """It was the second control the operator had to find, three hundred
    pixels from the refusal that asked for it."""
    code = _flat()
    assert "replace the data.yaml already there" not in code


# --- 2. the classes follow the model already chosen --------------------------------

def test_the_card_reads_the_model_chosen_for_the_run():
    code = _code()
    assert "status.config.default_detector?.slug" in code


def test_the_model_wins_over_anything_saved_here():
    """There is no second answer to win any more.

    This first allowed the card to keep its own override, and said so in
    blue or red when the two differed. Overriding was removed with the
    dropdown: one Model Name decides the registry the run versions into
    AND the class names the yaml declares, and those have to be one
    list. The saved slug is now only a fallback for a page with no
    choice made yet.
    """
    code = _code()
    assert "if (inherited) setDetectorSlug(inherited);" in code
    assert "else if (saved.detector_slug) setDetectorSlug(saved.detector_slug);" in code


def test_there_is_no_way_to_disagree_with_the_model_any_more():
    code = _code()
    assert "setDetectorSlug(e.target.value)" not in code, "the override is back"


def test_the_classes_are_displayed_rather_than_asked_for():
    code = " ".join(_code().split())
    assert "(from your Model Name, in index order)" in code


# --- what must not have broken -----------------------------------------------------

def test_one_button_previews_and_writes():
    """Asked for on 2026-10-20: the separate Preview button removed; the
    one button builds, shows, and writes."""
    code = _code()
    # 2026-10-26: no button at all -- the preview keeps itself current
    # and the run writes the yaml. Sent with overwrite, as the chain is.
    assert "Generate yaml training file" not in code
    assert "previewTrack2DatasetYaml(f, classesOf(), true, dropUndeclared," in code
    assert "onClick={handlePreview}" not in code
    assert "async function handlePreview" not in code


def test_the_overwrite_setting_is_still_saved():
    """Only its checkbox went. The value still rides to the server."""
    code = _code()
    assert "overwrite," in code
    assert "setOverwrite(saved.overwrite)" in code


def test_the_scheduled_key_is_declared_client_side():
    """Pydantic drops fields the model does not declare, and TypeScript
    drops nothing but would not have caught a typo here either."""
    assert "scheduled?: boolean;" in API.read_text(encoding="utf-8")


def test_the_card_still_saves_everything_it_did_before():
    code = _code()
    for key in ("train_sources", "eval_sources", "folders", "detector_slug",
                "overwrite", "built"):
        assert f"{key}," in code or f"{key}:" in code, key
