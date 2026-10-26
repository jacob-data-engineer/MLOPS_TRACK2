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

def test_write_it_is_not_gated_on_having_previewed():
    """The preview is for reading the file first. It was a toll gate."""
    # From the LAST "Preview the yaml" -- the button itself. The earlier
    # one is in the comment on handleWrite explaining why this changed,
    # and slicing from it stops before the button rather than at it.
    code = _flat()
    # One button now: it previews and writes in the same press.
    row = code[code.index("onClick={() => void handleWrite()}") - 200:code.index("Generate yaml training file")]
    assert "!preview" not in row, "the button still waits for a preview"


def test_write_it_is_gated_on_the_same_things_the_preview_is():
    """Complete inputs and a Model Name -- the two questions the card
    actually asks."""
    code = _flat()
    assert code.count("disabled={busy || !complete || !detectorSlug}") == 1


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


def test_the_write_checks_before_it_writes():
    """So an existing data.yaml becomes a dialog naming it, rather than
    a refusal arriving after the press from a server the button cannot
    quote."""
    code = _code()
    body = code.split("async function handleWrite(")[1][:1600]
    assert "previewTrack2DatasetYaml(" in body
    assert "setConfirming(check)" in body


def test_the_dialog_is_the_shared_one_not_window_confirm():
    code = _code()
    assert "<ConfirmDialog" in code
    assert "window.confirm" not in code


def test_the_dialog_says_what_is_not_touched():
    """What survives is the sentence that lets somebody press it."""
    code = _flat()
    assert "No images or labels are touched" in code
    assert "no dataset or trained weights are affected" in code


def test_the_dialog_names_the_file():
    code = _code()
    assert "{confirming?.key}" in code


def test_the_dialog_quotes_what_the_server_said_about_the_existing_file():
    code = _code()
    assert "confirming?.existing?.bytes" in code


def test_confirming_writes_with_force():
    code = _code()
    assert "onConfirm={() => void handleWrite(true)}" in code


def test_cancel_just_closes_it():
    code = _code()
    assert "onCancel={() => setConfirming(null)}" in code


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
    assert "Generate yaml training file" in code
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
