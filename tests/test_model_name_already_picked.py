"""A warning must not tell somebody to make a second Model Name.

Reported from the page: FAKECAD-IT defined and picked, and two red
lines still read "define the Model Name below and save it" and "a
checkpoint is waiting for a Model Name". Both were correct that a
CHOICE was missing in the import card, and both said it in words that
mean a Model Name does not exist.
"""
from __future__ import annotations

import pathlib

IMPORT_CARD = (pathlib.Path(__file__).resolve().parents[1]
               / "web" / "src" / "components" / "Track2ImportCheckpointPanel.tsx")
DETECTORS = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "src" / "components" / "Track2DetectorsPanel.tsx")


def _code(path: pathlib.Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


def test_the_import_card_checks_whether_any_model_name_exists():
    code = _code(IMPORT_CARD)
    assert "detectors.filter((d) => !d.builtin)" in code, (
        "the message has to know whether there is anything to pick"
    )


def test_it_says_already_picked_when_one_exists():
    """One sentence, and the SAME one the define card shows. Two
    wordings for one state read as two different states, which is what
    the longer version -- naming the model and where to choose it --
    turned into once both cards carried a message."""
    code = _code(IMPORT_CARD)
    assert "A model has already been selected." in code


def test_both_cards_word_it_identically():
    """Each locked card says it once, beside its own dead button. The
    page used to carry a third copy in a `SwitchedOff` banner above the
    import card -- a second line saying what the card already said."""
    page = _code(pathlib.Path(__file__).resolve().parents[1]
                 / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
    for where, code in (("import card", _code(IMPORT_CARD)),
                        ("define card", _code(DETECTORS))):
        assert "A model has already been selected." in code, where
    assert "A model has already been selected." not in page, (
        "the banner's copy should be gone, not duplicated on the page"
    )


def test_it_still_says_define_one_when_there_are_none():
    """The original wording is right in the original case."""
    code = _code(IMPORT_CARD)
    assert "Define the Model Name below and save it" in code


def test_the_already_picked_line_is_blue_not_red():
    """Red means "you cannot". A line whose whole job is to say where
    something already is read as a refusal in the same alarm ink."""
    code = _code(IMPORT_CARD)
    block = code.split("A model has already been selected.")[0][-400:]
    assert "text-info" in block
    assert "text-critical" not in block.split("if (defined.length)")[-1]


def test_the_detectors_summary_names_the_picked_model():
    code = _code(DETECTORS)
    assert "chosenSlug ? (" in code
    assert "A model has already been selected." in code
    block = code.split("A model has already been selected.")[0][-300:]
    assert "text-info" in block, "both sections use the same blue"


def test_the_info_colour_is_a_token_not_a_hex():
    """So it themes with everything else and is defined in one place."""
    css = (pathlib.Path(__file__).resolve().parents[1]
           / "web" / "src" / "index.css").read_text(encoding="utf-8")
    assert "--color-info:" in css


def test_the_detectors_summary_stays_red_with_nothing_picked():
    """Then it really is waiting for a Model Name."""
    code = _code(DETECTORS)
    assert "A checkpoint is waiting for a Model Name" in code
    assert "text-critical" in code.split("A checkpoint is waiting for a Model Name")[0][-300:]
