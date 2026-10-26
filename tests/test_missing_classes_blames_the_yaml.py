"""The refusal names the yaml, because the yaml is what it read.

Both "cannot be trained from it" refusals compare a detector's class
list against `dataset_classes`, which reads data.yaml and opens no .txt
file at all. They said "has no labels for gloves, person, ...".

That is a claim about the dataset, and it was false. The labels had
every box. What had happened: the Assemble card writes data.yaml from
the selected Model Name, so selecting `helmet` and pressing Write it
stamped

    names:
      0: helmet

onto six-class data -- snapshot 1e6cad7b5efb2712, frozen 17:08, is that
exact file. Queueing as `fakecad-it` then refused, and the message sent
the operator to check a dataset that was completely intact.

The guard itself was right. Only its sentence was wrong.
"""
from __future__ import annotations

import pathlib
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402

SIX = {0: "helmet", 1: "gloves", 2: "person",
       3: "working_boots", 4: "man_cage", 5: "safe_area"}


def _refuse(declared: dict, expected: dict = None):
    """The refusal `validate_dataset_folder` raises, as text."""
    with mock.patch.object(t2, "_s3_object_exists", lambda k: True), \
         mock.patch.object(t2, "dataset_classes", lambda f: declared), \
         mock.patch.object(t2, "get_detector",
                           lambda s: {"slug": s, "classes": expected or SIX}):
        with pytest.raises(ValueError) as exc:
            t2.validate_dataset_folder("ds", used_for_evaluation=False,
                                       detector_slug="fakecad-it")
        return str(exc.value)


# --- it blames the right thing -----------------------------------------------------

def test_it_says_the_yaml_does_not_declare_them():
    text = _refuse({0: "helmet"})
    assert "does not declare" in text
    assert "in its data.yaml" in text


def test_it_no_longer_claims_there_are_no_labels():
    """The sentence that cost the time."""
    assert "has no labels for" not in _refuse({0: "helmet"})


def test_it_names_every_class_the_yaml_is_missing():
    text = _refuse({0: "helmet"})
    for name in ("gloves", "person", "working_boots", "man_cage", "safe_area"):
        assert name in text, name


def test_it_says_what_the_yaml_does_declare():
    assert "The yaml declares helmet" in _refuse({0: "helmet"})


def test_an_empty_yaml_reads_as_nothing_rather_than_blank():
    assert "declares nothing" in _refuse({})


# --- and says how to fix it ----------------------------------------------------------

def test_it_explains_that_labels_carry_no_names():
    """Otherwise the next person checks the .txt files too."""
    text = _refuse({0: "helmet"})
    assert "Class names live ONLY in data.yaml" in text
    assert "indices and no names" in text


def test_it_points_at_the_card_that_overwrote_it():
    text = _refuse({0: "helmet"})
    assert "a different Model Name selected" in text
    assert "Assemble a dataset" in text


def test_the_sentence_is_defined_once():
    """Two refusals, one explanation -- or they drift."""
    assert t2._YAML_IS_THE_ONLY_SOURCE_OF_NAMES
    import inspect
    for fn in (t2.validate_dataset_folder, t2.derive_filtered_dataset):
        assert "_YAML_IS_THE_ONLY_SOURCE_OF_NAMES" in inspect.getsource(fn)


# --- the guard still guards ------------------------------------------------------------

def test_a_matching_yaml_is_accepted():
    with mock.patch.object(t2, "_s3_object_exists", lambda k: True), \
         mock.patch.object(t2, "dataset_classes", lambda f: SIX), \
         mock.patch.object(t2, "get_detector",
                           lambda s: {"slug": s, "classes": SIX}):
        t2.validate_dataset_folder("ds", used_for_evaluation=False,
                                   detector_slug="fakecad-it")


def test_a_superset_is_still_fine():
    """A site dataset labelled once with everything on it; the run
    derives a copy holding only this detector's classes."""
    more = {**SIX, 6: "forklift"}
    with mock.patch.object(t2, "_s3_object_exists", lambda k: True), \
         mock.patch.object(t2, "dataset_classes", lambda f: more), \
         mock.patch.object(t2, "get_detector",
                           lambda s: {"slug": s, "classes": SIX}):
        t2.validate_dataset_folder("ds", used_for_evaluation=False,
                                   detector_slug="fakecad-it")


def test_one_missing_class_is_still_refused():
    """The case this guard exists for: a run that would train happily,
    report metrics, and file them under a taxonomy it does not have."""
    short = {k: v for k, v in SIX.items() if v != "man_cage"}
    assert "man_cage" in _refuse(short)


def test_a_missing_data_yaml_is_still_its_own_refusal():
    with mock.patch.object(t2, "_s3_object_exists", lambda k: False):
        with pytest.raises(ValueError, match="has no data.yaml"):
            t2.validate_dataset_folder("ds", used_for_evaluation=False,
                                       detector_slug="fakecad-it")
