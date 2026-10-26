"""Why "Write it" did nothing, and why nothing said so.

TWO faults, one symptom.

1. `routers/track2_models._splits_of` builds `(images, labels)` TUPLES,
   because the two older consumers -- `assemble_dataset` and
   `inspect_dataset_layout` -- take tuples. `build_in_place` read
   `source.get("images")`. So every press in multi-source mode died on
   `'tuple' object has no attribute 'get'` and came back as a 400.

2. That 400 was rendered at the TOP of the card, four hundred pixels
   above the button that caused it. Scrolled out of view, a failed write
   is indistinguishable from a button that does nothing -- which is how
   it was reported.

The card had never written a yaml this way, so "Use the assembled
dataset" in step 6 stayed greyed out with "Nothing assembled yet".
"""
from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy  # noqa: E402

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx")
PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")

ROOT = "ds"
NAMES = {0: "helmet", 1: "gloves"}


@pytest.fixture
def bucket(monkeypatch):
    """Enough of a bucket to size a download and find no existing yaml."""
    monkeypatch.setattr(dy.pc, "_list_objects", lambda prefix, b: [])
    monkeypatch.setattr(dy.pc, "destination_bucket", lambda: "track2-s3")
    monkeypatch.setattr(dy, "_existing_yaml", lambda bucket, key: {})


# --- 1. both shapes reach the builder --------------------------------------------

TUPLES = {"train": [(f"{ROOT}/images/train", "")],
          "val": [(f"{ROOT}/images/val", "")]}
DICTS = {"train": [{"images": f"{ROOT}/images/train", "labels": ""}],
         "val": [{"images": f"{ROOT}/images/val", "labels": ""}]}


def test_the_tuples_the_route_sends_are_accepted(bucket):
    """This is the shape the card has always sent. It raised
    AttributeError before, which the route turned into a 400."""
    out = dy.build_in_place(splits=TUPLES, classes=NAMES, bucket="track2-s3")
    assert out["key"] == f"{ROOT}/data.yaml"


def test_dicts_still_work(bucket):
    """What the scheduled path sends."""
    out = dy.build_in_place(splits=DICTS, classes=NAMES, bucket="track2-s3")
    assert out["key"] == f"{ROOT}/data.yaml"


def test_both_shapes_produce_the_same_file(bucket):
    a = dy.build_in_place(splits=TUPLES, classes=NAMES, bucket="track2-s3")
    b = dy.build_in_place(splits=DICTS, classes=NAMES, bucket="track2-s3")
    assert a["train"] == b["train"] and a["val"] == b["val"]
    assert a["key"] == b["key"]


def test_a_three_element_tuple_does_not_break_it(bucket):
    """Defensive: the helper slices to two rather than unpacking, so a
    producer that grows a field does not take the card down."""
    splits = {"train": [(f"{ROOT}/images/train", "", "extra")],
              "val": [(f"{ROOT}/images/val", "")]}
    assert dy.build_in_place(splits=splits, classes=NAMES,
                             bucket="track2-s3")["key"] == f"{ROOT}/data.yaml"


def test_a_one_element_tuple_is_images_only(bucket):
    splits = {"train": [(f"{ROOT}/images/train",)],
              "val": [(f"{ROOT}/images/val",)]}
    assert dy.build_in_place(splits=splits, classes=NAMES,
                             bucket="track2-s3")["train"] == ["images/train"]


def test_the_labels_check_still_bites_in_tuple_form(bucket):
    """A labels folder that is not the one Ultralytics will read must
    still be refused -- it would train every image as a background
    example and report metrics for a model that learned nothing."""
    splits = {"train": [(f"{ROOT}/images/train", "somewhere/else")], "val": []}
    with pytest.raises(ValueError, match="would be read from"):
        dy.build_in_place(splits=splits, classes=NAMES, bucket="track2-s3")


def test_an_empty_images_entry_is_skipped_in_tuple_form(bucket):
    splits = {"train": [("", "x"), (f"{ROOT}/images/train", "")],
              "val": [(f"{ROOT}/images/val", "")]}
    assert dy.build_in_place(splits=splits, classes=NAMES,
                             bucket="track2-s3")["train"] == ["images/train"]


def test_the_helper_takes_both_shapes_directly():
    assert dy._images_and_labels({"images": "a/images", "labels": "a/labels"}) \
        == ("a/images", "a/labels")
    assert dy._images_and_labels(("a/images", "a/labels")) == ("a/images", "a/labels")
    assert dy._images_and_labels(("a/images/", "")) == ("a/images", "")


# --- 2. the answer is where the question was asked --------------------------------

def _panel() -> str:
    return PANEL.read_text(encoding="utf-8")


def test_the_failure_is_shown_beside_the_button():
    """Not at the top of the card. A refusal four hundred pixels above
    the control that caused it reads as nothing happening."""
    code = _panel()
    note = code.index("Preview only -- nothing is written.")
    assert code.index('{error && <p className="mt-2 text-xs text-critical">') > note


def test_neither_is_still_at_the_top():
    code = _panel()
    assert 'className="mb-2 text-xs text-critical"' not in code


# --- the classes cascade ----------------------------------------------------------

def test_the_dropdown_is_gone():
    """One question, asked once, in the Model Names card."""
    code = _panel()
    assert "Pick the Model Name whose class names these labels use" not in code
    assert "onChange={(e) => setDetectorSlug(e.target.value)}" not in code


def test_the_classes_are_shown_in_index_order():
    """A label line is `2 0.51 ...`; the order IS the meaning, so a list
    sorted by anything else would be a different mapping."""
    code = " ".join(_panel().split())
    assert "sort((a, b) => Number(a[0]) - Number(b[0]))" in code


def test_the_card_takes_the_model_as_a_prop():
    code = _panel()
    assert "modelSlug?: string;" in code


def test_the_prop_is_actually_destructured():
    """Declaring it in the type only is not taking it.

    `tsc` catches this -- "Cannot find name 'modelSlug'" -- but seven
    minutes into a CodeBuild, and the two halves are forty lines apart
    in the file.
    """
    code = _panel()
    head = code.split("export function Track2DatasetAssemblePanel({")[1]
    head = head[:head.index("}: {")]
    assert "modelSlug" in head


def test_it_follows_a_change_made_after_it_mounted():
    """The Model Name card is on the same page. A class list fixed at
    mount would be stale and silently so."""
    code = " ".join(_panel().split())
    assert "if (modelSlug) setDetectorSlug(modelSlug); }, [modelSlug]);" in code


def test_the_page_passes_it_to_both_copies_of_the_card():
    """Standard and multi tabs render the same component; passing it to
    one is how the two tabs drift."""
    assert PAGE.read_text(encoding="utf-8").count("modelSlug={modelChoice.slug}") == 2


def test_it_says_where_to_go_when_no_model_is_chosen():
    code = " ".join(_panel().split())
    assert "Pick a Model Name in <strong>Your Model Names</strong> above" in code


# --- what must not have broken ----------------------------------------------------

def test_the_slug_is_still_saved_for_the_schedule():
    """`assemble_plan` reads it to find the classes for an unattended
    run. A displayed-only value that was never written would leave the
    schedule with no classes at all."""
    code = _panel()
    assert "detector_slug: detectorSlug," in code


def test_the_buttons_are_still_gated_on_having_one():
    code = " ".join(_panel().split())
    # The preview runs only once the card is complete and a Model Name
    # is chosen -- the gate the button had.
    assert "if (!chosen || !complete) {" in code
