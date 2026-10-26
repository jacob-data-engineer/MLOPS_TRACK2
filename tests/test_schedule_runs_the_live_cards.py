"""A scheduled run acts on the cards that are not greyed out.

"Use these folders only" greys exactly two: Pull Control, and Auto
Training (which carries Retention inside it). Everything else on the
page stays live -- and the chain was acting on only some of them.

The one it ignored entirely was the GOLDEN SET. `golden_set` was written
by its card and read by no code anywhere in the repo: saved on every
edit, carried in the settings file, consumed by nothing. Which is why
the preflight went on warning

    "No separate evaluation set, so metrics are measured on the training
     corpus's own val split"

while a golden set sat configured a few cards above it.

Now the chain builds it, first, and its val side becomes the evaluation
set -- "the val side is the fixed evaluation set", in the card's own
words.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")

LIVE = {
    "retention": {"use_golden": True},
    "golden_set": {"source": "curated/images", "target_folder": "golden_set",
                   "split": True, "train_percent": 80},
}


# --- which cards are greyed ---------------------------------------------------------

def test_only_two_cards_are_switched_off_by_building_by_hand():
    """If a third ever joins them, the chain's idea of "live" has to
    change with it -- so this is pinned rather than assumed.

    Two cards, but only ONE wrapper now: Pull Control dims its own body
    instead, because the tick that turns it back on lives in its
    heading and the wrapper would have made that tick inert."""
    code = PAGE.read_text(encoding="utf-8")
    assert code.count("off={pickFoldersByHand}") == 1
    assert code.count("only={pickFoldersByHand}") == 2


def test_the_golden_set_card_is_not_one_of_them():
    """It is a folder somebody curated by hand. It has nothing to do
    with week folders, so "Use these folders only" does not touch it."""
    code = PAGE.read_text(encoding="utf-8")
    at = code.index("<Track2GoldenSetPanel")
    assert "SwitchedOff" not in code[max(0, at - 400):at]


# --- the golden set is read now -------------------------------------------------------

def test_it_runs_when_the_card_is_in_use_and_filled_in():
    assert rs.golden_plan(LIVE) is not None


def test_it_does_not_run_when_the_card_is_switched_off():
    assert rs.golden_plan({**LIVE, "retention": {"use_golden": False}}) is None
    assert rs.golden_plan({"golden_set": LIVE["golden_set"]}) is None


@pytest.mark.parametrize("missing", ["source", "target_folder"])
def test_a_half_filled_card_does_not_run(missing):
    """Leaves the evaluation set wherever it already points, rather than
    building half a golden set."""
    card = {k: v for k, v in LIVE["golden_set"].items() if k != missing}
    assert rs.golden_plan({**LIVE, "golden_set": card}) is None


def test_the_split_settings_travel():
    plan = rs.golden_plan(LIVE)
    assert plan["split"] is True
    assert plan["train_percent"] == 80


# --- the chain builds it, first -------------------------------------------------------

def test_the_chain_builds_it_before_anything_names_it():
    code = inspect.getsource(rs._run_chain_body)
    assert "gs.build(" in code
    assert code.index('record("golden"') < code.index("frozen = frozen_plan")


def test_it_has_a_stage_of_its_own_in_the_progress_bar():
    keys = [key for key, _ in rs.CHAIN_STAGES]
    assert keys.index("golden") == keys.index("preflight") + 1
    assert keys.index("golden") < keys.index("data.yaml")


def test_a_failed_golden_build_stops_the_chain():
    """Falling back to the training corpus's own val split would report
    optimistic numbers under a label promising held-out ones."""
    code = inspect.getsource(rs._run_chain_body)
    # The golden stage is everything between the build and the frozen
    # check that follows it. A fixed character count stood in for that
    # once and broke the moment the stage grew a comment.
    block = code.split("gs.build(")[1].split("frozen = frozen_plan(")[0]
    assert 'record("golden", False' in block
    assert '"ok": False' in block


def test_not_using_one_is_recorded_rather_than_skipped_silently():
    code = inspect.getsource(rs._run_chain_body)
    assert "not in use -- the evaluation set is" in code


# --- and its val side becomes the evaluation set ----------------------------------------

def test_golden_val_is_added_to_the_val_split():
    """Decided 2026-10-20: substituting left the dataset's own 1,693 val
    images used by nothing. Both are scored now."""
    code = inspect.getsource(rs._run_chain_body)
    assert 'kept = list(splits.get("val") or []) if picks["folders"] else []' in code
    assert 'splits = {**splits, "val": kept + [{"images": golden_val, "labels": ""}]}' in code


def test_the_val_side_is_the_images_val_folder():
    code = inspect.getsource(rs._run_chain_body)
    assert 'golden_val = f"{golden[\'target_folder\']}/images/val"' in code


def test_the_training_side_is_left_alone():
    """The golden set's train side is trained on in full and never
    decays; the chain does not touch which folders train."""
    code = inspect.getsource(rs._run_chain_body)
    block = code.split("if golden_val:")[1][:400]
    assert '"train"' not in block


# --- what must not have broken ------------------------------------------------------------

def test_the_greyed_cards_are_still_skipped_when_building_by_hand():
    assert rs.follows_pull({"dataset_assemble": {"only": True}}) is False


def test_the_week_folder_path_still_exists_for_everyone_else():
    code = inspect.getsource(rs._run_chain_body)
    assert "rl.write(root=root, config=retention)" in code
    assert "dy.build(root=root" in code


def test_the_chain_still_ends_in_queue_and_start():
    code = inspect.getsource(rs._run_chain_body)
    assert "t2.queue_add(entry)" in code
    assert "t2.queue_start()" in code
