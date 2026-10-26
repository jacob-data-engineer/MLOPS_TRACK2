"""The Assemble card can be the chain's build step.

Until now `build_in_place` was reachable only from the HTTP route, so a
card somebody had filled in completely still did nothing unattended --
the schedule rebuilt from the root and week folders, which is a
different dataset built by different code.

The tick is what joins them. OFF is the default and is what every config
written before this has, so an existing settings file cannot start
assembling in place by surprise.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx")

FULL = {
    "scheduled": True,
    "train_sources": [{"images": "ds/images/train", "labels": ""}],
    "eval_sources": [{"images": "ds/images/val", "labels": ""}],
    "detector_slug": "fakecad-it",
}
CLASSES = {0: "helmet", 1: "gloves", 2: "person"}


@pytest.fixture
def detector(monkeypatch):
    """A Model Name with three classes."""
    import ppe_pipeline_api.track2_shared as t2
    monkeypatch.setattr(
        t2, "get_detector",
        lambda slug: {"slug": slug, "label": slug.upper(), "classes": dict(CLASSES)})
    return t2


def _blocking(problems: list) -> str:
    return " ".join(p["text"] for p in problems if p["level"] == "blocking")


# --- off is the default ----------------------------------------------------------

def test_no_tick_means_the_old_path():
    assert rs.assemble_plan({}) is None
    assert rs.assemble_plan({"dataset_assemble": {}}) is None


def test_an_empty_card_is_still_the_old_path():
    """"assembled" is the DEFAULT source, so a config that has never
    touched the Assemble card must still reach the root-and-week-folders
    build -- otherwise every existing settings file breaks."""
    assert rs.assemble_plan({"dataset_assemble": {}}) is None
    assert rs.assemble_plan({"data_source": "assembled"}) is None


def test_another_source_being_chosen_takes_this_path_out(detector):
    """The radio decides. A filled card does not run when the operator
    asked for a frozen snapshot."""
    assert rs.assemble_plan({"data_source": "snapshot",
                             "dataset_assemble": FULL}) is None
    assert rs.assemble_plan({"data_source": "pull",
                             "dataset_assemble": FULL}) is None


def test_the_tick_turns_it_on(detector):
    assert rs.assemble_plan({"dataset_assemble": FULL}) is not None


# --- what it reads ---------------------------------------------------------------

def test_the_classes_come_from_the_model_name_not_the_card(detector):
    """The card saves a slug, never a copy of the names. A corrected
    class list has to reach the schedule without the card being
    reopened -- the yaml is the only place those names exist, so nothing
    downstream could notice they were stale."""
    plan = rs.assemble_plan({"dataset_assemble": FULL})
    assert plan["classes"] == CLASSES


def test_the_folders_come_across(detector):
    plan = rs.assemble_plan({"dataset_assemble": FULL})
    assert plan["splits"]["train"] == [{"images": "ds/images/train", "labels": ""}]
    assert plan["splits"]["val"] == [{"images": "ds/images/val", "labels": ""}]


def test_a_source_with_no_images_folder_is_dropped(detector):
    plan = rs.assemble_plan({"dataset_assemble": {
        **FULL, "train_sources": [{"images": "", "labels": "x"},
                                  {"images": "ds/images/train", "labels": ""}]}})
    assert plan["splits"]["train"] == [{"images": "ds/images/train", "labels": ""}]


def test_it_may_replace_the_file_it_wrote_last_time(detector):
    """Running weekly means writing the same key weekly. `write_in_place`
    still refuses a data.yaml this card did NOT write."""
    assert rs.assemble_plan({"dataset_assemble": FULL})["overwrite"] is True


# --- what it refuses -------------------------------------------------------------

def test_no_training_folder_blocks(detector):
    plan = rs.assemble_plan({"dataset_assemble": {**FULL, "train_sources": []}})
    assert "no training folder" in _blocking(plan["problems"]).lower()


def test_no_evaluation_folder_blocks(detector):
    """Evaluating on the training images reports metrics that are not
    held out, and they look like good ones."""
    plan = rs.assemble_plan({"dataset_assemble": {**FULL, "eval_sources": []}})
    assert "no evaluation folder" in _blocking(plan["problems"]).lower()


def test_no_model_name_blocks(detector):
    plan = rs.assemble_plan({"dataset_assemble": {**FULL, "detector_slug": ""}})
    assert "model name" in _blocking(plan["problems"]).lower()


def test_a_model_name_that_no_longer_exists_blocks(monkeypatch):
    import ppe_pipeline_api.track2_shared as t2

    def gone(slug):
        raise ValueError("No detector " + repr(slug) + ".")

    monkeypatch.setattr(t2, "get_detector", gone)
    plan = rs.assemble_plan({"dataset_assemble": FULL})
    assert "could not be read" in _blocking(plan["problems"])


def test_a_model_name_with_no_classes_blocks(monkeypatch):
    import ppe_pipeline_api.track2_shared as t2
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"slug": slug, "classes": {}})
    plan = rs.assemble_plan({"dataset_assemble": FULL})
    assert "no classes" in _blocking(plan["problems"])


# --- preflight -------------------------------------------------------------------

def test_a_missing_root_does_not_block_the_in_place_path(detector, monkeypatch):
    """`root` is the week-folders answer. Blocking on it would stop every
    scheduled run of a card that is working perfectly."""
    monkeypatch.setattr(rs.pc, "orphan_week_folders", lambda config: [])
    checks = rs.preflight({"dataset_assemble": FULL,
                           "default_detector": {"slug": "fakecad-it"}})
    assert "No root folder is set" not in _blocking(checks["problems"])


def test_a_missing_root_still_blocks_the_old_path(monkeypatch):
    monkeypatch.setattr(rs.pc, "orphan_week_folders", lambda config: [])
    checks = rs.preflight({"default_detector": {"slug": "fakecad-it"}})
    assert "No root folder is set" in _blocking(checks["problems"])


def test_the_cards_own_problems_reach_preflight(detector, monkeypatch):
    monkeypatch.setattr(rs.pc, "orphan_week_folders", lambda config: [])
    checks = rs.preflight({"dataset_assemble": {**FULL, "eval_sources": []},
                           "default_detector": {"slug": "fakecad-it"}})
    assert "no evaluation folder" in _blocking(checks["problems"]).lower()
    assert checks["ready"] is False


# --- the chain uses the card's own function --------------------------------------

def test_the_chain_calls_write_in_place_not_a_second_implementation():
    code = inspect.getsource(rs._run_chain_body)
    assert "dy.write_in_place(" in code
    assert "assemble_plan(config)" in code


def test_the_chain_moves_root_to_where_the_yaml_landed():
    """The freeze, the published folder and the prefix the run downloads
    all work on ONE folder. Leaving `root` at the week-folders answer
    would freeze and train something the yaml does not describe."""
    code = inspect.getsource(rs._run_chain_body)
    # Everything between the write and the second-download note that
    # follows it. A fixed character count stood in for that once and
    # broke each time the call grew an argument or a comment.
    after = code.split("dy.write_in_place(")[1].split("# The SECOND download")[0]
    assert 'root = out["folder"]' in after


def test_the_old_path_is_still_there():
    """This is a branch, not a replacement. Every config written before
    the tick existed goes down the root-and-week-folders side."""
    code = inspect.getsource(rs._run_chain_body)
    assert "rl.write(root=root, config=retention)" in code
    assert "dy.build(root=root" in code


# --- the card --------------------------------------------------------------------

def _panel() -> str:
    return PANEL.read_text(encoding="utf-8")


def test_the_card_no_longer_carries_its_own_scheduling_tick():
    """It was a SECOND answer to the question the Training data radios
    already ask, and nothing said which won."""
    # Comments EXPLAINING the move legitimately name it, so this looks
    # for the control rather than the phrase: no state, no setter, no
    # checkbox bound to it.
    code = "\n".join(
        line for line in _panel().splitlines()
        if not line.strip().startswith(("//", "*", "/*"))
    )
    # The CONTROL, not the phrase: a JSX comment explaining where the
    # tick went legitimately names it, and asserting on the words would
    # fail for a comment that is doing its job.
    assert "setScheduled" not in code
    assert "checked={scheduled}" not in code
    assert "const [scheduled" not in code


def test_the_tick_lives_with_the_radios_that_decide_the_source():
    page = (pathlib.Path(__file__).resolve().parents[1]
            / "web" / "src" / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    flat = " ".join(page.split())
    assert "Enabled for scheduling" in flat
    # Below the group it depends on, not floating somewhere else.
    assert flat.index("Use the assembled dataset") < flat.index("Enabled for scheduling")
