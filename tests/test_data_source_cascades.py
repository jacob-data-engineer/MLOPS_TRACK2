"""The ticked source decides the folder, on a reload as well as a click.

The radio was restored from `data_source`, and then the folder was set
from `default_dataset` WHATEVER the radio said. So every page load came
back with "Use the assembled dataset" ticked and `retraining` -- the
frozen-snapshot slot -- underneath it, and the run that followed trained
`retraining`.

Clicking the radio cascaded correctly. Reloading undid it. Nothing on
screen said which of the two to believe, and the queue entry recorded
the one nobody had chosen:

    Use the assembled dataset  (ticked)
      Testing_Dataset/object_detection_dataset . built Oct 15, 03:22 PM
    retraining                                  <- what actually ran
    Evaluation: retraining
    Built by the snapshot run ... frozen as 532f55d72b51a774 (451 files)

Three records, one slot each, and they are not interchangeable:

    assembled  ->  dataset_assemble.built     what the card wrote
    snapshot   ->  default_dataset            what a run would use now
    pull       ->  last_pull_dataset          what the chain last froze
"""
from __future__ import annotations

import pathlib

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")


def _code() -> str:
    return PAGE.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


def _mount() -> str:
    """The block that restores the three slots on load."""
    code = _code()
    start = code.index("const saved = status.config.data_source;")
    return code[start:code.index("getTrack2QueueAfterPull()", start)]


# --- the mapping exists, and it is one mapping --------------------------------------

def test_each_source_reads_its_own_record():
    block = " ".join(_mount().split())
    assert 'key === "assembled" ? assembledRecord' in block
    assert 'key === "pull" ? (status.config.last_pull_dataset ?? null)' in block
    assert '(built.folder ? built : null)' in block


def test_the_folder_comes_from_the_chosen_source_not_the_frozen_slot():
    block = " ".join(_mount().split())
    assert "let settled = recordFor(source);" in block
    assert "datasetFolder: settled.folder," in block


def test_the_frozen_slot_no_longer_sets_the_folder_unconditionally():
    """The whole bug in one line."""
    block = " ".join(_mount().split())
    assert "setSelection({ datasetFolder: built.folder, evalFolder: built.folder });" \
        not in block


def test_an_explicit_pick_in_this_tab_still_wins():
    """A refresh must not move the folder out from under someone
    mid-setup."""
    block = " ".join(_mount().split())
    assert "if (!datasetFolder && settled?.folder)" in block


def test_the_assembled_record_carries_its_own_evaluation_folder():
    """A SEPARATE download -- EVAL_DATA_S3_URI, read by a different
    container. Collapsing it onto the training folder would evaluate on
    the images it trained on and report that as held out."""
    block = " ".join(_mount().split())
    assert "eval_folder || settled.folder," in block


# --- when the chosen source is empty ------------------------------------------------

def test_an_empty_source_moves_the_radio_with_the_fallback():
    """Falling back to the frozen slot while leaving the radio on
    "assembled" is the same disagreement wearing different clothes."""
    block = " ".join(_mount().split())
    assert 'settledSource = "snapshot";' in block
    assert 'setDataSource("snapshot");' in block


def test_the_pull_record_can_arrive_late_and_still_cascade():
    """`last_pull_dataset` is often empty in the config; the route also
    reads the snapshot store, and answers after this block."""
    block = " ".join(_mount().split())
    assert 'if (settledSource === "pull" && !datasetFolder)' in block


# --- the line under the radios describes the radio ----------------------------------

def test_the_frozen_provenance_only_shows_under_the_frozen_radio():
    """It records `default_dataset`. Rendered whenever that slot had
    anything in it, it sat under a ticked "Use the assembled dataset"
    describing a different dataset."""
    code = _flat()
    assert '{dataSource === "snapshot" && autoDataset?.folder && (' in code


def test_it_is_not_rendered_on_the_bare_slot_any_more():
    code = _flat()
    assert "{autoDataset?.folder && ( <p className=\"tabular mt-1" not in code


# --- clicking still works, and still agrees ------------------------------------------

def test_clicking_assembled_still_points_the_run_at_it():
    code = _flat()
    assert 'if (option.key === "assembled" && assembled?.folder)' in code


def test_clicking_pull_still_restores_the_frozen_lists():
    """A snapshot's folder is the live root. Pointing at it without
    putting the frozen file lists back would train on whatever has been
    pulled since, under a label promising the pull's images."""
    code = _flat()
    assert 'else if (option.key === "pull" && lastPull?.folder)' in code
    assert "restoreTrack2Snapshot(lastPull.snapshot_id)" in code


def test_the_saved_choice_is_still_written_on_click():
    code = _flat()
    assert "saveTrack2PullConfig({ data_source: option.key })" in code


# --- what must not have broken --------------------------------------------------------

def test_all_three_options_are_still_offered():
    code = _flat()
    for label in ("Use the assembled dataset", "Use a frozen snapshot",
                  "Use snapshot generated during auto pull"):
        assert label in code, label


def test_the_old_live_setting_still_maps_onto_assembled():
    """A settings file written when there were two choices."""
    block = " ".join(_mount().split())
    assert 'saved === "snapshot" || saved === "pull" ? saved : "assembled"' in block


def test_the_snapshot_option_is_still_always_pickable():
    """Picking one is how it gets filled, so greying it until it has a
    value would be a control that can never be reached."""
    code = _flat()
    assert 'const ready = option.key === "snapshot" || Boolean(option.folder);' in code
