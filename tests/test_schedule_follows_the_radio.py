"""A scheduled run trains whatever "Training data" is set to.

The chain had its own idea of the source: a tick on the Assemble card,
or else the root-and-week-folders build. The radios in step 6 already
answer "what does a run train on", and the two could disagree with
nothing on screen saying which won.

Now the radio decides, and the three options mean three different
things:

    assembled  the Assemble card's folders, rebuilt into a data.yaml
               and frozen each time
    snapshot   the frozen snapshot, trained EXACTLY as frozen
    pull       the snapshot the pull chain last froze, likewise

NOTHING IS REBUILT for the two snapshot sources. "The same images every
time, whatever has been pulled since" is the whole promise of a frozen
snapshot; rebuilding the lists under the current retention policy would
break it quietly -- the run would still succeed, on a different corpus
than the one that was pinned. Re-freezing would also mint a second id
for the same images and lose the link to the one chosen.
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

CARD = {
    "only": True,   # assemble mode
    "train_sources": [{"images": "ds/images/train", "labels": ""}],
    "eval_sources": [{"images": "ds/images/val", "labels": ""}],
    "detector_slug": "fakecad-it",
}


@pytest.fixture
def detector(monkeypatch):
    import ppe_pipeline_api.track2_shared as t2
    monkeypatch.setattr(t2, "get_detector", lambda slug: {
        "slug": slug, "classes": {0: "helmet", 1: "gloves"}})


# --- which source ---------------------------------------------------------------------

def test_assembled_is_the_default_for_a_config_that_never_chose():
    assert rs.data_source({}) == "assembled"


@pytest.mark.parametrize("value", ["snapshot", "pull", "assembled"])
def test_the_three_options_are_read_through(value):
    assert rs.data_source({"data_source": value}) == value


def test_an_unknown_value_falls_back_rather_than_breaking():
    """"live" is the old name, and a settings file can still carry it."""
    assert rs.data_source({"data_source": "live"}) == "assembled"
    assert rs.data_source({"data_source": ""}) == "assembled"


# --- the frozen sources ------------------------------------------------------------------

def test_the_frozen_snapshot_comes_from_the_live_slot():
    plan = rs.frozen_plan({
        "data_source": "snapshot",
        "default_dataset": {"folder": "retraining", "snapshot_id": "abc", "files": 2214},
    })
    assert plan["folder"] == "retraining"
    assert plan["snapshot_id"] == "abc"
    assert plan["files"] == 2214
    assert plan["problems"] == []


def test_the_pull_snapshot_comes_from_its_own_slot():
    """`default_dataset` is "what a run would use now" and anything may
    overwrite it; `last_pull_dataset` survives every manual pick."""
    plan = rs.frozen_plan({
        "data_source": "pull",
        "default_dataset": {"folder": "WRONG"},
        "last_pull_dataset": {"folder": "retraining", "snapshot_id": "xyz"},
    })
    assert plan["folder"] == "retraining"
    assert plan["snapshot_id"] == "xyz"


def test_an_unchosen_snapshot_is_a_blocking_problem():
    plan = rs.frozen_plan({"data_source": "snapshot"})
    assert plan["folder"] == ""
    assert plan["problems"][0]["level"] == "blocking"
    assert "none has been chosen yet" in plan["problems"][0]["text"]


def test_the_message_names_which_source_it_means():
    assert "frozen snapshot" in rs.frozen_plan(
        {"data_source": "snapshot"})["problems"][0]["text"]
    assert "auto pull" in rs.frozen_plan(
        {"data_source": "pull"})["problems"][0]["text"]


def test_assembled_is_not_a_frozen_source():
    assert rs.frozen_plan({"data_source": "assembled"}) is None
    assert rs.frozen_plan({}) is None


# --- the assembled source ------------------------------------------------------------------

def test_the_card_runs_when_its_radio_is_chosen(detector):
    assert rs.assemble_plan({"data_source": "assembled",
                             "dataset_assemble": CARD}) is not None


def test_the_card_does_not_run_when_another_source_is_chosen(detector):
    for other in ("snapshot", "pull"):
        assert rs.assemble_plan({"data_source": other,
                                 "dataset_assemble": CARD}) is None, other


def test_an_untouched_card_falls_through_to_the_week_folders(detector):
    """"assembled" is the default source, so treating an empty card as a
    misconfiguration would break every config written before it."""
    assert rs.assemble_plan({"data_source": "assembled"}) is None
    assert rs.assemble_plan({"dataset_assemble": {}}) is None


# --- the chain honours it --------------------------------------------------------------------

def test_the_chain_asks_for_the_frozen_plan_first():
    code = inspect.getsource(rs._run_chain_body)
    assert "frozen = frozen_plan(config)" in code
    assert code.index("if frozen is not None:") < code.index("elif assembled is not None:")


def test_a_frozen_source_rebuilds_nothing():
    code = inspect.getsource(rs._run_chain_body)
    block = code.split("if frozen is not None:")[1]
    block = block[:block.index("elif assembled is not None:")]
    assert "dy.write_in_place(" not in block
    assert "rl.write(" not in block
    assert "already frozen" in block or "nothing is rebuilt" in block


def test_a_frozen_source_is_not_re_frozen():
    """A second id for the same images loses the link to the one that
    was chosen."""
    code = inspect.getsource(rs._run_chain_body)
    at = code.index("# 2. freeze it")
    block = code[at:at + 1200]
    assert "if frozen is not None:" in block
    assert block.index("if frozen is not None:") < block.index("ds.create(")


def test_the_chain_trains_the_folder_that_was_frozen():
    code = inspect.getsource(rs._run_chain_body)
    block = code.split("if frozen is not None:")[1][:700]
    assert 'root = frozen["folder"]' in block


def test_an_unchosen_snapshot_stops_the_chain():
    code = inspect.getsource(rs._run_chain_body)
    block = code.split("if frozen is not None:")[1][:700]
    assert 'if not frozen["folder"]:' in block


def test_preflight_reports_the_frozen_problems():
    code = inspect.getsource(rs.preflight)
    assert "frozen = frozen_plan(config)" in code
    assert 'problems.extend(frozen["problems"])' in code


# --- the card says so ------------------------------------------------------------------------

def _flat() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())


def test_the_caption_no_longer_claims_it_does_not_start():
    """`run_chain(submit="queue")` calls `queue_start()`, and has since
    the chain was written -- so the old sentence described something the
    code has never done."""
    # The tail of the old sentence, which the comment above the new
    # caption does NOT quote -- asserting on the quoted half would fail
    # for a comment that is doing its job.
    assert "press Start Queue for that" not in _flat()




def test_the_card_says_automatic_runs_use_the_ticked_sources():
    """An automatic run composes from the ticked sources; the radio is for
    runs added by hand (2026-10-26)."""
    code = _flat()
    assert "Switched on and off by <strong>PULL, BUILD, QUEUE, TRAIN</strong>" in code
    assert "queues the run and starts it" in code
    assert "The Training data radio above is for runs you add here by hand." in code


def test_unattended_runs_ignore_a_frozen_radio():
    assert rs.automatic({"data_source": "snapshot"})["data_source"] == "assembled"
    assert "run_chain(config=automatic(config), trigger=\"schedule\", submit=\"queue\")"         in inspect.getsource(rs.check_training_schedule)
