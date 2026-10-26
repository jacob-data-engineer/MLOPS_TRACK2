"""Step 6 offers three sources, and each one has its own record.

The three are different questions: what this page assembled, a snapshot
somebody froze by hand, and what the pull chain last produced. They
cannot share one slot -- loading a snapshot by hand writes
`default_dataset`, which would erase the answer to "what did the pull
build" the moment anybody picked one.
"""
from __future__ import annotations

import pathlib
import re

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
ASSEMBLE = (pathlib.Path(__file__).resolve().parents[1]
            / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx")


def _code(path: pathlib.Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


# --- the pull's own record ------------------------------------------------------

def test_the_chain_keeps_its_own_dataset_record():
    """`default_dataset` is the live slot and anything may overwrite it.
    The pull's answer needs a slot nothing else writes."""
    from ppe_pipeline_api.pull_control import DEFAULTS

    assert "last_pull_dataset" in DEFAULTS


def test_the_chain_writes_both_slots():
    import inspect

    from ppe_pipeline_api import retrain_schedule as rs

    source = inspect.getsource(rs)
    assert '"last_pull_dataset": produced' in source, (
        "the chain must record its dataset where a manual pick cannot "
        "overwrite it"
    )
    assert '"default_dataset": produced' in source, (
        "and still fill the live slot, which is what a run reads"
    )


def test_the_route_accepts_it_so_a_settings_file_carries_it():
    from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest

    assert "last_pull_dataset" in ConfigRequest.model_fields


# --- what the assemble card leaves behind ---------------------------------------

def test_the_assemble_card_records_what_it_built():
    """Without this the folder lived in a browser tab, so "use the
    assembled dataset" was empty on every reload."""
    code = _code(ASSEMBLE)
    # The literal was named `record` so the SAME value could be both set
    # in state and written to the config immediately -- see
    # tests/test_assembled_option_lights_up.py. Two literals would be
    # two chances for the screen and the file to say different things.
    assert "setBuilt(record)" in code
    assert "const record = {" in code
    assert "built," in code, "it has to ride in the same writer as the rest"


def test_the_built_record_is_restored_on_mount():
    code = _code(ASSEMBLE)
    assert "if (saved.built?.folder) setBuilt(saved.built);" in code


# --- the three options ----------------------------------------------------------

def test_all_three_are_offered():
    code = _code(PAGE)
    for key in ('key: "assembled"', 'key: "snapshot"', 'key: "pull"'):
        assert key in code, f"missing option {key}"


def test_pick_any_folder_is_gone():
    """Dropped deliberately. Leaving the button behind would be a control
    permanently greyed, since nothing can select its mode any more."""
    code = _code(PAGE)
    # Word-bounded: `pickFoldersByHand` is the Assemble card's "use these
    # folders only" switch and has nothing to do with this picker.
    for dead in ("pickOpen", "pickPrefix", "pickFolders", "pickBusy", "pickNote"):
        assert not re.search(rf"\b{dead}\b", code), f"{dead} survived the removal"
    assert "Choose a folder" not in code


def test_an_old_settings_file_still_loads():
    """`live` is the old name. A file still carrying it must land on one
    of the three rather than being ignored, which would silently put the
    card back on Assembled while the file said otherwise."""
    # The named `live` branch folded into the default when the restore
    # was rewritten to cascade (see tests/test_data_source_cascades.py).
    # Same landing, wider net: ANY value that is not one of the other
    # two becomes "assembled", so a name nobody anticipated lands
    # somewhere real instead of leaving the radio and the folder
    # disagreeing.
    code = " ".join(_code(PAGE).split())
    assert 'saved === "snapshot" || saved === "pull" ? saved : "assembled"' in code
    assert "setDataSource(source);" in code


def test_an_unavailable_source_is_greyed_with_a_reason():
    """Hiding it would mean a reader cannot tell the option exists or
    how to get it."""
    code = _code(PAGE)
    assert "missing:" in code
    assert "disabled={!ready" in code
    assert "opacity-55" in code, "the greyed state has to be visible"


def test_the_snapshot_option_is_never_greyed():
    """Picking one is how it gets filled, so greying it until it has a
    value would be a control that can never be reached."""
    code = _code(PAGE)
    assert 'option.key === "snapshot" || Boolean(option.folder)' in code


def test_choosing_a_source_points_the_run_at_that_folder():
    """Otherwise the radio and the line beneath it disagree, and the run
    trains on whatever was selected last."""
    code = _code(PAGE)
    block = code.split("onChange={() => {")[1][:1400]
    assert 'option.key === "assembled" && assembled?.folder' in block
    assert 'option.key === "pull" && lastPull?.folder' in block
    assert re.search(r"setSelection\(\{\s*datasetFolder: assembled\.folder", block)


# --- the auto-pull snapshot is found, not waited for ----------------------------

def test_only_automatic_triggers_count_as_an_auto_pull():
    """`manual retrain` is somebody pressing Run it now, and `queued run
    ...` is the queue advancing. Neither is an auto pull."""
    from ppe_pipeline_api.routers.track2_retention import AUTO_PULL_TRIGGERS

    offered = ["after-pull retrain", "schedule retrain"]
    skipped = ["manual retrain", "queued run 99cadb7dacf3", ""]
    for note in offered:
        assert note.startswith(AUTO_PULL_TRIGGERS), note
    for note in skipped:
        assert not note.startswith(AUTO_PULL_TRIGGERS), note


def test_the_route_falls_back_to_the_snapshot_store(monkeypatch):
    """The config field is only written by chains that ran after it was
    added -- which, when it shipped, was none of them. Sixteen snapshots
    from previous pulls were already in the table, so the option would
    have sat greyed with its own answer sitting one query away.
    """
    from ppe_pipeline_api import pull_control as pc
    from ppe_pipeline_api.routers import track2_retention as r

    monkeypatch.setattr(pc, "read_config", lambda: {"last_pull_dataset": {}})
    monkeypatch.setattr(r.ds, "listing", lambda limit=50: [
        {"snapshot_id": "newer", "note": "queued run abc",
         "root": "retraining", "file_count": 10, "created_at": "2026-10-08"},
        {"snapshot_id": "chain", "note": "after-pull retrain",
         "root": "retraining", "file_count": 2214, "created_at": "2026-10-07"},
    ])
    out = r.last_pull_dataset(session=None)
    assert out["dataset"]["snapshot_id"] == "chain", (
        "a queued run is not an auto pull, however recent it is"
    )
    assert out["dataset"]["found_in"] == "snapshots"


def test_the_config_record_wins_when_it_exists(monkeypatch):
    from ppe_pipeline_api import pull_control as pc
    from ppe_pipeline_api.routers import track2_retention as r

    monkeypatch.setattr(pc, "read_config", lambda: {
        "last_pull_dataset": {"folder": "retraining", "snapshot_id": "fresh"}})
    called = {"listing": False}
    monkeypatch.setattr(r.ds, "listing",
                        lambda limit=50: called.__setitem__("listing", True) or [])
    out = r.last_pull_dataset(session=None)
    assert out["dataset"]["snapshot_id"] == "fresh"
    assert out["dataset"]["found_in"] == "config"
    assert not called["listing"], "no need to scan when the record is there"


def test_nothing_anywhere_is_reported_as_nothing(monkeypatch):
    from ppe_pipeline_api import pull_control as pc
    from ppe_pipeline_api.routers import track2_retention as r

    monkeypatch.setattr(pc, "read_config", lambda: {})
    monkeypatch.setattr(r.ds, "listing", lambda limit=50: [])
    assert r.last_pull_dataset(session=None)["dataset"] is None


def test_choosing_it_restores_the_snapshot():
    """A snapshot's folder IS the live root. Pointing at it without
    putting the frozen file lists back would train on whatever has been
    pulled since, under a label promising the pull's images."""
    code = _code(PAGE)
    block = code.split('option.key === "pull" && lastPull?.folder')[1][:900]
    assert "restoreTrack2Snapshot(lastPull.snapshot_id)" in block
    assert "refreshLiveDataset()" in block


def test_the_label_says_auto_pull():
    code = _code(PAGE)
    assert "Use snapshot generated during auto pull" in code
