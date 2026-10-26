"""A queued run has to say which frozen dataset it trains on.

Two entries can carry the same folder, the same epochs and the same seed
and still be different runs: one pinned to a snapshot, one following the
live corpus. The chain's own entries recorded the snapshot id; a
queue-add from the training card did not, so a run on a deliberately
chosen snapshot was indistinguishable afterwards from one on whatever the
week folders held that day.

Captured at ADD time, not at run time. An entry can sit in the queue for
hours while the live slot is restored to something else, and reading it
when the run starts would name whatever happened to be current then.
"""
from __future__ import annotations

import importlib

import pytest

models = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
pc = importlib.import_module("src.ppe_pipeline_api.pull_control")

CHOSEN = {
    "folder": "retraining",
    "snapshot_id": "f458acb0db7fcb7b",
    "files": 748,
    "set_by": "snapshot",
    "snapshot_created_at": "2026-09-30T10:53:10+00:00",
    "snapshot_note": "after-pull retrain",
    "pinned": True,
    "at": "2026-10-06T07:55:15+00:00",
}


@pytest.fixture
def chosen(monkeypatch):
    def use(value):
        monkeypatch.setattr(pc, "read_config", lambda: {"default_dataset": value})
    return use


def _entry(**over):
    base = {"id": "e1", "dataset_folder": "retraining", "notes": ""}
    base.update(over)
    return base


def test_the_snapshot_is_stamped_onto_the_entry(chosen):
    chosen(CHOSEN)
    entry = _entry()
    models._stamp_snapshot(entry)
    assert entry["snapshot_id"] == "f458acb0db7fcb7b"
    assert entry["snapshot_created_at"] == "2026-09-30T10:53:10+00:00"
    assert entry["snapshot_files"] == 748
    assert entry["snapshot_pinned"] is True
    assert entry["snapshot_note"] == "after-pull retrain"


def test_it_reaches_the_notes_too(chosen):
    """Notes is the field that reaches the registry. A package saying
    only "1 epoch" cannot be traced back to the images it saw."""
    chosen(CHOSEN)
    entry = _entry(notes="This trains from version 2")
    models._stamp_snapshot(entry)
    assert "This trains from version 2" in entry["notes"]
    assert "f458acb0db7fcb7b" in entry["notes"]


def test_empty_notes_do_not_start_with_a_dash(chosen):
    chosen(CHOSEN)
    entry = _entry(notes="")
    models._stamp_snapshot(entry)
    assert entry["notes"] == "frozen snapshot f458acb0db7fcb7b"


def test_the_id_is_not_added_twice(chosen):
    """Notes that already name it -- the chain writes them that way."""
    chosen(CHOSEN)
    entry = _entry(notes="after-pull retrain from snapshot f458acb0db7fcb7b")
    models._stamp_snapshot(entry)
    assert entry["notes"].count("f458acb0db7fcb7b") == 1


def test_a_different_folder_is_NOT_stamped(chosen):
    """A snapshot restored into `retraining` says nothing about a run
    pointed somewhere else. Stamping it there would be a label that is
    untrue, which is worse than none because it reads as provenance."""
    chosen(CHOSEN)
    entry = _entry(dataset_folder="some_other_dataset")
    models._stamp_snapshot(entry)
    assert "snapshot_id" not in entry
    assert entry["notes"] == ""


def test_a_trailing_slash_does_not_defeat_the_match(chosen):
    chosen({**CHOSEN, "folder": "retraining/"})
    entry = _entry(dataset_folder="retraining")
    models._stamp_snapshot(entry)
    assert entry["snapshot_id"] == "f458acb0db7fcb7b"


def test_a_dataset_the_chain_built_is_not_called_a_snapshot(chosen):
    """`set_by` is the discriminator. The chain freezes a snapshot on
    every run, so the id alone would mark every live-corpus run as
    frozen."""
    chosen({**CHOSEN, "set_by": "after-pull"})
    entry = _entry()
    models._stamp_snapshot(entry)
    assert "snapshot_id" not in entry


def test_an_unpinned_snapshot_is_recorded_as_such(chosen):
    """It still names week folders, so it is not repeatable -- and the
    card says so rather than implying it is."""
    chosen({**CHOSEN, "pinned": False})
    entry = _entry()
    models._stamp_snapshot(entry)
    assert entry["snapshot_pinned"] is False


def test_nothing_selected_at_all(chosen):
    chosen({})
    entry = _entry()
    models._stamp_snapshot(entry)
    assert "snapshot_id" not in entry


def test_a_broken_config_read_does_not_stop_the_queue_add(monkeypatch):
    """Annotation. An entry that queues without its snapshot id is a
    smaller problem than one that does not queue."""
    def boom():
        raise RuntimeError("ddb is having a day")
    monkeypatch.setattr(pc, "read_config", boom)
    entry = _entry()
    models._stamp_snapshot(entry)
    assert entry == _entry()


def test_queue_add_stamps_before_it_queues():
    """After `queue_add` the entry is already in the list, and the copy
    held there would not carry the fields."""
    import inspect
    source = inspect.getsource(models.add_to_queue)
    assert "_stamp_snapshot(entry)" in source
    assert source.index("_stamp_snapshot(entry)") < source.index("t2.queue_add(entry)")


def test_a_rebuilt_live_yaml_is_not_stamped_as_a_snapshot(chosen):
    """Choosing "Use live data" rebuilds the yaml from the week folders,
    which un-restores the snapshot. The record follows, so a run queued
    afterwards is not labelled with a frozen dataset it does not use --
    a false provenance label is worse than none."""
    chosen({**CHOSEN, "set_by": "rebuilt"})
    entry = _entry()
    models._stamp_snapshot(entry)
    assert "snapshot_id" not in entry
    assert entry["notes"] == ""


def test_the_rebuild_clears_the_marker():
    import inspect
    router = importlib.import_module("src.ppe_pipeline_api.routers.track2_retention")
    source = inspect.getsource(router.save_dataset_yaml)
    assert '"set_by": "rebuilt"' in source
    # Only when a snapshot WAS live -- rewriting the record on every save
    # would overwrite what the chain had just set about its own dataset.
    assert 'set_by") or "") == "snapshot"' in source


def test_the_audit_records_the_class_and_the_snapshot():
    """So the history answers both outright. The registry group implies
    the Class, but only to somebody who knows how those names are built,
    and a run that never registers has no group at all. The snapshot was
    reaching the history only inside the free-text notes -- a habit, not
    a record: nothing guaranteed it and nothing could filter on it."""
    import inspect
    t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")
    source = inspect.getsource(t2.write_training_audit)
    assert '"detector_slug": str(detector_slug or "")' in source
    assert '"snapshot": dict(snapshot or {})' in source


def test_the_queue_runner_passes_the_entrys_own_snapshot():
    """From the ENTRY, not from the live slot -- which may have been
    restored to something else between queueing and running."""
    import inspect
    t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")
    source = inspect.getsource(t2.try_advance_queue)
    assert 'entry.get("snapshot_id")' in source
    assert "snapshot=" in source


def test_a_run_with_no_snapshot_records_an_empty_block():
    """Not None: the history card tests `snapshot?.snapshot_id`, and a
    missing key and an empty one must read the same."""
    import inspect
    t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")
    source = inspect.getsource(t2.write_training_audit)
    assert "dict(snapshot or {})" in source


def test_a_live_data_run_gets_a_snapshot_frozen_for_it(chosen, monkeypatch):
    """The gap this closes. The CHAIN freezes a snapshot as one of its
    steps, so its runs could always answer "which images was this
    trained on". A queue-add from the training card never did: the
    history showed an empty snapshot block and a folder name that had
    moved on since, and asked of a finished run the question had no
    answer at all."""
    chosen({})
    ds = importlib.import_module("src.ppe_pipeline_api.dataset_snapshots")
    monkeypatch.setattr(ds, "create", lambda **kw: {
        "snapshot_id": "fresh123", "created_at": "2026-10-06T10:30:00+00:00",
        "file_count": 1107, "names_from": "folder"})
    monkeypatch.setattr(pc, "read_config", lambda: {})
    entry = _entry(notes="This trains from version 2")
    models._stamp_snapshot(entry)
    models._freeze_for_entry(entry)
    assert entry["snapshot_id"] == "fresh123"
    assert entry["snapshot_files"] == 1107
    assert "fresh123" in entry["notes"]
    assert "This trains from version 2" in entry["notes"]


def test_a_chosen_snapshot_is_not_overwritten_by_a_fresh_freeze(chosen, monkeypatch):
    """Freezing over a deliberately restored one would silently discard
    the choice -- and name the new snapshot as this run's provenance."""
    chosen(CHOSEN)
    ds = importlib.import_module("src.ppe_pipeline_api.dataset_snapshots")
    monkeypatch.setattr(ds, "create", lambda **kw: pytest.fail("must not freeze over a choice"))
    entry = _entry()
    models._stamp_snapshot(entry)
    models._freeze_for_entry(entry)
    assert entry["snapshot_id"] == "f458acb0db7fcb7b"


def test_a_failed_freeze_does_not_stop_the_queue_add(monkeypatch):
    """A run that queues without a frozen record is worse than one with
    it, and far better than one that will not queue."""
    ds = importlib.import_module("src.ppe_pipeline_api.dataset_snapshots")

    def boom(**kw):
        raise RuntimeError("S3 is having a day")
    monkeypatch.setattr(ds, "create", boom)
    monkeypatch.setattr(pc, "read_config", lambda: {})
    entry = _entry()
    models._freeze_for_entry(entry)
    assert "snapshot_id" not in entry


def test_nothing_is_frozen_without_a_dataset_folder(monkeypatch):
    ds = importlib.import_module("src.ppe_pipeline_api.dataset_snapshots")
    monkeypatch.setattr(ds, "create", lambda **kw: pytest.fail("nothing to freeze"))
    entry = _entry(dataset_folder="")
    models._freeze_for_entry(entry)
    assert "snapshot_id" not in entry


def test_queue_add_freezes_after_it_checks_for_a_chosen_one():
    """Order matters: freezing first would always win, and the
    deliberately restored snapshot would never be used."""
    import inspect
    source = inspect.getsource(models.add_to_queue)
    assert source.index("_stamp_snapshot(entry)") < source.index("_freeze_for_entry(entry)")
    assert source.index("_freeze_for_entry(entry)") < source.index("t2.queue_add(entry)")
