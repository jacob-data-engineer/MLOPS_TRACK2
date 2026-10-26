"""Renaming a Model Name, and what a rename can and cannot move.

The identifier derives the registry group name. **A model package group
cannot be renamed** -- SageMaker exposes create, delete, describe and
list for them, and no update. That is an AWS limit, not ours, and it
decides the whole design:

  * future runs register into the NEW groups;
  * the old identifier is remembered, so versions already registered
    stay findable under the name they were filed under;
  * nothing is deleted and no weights move.

And the part that makes it a rename rather than a rename plus five
silent breakages: the live configuration is retargeted, so the training
form, the two cards, the unattended retrain and anything queued follow
the new name.
"""
from __future__ import annotations

import pytest

from ppe_pipeline_api import track2_shared as t2


@pytest.fixture
def registry(monkeypatch):
    """A detector table in memory, plus a record of what was written."""
    rows = {
        "cow-model": {
            "id": "detector#cow-model", "slug": "cow-model",
            "record_type": "detector", "label": "Cow Model",
            "classes": {"0": "cow"},
            "group_pattern": "yolo{generation}{size}-cow-model-detector",
        },
    }
    written: list = []
    deleted: list = []
    ensured: list = []

    class _Table:
        def put_item(self, Item):            # noqa: N803
            written.append(Item)
            rows[Item["slug"]] = Item

        def delete_item(self, Key):          # noqa: N803
            deleted.append(Key["id"])
            rows.pop(Key["id"].removeprefix("detector#"), None)

    monkeypatch.setattr(t2, "training_audit_table", lambda: _Table())
    monkeypatch.setattr(t2, "_invalidate_detector_cache", lambda: None)
    monkeypatch.setattr(t2, "ensure_model_package_group",
                        lambda group, description="": ensured.append(group))
    monkeypatch.setattr(
        t2, "get_detector",
        lambda slug: t2._detector_from_item(rows[slug]) if slug in rows else None)
    return {"rows": rows, "written": written, "deleted": deleted, "ensured": ensured}


# --- what it refuses ------------------------------------------------------------

def test_the_builtin_cannot_be_renamed(registry):
    """It owns the production registries, and those four names are the
    exact-match list the approval rule watches."""
    with pytest.raises(ValueError, match="built-in"):
        t2.rename_detector("ppe", "something-else")


def test_an_invalid_identifier_is_refused(registry):
    for bad in ("Has Capitals", "has spaces", "-starts-with-hyphen", ""):
        with pytest.raises(ValueError):
            t2.rename_detector("cow-model", bad)


def test_renaming_onto_an_existing_one_is_refused(registry):
    registry["rows"]["sheep"] = {
        "id": "detector#sheep", "slug": "sheep", "record_type": "detector",
        "label": "Sheep", "classes": {"0": "sheep"},
    }
    with pytest.raises(ValueError, match="already a Model Name"):
        t2.rename_detector("cow-model", "sheep")


def test_renaming_something_that_does_not_exist_is_refused(registry):
    with pytest.raises(ValueError, match="No Model Name"):
        t2.rename_detector("ghost", "whatever")


# --- what it does ---------------------------------------------------------------

def test_the_old_identifier_is_remembered(registry):
    out = t2.rename_detector("cow-model", "cattle", "Cattle")
    assert out["slug"] == "cattle"
    assert out["label"] == "Cattle"
    assert "cow-model" in out["previous_slugs"]


def test_the_classes_survive_unchanged(registry):
    """A rename is a name change. Touching the class list would silently
    retrain everything against different labels."""
    out = t2.rename_detector("cow-model", "cattle")
    assert out["classes"] == {0: "cow"}


def test_the_new_row_is_written_before_the_old_is_removed(registry):
    """A crash between the two should leave BOTH, which is visible and
    fixable. The other order leaves neither."""
    t2.rename_detector("cow-model", "cattle")
    assert registry["written"], "the new row was never written"
    assert registry["deleted"] == ["detector#cow-model"]


def test_the_new_groups_are_created(registry):
    t2.rename_detector("cow-model", "cattle")
    assert any("cattle-detector" in g for g in registry["ensured"])


def test_renaming_to_the_same_name_keeps_the_row(registry):
    """Used to change only the label. It must not delete the row it just
    wrote."""
    out = t2.rename_detector("cow-model", "cow-model", "Cows, renamed")
    assert out["label"] == "Cows, renamed"
    assert registry["deleted"] == []


# --- the old versions stay findable ---------------------------------------------

def test_history_groups_include_the_old_name(registry):
    renamed = t2.rename_detector("cow-model", "cattle")
    current = {g for _, _, g in t2.detector_groups(renamed)}
    history = {g for _, _, g in t2.detector_history_groups(renamed)}
    assert "yolo11s-cattle-detector" in current
    assert "yolo11s-cow-model-detector" not in current, (
        "a new run must never register into the old group"
    )
    assert "yolo11s-cow-model-detector" in history, (
        "versions trained under the old name must stay findable"
    )
    assert current < history


def test_a_detector_that_was_never_renamed_has_no_extra_groups(registry):
    plain = t2.get_detector("cow-model")
    assert ({g for _, _, g in t2.detector_groups(plain)}
            == {g for _, _, g in t2.detector_history_groups(plain)})


# --- the cascade ----------------------------------------------------------------

@pytest.fixture
def config(monkeypatch):
    """The live configuration, and what the cascade writes back."""
    from ppe_pipeline_api import pull_control as pc

    live = {
        "default_detector": {"slug": "cow-model", "architecture": "yolo11s"},
        "dataset_assemble": {"detector_slug": "cow-model", "overwrite": False},
        "import_checkpoint": {"detector_slug": "someone-else"},
        "retrain_schedule": {"detector_slug": "cow-model"},
    }
    monkeypatch.setattr(pc, "read_config", lambda: dict(live))
    monkeypatch.setattr(pc, "write_config", lambda patch: live.update(patch) or live)
    return live


def test_the_rename_follows_through_the_workflow(config):
    changed = t2.retarget_detector_references("cow-model", "cattle")
    assert config["default_detector"]["slug"] == "cattle"
    assert config["dataset_assemble"]["detector_slug"] == "cattle"
    assert config["retrain_schedule"]["detector_slug"] == "cattle"
    assert changed, "the card needs something to show"


def test_something_pointing_at_a_different_model_is_left_alone(config):
    t2.retarget_detector_references("cow-model", "cattle")
    assert config["import_checkpoint"]["detector_slug"] == "someone-else"


def test_other_settings_in_the_same_section_survive(config):
    """The cascade rewrites one field, not the section."""
    t2.retarget_detector_references("cow-model", "cattle")
    assert config["dataset_assemble"]["overwrite"] is False
    assert config["default_detector"]["architecture"] == "yolo11s"


def test_renaming_to_the_same_name_changes_nothing(config):
    assert t2.retarget_detector_references("cow-model", "cow-model") == []


def test_queued_runs_follow_the_new_name(monkeypatch, config):
    """An entry submitted against a slug that no longer resolves fails at
    the preflight, hours after anybody was watching."""
    monkeypatch.setattr(t2, "_persist_queue_entry", lambda entry: None)
    t2._training_queue[:] = [
        {"id": "a", "detector_slug": "cow-model"},
        {"id": "b", "detector_slug": "other"},
    ]
    try:
        t2.retarget_detector_references("cow-model", "cattle")
        assert t2._training_queue[0]["detector_slug"] == "cattle"
        assert t2._training_queue[1]["detector_slug"] == "other"
    finally:
        t2._training_queue[:] = []
