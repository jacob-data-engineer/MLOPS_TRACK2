"""The class and model registry travels in the settings file.

A settings file used to name a detector by slug and carry nothing else
about it, so a file was only restorable on a machine that already had
that detector -- which is the one machine that did not need the file.

The rule, chosen deliberately: **the file can add, and that is all.**
This machine's detectors always win. A load must not be able to change
the classes of a detector a run is training against, nor delete one that
registered model packages are filed under -- the registry group name is
derived from the detector, and the approval rule matches those names
exactly.
"""
from __future__ import annotations

import json

import pytest

from ppe_pipeline_api import pull_control as pc
from ppe_pipeline_api.routers import track2_pull_control as router


# --- the file carries them ------------------------------------------------------

def test_the_snapshot_skips_the_builtin(monkeypatch):
    """It ships in the image. Every machine has it, and a stored copy of
    something the code defines is a copy that can go stale."""
    monkeypatch.setattr(
        "ppe_pipeline_api.track2_shared.list_detectors",
        lambda fresh=False: [
            {"slug": "ppe", "label": "PPE", "classes": {0: "person"}, "builtin": True},
            {"slug": "cow-model", "label": "Cows", "classes": {0: "cow"}},
        ],
    )
    snapshot = router._registry_snapshot()
    assert [d["slug"] for d in snapshot["detectors"]] == ["cow-model"]


def test_classes_survive_json(monkeypatch):
    """JSON has no integer keys, and a class list is index -> name. The
    round trip has to put the integers back or the restored detector
    trains the wrong labels."""
    monkeypatch.setattr(
        "ppe_pipeline_api.track2_shared.list_detectors",
        lambda fresh=False: [
            {"slug": "d", "label": "D", "classes": {0: "cow", 1: "sheep"}},
        ],
    )
    written = json.loads(json.dumps(router._registry_snapshot()))

    created: list = []
    monkeypatch.setattr("ppe_pipeline_api.track2_shared.list_detectors",
                        lambda fresh=False: [])
    monkeypatch.setattr("ppe_pipeline_api.track2_shared.upsert_detector",
                        lambda slug, label, classes: created.append((slug, label, classes)))

    router._restore_registry(written)
    assert created == [("d", "D", {0: "cow", 1: "sheep"})]


# --- what a load may and may not do ---------------------------------------------

@pytest.fixture
def registry(monkeypatch):
    """A live registry, and a record of what the restore changed."""
    live = [{"slug": "cow-model", "label": "Cows (live)", "classes": {0: "cow"}}]
    written: list = []
    monkeypatch.setattr("ppe_pipeline_api.track2_shared.list_detectors",
                        lambda fresh=False: list(live))
    monkeypatch.setattr("ppe_pipeline_api.track2_shared.upsert_detector",
                        lambda slug, label, classes: written.append(slug))

    def refuse_delete(*args, **kwargs):
        raise AssertionError("a settings file load must never delete a detector")

    monkeypatch.setattr("ppe_pipeline_api.track2_shared.delete_detector", refuse_delete)
    return written


def test_a_detector_the_file_has_and_the_machine_does_not_is_created(registry):
    out = router._restore_registry({"detectors": [
        {"slug": "helmet-v2", "label": "Helmets", "classes": {"0": "helmet"}},
    ]})
    assert out["created"] == ["helmet-v2"]
    assert registry == ["helmet-v2"]


def test_an_existing_detector_is_left_exactly_as_it_is(registry):
    """The file disagreeing about a detector this machine already has is
    the case that must not write. A run may be training against it."""
    out = router._restore_registry({"detectors": [
        {"slug": "cow-model", "label": "Cows (from the file)",
         "classes": {"0": "cow", "1": "calf"}},
    ]})
    assert out["kept"] == ["cow-model"]
    assert out["created"] == []
    assert registry == [], "the live detector was overwritten"


def test_a_detector_the_file_omits_is_not_deleted(registry):
    """`delete_detector` is wired to fail the test if it is called."""
    out = router._restore_registry({"detectors": []})
    assert out == {"created": [], "kept": [], "problems": []}


def test_a_detector_with_no_classes_is_refused_with_a_reason(registry):
    out = router._restore_registry({"detectors": [
        {"slug": "empty", "label": "Empty", "classes": {}},
    ]})
    assert registry == []
    assert out["problems"] and "no classes" in out["problems"][0]


def test_one_bad_entry_does_not_stop_the_others(registry):
    out = router._restore_registry({"detectors": [
        {"slug": "broken", "label": "B", "classes": {"nope": "x"}},
        {"slug": "good", "label": "G", "classes": {"0": "thing"}},
    ]})
    assert out["created"] == ["good"]
    assert out["problems"]


def test_a_file_with_no_registry_section_is_fine(registry):
    """Every settings file written before this existed."""
    assert router._restore_registry({}) == {"created": [], "kept": [], "problems": []}


# --- the file format ------------------------------------------------------------

def test_sections_sit_beside_config_not_inside_it(tmp_path, monkeypatch):
    """`load_settings_file` writes `config` straight into the live
    configuration. A registry copy inside that dict would be written into
    the live config on every load, where nothing reads it, and
    `write_config` would then carry it forever."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    monkeypatch.setattr(pc, "write_config", lambda patch: dict(patch))
    monkeypatch.setattr(pc, "settings_owner", lambda *a, **k: "")

    pc.save_settings_file(name="f", config={"root": "retraining"},
                          sections={"registry": {"detectors": [{"slug": "d"}]}})

    body = json.loads((tmp_path / "f.json").read_text(encoding="utf-8"))
    assert "registry" not in body["config"]
    assert body["registry"]["detectors"] == [{"slug": "d"}]


def test_reading_a_section_from_a_file_without_one_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    (tmp_path / "old.json").write_text(
        json.dumps({"saved_at": "x", "config": {"root": "r"}}), encoding="utf-8")
    assert pc.read_settings_section("old", "registry") == {}


def test_reading_a_section_from_a_missing_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    assert pc.read_settings_section("nope", "registry") == {}
