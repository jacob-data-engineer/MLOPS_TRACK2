import json
from copy import deepcopy

import pytest

from ppe_pipeline_api import pull_control as pc


def setup_recipe(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    body = {"config": {"mode": "schedule", "schedule_time": "03:00",
                       "sources": [], "picked": [],
                       "dataset_assemble": {"use": True, "only": True,
                                            "train_sources": [{"prefix": "chosen"}],
                                            "built": {"folder": "stale"}}}}
    (tmp_path / "sample_DP.json").write_text(json.dumps(body))
    live = deepcopy(pc.DEFAULTS)
    live.update(mode="off", schedule_time="12:00", data_source="snapshot",
                sources=[{"prefix": "previous"}],
                retrain_schedule={"enabled": False}, queue_after_pull={"enabled": False})
    monkeypatch.setattr(pc, "read_config", lambda: deepcopy(live))
    monkeypatch.setattr(pc, "write_config", lambda update: live.update(update) or deepcopy(live))
    monkeypatch.setattr(pc, "set_mode", lambda **kw: pytest.fail("Schedule must not be changed"))
    return live


def test_preview_does_not_change_live_config(tmp_path, monkeypatch):
    live = setup_recipe(tmp_path, monkeypatch)
    before = deepcopy(live)
    assert not pc.load_training_data_file(name="sample_DP")["applied"]
    assert live == before


def test_import_selects_recipe_and_preserves_clocks(tmp_path, monkeypatch):
    live = setup_recipe(tmp_path, monkeypatch)
    before = deepcopy(live)
    pc.load_training_data_file(name="sample_DP", apply=True)
    for key in ("mode", "schedule_time", "retrain_schedule", "queue_after_pull"):
        assert live[key] == before[key]
    assert live["sources"] == []
    assert live["data_source"] == "assembled"
    assert "built" not in live["dataset_assemble"]
    assert live["dataset_assemble"]["train_sources"] == [{"prefix": "chosen"}]


def test_training_file_roundtrip_preserves_edited_recipe(tmp_path, monkeypatch):
    live = setup_recipe(tmp_path, monkeypatch)
    pc.load_training_data_file(name="sample_DP", apply=True)
    live["dataset_assemble"]["train_sources"] = [{"prefix": "edited"}]
    saved = pc.scope_slice(live, "at")
    (tmp_path / "training_AT.json").write_text(json.dumps({"config": saved}))
    live["dataset_assemble"] = {}
    monkeypatch.setattr(pc, "set_mode", lambda **kw: {"problems": []})
    pc.load_settings_file(name="training_AT", scope="at", apply=True)
    assert live["dataset_assemble"]["train_sources"] == [{"prefix": "edited"}]
    assert live["mode"] == "off"


def test_reject_other_scope():
    with pytest.raises(ValueError, match="Data Preparation"):
        pc.load_training_data_file(name="training_AT", apply=True)
