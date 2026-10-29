"""Each tab saves its own settings file, with its own schedule (2026-10-29).

Data Preparation saves `<name>_DP` holding only the data side and the
PULL schedule; Multi-Auto Training saves `<name>_AT` holding only the
training side and the TRAINING schedule. Saving, loading or resetting one
tab must leave the other's live settings exactly as they were.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ppe_pipeline_api import pull_control as pc
from ppe_pipeline_api.chain import queue_after_pull as qap
from ppe_pipeline_api.chain import retrain_schedule as rs

from test_settings_roundtrip import FakeDynamo

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"


@pytest.fixture
def live(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(pc, "ddb", FakeDynamo())
    armed = []
    monkeypatch.setattr(pc, "set_mode", lambda **kw: armed.append(kw) or {"problems": []})
    pc.write_config({
        "root": "retraining", "mode": "schedule", "schedule_time": "02:00",
        "golden_set": {"source": "gold/"},
        "default_detector": {"slug": "helmet", "architecture": "yolo11s"},
        "retrain_schedule": {"trigger": "clock", "every_days": 3, "at": "04:00"},
        "chain_state": {"running": True},
        "card_state": {"pull_control": True, "training_advanced": True},
    })
    return armed


def _file(name: str) -> dict:
    return json.loads(pathlib.Path(pc.SETTINGS_DIR, f"{name}.json").read_text())


def test_each_tab_saves_only_its_own_settings_under_its_own_ending(live):
    assert pc.save_settings_file(name="site", scope="dp")["name"] == "site_DP"
    assert pc.save_settings_file(name="site", scope="at")["name"] == "site_AT"

    dp, at = _file("site_DP")["config"], _file("site_AT")["config"]
    assert dp["root"] == "retraining" and dp["mode"] == "schedule"
    assert "default_detector" not in dp and "retrain_schedule" not in dp
    assert at["default_detector"]["slug"] == "helmet"
    assert at["retrain_schedule"]["trigger"] == "clock"
    assert "root" not in at and "mode" not in at
    # Run state is a result, saved by neither.
    assert "chain_state" not in dp and "chain_state" not in at
    # Each file carries only its own cards' open state.
    assert dp["card_state"] == {"pull_control": True}
    assert at["card_state"] == {"training_advanced": True}

    config = pc.read_config()
    assert config["settings_file_dp"] == "site_DP"
    assert config["settings_file_at"] == "site_AT"


def test_the_ending_is_added_once():
    assert pc.scoped_name("site_DP", "dp") == "site_DP"
    assert pc.scoped_name("site_dp", "dp") == "site_DP"
    assert pc.scoped_name("site", "at") == "site_AT"
    with pytest.raises(ValueError):
        pc.scoped_name("_AT", "at")


def test_loading_one_tab_leaves_the_other_alone(live):
    pc.save_settings_file(name="before", scope="dp")
    pc.write_config({"root": "elsewhere", "mode": "off",
                     "default_detector": {"slug": "vest", "architecture": "yolo11m"},
                     "card_state": {"pull_control": False, "training_advanced": False}})

    pc.load_settings_file(name="before_DP", scope="dp")
    config = pc.read_config()
    assert config["root"] == "retraining" and config["mode"] == "schedule"
    assert config["default_detector"]["slug"] == "vest"          # untouched
    assert config["card_state"] == {"pull_control": True, "training_advanced": False}
    assert live, "the pull's trigger is re-armed on a Data Preparation load"


def test_a_training_load_does_not_touch_the_pull(live):
    pc.save_settings_file(name="t", scope="at")
    pc.write_config({"mode": "arrival",
                     "default_detector": {"slug": "vest", "architecture": "yolo11m"}})
    live.clear()

    pc.load_settings_file(name="t_AT", scope="at")
    config = pc.read_config()
    assert config["default_detector"]["slug"] == "helmet"
    assert config["mode"] == "arrival"
    assert not live, "no EventBridge rule is touched by a training load"


def test_a_file_from_the_other_tab_is_refused(live):
    pc.save_settings_file(name="t", scope="at")
    with pytest.raises(ValueError, match="Multi-Auto Training"):
        pc.load_settings_file(name="t_AT", scope="dp")


def test_an_old_whole_page_file_loads_one_half_at_a_time(live):
    pc.save_settings_file(name="old")                # no scope: both pages
    pc.write_config({"root": "elsewhere",
                     "default_detector": {"slug": "vest", "architecture": "yolo11m"}})
    pc.load_settings_file(name="old", scope="at")
    config = pc.read_config()
    assert config["default_detector"]["slug"] == "helmet"
    assert config["root"] == "elsewhere"


def test_new_resets_one_tab_only(live):
    pc.reset_to_defaults(scope="at")
    config = pc.read_config()
    assert config["default_detector"] == {}
    assert config["root"] == "retraining" and config["mode"] == "schedule"
    assert config["card_state"] == {"pull_control": True}

    pc.reset_to_defaults(scope="dp")
    config = pc.read_config()
    assert config["mode"] == "off" and config["golden_set"] == {}


def test_the_listing_says_which_tab_each_file_is_for(live):
    pc.save_settings_file(name="a", scope="dp")
    pc.save_settings_file(name="a", scope="at")
    pc.save_settings_file(name="legacy")
    scopes = {f["name"]: f["scope"] for f in pc.list_settings_files()["files"]}
    assert scopes == {"a_DP": "dp", "a_AT": "at", "legacy": ""}


def test_every_setting_belongs_to_at_most_one_tab():
    assert not set(pc.SCOPES["dp"]) & set(pc.SCOPES["at"])


# --- the training schedule is Multi-Auto Training's own ---------------------

def test_an_old_config_still_trains_after_each_pull():
    """No `trigger` stored reads as it always behaved."""
    assert rs.training_trigger({"retrain_schedule": {"enabled": True, "follow_pull": True}}) == "pull"
    assert rs.training_trigger({}) == "pull"
    assert rs.training_trigger({"retrain_schedule": {"follow_pull": False, "enabled": True}}) == "clock"


@pytest.mark.parametrize("trigger", ["clock", "off"])
def test_the_pull_does_not_train_unless_training_follows_it(trigger):
    config = {"mode": "schedule",
              "retrain_schedule": rs.trigger_fields(trigger)}
    decision = qap.pending(config)
    assert decision["act"] is False
    assert "Multi-Auto Training" in decision["reason"]


def test_its_own_clock_runs_without_the_pull():
    config = {"mode": "off", "retrain_schedule": rs.trigger_fields("clock")}
    assert rs.follows_pull(config) is False
    assert rs.settings(config)["enabled"] is True


# --- the screen ------------------------------------------------------------

def test_each_page_has_its_own_toolbar():
    prep = (WEB / "routes" / "Track2DataPrepPage.tsx").read_text(encoding="utf-8")
    models = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    bar = (WEB / "components" / "Track2PinnedBar.tsx").read_text(encoding="utf-8")
    assert '<Track2PinnedBar scope="dp" />' in prep
    assert '<Track2PinnedBar scope="at" />' in models
    assert "saveTrack2SettingsFile(target, allowClash, scope)" in bar
    assert "loadTrack2SettingsFile(picked, true, scope)" in bar
    assert "newTrack2Settings(scope)" in bar
    assert 'const SUFFIX: Record<Track2SettingsScope, string> = { dp: "_DP", at: "_AT" };' in bar


def test_each_tab_lists_only_its_own_files():
    """Old whole-page files are offered on neither tab (2026-10-29)."""
    bar = (WEB / "components" / "Track2PinnedBar.tsx").read_text(encoding="utf-8")
    assert "const mine = out.files.filter((f) => f.scope === scope);" in bar
    assert "old, both tabs" not in bar
