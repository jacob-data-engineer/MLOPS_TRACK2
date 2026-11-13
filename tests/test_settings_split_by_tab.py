"""Data Preparation owns recipes; Multi-Auto Training owns both schedules."""
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
    assert dp["root"] == "retraining" and "mode" not in dp
    assert "default_detector" not in dp and "retrain_schedule" not in dp
    assert at["default_detector"]["slug"] == "helmet"
    assert at["retrain_schedule"]["trigger"] == "clock"
    assert "root" not in at and at["mode"] == "schedule"
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
    assert config["root"] == "retraining" and config["mode"] == "off"
    assert config["default_detector"]["slug"] == "vest"          # untouched
    assert config["card_state"] == {"pull_control": True, "training_advanced": False}
    assert not live, "Data Preparation must not re-arm schedules"


def test_a_training_load_restores_the_pull_schedule(live):
    pc.save_settings_file(name="t", scope="at")
    pc.write_config({"mode": "arrival",
                     "default_detector": {"slug": "vest", "architecture": "yolo11m"}})
    live.clear()

    pc.load_settings_file(name="t_AT", scope="at")
    config = pc.read_config()
    assert config["default_detector"]["slug"] == "helmet"
    assert config["mode"] == "schedule"
    assert live[-1]["mode"] == "schedule"


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
    assert config["root"] == "retraining" and config["mode"] == "off"
    assert config["card_state"] == {"pull_control": True}

    pc.write_config({"mode": "arrival"})
    live.clear()
    pc.reset_to_defaults(scope="dp")
    config = pc.read_config()
    assert config["mode"] == "arrival" and config["golden_set"] == {}
    assert not live


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


def test_the_central_data_store_is_a_saved_data_preparation_setting():
    from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest
    assert pc.DEFAULTS["central_store"] == {}
    assert "central_store" in ConfigRequest.model_fields
    assert "central_store" in pc.SCOPES["dp"]
    assert "central data store" in pc.DP_CARDS
    panel = (WEB / "components" / "Track2CentralStorePanel.tsx").read_text(encoding="utf-8")
    assert "saveTrack2PullConfig({ central_store: { bucket, folder, train_from_combined: trainFromCombined } })" in panel
    page = (WEB / "routes" / "Track2DataPrepPage.tsx").read_text(encoding="utf-8")
    assert page.index("<Track2AutoTrainingPanel />") < page.index("<Track2CentralStorePanel />")


def test_legacy_training_file_preserves_pull_schedule(live):
    path = pathlib.Path(pc.SETTINGS_DIR)
    path.mkdir(parents=True, exist_ok=True)
    (path / "legacy_AT.json").write_text(json.dumps({"config": {"default_training": {"epochs": 2}}}))
    pc.load_settings_file(name="legacy_AT", scope="at")
    assert pc.read_config()["mode"] == "schedule"
    assert not live


# --- UAT-03 (2026-11-04): a loaded file puts this tab's cards back -------------

def test_a_card_left_at_its_default_returns_to_it_on_load(live):
    """Saved with Assemble open BY DEFAULT (no key recorded), closed later,
    loaded: Assemble must open again -- the closed toggle may not survive."""
    pc.write_config({"card_state": {"training_advanced": True}})   # Assemble untouched
    pc.save_settings_file(name="uat03", scope="dp")
    pc.write_config({"card_state": {"assemble": False, "training_advanced": True}})
    pc.load_settings_file(name="uat03_DP", scope="dp")
    cards = pc.read_config()["card_state"]
    assert "assemble" not in cards                    # back to its default (open)
    assert cards["training_advanced"] is True         # the other tab untouched


def test_explicit_card_states_round_trip_in_both_tabs(live):
    pc.write_config({"card_state": {"golden_set": True, "pull_control": False,
                                    "training_advanced": False}})
    pc.save_settings_file(name="r", scope="dp")
    pc.save_settings_file(name="r", scope="at")
    pc.write_config({"card_state": {"golden_set": False, "pull_control": True,
                                    "training_advanced": True}})
    pc.load_settings_file(name="r_DP", scope="dp")
    assert pc.read_config()["card_state"] == {"golden_set": True, "pull_control": False,
                                              "training_advanced": True}
    pc.load_settings_file(name="r_AT", scope="at")
    assert pc.read_config()["card_state"] == {"golden_set": True, "pull_control": False,
                                              "training_advanced": False}


# --- UAT-05 (2026-11-04): a deleted file is not "loaded" any more ----------------

def test_deleting_the_loaded_file_clears_only_its_name(live):
    pc.save_settings_file(name="gone", scope="at")
    pc.save_settings_file(name="kept", scope="dp")
    before = pc.read_config()
    assert before["settings_file_at"] == "gone_AT" and before["settings_file_dp"] == "kept_DP"
    out = pc.delete_settings_file(name="gone_AT")
    after = pc.read_config()
    assert out["cleared"] == ["settings_file_at"]
    assert after["settings_file_at"] == "" and after["settings_file_dp"] == "kept_DP"
    # nothing else moved -- settings and schedules stay as they were
    for key in ("mode", "retrain_schedule", "default_detector", "root"):
        assert after.get(key) == before.get(key)


def test_deleting_a_file_that_is_not_loaded_changes_nothing(live):
    pc.save_settings_file(name="other", scope="dp")
    pc.save_settings_file(name="live", scope="dp")             # now the loaded one
    pc.delete_settings_file(name="other_DP")
    assert pc.read_config()["settings_file_dp"] == "live_DP"


def test_scheduled_does_not_call_a_missing_file_loaded():
    import inspect
    from ppe_pipeline_api.routers import track2_retention as tr
    src = inspect.getsource(tr)
    assert "if live_name and live_name not in {f.get(\"name\") for f in" in src


# --- R4 (independent UAT, 2026-11-04): a delete is not undone by a save that
# read the settings before it ---------------------------------------------------

def test_a_save_that_read_before_a_delete_does_not_bring_the_name_back(live, monkeypatch):
    pc.save_settings_file(name="race", scope="dp")
    assert pc.read_config()["settings_file_dp"] == "race_DP"
    real_read = pc._read_config_item
    state = {"interleaved": False}

    def read_then_delete():
        got = real_read()
        if not state["interleaved"]:
            # The card's autosave has read; the delete lands before it writes.
            state["interleaved"] = True
            monkeypatch.setattr(pc, "_read_config_item", real_read)
            assert pc.delete_settings_file(name="race_DP")["cleared"] == ["settings_file_dp"]
            monkeypatch.setattr(pc, "_read_config_item", read_then_delete)
        return got

    monkeypatch.setattr(pc, "_read_config_item", read_then_delete)
    pc.write_config({"card_state": {"assemble": True}})          # the stale autosave
    after = pc.read_config()
    assert state["interleaved"]
    assert after["settings_file_dp"] == ""                       # the delete stands
    assert after["card_state"] == {"assemble": True}             # and so does the save


def test_a_write_that_keeps_losing_says_so_instead_of_overwriting(live, monkeypatch):
    from botocore.exceptions import ClientError

    def always_conflicts(**kw):
        raise ClientError({"Error": {"Code": "ConditionalCheckFailedException"}}, "PutItem")

    monkeypatch.setattr(pc.ddb, "put_item", always_conflicts)
    monkeypatch.setattr(pc.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError):
        pc.write_config({"mode": "manual"})


def test_a_delete_never_clears_a_file_loaded_while_it_ran(live, monkeypatch):
    pc.save_settings_file(name="old", scope="dp")
    real_read = pc._read_config_item
    state = {"interleaved": False}

    def read_then_load_another():
        got = real_read()
        if not state["interleaved"]:
            state["interleaved"] = True
            monkeypatch.setattr(pc, "_read_config_item", real_read)
            pc.write_config({"settings_file_dp": "new_DP"})       # loaded meanwhile
            monkeypatch.setattr(pc, "_read_config_item", read_then_load_another)
        return got

    monkeypatch.setattr(pc, "_read_config_item", read_then_load_another)
    out = pc.delete_settings_file(name="old_DP")
    assert state["interleaved"] and out["cleared"] == []
    assert pc.read_config()["settings_file_dp"] == "new_DP"
