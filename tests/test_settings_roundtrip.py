"""A settings file must come back holding the run it was saved from.

The reported failure was "I selected version 1 of the helmet Class at
yolo11s, saved it in meow, and on reload it was not recorded". The file
on disk turned out to be CORRECT -- it held both the Class and the
version -- so the loss was on the way back in, not on the way out. These
pin the server half, which must stay correct for the page half to have
anything to restore.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402


class FakeDynamo:
    """The config is one JSON blob in one DynamoDB item, so this is all of it."""

    def __init__(self):
        self.items: dict = {}

    @staticmethod
    def _key(Key):
        return (Key["kind"]["S"], Key["sort"]["S"])

    def get_item(self, TableName, Key, ConsistentRead=False):
        item = self.items.get(self._key(Key))
        return {"Item": item} if item else {}

    def put_item(self, TableName, Item):
        self.items[(Item["kind"]["S"], Item["sort"]["S"])] = Item


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path / "settings"))
    monkeypatch.setattr(pc, "ddb", FakeDynamo())
    # The trigger is re-applied on load; EventBridge is not the subject here.
    monkeypatch.setattr(pc, "set_mode", lambda **kw: {"problems": []})
    return tmp_path


# What the page writes when somebody picks a Class, a size and a version.
A_RUN = {
    "default_detector": {"slug": "helmet", "architecture": "yolo11s"},
    "default_version": (
        "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-helmet-detector/1"
    ),
    "default_source": "existing",
    "default_training": {
        "epochs": 50, "img_size": 640, "batch": 16,
        "seed": 42, "patience": 0, "notes": "",
    },
}


class TestRoundTrip:
    def test_the_class_and_the_version_both_come_back(self, settings_dir):
        pc.write_config(dict(A_RUN))
        pc.save_settings_file(name="meow")

        # Somebody changes their mind about everything.
        pc.write_config({
            "default_detector": {"slug": "vehicle", "architecture": "yolo26x"},
            "default_version": "",
            "default_source": "scratch",
        })

        pc.load_settings_file(name="meow", apply=True)
        live = pc.read_config()
        assert live["default_detector"] == A_RUN["default_detector"]
        assert live["default_version"] == A_RUN["default_version"]
        assert live["default_source"] == "existing"

    def test_the_version_is_not_dropped_when_only_it_changed(self, settings_dir):
        pc.write_config(dict(A_RUN))
        pc.save_settings_file(name="meow")
        pc.write_config({"default_version": ""})
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["default_version"] == A_RUN["default_version"]

    def test_the_training_shape_round_trips(self, settings_dir):
        # It used to save as {} unless a field had been edited and blurred,
        # so a file restored "whatever the page was showing" instead of
        # what was saved.
        pc.write_config(dict(A_RUN))
        pc.save_settings_file(name="meow")
        pc.write_config({"default_training": {"epochs": 999, "img_size": 320,
                                              "batch": 1, "seed": 1,
                                              "patience": 9, "notes": "other"}})
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["default_training"] == A_RUN["default_training"]

    def test_the_file_records_which_file_it_is(self, settings_dir):
        pc.write_config(dict(A_RUN))
        pc.save_settings_file(name="meow")
        pc.write_config({"settings_file": "something-else"})
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["settings_file"] == "meow"

    def test_a_file_renamed_on_disk_loads_as_its_own_name(self, settings_dir):
        import os
        import shutil

        pc.write_config(dict(A_RUN))
        pc.save_settings_file(name="meow")
        shutil.copy(pc._settings_path("meow"), pc._settings_path("copy"))
        assert os.path.exists(pc._settings_path("copy"))
        pc.load_settings_file(name="copy", apply=True)
        # Not "meow", which is what the copied file claims inside.
        assert pc.read_config()["settings_file"] == "copy"


class TestAutoTrainingRoot:
    """Auto Training's root follows the pull control's by default.

    The card used to seed its root once and then let its own saved value
    win forever. Change the pull's root afterwards and the card showed the
    old one while `run_chain` -- which reads `config["root"]` directly --
    used the new one. Two folders, one of them on screen, nothing saying
    they differed.
    """

    def test_the_flag_rides_along_in_a_settings_file(self, settings_dir):
        pc.write_config({
            "root": "retraining",
            "auto_training": {"root": "retraining", "excluded": [], "follow_pull": True},
        })
        pc.save_settings_file(name="meow")
        pc.write_config({"auto_training": {}})
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["auto_training"]["follow_pull"] is True

    def test_an_override_rides_along_too(self, settings_dir):
        pc.write_config({
            "root": "retraining",
            "auto_training": {"root": "somewhere_else", "excluded": [],
                              "follow_pull": False},
        })
        pc.save_settings_file(name="meow")
        pc.write_config({"auto_training": {}})
        pc.load_settings_file(name="meow", apply=True)
        saved = pc.read_config()["auto_training"]
        assert saved["follow_pull"] is False
        assert saved["root"] == "somewhere_else"

    def test_the_chain_reads_the_pull_root_not_the_cards(self, settings_dir):
        # Pins the asymmetry the card now has to tell the user about: an
        # override moves what is on screen and nothing else.
        pc.write_config({
            "root": "retraining",
            "auto_training": {"root": "somewhere_else", "follow_pull": False},
        })
        config = pc.read_config()
        assert config["root"] == "retraining"
        assert config["auto_training"]["root"] == "somewhere_else"


class TestNewBlankTemplate:
    """`New` puts the LIVE settings back to the shipped defaults.

    Not a form reset: every card reads the live config, so clearing only
    what is on screen would leave the old settings running underneath.
    """

    def test_it_restores_the_defaults(self, settings_dir):
        pc.write_config({"root": "somewhere_else", "mode": "schedule",
                         "week_prefix": "batch_"})
        out = pc.reset_to_defaults()
        assert out["config"]["root"] == pc.DEFAULTS["root"]
        assert out["config"]["mode"] == pc.DEFAULTS["mode"] == "off"
        assert out["config"]["week_prefix"] == pc.DEFAULTS["week_prefix"]

    def test_it_is_a_replace_not_a_merge(self, settings_dir):
        # The bug this guards: `write_config` keeps what the caller did not
        # mention, which is right everywhere else and wrong here. A stored
        # key with no default -- queue_after_pull is a real one -- would
        # survive into a "fresh start" and the new file would carry a
        # setting nobody chose.
        pc.write_config({"queue_after_pull": {"enabled": True}})
        assert pc.read_config().get("queue_after_pull") == {"enabled": True}
        pc.reset_to_defaults()
        assert not pc.read_config().get("queue_after_pull")

    def test_it_clears_the_picked_folders_and_the_file_pointer(self, settings_dir):
        pc.write_config({
            "picked": {"train_images": ["a/b"], "train_labels": [],
                       "val_images": [], "val_labels": []},
            "settings_file": "helmet_settings",
        })
        config = pc.reset_to_defaults()["config"]
        assert config["picked"] == pc.DEFAULTS["picked"]
        assert config["settings_file"] == ""

    def test_it_does_not_delete_saved_files(self, settings_dir):
        # "Start a new one", not "throw the old ones away".
        pc.write_config(dict(A_RUN))
        pc.save_settings_file(name="meow")
        pc.reset_to_defaults()
        assert "meow" in [f["name"] for f in pc.list_settings_files()["files"]]
        # ...and it still holds what it held.
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["default_version"] == A_RUN["default_version"]

    def test_every_default_key_comes_back(self, settings_dir):
        config = pc.reset_to_defaults()["config"]
        for key, value in pc.DEFAULTS.items():
            assert config[key] == value, key


class TestEveryCardsSettingsRideInTheFile:
    """A settings file is only useful if it holds every card's choices.

    Reported as "sections 1, 2 and 3 are not saved". Section 1 was fine.
    Section 2 kept its choices in the browser tab and nowhere else, so the
    file recorded the golden set's on/off switch and none of the folders
    it pointed at. Section 3 writes the config only when Apply is pressed,
    so saving with unapplied changes wrote the PREVIOUS settings and still
    said "Saved" -- handled in the toolbar, which now refuses.
    """

    def test_the_golden_set_card_has_somewhere_to_store_its_choices(self, settings_dir):
        assert "golden_set" in pc.DEFAULTS

    def test_the_golden_set_choices_round_trip(self, settings_dir):
        chosen = {
            "bucket": "track2-s3", "source": "golden/edge_cases",
            "target_bucket": "track2-s3", "target_folder": "golden_set",
            "split": True, "train_percent": 80,
        }
        pc.write_config({"golden_set": chosen})
        pc.save_settings_file(name="meow")
        pc.write_config({"golden_set": {}})
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["golden_set"] == chosen

    def test_a_new_template_clears_them(self, settings_dir):
        pc.write_config({"golden_set": {"source": "golden/edge_cases"}})
        assert pc.reset_to_defaults()["config"]["golden_set"] == {}

    def test_section_3s_folders_round_trip(self, settings_dir):
        # Pull Control's own settings were always storable; the bug was
        # that pressing the toolbar Save before Apply never put them here.
        picked = {"train_images": ["drop/a"], "train_labels": ["drop/b"],
                  "val_images": [], "val_labels": []}
        pc.write_config({"picked": picked, "week_cutoff": "2026-09-01T00:00:00+00:00"})
        pc.save_settings_file(name="meow")
        pc.write_config({"picked": pc.DEFAULTS["picked"], "week_cutoff": ""})
        pc.load_settings_file(name="meow", apply=True)
        live = pc.read_config()
        assert live["picked"] == picked
        assert live["week_cutoff"] == "2026-09-01T00:00:00+00:00"

    def test_retention_rides_in_the_same_file(self, settings_dir):
        # Section 4b writes into the pull config under "retention", so it
        # was already covered -- pinned so a future move out of it fails.
        pc.write_config({"retention": {"enabled": True, "weekly_rate": 0.1}})
        pc.save_settings_file(name="meow")
        pc.write_config({"retention": {}})
        pc.load_settings_file(name="meow", apply=True)
        assert pc.read_config()["retention"] == {"enabled": True, "weekly_rate": 0.1}


class TestTheClashRefusalCanBeOverruled:
    """Two configs filing into one set of week folders is a warning.

    Usually a mistake -- two different source lists merging into one set
    of weeks with no record of which images came from where -- and
    occasionally exactly what was meant, such as saving a variant of a
    config being iterated on. So it refuses by default and can be
    overruled explicitly, never by default.
    """

    @staticmethod
    def _two_files_over_one_root():
        pc.write_config({
            "root": "retraining", "week_prefix": "week_",
            "picked": {"train_images": ["drop/a"], "train_labels": [],
                       "val_images": [], "val_labels": []},
        })
        pc.save_settings_file(name="test")
        # A different source list over the same root and prefix.
        pc.write_config({
            "picked": {"train_images": ["drop/b"], "train_labels": [],
                       "val_images": [], "val_labels": []},
        })

    def test_it_refuses_by_default(self, settings_dir):
        self._two_files_over_one_root()
        with pytest.raises(ValueError) as caught:
            pc.save_settings_file(name="hello")
        assert "already writes" in str(caught.value)

    def test_the_override_saves_it(self, settings_dir):
        self._two_files_over_one_root()
        out = pc.save_settings_file(name="hello", allow_clash=True)
        assert out["name"] == "hello"
        assert "hello" in [f["name"] for f in pc.list_settings_files()["files"]]

    def test_the_override_is_off_unless_asked_for(self, settings_dir):
        import inspect

        sig = inspect.signature(pc.save_settings_file)
        assert sig.parameters["allow_clash"].default is False

    def test_saving_over_the_same_file_was_never_a_clash(self, settings_dir):
        # `ignore=name` -- a file does not collide with itself, or no
        # config could ever be saved twice.
        self._two_files_over_one_root()
        assert pc.save_settings_file(name="test")["name"] == "test"


class TestPullControlPicksArePlainConfig:
    """Picking a folder has to persist without pressing Apply.

    Reported as "saving the folder in pull control is not working": the
    card wrote to the server only on Apply settings, which sits at the
    bottom of a long card while Save is pinned at the top. `picked` stayed
    empty and every settings file saved without the folders in it.

    Saving is not the same as taking effect: the sweep reads `sources`,
    which is rebuilt from `picked` by Apply and by loading a file. These
    pin that separation, because it is what makes saving-as-you-go safe.
    """

    def test_picked_survives_a_round_trip(self, settings_dir):
        picked = {"train_images": ["deployment"], "train_labels": ["labels/t"],
                  "val_images": [], "val_labels": []}
        pc.write_config({"picked": picked, "source_bucket": "safety-model-dataset-dev"})
        pc.save_settings_file(name="meow")
        pc.write_config({"picked": pc.DEFAULTS["picked"], "source_bucket": ""})
        pc.load_settings_file(name="meow", apply=True)
        live = pc.read_config()
        assert live["picked"] == picked
        assert live["source_bucket"] == "safety-model-dataset-dev"

    def test_the_low_level_writer_leaves_sources_alone(self, settings_dir):
        # `write_config` is a plain writer: it stores what it is given and
        # nothing more. The REBUILD lives in the /config endpoint below,
        # which is what the card calls -- a distinction worth pinning,
        # because an earlier version of this test read as though picking a
        # folder never reached the sweep, which is not true of the route.
        pc.write_config({"sources": [{"prefix": "old/folder", "kind": "train_images"}]})
        pc.write_config({"picked": {"train_images": ["new/folder"], "train_labels": [],
                                    "val_images": [], "val_labels": []}})
        assert pc.read_config()["sources"] == [
            {"prefix": "old/folder", "kind": "train_images"}
        ]

    def test_the_config_route_rebuilds_sources_from_the_picks(self, settings_dir):
        # What the card actually does. Picking a folder and letting the
        # card save makes that folder one the next sweep collects from --
        # there is no staging step, and the UI must not claim one.
        merged = pc.write_config({"picked": {"train_images": ["deployment"],
                                             "train_labels": [], "val_images": [],
                                             "val_labels": []}})
        rebuilt = pc.sources_from_picked(merged)
        assert [s.get("prefix") for s in rebuilt] == ["deployment"]

    def test_loading_a_file_rebuilds_sources_from_picked(self, settings_dir):
        picked = {"train_images": ["deployment"], "train_labels": [],
                  "val_images": [], "val_labels": []}
        pc.write_config({"picked": picked})
        pc.save_settings_file(name="meow")
        pc.write_config({"sources": [], "picked": pc.DEFAULTS["picked"]})
        pc.load_settings_file(name="meow", apply=True)
        prefixes = [s.get("prefix") for s in pc.read_config()["sources"]]
        assert "deployment" in prefixes
