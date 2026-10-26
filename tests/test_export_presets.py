"""The export card's own settings file, kept apart from the pull config.

Both are "settings" and both live on the EC2 host, but they are not
interchangeable. The pull-control file holds a whole retraining
configuration -- sources, cut-off, schedule, detector -- and loading it
rearms a schedule. An export preset holds which Class, which criterion
and where the checkpoint goes. One dropdown for both would make Load a
button whose consequences depend on which name you happened to pick.
"""
from __future__ import annotations

import importlib
import json
import os

import pytest
from fastapi import HTTPException

mod = importlib.import_module("src.ppe_pipeline_api.routers.track2_metrics")


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    folder = tmp_path / "export"
    monkeypatch.setattr(mod, "EXPORT_SETTINGS_DIR", str(folder))
    return folder


def save(name, **fields):
    return mod.save_export_settings(
        mod.ExportSettingsRequest(name=name, settings=mod.ExportSettings(**fields)),
        session=None)


def test_a_preset_round_trips(settings_dir):
    """Checks the fields it SET, not an exhaustive dict. An exhaustive
    comparison fails every time a field is added, which teaches whoever
    adds one to edit the assertion rather than to think about it."""
    save("weekly", registry="helmet", generation="26", size="m",
         group_name="yolo26m-helmet-detector", criterion="recall",
         bucket="track2-s3", prefix="exports")
    got = mod.load_export_settings(mod.ExportSettingsRequest(name="weekly"), session=None)["settings"]
    assert got["group_name"] == "yolo26m-helmet-detector"
    assert got["criterion"] == "recall"
    assert got["bucket"] == "track2-s3"
    assert got["prefix"] == "exports"


def test_it_lives_under_its_own_folder_not_the_pull_config(monkeypatch, tmp_path):
    """Same host volume -- so it survives a deploy -- different folder."""
    monkeypatch.setenv("PULL_CONTROL_SETTINGS_DIR", str(tmp_path))
    monkeypatch.delenv("METRICS_EXPORT_SETTINGS_DIR", raising=False)
    reloaded = importlib.reload(mod)
    try:
        assert reloaded.EXPORT_SETTINGS_DIR == os.path.join(str(tmp_path), "export")
    finally:
        importlib.reload(mod)


def test_listing_is_newest_first(settings_dir):
    save("older")
    save("newer")
    os.utime(settings_dir / "older.json", (1, 1))
    names = [f["name"] for f in mod.list_export_settings(session=None)["files"]]
    assert names[0] == "newer"


def test_a_missing_folder_is_not_an_error(settings_dir):
    """Nothing saved yet. The card should offer Save, not report a fault."""
    out = mod.list_export_settings(session=None)
    assert out["files"] == []


def test_deleting_removes_it(settings_dir):
    save("scratch")
    mod.delete_export_settings(mod.ExportSettingsRequest(name="scratch"), session=None)
    assert mod.list_export_settings(session=None)["files"] == []


def test_loading_a_name_that_is_not_there_404s(settings_dir):
    with pytest.raises(HTTPException) as exc:
        mod.load_export_settings(mod.ExportSettingsRequest(name="nope"), session=None)
    assert exc.value.status_code == 404


@pytest.mark.parametrize("name", ["../escape", "a/b", "..", "", " ", "x" * 100,
                                  ".hidden", "a\b"])
def test_a_name_that_could_write_outside_the_folder_is_refused(settings_dir, name):
    """A file name has to survive being a path segment. Anything else is
    a way to write somewhere nobody asked for."""
    with pytest.raises(HTTPException) as exc:
        save(name)
    assert exc.value.status_code == 422


def test_a_preset_written_before_a_field_existed_loads_as_the_default(settings_dir):
    """Presets outlive the code that wrote them."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "old.json").write_text(
        json.dumps({"saved_at": "", "settings": {"bucket": "b"}}), encoding="utf-8")
    got = mod.load_export_settings(mod.ExportSettingsRequest(name="old"), session=None)
    assert got["settings"]["criterion"] == "mAP50"
    assert got["settings"]["group_name"] == ""


def test_a_save_is_written_whole(settings_dir):
    """Written to a temporary name then moved, so a page load can never
    read a half-written preset."""
    import inspect
    source = inspect.getsource(mod.save_export_settings)
    assert "os.replace(" in source
    assert ".tmp" in source


def test_saving_nothing_is_refused(settings_dir):
    with pytest.raises(HTTPException) as exc:
        mod.save_export_settings(mod.ExportSettingsRequest(name="x"), session=None)
    assert exc.value.status_code == 422


def test_loading_a_preset_exports_nothing(settings_dir):
    """Load fills the form. The export is still a separate press -- a
    Load that wrote a file to S3 would be a button nobody could undo."""
    import inspect
    source = inspect.getsource(mod.load_export_settings)
    assert "put_object" not in source
    assert "build_metrics_csv" not in source


def test_a_preset_carries_the_destination(settings_dir):
    """The same checkpoint downloaded to a laptop and written to a shared
    bucket are two different jobs. A preset that restored everything but
    that would reproduce the wrong one."""
    save("to_s3", destination="s3", bucket="track2-s3", prefix="exports")
    got = mod.load_export_settings(mod.ExportSettingsRequest(name="to_s3"), session=None)
    assert got["settings"]["destination"] == "s3"


def test_a_preset_written_before_the_destination_existed_loads_as_local(settings_dir):
    """The safe reading of silence: a download writes nothing anybody
    else can see."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / "older.json").write_text(
        json.dumps({"settings": {"group_name": "yolo11s-helmet-detector",
                                 "bucket": "b", "prefix": "p"}}), encoding="utf-8")
    got = mod.load_export_settings(mod.ExportSettingsRequest(name="older"), session=None)
    assert got["settings"]["destination"] == "local"


def test_a_preset_does_not_carry_the_watcher(settings_dir):
    """Arming is its own call, deliberately.

    A preset that could arm the watcher would make Load a button that
    starts writing to a shared bucket on a schedule, depending on which
    name happened to be in the dropdown. The pull schedule is armed in
    its card for the same reason.
    """
    fields = set(mod.ExportSettings().model_dump())
    assert "enabled" not in fields
    assert "last_version" not in fields
    assert "auto" not in fields


def test_the_watchers_own_file_cannot_be_reached_as_a_preset(settings_dir):
    """`_auto_export.json` sits in the same folder, and must not appear in
    the Load dropdown or be writable through Save."""
    from src.ppe_pipeline_api import metrics_export as mx

    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / mx.AUTO_FILE).write_text(
        json.dumps({"enabled": True, "group_name": "x"}), encoding="utf-8")
    assert mod.list_export_settings(session=None)["files"] == []
    with pytest.raises(HTTPException) as exc:
        save(mx.AUTO_FILE.removesuffix(".json"))
    assert exc.value.status_code == 422
