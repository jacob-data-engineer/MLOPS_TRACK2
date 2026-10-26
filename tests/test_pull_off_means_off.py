"""Pull Control unticked means the pull does nothing and checks nothing.

A configuration with Pull Control off writes no week folders. Its save was
still refused for "already writes retraining with prefix 'week_'", and the
router Lambda still swept the drop folders on schedule -- runs that
"moved 0" every morning for a configuration that never pulls.
"""
import inspect
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from ppe_pipeline_api import pull_control as pc  # noqa: E402

OFF = {"dataset_assemble": {"only": True}}
ON = {"dataset_assemble": {"only": False}}


def test_the_tick_reads_the_stored_field_inverted():
    assert pc.pull_in_use({}) is True
    assert pc.pull_in_use(ON) is True
    assert pc.pull_in_use(OFF) is False


def _write(tmp, name, config):
    (tmp / f"{name}.json").write_text(json.dumps({"config": config}), encoding="utf-8")


def test_a_config_with_the_pull_off_saves_without_the_clash_check(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    other = {**ON, "root": "retraining", "picked": {"train_images": ["a"]}}
    _write(tmp_path, "test4", other)
    mine = {**OFF, "root": "retraining", "picked": {"train_images": ["b"]}}
    pc.save_settings_file(name="full_golden", config=mine)       # must not raise
    assert (tmp_path / "full_golden.json").exists()


def test_a_file_with_the_pull_off_never_counts_as_an_owner(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    _write(tmp_path, "full_golden", {**OFF, "root": "retraining", "picked": {"train_images": ["b"]}})
    mine = {**ON, "root": "retraining", "picked": {"train_images": ["a"]}}
    assert pc.settings_owner(pc.folder_identity(mine), ignore="x",
                             sources=pc._picked_sources(mine)) == ""


def test_two_configs_that_both_pull_are_still_warned(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    _write(tmp_path, "test4", {**ON, "root": "retraining", "picked": {"train_images": ["a"]}})
    mine = {**ON, "root": "retraining", "picked": {"train_images": ["b"]}}
    try:
        pc.save_settings_file(name="other", config=mine)
        raise AssertionError("expected the clash refusal")
    except ValueError as exc:
        assert "already writes" in str(exc)


def test_the_router_stands_down_when_the_pull_is_off(monkeypatch):
    monkeypatch.setattr(pc, "read_config", lambda: OFF)
    called = []
    monkeypatch.setattr(pc, "active_run", lambda: called.append("swept"))
    out = pc.lambda_handler({"detail-type": "Scheduled Event"})
    assert out["off"] is True and called == []


def test_the_pull_queue_tick_stands_down_too(monkeypatch):
    monkeypatch.setattr(pc, "read_config", lambda: OFF)
    monkeypatch.setattr(pc, "advance_pull_queue", lambda **k: (_ for _ in ()).throw(AssertionError("ran")))
    assert pc.lambda_handler({"tick": True}) == {"pull_control": "off"}
