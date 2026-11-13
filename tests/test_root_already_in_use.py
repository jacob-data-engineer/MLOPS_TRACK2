"""Picking a root another saved configuration already uses says so.

Four settings files on this host, and nothing anywhere said that three
of them trained out of the same `retraining`. That is not an error --
the same week folders CAN feed two Classes -- but it is almost never
meant, and the consequences are silent:

  * the pull WRITES into that root, so one configuration's sweeps
    reshape what the other trains on;
  * retention PRUNES it, so one configuration's policy deletes the
    other's weeks;
  * the chain trains from it, so the second inherits the first's data.

Read from the FILES, not from live config. The live one is whatever was
loaded last; the question is what every OTHER saved configuration
already claims.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402

def _write(tmp_path, name, config):
    """A settings file as `save_settings_file` writes one.

    `read_settings_file` returns `body["config"]`, so a fixture without
    that wrapper reads back as {} -- which is how this test first
    claimed the scanner was broken when the fixture was.
    """
    (tmp_path / f"{name}.json").write_text(
        json.dumps({"config": config, "registry": {}}), encoding="utf-8")


# --- the scan ------------------------------------------------------------------------

def test_it_reports_each_files_root_and_model(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    _write(tmp_path, "test4", {"root": "retraining",
                               "default_detector": {"slug": "helmet"}})
    _write(tmp_path, "other", {"root": "elsewhere",
                               "default_detector": {"slug": "fakecad-it"}})
    claims = {c["name"]: c for c in pc.settings_claims()}
    assert claims["test4"]["root"] == "retraining"
    assert claims["test4"]["model"] == "helmet"
    assert claims["other"]["root"] == "elsewhere"


def test_the_config_section_is_what_is_read(tmp_path, monkeypatch):
    """A settings file carries `config` beside `registry`, and
    `read_settings_file` hands back the first of those."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    _write(tmp_path, "wrapped", {"root": "retraining",
                                 "default_detector": {"slug": "ppe2"}})
    claim = pc.settings_claims()[0]
    assert claim["root"] == "retraining"
    assert claim["model"] == "ppe2"


def test_the_assembled_folder_is_a_claim_as_well(tmp_path, monkeypatch):
    """Two files writing one data.yaml fight over the same file."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    _write(tmp_path, "a", {"dataset_assemble": {"built": {"folder": "Testing_Dataset/ds"}}})
    assert pc.settings_claims()[0]["assembled"] == "Testing_Dataset/ds"


def test_a_file_that_will_not_parse_does_not_take_the_scan_down(tmp_path, monkeypatch):
    """Failing the whole scan for one bad file would take the warning
    away from every other."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text("{not json", encoding="utf-8")
    _write(tmp_path, "good", {"root": "retraining"})
    names = [c["name"] for c in pc.settings_claims()]
    assert names == ["good"]


def test_no_files_is_no_claims(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    assert pc.settings_claims() == []


def test_a_missing_directory_is_no_claims(monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", "/definitely/not/here")
    assert pc.settings_claims() == []


def test_the_listing_carries_them(tmp_path, monkeypatch):
    """One fetch, not a second round trip -- the card already asks for
    the file list."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    _write(tmp_path, "a", {"root": "retraining"})
    assert "claims" in pc.list_settings_files()

