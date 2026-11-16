"""A Data Preparation file says when it was created, last saved and loaded
for training -- on one line of the training card (2026-11-16)."""
import json
import pathlib

from ppe_pipeline_api import pull_control as pc

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _setup(monkeypatch, tmp_path, config):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    monkeypatch.setattr(pc, "read_config", lambda: dict(config))
    monkeypatch.setattr(pc, "write_config", lambda patch: {**config, **patch})
    monkeypatch.setattr(pc, "pull_in_use", lambda body: False)


def test_created_is_set_once_and_kept_on_every_resave(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, {"root": "r"})
    times = iter(["2026-11-01T00:00:00+00:00", "2026-11-05T00:00:00+00:00"])
    monkeypatch.setattr(pc, "_now", lambda: type("T", (), {"isoformat": lambda self, **k: next(times)})())
    pc.save_settings_file(name="a", scope="dp")
    first = json.loads((tmp_path / "a_DP.json").read_text())
    monkeypatch.setattr(pc, "_now", lambda: type("T", (), {"isoformat": lambda self, **k: "2026-11-09T00:00:00+00:00"})())
    pc.save_settings_file(name="a", scope="dp")
    second = json.loads((tmp_path / "a_DP.json").read_text())
    assert second["created_at"] == first["created_at"]
    assert second["saved_at"] != first["saved_at"]


def test_a_file_from_before_has_no_invented_creation_date(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, {"root": "r"})
    (tmp_path / "old_DP.json").write_text(json.dumps({"saved_at": "x", "config": {}}))
    pc.save_settings_file(name="old", scope="dp")
    assert "created_at" not in json.loads((tmp_path / "old_DP.json").read_text())


def test_the_card_shows_all_three_on_the_loaded_line():
    card = (ROOT / "web/src/components/Track2TrainingDataFile.tsx").read_text(encoding="utf-8")
    line = card[card.index("Loaded: {loaded"):card.index("</p>", card.index("Loaded: {loaded"))]
    assert "loadedTimes.loaded_at" in line and "loadedTimes.created_at" in line and "loadedTimes.saved_at" in line
    src = (ROOT / "src/ppe_pipeline_api/pull_control.py").read_text(encoding="utf-8")
    assert '"loaded_at": loaded_at, "config": recipe' in src
