"""The Scheduled page: what the clocks will start, when, and from what."""
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppe_pipeline_api.routers import track2_retention as tr  # noqa: E402

NOW = dt.datetime(2026, 10, 20, 4, 0, tzinfo=dt.timezone.utc)  # 12:00 SGT


def _config(**over):
    base = {
        "settings_file": "helmet.json",
        "mode": "schedule", "schedule_every_days": 7, "schedule_time": "02:00",
        "week_cutoff": "2026-09-01T02:00:00+08:00",
        "dataset_assemble": {"use": True, "only": False},
        "retention": {"use_golden": True},
        "queue_after_pull": {"enabled": True, "triggers": ["schedule"]},
        "retrain_schedule": {"enabled": False},
    }
    base.update(over)
    return base


def test_pulls_and_the_training_they_set_off(monkeypatch):
    out = tr.scheduled_events(_config(), days=21, now=NOW)
    pulls = [e for e in out["events"] if e["kind"] == "pull"]
    trains = [e for e in out["events"] if e["kind"] == "train"]
    assert len(pulls) == 3 and len(trains) == 3
    assert [p["at"] for p in pulls] == [t["at"] for t in trains]
    assert out["settings_file"] == "helmet.json"
    assert out["sources"] == ["Pull", "Golden set", "Assembled folders"]
    assert out["next"]["kind"] == "train" and out["next"]["at"] == pulls[0]["at"]


def test_a_pull_that_does_not_chain_trains_nothing():
    out = tr.scheduled_events(_config(queue_after_pull={"enabled": False}), days=21, now=NOW)
    assert all(e["kind"] == "pull" for e in out["events"]) and out["next"] is None


def test_the_training_clock_on_its_own(monkeypatch):
    monkeypatch.setattr(tr.rs, "follows_pull", lambda config: False)
    cfg = _config(mode="off", queue_after_pull={"enabled": False},
                  retrain_schedule={"enabled": True, "every_days": 7, "at": "03:00"})
    out = tr.scheduled_events(cfg, days=30, now=NOW)
    assert out["events"] and all(e["kind"] == "train" for e in out["events"])
    assert all(e["at"].endswith("19:00:00+00:00") for e in out["events"])  # 03:00 SGT


def test_arrival_is_a_note_not_a_date():
    out = tr.scheduled_events(_config(mode="arrival",
                                      queue_after_pull={"enabled": True, "triggers": ["arrival"],
                                                        "min_files": 20}), days=21, now=NOW)
    assert out["events"] == []
    assert any("at least 20 file" in n for n in out["notes"])


def test_pull_off_means_no_pulls():
    out = tr.scheduled_events(_config(dataset_assemble={"use": True, "only": True}), days=21, now=NOW)
    assert not [e for e in out["events"] if e["kind"] == "pull"]
    assert "Pull" not in out["sources"]


def test_the_page_is_routed_and_in_the_nav_under_training():
    app = (ROOT / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
    nav = (ROOT / "web" / "src" / "components" / "Layout.tsx").read_text(encoding="utf-8")
    assert '<Route path="/track2/scheduled" element={<Track2ScheduledPage />} />' in app
    assert nav.index("/track2/multi-auto-training") < nav.index("/track2/scheduled") \
        < nav.index("/track2/metrics")
    page = (ROOT / "web" / "src" / "routes" / "Track2ScheduledPage.tsx").read_text(encoding="utf-8")
    assert '"Weekly"' in page and '"Monthly"' in page


def test_saved_files_are_shown_beside_the_live_one_and_marked(monkeypatch):
    live = _config(settings_file="live_one")
    other = _config(settings_file="other", schedule_time="09:00",
                    retention={"use_golden": False})
    monkeypatch.setattr(tr.pc, "read_config", lambda: live)
    monkeypatch.setattr(tr.pc, "read_settings_file",
                        lambda name: other if name == "other" else (_ for _ in ()).throw(ValueError("No settings file called x.")))
    monkeypatch.setattr(tr.pc, "list_settings_files",
                        lambda: {"files": [{"name": "live_one"}, {"name": "other"}, {"name": "x"}]})
    out = tr.scheduled_for_files(["live_one", "other", "x"], days=14, now=NOW)
    names = [(f["name"], f["live"]) for f in out["files"]]
    assert names == [("live_one", True), ("other", False), ("x", False)]
    assert out["files"][2]["error"]
    assert {e["file"] for e in out["events"]} == {"live_one", "other"}
    assert all(e["live"] == (e["file"] == "live_one") for e in out["events"])
    assert "Golden set" not in out["files"][1]["sources"]
    assert out["available"] == ["live_one", "other", "x"]


def test_the_page_offers_all_files_and_a_hover_popup():
    page = (ROOT / "web" / "src" / "routes" / "Track2ScheduledPage.tsx").read_text(encoding="utf-8")
    assert "All saved files" in page and "group-hover:block" in page
    assert "Not loaded -- would run if this file were loaded" in page
