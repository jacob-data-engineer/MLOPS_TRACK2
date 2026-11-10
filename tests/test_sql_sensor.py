import datetime as dt
import json

import pytest

from ppe_pipeline_api.data import sql_sensor as ss

UTC = dt.timezone.utc


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.pull_control, "SETTINGS_DIR", str(tmp_path))
    monkeypatch.setattr(ss, "_running", False)
    monkeypatch.setattr(ss, "_recovered", False)
    return tmp_path


def config(**kwargs):
    return ss.SensorConfig(settings={"database": "violation-review", "preset_class": "no_helmet",
        "pull_bucket": "target", "pull_folder": "frames", "class_ids": {"helmet": 1, "person": 0}}, **kwargs)


def test_midnight_is_singapore_midnight_and_does_not_fire_on_save(env):
    current = dt.datetime(2026, 11, 10, 6, tzinfo=UTC)
    result = ss.save(config(enabled=True), current)
    assert result["next_at"] == "2026-11-10T16:00:00+00:00"
    assert not result["running"]
    assert "last_run" not in result
    assert ss.read()["config"]["settings"]["class_ids"] == {"helmet": 1, "person": 0}


def test_custom_interval_keeps_anchor_across_other_settings_saves(env):
    current = dt.datetime(2026, 11, 10, 6, tzinfo=UTC)
    original = config(enabled=True, every_days=7, at="03:15")
    result = ss.save(original, current)
    assert result["next_at"] == "2026-11-10T19:15:00+00:00"
    original.settings.class_ids = {"helmet": 0, "person": 1}
    result = ss.save(original, dt.datetime(2026, 11, 12, 6, tzinfo=UTC))
    assert result["next_at"] == "2026-11-17T19:15:00+00:00"


def test_missed_schedule_claims_once_and_skips_backlog(env, monkeypatch):
    current = dt.datetime(2026, 11, 10, 6, tzinfo=UTC)
    ss.save(config(enabled=True), current)
    started = []
    class Thread:
        def __init__(self, **kwargs):
            started.append(kwargs)
        def start(self):
            pass
    monkeypatch.setattr(ss.threading, "Thread", Thread)
    assert not ss.start(moment=current)
    late = dt.datetime(2026, 11, 14, 10, tzinfo=UTC)
    assert ss.start(moment=late)
    assert not ss.start(moment=late)
    assert len(started) == 1
    assert ss.read()["next_at"] == "2026-11-14T16:00:00+00:00"
    with pytest.raises(ValueError, match="running"):
        ss.save(config(), late)


def test_off_schedule_does_not_run_and_restart_recovers_interrupted(env, monkeypatch):
    ss.save(config(enabled=False))
    assert not ss.start()
    state = ss.read()
    state["last_run"] = {"status": "running", "id": "old"}
    ss.write(state)
    monkeypatch.setattr(ss, "_recovered", False)
    ss.recover()
    assert ss.status()["last_run"]["status"] == "interrupted"
    assert not ss.status()["next_at"]


@pytest.mark.parametrize("params", [{"at": "24:00"}, {"at": "wrong"}, {"every_days": 0}, {"every_days": 366}, {"from_date": "bad"}])
def test_invalid_schedule_is_rejected(params):
    with pytest.raises(ValueError):
        config(**params)


def setup_execution(monkeypatch, cfg, *, present=False, label_present=False):
    image = {"uri": "s3://src/a.jpg", "bucket": "src", "key": "a.jpg", "labels": {"complete": False, "boxes": [{"name": "person", "cls": 1, "x1": .1, "y1": .1, "x2": .5, "y2": .9}]}}
    result = {"images": [image]}
    monkeypatch.setattr(ss.sx, "has_crop_verdicts", lambda key: True)
    monkeypatch.setattr(ss.sx, "preset_sql", lambda *a, **kw: "fresh query")
    monkeypatch.setattr(ss.sx, "run", lambda *a, **kw: result)
    monkeypatch.setattr(ss, "exists", lambda bucket, key: label_present if "/labels/" in key else present if "/images/" in key else False)
    ss.save(cfg)
    state = ss.read()
    state["last_run"] = {"id": "test", "status": "running"}
    ss.write(state)
    return result


def test_sensor_pulls_missing_labels_with_exact_saved_settings(env, monkeypatch):
    cfg = config()
    result = setup_execution(monkeypatch, cfg, present=True, label_present=False)
    called = []
    def pull(key, sql, **kwargs):
        called.append((key, sql, kwargs))
        return {"failed": 0, "copied": 0, "skipped": 1, "with_label": 1}
    monkeypatch.setattr(ss.sx, "pull", pull)
    ss.execute(cfg, "test")
    assert len(called) == 1
    key, sql, options = called[0]
    assert key == cfg.settings.database and sql == "fresh query"
    assert options["class_ids"] == {"helmet": 1, "person": 0}
    assert options["dest_bucket"] == "target" and options["dest_folder"] == "frames"
    assert options["train_percent"] == 80 and options["with_labels"] is True
    assert options["query_result"] is result
    last = ss.status()["last_run"]
    assert last["status"] == "complete" and last["pending"] == 1


def test_unchanged_destination_does_not_pull(env, monkeypatch):
    cfg = config()
    setup_execution(monkeypatch, cfg, present=True, label_present=True)
    monkeypatch.setattr(ss.sx, "pull", lambda *a, **kw: pytest.fail("should not copy"))
    ss.execute(cfg, "test")
    assert ss.status()["last_run"]["status"] == "unchanged"
    assert ss.read()["seen"] == ["s3://src/a.jpg"]


def test_query_failure_is_visible_and_releases_run(env, monkeypatch):
    cfg = config()
    setup_execution(monkeypatch, cfg)
    def fail(*a, **kw):
        raise ValueError("database stopped")
    monkeypatch.setattr(ss.sx, "run", fail)
    ss.execute(cfg, "test")
    assert ss.status()["last_run"]["error"] == "database stopped"
    assert not ss.status()["running"]
    assert ss.read()["seen"] == []


def test_truncation_never_triggers_partial_pull(env, monkeypatch):
    cfg = config()
    result = setup_execution(monkeypatch, cfg)
    result["truncated"] = True
    monkeypatch.setattr(ss.sx, "pull", lambda *a, **kw: pytest.fail("should not copy"))
    ss.execute(cfg, "test")
    assert "truncated" in ss.status()["last_run"]["error"]


def test_empty_selection_reports_unchanged(env, monkeypatch):
    cfg = config()
    result = setup_execution(monkeypatch, cfg)
    result["images"] = []
    monkeypatch.setattr(ss.sx, "pull", lambda *a, **kw: pytest.fail("should not copy"))
    ss.execute(cfg, "test")
    assert ss.status()["last_run"]["status"] == "unchanged"


def test_status_does_not_leak_or_return_full_seen_inventory(env):
    ss.save(config())
    state = ss.read()
    state["seen"] = ["source"]
    ss.write(state)
    assert "seen" not in ss.status()
    assert json.loads(ss.path().read_text())["seen"] == ["source"]


def test_manual_check_preserves_next_scheduled_time(env, monkeypatch):
    ss.save(config(enabled=True), dt.datetime(2026, 11, 10, 6, tzinfo=UTC))
    due = ss.status()["next_at"]
    class Thread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(ss.threading, "Thread", Thread)
    assert ss.start(manual=True)
    assert ss.status()["next_at"] == due


def test_background_off_cannot_arm_a_schedule(env, monkeypatch):
    monkeypatch.setenv("APP_BACKGROUND", "off")
    with pytest.raises(ValueError, match="disabled"):
        ss.save(config(enabled=True))


def test_real_pull_path_copies_once_then_discovers_new_image(env, monkeypatch):
    import io
    from botocore.exceptions import ClientError
    objects = {("src", "a.jpg"): b"image-a", ("src", "b.jpg"): b"image-b"}
    class Store:
        def head_object(self, Bucket, Key):
            if (Bucket, Key) not in objects:
                raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
            return {}
        def get_object(self, Bucket, Key):
            return {"Body": io.BytesIO(objects[Bucket, Key])}
        def put_object(self, Bucket, Key, Body, **kwargs):
            objects[Bucket, Key] = Body
        def copy_object(self, Bucket, Key, CopySource):
            objects[Bucket, Key] = objects[CopySource["Bucket"], CopySource["Key"]]
        def list_objects_v2(self, Bucket, Prefix, **kwargs):
            return {"Contents": [{"Key": k} for b, k in objects if b == Bucket and k.startswith(Prefix)]}
    monkeypatch.setattr(ss.sx.storage, "s3", Store())
    cfg = config()
    cfg.settings.class_ids = {"person": 0}
    def image(name):
        return {"bucket": "src", "key": name, "uri": "s3://src/" + name,
                "labels": {"complete": True, "boxes": [{"name": "person", "cls": 2, "x1": .1, "y1": .2, "x2": .5, "y2": .9}]}}
    result = {"images": [image("a.jpg")]}
    monkeypatch.setattr(ss.sx, "has_crop_verdicts", lambda key: True)
    monkeypatch.setattr(ss.sx, "preset_sql", lambda *a, **kw: "fresh query")
    monkeypatch.setattr(ss.sx, "run", lambda *a, **kw: result)
    ss.save(cfg)
    state = ss.read()
    state["last_run"] = {"id": "test", "status": "running"}
    ss.write(state)
    ss.execute(cfg, "test")
    assert ss.status()["last_run"]["summary"]["copied"] == 1
    label = next(v for (b, k), v in objects.items() if b == "target" and k.endswith(".txt"))
    assert label.startswith(b"0 ")
    ss.execute(cfg, "test")
    assert ss.status()["last_run"]["status"] == "unchanged"
    result["images"].append(image("b.jpg"))
    ss.execute(cfg, "test")
    assert ss.status()["last_run"]["summary"]["copied"] == 1
    assert ss.status()["last_run"]["newly_eligible"] == 1
