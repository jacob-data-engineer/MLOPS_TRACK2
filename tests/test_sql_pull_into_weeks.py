"""SQL frames filed into Pull Control's weeks by capture date (2026-11-13).

The data preparation flow is: pull from the review database, file each
frame into the week it was CAPTURED in, let Retention decay the weeks, and
build one data.yaml from every ticked source. Before this the SQL pull
wrote one flat folder, so Retention could not see its frames at all, and
pointing Pull Control at that folder would have dated every frame by the
moment it was copied -- one week, no decay.

The capture moment is the stamp inside `case_key`. `data->>'timestamp'`
and `judged_at` are the REVIEW time and must not be read as it.
"""
import datetime as dt
import io
import json

import pytest

from ppe_pipeline_api import pull_control as pc
from ppe_pipeline_api.data import sql_explorer as sx
from ppe_pipeline_api.data import sql_sensor
from ppe_pipeline_api.data.sql_settings import SqlSettings

UTC = dt.timezone.utc
CONFIG = {"root": "retraining/weeks", "destination_bucket": "dest",
          "week_cutoff": "2026-03-24T00:00:00Z", "week_length_days": 7,
          "week_prefix": "week_", "auto_create_weeks": True, "first_week_takes_all": True}


# --- the capture moment ----------------------------------------------------------

def test_the_capture_moment_is_the_stamp_in_the_case_key():
    row = {"case_key": "unk__unk__20260602T014129Z__9a2c__f000076",
           "timestamp": "2026-11-05T06:59:39+00:00", "judged_at": "2026-11-05T06:59:39+00:00"}
    assert sx.capture_moment(row) == dt.datetime(2026, 6, 2, 1, 41, 29, tzinfo=UTC)


def test_a_capture_column_wins_over_the_stamp():
    row = {"captured_at": "2026-04-01T00:00:00Z", "case_key": "x__20260602T014129Z"}
    assert sx.capture_moment(row) == dt.datetime(2026, 4, 1, tzinfo=UTC)


def test_the_review_time_is_never_taken_for_the_capture_time():
    row = {"case_key": "proximity_to_vehicle__19d606a27f", "timestamp": "2026-10-06T14:56:05+00:00",
           "judged_at": "2026-10-06T14:56:05+00:00"}
    assert sx.capture_moment(row, "eval/alarm_images/x.jpg") is None


# --- which week ------------------------------------------------------------------

def _image(name, captured):
    return {"uri": f"s3://src/f/{name}.jpg", "bucket": "src", "key": f"f/{name}.jpg",
            "captured_at": captured}


def test_frames_land_in_the_week_they_were_captured_in():
    images = [_image("a", "2026-03-18T07:49:25+00:00"),     # before the cut-off -> week 1
              _image("b", "2026-03-25T00:00:00+00:00"),     # first week after -> week 2
              _image("c", "2026-06-02T01:41:29+00:00")]
    placed = sx.week_plan(images, 100, by_week=True, config=CONFIG, weeks=[])
    weeks = {e[0]["key"]: e[3] for e in placed["entries"]}
    assert weeks == {"f/a.jpg": "week_1", "f/b.jpg": "week_2",
                     "f/c.jpg": pc.week_at(dt.datetime(2026, 6, 2, 1, 41, 29, tzinfo=UTC), CONFIG)["name"]}
    assert {w["name"] for w in placed["new_weeks"]} == set(weeks.values())


def test_a_frame_with_no_date_is_left_out_not_filed_as_new():
    placed = sx.week_plan([_image("a", "")], 100, by_week=True, config=CONFIG, weeks=[])
    assert placed["entries"] == [] and placed["undated"] == 1


def test_the_backlog_is_left_out_when_week_one_does_not_take_it():
    config = {**CONFIG, "first_week_takes_all": False}
    placed = sx.week_plan([_image("a", "2026-03-01T00:00:00+00:00")], 100,
                          by_week=True, config=config, weeks=[])
    assert placed["entries"] == [] and placed["no_week"] == 1


def test_without_weeks_the_plan_is_the_flat_one():
    placed = sx.week_plan([_image("a", "")], 100)
    assert [e[3] for e in placed["entries"]] == [""] and placed["undated"] == 0


# --- the pull --------------------------------------------------------------------

class _S3:
    def __init__(self, objects):
        self.objects = dict(objects)

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {}

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=""):
        self.objects[(Bucket, Key)] = Body

    def list_objects_v2(self, Bucket, Prefix, MaxKeys=1000):
        return {"Contents": [{"Key": k} for (b, k) in self.objects if b == Bucket and k.startswith(Prefix)]}


class _Lambda:
    def __init__(self, body):
        self.body = body

    def invoke(self, **kw):
        return {"Payload": io.BytesIO(json.dumps(self.body).encode())}


@pytest.fixture
def weeks_setup(monkeypatch):
    rows = [["s3://src/f/a.jpg", "unk__20260318T074925Z__a"],
            ["s3://src/f/b.jpg", "unk__20260602T014129Z__b"],
            ["s3://src/f/c.jpg", "proximity__nostamp"]]
    monkeypatch.setattr(sx, "_client", lambda kind: _Lambda(
        {"ok": True, "columns": ["image", "case_key"], "rows": rows, "truncated": False}))
    s3 = _S3({("src", "f/a.jpg"): b"A", ("src", "f/b.jpg"): b"B", ("src", "f/c.jpg"): b"C"})
    monkeypatch.setattr(sx.storage, "s3", s3)
    monkeypatch.setattr(pc, "read_config", lambda: dict(CONFIG))
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    created = []
    monkeypatch.setattr(pc, "create_week", lambda **kw: created.append(kw["name"]))
    return s3, created


def test_a_week_pull_files_into_pull_controls_root_by_capture_date(weeks_setup):
    s3, created = weeks_setup
    out = sx.pull("violation-review", "q", dest_bucket="ignored", dest_folder="ignored",
                  train_percent=100, by_week=True)
    june = pc.week_at(dt.datetime(2026, 6, 2, tzinfo=UTC), CONFIG)["name"]
    assert ("dest", "retraining/weeks/week_1/images/train/a.jpg") in s3.objects
    assert ("dest", f"retraining/weeks/{june}/images/train/b.jpg") in s3.objects
    assert not any(k.endswith("/c.jpg") for (b, k) in s3.objects if b == "dest")
    assert out["left_out_undated"] == 1 and out["copied"] == 2
    assert out["weeks"] == {"week_1": 1, june: 1}
    assert set(created) == {"week_1", june}
    assert out["bucket"] == "dest" and out["folder"] == "retraining/weeks"


def test_the_layout_is_what_retentions_inventory_reads(weeks_setup, monkeypatch):
    s3, _ = weeks_setup
    sx.pull("violation-review", "q", dest_bucket="", dest_folder="", train_percent=100, by_week=True)
    from ppe_pipeline_api.data import retention as rt
    from ppe_pipeline_api import storage
    monkeypatch.setattr(rt, "week_names", lambda root, prefix="week_": sorted(
        {k.split("/")[2] for (b, k) in s3.objects if b == "dest" and "/week_" in k}, key=rt.week_order))
    monkeypatch.setattr(storage, "_list_keys", lambda prefix: [k for (b, k) in s3.objects
                                                               if b == "dest" and k.startswith(prefix)])
    inventory = rt.read_inventory("retraining/weeks")
    assert inventory[0]["name"] == "week_1" and list(inventory[0]["train"]) == ["a"]


def test_no_root_means_no_week_pull(monkeypatch):
    monkeypatch.setattr(pc, "read_config", lambda: {**CONFIG, "root": ""})
    with pytest.raises(ValueError, match="no root folder"):
        sx.week_destination()


# --- saved with the page, and the chain stays opt-in ------------------------------

def test_both_choices_are_saved_settings_and_off_by_default():
    settings = SqlSettings()
    assert settings.by_week is False and settings.train_after is False
    assert SqlSettings(by_week=True, train_after=True).model_dump()["train_after"] is True


def test_training_after_a_pull_needs_the_weeks(monkeypatch):
    config = sql_sensor.SensorConfig(enabled=True, settings=SqlSettings(
        database="violation-review", preset_class="no_helmet", pull_folder="x", train_after=True))
    with pytest.raises(ValueError, match="Sort into weeks"):
        sql_sensor.validate_selection(config)


def test_a_small_pull_does_not_start_a_gpu_run(monkeypatch):
    from ppe_pipeline_api.chain import queue_after_pull as qap
    monkeypatch.setattr(qap, "settings", lambda config=None: {"min_files": 50})
    monkeypatch.setattr(pc, "claim_once", lambda *a, **k: pytest.fail("must not claim"))
    out = qap.after_sql_pull("run1", {"copied": 3})
    assert out["queued"] is False and "below the 50" in out["reason"]


def test_one_pull_chains_once_across_containers(monkeypatch):
    from ppe_pipeline_api.chain import queue_after_pull as qap
    monkeypatch.setattr(qap, "settings", lambda config=None: {"min_files": 1})
    monkeypatch.setattr(pc, "claim_once", lambda *a, **k: False)
    out = qap.after_sql_pull("run1", {"copied": 30})
    assert out == {"queued": False, "reason": "Another app instance is handling this pull."}


def test_the_sensor_reaches_the_chain_only_through_the_registered_hook(monkeypatch):
    monkeypatch.setattr(sql_sensor, "after_pull", None)
    assert sql_sensor.chain_after("r", {"copied": 9})["queued"] is False
    monkeypatch.setattr(sql_sensor, "after_pull", lambda run_id, summary: 1 / 0)
    assert sql_sensor.chain_after("r", {"copied": 9})["reason"].startswith("chain:")


def test_the_ticker_registers_the_chain():
    import pathlib
    ticker = (pathlib.Path(__file__).resolve().parents[1]
              / "src/ppe_pipeline_api/chain/ticker.py").read_text(encoding="utf-8")
    assert "sql_sensor.after_pull = queue_after_pull.after_sql_pull" in ticker
