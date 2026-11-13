"""Combine every source into the central store, then decay a copy for the
run (2026-11-13).

The operator's rules, each pinned here:
  * the store is only ever ADDED to -- never pruned, never overwritten;
  * `<store>_decay` is rebuilt from the store before every run, and the
    run trains on it alone;
  * it is deleted once the run has finished -- never while a run, or a
    queued entry, could still read it;
  * a button does the combine and decay by hand, without training.
"""
import datetime as dt
import io
import json
import pathlib

import pytest

from ppe_pipeline_api import pull_control as pc
from ppe_pipeline_api import storage
from ppe_pipeline_api.data import central_store, combined_store as cs

ROOT = pathlib.Path(__file__).resolve().parents[1]
B = storage.BUCKET


class _Paginator:
    def __init__(self, s3):
        self.s3 = s3

    def paginate(self, Bucket, Prefix):
        yield {"Contents": [{"Key": k} for (b, k) in sorted(self.s3.objects)
                            if b == Bucket and k.startswith(Prefix)]}


class _S3:
    def __init__(self, objects):
        self.objects = dict(objects)
        self.copies = []

    def list_objects_v2(self, Bucket, Prefix="", Delimiter=None, ContinuationToken=None, MaxKeys=None):
        keys = sorted(k for (b, k) in self.objects if b == Bucket and k.startswith(Prefix))
        if Delimiter:
            folders = sorted({Prefix + k[len(Prefix):].split("/", 1)[0] + "/"
                              for k in keys if "/" in k[len(Prefix):]})
            return {"CommonPrefixes": [{"Prefix": f} for f in folders]}
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_paginator(self, name):
        return _Paginator(self)

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {"ETag": '"x"'}

    def copy_object(self, Bucket, Key, CopySource):
        self.objects[(Bucket, Key)] = self.objects[(CopySource["Bucket"], CopySource["Key"])]
        self.copies.append(Key)

    def get_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise Exception("404")
        return {"Body": io.BytesIO(self.objects[(Bucket, Key)])}

    def put_object(self, Bucket, Key, Body, ContentType=""):
        self.objects[(Bucket, Key)] = Body

    def delete_objects(self, Bucket, Delete):
        for o in Delete["Objects"]:
            self.objects.pop((Bucket, o["Key"]), None)


def _week(n, names, split="train"):
    out = {}
    for name in names:
        out[(B, f"retraining/week_{n}/images/{split}/{name}.jpg")] = b"I"
        out[(B, f"retraining/week_{n}/labels/{split}/{name}.txt")] = b"0 .5 .5 .1 .1"
    return out


@pytest.fixture
def s3(monkeypatch):
    objects = {**_week(1, [f"a{i}" for i in range(10)]), **_week(2, [f"b{i}" for i in range(10)]),
               **_week(2, ["v0"], split="val"),
               (B, "retraining/_retained/train.txt"): b"x",
               (B, "golden/images/train/g.jpg"): b"G", (B, "golden/labels/train/g.txt"): b"0",
               (B, "golden/images/val/h.jpg"): b"H", (B, "golden/labels/val/h.txt"): b"0",
               (B, "sets/one/images/train/f.jpg"): b"F", (B, "sets/one/labels/train/f.txt"): b"0",
               (B, "sets/one/data.yaml"): b"names: {0: helmet}"}
    fake = _S3(objects)
    monkeypatch.setattr(storage, "s3", fake)
    monkeypatch.setattr(central_store, "location", lambda: (B, "central"))
    return fake


def _combine():
    return cs.combine(weeks_root="retraining", weeks_bucket=B, golden_folder="golden",
                      golden_bucket=B, folders=[{"images": "sets/one/images/train",
                                                 "labels": "sets/one/labels/train", "split": "train"}])


# --- the store: added to, never pruned or overwritten -----------------------------

def test_every_source_lands_apart_in_the_store(s3):
    out = _combine()
    assert (B, "central/weeks/week_1/images/train/a0.jpg") in s3.objects
    assert (B, "central/weeks/week_2/labels/val/v0.txt") in s3.objects
    assert (B, "central/golden/images/val/h.jpg") in s3.objects
    assert (B, "central/folders/sets__one/images/train/f.jpg") in s3.objects
    assert (B, "central/folders/sets__one/data.yaml") in s3.objects, "the source's numbering travels"
    assert not any("_retained" in k for (b, k) in s3.objects if k.startswith("central/"))
    assert out["slugs"] == {"sets__one": ["train"]}


def test_a_second_combine_adds_only_what_is_new_and_overwrites_nothing(s3):
    _combine()
    s3.objects[(B, "central/golden/images/train/g.jpg")] = b"FIRST"
    s3.objects[(B, "retraining/week_3/images/train/c.jpg")] = b"C"
    s3.copies.clear()
    out = _combine()
    assert s3.copies == ["central/weeks/week_3/images/train/c.jpg"]
    assert s3.objects[(B, "central/golden/images/train/g.jpg")] == b"FIRST"
    assert out["copied"] == 1


def test_files_gone_from_a_source_stay_in_the_store(s3):
    _combine()
    del s3.objects[(B, "retraining/week_1/images/train/a0.jpg")]
    _combine()
    assert (B, "central/weeks/week_1/images/train/a0.jpg") in s3.objects


def test_the_store_must_be_in_the_bucket_runs_read_from(monkeypatch):
    monkeypatch.setattr(central_store, "location", lambda: ("some-other-bucket", "central"))
    with pytest.raises(ValueError, match="bucket runs read their dataset from"):
        cs.location()


# --- the decayed copy ------------------------------------------------------------

POLICY = {"enabled": True, "weekly_rate": 0.5, "curve": "exponential", "full_weeks": 1,
          "use_floor": False, "week_prefix": "week_"}


def test_the_decayed_copy_thins_old_weeks_and_keeps_golden_and_folders_whole(s3):
    out = _combine()
    built = cs.rebuild_decayed(retention=POLICY, use_weeks=True, use_golden=True,
                               slugs=out["slugs"], for_run=True)
    decayed = lambda w: [k for (b, k) in s3.objects if k.startswith(f"central_decay/weeks/{w}/images/train/")]
    assert len(decayed("week_2")) == 10, "the newest week counts in full"
    assert 0 < len(decayed("week_1")) < 10, "an older week is thinned"
    for k in decayed("week_1"):
        assert (B, k.replace("/images/", "/labels/").replace(".jpg", ".txt")) in s3.objects
    assert not any("/val/" in k for (b, k) in s3.objects if k.startswith("central_decay/weeks/")), \
        "week val is never decayed or trained on"
    assert (B, "central_decay/golden/images/val/h.jpg") in s3.objects
    assert (B, "central_decay/folders/sets__one/images/train/f.jpg") in s3.objects
    assert {"images": "central_decay/golden/images/val", "labels": ""} in built["val"]
    assert {"images": "central_decay/weeks/week_2/images/train", "labels": ""} in built["train"]
    assert json.loads(s3.objects[(B, "central_decay/_decay.json")])["status"] == "built"


def test_a_rebuild_starts_from_empty(s3):
    out = _combine()
    s3.objects[(B, "central_decay/weeks/week_9/images/train/stale.jpg")] = b"S"
    cs.rebuild_decayed(retention=POLICY, use_weeks=True, use_golden=False, slugs=out["slugs"])
    assert (B, "central_decay/weeks/week_9/images/train/stale.jpg") not in s3.objects


def test_deleting_the_decayed_copy_never_touches_the_store(s3):
    out = _combine()
    cs.rebuild_decayed(retention=POLICY, use_weeks=True, use_golden=True, slugs=out["slugs"])
    store_before = {k for k in s3.objects if k[1].startswith("central/")}
    cs.delete_decayed()
    assert not any(k.startswith("central_decay/") for (b, k) in s3.objects)
    assert {k for k in s3.objects if k[1].startswith("central/")} == store_before


# --- the chain: when it is cleaned up --------------------------------------------

def _cleanup_setup(monkeypatch, *, marker, running="", pending=()):
    from ppe_pipeline_api.chain import retrain_schedule as rs
    from ppe_pipeline_api import data, training
    deleted = []
    monkeypatch.setattr(rs, "combined_enabled", lambda config=None: True)
    monkeypatch.setattr(data, "decay_marker", lambda: marker)
    monkeypatch.setattr(data, "combined_location", lambda: (B, "central", "central_decay"))
    monkeypatch.setattr(data, "delete_decayed", lambda: deleted.append(1) or 5)
    monkeypatch.setattr(training, "current_training_reason", lambda: running)
    monkeypatch.setattr(training, "queue_snapshot", lambda: {"pending": list(pending), "running": None})
    return rs, deleted


OLD = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat()
NEW = dt.datetime.now(dt.timezone.utc).isoformat()


def test_it_is_deleted_once_the_run_has_finished(monkeypatch):
    rs, deleted = _cleanup_setup(monkeypatch, marker={"status": "built", "for_run": True, "built_at": OLD})
    assert rs.cleanup_decayed().startswith("deleted") and deleted


@pytest.mark.parametrize("why", ["running", "queued", "just built", "by hand", "building"])
def test_it_is_kept_while_anything_could_still_read_it(monkeypatch, why):
    marker = {"status": "building" if why == "building" else "built",
              "for_run": why != "by hand", "built_at": NEW if why == "just built" else OLD}
    rs, deleted = _cleanup_setup(
        monkeypatch, marker=marker, running="Executing" if why == "running" else "",
        pending=[{"train_data_uris": f"s3://{B}/central_decay/"}] if why == "queued" else ())
    assert rs.cleanup_decayed() == "" and not deleted


def test_the_chain_trains_on_the_decayed_copy_when_ticked():
    code = (ROOT / "src/ppe_pipeline_api/chain/retrain_schedule.py").read_text(encoding="utf-8")
    assert 'splits = {"train": decayed["train"], "val": decayed["val"]}' in code
    assert "both = combine_and_decay(config, for_run=True)" in code


def test_the_ticker_cleans_up():
    ticker = (ROOT / "src/ppe_pipeline_api/chain/ticker.py").read_text(encoding="utf-8")
    assert "retrain_schedule.cleanup_decayed" in ticker


def test_the_card_has_the_button_and_saves_the_tick():
    card = (ROOT / "web/src/components/Track2CentralStorePanel.tsx").read_text(encoding="utf-8")
    assert "Combine data sources" in card
    assert "train_from_combined: trainFromCombined" in card


def test_stop_skips_the_rest_and_keeps_what_reached_the_store(s3):
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 5
    with pytest.raises(cs.Stopped):
        cs.combine(weeks_root="retraining", weeks_bucket=B, stop=stop)
    copied = [k for (b, k) in s3.objects if k.startswith("central/")]
    assert 0 < len(copied) < 40, "some copies made, the rest skipped"
    out = _combine()
    assert out["already"] == len(copied), "the next combine carries on from them"


def test_stop_during_the_decay_leaves_no_half_built_copy(s3):
    out = _combine()
    with pytest.raises(cs.Stopped):
        cs.rebuild_decayed(retention=POLICY, use_weeks=True, use_golden=True,
                           slugs=out["slugs"], stop=lambda: True)
    assert not any(k.startswith("central_decay/") for (b, k) in s3.objects)
    assert any(k.startswith("central/") for (b, k) in s3.objects), "the store is untouched"


def test_the_stop_button_is_wired():
    from ppe_pipeline_api.main import app
    assert "/track2/pull-control/central-store/combine/stop" in app.openapi()["paths"]
    card = (ROOT / "web/src/components/Track2CentralStorePanel.tsx").read_text(encoding="utf-8")
    assert "onClick={stopCombine}" in card
    assert "Catalog" not in card, "the catalog section left this card"


# --- progress, and outliving the page and the container ---------------------------

def test_progress_is_reported_for_each_copy(s3):
    seen = []
    cs.combine(weeks_root="retraining", weeks_bucket=B,
               progress=lambda phase, done, total: seen.append((phase, done, total)))
    copies = [s for s in seen if s[0] == "store"]
    assert copies[0][1] == 0 and copies[-1][1] == copies[-1][2] > 0


def test_a_broken_progress_display_never_stops_a_copy(s3):
    def boom(*a):
        raise RuntimeError("display gone")
    assert cs.combine(weeks_root="retraining", weeks_bucket=B, progress=boom)["copied"] > 0


def _resume_setup(monkeypatch, saved):
    from ppe_pipeline_api.chain import retrain_schedule as rs
    from ppe_pipeline_api import data
    started, written = [], []
    monkeypatch.setattr(rs, "_combine_job", {})
    monkeypatch.setattr(rs, "_combine_resume_checked", [False])
    monkeypatch.setattr(rs, "start_combine", lambda **kw: started.append(kw))
    monkeypatch.setattr(data, "load_combine_job", lambda: saved)
    monkeypatch.setattr(data, "save_combine_job", lambda job: written.append(job))
    monkeypatch.setattr(data, "decay_marker", lambda: {"status": "building"})
    monkeypatch.setattr(data, "delete_decayed", lambda: 0)
    monkeypatch.setenv("APP_ROLE", "dev")
    return rs, started, written


def test_a_restart_carries_on_a_running_combine(monkeypatch):
    rs, started, _ = _resume_setup(monkeypatch, {"running": True, "owner": "dev", "started_at": "t0"})
    assert rs.resume_combine().startswith("resumed")
    assert started == [{"resumed_from": {"running": True, "owner": "dev", "started_at": "t0"}}]
    assert rs.resume_combine() == "", "once per process"


@pytest.mark.parametrize("saved", [{"running": False}, {"running": True, "owner": "stable"}])
def test_nothing_to_carry_on(monkeypatch, saved):
    rs, started, _ = _resume_setup(monkeypatch, saved)
    assert rs.resume_combine() == "" and not started


def test_a_combine_being_stopped_is_finished_as_stopped(monkeypatch):
    rs, started, written = _resume_setup(monkeypatch, {"running": True, "owner": "dev", "stopping": True})
    rs.resume_combine()
    assert not started and written[-1]["stopped"] and not written[-1]["running"]


def test_the_ticker_resumes_and_the_card_shows_progress():
    ticker = (ROOT / "src/ppe_pipeline_api/chain/ticker.py").read_text(encoding="utf-8")
    assert "retrain_schedule.resume_combine" in ticker
    card = (ROOT / "web/src/components/Track2CentralStorePanel.tsx").read_text(encoding="utf-8")
    assert 'role="progressbar"' in card and "<CombineProgress job={combine} />" in card
