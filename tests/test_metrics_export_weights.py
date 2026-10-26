"""Exporting the best registered checkpoint of one Class.

What this pins, and why each one is here rather than being obvious:

  * **No eligibility guards.** Approval status and validation count do
    not exclude a version -- that was asked and answered -- so the tests
    assert the un-guarded behaviour deliberately, and also assert that
    both facts come back in the result. Not gating is a decision; hiding
    what it costs would be a defect.
  * **A missing metric is not zero.** The usual reason a version has no
    mAP50 is an imported checkpoint, which is exactly what somebody would
    otherwise read as the worst model in the group.
  * **The watcher triggers on a new version and writes on a new WINNER.**
    Those are two different conditions and conflating them writes a
    byte-identical copy of one checkpoint after every training run.
"""
from __future__ import annotations

import datetime as _dt
import importlib
import json

import pytest

mx = importlib.import_module("src.ppe_pipeline_api.metrics_export")
router = importlib.import_module("src.ppe_pipeline_api.routers.track2_metrics")


def row(version, *, mAP50=None, recall=None, status="Approved", val=None,
        group="yolo11s-helmet-detector", size="s"):
    metrics = None
    if mAP50 is not None or recall is not None or val is not None:
        metrics = {"mAP50": mAP50, "recall": recall,
                   "dataset": {"validation_count": val}}
    return {"group_name": group, "generation": "11", "size": size,
            "version": str(version),
            "model_package_arn": f"arn:aws:sagemaker:ap-southeast-1:1:model-package/{group}/{version}",
            "status": status, "metrics": metrics}


@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    folder = tmp_path / "export"
    monkeypatch.setattr(router, "EXPORT_SETTINGS_DIR", str(folder))
    return folder


# --- picking ------------------------------------------------------------

def test_highest_wins():
    rows = [row(1, mAP50=0.4), row(2, mAP50=0.9), row(3, mAP50=0.7)]
    best, why = mx.pick_best(rows, "mAP50")
    assert best["version"] == "2" and why == ""


def test_a_tie_goes_to_the_higher_version():
    """Something has to break it, and version order is the one ordering
    that is stable between two identical calls. Without this, "the best
    version" of three runs that all landed on 0.727 is whichever the
    registry happened to list first."""
    best, _ = mx.pick_best([row(3, mAP50=0.727), row(7, mAP50=0.727),
                            row(5, mAP50=0.727)], "mAP50")
    assert best["version"] == "7"


def test_a_version_with_no_metric_is_not_a_version_that_scored_zero():
    """An imported checkpoint has not been measured. Ranked as 0.0 it
    would sit at the bottom of a list it does not belong in at all -- and
    worse, it would WIN a group where nothing else was measured either."""
    rows = [row(1), row(2, mAP50=0.3)]
    best, _ = mx.pick_best(rows, "mAP50")
    assert best["version"] == "2"
    assert mx.metric_value(rows[0], "mAP50") is None


def test_a_group_with_nothing_measured_refuses_rather_than_picking_one():
    best, why = mx.pick_best([row(1), row(2)], "mAP50")
    assert best is None
    assert "mAP50" in why and "scored zero" in why


def test_newest_ranks_on_the_version_number():
    """A real answer, not a fallback: for a Class trained on a growing
    corpus the newest model saw the most data, whatever it scored against
    a validation set that also moved."""
    best, _ = mx.pick_best([row(9, mAP50=0.99), row(14, mAP50=0.0)], "newest")
    assert best["version"] == "14"


def test_version_order_is_numeric_not_textual():
    """As text, v10 sorts before v9 and the answer silently goes wrong at
    the tenth version."""
    best, _ = mx.pick_best([row(9, mAP50=0.5), row(10, mAP50=0.5)], "newest")
    assert best["version"] == "10"


def test_nothing_is_excluded_for_being_rejected_or_thinly_validated():
    """The live shape of this estate: three versions at 0.995 on FIVE
    validation images against one at 0.000 on 149. Ranked on mAP50 with
    no floor, five images win -- which is the behaviour that was asked
    for, so it is pinned rather than left to drift."""
    rows = [row(4, mAP50=0.995, val=5, status="Rejected"),
            row(9, mAP50=0.000, val=149)]
    best, _ = mx.pick_best(rows, "mAP50")
    assert best["version"] == "4"


def test_the_pick_states_what_it_did_not_filter_on():
    """Not gating is a decision. Hiding it would be a defect, so the two
    facts that make a high score meaningless travel WITH the pick."""
    pick = mx.describe_pick(row(4, mAP50=0.995, val=5, status="Rejected"), "mAP50")
    assert pick["validation_images"] == 5
    assert pick["thin_validation"] is True
    assert pick["status"] == "Rejected"


def test_a_healthy_validation_count_is_not_flagged():
    assert mx.describe_pick(row(9, mAP50=0.4, val=149), "mAP50")["thin_validation"] is False


def test_an_unknown_criterion_is_refused_rather_than_defaulted():
    """Falling back to mAP50 would export a different file from the one
    asked for and say nothing."""
    best, why = mx.pick_best([row(1, mAP50=0.4)], "vibes")
    assert best is None and "ranked by" in why


# --- naming -------------------------------------------------------------

def test_the_name_says_which_model_it_is():
    when = _dt.datetime(2026, 10, 8, 9, 15, 14, tzinfo=_dt.timezone.utc)
    assert mx.export_filename("yolo11s-helmet-detector", "7", when) \
        == "yolo11-s-helmet-v7-20261008-091514.pt"


def test_a_yolo26_export_does_not_carry_eleven_in_its_name():
    when = _dt.datetime(2026, 10, 8, 9, 15, 14, tzinfo=_dt.timezone.utc)
    name = mx.export_filename("yolo26m-open-detector", "3", when)
    assert name.startswith("yolo26-m-open-v3-")
    assert "yolo11" not in name


def test_two_exports_of_the_same_version_do_not_collide():
    """Timestamped, never overwritten: an automatic export must not be
    able to replace a file something downstream already fetched."""
    a = mx.export_filename("yolo11s-helmet-detector", "7",
                           _dt.datetime(2026, 10, 8, 9, 15, 14, tzinfo=_dt.timezone.utc))
    b = mx.export_filename("yolo11s-helmet-detector", "7",
                           _dt.datetime(2026, 10, 8, 9, 15, 15, tzinfo=_dt.timezone.utc))
    assert a != b


def test_the_two_export_paths_name_a_checkpoint_the_same_way():
    """`export_weights_to_folder` has always built its stem one way. Two
    stems in one folder for the same group would make the folder unreadable
    at exactly the moment it holds exports from both paths."""
    from src.ppe_pipeline_api import track2_shared as t2

    for group in ("yolo11s-detector", "yolo26m-open-detector",
                  "yolo11l-helmet-detector", "track2-yolo-custom-detector"):
        assert mx.export_filename(group, "1").startswith(f"{t2.export_stem(group)}-v1-")


# --- the watcher --------------------------------------------------------

def test_it_is_off_until_it_is_armed(settings_dir):
    assert mx.auto_settings()["enabled"] is False
    assert mx.pending()["act"] is False


def test_arming_marks_the_current_newest_as_already_seen(settings_dir, monkeypatch):
    """Otherwise arming a watcher on a group holding fourteen versions
    immediately exports v14 -- a version registered weeks ago, which is
    not "a new version registered"."""
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    plan = mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
                  bucket="track2-s3", prefix="exports")
    assert plan["enabled"] is True
    assert plan["last_version"] == 14
    assert mx.pending(plan)["act"] is False


def test_a_new_version_makes_it_act(settings_dir, monkeypatch):
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    plan = mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
                  bucket="track2-s3", prefix="")
    monkeypatch.setattr(mx, "newest_version", lambda group: 15)
    decision = mx.pending(plan)
    assert decision["act"] is True and decision["version"] == 15


def test_an_automatic_export_to_a_laptop_is_refused(settings_dir):
    """There is no browser attached when it fires, so "this computer" has
    nowhere to put the file. Said, rather than accepted and then silently
    doing nothing."""
    with pytest.raises(ValueError) as exc:
        mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
               bucket="", prefix="")
    assert "S3 bucket" in str(exc.value)


def test_arming_without_a_group_is_refused(settings_dir):
    with pytest.raises(ValueError):
        mx.arm(group_name="", criterion="mAP50", bucket="b", prefix="")


def test_a_new_version_that_does_not_win_writes_nothing(settings_dir, monkeypatch):
    """A training run finishing is not a reason to re-export. v15 scoring
    0.3 against v4's 0.995 leaves v4 the best, and a second byte-identical
    timestamped copy of v4 only makes the folder harder to read."""
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
           bucket="track2-s3", prefix="")
    mx.write_auto({"last_exported_version": "4"})
    monkeypatch.setattr(mx, "newest_version", lambda group: 15)
    monkeypatch.setattr(mx, "group_versions",
                        lambda group: [row(4, mAP50=0.995, val=5), row(15, mAP50=0.3, val=149)])

    def refuse(*args, **kwargs):
        raise AssertionError("exported a checkpoint that was already exported")

    monkeypatch.setattr(mx, "export_checkpoint", refuse)
    out = mx.check()
    assert out["exported"] is False
    assert out["result"]["skipped"] is True
    # The trigger still moves, or this decision is retaken every 25s.
    assert mx.auto_settings()["last_version"] == 15


def test_a_new_winner_is_exported_and_recorded(settings_dir, monkeypatch):
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
           bucket="track2-s3", prefix="exports")
    mx.write_auto({"last_exported_version": "4"})
    monkeypatch.setattr(mx, "newest_version", lambda group: 15)
    monkeypatch.setattr(mx, "group_versions",
                        lambda group: [row(4, mAP50=0.5, val=5), row(15, mAP50=0.9, val=149)])
    sent = {}
    monkeypatch.setattr(mx, "export_checkpoint", lambda best, **kw: sent.update(
        {"version": best["version"], **kw}) or {"written_to": "s3://track2-s3/exports/x.pt",
                                                "destination": "s3", "filename": "x.pt", "url": ""})
    out = mx.check()
    assert out["exported"] is True
    assert sent["version"] == "15"
    # S3, always: the watcher has no browser to download to.
    assert sent["destination"] == "s3" and sent["bucket"] == "track2-s3"
    assert mx.auto_settings()["last_exported_version"] == "15"


def test_a_failed_export_still_moves_the_marker(settings_dir, monkeypatch):
    """A bucket the role cannot write to will not become writable 25
    seconds later. Not recording the version turns one misconfiguration
    into a failed S3 call every 25 seconds until somebody notices."""
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
           bucket="track2-s3", prefix="")
    monkeypatch.setattr(mx, "newest_version", lambda group: 15)
    monkeypatch.setattr(mx, "group_versions", lambda group: [row(15, mAP50=0.9, val=149)])

    def denied(*args, **kwargs):
        raise RuntimeError("AccessDenied")

    monkeypatch.setattr(mx, "export_checkpoint", denied)
    out = mx.check()
    assert out["exported"] is False
    assert "AccessDenied" in out["result"]["detail"]
    assert mx.auto_settings()["last_version"] == 15
    # NOT recorded as exported -- the next winner must still go out.
    assert mx.auto_settings()["last_exported_version"] == ""


def test_switching_it_off_stops_it(settings_dir, monkeypatch):
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
           bucket="track2-s3", prefix="")
    mx.disarm()
    assert mx.check()["exported"] is False
    assert mx.auto_settings()["enabled"] is False


def test_rearming_on_another_group_forgets_what_it_exported(settings_dir, monkeypatch):
    """A stale "already exported v4" would skip the first export into a
    bucket holding nothing at all."""
    monkeypatch.setattr(mx, "newest_version", lambda group: 14)
    mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
           bucket="b", prefix="")
    mx.write_auto({"last_exported_version": "4"})
    mx.arm(group_name="yolo26m-open-detector", criterion="recall",
           bucket="b", prefix="")
    assert mx.auto_settings()["last_exported_version"] == ""


def test_the_watcher_file_is_written_whole(settings_dir, monkeypatch):
    """A half-written file read back by the next tick is a watcher with no
    marker, which re-exports whatever it last exported."""
    import inspect

    source = inspect.getsource(mx.write_auto)
    assert "os.replace(" in source and ".tmp" in source


def test_it_survives_a_corrupt_watcher_file(settings_dir):
    """Returning the defaults -- i.e. OFF -- is the safe reading. A parse
    error that armed something would be the worst possible answer."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / mx.AUTO_FILE).write_text("{not json", encoding="utf-8")
    assert mx.auto_settings()["enabled"] is False


def test_the_tick_calls_it(settings_dir):
    """The watcher only exists if something runs it. It rides the app's
    existing 25s poll rather than a loop of its own, so this asserts the
    call site rather than trusting that it is there."""
    import inspect

    from src.ppe_pipeline_api import track2_shared as t2

    source = inspect.getsource(t2.queue_poll_loop)
    # Handed to a thread rather than called inline -- it is synchronous
    # boto3 and the loop it rides is the one serving every request -- so
    # the call reads `to_thread(metrics_export.check)`, with no parens.
    assert "metrics_export" in source
    assert "asyncio.to_thread(metrics_export.check" in source


# --- what gets exported -------------------------------------------------

def test_the_checkpoint_never_travels_through_this_container():
    """A 19MB-110MB .pt streamed through the API would stall every request
    in this single-worker process for its duration. S3 is a server-side
    copy; a download is a presigned link the browser follows itself."""
    import inspect

    source = inspect.getsource(mx.export_checkpoint)
    assert "copy_object" in source
    assert "generate_presigned_url" in source
    assert "download_file" not in source
    assert "upload_file" not in source


def test_both_destinations_pick_the_same_version():
    """One `run_export`, so a manual export and an automatic one cannot
    choose differently -- a difference that would only show up months
    later in whatever was consuming the folder."""
    import inspect

    source = inspect.getsource(mx.run_export)
    assert "pick_best(" in source
    assert "export_checkpoint(" in source


def test_listing_a_groups_versions_is_paginated():
    """`model_registry_summary` passes no MaxResults and takes whatever
    the service returns -- tolerable where it shows the top few, not where
    "the best version" would become "the best of whichever came back"."""
    import inspect

    source = inspect.getsource(mx.registry_packages)
    assert "NextToken" in source


def test_the_preview_and_the_export_rank_the_same_way():
    """Two implementations of "best" -- one deciding what the card says,
    one deciding what the button does -- is how a card comes to promise v4
    and ship v9."""
    import inspect

    assert "mx.pick_best(" in inspect.getsource(router.preview_best_version)
    assert "mx.run_export(" in inspect.getsource(router.export_best_weights)


# --- the group matrix ---------------------------------------------------

def test_every_group_is_offered_before_anything_registers_in_it(monkeypatch):
    """The card's dropdowns come from here, not from the versions that
    exist. Built from what exists, a Class that has only ever trained at
    YOLO11-S offers YOLO11 and S and nothing else -- so the group a
    YOLO26-X run will land in next week is unreachable, which is exactly
    when you would want it armed."""
    from src.ppe_pipeline_api import track2_shared as t2

    helmet = {"slug": "helmet", "builtin": False,
              "group_pattern": "yolo{generation}{size}-helmet-detector"}
    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [helmet])
    names = {g["group_name"] for g in t2.exportable_groups()}
    for generation in ("11", "26"):
        for size in ("s", "m", "l", "x"):
            assert f"yolo{generation}{size}-helmet-detector" in names


def test_the_matrix_carries_the_generation_the_metrics_scan_threw_away(monkeypatch):
    from src.ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [])
    entry = next(g for g in t2.exportable_groups()
                 if g["group_name"] == "yolo26x-open-detector")
    assert entry["generation"] == "26" and entry["size"] == "x"


def test_production_stays_yolo11_only(monkeypatch):
    """`yolo26{size}-detector` does not exist in AWS and is not on the
    exact-match list the approval rule watches. Offering it would be a
    group that looks like production and can never deploy."""
    from src.ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [])
    production = [g for g in t2.exportable_groups() if g["registry"] == "production"]
    assert {g["generation"] for g in production} == {"11"}
    assert {g["size"] for g in production} == {"s", "m", "l", "x"}


def test_the_legacy_group_is_not_given_a_generation_it_never_had(monkeypatch):
    from src.ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [])
    entry = next(g for g in t2.exportable_groups()
                 if g["group_name"] == t2.LEGACY_OPEN_REGISTRY_GROUP)
    assert entry["generation"] == "" and entry["size"] == ""


def test_the_chart_and_the_export_card_enumerate_the_same_groups():
    """Two lists of "every group" is two chances for a group to be
    exportable and unchartable, or the reverse."""
    import inspect

    from src.ppe_pipeline_api import track2_shared as t2

    assert "exportable_groups()" in inspect.getsource(t2.scan_all_model_metrics)


def test_a_group_name_is_never_rebuilt_from_parts(monkeypatch):
    """Composing "yolo{gen}{size}-{slug}-detector" by hand is how the
    Models card once told somebody a YOLO26 run would register into
    yolo11s-helmet-detector while the run used the right name."""
    import inspect

    from src.ppe_pipeline_api import track2_shared as t2

    source = inspect.getsource(t2.exportable_groups)
    # Either helper is fine -- the point is that the name comes from one
    # of them and is never composed here. `detector_history_groups` is
    # the right one for a LISTING: after a rename the versions trained
    # under the old identifier are still in the old groups, and exporting
    # should still find them.
    assert ("detector_history_groups(" in source
            or "detector_groups(" in source), (
        "build the group name with the shared helper, never from parts"
    )
    assert "builtin_open_groups()" in source


def test_arming_a_group_with_nothing_in_it_waits_rather_than_erroring(settings_dir, monkeypatch):
    """Setting one up before its first run is the whole point. v1 landing
    later must then count as new."""
    monkeypatch.setattr(mx, "newest_version", lambda group: 0)
    plan = mx.arm(group_name="yolo26x-helmet-detector", criterion="mAP50",
                  bucket="track2-s3", prefix="")
    assert plan["last_version"] == 0
    assert mx.pending(plan)["act"] is False
    monkeypatch.setattr(mx, "newest_version", lambda group: 1)
    assert mx.pending(plan)["act"] is True


# --- what happens to the earlier exports --------------------------------

class FakeS3:
    """Enough of S3 to watch what retention touches, and what it leaves."""

    def __init__(self, keys):
        self.keys = list(keys)
        self.copied = []
        self.deleted = []

    def list_objects_v2(self, **kw):
        prefix = kw.get("Prefix", "")
        return {"Contents": [{"Key": k} for k in self.keys if k.startswith(prefix)],
                "IsTruncated": False}

    def copy_object(self, **kw):
        self.copied.append((kw["CopySource"]["Key"], kw["Key"]))
        self.keys.append(kw["Key"])

    def delete_object(self, **kw):
        self.deleted.append(kw["Key"])
        self.keys.remove(kw["Key"])


FOLDER = [
    "exports/yolo11-s-helmet-v4-20261001-090000.pt",
    "exports/yolo11-s-helmet-v7-20261007-090000.pt",
    "exports/yolo11-s-helmet-v9-20261008-100000.pt",   # the one just written
    # Everything below is somebody else's and must survive all of it.
    "exports/yolo11-m-segment-v2-20261001-090000.pt",
    "exports/README.md",
    "exports/notes.txt",
    "exports/archive/yolo11-s-helmet-v1-20260901-090000.pt",
    "exports/old/yolo11-s-helmet-v2-20260902-090000.pt",
]
JUST_WRITTEN = "exports/yolo11-s-helmet-v9-20261008-100000.pt"


def fake_s3(monkeypatch, keys=None):
    from src.ppe_pipeline_api import track2_shared as t2

    client = FakeS3(keys if keys is not None else FOLDER)
    monkeypatch.setattr(t2, "s3", client)
    return client


def test_it_only_sees_this_classes_own_exports(monkeypatch):
    """An export folder is a normal S3 prefix that may hold somebody
    else's data. A retention rule on a checkpoint export has no business
    removing a README, and must never reach the segment Class's files."""
    fake_s3(monkeypatch)
    found = mx.export_siblings("b", "exports", "yolo11s-helmet-detector",
                               exclude=JUST_WRITTEN)
    assert found == ["exports/yolo11-s-helmet-v4-20261001-090000.pt",
                     "exports/yolo11-s-helmet-v7-20261007-090000.pt"]


def test_it_never_reaches_into_a_subfolder(monkeypatch):
    """Which also keeps archive/ out of its own reach -- otherwise
    archiving would feed itself every run, moving yesterday's archive
    into today's."""
    fake_s3(monkeypatch)
    found = mx.export_siblings("b", "exports", "yolo11s-helmet-detector")
    assert not any("/archive/" in k or "/old/" in k for k in found)


def test_keeping_everything_touches_nothing(monkeypatch):
    client = fake_s3(monkeypatch)
    out = mx.apply_retention("b", "exports", "yolo11s-helmet-detector",
                             mode="keep_all", keep_key=JUST_WRITTEN)
    assert out == {"mode": "keep_all", "archived": [], "deleted": []}
    assert client.deleted == [] and client.copied == []


def test_archiving_copies_before_it_deletes(monkeypatch):
    """A delete that ran first, or ran regardless, turns "archive it" into
    "lose it" the moment the copy is refused."""
    client = fake_s3(monkeypatch)
    out = mx.apply_retention("b", "exports", "yolo11s-helmet-detector",
                             mode="archive_oldest", keep_key=JUST_WRITTEN)
    assert len(out["archived"]) == 2
    assert all(k.startswith("exports/archive/") for k in out["archived"])
    for index, (source, _target) in enumerate(client.copied):
        assert index <= client.deleted.index(source)
    assert JUST_WRITTEN in client.keys


def test_keep_latest_removes_the_earlier_ones_and_nothing_else(monkeypatch):
    client = fake_s3(monkeypatch)
    out = mx.apply_retention("b", "exports", "yolo11s-helmet-detector",
                             mode="keep_latest", keep_key=JUST_WRITTEN)
    assert len(out["deleted"]) == 2
    assert JUST_WRITTEN in client.keys
    for survivor in ("exports/README.md", "exports/notes.txt",
                     "exports/yolo11-m-segment-v2-20261001-090000.pt",
                     "exports/archive/yolo11-s-helmet-v1-20260901-090000.pt"):
        assert survivor in client.keys


def test_the_new_export_is_never_the_one_removed(monkeypatch):
    for mode in ("archive_oldest", "keep_latest"):
        client = fake_s3(monkeypatch)
        mx.apply_retention("b", "exports", "yolo11s-helmet-detector",
                           mode=mode, keep_key=JUST_WRITTEN)
        assert JUST_WRITTEN not in client.deleted, mode


def test_tidying_happens_after_the_new_file_lands():
    """The other order leaves a window where the folder holds nothing at
    all, and a consumer reading it then gets an empty answer rather than
    an old one -- the worse failure, because an old checkpoint is still a
    checkpoint."""
    import inspect

    source = inspect.getsource(mx.export_checkpoint)
    assert source.index("copy_object") < source.index("apply_retention")


def test_a_download_tidies_nothing(monkeypatch):
    """A download adds no file to anybody's folder, so there is no earlier
    export of it to remove -- and a retention mode that reached into S3 on
    a download would be acting somewhere nobody asked it to."""
    from src.ppe_pipeline_api import track2_shared as t2

    client = fake_s3(monkeypatch)
    monkeypatch.setattr(t2, "publish_versioned_weights",
                        lambda arn: "s3://track2-s3/models/g/v9/model.pt")
    client.generate_presigned_url = lambda *a, **kw: "https://signed"
    out = mx.export_checkpoint(row(9, mAP50=0.9), destination="local",
                               retention="keep_latest")
    assert out["retention"]["deleted"] == []
    assert client.deleted == []


def test_an_unknown_retention_mode_is_refused_rather_than_defaulted(monkeypatch):
    fake_s3(monkeypatch)
    with pytest.raises(ValueError):
        mx.apply_retention("b", "exports", "yolo11s-helmet-detector",
                           mode="burn_it_all", keep_key=JUST_WRITTEN)


def test_the_watcher_carries_its_own_retention_mode(settings_dir, monkeypatch):
    """Taken at arming time. Reading the card's radio instead would mean a
    mode changed on screen silently changed what an already-armed watcher
    does to a folder."""
    monkeypatch.setattr(mx, "newest_version", lambda group: 3)
    plan = mx.arm(group_name="yolo11s-helmet-detector", criterion="mAP50",
                  bucket="b", prefix="exports", retention="keep_latest")
    assert plan["retention"] == "keep_latest"


def test_arming_with_a_mode_that_is_not_one_is_refused(settings_dir):
    with pytest.raises(ValueError):
        mx.arm(group_name="g", criterion="mAP50", bucket="b", prefix="",
               retention="burn_it_all")


def test_a_watcher_armed_before_retention_existed_keeps_everything(settings_dir):
    """The safe reading of silence: a mode nobody chose must not delete."""
    settings_dir.mkdir(parents=True, exist_ok=True)
    (settings_dir / mx.AUTO_FILE).write_text(
        json.dumps({"enabled": True, "group_name": "g", "bucket": "b"}),
        encoding="utf-8")
    assert mx.auto_settings()["retention"] == "keep_all"


def test_the_preview_lists_exactly_what_the_modes_would_act_on():
    """A preview scoped differently from the deletion would be worse than
    no preview at all."""
    import inspect

    assert "mx.export_siblings(" in inspect.getsource(router.export_folder_contents)
    assert "export_siblings(" in inspect.getsource(mx.apply_retention)


# --- the Class list -----------------------------------------------------

def test_the_open_registry_is_not_offered_as_a_class(monkeypatch):
    """`custom` is where a run with no declared Class lands -- a different
    kind of thing from a Class, in a dropdown labelled Class."""
    from src.ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [])
    offered = {g["registry"] for g in router.exportable_groups(session=None)["groups"]}
    assert "custom" not in offered
    assert "production" in offered


def test_those_groups_stay_on_the_comparison_chart(monkeypatch):
    """Removed from a picker, not from the platform. A registered model
    that appears on no page is how a YOLO26 open run was once trained,
    Approved and invisible."""
    from src.ppe_pipeline_api import track2_shared as t2

    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [])
    assert any(g["registry"] == "custom" for g in t2.exportable_groups())


def test_the_card_is_handed_the_name_a_file_will_really_carry(monkeypatch):
    """Composed in the browser as `group_name.replace(/-detector$/, "")`
    this came out `yolo11s-helmet`, while the file lands as
    `yolo11-s-helmet` -- so the line promising where a file would be
    written named a key that never appears in the bucket."""
    from src.ppe_pipeline_api import track2_shared as t2

    helmet = {"slug": "helmet", "builtin": False,
              "group_pattern": "yolo{generation}{size}-helmet-detector"}
    monkeypatch.setattr(t2, "list_detectors", lambda **kw: [helmet])
    entry = next(g for g in t2.exportable_groups()
                 if g["group_name"] == "yolo11s-helmet-detector")
    assert entry["export_stem"] == "yolo11-s-helmet"
    assert mx.export_filename(entry["group_name"], "7").startswith(
        f"{entry['export_stem']}-v7-")


# --- what the audit of this page turned up ------------------------------

def test_a_group_past_one_page_is_read_whole(monkeypatch):
    """`model_registry_summary` passed no MaxResults and took whatever one
    call returned. Enough today -- the 14-version helmet group comes back
    whole -- and silently not enough at the service cap, which would be
    versions missing from the comparison chart and from every count on the
    page with nothing erroring to say so."""
    from src.ppe_pipeline_api import track2_shared as t2

    pages = [
        {"ModelPackageSummaryList": [
            {"ModelPackageArn": f"arn:...:model-package/g/{n}",
             "ModelApprovalStatus": "Approved"} for n in range(1, 101)],
         "NextToken": "more"},
        {"ModelPackageSummaryList": [
            {"ModelPackageArn": "arn:...:model-package/g/101",
             "ModelApprovalStatus": "Rejected"}]},
    ]
    seen = []

    class FakeSm:
        def list_model_packages(self, **kw):
            seen.append(kw.get("NextToken", ""))
            return pages[len(seen) - 1]

        def describe_model_package(self, **kw):
            return {}

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "fetch_lineage", lambda i: None)
    out = t2.model_registry_summary("g")
    assert out["total"] == 101
    assert out["approved"] == 100
    assert seen == ["", "more"]


def test_the_preview_may_cache_but_the_watcher_may_not(settings_dir, monkeypatch):
    """One `group_versions` is a describe plus an S3 read per version --
    1.7s for the helmet group -- and the card calls it every time the
    criterion dropdown moves. The WATCHER runs precisely because a new
    version just appeared, so a cached answer is the one answer it cannot
    use: it would decide "no new winner" off a list that predates the
    thing that woke it."""
    import inspect

    assert "max_age=20" in inspect.getsource(router.preview_best_version)
    # The watcher's own call takes the default, which is no cache at all.
    check_source = inspect.getsource(mx.check)
    assert "group_versions(str(plan[\"group_name\"]))" in check_source
    assert "max_age" not in check_source
    assert inspect.signature(mx.group_versions).parameters["max_age"].default == 0.0


def test_a_cached_read_is_a_copy_not_the_cache(monkeypatch):
    """A caller mutating a row it was handed would otherwise poison every
    later read of that group."""
    from src.ppe_pipeline_api import track2_shared as t2

    mx._versions_cache.clear()

    class FakeSm:
        def list_model_packages(self, **kw):
            return {"ModelPackageSummaryList": [
                {"ModelPackageArn": "arn:...:model-package/g/1"}]}

        def describe_model_package(self, **kw):
            return {"ModelApprovalStatus": "Approved"}

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "fetch_metrics_and_lineage", lambda arn: ({"mAP50": 0.5}, "", None))
    first = mx.group_versions("g")
    first[0]["status"] = "TAMPERED"
    assert mx.group_versions("g", max_age=60)[0]["status"] == "Approved"
    mx._versions_cache.clear()
