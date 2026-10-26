"""Where an arriving file ends up.

The rule under test is the one the Lambda runs unattended overnight, so
the cases that matter are the awkward ones: a file older than every week,
a file a month past the newest one, a name that already exists at the
destination.
"""

from __future__ import annotations

import datetime as _dt

import pytest
from botocore.exceptions import ClientError

from ppe_pipeline_api import pull_control as pc

UTC = _dt.timezone.utc


def _at(day: int, hour: int = 12) -> _dt.datetime:
    return _dt.datetime(2026, 9, day, hour, tzinfo=UTC)


WEEKS = [
    {"name": "week_1", "starts_at": _at(1, 0).isoformat()},
    {"name": "week_2", "starts_at": _at(8, 0).isoformat()},
]

#: The cut-off these two weeks are the same thing as. Week membership is
#: arithmetic from the cut-off now, not a lookup in the rows above --
#: `week_at` is the only implementation, so a stored row records only
#: that a week EXISTS, never where a file goes. Week 1 is everything
#: before the 8th; week 2 is the 8th to the 15th.
WEEKS_CUTOFF = _at(8, 0)


class FakeDdb:
    """An in-memory stand-in for the pull-control table.

    Autoused below. Two tests here drove `lambda_handler` end to end with
    only `_record_run` stubbed, so `write_run_state` reached the LIVE
    table on every run of the suite -- writing run rows into production
    and, on any path that touched `write_config`, able to overwrite the
    live configuration outright. The conftest guard turned that from
    silent into a failure; this makes the tests honest instead.
    """

    def __init__(self) -> None:
        self.items: dict = {}

    @staticmethod
    def _key(key: dict) -> tuple:
        return tuple(sorted((k, tuple(v.items())[0][1]) for k, v in key.items()))

    def put_item(self, **kwargs):
        item = kwargs["Item"]
        self.items[self._key({k: item[k] for k in ("kind", "sort") if k in item})] = item
        return {}

    def get_item(self, **kwargs):
        got = self.items.get(self._key(kwargs["Key"]))
        return {"Item": got} if got else {}

    def query(self, **kwargs):
        return {"Items": []}

    def delete_item(self, **kwargs):
        self.items.pop(self._key(kwargs["Key"]), None)
        return {}


@pytest.fixture(autouse=True)
def _fake_table(monkeypatch):
    monkeypatch.setattr(pc, "ddb", FakeDdb())


class FakeS3:
    """Enough S3 to route with: list, head, copy, delete.

    One dict is the bucket. A copied object becomes listable, which real
    S3 does and which the cross-week dedup depends on -- keeping copies in
    a side set instead made the fake claim a dedup rule worked when it
    could not have seen the file.
    """

    def __init__(self, objects: dict, existing: set | None = None):
        # {key: LastModified}
        self.objects = dict(objects)
        for key in existing or []:
            self.objects.setdefault(key, _at(1))
        # {key: etag}. Distinct per key by default, so two different files
        # never look identical; a copy carries the source's across, which
        # is what real S3 does and what makes "same name, same bytes"
        # distinguishable from "same name, revised upstream".
        self.etags = {key: f"etag-of-{key}" for key in self.objects}
        self.copied: list = []
        self.deleted: list = []
        self.put: list = []

    def list_objects_v2(self, **kwargs):
        prefix = kwargs["Prefix"]
        contents = [
            {"Key": key, "LastModified": when,
             "ETag": '"' + self.etags.get(key, f"etag-of-{key}") + '"'}
            for key, when in sorted(self.objects.items())
            if key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def head_object(self, **kwargs):
        if kwargs["Key"] in self.objects:
            return {}
        raise ClientError({"Error": {"Code": "404"}}, "HeadObject")

    def copy_object(self, **kwargs):
        source = kwargs["CopySource"]["Key"]
        self.copied.append((source, kwargs["Key"]))
        self.objects[kwargs["Key"]] = self.objects.get(source, _at(1))
        self.etags[kwargs["Key"]] = self.etags.get(source, f"etag-of-{source}")

    def delete_object(self, **kwargs):
        self.deleted.append(kwargs["Key"])
        self.objects.pop(kwargs["Key"], None)

    def put_object(self, **kwargs):
        self.put.append(kwargs["Key"])


@pytest.fixture
def config():
    """A genuine staging drop folder, so `move` is the right action.

    Spelled out rather than left to the default, which is `copy` -- the
    default has to be the one that cannot empty a folder that turns out to
    be a dataset, and these tests are about the other case.
    """
    return {
        **pc.DEFAULTS,
        # These exercise the per-file model, where a file's own S3
        # timestamp picks its week. The default is `scan` -- one week per
        # sweep -- which is what a source edited in place needs, since it
        # has no per-batch event to date anything by.
        "week_by": "file_time",
        "week_cutoff": WEEKS_CUTOFF.isoformat(),
        "week_length_days": 7,
        "sources": [{
            "prefix": "drop/train",
            "destination": "retraining_training",
            "action": "move",
        }],
    }


def test_a_file_lands_in_the_last_week_that_had_started():
    assert pc.week_for(_at(3), WEEKS)["name"] == "week_1"
    assert pc.week_for(_at(8, 0), WEEKS)["name"] == "week_2"
    # Unbounded, the newest week would swallow every future file and no
    # new week would ever be created.
    assert pc.week_for(_at(12), WEEKS, 7)["name"] == "week_2"
    assert pc.week_for(_at(30), WEEKS, 7) is None


def test_a_file_older_than_week_1_belongs_to_no_week():
    """Not week 1 by default.

    Silently filing it into the first week would put data from before the
    programme started inside it, which is worse than refusing: the refusal
    is visible, the wrong week is not.
    """
    assert pc.week_for(_dt.datetime(2026, 8, 1, tzinfo=UTC), WEEKS) is None


def test_a_late_file_lands_in_its_own_week_not_a_long_one(monkeypatch, config):
    """One week per length, not one long week.

    A file two weeks past the newest week goes into the week its own date
    falls in -- not into a single stretched "week 3", and not into the
    newest week that happens to exist.

    It no longer creates the weeks it skipped. That used to be necessary
    because the next week's NUMBER came from counting rows, so a gap
    renumbered everything after it. The number is arithmetic from the
    cut-off now, so week_4 is week_4 whether or not week_3 was ever
    written -- and the gap is shown in the forecast instead of being
    written to S3 as empty folders.
    """
    made = []

    def fake_create(*, name, starts_at, make_folders=True):
        week = {"name": name, "starts_at": pc._parse_time(starts_at).isoformat()}
        made.append(week)
        return week

    monkeypatch.setattr(pc, "create_week", fake_create)

    # 24 Sep is 16 days past the cut-off: two whole weeks elapsed, so the
    # third one after the cut-off -- week_4.
    week, weeks = pc.ensure_week_covering(_at(24), list(WEEKS), config)
    assert week["name"] == "week_4"
    assert [w["name"] for w in made] == ["week_4"]
    assert made[0]["starts_at"] == _at(22, 0).isoformat()
    assert len(weeks) == 3


def test_a_skipped_week_still_shows_in_the_forecast(monkeypatch, config):
    """The gap is visible, even though nothing was written for it.

    This is the half of the old behaviour worth keeping: jumping from
    week_2 to week_4 with no week_3 row at all would read as a numbering
    bug rather than as a week when nothing arrived.
    """
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {
        "weeks": [
            {"week": "week_1", "images": 5, "labels": 5, "bytes": 10,
             "roots": ["retraining_training"], "defined": True},
            {"week": "week_4", "images": 2, "labels": 2, "bytes": 4,
             "roots": ["retraining_training"], "defined": True},
        ],
        "roots": ["retraining_training"],
    })
    monkeypatch.setattr(pc, "_now", lambda: _at(24))

    out = pc.week_forecast(config=config)
    names = [w["week"] for w in out["weeks"]]
    assert names[:4] == ["week_1", "week_2", "week_3", "week_4"]

    gap = {w["week"]: w for w in out["weeks"]}["week_3"]
    assert gap["exists"] is False
    assert gap["images"] == 0
    # A past week whose folder was never created. Not "empty" -- that is
    # reserved for a folder that exists and holds nothing, which is a
    # different thing you can open and delete.
    assert gap["state"] == "missing"

    # And every row says when it closes, week 1 included.
    assert gap["closes_at"] == _at(22, 0).isoformat()
    first = {w["week"]: w for w in out["weeks"]}["week_1"]
    assert first["closes_at"] == WEEKS_CUTOFF.isoformat()
    assert first["is_cutoff"] is True


def test_auto_create_off_refuses_rather_than_guessing(monkeypatch, config):
    config["auto_create_weeks"] = False
    monkeypatch.setattr(pc, "create_week", lambda **kw: pytest.fail("should not create"))
    week, _ = pc.ensure_week_covering(_at(24), list(WEEKS), config)
    assert week is None


def test_extension_decides_images_or_labels(config):
    assert pc._subfolder_for("frame_001.JPG", config) == "images"
    assert pc._subfolder_for("frame_001.txt", config) == "labels"
    # data.yaml, index.json and friends are not the dataset.
    assert pc._subfolder_for("metadata.json", config) is None


def test_a_sweep_moves_each_file_into_its_week(monkeypatch, config):
    s3 = FakeS3({
        "drop/train/a.jpg": _at(3),
        "drop/train/a.txt": _at(3),
        "drop/train/b.jpg": _at(9),
        "drop/train/notes.json": _at(9),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=config)

    assert out["moved"] == 3
    assert dict(s3.copied) == {
        "drop/train/a.jpg": "retraining_training/week_1/images/a.jpg",
        "drop/train/a.txt": "retraining_training/week_1/labels/a.txt",
        "drop/train/b.jpg": "retraining_training/week_2/images/b.jpg",
    }
    # The copy has to happen before the delete, or a failed copy loses the
    # file outright -- the drop folder is the only record it is unfiled.
    assert set(s3.deleted) == set(dict(s3.copied))
    assert any("not an image or a label" in s["reason"] for s in out["skipped_files"])


def test_an_identical_file_already_there_is_left_alone(monkeypatch, config):
    """Same name AND same bytes is nothing to do.

    Never renamed: an image and its label pair on their shared stem, so
    suffixing one half of a collision and not the other unpairs them and
    the label quietly stops describing any image.
    """
    s3 = FakeS3(
        {"drop/train/a.jpg": _at(3)},
        existing={"retraining_training/week_1/images/a.jpg"},
    )
    # Same content, which is what "already filed" means.
    s3.etags["retraining_training/week_1/images/a.jpg"] = s3.etags["drop/train/a.jpg"]
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=config)
    assert out["moved"] == 0 and out["skipped"] == 1
    assert s3.copied == [] and s3.deleted == []


def test_the_same_name_with_new_bytes_is_treated_as_a_revision(monkeypatch, config):
    """Same name, different content means the source was edited.

    The upstream folder is authoritative and is edited in place, so a
    changed file is a correction to take, not a collision to refuse. That
    is a judgement about THIS data: the filenames are content hashes or
    unique alarm ids, and every repeated name measured across five days of
    the live dataset was byte-identical. Two genuinely different files
    sharing a name would be indistinguishable from a revision, so the
    count is reported separately rather than folded into `moved`.
    """
    s3 = FakeS3(
        {"drop/train/a.jpg": _at(3)},
        existing={"retraining_training/week_1/images/a.jpg"},
    )
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=config)
    assert out["revised"] == 1
    # Rewritten where it already sits, so no second copy appears in a
    # later week holding a contradicting label.
    assert dict(s3.copied) == {
        "drop/train/a.jpg": "retraining_training/week_1/images/a.jpg",
    }


def test_a_preview_projects_the_new_week_without_creating_it(monkeypatch, config):
    """Looking is not doing, but looking still has to answer the question.

    A file past the newest week goes into a week the sweep would create.
    The preview names that week and flags it as new -- reporting the file
    as merely "skipped" would describe an outcome the sweep does not have.
    """
    s3 = FakeS3({"drop/train/late.jpg": _at(24)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "create_week", lambda **kw: pytest.fail("preview created a week"))

    out = pc.plan_routing(config=config)

    assert out["planned"][0]["destination"] == "retraining_training/week_4/images/late.jpg"
    assert out["planned"][0]["new_week"] is True
    # Only the week the file actually belongs to. The weeks it skipped
    # are not invented -- see test_a_late_file_lands_in_its_own_week.
    assert out["would_create"] == ["week_4"]
    # Nothing was written: no copy, and no folder markers either.
    assert s3.copied == [] and s3.put == [] and s3.deleted == []


def test_the_preview_and_the_sweep_agree(monkeypatch, config):
    """The one invariant worth a test of its own.

    They share `project_week`; this is what stops them drifting apart if
    one of them grows a special case later.
    """
    files = {"drop/train/a.jpg": _at(3), "drop/train/late.jpg": _at(24)}
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "create_week",
                        lambda *, name, starts_at, make_folders=True: {
                            "name": name, "starts_at": pc._parse_time(starts_at).isoformat()})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    monkeypatch.setattr(pc, "s3", FakeS3(dict(files)))
    predicted = {p["key"]: p["destination"] for p in pc.plan_routing(config=config)["planned"]}

    swept = FakeS3(dict(files))
    monkeypatch.setattr(pc, "s3", swept)
    pc.route_pending(config=config)

    assert predicted == dict(swept.copied)


def test_the_next_week_is_numbered_not_parsed(config):
    """`week_pilot` must not stop the next one being `week_3`."""
    weeks = [{"name": "week_pilot", "starts_at": _at(1).isoformat()},
             {"name": "week_2", "starts_at": _at(8).isoformat()}]
    assert pc.next_week_name(config, weeks) == "week_3"
    assert pc.next_week_name({**config, "week_prefix": "batch-"}, weeks) == "batch-3"


def test_a_bare_date_means_midnight_utc():
    assert pc._parse_time("2026-09-01") == _dt.datetime(2026, 9, 1, tzinfo=UTC)
    assert pc._parse_time("2026-09-01T06:30:00Z") == _dt.datetime(2026, 9, 1, 6, 30, tzinfo=UTC)
    with pytest.raises(ValueError):
        pc._parse_time("   ")


def test_an_arrival_outside_the_drop_folders_does_nothing(monkeypatch, config):
    """The rule cannot filter by prefix, so the handler must.

    Every object created anywhere in the bucket reaches this function.
    Assembling a dataset copies thousands of files; without this check
    each one would be a full sweep of every drop folder.
    """
    s3 = FakeS3({"drop/train/a.jpg": _at(3)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "list_weeks", lambda: pytest.fail("listed S3 for an unrelated key"))

    out = pc.lambda_handler({
        "detail-type": "Object Created",
        "detail": {"object": {"key": "data/assembled/run-1/images/train/x.jpg"}},
    })
    assert out["moved"] == 0 and out["ignored"].endswith("x.jpg")
    assert s3.copied == []


def test_an_arrival_inside_a_drop_folder_sweeps(monkeypatch, config):
    s3 = FakeS3({"drop/train/a.jpg": _at(3)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.lambda_handler({
        "detail-type": "Object Created",
        "detail": {"object": {"key": "drop/train/a.jpg"}},
    })
    assert out["moved"] == 1
    assert s3.copied == [("drop/train/a.jpg", "retraining_training/week_1/images/a.jpg")]


def test_a_failed_delete_stops_the_sweep(monkeypatch, config):
    """One duplicate, not thousands.

    The copy runs before the delete so a crash never loses a file. The
    cost is that a delete which fails leaves the file in both places --
    and whatever stopped it (a missing s3:DeleteObject, in the case that
    prompted this) stops every remaining file identically. Carrying on
    duplicated ~3,000 files before anyone looked.
    """
    s3 = FakeS3({
        "drop/train/a.jpg": _at(3),
        "drop/train/b.jpg": _at(3),
        "drop/train/c.jpg": _at(3),
    })

    def refuse(**kwargs):
        raise ClientError({"Error": {"Code": "AccessDenied"}}, "DeleteObject")

    s3.delete_object = refuse
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=config)

    assert out["moved"] == 0
    # Exactly one file was copied before it gave up -- not all three.
    assert len(s3.copied) == 1
    assert out["errors"] == 1
    assert "both places" in out["error_files"][0]["error"]
    assert "Stopped after the first file" in out["aborted"]


def test_a_failed_copy_keeps_going(monkeypatch, config):
    """A copy that fails wrote nothing, so the next file is worth trying.

    The opposite of the delete case above, and the reason the two are
    handled separately rather than in one try block.
    """
    s3 = FakeS3({"drop/train/a.jpg": _at(3), "drop/train/b.jpg": _at(3)})
    real_copy = s3.copy_object
    calls = {"n": 0}

    def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ClientError({"Error": {"Code": "SlowDown"}}, "CopyObject")
        return real_copy(**kwargs)

    s3.copy_object = flaky
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=config)
    assert out["moved"] == 1 and out["errors"] == 1
    assert not out["aborted"]


def test_week_one_is_the_backlog(monkeypatch, config):
    """Set a start date and week 1 takes everything already there.

    The usual way to begin: point it at a folder holding months of files,
    say "week 1 starts now", and the backlog lands in week 1 while only
    what arrives afterwards rolls into week 2.
    """
    ancient = _dt.datetime(2026, 1, 15, tzinfo=UTC)
    # Bounded on both sides, an old file belongs nowhere...
    assert pc.week_for(ancient, WEEKS, 7, False) is None
    # ...and open-ended backwards it is week 1's.
    assert pc.week_for(ancient, WEEKS, 7, True)["name"] == "week_1"

    s3 = FakeS3({
        "drop/train/old.jpg": ancient,
        "drop/train/older.jpg": _dt.datetime(2025, 6, 1, tzinfo=UTC),
        "drop/train/recent.jpg": _at(9),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config={**config, "first_week_takes_all": True})

    assert out["moved"] == 3
    assert dict(s3.copied) == {
        "drop/train/old.jpg": "retraining_training/week_1/images/old.jpg",
        "drop/train/older.jpg": "retraining_training/week_1/images/older.jpg",
        # Still dated into its own week -- backfilling only changes what
        # happens BEFORE week 1, not the weeks after it.
        "drop/train/recent.jpg": "retraining_training/week_2/images/recent.jpg",
    }


def test_backfilling_off_still_refuses_older_files(monkeypatch, config):
    """The other reading has to keep working.

    A start date meant to EXCLUDE older data is a real setting, so turning
    this off must still leave those files where they are.
    """
    s3 = FakeS3({"drop/train/old.jpg": _dt.datetime(2026, 1, 15, tzinfo=UTC)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config={**config, "first_week_takes_all": False})
    assert out["moved"] == 0 and out["skipped"] == 1
    assert s3.copied == []


SPLIT_SOURCE = {
    "prefix": "pulls",
    "destination": "retraining_training",
    "eval_destination": "retraining_evaluation",
    "action": "copy",
}


def test_an_already_split_dataset_keeps_its_split(config):
    """val stays val.

    The pulls arrive as complete Ultralytics datasets. Merging train and
    val into one pile and re-splitting later puts near-duplicate frames
    from the same camera feed on both sides of the evaluation, and the
    mAP that comes out is inflated and says nothing about it.
    """
    base = "pulls/pull18/track2/escalation_dataset"
    # The source has to be IN the config: which folders a root needs
    # depends on every source filing into it, not just this one.
    split_config = {**config, "sources": [SPLIT_SOURCE]}

    # No train/ or val/ level -- each root receives exactly one split, so
    # the root already says which it is.
    cases = {
        f"{base}/images/train/a.jpg": ("retraining_training", "images"),
        f"{base}/labels/train/a.txt": ("retraining_training", "labels"),
        f"{base}/images/val/b.jpg": ("retraining_evaluation", "images"),
        f"{base}/labels/val/b.txt": ("retraining_evaluation", "labels"),
    }
    for key, want in cases.items():
        assert pc.placement(key, SPLIT_SOURCE, split_config) == want, key

    # A substring search would call this one training.
    assert pc.split_of("pulls/train2/images/val/x.jpg") == "val"
    # Unsplit folders are a training set until someone says otherwise.
    assert pc.split_of("pulls/flat/images/x.jpg") == ""
    assert pc.placement("pulls/flat/images/x.jpg", SPLIT_SOURCE, split_config)[0] \
        == "retraining_training"


def test_one_root_for_both_splits_keeps_them_apart():
    """The exception, and the reason the rule is not just "drop it".

    With no evaluation root, both splits land under the training root.
    Flattening there would tip val in with train -- the held-out set
    stops being held out, and the mAP that follows is measured partly on
    images the model trained on.
    """
    mixed = {
        **pc.DEFAULTS,
        "sources": [{"prefix": "drop", "destination": "one_root",
                     "eval_destination": "", "split": "", "action": "copy"}],
    }
    source = mixed["sources"][0]
    assert pc.splits_per_root(mixed) == {"one_root": {"train", "val"}}
    assert pc.placement("drop/images/train/a.jpg", source, mixed) == ("one_root", "images/train")
    assert pc.placement("drop/images/val/b.jpg", source, mixed) == ("one_root", "images/val")
    # A file on neither side stays flat, as it always did.
    assert pc.placement("drop/images/c.jpg", source, mixed) == ("one_root", "images")


PICKED_FOUR = {
    "train_images": ["ds/images/train"], "train_labels": ["ds/labels/train"],
    "val_images": ["ds/images/val"], "val_labels": ["ds/labels/val"],
}


def _placed(config):
    return {
        src["prefix"]: "/".join(
            pc.placement(f"{src['prefix']}/f.jpg" if "images" in src["prefix"]
                         else f"{src['prefix']}/f.txt", src, config))
        for src in config["sources"]
    }


def test_the_card_s_own_config_gives_one_root_with_the_split_inside():
    """What the four pickers produce, end to end.

    One root holds everything, and the split stays in the path BECAUSE
    that root receives both -- `splits_per_root` sees two splits and
    `placement` keeps the level. Flattening here would file the
    evaluation images in with the training ones.
    """
    config = {**pc.DEFAULTS, "root": "retraining", "picked": PICKED_FOUR}
    config["sources"] = pc.sources_from_picked(config)

    assert pc.splits_per_root(config) == {"retraining": {"train", "val"}}
    assert _placed(config) == {
        "ds/images/train": "retraining/images/train",
        "ds/labels/train": "retraining/labels/train",
        "ds/images/val": "retraining/images/val",
        "ds/labels/val": "retraining/labels/val",
    }


def test_blanking_the_root_falls_back_to_the_two_older_roots():
    """The escape hatch, and the shape a pre-merge settings file had.

    With no single root, each split points at its own root, and the root
    alone says which is which -- so no split subfolder is needed.
    """
    config = {
        **pc.DEFAULTS,
        "root": "",
        "training_root": "retraining_training",
        "eval_root": "retraining_evaluation",
        "picked": PICKED_FOUR,
    }
    config["sources"] = pc.sources_from_picked(config)

    assert pc.splits_per_root(config) == {
        "retraining_training": {"train"},
        "retraining_evaluation": {"val"},
    }
    assert _placed(config) == {
        "ds/images/train": "retraining_training/images",
        "ds/labels/train": "retraining_training/labels",
        "ds/images/val": "retraining_evaluation/images",
        "ds/labels/val": "retraining_evaluation/labels",
    }


def test_defining_a_week_creates_no_folders(monkeypatch):
    """An unused folder is not created.

    Empty marker keys used to be written under both roots, for both
    splits and both of images/ and labels/ -- eight keys per week, half
    of them for splits that week would never receive. A folder now means
    something is in it.
    """
    s3 = FakeS3({})
    monkeypatch.setattr(pc, "s3", s3)
    # The week ROW still gets written; only the S3 folders are gone.
    monkeypatch.setattr(pc, "ddb", type("Ddb", (), {"put_item": lambda self, **kw: {}})())

    pc.create_week(name="week_4", starts_at=_at(22, 0).isoformat(), make_folders=True)
    assert s3.put == []
    assert s3.copied == [] and s3.deleted == []


def test_copy_leaves_the_original_where_it_is(monkeypatch, config):
    """The pulls are the system of record; a move would empty them."""
    s3 = FakeS3({
        "pulls/p1/images/train/a.jpg": _at(3),
        "pulls/p1/images/val/b.jpg": _at(3),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config={**config, "sources": [SPLIT_SOURCE]})

    assert out["moved"] == 2
    assert dict(s3.copied) == {
        "pulls/p1/images/train/a.jpg": "retraining_training/week_1/images/a.jpg",
        "pulls/p1/images/val/b.jpg": "retraining_evaluation/week_1/images/b.jpg",
    }
    # Nothing removed -- that is the whole difference from a move.
    assert s3.deleted == []


def test_copying_twice_changes_nothing(monkeypatch, config):
    """Content-hash filenames make a re-run free.

    The same frame pulled twice has the same name, so the second run finds
    it already there and skips it rather than duplicating it.
    """
    s3 = FakeS3({"pulls/p1/images/train/a.jpg": _at(3)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    first = pc.route_pending(config={**config, "sources": [SPLIT_SOURCE]})
    second = pc.route_pending(config={**config, "sources": [SPLIT_SOURCE]})

    assert first["moved"] == 1
    assert second["moved"] == 0 and second["skipped"] == 1
    assert len(s3.copied) == 1


def test_move_is_still_available_for_a_real_staging_folder(monkeypatch, config):
    s3 = FakeS3({"drop/train/a.jpg": _at(3)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    staging = {"prefix": "drop/train", "destination": "retraining_training", "action": "move"}
    out = pc.route_pending(config={**config, "sources": [staging]})
    assert out["moved"] == 1
    assert s3.deleted == ["drop/train/a.jpg"]


def test_copy_is_the_default(config):
    """The default must be the one that cannot empty a dataset."""
    assert pc._action_of({"prefix": "x", "destination": "y"}) == "copy"
    assert pc._action_of({"action": "move"}) == "move"


def test_a_cumulative_source_does_not_refile_its_history(monkeypatch, config):
    """The case that makes per-week dedup wrong.

    track2_pull publishes cumulative snapshots: pull1 held 153 images,
    pull18 held 2,954, and every pull contains everything before it. A
    check scoped to the week being written sees `week_2/images/a.jpg`
    missing, copies it, and repeats into week_3 -- every week becomes a
    full copy of the dataset and storage grows with the square of the
    weeks. Checked across the whole root, a week holds only what was new.
    """
    source = {"prefix": "pulls", "destination": "retraining_training", "action": "copy"}
    s3 = FakeS3({
        # Two snapshots. The second repeats the first and adds one.
        "pulls/snap1/images/train/a.jpg": _at(3),
        "pulls/snap2/images/train/a.jpg": _at(9),
        "pulls/snap2/images/train/b.jpg": _at(9),
    })
    # A repeated file in a later snapshot is the SAME file, byte for byte
    # -- which is what makes it recognisable as already filed rather than
    # as a revision.
    s3.etags["pulls/snap2/images/train/a.jpg"] = s3.etags["pulls/snap1/images/train/a.jpg"]
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config={**config, "sources": [source]})

    # `a.jpg` is filed once, under the week it first appeared in. Its
    # repeat in the later snapshot is recognised, not filed again.
    assert out["moved"] == 2
    # This source declares no split and has no evaluation root, so the
    # one root may receive both -- and the train/ level stays, for the
    # reason in test_one_root_for_both_splits_keeps_them_apart. The four
    # pickers in the card never produce this shape.
    assert dict(s3.copied) == {
        "pulls/snap1/images/train/a.jpg": "retraining_training/week_1/images/train/a.jpg",
        "pulls/snap2/images/train/b.jpg": "retraining_training/week_2/images/train/b.jpg",
    }
    assert out["skipped"] == 1


def test_the_two_roots_dedupe_separately(monkeypatch, config):
    """A name on the val side is not "already filed" because train has it.

    The roots are different datasets. Sharing one set of seen names would
    let a training file suppress its evaluation counterpart.
    """
    s3 = FakeS3({
        "pulls/p/images/train/same.jpg": _at(3),
        "pulls/p/images/val/same.jpg": _at(3),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: list(WEEKS))
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config={**config, "sources": [SPLIT_SOURCE | {"prefix": "pulls"}]})
    assert out["moved"] == 2
    assert set(dict(s3.copied).values()) == {
        "retraining_training/week_1/images/same.jpg",
        "retraining_evaluation/week_1/images/same.jpg",
    }



# ---------------------------------------------------------------------------
# The scan-batch model, for a source edited in place
# ---------------------------------------------------------------------------

DATASET = {
    "bucket": "safety-model-dataset-dev",
    "prefix": "track2/escalation_dataset",
    "destination": "retraining_training",
    "eval_destination": "retraining_evaluation",
    "action": "copy",
}


@pytest.fixture
def scanned():
    """A live dataset folder: no pull events, so the scan is the batch."""
    return {**pc.DEFAULTS, "week_by": "scan", "sources": [DATASET]}


def test_the_first_scan_takes_everything_as_week_one(monkeypatch, scanned):
    """No pull to date a batch by, so the scan is the batch.

    track2/escalation_dataset is rewritten in place -- the snapshot
    folders that carried a date are gone. Everything present on the first
    scan is week 1, whatever each file's own timestamp says.
    """
    s3 = FakeS3({
        "track2/escalation_dataset/images/train/a.jpg": _at(1),
        # Months apart. Under the per-file model these land in different
        # weeks; as one batch they do not.
        "track2/escalation_dataset/images/train/b.jpg": _at(28),
        "track2/escalation_dataset/images/val/c.jpg": _at(14),
    })
    made = []
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    monkeypatch.setattr(pc, "create_week", lambda *, name, starts_at, make_folders=True: (
        made.append(name) or {"name": name, "starts_at": pc._parse_time(starts_at).isoformat()}))

    out = pc.route_pending(config=scanned)

    assert made == ["week_1"]
    assert out["moved"] == 3
    assert set(dict(s3.copied).values()) == {
        "retraining_training/week_1/images/a.jpg",
        "retraining_training/week_1/images/b.jpg",
        "retraining_evaluation/week_1/images/c.jpg",
    }



def test_a_one_day_week_is_allowed(scanned):
    """1 day is the setting for exercising this without waiting a week."""
    anchor = _dt.datetime(2026, 9, 1, tzinfo=UTC)
    weeks = [{"name": "week_1", "starts_at": anchor.isoformat()}]
    week, _ = pc.project_scan_week({**scanned, "week_length_days": 1} and
                                   {**scanned, "week_length_days": 1},
                                   weeks) if False else (None, None)
    week, _ = pc.project_scan_week(weeks, {**scanned, "week_length_days": 1},
                                   anchor + _dt.timedelta(days=3))
    assert week["starts_at"] == (anchor + _dt.timedelta(days=3)).isoformat()


def test_preview_and_sweep_pick_the_same_scan_week(monkeypatch, scanned):
    """The invariant, restated for the batch model."""
    files = {"track2/escalation_dataset/images/train/a.jpg": _at(3)}
    weeks = [{"name": "week_1", "starts_at": _at(1, 0).isoformat()}]
    monkeypatch.setattr(pc, "list_weeks", lambda: list(weeks))
    monkeypatch.setattr(pc, "create_week",
                        lambda *, name, starts_at, make_folders=True: {
                            "name": name, "starts_at": pc._parse_time(starts_at).isoformat()})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    monkeypatch.setattr(pc, "s3", FakeS3(dict(files)))
    predicted = {p["key"]: p["destination"] for p in pc.plan_routing(config=scanned)["planned"]}

    swept = FakeS3(dict(files))
    monkeypatch.setattr(pc, "s3", swept)
    pc.route_pending(config=scanned)
    assert predicted == dict(swept.copied)


def test_a_source_reads_from_its_own_bucket(monkeypatch, scanned):
    """The datasets live in a different bucket, read-only.

    The copy has to name that bucket as the SOURCE and ours as the
    destination; reading from our own would find nothing and writing to
    theirs is exactly what the IAM grant refuses.
    """
    seen = {}
    s3 = FakeS3({"track2/escalation_dataset/images/train/a.jpg": _at(3)})
    real_copy = s3.copy_object

    def watch(**kw):
        seen["from"] = kw["CopySource"]["Bucket"]
        seen["to"] = kw["Bucket"]
        return real_copy(**kw)

    s3.copy_object = watch
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [{"name": "week_1",
                                                    "starts_at": _at(1, 0).isoformat()}])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    pc.route_pending(config=scanned)
    assert seen["from"] == "safety-model-dataset-dev"
    assert seen["to"] == pc.BUCKET
    assert pc.source_bucket({}) == pc.BUCKET


def test_a_revised_label_is_rewritten_where_it_already_sits(monkeypatch, scanned):
    """The source is edited in place, so a name can come back changed.

    Measured at about 1 file in 700 over five days of the live dataset.
    Filing the corrected version into the CURRENT week would leave the
    stale copy in its old week, and the dataset would hold the same image
    twice with contradicting labels.
    """
    s3 = FakeS3({"track2/escalation_dataset/labels/train/a.txt": _at(3)})
    # Already filed, under an older week, with different bytes.
    s3.objects["retraining_training/week_1/labels/a.txt"] = _at(1)
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [
        {"name": "week_1", "starts_at": _at(1, 0).isoformat()},
        {"name": "week_9", "starts_at": _at(20, 0).isoformat()},
    ])
    monkeypatch.setattr(pc, "create_week",
                        lambda *, name, starts_at, make_folders=True: {
                            "name": name, "starts_at": pc._parse_time(starts_at).isoformat()})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=scanned)

    assert out["revised"] == 1
    # Rewritten in week_1, where it already lives -- not added to the
    # week this scan happens to be filing into.
    assert dict(s3.copied) == {
        "track2/escalation_dataset/labels/train/a.txt":
            "retraining_training/week_1/labels/a.txt",
    }


def test_an_unchanged_file_is_skipped_not_recopied(monkeypatch, scanned):
    """Same name AND same bytes is the overwhelmingly common case."""
    s3 = FakeS3({"track2/escalation_dataset/images/train/a.jpg": _at(3)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [{"name": "week_1",
                                                    "starts_at": _at(1, 0).isoformat()}])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    first = pc.route_pending(config=scanned)
    second = pc.route_pending(config=scanned)
    assert first["moved"] == 1
    assert second["moved"] == 0 and second["revised"] == 0 and second["skipped"] == 1



def test_a_hand_picked_folder_needs_no_data_yaml(monkeypatch, scanned):
    """Four raw folders, no manifest anywhere.

    Images and labels are told apart by extension, and which side a
    folder belongs to is the source it was picked in -- so a dataset that
    has never had a data.yaml can still be filed into weeks.
    """
    s3 = FakeS3({
        "raw/train_imgs/a.jpg": _at(3),
        "raw/train_lbls/a.txt": _at(3),
        "raw/val_imgs/b.jpg": _at(3),
        "raw/val_lbls/b.txt": _at(3),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [{"name": "week_1",
                                                    "starts_at": _at(1, 0).isoformat()}])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    # Pinned inside week 1, so this asserts WHERE files land rather than
    # which week the calendar happens to be on today.
    monkeypatch.setattr(pc, "_now", lambda: _at(3))

    picked = [
        {"prefix": "raw/train_imgs", "destination": "retraining_training",
         "split": "train", "action": "copy"},
        {"prefix": "raw/train_lbls", "destination": "retraining_training",
         "split": "train", "action": "copy"},
        {"prefix": "raw/val_imgs", "destination": "retraining_evaluation",
         "split": "val", "action": "copy"},
        {"prefix": "raw/val_lbls", "destination": "retraining_evaluation",
         "split": "val", "action": "copy"},
    ]
    out = pc.route_pending(config={**scanned, "sources": picked})

    assert out["moved"] == 4
    # The shape a trainer reads directly, built from raw folders.
    assert dict(s3.copied) == {
        "raw/train_imgs/a.jpg": "retraining_training/week_1/images/a.jpg",
        "raw/train_lbls/a.txt": "retraining_training/week_1/labels/a.txt",
        "raw/val_imgs/b.jpg": "retraining_evaluation/week_1/images/b.jpg",
        "raw/val_lbls/b.txt": "retraining_evaluation/week_1/labels/b.txt",
    }


def test_a_hand_picked_folder_is_not_re_split_by_its_path(monkeypatch, scanned):
    """A folder literally named `val` still goes where it was picked.

    Path-based split detection applies only when an evaluation root is
    configured on the source. A folder chosen by hand for TRAINING must
    go to training even if `val` appears in its path, or picking it would
    not mean what it says.
    """
    s3 = FakeS3({"raw/images/val/a.jpg": _at(3)})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [{"name": "week_1",
                                                    "starts_at": _at(1, 0).isoformat()}])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    monkeypatch.setattr(pc, "_now", lambda: _at(3))

    out = pc.route_pending(config={**scanned, "sources": [
        {"prefix": "raw/images/val", "destination": "retraining_training",
         "split": "train", "action": "copy"},
    ]})
    assert out["moved"] == 1
    # `val` in the path, `train` in the picker -- the picker wins.
    assert dict(s3.copied) == {
        "raw/images/val/a.jpg": "retraining_training/week_1/images/a.jpg",
    }


# ---------------------------------------------------------------------------
# Do the chosen images and labels describe each other?
# ---------------------------------------------------------------------------

def test_pairing_reports_a_clean_match(monkeypatch):
    s3 = FakeS3({
        "imgs/a.jpg": _at(1), "imgs/b.jpg": _at(1),
        "lbls/a.txt": _at(1), "lbls/b.txt": _at(1),
    })
    monkeypatch.setattr(pc, "s3", s3)
    out = pc.check_pairing(image_prefixes=["imgs"], label_prefixes=["lbls"])
    assert out["paired"] == 2
    assert out["images_without_labels"] == 0
    assert out["labels_without_images"] == 0


def test_an_orphan_label_is_reported_separately_from_a_background_image(monkeypatch):
    """The two mismatches are not equally serious.

    A label with no image describes nothing and is never read, so whatever
    it annotated is silently missing from the run -- always a mistake. An
    image with no label is a background example, which Ultralytics accepts
    deliberately and which improves precision. Reporting them as one
    number would either cry wolf about backgrounds or hide orphan labels.
    """
    s3 = FakeS3({
        "imgs/a.jpg": _at(1),
        "imgs/background.jpg": _at(1),   # legitimately unlabelled
        "lbls/a.txt": _at(1),
        "lbls/ghost.txt": _at(1),        # describes nothing
    })
    monkeypatch.setattr(pc, "s3", s3)
    out = pc.check_pairing(image_prefixes=["imgs"], label_prefixes=["lbls"])
    assert out["paired"] == 1
    assert out["images_without_labels"] == 1
    assert out["sample_images_without_labels"] == ["background"]
    assert out["labels_without_images"] == 1
    assert out["sample_labels_without_images"] == ["ghost"]


def test_pairing_matches_on_the_stem_across_folders(monkeypatch):
    """Images and labels are picked independently, so they rarely share a
    folder -- the stem is the only thing that links them."""
    s3 = FakeS3({
        "a/one/x.jpg": _at(1), "a/two/y.png": _at(1),
        "b/three/x.txt": _at(1), "b/four/y.txt": _at(1),
    })
    monkeypatch.setattr(pc, "s3", s3)
    out = pc.check_pairing(image_prefixes=["a/one", "a/two"],
                           label_prefixes=["b/three", "b/four"])
    assert out["paired"] == 2 and out["images"] == 2 and out["labels"] == 2


def test_completely_mismatched_folders_pair_nothing(monkeypatch):
    """The case the warning exists for: the wrong label folder entirely."""
    s3 = FakeS3({"imgs/a.jpg": _at(1), "lbls/completely_different.txt": _at(1)})
    monkeypatch.setattr(pc, "s3", s3)
    out = pc.check_pairing(image_prefixes=["imgs"], label_prefixes=["lbls"])
    assert out["paired"] == 0
    assert out["images_without_labels"] == 1 and out["labels_without_images"] == 1


def test_nothing_picked_is_not_an_error(monkeypatch):
    monkeypatch.setattr(pc, "s3", FakeS3({}))
    out = pc.check_pairing(image_prefixes=[], label_prefixes=[])
    assert out["checked"] is False and out["paired"] == 0


# ---------------------------------------------------------------------------
# Saving and loading a settings file
# ---------------------------------------------------------------------------

@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    return tmp_path


def test_a_saved_setting_restores_the_folder_lists(settings_dir, monkeypatch):
    """The round trip that matters: pick folders, save, load, get them back.

    `sources` alone cannot do this. A source records a prefix and a split
    but not whether the folder holds images or labels -- that is decided
    per file by extension -- so the four lists have to be stored as well.
    """
    picked = {
        "train_images": ["ds/images/train"], "train_labels": ["ds/labels/train"],
        "val_images": ["ds/images/val"], "val_labels": ["ds/labels/val"],
    }
    live = {**pc.DEFAULTS, "picked": picked, "source_bucket": "other-bucket",
            "training_root": "T", "eval_root": "E", "week_length_days": 5,
            "week_prefix": "batch_", "mode": "off"}

    monkeypatch.setattr(pc, "read_config", lambda: live)
    pc.save_settings_file(name="mine", config=live)

    written = {}
    monkeypatch.setattr(pc, "write_config", lambda c: written.update(c) or {**live, **written})
    monkeypatch.setattr(pc, "set_mode", lambda **kw: {"problems": []})

    out = pc.load_settings_file(name="mine")
    assert out["config"]["picked"] == picked
    assert out["config"]["source_bucket"] == "other-bucket"
    assert out["config"]["week_prefix"] == "batch_"
    # Four folders in, four sources out.
    assert out["sources"] == 4


def test_loading_rebuilds_sources_from_the_restored_picks(settings_dir, monkeypatch):
    """Restoring the lists without rebuilding sources is the worst failure.

    The card would show the loaded folders while the sweep kept using the
    ones from before -- and both halves look correct on their own.
    """
    live = {**pc.DEFAULTS, "training_root": "T", "eval_root": "E",
            "picked": {"train_images": ["new/imgs"], "train_labels": [],
                       "val_images": [], "val_labels": []},
            "sources": [{"prefix": "STALE", "destination": "T"}], "mode": "off"}
    monkeypatch.setattr(pc, "read_config", lambda: live)
    pc.save_settings_file(name="x", config=live)

    writes = []
    monkeypatch.setattr(pc, "write_config",
                        lambda c: writes.append(c) or {**live, **c})
    monkeypatch.setattr(pc, "set_mode", lambda **kw: {"problems": []})
    pc.load_settings_file(name="x")

    rebuilt = [w for w in writes if "sources" in w]
    assert rebuilt, "sources were never rebuilt from the restored picks"
    assert [s["prefix"] for s in rebuilt[-1]["sources"]] == ["new/imgs"]
    assert "STALE" not in str(rebuilt[-1]["sources"])


def test_loading_reapplies_the_trigger(settings_dir, monkeypatch):
    """A file saved with the trigger off must not load into an armed rule.

    Writing the config alone would leave the card reading "off" while the
    EventBridge rule kept firing.
    """
    live = {**pc.DEFAULTS, "mode": "schedule", "schedule_expression": "rate(2 hours)"}
    monkeypatch.setattr(pc, "read_config", lambda: live)
    pc.save_settings_file(name="sched", config=live)

    applied = {}
    monkeypatch.setattr(pc, "write_config", lambda c: {**live, **c})
    monkeypatch.setattr(pc, "set_mode",
                        lambda **kw: applied.update(kw) or {"problems": []})

    out = pc.load_settings_file(name="sched")
    assert applied["mode"] == "schedule"
    assert applied["schedule_expression"] == "rate(2 hours)"
    assert out["mode"] == "schedule"


def test_a_trigger_that_cannot_be_applied_is_reported_not_raised(settings_dir, monkeypatch):
    """The settings are live either way; only the rules are not."""
    live = {**pc.DEFAULTS, "mode": "arrival"}
    monkeypatch.setattr(pc, "read_config", lambda: live)
    pc.save_settings_file(name="y", config=live)

    monkeypatch.setattr(pc, "write_config", lambda c: {**live, **c})

    def boom(**kw):
        raise RuntimeError("EventBridge unreachable")

    monkeypatch.setattr(pc, "set_mode", boom)
    out = pc.load_settings_file(name="y")
    assert out["applied"] is True
    assert any("trigger was not applied" in p for p in out["problems"])





def test_the_destination_bucket_defaults_to_this_app(monkeypatch):
    assert pc.destination_bucket({}) == pc.BUCKET
    assert pc.destination_bucket({"destination_bucket": "  "}) == pc.BUCKET
    assert pc.destination_bucket({"destination_bucket": "other"}) == "other"


# ---------------------------------------------------------------------------
# The router must not silently run different routing code
# ---------------------------------------------------------------------------

def test_matching_hashes_report_current(monkeypatch):
    monkeypatch.setattr(pc, "source_sha", lambda: "a" * 64)

    class Lam:
        def get_function_configuration(self, **kw):
            return {"Environment": {"Variables": {"ROUTER_SOURCE_SHA": "a" * 64}},
                    "LastModified": "2026-09-24T11:00:28.000+0000"}

    monkeypatch.setattr(pc.boto3, "client", lambda *a, **k: Lam())
    assert pc.router_freshness()["state"] == "current"


def test_a_different_hash_reports_stale_and_says_what_to_run(monkeypatch):
    """The failure this exists for.

    The container deploys and the Lambda does not. Each half is
    internally consistent, so nothing errors -- files just start landing
    in a shape the card does not describe. It happened three times in one
    day before the check existed.
    """
    monkeypatch.setattr(pc, "source_sha", lambda: "a" * 64)

    class Lam:
        def get_function_configuration(self, **kw):
            return {"Environment": {"Variables": {"ROUTER_SOURCE_SHA": "b" * 64}},
                    "LastModified": "2026-09-23T10:41:49.000+0000"}

    monkeypatch.setattr(pc.boto3, "client", lambda *a, **k: Lam())
    out = pc.router_freshness()
    assert out["state"] == "stale"
    assert out["app_sha"] != out["router_sha"]
    # The message has to carry the fix, not just the diagnosis.
    assert "terraform apply" in out["detail"]


def test_a_router_that_cannot_be_read_is_unknown_not_stale(monkeypatch):
    """A missing permission must not be reported as a code mismatch.

    Claiming the router is stale when it might be fine would send someone
    redeploying to fix a problem they do not have.
    """
    monkeypatch.setattr(pc, "source_sha", lambda: "a" * 64)

    class Lam:
        def get_function_configuration(self, **kw):
            raise RuntimeError("AccessDenied")

    monkeypatch.setattr(pc.boto3, "client", lambda *a, **k: Lam())
    out = pc.router_freshness()
    assert out["state"] == "unknown"
    assert "AccessDenied" in out["detail"]


def test_a_router_with_no_hash_is_unknown(monkeypatch):
    """Deployed before the hash existed -- which is itself a hint."""
    monkeypatch.setattr(pc, "source_sha", lambda: "a" * 64)

    class Lam:
        def get_function_configuration(self, **kw):
            return {"Environment": {"Variables": {}}, "LastModified": "x"}

    monkeypatch.setattr(pc.boto3, "client", lambda *a, **k: Lam())
    assert pc.router_freshness()["state"] == "unknown"


def test_the_hash_is_of_this_file(monkeypatch):
    """It must hash the routing source, not something incidental.

    Terraform runs filesha256 over the same path, so the two only agree
    if this reads exactly that file.
    """
    import hashlib
    import pathlib

    want = hashlib.sha256(pathlib.Path(pc.__file__).read_bytes()).hexdigest()
    assert pc.source_sha() == want


# ---------------------------------------------------------------------------
# The cut-off: week 1 is the backlog, week 2 starts the clock
# ---------------------------------------------------------------------------

CUTOFF = _dt.datetime(2026, 9, 25, tzinfo=UTC)
CUT_CONFIG = {**pc.DEFAULTS, "week_cutoff": CUTOFF.isoformat(), "week_length_days": 7}


def test_week_one_is_everything_before_the_cutoff():
    """No start at all. A cut-off says "everything so far is the baseline",
    which only means something if nothing falls off the back of it."""
    for when in (CUTOFF - _dt.timedelta(days=365),
                 CUTOFF - _dt.timedelta(days=1),
                 CUTOFF - _dt.timedelta(minutes=1)):
        got = pc.week_at(when, CUT_CONFIG)
        assert got["name"] == "week_1"
        assert got["open_start"] is True
        assert got["ends_at"] == CUTOFF.isoformat()


def test_the_clock_starts_at_the_cutoff():
    """The cut-off belongs to week 2, not week 1 -- it is the boundary."""
    assert pc.week_at(CUTOFF, CUT_CONFIG)["name"] == "week_2"
    assert pc.week_at(CUTOFF + _dt.timedelta(days=6, hours=23), CUT_CONFIG)["name"] == "week_2"
    assert pc.week_at(CUTOFF + _dt.timedelta(days=7), CUT_CONFIG)["name"] == "week_3"


def test_week_numbers_come_from_elapsed_time_not_from_a_count():
    """`week_5` must mean the fifth week, whatever was collected before.

    Numbering from how many weeks happen to exist produced a folder named
    week_3 that started on the fifth interval, because two intervals had
    no data -- and "after N weeks it lands in week_N+1" stopped being true.
    """
    for weeks_on, expected in ((0, "week_2"), (1, "week_3"), (4, "week_6"), (51, "week_53")):
        when = CUTOFF + _dt.timedelta(days=7 * weeks_on)
        assert pc.week_at(when, CUT_CONFIG)["name"] == expected


def test_bounds_do_not_overlap_or_leave_gaps():
    """Every moment belongs to exactly one week."""
    previous_end = None
    for weeks_on in range(6):
        got = pc.week_at(CUTOFF + _dt.timedelta(days=7 * weeks_on), CUT_CONFIG)
        if previous_end is not None:
            assert got["starts_at"] == previous_end
        previous_end = got["ends_at"]


def test_no_cutoff_means_there_is_only_week_one():
    """Nothing has said when the baseline stops, so nothing is after it."""
    config = {**pc.DEFAULTS, "week_cutoff": ""}
    assert pc.week_at(_at(1), config)["name"] == "week_1"
    assert pc.week_at(_at(28), config)["name"] == "week_1"


def test_an_unparseable_cutoff_falls_back_to_week_one():
    """A malformed date must not stop a sweep that would otherwise work."""
    config = {**pc.DEFAULTS, "week_cutoff": "next tuesday"}
    assert pc.week_at(_at(12), config)["name"] == "week_1"


def test_the_week_prefix_is_honoured():
    config = {**CUT_CONFIG, "week_prefix": "batch-"}
    assert pc.week_at(CUTOFF, config)["name"] == "batch-2"


def test_the_forecast_shows_filled_current_and_future(monkeypatch):
    """One table: what exists with real counts, what is being filled, and
    what is coming. A week that is coming has no counts because it does
    not exist -- saying 0 images would read as "empty" rather than "not
    yet"."""
    monkeypatch.setattr(pc, "week_contents", lambda config=None: {
        "weeks": [{"week": "week_1", "images": 730, "labels": 730,
                   "bytes": 1024, "roots": ["r"], "defined": True}],
        "roots": ["r"],
    })
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=3))

    out = pc.week_forecast(config=CUT_CONFIG, ahead=6)
    rows = {r["week"]: r for r in out["weeks"]}

    assert out["current"] == "week_2"
    assert rows["week_1"]["state"] == "filled" and rows["week_1"]["images"] == 730
    assert rows["week_2"]["state"] == "current" and rows["week_2"]["exists"] is False
    # Six ahead of the current one, so up to week_8.
    assert rows["week_8"]["state"] == "future"
    assert "week_9" not in rows
    assert rows["week_5"]["starts_at"] == (CUTOFF + _dt.timedelta(days=21)).isoformat()

# The four tests that lived here described `week_anchor` -- a date week 1
# STARTED, with later weeks numbered by how many already existed. The
# cut-off replaced both halves of that: week 1 is the open-ended backlog
# and week numbers come from elapsed intervals. See the cut-off tests.


# ---------------------------------------------------------------------------
# The pull schedule
#
# A pull is "come and look for new files". A week cut-off is "which folder
# does a file belong to". They are deliberately unrelated, and the tests
# that matter are the ones where they disagree -- pulling fortnightly into
# weekly folders is the case this was all built for.
# ---------------------------------------------------------------------------

def _sched(**over):
    return {
        **pc.DEFAULTS,
        # The cut-off is the anchor: the pull series is counted from it
        # and inherits its time of day, so there is no second date.
        "week_cutoff": CUTOFF.isoformat(),
        "week_length_days": 7,
        "schedule_every_days": 14,
        **over,
    }


def test_the_first_pull_is_one_interval_after_the_cut_off():
    """The first pull is the next slot from NOW, not an interval after
    the cut-off.

    CHANGED DELIBERATELY. It used to be `cut-off + interval`, rolled
    forward by whole intervals -- a grid fixed by a date chosen to define
    week boundaries. Setting "every 7 days" on 5 October against a March
    cut-off armed the 7th, because the grid did not happen to include
    today, and the page gave no hint why. The cut-off still decides which
    WEEK a file lands in; it no longer decides when collecting happens.
    """
    config = _sched(schedule_every_days=7)
    now = CUTOFF - _dt.timedelta(hours=1)
    runs = [pc._parse_time(t) for t in pc.next_pull_times(config, count=3, now=now)]

    # Within a day of NOW, not a week after the cut-off.
    assert runs[0] - now < _dt.timedelta(days=1)
    assert runs[0] > now
    # ...and the interval still holds from there.
    assert (runs[1] - runs[0]).days == 7
    assert (runs[2] - runs[1]).days == 7


def test_a_cut_off_in_the_past_rolls_forward_keeping_its_time_of_day():
    """09:00 every fortnight stays at 09:00.

    The TIME is still preserved exactly -- that part of the old design is
    worth keeping and is what this now pins. What changed is the DATE: the
    series starts at the next 09:00 from now rather than on a grid counted
    off the cut-off. Whole-day intervals and no daylight saving in
    Singapore mean the wall clock cannot drift once it starts.
    """
    # 09:00 Singapore time.
    cutoff = _dt.datetime(2026, 9, 25, 1, 0, tzinfo=UTC)
    config = _sched(week_cutoff=cutoff.isoformat(), schedule_every_days=14)
    now = _dt.datetime(2026, 11, 3, 6, 37, tzinfo=UTC)
    runs = [pc._parse_time(t) for t in pc.next_pull_times(config, count=3, now=now)]

    assert all(t > now for t in runs)
    # The TIME is what must survive, and does: 09:00 SGT is 01:00 UTC.
    assert all(t.hour == 1 and t.minute == 0 for t in runs)
    assert (runs[1] - runs[0]).days == 14
    # Restarted from today, NOT kept on the cut-off's rhythm -- the
    # reversal this test now pins. The first run is the next 09:00.
    assert runs[0] - now < _dt.timedelta(days=1)


def test_no_cut_off_arms_nothing():
    """Without a cut-off there is nothing to count from.

    A timer that cannot say when it first fires is worse than no timer:
    it sits ENABLED and silent.
    """
    assert pc.next_pull_times(_sched(week_cutoff=""), count=3) == []
    assert pc.schedule_plan(_sched(week_cutoff=""))["ready"] is False


def test_arming_a_schedule_without_a_cut_off_is_refused(monkeypatch):
    """And it is refused at the point of arming, not silently accepted."""
    monkeypatch.setattr(pc, "read_config", lambda: _sched(week_cutoff=""))
    with pytest.raises(ValueError, match="cut-off"):
        pc.set_mode(mode="schedule")


def test_the_cut_off_time_of_day_is_the_pull_time_of_day():
    """One clock. Change the cut-off's time and the pulls move with it."""
    morning = _sched(week_cutoff=_dt.datetime(2026, 9, 25, 1, 0, tzinfo=UTC).isoformat(),
                     schedule_every_days=7)
    evening = _sched(week_cutoff=_dt.datetime(2026, 9, 25, 15, 30, tzinfo=UTC).isoformat(),
                     schedule_every_days=7)
    now = _dt.datetime(2026, 9, 26, tzinfo=UTC)

    first_morning = pc._parse_time(pc.next_pull_times(morning, count=1, now=now)[0])
    first_evening = pc._parse_time(pc.next_pull_times(evening, count=1, now=now)[0])
    assert (first_morning.hour, first_morning.minute) == (1, 0)
    assert (first_evening.hour, first_evening.minute) == (15, 30)


def test_a_fortnightly_pull_fills_more_than_one_week():
    """The case the whole per-file model exists for.

    Pulling every 14 days into 7-day weeks means one pull carries files
    belonging to several different weeks.
    """
    now = _dt.datetime(2026, 10, 20, tzinfo=UTC)
    plan = pc.schedule_plan(_sched(), now=now)

    assert plan["every_days"] == 14
    assert plan["week_length_days"] == 7
    assert plan["spans_multiple_weeks"] is True
    assert plan["weeks_per_pull"] > 1
    # Dated individually -- by capture time now -- so the several weeks
    # are filled correctly. Only `scan` misfiles across a long interval.
    assert plan["week_by"] == "capture_time"
    assert plan["misfiling_risk"] is False


def test_scan_mode_plus_a_long_interval_is_flagged():
    """The combination that silently puts a fortnight in one folder."""
    plan = pc.schedule_plan(_sched(week_by="scan"),
                            now=_dt.datetime(2026, 10, 20, tzinfo=UTC))
    assert plan["spans_multiple_weeks"] is True
    assert plan["misfiling_risk"] is True


def test_a_fortnightly_pull_files_each_file_into_its_own_week(monkeypatch):
    """End to end: one sweep, three weeks, each file where it belongs.

    This is the behaviour the card promises when it says a pull may fill
    several week folders at once.
    """
    config = _sched(sources=[{
        "prefix": "drop/train", "destination": "retraining_training", "action": "copy",
    }])
    # Three files spread across a fortnight either side of the cut-off.
    s3 = FakeS3({
        "drop/train/backlog.jpg": _dt.datetime(2026, 9, 1, tzinfo=UTC),
        "drop/train/first.jpg": _dt.datetime(2026, 9, 27, tzinfo=UTC),
        "drop/train/second.jpg": _dt.datetime(2026, 10, 5, tzinfo=UTC),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "create_week", lambda **kw: {
        "name": kw["name"], "starts_at": kw["starts_at"]})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    out = pc.route_pending(config=config)

    assert out["moved"] == 3
    assert dict(s3.copied) == {
        # Before the cut-off: the backlog.
        "drop/train/backlog.jpg": "retraining_training/week_1/images/backlog.jpg",
        # 25 Sep + 2 days.
        "drop/train/first.jpg": "retraining_training/week_2/images/first.jpg",
        # 25 Sep + 10 days -- one whole week elapsed.
        "drop/train/second.jpg": "retraining_training/week_3/images/second.jpg",
    }


def test_the_pull_interval_does_not_change_the_week_boundaries():
    """Changing how often you look cannot move which folder a file is in.

    If it could, a file's week would depend on when someone happened to
    press Pull -- and a week would stop meaning a span of dates.
    """
    when = _dt.datetime(2026, 10, 5, tzinfo=UTC)
    weekly = pc.week_at(when, _sched(schedule_every_days=7))
    fortnightly = pc.week_at(when, _sched(schedule_every_days=14))
    monthly = pc.week_at(when, _sched(schedule_every_days=30))
    assert weekly == fortnightly == monthly


# ---------------------------------------------------------------------------
# Re-dating the table under settings that have not been applied
#
# The card shows what a week length WOULD mean while it is being chosen.
# The thing that must not happen is a preview that moves files, or one
# that disagrees with what Apply then does.
# ---------------------------------------------------------------------------

def _forecast_world(monkeypatch, config, weeks, now):
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {
        "weeks": [
            {"week": name, "images": images, "labels": images, "bytes": 0,
             "roots": ["retraining_training"], "defined": True}
            for name, images in weeks
        ],
        "roots": ["retraining_training"],
    })
    monkeypatch.setattr(pc, "_now", lambda: now)


def test_a_longer_week_re_dates_the_table_without_moving_anything(monkeypatch):
    """The preview changes boundaries, never contents.

    A folder gathered under seven-day weeks keeps every file it holds; a
    fourteen-day setting only relabels the span it is said to cover. If a
    preview could move files it would not be a preview.
    """
    config = _sched(schedule_every_days=7)
    _forecast_world(monkeypatch, config,
                    [("week_1", 2954), ("week_2", 120)],
                    CUTOFF + _dt.timedelta(days=10))

    weekly = pc.week_forecast()
    fortnightly = pc.week_forecast(overrides={"week_length_days": 14})

    counts = lambda out: {w["week"]: w["images"] for w in out["weeks"] if w["exists"]}
    assert counts(weekly) == counts(fortnightly) == {"week_1": 2954, "week_2": 120}

    closes = lambda out, name: {w["week"]: w["closes_at"] for w in out["weeks"]}[name]
    # week_2 closed a week after the cut-off; now it closes a fortnight after.
    assert closes(weekly, "week_2") == (CUTOFF + _dt.timedelta(days=7)).isoformat()
    assert closes(fortnightly, "week_2") == (CUTOFF + _dt.timedelta(days=14)).isoformat()

    assert weekly["previewing"] is False
    assert fortnightly["previewing"] is True
    # And it says how much there is to re-date, which is what the warning
    # in the card is counting.
    assert fortnightly["existing_weeks"] == 2


def test_week_one_still_closes_on_the_cut_off_whatever_the_length(monkeypatch):
    """The cut-off is week 1's boundary and no length changes that."""
    config = _sched()
    _forecast_world(monkeypatch, config, [("week_1", 10)],
                    CUTOFF + _dt.timedelta(days=3))

    for length in (1, 7, 14, 30):
        out = pc.week_forecast(overrides={"week_length_days": length})
        first = {w["week"]: w for w in out["weeks"]}["week_1"]
        assert first["closes_at"] == CUTOFF.isoformat()
        assert first["is_cutoff"] is True


def test_the_preview_is_what_applying_it_would_give(monkeypatch):
    """The one invariant: previewing a length equals saving it.

    Both go through `week_forecast`, so this is what stops the card
    growing its own copy of the arithmetic and drifting from the sweep.
    """
    now = CUTOFF + _dt.timedelta(days=20)
    saved_as_seven = _sched(schedule_every_days=7)
    _forecast_world(monkeypatch, saved_as_seven, [("week_1", 5)], now)
    previewed = pc.week_forecast(overrides={"week_length_days": 14})

    saved_as_fourteen = _sched(week_length_days=14, schedule_every_days=7)
    _forecast_world(monkeypatch, saved_as_fourteen, [("week_1", 5)], now)
    applied = pc.week_forecast()

    strip = lambda out: [
        (w["week"], w["starts_at"], w["ends_at"], w["state"]) for w in out["weeks"]
    ]
    assert strip(previewed) == strip(applied)


def test_a_preview_writes_nothing(monkeypatch):
    """Looking at a different week length must not create week folders."""
    config = _sched()
    _forecast_world(monkeypatch, config, [("week_1", 5)],
                    CUTOFF + _dt.timedelta(days=40))
    monkeypatch.setattr(pc, "create_week",
                        lambda **kw: pytest.fail("the forecast created a week"))
    monkeypatch.setattr(pc, "s3", FakeS3({}))

    out = pc.week_forecast(overrides={"week_length_days": 1})
    assert out["weeks"]
    assert pc.s3.put == [] and pc.s3.copied == [] and pc.s3.deleted == []


# ---------------------------------------------------------------------------
# Choosing what time of day a pull lands on
#
# The whole point of the setting is that it moves the CLOCK and not the
# dates. A test that only checked "the hour changed" would pass on an
# implementation that also dragged the series off the cut-off.
# ---------------------------------------------------------------------------

SGT = pc.SGT


def _sgt(day, hour, minute=0, month=9, year=2026):
    return _dt.datetime(year, month, day, hour, minute, tzinfo=SGT)


def _timed(time_of_day, every=7):
    return {
        **pc.DEFAULTS,
        "week_cutoff": _sgt(23, 9).astimezone(UTC).isoformat(),
        "week_length_days": 7,
        "schedule_every_days": every,
        "schedule_time": time_of_day,
    }


def test_a_chosen_pull_time_moves_the_clock_and_not_the_dates():
    """Same interval, different time of day: same spacing, and the days
    may shift by one.

    Previously the days were identical whatever the time, because the
    series hung off the cut-off. Now it starts at the next occurrence of
    the chosen time, so a time already past today starts tomorrow and one
    still ahead starts today -- at most a day apart. The SPACING is what
    must hold, and does.
    """
    now = _sgt(25, 11).astimezone(UTC)
    days_of = lambda t: [
        pc._parse_time(x).astimezone(SGT).date()
        for x in pc.next_pull_times(_timed(t), count=4, now=now)
    ]
    hours_of = lambda t: {
        (pc._parse_time(x).astimezone(SGT).hour, pc._parse_time(x).astimezone(SGT).minute)
        for x in pc.next_pull_times(_timed(t), count=4, now=now)
    }

    # The SPACING is what must hold whatever the time of day.
    spacing = lambda t: {(b - a).days for a, b in zip(days_of(t), days_of(t)[1:])}
    assert spacing("") == spacing("02:00") == spacing("23:30")
    # The days may now differ by at most one, because the series starts at
    # the next occurrence of the chosen time rather than on a fixed grid.
    assert abs((days_of("02:00")[0] - days_of("23:30")[0]).days) <= 1
    # The clock is exactly as asked, and blank still means the cut-off's
    # own hour -- one clock, not two.
    assert hours_of("") == {(9, 0)}
    assert hours_of("02:00") == {(2, 0)}
    assert hours_of("23:30") == {(23, 30)}


def test_the_chosen_time_applies_to_every_cadence():
    """It sits under all four choices, so it has to work for all four."""
    now = _sgt(25, 11).astimezone(UTC)
    for every in (1, 2, 7, 14, 28):
        runs = pc.next_pull_times(_timed("02:00", every=every), count=3, now=now)
        local = [pc._parse_time(x).astimezone(SGT) for x in runs]
        assert {(t.hour, t.minute) for t in local} == {(2, 0)}, every
        # ...and the interval is still the interval.
        assert (local[1] - local[0]).days == every, every


def test_a_chosen_time_cannot_bring_a_fortnightly_pull_forward():
    """What the user chose, spelled out -- and the reading has changed.

    This used to assert that 23:30 on a fortnightly pull meant a fortnight
    after the cut-off at 23:30, NOT one tonight; the alternative was
    considered and rejected at the time. It was then reversed on request,
    after the old reading twice produced a schedule that sat two days out
    with no explanation: "set it to pull every 2 days" should mean the
    next one is two days away, not wherever a March grid happens to fall.

    The cost, accepted knowingly: changing the time of day on a RUNNING
    schedule now re-anchors it from today rather than keeping its grid.
    """
    now = _sgt(25, 11).astimezone(UTC)
    first = pc._parse_time(
        pc.next_pull_times(_timed("23:30", every=14), count=1, now=now)[0]
    ).astimezone(SGT)
    # 23:30 tonight, because now is 11:00 and tonight's slot is still
    # ahead. The old reading put this a fortnight after the cut-off.
    assert first.date() == now.astimezone(SGT).date()
    assert (first.hour, first.minute) == (23, 30)


def test_a_chosen_time_does_not_touch_the_weeks(monkeypatch):
    """Weeks stay on the cut-off whatever the pull time says."""
    monkeypatch.setattr(pc, "_now", lambda: _sgt(25, 11).astimezone(UTC))
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {"weeks": [], "roots": []})

    plain = pc.week_forecast(config=_timed(""))
    timed = pc.week_forecast(config=_timed("02:00"))
    shape = lambda out: [(w["week"], w["starts_at"], w["ends_at"]) for w in out["weeks"]]
    assert shape(plain) == shape(timed)

    # And a file's week is decided by the cut-off, never by the pull time.
    when = _sgt(26, 4).astimezone(UTC)
    assert pc.week_at(when, _timed("")) == pc.week_at(when, _timed("23:30"))


def test_a_time_that_is_not_a_time_is_refused(monkeypatch):
    """Refused, not quietly ignored.

    Falling back to the cut-off's hour would leave a schedule running at
    a different time than the one on screen, which is worse than either.
    """
    for bad in ("25:00", "12:99", "9", "abc", "2:0:0"):
        assert pc.parse_time_of_day(bad) is None, bad
    for good, expected in (("02:00", (2, 0)), ("9:05", (9, 5)), ("23:59", (23, 59))):
        assert pc.parse_time_of_day(good) == expected

    monkeypatch.setattr(pc, "read_config", lambda: _timed(""))
    with pytest.raises(ValueError, match="not a time"):
        pc.set_mode(mode="off", schedule_time="25:00")


def test_the_default_is_the_cut_off_time():
    """Blank means the cut-off's own hour -- one clock, not two."""
    now = _sgt(25, 11).astimezone(UTC)
    first = pc._parse_time(pc.next_pull_times(_timed(""), count=1, now=now)[0])
    cutoff = pc._parse_time(_timed("")["week_cutoff"]).astimezone(SGT)
    assert first.astimezone(SGT).hour == cutoff.hour
    assert pc.schedule_plan(_timed(""), now=now)["time_of_day"] == ""
    assert pc.schedule_plan(_timed("02:00"), now=now)["time_of_day"] == "02:00"


# ---------------------------------------------------------------------------
# Which week the next pull actually fills
#
# The clock's week and the week that gets written are different things
# while the backlog is waiting, and saying the clock's one was wrong on
# live data by 1,458 files to 34.
# ---------------------------------------------------------------------------

def _active_cfg(**over):
    return {**pc.DEFAULTS, "week_cutoff": CUTOFF.isoformat(),
            "week_length_days": 7, **over}


def _contents(**weeks):
    return {"weeks": [{"week": name, "images": n, "labels": n, "bytes": 0,
                       "roots": [], "defined": True}
                      for name, n in weeks.items()],
            "roots": []}


def test_the_uncollected_backlog_is_the_active_week():
    """Past the cut-off, but week 1 has not been pulled yet.

    Every file waiting in the source folders predates the cut-off, so
    that is where they go -- however far past it today is.
    """
    config = _active_cfg()
    now = CUTOFF + _dt.timedelta(days=2)
    assert pc.week_at(now, config)["name"] == "week_2"
    assert pc.active_week(config, _contents(), now) == "week_1"
    # A folder that exists but holds nothing is still uncollected.
    assert pc.active_week(config, _contents(week_1=0), now) == "week_1"


def test_once_the_backlog_is_in_the_clock_takes_over():
    config = _active_cfg()
    now = CUTOFF + _dt.timedelta(days=2)
    assert pc.active_week(config, _contents(week_1=1458), now) == "week_2"

    # And it keeps following the clock from then on.
    later = CUTOFF + _dt.timedelta(days=30)
    assert pc.active_week(config, _contents(week_1=1458), later) \
        == pc.week_at(later, config)["name"]


def test_no_backlog_wanted_means_the_clock_decides():
    """`first_week_takes_all` off: nothing reaches back, so nothing waits."""
    config = _active_cfg(first_week_takes_all=False)
    now = CUTOFF + _dt.timedelta(days=2)
    assert pc.active_week(config, _contents(), now) == "week_2"


def test_before_the_cut_off_the_active_week_is_week_one():
    config = _active_cfg()
    assert pc.active_week(config, _contents(), CUTOFF - _dt.timedelta(days=3)) == "week_1"


def test_the_forecast_reports_both(monkeypatch):
    """The card shows one and explains it with the other."""
    config = _active_cfg()
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: _contents())
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=2))

    out = pc.week_forecast()
    assert out["current"] == "week_2"
    assert out["active"] == "week_1"


def test_the_week_prefix_previews_like_the_length_does(monkeypatch):
    """Typing a new prefix renames the table before it is applied.

    The rest of the card already followed the typed prefix; only this
    table did not, so it sat there saying week_1 while everything around
    it said day_1.
    """
    config = _active_cfg()
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: _contents())
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=2))

    assert [w["week"] for w in pc.week_forecast()["weeks"][:3]] \
        == ["week_1", "week_2", "week_3"]

    for prefix, want in (
        ("day_", ["day_1", "day_2", "day_3"]),
        ("batch-", ["batch-1", "batch-2", "batch-3"]),
    ):
        out = pc.week_forecast(overrides={"week_prefix": prefix})
        assert [w["week"] for w in out["weeks"][:3]] == want
        # Everything that names a week follows it, not just the rows.
        assert out["active"] == want[0]
        assert out["current"] == want[1]


def test_a_blank_prefix_falls_back_rather_than_naming_folders_by_number():
    """`1/` and `2/` are not folder names anyone meant to ask for."""
    config = _active_cfg(week_prefix="")
    assert pc.week_at(CUTOFF + _dt.timedelta(days=2), config)["name"] == "week_2"
    assert pc.active_week(config, _contents(), CUTOFF) == "week_1"


def test_renaming_the_prefix_does_not_re_copy_what_is_already_filed(monkeypatch):
    """Dedup is by filename across the whole root, not per week folder.

    So a file sitting in `week_1/images/` is still recognised when the
    prefix becomes `day_`. Otherwise every rename would duplicate the
    entire dataset into freshly named folders.
    """
    s3 = FakeS3({
        "src/a.jpg": _at(3),
        # Already filed, under the OLD prefix.
        "retraining_training/week_1/images/a.jpg": _at(3),
    })
    s3.etags["retraining_training/week_1/images/a.jpg"] = s3.etags["src/a.jpg"]
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    config = {
        **pc.DEFAULTS,
        "week_cutoff": CUTOFF.isoformat(),
        "week_prefix": "day_",
        "sources": [{"prefix": "src", "destination": "retraining_training",
                     "eval_destination": "", "split": "train", "action": "copy"}],
    }
    out = pc.route_pending(config=config)

    assert out["moved"] == 0
    assert s3.copied == []
    # Recognised and skipped, not quietly missed: one file in, one
    # accounted for. (route_pending reports skipped as a count.)
    assert out["skipped"] == 1


def test_a_week_that_is_only_defined_is_not_present(monkeypatch):
    """A row in the table is not a folder in the bucket.

    `week_contents` reports both: weeks found as folders, which carry
    the roots they were found under, and weeks that were merely defined,
    which carry none. Treating the second as present made the state
    column say `present` for a week with nothing behind it -- which is
    exactly the thing that column exists to distinguish.
    """
    config = _active_cfg()
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=2))
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {"weeks": [
        # Defined only -- no roots, nothing filed.
        {"week": "week_1", "images": 0, "labels": 0, "bytes": 0,
         "roots": [], "defined": True},
        # A real folder holding real files.
        {"week": "week_2", "images": 12, "labels": 12, "bytes": 99,
         "roots": ["retraining_training"], "defined": True},
    ], "roots": ["retraining_training"]})

    rows = {w["week"]: w for w in pc.week_forecast()["weeks"]}
    assert rows["week_1"]["exists"] is False
    assert rows["week_2"]["exists"] is True
    # And a defined-but-empty week 1 does not count as collected, so the
    # backlog is still where the next pull goes.
    assert pc.week_forecast()["active"] == "week_1"


def test_a_leftover_week_from_an_old_prefix_is_not_drawn(monkeypatch):
    """Renaming the prefix leaves its rows behind. Empty ones are noise.

    A week DEFINED under `day_` with no folder behind it has nothing in
    it and nothing will ever put anything in it, because every new week
    is now named `meow_`. Drawing it puts a `day_1` in a list of
    `meow_N` and makes the table look broken.
    """
    config = _active_cfg()
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=2))
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {"weeks": [
        {"week": "day_1", "images": 0, "labels": 0, "bytes": 0,
         "roots": [], "defined": True},
    ], "roots": []})

    names = [w["week"] for w in pc.week_forecast(overrides={"week_prefix": "meow_"})["weeks"]]
    assert "day_1" not in names
    assert names[:3] == ["meow_1", "meow_2", "meow_3"]


def test_a_week_holding_files_is_kept_whatever_it_is_called(monkeypatch):
    """The files are the thing that matters.

    A folder full of images does not stop existing because the prefix
    was renamed, and hiding it would be a lie about what is on S3 --
    the opposite failure to the one above.
    """
    config = _active_cfg()
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=2))
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {"weeks": [
        {"week": "week_1", "images": 900, "labels": 900, "bytes": 1,
         "roots": ["retraining_training"], "defined": True},
    ], "roots": ["retraining_training"]})

    rows = {w["week"]: w for w in
            pc.week_forecast(overrides={"week_prefix": "meow_"})["weeks"]}
    assert "week_1" in rows
    assert rows["week_1"]["images"] == 900
    assert rows["week_1"]["exists"] is True
    # ...and the new prefix still drives everything projected.
    assert "meow_1" in rows and "meow_2" in rows


# ---------------------------------------------------------------------------
# One settings file, one set of week folders
#
# Two configurations filing into the same folders merge two datasets with
# no record of the split, and nothing afterwards can tell the images
# apart -- a filename does not say which configuration filed it.
# ---------------------------------------------------------------------------

def test_a_different_prefix_or_root_gives_a_file_its_own_folders(tmp_path, monkeypatch):
    """What separates two settings files: where they write.

    The collision rule is about SOURCES fighting over one destination --
    see test_two_different_datasets_into_one_root_are_still_refused.
    Changing the prefix or the root gives a configuration its own
    folders, so it can never fight with another.
    """
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    # `root` is what identity reads, so it is what has to differ. Setting
    # the legacy pair alone changes nothing while a single root is set.
    base = {**pc.DEFAULTS, "root": "retraining", "week_prefix": "eww_",
            "picked": {"train_images": ["a/images/train"], "train_labels": [],
                       "val_images": [], "val_labels": []}}
    other = {**base, "picked": {"train_images": ["b/images/train"],
                                "train_labels": [], "val_images": [], "val_labels": []}}

    pc.save_settings_file(name="dataset-a", config=base)
    # Different source into the SAME folders -- refused.
    with pytest.raises(ValueError, match="dataset-a already writes"):
        pc.save_settings_file(name="dataset-b", config=other)

    # A different prefix is a different set of folders, so it is allowed.
    pc.save_settings_file(name="dataset-b", config={**other, "week_prefix": "other_"})
    # ...and so is a different root.
    pc.save_settings_file(name="dataset-c", config={**other, "root": "somewhere_else"})

    names = {row["name"] for row in pc.list_settings_files()["files"]}
    assert names == {"dataset-a", "dataset-b", "dataset-c"}


def test_saving_over_a_file_with_its_own_identity_is_fine(tmp_path, monkeypatch):
    """Re-saving is not a collision with yourself."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    base = {**pc.DEFAULTS, "training_root": "r", "eval_root": "e",
            "week_prefix": "eww_"}
    pc.save_settings_file(name="dataset-a", config=base)
    pc.save_settings_file(name="dataset-a", config={**base, "week_length_days": 14})
    assert pc.read_settings_file("dataset-a")["week_length_days"] == 14


def test_folder_identity_is_the_roots_and_the_prefix():
    """What makes two configurations collide, spelled out."""
    a = {"training_root": "r", "eval_root": "e", "week_prefix": "w_"}
    assert pc.folder_identity(a) == ("r", "e", "w_")
    # Trailing slashes and spaces are the same folders.
    assert pc.folder_identity({"training_root": " r/ ", "eval_root": "e",
                               "week_prefix": "w_"}) == ("r", "e", "w_")
    # A blank prefix means the default, and must collide with it.
    assert pc.folder_identity({"training_root": "r", "eval_root": "e"})[2] == "week_"


def test_reading_a_settings_file_applies_nothing(tmp_path, monkeypatch):
    """Viewing another dataset must not re-point the next pull."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    saved = {**pc.DEFAULTS, "training_root": "other_root", "week_prefix": "eww_"}
    pc.save_settings_file(name="dataset-a", config=saved)

    monkeypatch.setattr(pc, "write_config",
                        lambda patch: pytest.fail("reading a file wrote config"))
    monkeypatch.setattr(pc, "set_mode",
                        lambda **kw: pytest.fail("reading a file changed the trigger"))

    got = pc.read_settings_file("dataset-a")
    assert got["week_prefix"] == "eww_"
    assert got["training_root"] == "other_root"


# ---------------------------------------------------------------------------
# Watching and stopping a sweep
#
# The two rules that matter are about what a stop DELETES. Undoing must
# remove only what the run created -- never a file it merely rewrote, and
# never a file it moved, because a move already deleted the original.
# ---------------------------------------------------------------------------

class _State:
    """The run-state row and the manifest, in memory."""

    def __init__(self):
        self.rows: dict = {}
        self.manifests: dict = {}

    def install(self, monkeypatch, stop_after=None):
        monkeypatch.setattr(pc, "write_run_state",
                            lambda run_id, **f: self.rows.setdefault(run_id, {}).update(f)
                            or self.rows[run_id])
        monkeypatch.setattr(pc, "read_run_state", lambda run_id: self.rows.get(run_id))
        monkeypatch.setattr(pc, "_write_manifest",
                            lambda run_id, created: self.manifests.update({run_id: list(created)}))
        monkeypatch.setattr(pc, "read_manifest", lambda run_id: self.manifests.get(run_id, []))
        if stop_after is not None:
            seen = {"n": 0}

            def asked(run_id):
                seen["n"] += 1
                return seen["n"] > stop_after
            monkeypatch.setattr(pc, "stop_requested", asked)
        else:
            monkeypatch.setattr(pc, "stop_requested", lambda run_id: False)


def _copy_config(**over):
    return {
        **pc.DEFAULTS,
        "week_cutoff": CUTOFF.isoformat(),
        "week_by": "file_time",
        "sources": [{"prefix": "src", "destination": "retraining_training",
                     "eval_destination": "", "split": "train", "action": "copy"}],
        **over,
    }


def test_a_stopped_run_deletes_what_it_created(monkeypatch):
    """Stop undoes the run, as asked."""
    files = {f"src/a{i}.jpg": _at(3) for i in range(80)}
    s3 = FakeS3(files)
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "create_week", lambda **kw: {"name": kw["name"],
                                                         "starts_at": kw["starts_at"]})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    _State().install(monkeypatch, stop_after=0)   # stop at the first checkpoint

    out = pc.route_pending(config=_copy_config(), run_id="r1")

    assert out["aborted"] == "stopped by hand"
    undone = out["undone"]
    # Everything it created was removed, and the count matches.
    assert undone["removed"] == undone["created"]
    assert undone["failed"] == []
    # Data keys only: the manifest itself is also deleted, on success.
    data_deletes = [k for k in s3.deleted if not k.startswith(pc.STATE_PREFIX)]
    assert len(data_deletes) == undone["removed"]
    assert all(k.startswith("retraining_training/") for k in data_deletes)
    # It stopped early rather than finishing all 80.
    assert out["moved"] < 80


def test_undo_never_touches_a_file_the_run_only_rewrote(monkeypatch):
    """A revised destination existed before. Deleting it would lose data.

    The run rewrote it because the bytes changed upstream; it did not
    create it. An undo that removed it would delete a file that was there
    before the run started.
    """
    files = {"src/a.jpg": _at(3), "retraining_training/week_1/images/a.jpg": _at(1)}
    s3 = FakeS3(files)
    # Same name, DIFFERENT bytes -> revised, rewritten in place.
    s3.etags["src/a.jpg"] = "new-bytes"
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "create_week", lambda **kw: {"name": kw["name"],
                                                         "starts_at": kw["starts_at"]})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    state = _State()
    state.install(monkeypatch)

    pc.route_pending(config=_copy_config(), run_id="r2")

    assert state.manifests["r2"] == [], \
        "a rewritten file must not be recorded as created by this run"


def test_undo_refuses_when_the_run_moved_anything(monkeypatch):
    """A move deleted the original, so the copy is the only one left.

    Deleting it would destroy the file outright -- the opposite of
    undoing. Refused, loudly, rather than half-done.
    """
    files = {f"src/a{i}.jpg": _at(3) for i in range(80)}
    s3 = FakeS3(files)
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "create_week", lambda **kw: {"name": kw["name"],
                                                         "starts_at": kw["starts_at"]})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    _State().install(monkeypatch, stop_after=0)

    moving = _copy_config(sources=[{
        "prefix": "src", "destination": "retraining_training",
        "eval_destination": "", "split": "train", "action": "move"}])
    out = pc.route_pending(config=moving, run_id="r3")

    assert out["aborted"] == "stopped by hand"
    assert out["undone"]["removed"] == 0
    assert "MOVED" in out["undone"]["refused"]
    # The destinations are still there.
    assert not any(k.startswith("retraining_training/") for k in s3.deleted)


def test_progress_is_published_as_the_sweep_runs(monkeypatch):
    """A percentage needs a denominator, and it has to arrive early."""
    files = {f"src/a{i}.jpg": _at(3) for i in range(60)}
    s3 = FakeS3(files)
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "create_week", lambda **kw: {"name": kw["name"],
                                                         "starts_at": kw["starts_at"]})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    state = _State()
    state.install(monkeypatch)

    pc.route_pending(config=_copy_config(), run_id="r4")

    row = state.rows["r4"]
    assert row["total"] == 60
    assert row["done"] == 60
    assert row["status"] == "done"
    assert row["moved"] == 60


def test_counting_the_denominator_reads_nothing_twice(monkeypatch):
    """count_pending is a LIST-only pass."""
    s3 = FakeS3({f"src/a{i}.jpg": _at(3) for i in range(42)})
    monkeypatch.setattr(pc, "s3", s3)
    assert pc.count_pending(_copy_config()) == 42
    assert s3.copied == [] and s3.deleted == [] and s3.put == []


def test_pending_summary_says_where_waiting_files_would_land(monkeypatch):
    """The visibility a scheduled pull otherwise never gives."""
    s3 = FakeS3({
        "src/old.jpg": CUTOFF - _dt.timedelta(days=30),
        "src/new.jpg": CUTOFF + _dt.timedelta(days=1),
        "src/notes.json": CUTOFF,
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])

    out = pc.pending_summary(config=_copy_config())
    assert out["waiting"] == 2
    assert out["ignored"] == 1          # the .json is not dataset content
    by_week = {row["week"]: row["files"] for row in out["weeks"]}
    assert by_week == {"week_1": 1, "week_2": 1}
    # Looking must not create anything.
    assert s3.copied == [] and s3.put == [] and s3.deleted == []


# ---------------------------------------------------------------------------
# Dating a file by the time written into its name
#
# Measured on the live dataset: 748 images captured across six weeks were
# all uploaded inside one fortnight, so dating by upload put 709 in week_1
# and 39 in week_2. The capture time is in the filename the whole time.
# ---------------------------------------------------------------------------

REAL_NAME = "unk__unk__20260318T075125Z__b668a790-7fcb-4e83-8354-ef2e9f2f974b__f000000.jpg"


def test_the_capture_time_is_read_out_of_the_filename():
    got = pc.capture_time(REAL_NAME)
    assert got == _dt.datetime(2026, 3, 18, 7, 51, 25, tzinfo=UTC)
    # Path in front makes no difference; only the basename is read.
    assert pc.capture_time(f"a/b/c/{REAL_NAME}") == got


def test_a_name_without_a_timestamp_returns_nothing():
    """None, never a guess. The caller decides what to do about it."""
    for name in ("plain.jpg", "frame_0001.jpg", "20260318.jpg",
                 "batch_12345678_frame.jpg", "x__20260318T0751Z__y.jpg"):
        assert pc.capture_time(name) is None, name


def test_eight_digits_that_are_not_a_date_are_refused():
    """An id can look like a date. A month of 13 is not one."""
    assert pc.capture_time("x__20261332T075125Z__y.jpg") is None
    assert pc.capture_time("x__20260230T075125Z__y.jpg") is None   # 30 Feb
    assert pc.capture_time("x__20260318T995125Z__y.jpg") is None   # hour 99


def test_which_clock_answers_is_reported():
    """`file_moment` says where the moment came from, so a week is checkable."""
    uploaded = _dt.datetime(2026, 9, 22, tzinfo=UTC)
    obj = {"Key": f"src/{REAL_NAME}", "LastModified": uploaded}

    when, clock = pc.file_moment(obj, {"week_by": "capture_time"})
    assert (when.date(), clock) == (_dt.date(2026, 3, 18), "capture")

    when, clock = pc.file_moment(obj, {"week_by": "file_time"})
    assert (when, clock) == (uploaded, "upload")

    plain = {"Key": "src/plain.jpg", "LastModified": uploaded}
    when, clock = pc.file_moment(plain, {"week_by": "capture_time"})
    assert (when, clock) == (uploaded, "fallback")


def test_capture_dating_spreads_what_upload_dating_collapses(monkeypatch):
    """The live failure, reproduced.

    Frames taken weeks apart, uploaded together. Under upload time they
    land in one week; under capture time they land in the weeks they were
    taken in.
    """
    together = _dt.datetime(2026, 9, 22, tzinfo=UTC)
    files = {
        f"src/unk__unk__20260318T075125Z__a__f000000.jpg": together,
        f"src/unk__unk__20260401T090000Z__b__f000000.jpg": together,
        f"src/unk__unk__20260429T120000Z__c__f000000.jpg": together,
    }
    config = {
        **pc.DEFAULTS,
        "week_cutoff": _dt.datetime(2026, 3, 17, tzinfo=UTC).isoformat(),
        "week_length_days": 7,
        "sources": [{"prefix": "src", "destination": "retraining",
                     "eval_destination": "", "split": "train", "action": "copy"}],
    }

    def weeks_used(mode):
        s3 = FakeS3(dict(files))
        monkeypatch.setattr(pc, "s3", s3)
        monkeypatch.setattr(pc, "list_weeks", lambda: [])
        monkeypatch.setattr(pc, "create_week",
                            lambda **kw: {"name": kw["name"], "starts_at": kw["starts_at"]})
        monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
        pc.route_pending(config={**config, "week_by": mode})
        return sorted({d.split("/")[1] for _, d in s3.copied})

    # All uploaded the same day -> one week.
    assert len(weeks_used("file_time")) == 1
    # Taken weeks apart -> three weeks.
    assert len(weeks_used("capture_time")) == 3


def test_a_fallback_is_counted_not_hidden(monkeypatch):
    """A source that does not use the convention should be visible."""
    s3 = FakeS3({
        "src/unk__unk__20260318T075125Z__a__f000000.jpg": _at(3),
        "src/no_timestamp_here.jpg": _at(3),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "create_week",
                        lambda **kw: {"name": kw["name"], "starts_at": kw["starts_at"]})
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)

    config = {
        **pc.DEFAULTS, "week_by": "capture_time",
        "week_cutoff": CUTOFF.isoformat(),
        "sources": [{"prefix": "src", "destination": "retraining",
                     "eval_destination": "", "split": "train", "action": "copy"}],
    }
    out = pc.route_pending(config=config)
    assert out["dated_by"] == "capture_time"
    assert out["fell_back_to_upload_time"] == 1
    # And nothing was dropped -- both files were filed.
    assert out["moved"] == 2


def test_capture_dating_is_not_a_misfiling_risk():
    """Only `scan` collapses a long interval into one week."""
    now = _dt.datetime(2026, 10, 20, tzinfo=UTC)
    for mode in ("capture_time", "file_time"):
        plan = pc.schedule_plan(_sched(week_by=mode), now=now)
        assert plan["spans_multiple_weeks"] is True
        assert plan["misfiling_risk"] is False, mode
    assert pc.schedule_plan(_sched(week_by="scan"), now=now)["misfiling_risk"] is True


def test_folder_identity_follows_the_single_root(tmp_path, monkeypatch):
    """It used to read only the legacy pair.

    Every config still carries `training_root`/`eval_root` from DEFAULTS
    whether or not it uses them, so reading those alone gave EVERY
    configuration the same identity -- and every save after the first was
    refused whatever had been configured.
    """
    single = {"root": "retraining", "week_prefix": "week_",
              "training_root": "retraining_training",
              "eval_root": "retraining_evaluation"}
    legacy = {"training_root": "retraining_training",
              "eval_root": "retraining_evaluation", "week_prefix": "week_"}
    assert pc.folder_identity(single) == ("retraining", "", "week_")
    assert pc.folder_identity(legacy) == ("retraining_training",
                                          "retraining_evaluation", "week_")
    assert pc.folder_identity(single) != pc.folder_identity(legacy)
    # The prefix still separates two configs on one root.
    assert pc.folder_identity({**single, "week_prefix": "eww_"}) != pc.folder_identity(single)


def test_the_same_configuration_can_be_saved_under_several_names(tmp_path, monkeypatch):
    """Snapshots are not collisions.

    `..._v3` and `..._v4` of one configuration file identical images to
    identical places; the second is a no-op the dedup already handles.
    Refusing them blocked an ordinary habit.
    """
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    config = {**pc.DEFAULTS, "root": "retraining", "week_prefix": "week_",
              "picked": {"train_images": ["ds/images/train"], "train_labels": [],
                         "val_images": [], "val_labels": []}}

    pc.save_settings_file(name="pull_control_v3", config=config)
    pc.save_settings_file(name="pull_control_v4", config=config)
    pc.save_settings_file(name="pull_control_5rrrr", config=config)

    names = {row["name"] for row in pc.list_settings_files()["files"]}
    assert names == {"pull_control_v3", "pull_control_v4", "pull_control_5rrrr"}


def test_two_different_datasets_into_one_root_are_still_refused(tmp_path, monkeypatch):
    """The case the guard exists for, which is about sources, not names."""
    monkeypatch.setattr(pc, "SETTINGS_DIR", str(tmp_path))
    base = {**pc.DEFAULTS, "root": "retraining", "week_prefix": "week_"}
    first = {**base, "picked": {"train_images": ["ship_dataset/images/train"],
                                "train_labels": [], "val_images": [], "val_labels": []}}
    second = {**base, "picked": {"train_images": ["helmet_dataset/images/train"],
                                 "train_labels": [], "val_images": [], "val_labels": []}}

    pc.save_settings_file(name="ships", config=first)
    with pytest.raises(ValueError, match="different source folders"):
        pc.save_settings_file(name="helmets", config=second)

    # A different prefix gives it its own folders, so it saves.
    pc.save_settings_file(name="helmets", config={**second, "week_prefix": "helmet_"})
    assert {row["name"] for row in pc.list_settings_files()["files"]} == {"ships", "helmets"}


# ---------------------------------------------------------------------------
# Weeks that received nothing
# ---------------------------------------------------------------------------

def _contents_of(**weeks):
    return {"weeks": [{"week": name, "images": n, "labels": n, "bytes": 0,
                       "roots": ["retraining"], "defined": True}
                      for name, n in weeks.items()],
            "roots": ["retraining"]}


def _gap_config():
    config = {**pc.DEFAULTS, "root": "retraining", "week_prefix": "week_",
              "week_cutoff": CUTOFF.isoformat(),
              "picked": {"train_images": ["a"], "train_labels": ["b"],
                         "val_images": ["c"], "val_labels": ["d"]}}
    config["sources"] = pc.sources_from_picked(config)
    return config


def test_only_the_holes_between_filled_weeks_count_as_gaps():
    """The stretch after the last collection is the present, not a gap.

    Giving every passing week a folder would fill the bucket during any
    quiet period, and say nothing true about the data.
    """
    config = _gap_config()
    contents = _contents_of(week_1=26, week_2=392, week_3=203, week_6=8, week_7=119)
    assert pc.gap_weeks(config, contents) == ["week_4", "week_5"]

    # Nothing to be between.
    assert pc.gap_weeks(config, _contents_of(week_1=5)) == []
    assert pc.gap_weeks(config, _contents_of()) == []
    # Consecutive weeks have no holes.
    assert pc.gap_weeks(config, _contents_of(week_1=1, week_2=1, week_3=1)) == []


def test_a_gap_folder_is_made_only_for_the_splits_the_root_receives(monkeypatch):
    """A val/ folder under a root that never sees validation data would be
    exactly the clutter this avoids elsewhere."""
    config = _gap_config()
    s3 = FakeS3({})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "week_contents",
                        lambda cfg=None: _contents_of(week_1=5, week_4=5))

    out = pc.create_gap_folders(config)
    assert out["created"] == ["week_2", "week_3"]
    keys = {k for k, _ in s3.put} if s3.put and isinstance(s3.put[0], tuple) else set(s3.put)
    # One root carrying both splits -> train and val under each subfolder.
    for week in ("week_2", "week_3"):
        for sub in ("images", "labels"):
            for side in ("train", "val"):
                assert f"retraining/{week}/{sub}/{side}/" in keys


def test_a_past_week_with_nothing_in_it_reads_as_empty(monkeypatch):
    """Not "to be created" -- it has been, and nothing arrived."""
    config = _gap_config()
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=40))
    monkeypatch.setattr(pc, "week_contents",
                        lambda cfg=None: _contents_of(week_1=26, week_4=8))

    rows = {w["week"]: w for w in pc.week_forecast()["weeks"]}
    assert rows["week_1"]["state"] == "filled"
    assert rows["week_4"]["state"] == "filled"
    # Between them, nothing arrived AND no folder was made for it.
    assert rows["week_2"]["state"] == "missing"
    assert rows["week_3"]["state"] == "missing"
    assert rows["week_2"]["exists"] is False
    # 40 days past the cut-off is inside week_7, so that is the current
    # one. A week beyond it has not happened, and stays "future".
    assert rows["week_7"]["state"] == "current"
    assert rows["week_8"]["state"] == "future"


def test_a_pull_that_moved_nothing_creates_no_gap_folders(monkeypatch):
    """No files landed, so nothing about the shape of the weeks changed."""
    config = {**_gap_config(), "sources": [{"prefix": "src",
              "destination": "retraining", "eval_destination": "",
              "split": "train", "action": "copy"}]}
    s3 = FakeS3({})
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])
    monkeypatch.setattr(pc, "_record_run", lambda *a, **k: None)
    monkeypatch.setattr(pc, "create_gap_folders",
                        lambda cfg=None: pytest.fail("made folders for an empty sweep"))

    out = pc.route_pending(config=config)
    assert out["moved"] == 0


def test_a_week_folder_holding_only_its_marker_is_still_visible(monkeypatch):
    """The bug that made creating empty folders pointless.

    An S3 "folder" is a zero-byte key ending in `/`, and the routing
    listing skips those -- so a week created and never filled was
    invisible, indistinguishable from one that never existed. Creating it
    would have changed nothing anybody could see.
    """
    config = {**pc.DEFAULTS, "root": "retraining", "week_cutoff": CUTOFF.isoformat(),
              "sources": [{"prefix": "src", "destination": "retraining",
                           "eval_destination": "", "split": "train", "action": "copy"}]}
    s3 = FakeS3({
        "retraining/week_1/images/train/a.jpg": _at(3),
        # week_2 exists as a folder and holds nothing.
        "retraining/week_2/images/train/": _at(3),
    })
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "list_weeks", lambda: [])

    rows = {r["week"]: r for r in pc.week_contents(config)["weeks"]}
    assert "week_2" in rows, "an empty folder must still be listed"
    assert rows["week_2"]["images"] == 0
    assert rows["week_2"]["roots"] == ["retraining"]   # it exists
    # ...and the marker is not counted as an image.
    assert rows["week_1"]["images"] == 1


def test_an_empty_folder_reads_as_empty_and_a_missing_one_does_not(monkeypatch):
    """The distinction the card needs to offer a delete button."""
    config = {**pc.DEFAULTS, "root": "retraining", "week_cutoff": CUTOFF.isoformat()}
    monkeypatch.setattr(pc, "read_config", lambda: config)
    monkeypatch.setattr(pc, "_now", lambda: CUTOFF + _dt.timedelta(days=30))
    monkeypatch.setattr(pc, "week_contents", lambda cfg=None: {"weeks": [
        {"week": "week_1", "images": 5, "labels": 5, "bytes": 1,
         "roots": ["retraining"], "defined": True},
        # created, never filled
        {"week": "week_2", "images": 0, "labels": 0, "bytes": 0,
         "roots": ["retraining"], "defined": True},
    ], "roots": ["retraining"]})

    rows = {r["week"]: r for r in pc.week_forecast()["weeks"]}
    assert (rows["week_1"]["state"], rows["week_1"]["exists"]) == ("filled", True)
    assert (rows["week_2"]["state"], rows["week_2"]["exists"]) == ("empty", True)
    # week_3 was never created at all.
    assert (rows["week_3"]["state"], rows["week_3"]["exists"]) == ("missing", False)
