"""Pull control: route arriving files into dated week folders.

Files land in a flat drop folder and have to end up in the shape the
trainer needs:

    retraining_training/
      week_1/  images/  labels/
      week_2/  images/  labels/
    retraining_evaluation/
      week_1/  images/  labels/

Which week a file belongs to comes from its S3 LastModified -- which is
when it was WRITTEN TO THE BUCKET, not when it was captured. S3 keeps no
capture time, and a JPEG's EXIF date is not read here. So a batch of
last month's photos uploaded today is filed under today's week. That is
the right answer when the drop folder is fed by a pipeline that uploads
as it goes, and the wrong one when someone uploads a backlog by hand --
in which case give it its own week and set that week's start date.

This module is the single source of truth for that routing. It is also
the Lambda's handler file -- Terraform zips this exact file -- so it
imports nothing beyond boto3 and the standard library, and nothing from
the rest of the app. Two copies of the rule that decides where a file
goes is how the drop folder and the trainer end up disagreeing.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import itertools
import json
import os
import re
import uuid

import boto3
from botocore.exceptions import ClientError

BUCKET = os.environ.get("PIPELINE_BUCKET", "")
TABLE_NAME = os.environ.get("PULL_CONTROL_TABLE", "track2-pull-control")
REGION = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-1"

s3 = boto3.client("s3", region_name=REGION)
ddb = boto3.client("dynamodb", region_name=REGION)

# The topic every human-facing notification already goes through. Named by
# environment so this file keeps importing nothing beyond boto3 and the
# standard library -- Terraform zips it as the Lambda handler, and an
# import of track2_shared (where the other callers get this ARN) would not
# resolve there.
NOTIFY_TOPIC_ARN = os.environ.get("PULL_NOTIFY_TOPIC_ARN", "")


CONFIG_SORT = "v1"
#: A week folder holds at most this many days before the next one starts.
DEFAULT_WEEK_DAYS = 7
#: Routing history is for seeing what happened last night, not forever.
EVENT_TTL_DAYS = 30

DEFAULTS = {
    # [{"prefix": "drop/training", "destination": "retraining_training"}]
    # One watched folder per destination root. The picture has two roots,
    # and a file cannot be sorted into "training or evaluation" from its
    # own contents -- where it was dropped is the only thing that says so.
    # Derived from `picked` below. Kept because the router and the Lambda
    # both walk it, but it is no longer edited directly.
    "sources": [],
    # The four folder lists, exactly as chosen in the card. These are the
    # configuration now: `sources` is rebuilt from them on every save.
    #
    # Stored rather than derived-and-discarded because a source records a
    # prefix and a split, but NOT whether the folder holds images or
    # labels -- that is decided per file by extension. So sources alone
    # cannot be turned back into the four lists, and a saved setting would
    # not round-trip.
    "picked": {"train_images": [], "train_labels": [], "val_images": [], "val_labels": []},
    "source_bucket": "",
    # Where week folders are written. Defaults to this app's own bucket;
    # the card may point it at any bucket the stack grants writes to.
    "destination_bucket": "",
    # ONE root holds everything: the week folders, and inside each the
    # images/ and labels/ subfolders with train/ and val/ under them.
    #
    #     <root>/week_1/images/train/   <root>/week_1/images/val/
    #     <root>/week_1/labels/train/   <root>/week_1/labels/val/
    #
    # The split stays in the path because one root now receives both --
    # `splits_per_root` sees that and `placement` keeps the level, which
    # is the whole reason that guard exists. Flattening here would put
    # the evaluation images in with the training ones.
    #
    # One root also means one S3 prefix, which is what the trainer
    # downloads: two top-level roots had only the bucket in common, so a
    # data.yaml naming both could only be resolved by fetching everything.
    "root": "retraining",
    # Superseded by `root` and kept so an older settings file still
    # loads. Only consulted when `root` is blank.
    "training_root": "retraining_training",
    "eval_root": "retraining_evaluation",
    # When week 1 ENDS, as an ISO timestamp. Week 1 is the backlog and has
    # no start: everything filed up to this moment goes into it, however
    # old. Week 2 begins here and runs for `week_length_days`, week 3
    # after that, and so on without further input.
    #
    # An end rather than a start, because that is the decision actually
    # being made -- "everything so far is the baseline, and the clock
    # starts now". A start date leaves the question of what happens to
    # everything before it unanswered.
    "week_cutoff": "",
    "images_subpath": "images",
    "labels_subpath": "labels",
    "week_prefix": "week_",
    "week_length_days": DEFAULT_WEEK_DAYS,
    # Off, the first file that falls past the newest week stops and says
    # so. On, the weeks it needs are created for it -- which is the point
    # of an unattended pull.
    "auto_create_weeks": True,
    # Week 1 is the backlog: everything already in the drop folder goes
    # into it, whatever its date, and later weeks take only what arrives
    # after them. Off, a file older than week 1 belongs to no week -- the
    # reading to want when the start date is meant to EXCLUDE older data
    # rather than gather it.
    "first_week_takes_all": True,
    # scan | file_time | capture_time. Which week a file lands in.
    #
    #   capture_time -- the moment written INTO the filename, e.g.
    #                   `unk__unk__20260318T075125Z__<uuid>__f000000.jpg`.
    #                   When the frame was taken.
    #   file_time    -- each file goes to the week its own S3 LastModified
    #                   falls in. When it was uploaded.
    #   scan         -- the whole sweep goes into the week that is current
    #                   when the sweep runs, whatever the files' dates.
    #
    # The difference is not academic. Measured on the live escalation
    # dataset: 748 images captured across six weeks (18 Mar - 29 Apr) were
    # all uploaded inside one fortnight in September. Dating by upload put
    # 709 in week_1 and 39 in week_2 -- two folders, decided entirely by
    # when somebody happened to copy them. Dating by capture puts the same
    # images in five weeks that correspond to when the work happened.
    #
    # capture_time is the default: the date in the filename is when the
    # frame was taken, and that is what a week is supposed to mean. Upload
    # time is an accident of when someone ran a copy -- on the live
    # dataset it collapsed six weeks of captures into two folders.
    #
    # It also survives a pull interval longer than a week, which `scan`
    # does not: under `scan` a fortnightly pull into 7-day weeks puts two
    # weeks of images in one folder and leaves the other empty.
    # The Auto Training card's own selection: which root it is pointed at
    # and which weeks have been ticked out. Kept HERE rather than in the
    # browser so it rides along in the settings file -- one file holding
    # every card's choices was the point of that file, and a selection
    # that only exists in a tab is one nobody can restore.
    "auto_training": {},
    # Step 2's own choices: which curated folder to split, where the split
    # goes, and how. Kept here with every other card's settings so the one
    # settings file holds them -- they used to live only in the browser
    # tab, so saving a file recorded the golden set's SWITCH and none of
    # the folders it pointed at. `{bucket, source, target_bucket,
    # target_folder, split, train_percent}`.
    "golden_set": {},
    # The SQL tab's choices: which database, the bucket and folder that
    # narrow the images, and the query last typed. `{database, bucket,
    # folder, sql}`. A setting like any other card's, so a settings file
    # brings the query back with everything else.
    "sql_explorer": {},
    # The CENTRAL DATA STORE (2026-10-29): the one bucket and folder where
    # all the data and images are meant to be kept. `{bucket, folder}`.
    # Chosen on its own card on Data Preparation, below the Pull queue,
    # and saved like every other card's choice -- a setting nobody can
    # restore is not a setting.
    "central_store": {},
    # The Assemble card's own choices: the folders it merges into a
    # dataset, the detector it writes classes for, whether it may replace
    # a data.yaml it did not write, and whether it is the ONLY source of
    # training data -- the tick that switches Pull Control, Auto Training
    # and Retention off.
    #
    # Kept here for the same reason as everything around it: a folder
    # chosen in a browser tab is not a setting anybody can restore, and a
    # settings file written while it was on screen did not contain it.
    # This card was added without this key, so every folder, class list
    # and destination picked in it vanished on reload while the file
    # reported a clean save -- the save was honest, the card simply never
    # put anything in the config for it to copy.
    #
    # `{only, use, train_sources, eval_sources, folders, detector_slug,
    #   overwrite, scheduled, built, label_classes, drop_undeclared_labels}`.
    #
    # `drop_undeclared_labels`: when the label files use a class the yaml
    # does not name, the training container drops those boxes from its
    # downloaded copy before Ultralytics reads them. S3 is never touched.
    # Off by default: a run must say it wants this.
    #
    # `label_classes` is what the LABEL FILES mean -- every class they
    # use, in index order -- as distinct from `detector_slug`, which is
    # what to TRAIN. They differ whenever a site dataset labelled with
    # everything on it feeds a one-class Model Name; the yaml must
    # declare the former or Ultralytics drops every image holding a box
    # it has no name for.
    #
    # `scheduled` is what lets the unattended chain run this card instead
    # of the root-and-week-folders build: `retrain_schedule.assemble_plan`
    # reads it, and OFF means the old path, so a settings file written
    # before this existed cannot start assembling in place by surprise.
    # The classes are NOT read from here -- they come from the Model Name
    # named by `detector_slug`, so a corrected class list reaches the
    # schedule without the card being reopened.
    # The after-pull chain's own settings -- whether it is on, which
    # triggers may chain, and the guards. Listed here, even though
    # queue_after_pull.py owns the shape, so that `read_config` always
    # returns the key and a settings file saved before anybody touched
    # the card still carries it. Without an entry here the key simply
    # was not in the file, and loading that file left the chain at
    # whatever the machine happened to be set to.
    "queue_after_pull": {},
    # Which cards are open, per card. Not a setting in the sense the
    # others are -- it changes nothing about what a run does -- but
    # restoring a file should put the page back the way it was left, and
    # a page that reopens with every card collapsed is one you have to
    # rebuild by hand before you can read it.
    "card_state": {},
    # Whether every run freezes a snapshot of its dataset before training
    # (the chain's freeze step, and a queue-add from the training card).
    # A setting because it is a choice about RECORDS, not about training:
    # off, a run trains on exactly the same images and simply leaves no
    # snapshot behind -- so it cannot be picked again under "Use a frozen
    # snapshot". True reproduces the behaviour before this existed. The
    # "Freeze this dataset" button freezes regardless.
    "freeze_on_run": True,
    "dataset_assemble": {},
    # The Import Checkpoint card's standing choices -- architecture, where
    # it looks for weights, which detector it files them under, and
    # whether class names come from the file.
    #
    # The STANDING ones only. A chosen file, an inspection result and the
    # outcome of a registration are not settings: they belong to one
    # action, and restoring them would show a finished import that never
    # happened on this machine.
    #
    # `{architecture, source, bucket_key, prefix, detector_slug,
    #   use_file_classes}`.
    "import_checkpoint": {},
    # Which detector everything trains as, and at what size, chosen once
    # in "Register a model and Classes". `{"slug": ..., "architecture":
    # ...}`.
    #
    # The DEFINITIONS of detectors are data and live where they always
    # have; this is the SELECTION, which is a setting -- so it sits with
    # the other settings and rides along in the one settings file.
    #
    # The unattended retrain always uses this, never the training form's
    # per-run choice. A one-off manual run with a different detector must
    # not be able to silently change what the scheduled runs do months
    # later, and that is only guaranteed if they read different places.
    "default_detector": {},
    # How every run is shaped: epochs, img_size, batch, seed, patience and
    # a default note. Chosen once in "Models and Classes" beside the
    # detector, for the same reason that is there -- a number typed into
    # the training form lives in a browser tab, so the scheduled retrain
    # had no way to read it and kept a second copy that could drift.
    #
    # One source. A schedule training for 50 epochs while the form says
    # 100 is a disagreement nobody notices until two models that should
    # match do not.
    "default_training": {},
    # The dataset the LAST automatic run built and froze:
    # {"folder", "snapshot_id", "files", "set_by", "at"}.
    #
    # Written by retrain_schedule.run_chain so the training card can open
    # already pointing at it. Without it the unattended path is invisible
    # -- it rebuilds the lists, writes the yaml and freezes a snapshot,
    # and the card still says "Nothing chosen yet" because its folder
    # lives in a browser tab.
    "default_dataset": {},
    # The dataset the PULL CHAIN last froze, kept apart from
    # `default_dataset` on purpose.
    #
    # `default_dataset` is the live slot: whatever was chosen most
    # recently, by anything. Loading a snapshot by hand overwrites it,
    # which is right for a slot that means "what a run would use" and
    # wrong for the question "what did the pull produce" -- and the
    # training card now offers those as two separate choices. One slot
    # cannot answer both.
    #
    # Written only by `run_chain`, so it survives any number of manual
    # picks. `{folder, snapshot_id, files, set_by, at}`.
    "last_pull_dataset": {},
    # WHICH KIND of training: "existing" (continue from a registered
    # version), "scratch" (random weights), or "import" (file a .pt
    # rather than train). Saved beside the model it applies to, because
    # "helmet / yolo11s" and "helmet / yolo11s from scratch" are two
    # different runs and a settings file that records only the first
    # restores the wrong one.
    "default_source": "existing",
    # The registered version a run continues FROM -- the model package
    # ARN picked in "Train on an existing model".
    #
    # Saved because it is half of what defines a run: "helmet / yolo11s"
    # says which registry, and this says which version of it. A settings
    # file that records the first but not the second restores a run that
    # starts from somewhere else, and nothing on screen says so.
    "default_version": "",
    # "live" -- train on the week folders as they stand. "snapshot" --
    # train on a frozen definition, the same images every time whatever
    # has been pulled since.
    #
    # Saved because it is part of what a settings file is FOR. Two runs
    # with identical folders, detector and epochs are still different
    # runs if one is pinned to a snapshot and the other follows the live
    # corpus, and a file that restored everything except that would
    # reproduce the wrong one of the two without saying so.
    "data_source": "live",
    # The settings file these settings came from, or were last written
    # to. Shown on every card that offers Save, so "which file am I
    # working in" is answerable without scrolling to find out -- and so
    # that saving over the wrong one is visible before it happens rather
    # than afterwards.
    "settings_file": "",
    # The same, per tab (2026-10-29): Data Preparation saves `<name>_DP`
    # and Multi-Auto Training `<name>_AT`, each holding only its own page,
    # so each tab says which of ITS files is loaded. See SCOPES.
    "settings_file_dp": "",
    "settings_file_at": "",
    # The unattended retrain: pull, rebuild the data.yaml under the
    # current retention policy, freeze it, and submit a run. Off until
    # somebody sets it up, and every field it needs is here rather than
    # in the schedule, so the settings file carries the whole thing.
    #
    # Kept beside the pull's own schedule deliberately. They are two
    # clocks doing different jobs -- one decides when data arrives, this
    # decides when a model is trained on it -- and a single schedule
    # doing both was the first design. It is wrong: a pull that finds
    # nothing should not spend GPU hours, and a retrain should be able to
    # run on a Sunday against a week that filled on Tuesday.
    "retrain_schedule": {},
    "week_by": "capture_time",
    # off | arrival | schedule. Held here as well as on the EventBridge
    # rules so the UI can show what is meant to be on when a rule has been
    # changed underneath it.
    "mode": "off",
    "schedule_expression": "rate(1 day)",
    # How many days between scheduled pulls. The schedule is ANCHORED ON
    # `week_cutoff`: the first pull is one interval after it, and every
    # pull inherits the cut-off's time of day. So one date drives both
    # clocks and there is no second date to keep in step.
    #
    # The two are still separate settings, and deliberately so. This says
    # how often the system goes and LOOKS; `week_length_days` says which
    # folder what it finds goes in. Pulling fortnightly into weekly
    # folders is a normal thing to want, and works because each file is
    # dated individually -- see `week_by`.
    #
    # No cut-off means no schedule: nothing has said when to start
    # counting from.
    "schedule_every_days": 7,
    # Where "every N days" counts from. "today": the next pull is the
    # next time the clock reaches `schedule_time`, then every N days from
    # there (re-anchored on each Apply). "date": every N days on the grid
    # that starts at `schedule_start_date` (YYYY-MM-DD, Singapore), so the
    # pull days stay the same whenever Apply is pressed. Both kept so the
    # date survives switching the radio back and forth.
    "schedule_from": "today",
    "schedule_start_date": "",
    # What time of day a scheduled pull lands on, as "HH:MM" Singapore
    # time. Blank means the cut-off's own time, which is the default and
    # keeps the card down to one clock.
    #
    # It moves the CLOCK only, never the date: the series is still
    # counted from the cut-off, so setting 02:00 on a fortnightly pull
    # gives a pull a fortnight after the cut-off at 02:00 -- not one
    # tonight. Overnight pulls are the reason to want it; testing at a
    # chosen minute is the other.
    "schedule_time": "",
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"],
    "label_extensions": [".txt"],
    # A name that already exists at the destination is left alone. An
    # image and its label share a stem, so renaming one of a colliding
    # pair and not the other silently unpairs them; skipping keeps both
    # halves of the decision visible instead.
    "overwrite": False,
    # Recency weighting, owned by the retention card. Kept here
    # because it decays the week folders this writes, and two
    # places to configure one pipeline is one too many.
    "retention": {},
}


# --------------------------------------------------------------------------
# Config and weeks
# --------------------------------------------------------------------------

def _plain(item: dict) -> dict:
    """A DynamoDB item as ordinary Python."""
    out = {}
    for key, value in item.items():
        if "S" in value:
            out[key] = value["S"]
        elif "N" in value:
            out[key] = float(value["N"]) if "." in value["N"] else int(value["N"])
        elif "BOOL" in value:
            out[key] = value["BOOL"]
    return out


def read_config() -> dict:
    """The saved settings, with every default filled in.

    Merged over the defaults rather than returned raw: a setting added
    after a config was written would otherwise come back missing, and the
    first caller to read it would be the one that crashed.
    """
    config = dict(DEFAULTS)
    try:
        got = ddb.get_item(
            TableName=TABLE_NAME,
            Key={"kind": {"S": "config"}, "sort": {"S": CONFIG_SORT}},
            ConsistentRead=True,
        ).get("Item")
    except ClientError:
        got = None
    if got and "body" in got:
        try:
            config.update(json.loads(got["body"]["S"]))
        except (ValueError, KeyError):
            pass
    return config


def write_config(config: dict) -> dict:
    """Saves the settings, keeping anything the caller did not mention."""
    merged = read_config()
    merged.update(config or {})
    # Whole thing as one JSON blob: it is a single small document that is
    # always read and written together, and a blob cannot half-apply.
    ddb.put_item(
        TableName=TABLE_NAME,
        Item={
            "kind": {"S": "config"},
            "sort": {"S": CONFIG_SORT},
            "body": {"S": json.dumps(merged)},
            "updated_at": {"S": _now().isoformat()},
        },
    )
    return merged


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _parse_time(value) -> _dt.datetime:
    """An ISO timestamp, or a date, as an aware UTC datetime."""
    if isinstance(value, _dt.datetime):
        return value if value.tzinfo else value.replace(tzinfo=_dt.timezone.utc)
    text = str(value or "").strip().replace("Z", "+00:00")
    if not text:
        raise ValueError("A week needs a start time.")
    # A bare date means midnight UTC that day, which is what someone
    # typing "2026-09-01" into a date box means by it.
    parsed = _dt.datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=_dt.timezone.utc)


def list_weeks() -> list:
    """Every week, oldest first.

    Sorted by the key rather than in Python: the range key starts with the
    start time, so DynamoDB returns them in the order the routing needs
    and a week inserted between two others lands in the right place.
    """
    weeks = []
    kwargs = {
        "TableName": TABLE_NAME,
        "KeyConditionExpression": "#k = :k",
        "ExpressionAttributeNames": {"#k": "kind"},
        "ExpressionAttributeValues": {":k": {"S": "week"}},
    }
    while True:
        page = ddb.query(**kwargs)
        for item in page.get("Items", []):
            row = _plain(item)
            weeks.append({"name": row.get("name", ""), "starts_at": row.get("starts_at", "")})
        if not page.get("LastEvaluatedKey"):
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    return weeks


def _week_sort(starts_at: str, name: str) -> str:
    return f"{starts_at}#{name}"


def create_week(*, name: str, starts_at, make_folders: bool = False) -> dict:
    """Defines a week. Creates no folders.

    It used to lay down empty marker keys under every root, for both
    splits and both of images/ and labels/, so an empty week showed up in
    a bucket listing. That is eight keys per week that nothing ever
    writes to, and half of them for splits the week will never receive.

    Nothing needs them now: the weeks table fills in every week between 1
    and the current one whether or not a folder exists, so an empty week
    is visible without inventing one. A folder appears when a file lands
    in it, and a folder that is there means something is in it.

    `make_folders` is kept so existing callers still type-check, and is
    ignored -- there is nothing to create.
    """
    name = (name or "").strip().strip("/")
    if not name:
        raise ValueError("Give the week a name.")
    when = _parse_time(starts_at)
    item = {
        "kind": {"S": "week"},
        "sort": {"S": _week_sort(when.isoformat(), name)},
        "name": {"S": name},
        "starts_at": {"S": when.isoformat()},
    }
    try:
        ddb.put_item(
            TableName=TABLE_NAME,
            Item=item,
            ConditionExpression="attribute_not_exists(#s)",
            ExpressionAttributeNames={"#s": "sort"},
        )
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "ConditionalCheckFailedException":
            raise
        # Two callers asking for the same week is the expected race when
        # a schedule sweep and an arrival event overlap, not an error.

    return {"name": name, "starts_at": when.isoformat()}


def delete_week(*, name: str, starts_at: str) -> None:
    """Forgets a week. The folders and their contents are left alone."""
    ddb.delete_item(
        TableName=TABLE_NAME,
        Key={
            "kind": {"S": "week"},
            "sort": {"S": _week_sort(_parse_time(starts_at).isoformat(), name)},
        },
    )


def week_folder_prefixes(config: dict | None = None,
                         names: list | None = None) -> list:
    """Every `<root>/<week>/` prefix the defined weeks occupy.

    Across all roots, because a week is not one folder: a source that
    files `val` into its own eval root gives the same week a second
    home, and deleting only the training one leaves half of it behind.
    """
    config = config or read_config()
    bucket = destination_bucket(config)
    week_prefix = str(config.get("week_prefix") or "week_").strip()

    roots: list = []
    for candidate in [*sorted(splits_per_root(config)),
                      str(config.get("root") or ""),
                      str(config.get("training_root") or ""),
                      str(config.get("eval_root") or "")]:
        root = str(candidate or "").strip().strip("/")
        if root and root not in roots:
            roots.append(root)

    wanted = {str(n) for n in (names or [])}
    out: list = []
    for root in roots:
        # FROM S3, not from the week rows. A week folder exists in the
        # bucket whether or not a row describes it, and the rows can be
        # forgotten independently -- `purge_week` and the bulk delete
        # both do exactly that. Deriving the list from the rows meant
        # that once they were gone the folders became invisible to this
        # and could never be deleted, while the card went on showing
        # them because IT reads the bucket. Two sources, one of them
        # wrong.
        try:
            names_here = t2_list_s3_folders(f"{root}/", bucket=bucket)
        except Exception:                                      # noqa: BLE001
            continue
        for name in sorted(names_here):
            if week_prefix and not name.startswith(week_prefix):
                continue
            if wanted and name not in wanted:
                continue
            key = f"{root}/{name}/"
            if key not in out:
                out.append(key)
    return out


def t2_list_s3_folders(prefix: str, *, bucket: str = "") -> list:
    """Folder names directly under `prefix`, from the bucket itself.

    A local copy rather than `track2_shared.list_s3_folders` because
    that one is pinned to its own module-level BUCKET, and the weeks
    live in whichever bucket the pull writes to.
    """
    bucket = bucket or destination_bucket()
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    out: list = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        for common in page.get("CommonPrefixes", []):
            out.append(common["Prefix"][len(prefix):].rstrip("/"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return out


def count_week_objects(config: dict | None = None,
                       names: list | None = None) -> dict:
    """How much `purge_week_folders` would destroy. Counted first and
    shown, because the number IS the warning: "delete the week folders"
    and "delete 1,496 files" are not the same sentence."""
    config = config or read_config()
    bucket = destination_bucket(config)
    per_prefix: dict = {}
    total = 0
    size = 0
    for prefix in week_folder_prefixes(config, names):
        n = 0
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            page = s3.list_objects_v2(**kw)
            for obj in page.get("Contents", []):
                n += 1
                size += int(obj.get("Size") or 0)
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        per_prefix[prefix] = n
        total += n
    return {"bucket": bucket, "prefixes": per_prefix,
            "objects": total, "bytes": size}


def purge_week_folders(config: dict | None = None,
                       names: list | None = None) -> dict:
    """PERMANENTLY delete every object under the week folders.

    There is no second copy. The pull MOVES files out of the drop
    folders rather than copying them, so these objects are the only
    ones; `undo_run` exists precisely because a move cannot be undone
    from the destination. Nothing here is recoverable.

    Paginated and batched: `delete_objects` takes 1000 keys at a time,
    and a corpus is routinely larger than one page -- a single
    unpaginated pass would delete the first thousand, report success,
    and leave the rest.

    The week ROWS are left alone. Forgetting the definition is a
    separate decision with its own button, and doing both from one
    press would make "I wanted the files gone, not the schedule" an
    unrecoverable mistake.
    """
    config = config or read_config()
    bucket = destination_bucket(config)
    deleted = 0
    errors: list = []
    for prefix in week_folder_prefixes(config, names):
        # LIST EVERYTHING FIRST, then delete. Deleting inside the
        # pagination loop shifts the listing under the continuation
        # token, so the page after each delete starts 1000 items further
        # on than it should and a whole page survives -- while the call
        # reports success. Found by test, not in production.
        keys: list = []
        token = None
        while True:
            kw = {"Bucket": bucket, "Prefix": prefix}
            if token:
                kw["ContinuationToken"] = token
            page = s3.list_objects_v2(**kw)
            keys += [o["Key"] for o in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")

        for start in range(0, len(keys), 1000):
            batch = [{"Key": k} for k in keys[start:start + 1000]]
            out = s3.delete_objects(Bucket=bucket,
                                    Delete={"Objects": batch, "Quiet": True})
            failed = out.get("Errors") or []
            deleted += len(batch) - len(failed)
            for err in failed:
                errors.append(f"{err.get('Key')}: {err.get('Message')}")
    return {"bucket": bucket, "deleted": deleted, "errors": errors}


def next_week_name(config: dict, weeks: list) -> str:
    """`week_3` when two weeks exist.

    Counted rather than parsed out of the last name so a week someone
    named `week_pilot` does not stop the next one being numbered.
    """
    prefix = str(config.get("week_prefix") or "week_")
    return f"{prefix}{len(weeks) + 1}"


def week_for(
    when: _dt.datetime,
    weeks: list,
    length_days: int | None = None,
    backfill_first: bool = False,
) -> dict | None:
    """The week a timestamp falls in, or None.

    A week runs from its own start until the next one begins. The NEWEST
    week is the one that needs `length_days`: without an end it would
    swallow every future file for ever and no new week would ever be
    created, which is the opposite of knowing when a week starts.

    No longer on the routing path -- `project_week` asks `week_at`
    instead, so where a file goes never depends on which week rows happen
    to exist. Kept because "which of these defined weeks contains this
    moment" is still a fair question to ask of a week list.

    `backfill_first` makes week 1 open-ended backwards, so it collects
    everything already sitting in the drop folder and later weeks take
    only what arrives after them. That is the usual way to start: the
    first week is the backlog. Without it a file older than week 1
    belongs to no week at all, which is the safer reading when the start
    date is meant to exclude older data rather than gather it.
    """
    chosen = None
    chosen_index = -1
    for index, week in enumerate(weeks):
        try:
            start = _parse_time(week["starts_at"])
        except ValueError:
            continue
        if start > when:
            break
        chosen = week
        chosen_index = index
    if chosen is None:
        return weeks[0] if backfill_first and weeks else None
    if length_days and chosen_index == len(weeks) - 1:
        ends = _parse_time(chosen["starts_at"]) + _dt.timedelta(days=max(int(length_days), 1))
        if when >= ends:
            return None
    return chosen


def project_week(when: _dt.datetime, weeks: list, config: dict) -> tuple:
    """Which week `when` belongs to, inventing the week if it is new.

    Returns `(week_or_None, projected)`, where `projected` are the weeks
    that do not exist yet and would have to be created. Pure -- it writes
    nothing. Both the preview and the sweep start here, which is what
    keeps a preview from describing a different outcome than the move.

    The arithmetic is `week_at` and nothing else. This used to walk the
    stored week rows and project forward one length at a time, which was
    a second answer to "which week is it" -- and the two could disagree
    whenever a week row had been created under an older cut-off. Now a
    stored row only records that a week EXISTS; where a file goes is
    always the cut-off plus the week length.
    """
    placed = week_at(when, config)

    if placed["index"] <= 1 and not bool(config.get("first_week_takes_all", True)):
        # The backlog is explicitly not wanted: only what arrives after
        # the cut-off is collected, and week 1 stays empty. Without a
        # cut-off nothing is after it, so nothing is collected at all --
        # which is why the card warns when both are set that way.
        return None, []

    week = {"name": placed["name"], "starts_at": placed["starts_at"]}
    if week["name"] in {w["name"] for w in weeks}:
        return week, []
    if not config.get("auto_create_weeks"):
        # Nothing may be created, so this file belongs to no week that
        # exists. It is skipped and reported rather than filed into the
        # newest week there happens to be: putting a week-6 file in week
        # 2 would corrupt the only thing a week means.
        return None, []
    return week, [week]


#: `20260318T075125Z` anywhere in a filename: ISO 8601 basic, UTC by the
#: trailing Z. Matched as a whole token so an eight-digit id cannot be
#: mistaken for a date.
CAPTURE_STAMP = re.compile(r"(\d{8})T(\d{6})Z")


def capture_time(key: str) -> _dt.datetime | None:
    """The moment written into a filename, or None if there is not one.

    Returns None rather than guessing. A filename that does not carry a
    timestamp is a fact about that file, and the caller decides what to
    do about it -- silently inventing a date here would be the worst of
    the available options.
    """
    found = CAPTURE_STAMP.search(key.rsplit("/", 1)[-1])
    if not found:
        return None
    try:
        return _dt.datetime.strptime(
            f"{found.group(1)}T{found.group(2)}Z", "%Y%m%dT%H%M%SZ"
        ).replace(tzinfo=_dt.timezone.utc)
    except ValueError:
        # Eight digits that are not a date -- 20261332, say.
        return None


def file_moment(obj: dict, config: dict) -> tuple:
    """`(moment, source)` -- when a file counts as having happened.

    One place, so the preview, the sweep and the waiting count cannot
    disagree about which week a file belongs to. `source` says which
    clock answered, because "which week is this in" is only checkable if
    you know what was compared.

    Falls back to the upload time when a filename carries no stamp. The
    alternative -- skipping it -- loses data silently to a naming
    convention, and the count of fallbacks is reported so a source that
    does not use the convention is visible rather than merely tolerated.
    """
    if str(config.get("week_by") or "") == "capture_time":
        taken = capture_time(obj["Key"])
        if taken is not None:
            return taken, "capture"
        return obj["LastModified"], "fallback"
    return obj["LastModified"], "upload"


def week_for_scan(weeks: list, config: dict, now: _dt.datetime | None = None) -> tuple:
    """The week THIS SCAN files into -- the batch model.

    The source is a folder edited in place, not a series of dated drops,
    so there is no per-file event to date a batch by. The scan is the
    batch instead: the first one takes everything present as week 1, and
    every later one files whatever is new into whichever week is current.

    That makes a week mean "what appeared between these two dates", which
    is what it has to mean when the upstream folder has no notion of a
    pull. Dating each file by its own S3 timestamp instead would scatter
    one scan's findings across several weeks -- see `week_by`.
    """
    week, to_create = project_scan_week(weeks, config, now)
    if to_create:
        week = create_week(name=to_create["name"], starts_at=to_create["starts_at"])
        return week, weeks + [week]
    return week, weeks


def project_scan_week(weeks: list, config: dict, now: _dt.datetime | None = None) -> tuple:
    """`week_for_scan` without writing. Returns `(week, week_to_create)`.

    The preview and the sweep have to name the same week, so both start
    here and only the sweep persists what comes back. The arithmetic is
    `week_at`; this only decides whether the row needs creating.
    """
    now = now or _now()
    placed = week_at(now, config)
    week = {"name": placed["name"], "starts_at": placed["starts_at"]}
    existing = {w["name"] for w in weeks}
    if week["name"] in existing:
        return week, None
    if not config.get("auto_create_weeks") and weeks:
        # Nothing may be created, so it goes to the newest week there is.
        return weeks[-1], None
    return week, week


def ensure_week_covering(when: _dt.datetime, weeks: list, config: dict) -> tuple:
    """`project_week`, with the projected weeks actually created."""
    week, projected = project_week(when, weeks, config)
    for pending in projected:
        create_week(name=pending["name"], starts_at=pending["starts_at"])
    return week, weeks + projected


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def _list_objects(prefix: str, bucket: str = "", *, markers: bool = False):
    """Every real object under a prefix, newest-agnostic.

    Zero-byte keys ending in `/` are the markers a folder is made of and
    are not files to route -- so they are skipped, except when asked for.
    `markers=True` is how an EMPTY folder can be seen at all: a week that
    holds nothing but its own marker is invisible to a listing that
    filters them out, which made a created-but-empty week indistinguish-
    able from one that was never created.

    `bucket` defaults to this app's own. The escalation datasets live in
    a separate bucket, read-only, so a source names the bucket it is in
    and the destination is always ours.
    """
    token = None
    while True:
        kwargs = {"Bucket": bucket or BUCKET, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/") and not markers:
                continue
            yield obj
        token = page.get("NextContinuationToken")
        if not page.get("IsTruncated"):
            break


def _subfolder_for(name: str, config: dict) -> str | None:
    lowered = name.lower()
    for ext in config.get("image_extensions") or []:
        if lowered.endswith(str(ext).lower()):
            return str(config.get("images_subpath") or "images").strip("/")
    for ext in config.get("label_extensions") or []:
        if lowered.endswith(str(ext).lower()):
            return str(config.get("labels_subpath") or "labels").strip("/")
    return None


def destination_bucket(config: dict | None = None) -> str:
    """Which bucket week folders are written to.

    Read from the settings rather than fixed to this app's bucket, so the
    built dataset can live beside whatever consumes it. Falls back to the
    app's own bucket, which is what every existing configuration means.
    """
    config = config if config is not None else read_config()
    return str(config.get("destination_bucket") or "").strip() or BUCKET


def source_bucket(source: dict) -> str:
    """Which bucket a source reads from; ours unless it says otherwise."""
    return str(source.get("bucket") or "").strip() or BUCKET


def _action_of(source: dict) -> str:
    """`copy` or `move` for one source.

    Copy is the default, and deliberately so. A drop folder that is really
    a staging area wants a move -- its emptiness is what "everything is
    filed" means. But pointing this at a folder that is ALSO a dataset,
    which is the easy mistake, empties it: the safe default is the one
    that cannot destroy the thing it was pointed at by accident.
    """
    return "move" if str(source.get("action") or "").strip().lower() == "move" else "copy"


def split_of(key: str) -> str:
    """`train`, `val` or `""` for a key already inside a split dataset.

    An Ultralytics dataset says which side a file is on by the segment
    directly after `images/` or `labels/`, so that is what is read --
    not a substring search, which would call `train2/images/val/x.jpg`
    training.

    This matters more than it looks. A source dataset that is already
    split was split by someone; merging the two sides and re-splitting
    later puts near-duplicate frames from one camera feed on both sides
    of the evaluation, and the resulting mAP is inflated and quiet about
    it. Preserving the split is the whole point of reading this.
    """
    parts = key.split("/")
    # Backwards: the nearest images/ or labels/ segment to the file is the
    # one that describes it.
    for index in range(len(parts) - 2, -1, -1):
        if parts[index].lower() in ("images", "labels"):
            following = parts[index + 1].lower()
            if following in ("train", "training"):
                return "train"
            if following in ("val", "valid", "validation"):
                return "val"
            return ""
    return ""


def splits_per_root(config: dict) -> dict:
    """`{root: {"train", "val"}}` -- which splits each root will receive.

    Answered across EVERY source rather than one at a time, because
    whether `retraining_training` needs a `train/` subfolder does not
    depend on the file being filed; it depends on whether anything else
    also files `val` into the same root.
    """
    out: dict = {}
    for source in config.get("sources") or []:
        training_root = str(source.get("destination") or "").strip().strip("/")
        eval_root = str(source.get("eval_destination") or "").strip().strip("/")
        split = str(source.get("split") or "").strip().lower()
        if split in ("train", "val"):
            root = eval_root if (split == "val" and eval_root) else training_root
            if root:
                out.setdefault(root, set()).add(split)
            continue
        # No split declared, so it is inferred per file and this source
        # can produce either. The training root therefore has to be
        # treated as receiving both, unless an evaluation root takes val.
        if training_root:
            out.setdefault(training_root, set()).add("train")
            if not eval_root:
                out[training_root].add("val")
        if eval_root:
            out.setdefault(eval_root, set()).add("val")
    return out


def placement(key: str, source: dict, config: dict,
              splits: dict | None = None) -> tuple | None:
    """`(root, subdir)` for a key, or None if it is not dataset content.

    A week folder ends up as:

        retraining_training/week_3/images/frame.jpg
        retraining_training/week_3/labels/frame.txt
        retraining_evaluation/week_3/images/frame.jpg
        retraining_evaluation/week_3/labels/frame.txt

    No `train/` or `val/` level, because the ROOT already says which one
    it is -- everything under `retraining_training` is training data, and
    `images/train/` under it would be saying so twice.

    The exception is the one case where it would not be saying it twice:
    when a single root receives both splits, dropping the subfolder would
    merge them into one folder. That is train/val leakage -- the
    held-out set stops being held out, and every metric measured against
    it becomes meaningless while still looking fine. So the split stays
    in the path exactly when one root has to carry both, and
    `splits_per_root` is what decides that.

    Which split a file is on comes from `source["split"]` when the folder
    was picked by hand -- the list it was put in IS the answer, and no
    manifest or filename convention is consulted. Failing that it is
    inferred from the path, for a source already split upstream.

    Nothing here reads a `data.yaml`: image or label comes from the file
    EXTENSION, and the root comes from the source. That is what lets raw
    folders be used with no manifest of any kind.

    `splits` is `splits_per_root(config)`, computed once by the caller
    because it is the same for every file in a sweep. Passing it is not
    an optimisation detail: it makes visible that where a file lands
    depends on the WHOLE configuration, not only on the source it came
    from -- another source filing `val` into this root is what decides
    whether a `train/` level is needed.
    """
    sub = _subfolder_for(key.rsplit("/", 1)[-1], config)
    if not sub:
        return None

    training_root = str(source.get("destination") or "").strip("/")
    eval_root = str(source.get("eval_destination") or "").strip("/")

    split = str(source.get("split") or "").strip().lower()
    if split not in ("train", "val"):
        # Only consulted when the source did not say. A hand-picked folder
        # must not be re-routed because `val` happens to be in its path.
        split = split_of(key)

    # An evaluation root only means anything when the split was inferred;
    # a hand-picked source points `destination` straight at the root it
    # belongs to and leaves this blank.
    root = eval_root if (split == "val" and eval_root) else training_root

    if splits is None:
        splits = splits_per_root(config)
    if split and len(splits.get(root, set())) > 1:
        return root, f"{sub}/{split}"
    return root, sub


def plan_routing(*, config: dict | None = None, limit: int | None = None) -> dict:
    """What would move, and where -- without moving anything.

    The same walk the move does, which is the point: a preview produced by
    different code is a preview of something else.
    """
    config = config or read_config()
    weeks = list_weeks()
    planned, skipped, would_create = [], [], []
    already: dict = {}
    by_scan = str(config.get("week_by") or "scan") == "scan"
    scan_week = None
    # The same for every file in this sweep -- see `placement`.
    splits = splits_per_root(config)
    # Weeks created during a preview would be a side effect of looking, so
    # the preview projects what it would need instead of creating it.
    pending_weeks = list(weeks)

    for source in config.get("sources") or []:
        prefix = str(source.get("prefix") or "").strip("/")
        if not prefix or not str(source.get("destination") or "").strip("/"):
            continue
        bucket = source_bucket(source)
        for obj in _list_objects(prefix + "/", bucket):
            # No cap by default, like the sweep it previews.
            if limit and len(planned) >= limit:
                break
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            where = placement(key, source, config, splits)
            if not where:
                skipped.append({"key": key, "reason": "not an image or a label"})
                continue
            root, sub = where
            when, _clock = file_moment(obj, config)

            if by_scan:
                # The same single week the sweep would choose. Projected,
                # not created -- a preview that laid down a week folder
                # would be doing rather than looking.
                if scan_week is None:
                    scan_week, made = project_scan_week(pending_weeks, config)
                    if made:
                        pending_weeks = pending_weeks + [made]
                        would_create.append(made["name"])
                week = scan_week
            else:
                week, projected = project_week(when, pending_weeks, config)
                if projected:
                    # Carried forward so a second late file sees the weeks
                    # the first would have created, and the preview numbers
                    # them the way the sweep will.
                    pending_weeks = pending_weeks + projected
                    would_create.extend(w["name"] for w in projected)
            if not week:
                skipped.append({
                    "key": key,
                    "reason": "older than week 1 -- no week covers it"
                    if config.get("auto_create_weeks")
                    else "past the last week, and new weeks are switched off",
                })
                continue

            destination = f"{root}/{week['name']}/{sub}/{name}"
            revision = False
            if not config.get("overwrite"):
                if root not in already:
                    already[root] = filed_fingerprints(root)
                seen = already[root].get(name)
                etag = obj.get("ETag", "").strip('"')
                if seen is not None:
                    if seen[0] == etag:
                        skipped.append({"key": key,
                                        "reason": "already filed under this root"})
                        continue
                    # Revised upstream: rewrite where it already sits.
                    destination, revision = seen[1], True
                already[root][name] = (etag, destination)
            planned.append({
                "key": key,
                "destination": destination,
                "week": week["name"],
                "new_week": week["name"] in would_create,
                "modified": when.isoformat(),
                "action": _action_of(source),
                # A corrected label landing on a file already in use, not
                # a new file. Worth showing apart in the preview.
                "revision": revision,
            })
    return {
        "planned": planned,
        "skipped": skipped,
        "weeks": weeks,
        # Named so the card can say "this will also create week_3" before
        # anyone presses the button that does it.
        "would_create": sorted(set(would_create)),
    }


def _exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError:
        return False


def claim_once(what: str, key: str, *, days: int = 7) -> bool:
    """True for the FIRST caller to claim `key`, False for every other.

    A conditional put, which is the only way two processes can agree on
    who acts. The development and stable containers share this table and
    both run the same poll tick, so "has this already been handled?"
    answered by read-then-write has a window between the two in which
    both read "no" -- and the thing that follows is a GPU training run.
    Two runs from one pull, serialised on the single slot, is an hour of
    compute spent producing a duplicate of a model nobody asked for.

    Expires, because a claim is a race guard and not a record: the thing
    it protects has its own durable marker written afterwards.
    """
    try:
        ddb.put_item(
            TableName=TABLE_NAME,
            Item={
                "kind": {"S": f"claim#{what}"},
                "sort": {"S": str(key)},
                "claimed_at": {"S": _now().isoformat()},
                "expires_at": {"N": str(int((_now() + _dt.timedelta(days=days)).timestamp()))},
            },
            ConditionExpression="attribute_not_exists(#k)",
            ExpressionAttributeNames={"#k": "kind"},
        )
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        # Anything else is a real failure. Returning True would be the
        # safe-looking choice and is the wrong one: it converts an
        # unavailable table into two concurrent training runs.
        print(json.dumps({"pull_control": "claim_failed", "what": what,
                          "key": str(key), "error": str(exc)}))
        return False


def week_definition_problems(config: dict | None = None) -> list:
    """Week DEFINITIONS that will route files somewhere surprising.

    Two kinds, both seen live on 2026-10-06:

    * A name that does not start the configured `week_prefix` --
      `day_1` survived an experiment with a different prefix. It still
      covers a date range, so the newest captures route into a folder
      whose name contradicts every other folder and which nothing
      downstream expects to find.

    * A GAP between one week's end and the next one's start. Files
      captured in the gap have no week. `auto_create_weeks` papers over
      it by inventing one at sweep time, so the numbering drifts further
      from the dates with every pull rather than being fixed once.

    Reported, never repaired automatically: a week defines where data
    LANDS, and silently moving that boundary would re-file a corpus
    somebody has already trained on.
    """
    config = config or read_config()
    prefix = str(config.get("week_prefix") or "week_")
    length = int(config.get("week_length_days") or 7)
    weeks = list_weeks()
    out = []

    for week in weeks:
        name = str(week.get("name") or "")
        if name and not name.startswith(prefix):
            out.append({
                "kind": "prefix",
                "week": name,
                "detail": f"{name} does not start with {prefix!r}, so files whose "
                          f"dates fall in it land outside the week folders "
                          f"everything else reads.",
            })

    previous = None
    for week in weeks:
        try:
            starts = _parse_time(week.get("starts_at") or "")
        except ValueError:
            continue
        if previous is not None:
            ends = previous[1] + _dt.timedelta(days=length)
            if starts > ends:
                out.append({
                    "kind": "gap",
                    "week": f"{previous[0]} -> {week.get('name')}",
                    "detail": f"No week covers {ends.date()} to {starts.date()}. "
                              f"Files captured then have nowhere to go.",
                })
        previous = (str(week.get("name") or ""), starts)
    return out


def claim_exists(what: str, key: str) -> bool:
    """Whether `key` has already been claimed.

    The read half of `claim_once`, so a caller can ASK without taking the
    claim -- which is what a status card wants: "will this act on the
    next tick" has to be answerable without acting.

    Consistent read, because the question is being asked seconds after
    the write that answers it.
    """
    if not key:
        return False
    try:
        got = ddb.get_item(
            TableName=TABLE_NAME,
            Key={"kind": {"S": f"claim#{what}"}, "sort": {"S": str(key)}},
            ConsistentRead=True,
        ).get("Item")
    except ClientError as exc:
        # Unknown, and "not claimed" is the answer that starts GPU work.
        # Say claimed; the worst case is one skipped chain, against a
        # duplicate training run for the other answer.
        print(json.dumps({"pull_control": "claim_read_failed", "what": what,
                          "key": str(key), "error": str(exc)}))
        return True
    return bool(got)


def orphan_week_folders(config: dict | None = None) -> list:
    """Week folders holding images that NO week definition names.

    These are invisible everywhere else and silently excluded from
    training. `list_weeks` reads the definitions in DynamoDB, the
    data.yaml is built from those, and the trainer only ever sees what
    the yaml lists -- so a folder in S3 with no matching definition holds
    images that were pulled, are paying for storage, and will never be
    learned from. Nothing reports that, because every component is
    working from the definitions and the folder is simply not in them.

    Seen live: `retraining/week_4` held a filed image while the
    definitions listed weeks 1, 2, 3, 6 and 7.

    Returns `{"name", "images"}` per folder, newest-looking first.
    """
    config = config or read_config()
    root = str(config.get("root") or "").strip("/")
    if not root:
        return []
    known = {str(week.get("name") or "") for week in list_weeks()}
    prefix = str(config.get("week_prefix") or "week_")
    into = destination_bucket(config)
    extensions = config.get("image_extensions") or DEFAULTS["image_extensions"]

    counts: dict = {}
    for obj in _list_objects(root + "/", into):
        rest = obj["Key"][len(root) + 1:]
        name, _, tail = rest.partition("/")
        if not tail or not name.startswith(prefix) or name in known:
            continue
        if "/images/" not in "/" + tail:
            continue
        if not any(tail.lower().endswith(str(ext).lower()) for ext in extensions):
            continue
        counts[name] = counts.get(name, 0) + 1

    def order(name: str):
        digits = "".join(ch for ch in name if ch.isdigit())
        return (int(digits) if digits else 0, name)

    return [{"name": name, "images": count}
            for name, count in sorted(counts.items(), key=lambda kv: order(kv[0]), reverse=True)]


def filed_fingerprints(root: str) -> dict:
    """`{filename: (etag, key)}` for everything already filed under a root.

    The key as well as the ETag, so a file whose bytes changed upstream
    can be rewritten WHERE IT ALREADY SITS. Filing the corrected version
    into the current week instead would leave the stale copy in its old
    week and the dataset would hold the same image twice with different
    labels -- which is worse than either version alone.

    The ETag as well as the name, because the source is edited in place:
    an alarm that is re-reviewed keeps its filename and gets a new label.
    Measured over five days of the live dataset that is about 1 file in
    700 -- rare, and completely silent if only names are compared, so the
    week folder would keep a label that has since been corrected.

    Costs nothing: list_objects_v2 returns ETags already, so this is the
    same single listing either way, and far cheaper than a HeadObject per
    file.
    """
    out: dict = {}
    for obj in _list_objects(root.strip("/") + "/", destination_bucket()):
        out[obj["Key"].rsplit("/", 1)[-1]] = (obj.get("ETag", "").strip('"'), obj["Key"])
    return out


def filed_names(root: str) -> set:
    """Every filename already filed under ANY week of a root.

    Across weeks, not just the one being written, and that distinction is
    the whole point. A source that publishes CUMULATIVE snapshots -- each
    pull containing everything before it plus an increment, which is what
    track2_pull does -- re-presents its entire history on every sweep. A
    per-week check sees `week_2/images/x.jpg` missing, copies it, and does
    the same again into week_3: every week becomes a full copy of the
    dataset and storage grows with the square of the weeks.

    Checked by name against the whole root instead, a week holds only what
    was genuinely new that week, which is what "later weeks take what
    arrives after them" has to mean for a cumulative source.

    One listing rather than a HeadObject per file, which is also the
    difference between a sweep that takes seconds and one that takes an
    hour.
    """
    names = set()
    for obj in _list_objects(root.strip("/") + "/"):
        names.add(obj["Key"].rsplit("/", 1)[-1])
    return names


def count_pending(config: dict | None = None, *, limit: int | None = None) -> int:
    """How many objects a sweep will walk. The progress denominator.

    A LIST-only pass, which is two or three calls for a few thousand
    keys -- cheap next to the per-object work that follows, and the only
    way a percentage can exist at all: without a total, progress is a
    running count with nothing to be a fraction of.
    """
    config = config or read_config()
    total = 0
    for source in config.get("sources") or []:
        prefix = str(source.get("prefix") or "").strip("/")
        if not prefix:
            continue
        for _ in _list_objects(prefix + "/", source_bucket(source)):
            total += 1
            # Uncapped by default, like the sweep it is the total for.
            if limit and total >= limit:
                return total
    return total


def unpaired_sides(config: dict | None = None) -> list:
    """Sides that would collect images and no labels: `["train"]`, say.

    An Ultralytics label is matched to its image by STEM, so the two can
    be picked from different folders -- but a side with image folders and
    no label folder at all collects pictures nobody annotated. The trainer
    takes those as BACKGROUND examples, which is a real thing and useful
    in small numbers, so nothing downstream errors: a whole week of them
    simply trains the model on nothing, quietly, and the first sign is a
    metric that will not move.

    A few images with no matching label is healthy and NOT reported here;
    this is only the unambiguous case of no label folder being picked.
    """
    config = config or read_config()
    picked = config.get("picked") or {}
    out = []
    for side in ("train", "val"):
        images = [p for p in (picked.get(f"{side}_images") or []) if str(p).strip()]
        labels = [p for p in (picked.get(f"{side}_labels") or []) if str(p).strip()]
        if images and not labels:
            out.append(side)
    return out


def route_pending(*, config: dict | None = None, limit: int | None = None,
                  trigger: str = "manual", run_id: str = "") -> dict:
    """Moves every waiting file into the week folder its time says.

    A move is a copy then a delete, in that order: the drop folder is the
    record that a file has not been filed yet, so deleting first and
    failing the copy would lose it outright.

    With a `run_id`, progress is published every PROGRESS_EVERY files and
    the stop flag is re-read at the same moment -- so a sweep can be
    watched and interrupted from anywhere, including a Lambda that
    nothing can kill from outside.
    """
    config = config or read_config()

    # Refused here rather than warned about in the card, because the sweep
    # that matters is the one nobody is watching: a scheduled pull at 18:33
    # that files a week of unlabelled images reports success, and the first
    # sign of trouble is a retrain whose metrics will not move. The card
    # shows the same thing, but the card is not what runs at 18:33.
    unpaired = unpaired_sides(config)
    if unpaired:
        where = " and ".join(unpaired)
        raise ValueError(
            f"The {where} side has image folders picked and no label folder. "
            f"Those images would be filed as background examples -- the trainer "
            f"accepts them and learns nothing from them. Pick the {where} label "
            f"folder, or remove the images, before pulling."
        )

    weeks = list_weeks()
    moved, skipped, errors, revised = [], [], [], []
    aborted = ""
    # Destinations this run CREATED -- new keys only. A revised
    # destination is excluded: the file was already there and this run
    # only rewrote it, so undoing the run must not delete it.
    created: list = []
    # Files whose name carried no timestamp, so their upload time was used
    # instead. Counted rather than ignored: a source that does not follow
    # the convention should be visible, not merely survivable.
    fell_back = 0
    seen_files = 0
    limit_reached = False
    total_files = count_pending(config) if run_id else 0
    if run_id:
        write_run_state(run_id, status="running", trigger=trigger,
                        total=total_files, done=0, moved=0, skipped=0, errors=0,
                        started_at=_now().isoformat(), stop_requested=False)
    # root -> names already filed under it, read once and kept current as
    # this sweep files more, so a name repeated within one sweep is caught
    # as well as one repeated across weeks.
    already: dict = {}

    by_scan = str(config.get("week_by") or "scan") == "scan"
    scan_week = None
    splits = splits_per_root(config)
    into = destination_bucket(config)

    for source in config.get("sources") or []:
        prefix = str(source.get("prefix") or "").strip("/")
        if not prefix or not str(source.get("destination") or "").strip("/"):
            continue
        action = _action_of(source)
        bucket = source_bucket(source)
        for obj in _list_objects(prefix + "/", bucket):
            # NO CAP by default (2026-10-26): a sweep files everything
            # waiting. The old 5,000 stopped a 5,240-file pull short and
            # left the last 240 -- the val labels, last in the source
            # list -- under images that had landed. And it counted skips,
            # so a copy source re-skipped the same 5,000 every sweep and
            # never reached the rest.
            #
            # The ceiling that remains is AWS's: the pull Lambda is
            # stopped at 15 minutes (~10,000 copies at the measured 12 a
            # second). A sweep cut off there is reported `lost`, and the
            # next one carries on -- copies already filed are skipped.
            # `limit` is still accepted for tests and callers that want a
            # bounded run; it counts files COPIED, never files skipped.
            if limit and len(moved) >= limit:
                limit_reached = True
                break
            seen_files += 1
            if run_id and seen_files % PROGRESS_EVERY == 0:
                # Publish, then ask whether to stop. In that order: the
                # numbers the card shows when it stops should be the
                # numbers that were true when it stopped.
                write_run_state(run_id, done=seen_files, moved=len(moved),
                                skipped=len(skipped), errors=len(errors),
                                total=total_files)
                _write_manifest(run_id, created)
                if stop_requested(run_id):
                    aborted = "stopped by hand"
                    break
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            where = placement(key, source, config, splits)
            if not where:
                skipped.append({"key": key, "reason": "not an image or a label"})
                continue
            root, sub = where

            if by_scan:
                # One week for the whole sweep, chosen once.
                if scan_week is None:
                    scan_week, weeks = week_for_scan(weeks, config)
                week = scan_week
            else:
                when, clock = file_moment(obj, config)
                if clock == "fallback":
                    fell_back += 1
                week, weeks = ensure_week_covering(when, weeks, config)
            if not week:
                skipped.append({"key": key, "reason": "no week covers it"})
                continue

            destination = f"{root}/{week['name']}/{sub}/{name}"
            if bucket == into and destination == key:
                continue
            if not config.get("overwrite"):
                if root not in already:
                    already[root] = filed_fingerprints(root)
                seen = already[root].get(name)
                if seen is not None:
                    if seen[0] == obj.get("ETag", "").strip('"'):
                        skipped.append({"key": key, "reason": "already filed under this root"})
                        continue
                    # Same name, different bytes: revised upstream. Rewrite
                    # it where it already sits, so the week that owns it
                    # keeps owning it and no second copy appears.
                    destination = seen[1]
                    revised.append({"key": key, "destination": destination})
            try:
                s3.copy_object(
                    Bucket=into,
                    Key=destination,
                    CopySource={"Bucket": bucket, "Key": key},
                )
            except ClientError as exc:
                # Nothing was written, so the next file is worth trying.
                errors.append({"key": key, "error": f"could not copy: {exc}"})
                continue

            if action == "copy":
                # The original is the system of record and stays put. The
                # names here are content hashes, so running again finds the
                # copy already in place and skips it -- a copy is idempotent
                # in a way a move never needs to be.
                already.setdefault(root, {})[name] = (
                    obj.get("ETag", "").strip('"'), destination)
                moved.append({"key": key, "destination": destination,
                              "week": week["name"], "action": "copy"})
                # Only a NEW key. A revised one was already there.
                if not any(r["destination"] == destination for r in revised):
                    created.append(destination)
                continue

            try:
                s3.delete_object(Bucket=bucket, Key=key)
            except ClientError as exc:
                # The copy succeeded, so this file now exists in BOTH
                # places. Whatever stopped the delete -- most likely a
                # missing s3:DeleteObject -- will stop every remaining one
                # the same way, and carrying on would duplicate the entire
                # drop folder rather than move it. One duplicate is
                # recoverable; thousands is a mess. Stop here.
                errors.append({
                    "key": key,
                    "error": f"copied to {destination}, but the original could not be "
                             f"removed, so it now exists in both places: {exc}",
                })
                aborted = (
                    "Stopped after the first file that could not be removed from the drop "
                    "folder. Everything before it moved; nothing after it was touched."
                )
                break

            already.setdefault(root, {})[name] = (
                obj.get("ETag", "").strip('"'), destination)
            moved.append({"key": key, "destination": destination,
                          "week": week["name"], "action": "move"})
            # NOT added to `created`. A move deleted the original, so the
            # destination is now the only copy of that file -- deleting it
            # on an undo would destroy it outright, which is the opposite
            # of undoing. `undo_run` refuses outright when a run moved
            # anything; see `moves` in the summary below.
        if aborted:
            break

    moves = sum(1 for m in moved if m.get("action") == "move")
    summary = {
        "moved": len(moved),
        "skipped": len(skipped),
        "errors": len(errors),
        "trigger": trigger,
        "at": _now().isoformat(),
        "aborted": aborted,
        # Counted apart from `moved`: a revision is a correction landing on
        # a file already in use, which is worth seeing rather than hiding
        # inside a copy count.
        "revised": len(revised),
        "run_id": run_id,
        "dated_by": str(config.get("week_by") or "scan"),
        "fell_back_to_upload_time": fell_back,
        # The sweep stopped at its cap with files still waiting; the next
        # one carries on from where this one left off.
        "limit_reached": limit_reached,
    }

    stopped = aborted == "stopped by hand"
    undone = None
    if run_id:
        _write_manifest(run_id, created)
        if stopped:
            # Stopping undoes the run, which is what was asked for -- but
            # only where undoing is possible. A run that MOVED files
            # cannot be undone: the originals are gone, so deleting the
            # destinations would destroy the only copies. Reported rather
            # than half-done.
            if moves:
                undone = {
                    "run_id": run_id, "created": len(created), "removed": 0,
                    "failed": [],
                    "refused": f"{moves} file(s) were MOVED, not copied. Their "
                               f"originals are already deleted, so removing the "
                               f"copies would destroy them. Nothing was undone.",
                }
            else:
                undone = undo_run(run_id)
            summary["undone"] = undone

        write_run_state(
            run_id,
            status="stopped" if stopped else ("failed" if aborted else "done"),
            done=seen_files, total=total_files, moved=len(moved),
            skipped=len(skipped), errors=len(errors),
            aborted=aborted, finished_at=_now().isoformat(),
            undone=undone,
        )

    if not aborted and moved:
        # Once the sweep knows what landed where, the holes between the
        # weeks that received files are known too. Filled in here rather
        # than by the card, so a scheduled pull leaves the same tree a
        # manual one does.
        try:
            summary["gaps_created"] = create_gap_folders(config)["created"]
        except Exception:                                   # noqa: BLE001
            summary["gaps_created"] = []

    # Here rather than in the router, because this is the one line both
    # triggers reach: a manual Pull now runs it in the app container, and
    # the schedule and arrival rules run it in the Lambda. Putting it in
    # the router would have notified for exactly the pulls somebody was
    # already watching, and stayed silent for every unattended one.
    _notify_pull_finished(summary, stopped=stopped, aborted=aborted)

    _record_run(summary, moved, skipped, errors)
    return {**summary, "moved_files": moved[:200], "skipped_files": skipped[:200],
            "error_files": errors[:50]}


# --------------------------------------------------------------------------
# Watching a sweep while it runs
#
# A sweep used to be a single blocking call that reported only when it was
# over: no progress, and no way to stop it once it had started on four
# thousand files. Both need the same thing -- state the sweep writes as it
# goes and anything else can read.
#
# The state lives in the settings table; the list of keys a run CREATED
# lives in S3 beside the data, because it can run to thousands of entries
# and a DynamoDB item stops at 400 KB.
# --------------------------------------------------------------------------

#: How often the sweep pauses to publish progress and re-read the stop
#: flag. Every file would be a DynamoDB round trip per image; every
#: hundred would mean Stop takes a visible while to bite. Twenty-five is
#: roughly a second of copying.
PROGRESS_EVERY = 25

#: A run that has not published progress for this long is presumed dead --
#: its container was replaced, or the Lambda hit its timeout. Without this
#: a crashed sweep leaves a row that says "running" for ever and the card
#: offers a Stop button for something that stopped days ago.
RUN_STALE_SECONDS = 180

STATE_PREFIX = "_pull_state"


def _manifest_key(run_id: str) -> str:
    """Where the keys a run created are listed, for undoing it."""
    return f"{STATE_PREFIX}/{run_id}.json"


def new_run_id() -> str:
    return f"{_now().strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"


def _notify_pull_finished(summary: dict, *, stopped: bool = False,
                          aborted: bool = False) -> None:
    """Emails the subscribers that a sweep has ended.

    QUIET SWEEPS DO NOT NOTIFY. A schedule that fires daily and finds
    nothing would otherwise send a daily "0 files", which is the fastest
    way to teach somebody to filter the address -- and then the one that
    mattered goes unread too. A run that moved files, failed, or was
    stopped is worth saying out loud; one that found nothing is not.

    Never raises. A pull that actually moved files has done its job, and
    failing it because SNS was unreachable would turn a notification
    problem into a data problem.
    """
    if not NOTIFY_TOPIC_ARN:
        return
    moved = int(summary.get("moved") or 0)
    errors = int(summary.get("errors") or 0)
    if not (moved or errors or stopped or aborted):
        return

    trigger = str(summary.get("trigger") or "pull")
    if aborted:
        headline = "Pull FAILED"
    elif stopped:
        headline = "Pull stopped"
    else:
        headline = f"Pull finished: {moved} file(s) filed"

    lines = [
        headline,
        "",
        f"Trigger:  {trigger}",
        f"Run:      {summary.get('run_id') or '(none)'}",
        f"Moved:    {moved}",
        f"Skipped:  {summary.get('skipped') or 0}",
        f"Errors:   {errors}",
    ]
    weeks = summary.get("weeks_touched") or summary.get("gaps_created")
    if weeks:
        lines.append(f"Weeks:    {', '.join(str(w) for w in weeks[:10])}")
    try:
        extra = {}
        if aborted or errors:
            # Labelled an ERROR, so it reaches whoever ticked "Errors" on
            # the Notifications page. An ordinary pull summary carries no
            # label and reaches everyone, as it always has.
            from . import notify

            extra["MessageAttributes"] = notify.attributes(["error"])
        boto3.client("sns", region_name=REGION).publish(
            TopicArn=NOTIFY_TOPIC_ARN,
            Subject=headline[:100],
            Message=chr(10).join(lines),
            **extra,
        )
    except Exception as exc:                                # noqa: BLE001
        # Logged, not raised. Visible in the Lambda's log and the app's
        # stdout, so a permissions problem is findable without being a
        # failed pull.
        print(json.dumps({"pull_control": "notify_failed", "error": str(exc)}))


def write_run_state(run_id: str, **fields) -> dict:
    """Merges fields into a run's state row."""
    state = read_run_state(run_id) or {"run_id": run_id}
    state.update(fields)
    state["updated_at"] = _now().isoformat()
    ddb.put_item(
        TableName=TABLE_NAME,
        Item={
            "kind": {"S": "run_state"},
            "sort": {"S": run_id},
            "body": {"S": json.dumps(state)},
            "expires_at": {"N": str(int((_now() + _dt.timedelta(days=2)).timestamp()))},
        },
    )
    return state


def read_run_state(run_id: str) -> dict | None:
    try:
        got = ddb.get_item(
            TableName=TABLE_NAME,
            Key={"kind": {"S": "run_state"}, "sort": {"S": run_id}},
            ConsistentRead=True,
        ).get("Item")
    except ClientError:
        return None
    if not got or "body" not in got:
        return None
    try:
        return json.loads(got["body"]["S"])
    except ValueError:
        return None


def request_stop(run_id: str) -> dict:
    """Asks a sweep to stop. It notices within PROGRESS_EVERY files.

    A flag rather than killing anything: the sweep is mid-copy somewhere,
    and the only safe moment to stop is between files. It works for the
    Lambda too, which cannot be killed from outside at all.
    """
    state = read_run_state(run_id)
    if not state:
        raise ValueError(f"No run called {run_id}.")
    if state.get("status") != "running":
        raise ValueError(f"That run is already {state.get('status')}.")
    return write_run_state(run_id, stop_requested=True)


def stop_requested(run_id: str) -> bool:
    state = read_run_state(run_id) if run_id else None
    return bool(state and state.get("stop_requested"))


def _write_manifest(run_id: str, created: list) -> None:
    """The keys this run CREATED, for undoing it.

    Only keys that did not exist before. A `revised` destination -- a file
    whose bytes changed upstream, rewritten where it already sat -- is
    deliberately excluded: this run did not create it, and deleting it on
    an undo would destroy data the run only updated.
    """
    try:
        s3.put_object(
            Bucket=destination_bucket(), Key=_manifest_key(run_id),
            Body=json.dumps({"run_id": run_id, "created": created}).encode("utf-8"),
            ContentType="application/json",
        )
    except ClientError:
        # A manifest that cannot be written costs an undo, not the sweep.
        pass


def read_manifest(run_id: str) -> list:
    try:
        body = s3.get_object(Bucket=destination_bucket(),
                             Key=_manifest_key(run_id))["Body"].read()
        return list(json.loads(body).get("created") or [])
    except Exception:
        return []


def undo_run(run_id: str) -> dict:
    """Deletes what a run created, and nothing else.

    Driven by the manifest rather than by "everything under the week
    folder": the week may hold files from earlier runs, and an undo that
    cleared the folder would take them too.

    Each key is confirmed gone before it is counted. A delete that fails
    is reported by name rather than swallowed -- a partial undo that
    claimed success would leave the destination in a state nobody could
    reason about, which is the one thing worth being loud about here.
    """
    created = read_manifest(run_id)
    into = destination_bucket()
    removed, failed = 0, []
    if not created:
        return {"run_id": run_id, "created": 0, "removed": 0, "failed": [],
                "refused": "Nothing recorded as created by that run."}
    for key in created:
        try:
            s3.delete_object(Bucket=into, Key=key)
        except ClientError as exc:
            failed.append({"key": key, "error": str(exc)})
            continue
        try:
            s3.head_object(Bucket=into, Key=key)
        except ClientError:
            removed += 1          # gone, as intended
            continue
        failed.append({"key": key, "error": "still present after delete"})

    if not failed:
        try:
            s3.delete_object(Bucket=into, Key=_manifest_key(run_id))
        except ClientError:
            pass
    return {"run_id": run_id, "created": len(created),
            "removed": removed, "failed": failed}


def active_run() -> dict | None:
    """The sweep that is running now, if one is.

    Anything that has gone quiet for longer than RUN_STALE_SECONDS is
    reported as `lost` rather than `running`: its container was replaced
    or its Lambda timed out, and offering Stop for it would be a button
    that does nothing.
    """
    try:
        rows = ddb.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="#k = :k",
            ExpressionAttributeNames={"#k": "kind"},
            ExpressionAttributeValues={":k": {"S": "run_state"}},
            ScanIndexForward=False,
            Limit=5,
        ).get("Items") or []
    except ClientError:
        return None

    for row in rows:
        try:
            state = json.loads(row["body"]["S"])
        except (KeyError, ValueError):
            continue
        if state.get("status") != "running":
            continue
        try:
            quiet = (_now() - _parse_time(state.get("updated_at") or "")).total_seconds()
        except ValueError:
            quiet = 0.0
        if quiet > RUN_STALE_SECONDS:
            state["status"] = "lost"
            state["quiet_seconds"] = int(quiet)
        return state
    return None


def pending_summary(*, config: dict | None = None, limit: int | None = None) -> dict:
    """What is waiting to be pulled, and which week each file would land in.

    The same decisions a sweep makes -- placement, week, dedup -- but
    counting only, so it can be shown continuously rather than only when
    someone presses Preview. Nothing is copied, created or written.

    It answers the question a scheduled pull otherwise leaves unanswered:
    how much is sitting there, and where will it go when the timer fires.
    """
    config = config or read_config()
    weeks = list_weeks()
    splits = splits_per_root(config)
    by_scan = str(config.get("week_by") or "scan") == "scan"
    scan_week = None
    pending_weeks = list(weeks)

    by_week: dict = {}
    waiting = 0
    fell_back = 0
    already_filed = 0
    ignored = 0
    scanned = 0
    capped = False
    already: dict = {}

    for source in config.get("sources") or []:
        prefix = str(source.get("prefix") or "").strip("/")
        if not prefix or not str(source.get("destination") or "").strip("/"):
            continue
        bucket = source_bucket(source)
        for obj in _list_objects(prefix + "/", bucket):
            scanned += 1
            if limit and scanned > limit:
                capped = True
                break
            key = obj["Key"]
            name = key.rsplit("/", 1)[-1]
            where = placement(key, source, config, splits)
            if not where:
                ignored += 1
                continue
            root, sub = where

            if by_scan:
                if scan_week is None:
                    scan_week, made = project_scan_week(pending_weeks, config)
                    if made:
                        pending_weeks = pending_weeks + [made]
                week = scan_week
            else:
                when, clock = file_moment(obj, config)
                if clock == "fallback":
                    fell_back += 1
                week, projected = project_week(when, pending_weeks, config)
                if projected:
                    pending_weeks = pending_weeks + projected
            if not week:
                ignored += 1
                continue

            if not config.get("overwrite"):
                if root not in already:
                    already[root] = filed_fingerprints(root)
                seen = already[root].get(name)
                if seen is not None and seen[0] == obj.get("ETag", "").strip('"'):
                    already_filed += 1
                    continue

            waiting += 1
            row = by_week.setdefault(week["name"], {
                "week": week["name"], "files": 0, "bytes": 0, "exists": False,
            })
            row["files"] += 1
            row["bytes"] += obj.get("Size", 0)
        if capped:
            break

    known = {w["name"] for w in weeks}
    for name, row in by_week.items():
        # A week that does not exist yet will be created by the pull. Worth
        # distinguishing from one that is merely gaining files.
        row["exists"] = name in known

    def order(row):
        digits = "".join(ch for ch in row["week"] if ch.isdigit())
        return (int(digits) if digits else 0, row["week"])

    return {
        "waiting": waiting,
        "dated_by": str(config.get("week_by") or "scan"),
        # Files whose name carries no timestamp. Zero on a source that
        # uses the convention; anything else says the source does not.
        "fell_back_to_upload_time": fell_back,
        "already_filed": already_filed,
        "ignored": ignored,
        "scanned": scanned,
        "capped": capped,
        "weeks": sorted(by_week.values(), key=order),
        "checked_at": _now().isoformat(),
    }


def _record_run(summary: dict, moved: list, skipped: list, errors: list) -> None:
    """One row per sweep, not per file.

    A sweep that files 4,000 images is one thing that happened; 4,000 rows
    describing it would cost more to write than the move itself and would
    bury the run that actually went wrong.
    """
    if not moved and not skipped and not errors and summary["trigger"] == "schedule":
        # A scheduled sweep that found nothing is the normal state. Writing
        # it every time turns the history into a wall of empty runs.
        return
    expires = int((_now() + _dt.timedelta(days=EVENT_TTL_DAYS)).timestamp())
    body = {
        **summary,
        "sample_moved": [m["destination"] for m in moved[:20]],
        "sample_skipped": skipped[:10],
        "sample_errors": errors[:5],
    }
    try:
        ddb.put_item(
            TableName=TABLE_NAME,
            Item={
                "kind": {"S": "event"},
                "sort": {"S": f"{summary['at']}#{uuid.uuid4().hex[:8]}"},
                "body": {"S": json.dumps(body)},
                "expires_at": {"N": str(expires)},
            },
        )
    except ClientError:
        # History is for looking at afterwards. Losing a row is not worth
        # failing a sweep that has already moved the files.
        pass


def recent_run_states(limit: int = 20) -> list:
    """Every sweep's final state, newest first.

    Unlike `recent_runs` -- which reads the `event` rows -- this is
    written for EVERY sweep, unconditionally, at the end of the run.
    `_record_run` deliberately skips a scheduled sweep that moved
    nothing, to keep the history from becoming a wall of empty rows; the
    consequence is that the event rows cannot see the very case a
    "queue even when nothing moved" option exists for.

    These rows carry `status` and `finished_at`, which the event rows do
    not, so a caller can tell a finished sweep from one in flight.

    Run ids are `<UTC timestamp>-<hex>`, so a descending key scan is
    newest-first without a sort.
    """
    try:
        page = ddb.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="#k = :k",
            ExpressionAttributeNames={"#k": "kind"},
            ExpressionAttributeValues={":k": {"S": "run_state"}},
            ScanIndexForward=False,
            Limit=max(1, min(int(limit), 100)),
        )
    except ClientError:
        return []
    out = []
    for item in page.get("Items", []):
        try:
            out.append(json.loads(item["body"]["S"]))
        except (ValueError, KeyError):
            continue
    return out


def recent_runs(limit: int = 20) -> list:
    """The last sweeps, newest first."""
    try:
        page = ddb.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="#k = :k",
            ExpressionAttributeNames={"#k": "kind"},
            ExpressionAttributeValues={":k": {"S": "event"}},
            ScanIndexForward=False,
            Limit=max(1, min(int(limit), 100)),
        )
    except ClientError:
        return []
    out = []
    for item in page.get("Items", []):
        try:
            out.append(json.loads(item["body"]["S"]))
        except (ValueError, KeyError):
            continue
    return out


# --------------------------------------------------------------------------
# The pull queue
# --------------------------------------------------------------------------
#
# Lines up sweeps so a second one waits for the first instead of running
# beside it. Two sweeps at once do not corrupt anything -- the week a file
# lands in is arithmetic, and `create_week` is a conditional put -- but they
# duplicate every copy, race the delete so the loser records errors that
# did not happen, and write overlapping undo manifests, which is the one
# with teeth: a pull MOVES, so undoing either run deletes a file nothing
# else has a copy of.
#
# WHY AN ENTRY CARRIES ITS OWN SETTINGS. A sweep reads the live config and
# moves whatever is waiting, so two entries that both said "sweep now"
# would be the same instruction twice -- the second would find an empty
# folder. Snapshotting the settings is what makes an entry a piece of work
# rather than a repeat: queue a pull of one set of folders, then another of
# a different set, and walk away.
#
# It also makes the queue honest about time. The settings that run are the
# ones that were on screen when the entry was added, not whatever the card
# happens to hold when its turn comes round.
#
# This lives in THIS file, not in track2_shared, because this file is what
# Terraform zips as the router Lambda -- stdlib and boto3, nothing else.
# A queue the Lambda cannot read is a queue that stops the moment the app
# container does, which is the whole thing it exists to survive.

PULL_QUEUE_KIND = "pull_queue"

#: Entries older than this are dropped rather than run. A sweep queued a
#: week ago was queued against folders that have since been pulled by
#: something else, and running it then is not catching up -- it is acting
#: on a decision nobody remembers making.
PULL_QUEUE_TTL_DAYS = 7


#: Breaks ties the clock cannot.
#:
#: The sort key was `added_at#entry_id`, on the assumption that two
#: entries never share a timestamp. They do: the clock has ~15ms
#: granularity, so entries queued in the same tick got the SAME
#: `added_at` and the tiebreak fell to the random hex in the entry id --
#: a queue that called itself FIFO and was not. `time.time_ns()` was
#: measured and is equally coarse, so a finer clock was never the answer.
#:
#: The first fix was `itertools.count()`, which was worse than useless
#: HERE: every Lambda execution environment starts its own count at zero,
#: and an arrival burst is many invocations. Five cold starts queuing in
#: one clock tick came back `second, fifth, fourth, third, first` --
#: scrambled in exactly the case the queue exists for.
#:
#: So the counter has to be shared, which means the table. One atomic ADD
#: per enqueue, which is cheap because enqueuing is rare: a handful per
#: burst, not per file.
_PULL_QUEUE_COUNTER_SORT = "pull_queue_seq"

#: Only reached when the counter is unavailable. Keeps enqueuing working
#: rather than failing the sweep that was trying to queue itself.
_pull_queue_fallback_seq = itertools.count()


def _next_pull_queue_seq() -> int:
    """A number that always goes up, across every process.

    `ADD` on one row is atomic in DynamoDB, so two Lambdas and the
    container cannot be handed the same value however closely they race.

    No `expires_at` on this row, unlike the queue entries: a counter that
    expired would restart at one and sort new entries ahead of old ones.
    """
    try:
        got = ddb.update_item(
            TableName=TABLE_NAME,
            Key={"kind": {"S": "counter"}, "sort": {"S": _PULL_QUEUE_COUNTER_SORT}},
            UpdateExpression="ADD #v :one",
            ExpressionAttributeNames={"#v": "value"},
            ExpressionAttributeValues={":one": {"N": "1"}},
            ReturnValues="UPDATED_NEW",
        )
        return int(got["Attributes"]["value"]["N"])
    except ClientError as exc:
        # Degraded, not broken: ordering within this process still holds,
        # which is the same guarantee the counter alone gave.
        print(f"Pull queue counter unavailable, falling back: {exc}")
        return next(_pull_queue_fallback_seq)


def _pull_queue_sort(added_at: str, entry_id: str, seq: int) -> str:
    """FIFO by construction: time first, then a sequence shared by every
    process, so a query returns the queue in the order it was added
    without re-sorting in memory.

    Time still leads so the key reads chronologically in the console, and
    because a later enqueue never has an earlier timestamp. The sequence
    only decides entries the clock could not separate.
    """
    return f"{added_at}#{seq:012d}#{entry_id}"


def enqueue_pull(*, config: dict | None = None, note: str = "",
                 added_by: str = "") -> dict:
    """Adds one sweep to the queue, with the settings it should run under.

    The snapshot is taken HERE rather than when the entry runs, so what
    was queued is what runs. A caller passing no config gets the live one
    snapshotted, which is what the card's "queue this" button means.
    """
    snapshot = dict(config or read_config())
    entry_id = new_run_id()
    added_at = _now().isoformat()
    seq = _next_pull_queue_seq()
    entry = {
        "id": entry_id,
        "added_at": added_at,
        "added_by": str(added_by or ""),
        "note": str(note or ""),
        "config": snapshot,
        # What this entry will actually do, worked out now so the card can
        # show it without re-deriving the plan for every row.
        "summary": _describe_pull_entry(snapshot),
    }
    ddb.put_item(
        TableName=TABLE_NAME,
        Item={
            "kind": {"S": PULL_QUEUE_KIND},
            "sort": {"S": _pull_queue_sort(added_at, entry_id, seq)},
            "body": {"S": json.dumps(entry)},
            "expires_at": {"N": str(int(
                (_now() + _dt.timedelta(days=PULL_QUEUE_TTL_DAYS)).timestamp()))},
        },
    )
    return entry


def _describe_pull_entry(config: dict) -> str:
    """One line saying what this entry sweeps, for the row in the card."""
    sources = [str(s.get("prefix") or "").strip("/")
               for s in (config.get("sources") or [])]
    sources = [s for s in sources if s]
    root = str(config.get("root") or config.get("training_root") or "").strip("/")
    where = f" into {root}" if root else ""
    if not sources:
        return f"No source folders{where}"
    if len(sources) == 1:
        return f"{sources[0]}{where}"
    return f"{len(sources)} source folders{where}"


def list_pull_queue() -> list:
    """Every queued sweep, oldest first. The sort key does the ordering."""
    try:
        rows = ddb.query(
            TableName=TABLE_NAME,
            KeyConditionExpression="#k = :k",
            ExpressionAttributeNames={"#k": "kind"},
            ExpressionAttributeValues={":k": {"S": PULL_QUEUE_KIND}},
            ScanIndexForward=True,
        ).get("Items") or []
    except ClientError as exc:
        print(f"Could not read the pull queue: {exc}")
        return []
    out = []
    for row in rows:
        try:
            entry = json.loads(row["body"]["S"])
        except (KeyError, ValueError):
            continue
        entry["sort"] = row["sort"]["S"]
        out.append(entry)
    return out


def remove_pull_queue_entry(entry_id: str) -> bool:
    """Drops one entry. True if it was there."""
    for entry in list_pull_queue():
        if entry.get("id") != entry_id:
            continue
        ddb.delete_item(
            TableName=TABLE_NAME,
            Key={"kind": {"S": PULL_QUEUE_KIND}, "sort": {"S": entry["sort"]}},
        )
        return True
    return False


def clear_pull_queue() -> int:
    """Drops every entry. Returns how many went."""
    entries = list_pull_queue()
    for entry in entries:
        ddb.delete_item(
            TableName=TABLE_NAME,
            Key={"kind": {"S": PULL_QUEUE_KIND}, "sort": {"S": entry["sort"]}},
        )
    return len(entries)


def advance_pull_queue(*, trigger: str = "queue") -> dict:
    """Starts the next queued sweep, if nothing is sweeping now.

    The guard the Lambda never had. `active_run` reports anything quiet
    for longer than RUN_STALE_SECONDS as `lost` rather than `running`, so
    a sweep whose container was replaced does not block the queue for
    ever -- which is the failure a naive "is something running?" check
    would introduce while fixing the overlap it was aimed at.

    Returns what it did, so the caller can log one line rather than
    guessing from the absence of an exception.
    """
    running = active_run()
    if running and running.get("status") == "running":
        return {"started": False, "reason": "A sweep is already running.",
                "run_id": running.get("run_id", "")}

    queue = list_pull_queue()
    if not queue:
        return {"started": False, "reason": "Nothing queued."}

    entry = queue[0]
    # Dropped BEFORE it runs, not after. A sweep can take fifteen minutes
    # and the process running it can die; an entry still in the queue when
    # that happens would be picked up again by the next tick and run a
    # second time. Losing one sweep is recoverable -- the next one moves
    # the same waiting files -- and running one twice is what this queue
    # exists to prevent.
    remove_pull_queue_entry(entry["id"])

    run_id = new_run_id()
    try:
        result = route_pending(config=entry.get("config") or None,
                               trigger=trigger, run_id=run_id)
    except ValueError as exc:
        # A refusal, not a crash: the snapshot is incomplete. Said in the
        # run row so it shows in the history rather than only in a log.
        write_run_state(run_id, status="failed", trigger=trigger,
                        error=str(exc), entry_id=entry["id"])
        return {"started": False, "reason": str(exc), "entry_id": entry["id"]}
    except Exception as exc:                                       # noqa: BLE001
        write_run_state(run_id, status="failed", trigger=trigger,
                        error=str(exc), entry_id=entry["id"])
        return {"started": False, "reason": str(exc), "entry_id": entry["id"]}

    return {"started": True, "run_id": run_id, "entry_id": entry["id"],
            "moved": result.get("moved", 0), "remaining": len(queue) - 1}


# --------------------------------------------------------------------------
# Lambda entry point
# --------------------------------------------------------------------------

def lambda_handler(event, _context=None):
    """Invoked by EventBridge, on a schedule or on an object arriving.

    Both do the same thing: sweep the drop folders. An arrival event names
    one object, but sweeping is what makes the two triggers equivalent and
    what lets a file that arrived while the function was throttled still
    get filed on the next event.
    """
    detail_type = (event or {}).get("detail-type", "")

    # A TICK drains the queue; it never sweeps on its own account.
    #
    # Separate from the schedule because they answer different questions.
    # The schedule asks "is it time to pull again?" once a day. The tick
    # asks "is anything waiting and is the slot free?" every couple of
    # minutes -- which is what turns a queue into something that actually
    # moves, and does it whether or not the app container is up.
    #
    # One entry per tick, deliberately. A sweep can run for fifteen
    # minutes against a Lambda ceiling of fifteen, so draining two in one
    # invocation risks the second being killed halfway through a copy.
    # The next tick takes the next one.
    if (event or {}).get("tick") or "Tick" in str(detail_type):
        if not pull_in_use(read_config()):
            return {"pull_control": "off"}
        outcome = advance_pull_queue(trigger="queue")
        print(json.dumps({"pull_control": "tick", **outcome}))
        return outcome

    trigger = "arrival" if "Object" in str(detail_type) else "schedule"
    config = read_config()
    # PULL CONTROL OFF: nothing to pull. The schedule and the arrival rule
    # can still fire -- they belong to AWS -- but the sweep stands down
    # rather than reading the drop folders, writing a run row that says
    # "moved 0", and waking the after-pull chain for nothing.
    if not pull_in_use(config):
        print(json.dumps({"pull_control": "off", "trigger": trigger}))
        return {"moved": 0, "skipped": 0, "errors": 0, "off": True}

    if trigger == "arrival":
        # The rule cannot filter by prefix -- the watched folders are
        # settings that change from the card, and a rule pattern rewritten
        # to match them would be a second place to keep them in step. So
        # EVERY object created anywhere in the bucket arrives here, and a
        # dataset assembly copying thousands of files would otherwise be
        # thousands of full sweeps. Checking the key first costs one
        # settings read and no S3 listing at all.
        key = str((((event or {}).get("detail") or {}).get("object") or {}).get("key", ""))
        watched = [str(s.get("prefix") or "").strip("/") for s in config.get("sources") or []]
        if key and not any(p and key.startswith(p + "/") for p in watched):
            print(json.dumps({"pull_control": "ignored", "key": key}))
            return {"moved": 0, "skipped": 0, "errors": 0, "ignored": key}

    # QUEUE IT, do not race it.
    #
    # This handler used to call `route_pending` straight away with no
    # check at all, while the manual route answered 409 if a sweep was
    # already running -- so the guard was on the door nobody uses
    # unattended. An arrival burst of 500 files is 500 invocations, none
    # of which looked to see whether the other 499 were already sweeping.
    #
    # Now every trigger lands in the queue and the queue is drained one at
    # a time. A burst collapses: each invocation adds an entry, the first
    # drain runs a sweep that moves everything waiting, and the rest find
    # nothing to do and cost one DynamoDB read each.
    running = active_run()
    if running and running.get("status") == "running":
        entry = enqueue_pull(config=config, note=f"{trigger} while a sweep was running",
                             added_by=trigger)
        print(json.dumps({"pull_control": "queued", "trigger": trigger,
                          "entry_id": entry["id"],
                          "behind_run": running.get("run_id", "")}))
        return {"queued": True, "entry_id": entry["id"],
                "behind_run": running.get("run_id", "")}

    # A run id even here, so an unattended sweep is as watchable and as
    # stoppable as one started by hand. Nothing can kill a Lambda from
    # outside; the stop flag it re-reads between files is the only way to
    # interrupt one, and it is the same flag the card sets.
    run_id = new_run_id()
    print(json.dumps({"pull_control": "started", "trigger": trigger, "run_id": run_id}))
    try:
        result = route_pending(config=config, trigger=trigger, run_id=run_id)
    except ValueError as exc:
        # A REFUSAL, not a crash: the configuration is incomplete, and the
        # sweep declined rather than filing a week of images nobody
        # annotated. Recorded and notified, because an unattended pull
        # that silently declines looks exactly like one that found
        # nothing -- and the difference is a retrain that will never
        # improve versus a quiet week. Raising instead would leave the
        # reason in a CloudWatch log nobody opens.
        write_run_state(run_id, status="failed", aborted=str(exc),
                        finished_at=_now().isoformat())
        _notify_pull_finished(
            {"run_id": run_id, "trigger": trigger, "moved": 0, "errors": 1},
            aborted=True,
        )
        print(json.dumps({"pull_control": "refused", "trigger": trigger,
                          "run_id": run_id, "reason": str(exc)}))
        return {"moved": 0, "skipped": 0, "errors": 1, "refused": str(exc)}
    print(json.dumps({"pull_control": result.get("moved"), "trigger": trigger,
                      "run_id": run_id, "skipped": result.get("skipped"),
                      "errors": result.get("errors"), "aborted": result.get("aborted")}))
    return result



def discover_units(*, bucket: str, root: str) -> dict:
    """The Ultralytics datasets sitting directly under a root.

    `track2/` holds `escalation_dataset` (person crops) and
    `escalation_dataset_full_frame` (the whole camera view), among other
    folders that are not datasets at all. A unit is a child that has a
    `data.yaml` -- which is also where its class list lives, so the same
    read answers "is this a dataset" and "what does it contain".

    The class list comes back per unit so the card can show it, and so a
    merge can refuse when two units disagree: combining datasets whose
    class ORDER differs silently redefines every label, and a label file
    carries an index and nothing else.
    """
    bucket = (bucket or BUCKET).strip()
    root = (root or "").strip().strip("/")
    if not root:
        raise ValueError("Give the folder that holds the datasets.")

    units = []
    response = s3.list_objects_v2(Bucket=bucket, Prefix=root + "/", Delimiter="/")
    for entry in response.get("CommonPrefixes", []):
        prefix = entry["Prefix"].rstrip("/")
        name = prefix.rsplit("/", 1)[-1]
        classes, counts, problem = {}, {}, ""
        try:
            body = s3.get_object(Bucket=bucket, Key=f"{prefix}/data.yaml")["Body"].read()
            classes = _classes_from_yaml(body.decode("utf-8", "replace"))
        except ClientError:
            # No data.yaml: not a dataset, just a sibling folder.
            continue
        for split in ("train", "val"):
            counts[split] = _count_under(bucket, f"{prefix}/images/{split}/")
        if not any(counts.values()):
            problem = "no images under images/train or images/val"
        units.append({
            "name": name,
            "prefix": prefix,
            "classes": classes,
            "train_images": counts.get("train", 0),
            "val_images": counts.get("val", 0),
            "problem": problem,
        })

    orders = {tuple(u["classes"].get(str(i), "") for i in range(len(u["classes"])))
              for u in units if u["classes"]}
    return {
        "bucket": bucket,
        "root": root,
        "units": units,
        # False means the units cannot safely be combined: same index,
        # different meaning.
        "classes_agree": len(orders) <= 1,
    }


def _count_under(bucket: str, prefix: str) -> int:
    total = 0
    for _ in _list_objects(prefix, bucket):
        total += 1
    return total


def _unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _classes_from_yaml(text: str) -> dict:
    """`{index: name}` from a data.yaml, without a YAML parser.

    No `yaml.safe_load` here on purpose: this module IS the Lambda handler
    file -- Terraform zips this exact file -- so it may import nothing
    beyond boto3 and the standard library.

    All four shapes Ultralytics accepts, because all four are things
    people really write and three of them used to return `{}`:

        names:            names:            names: [helmet, gloves]
          0: helmet         - helmet
          1: gloves         - gloves      names: {0: helmet, 1: gloves}

    Returning `{}` did not fail loudly -- it surfaced as "No data.yaml was
    found to read class names from", which blames a missing file for one
    that is present and valid. The Help page documents the list form, and
    the STANDARD page parses it with `yaml.safe_load` and accepts it, so
    the same dataset built on one page and not on another.

    An index that is not a number is skipped rather than guessed at: a
    wrong index is a silently mislabelled class, which is worse on a
    trained model than no classes at all.
    """
    classes: dict = {}
    inside = False
    flow = ""
    next_index = 0

    for line in text.splitlines():
        stripped = line.strip()

        if flow:
            # A bracketed list or mapping continued across lines.
            flow += " " + stripped
            if flow.rstrip().endswith(("]", "}")):
                return _classes_from_flow(flow)
            continue

        if not inside:
            if stripped.startswith("names:"):
                inline = stripped[len("names:"):].strip()
                if inline.startswith(("[", "{")):
                    if inline.rstrip().endswith(("]", "}")):
                        return _classes_from_flow(inline)
                    flow = inline
                    continue
                inside = True
            continue

        # Inside a block `names:`. A line that is not indented, or blank,
        # ends it -- the next top-level key has started.
        if not stripped or not line[:1].isspace():
            break
        if stripped.startswith("#"):
            continue

        if stripped.startswith("-"):
            # Block sequence: position IS the index, which is what
            # Ultralytics does with a list.
            classes[str(next_index)] = _unquote(stripped[1:])
            next_index += 1
        elif ":" in stripped:
            index, _, name = stripped.partition(":")
            index = index.strip()
            if index.isdigit():
                classes[index] = _unquote(name)
    return classes


def _classes_from_flow(text: str) -> dict:
    """`[helmet, gloves]` or `{0: helmet, 1: gloves}` -> `{index: name}`.

    Split on commas only, which is enough for class names and avoids
    pulling in a parser this file is not allowed to import. A name
    containing a comma would need real YAML, and is not a thing
    Ultralytics class names do.
    """
    body = text.strip()
    is_mapping = body.startswith("{")
    body = body.strip("[]{}").strip()
    if not body:
        return {}
    classes: dict = {}
    for position, part in enumerate(body.split(",")):
        part = part.strip()
        if not part:
            continue
        if is_mapping and ":" in part:
            index, _, name = part.partition(":")
            index = _unquote(index)
            if index.isdigit():
                classes[index] = _unquote(name)
        else:
            classes[str(position)] = _unquote(part)
    return classes



def list_buckets() -> dict:
    """Every bucket in the account, each marked readable or not.

    ListAllMyBuckets returns names and nothing else, and reading objects
    is granted on only a few buckets -- so a name appearing here does not
    mean the folder browser can open it. Each is probed with a one-key
    listing, which is cheap and turns "this choice will fail on the next
    click" into something the picker can show before the click.
    """
    try:
        found = s3.list_buckets().get("Buckets", [])
    except ClientError as exc:
        raise ValueError(f"Could not list buckets: {exc}")

    buckets = []
    for entry in found:
        name = entry.get("Name", "")
        if not name:
            continue
        readable = True
        try:
            s3.list_objects_v2(Bucket=name, MaxKeys=1)
        except ClientError:
            readable = False
        buckets.append({"name": name, "readable": readable})

    # Readable first, then alphabetical -- the ones that work should not
    # have to be hunted for among the ones that do not.
    buckets.sort(key=lambda b: (not b["readable"], b["name"]))
    return {"buckets": buckets, "default": BUCKET}


#: What the data.yaml browser offers as selectable files. Both spellings:
#: Ultralytics writes `.yaml`, plenty of exports write `.yml`.
YAML_SUFFIXES = (".yaml", ".yml")


def list_folders_and_yaml_files(*, bucket: str, prefix: str = "") -> dict:
    """Folders and the .yaml files beside them, in any readable bucket.

    `list_folders` below returns folders only, which is right for picking
    a dataset ROOT. Reading class names needs the file itself -- a folder
    tick box says nothing about which of three yamls it is about to
    read -- and the datasets live in a bucket that is not this app's, so
    this takes one as an argument the same way.
    """
    bucket = (bucket or BUCKET).strip()
    prefix = (prefix or "").strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    folders, files, token = [], [], None
    try:
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            for entry in page.get("CommonPrefixes", []):
                folders.append(entry["Prefix"].rstrip("/").rsplit("/", 1)[-1])
            for entry in page.get("Contents", []):
                key = entry["Key"]
                if key == prefix:
                    continue
                name = key[len(prefix):]
                # Only this level: a nested key would render as a name
                # with slashes in it and select something the browser is
                # not showing.
                if "/" in name:
                    continue
                if name.lower().endswith(YAML_SUFFIXES):
                    files.append(name)
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated"):
                break
    except ClientError as exc:
        raise ValueError(f"Could not read {bucket}/{prefix}: {exc}")

    return {"bucket": bucket, "prefix": prefix.rstrip("/"),
            "folders": sorted(folders), "yaml_files": sorted(files)}


def read_yaml_object(*, bucket: str, key: str) -> bytes:
    """One yaml's bytes, from any readable bucket.

    The extension is checked before the fetch: this is handed a key the
    browser chose, and "read whatever key you are given" is a wider door
    than reading class names needs.
    """
    bucket = (bucket or BUCKET).strip()
    key = (key or "").strip().lstrip("/")
    if not key.lower().endswith(YAML_SUFFIXES):
        raise ValueError(f"'{key}' is not a .yaml file.")
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        raise ValueError(f"Could not read s3://{bucket}/{key}: {exc}")


#: Prefixes no folder delete may touch, or contain. The pipeline's own
#: code, its build sources and the base weights: deleting any of them
#: breaks every run, not one dataset, so there is no "delete anyway".
PROTECTED_PREFIXES = ("custom", "codebuild-source", "track2-app-source",
                      "track2-yolo-weights")


def _overlaps(a: str, b: str) -> bool:
    """True when one prefix is the other or sits inside it."""
    a, b = a.strip("/"), b.strip("/")
    if not a or not b:
        return False
    return a == b or a.startswith(b + "/") or b.startswith(a + "/")


def folder_uses(prefix: str, config: dict | None = None) -> list:
    """Every setting that points at, into or above `prefix`, in words.

    Deleting one of these does not fail -- it empties a folder a card,
    the pull or the next run still reads, and the first anybody hears of
    it is a run with nothing to train on.
    """
    config = config if config is not None else read_config()
    named: list = []

    def add(path, what):
        if isinstance(path, str) and _overlaps(prefix, path):
            named.append(f"{what} ({path.strip('/')})")

    add(config.get("root"), "Pull Control's root -- the pulled files' only copy")
    for src in config.get("sources") or []:
        add(src.get("destination"), "a Pull Control destination")
        add(src.get("eval_destination"), "a Pull Control evaluation destination")
    golden = config.get("golden_set") or {}
    add(golden.get("target_folder"), "the golden set")
    retention = config.get("retention") or {}
    add(retention.get("root"), "the Retention card's root")
    add(retention.get("golden_root"), "the Retention card's golden root")
    assemble = config.get("dataset_assemble") or {}
    for side in ("train_sources", "eval_sources"):
        for src in assemble.get(side) or []:
            add((src or {}).get("images"), "an Assemble a dataset folder")
    add((assemble.get("built") or {}).get("folder"), "the last data.yaml Assemble built")
    add("_dataset_snapshots", "frozen snapshot records")
    add("_runs", "the record of what each run was sent")
    # Once each, in order.
    return list(dict.fromkeys(named))


def folder_delete_check(*, bucket: str, prefix: str) -> dict:
    """What deleting `prefix` would remove, and why it might not be safe.

    Refuses outright (ValueError) for the bucket root, a bucket that is
    not this app's, and the pipeline's own prefixes. Otherwise counts
    the objects and names every setting that uses the folder, so the
    dialog can say both before anything is deleted.
    """
    config = read_config()
    bucket = (bucket or "").strip() or destination_bucket(config)
    allowed = {b for b in (BUCKET, destination_bucket(config)) if b}
    if bucket not in allowed:
        raise ValueError(
            f"Folders can only be deleted in this app's bucket "
            f"({', '.join(sorted(allowed))}). {bucket} belongs to somebody else's "
            "data and is read-only from here.")
    prefix = (prefix or "").strip().strip("/")
    if not prefix:
        raise ValueError("Refusing to delete the whole bucket.")
    for guarded in PROTECTED_PREFIXES:
        if _overlaps(prefix, guarded):
            raise ValueError(
                f"{prefix} holds or is part of {guarded}/, the pipeline's own code or "
                "weights. Deleting it would break every run, so it is not offered.")

    count, size, token = 0, 0, None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix + "/"}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        for obj in page.get("Contents", []):
            count += 1
            size += int(obj.get("Size") or 0)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return {"bucket": bucket, "prefix": prefix, "objects": count, "bytes": size,
            "in_use": folder_uses(prefix, config)}


def delete_folder(*, bucket: str, prefix: str, force: bool = False) -> dict:
    """PERMANENTLY delete every object under `prefix`.

    The same refusals as `folder_delete_check`, plus one that `force`
    lifts: a folder some setting still uses. Listed in full before
    anything is deleted -- deleting inside the pagination loop shifts
    the listing under the continuation token and a page survives while
    the call reports success (see `purge_week_folders`).
    """
    check = folder_delete_check(bucket=bucket, prefix=prefix)
    if check["in_use"] and not force:
        raise PermissionError(
            "In use by " + "; ".join(check["in_use"]) + ".")
    bucket, prefix = check["bucket"], check["prefix"]
    keys: list = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix + "/"}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        keys += [o["Key"] for o in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    deleted, errors = 0, []
    for start in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[start:start + 1000]]
        out = s3.delete_objects(Bucket=bucket, Delete={"Objects": batch, "Quiet": True})
        failed = out.get("Errors") or []
        deleted += len(batch) - len(failed)
        errors += [f"{e.get('Key')}: {e.get('Message')}" for e in failed]
    return {"bucket": bucket, "prefix": prefix, "deleted": deleted, "errors": errors}


def list_folders(*, bucket: str, prefix: str = "") -> dict:
    """The folders directly under a prefix, in any readable bucket.

    The app's own folder browser is bound to this app's bucket; the
    datasets are in another one, so this takes the bucket as an argument.
    Delimited, so it returns one level at a time -- the browser walks
    down rather than listing a whole bucket to show ten names.
    """
    bucket = (bucket or BUCKET).strip()
    prefix = (prefix or "").strip().lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"

    folders, token = [], None
    try:
        while True:
            kwargs = {"Bucket": bucket, "Prefix": prefix, "Delimiter": "/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = s3.list_objects_v2(**kwargs)
            for entry in page.get("CommonPrefixes", []):
                folders.append(entry["Prefix"].rstrip("/").rsplit("/", 1)[-1])
            token = page.get("NextContinuationToken")
            if not page.get("IsTruncated"):
                break
    except ClientError as exc:
        raise ValueError(f"Could not read {bucket}/{prefix}: {exc}")

    return {"bucket": bucket, "prefix": prefix.rstrip("/"), "folders": sorted(folders)}



# --------------------------------------------------------------------------
# Settings on disk, and the weeks that exist
# --------------------------------------------------------------------------

#: A directory on the EC2 HOST, mounted into the container by deploy.sh.
#: Inside the container it would be destroyed on the next deploy, which is
#: the whole reason the mount exists.
SETTINGS_DIR = os.environ.get("PULL_CONTROL_SETTINGS_DIR", "/settings")

#: Which settings belong to which tab (2026-10-29). Each tab saves, loads
#: and resets ONLY its own keys, into a file named `<name>_DP` or
#: `<name>_AT`, so saving one page can no longer overwrite the other's
#: choices with whatever happened to be live.
#:
#: The live configuration is still ONE record -- the chain reads both
#: halves -- so this is a split of the FILES, not of the system.
#:
#: Data Preparation: the pull and its schedule, where weeks go, retention,
#: the golden set, Assemble, the week selection, and the SQL tab (data
#: side, and it has no toolbar of its own). Multi-Auto Training: the
#: detector and run shape, which version and dataset it trains from, the
#: Import card, and the TRAINING schedule with its after-pull guards.
#: Anything in neither (run state such as `chain_state`) is saved by
#: neither: it is a result, not a choice.
SCOPES = {
    "dp": (
        "sources", "picked", "source_bucket", "destination_bucket", "root",
        "training_root", "eval_root", "week_cutoff", "week_anchor",
        "images_subpath", "labels_subpath", "week_prefix", "week_length_days",
        "auto_create_weeks", "first_week_takes_all", "week_by",
        "mode", "schedule_expression", "schedule_every_days", "schedule_from",
        "schedule_start_date", "schedule_time",
        "image_extensions", "label_extensions", "overwrite",
        "retention", "golden_set", "dataset_assemble", "auto_training",
        "sql_explorer", "central_store",
    ),
    "at": (
        "default_detector", "default_training", "default_source",
        "default_version", "data_source", "default_dataset",
        "last_pull_dataset", "import_checkpoint", "freeze_on_run",
        "retrain_schedule", "queue_after_pull",
    ),
}
#: The ending each tab's files carry. The name typed is the XXX.
SUFFIXES = {"dp": "_DP", "at": "_AT"}
#: The cards that live on Data Preparation, by the key `useCardOpen`
#: remembers them under (a Card's lower-cased title, or its own key).
#: `card_state` is ONE shared record, so each tab's file carries its own
#: cards' entries and a load merges them in rather than replacing it.
DP_CARDS = frozenset({
    "assemble", "golden_set", "pull_control", "retention",
    "assemble a dataset", "pick golden set / edge cases", "pull control",
    "auto training", "pull queue", "central data store",
})


def _check_scope(scope: str) -> str:
    scope = (scope or "").strip().lower()
    if scope and scope not in SCOPES:
        raise ValueError(f"Unknown settings tab {scope!r}.")
    return scope


def scoped_name(name: str, scope: str) -> str:
    """`name` with the tab's ending, added once. Typing `helmet` or
    `helmet_DP` on Data Preparation both save `helmet_DP`."""
    name = (name or "").strip()
    if not scope:
        return name
    suffix = SUFFIXES[scope]
    if name.upper().endswith(suffix):
        name = name[:-len(suffix)]
    if not name:
        raise ValueError("Type a name for the settings file.")
    return name + suffix


def scope_of(name: str) -> str:
    """Which tab a file belongs to, from its name: "dp", "at", or "" for
    a file saved before the split, which holds both pages."""
    upper = (name or "").upper()
    for scope, suffix in SUFFIXES.items():
        if upper.endswith(suffix):
            return scope
    return ""


def scope_slice(config: dict, scope: str) -> dict:
    """The part of `config` one tab owns, with its own cards' open state."""
    if not scope:
        return dict(config)
    out = {k: config[k] for k in SCOPES[scope] if k in config}
    cards = config.get("card_state")
    if isinstance(cards, dict):
        out["card_state"] = {k: v for k, v in cards.items()
                             if (k in DP_CARDS) == (scope == "dp")}
    return out


def _settings_path(name: str) -> str:
    """A safe path inside the settings directory.

    The name comes from a text box, so anything with a separator in it is
    refused rather than cleaned -- a quietly renamed file is worse than a
    rejected one, and `../` in a filename is not a typo worth guessing at.
    """
    name = (name or "").strip() or "pull_control"
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError("A settings name cannot contain a path.")
    return os.path.join(SETTINGS_DIR, f"{name}.json")


def folder_identity(config: dict) -> tuple:
    """What a settings file OWNS: where it writes, and under what prefix.

    Two files sharing an identity write into the same week folders. The
    images then sit in one place with nothing recording which dataset
    each came from -- not recoverable afterwards, because a filename says
    nothing about which configuration filed it.

    Reads `root` first. It used to read only the legacy
    `training_root`/`eval_root` pair, which every config still carries
    from DEFAULTS whether or not it uses them -- so every configuration
    computed the SAME identity and every save after the first was
    refused, whatever had actually been configured.
    """
    prefix = str(config.get("week_prefix") or "week_").strip()
    single = str(config.get("root") or "").strip().strip("/")
    if single:
        return (single, "", prefix)
    return (
        str(config.get("training_root") or "").strip().strip("/"),
        str(config.get("eval_root") or "").strip().strip("/"),
        prefix,
    )


def _picked_sources(config: dict) -> tuple:
    """The folders a configuration reads, as a comparable value."""
    picked = config.get("picked") or {}
    return tuple(sorted(
        (key, tuple(sorted(str(f).strip().strip("/") for f in (picked.get(key) or []))))
        for key in ("train_images", "train_labels", "val_images", "val_labels")
    ))


def pull_in_use(config: dict) -> bool:
    """Is Pull Control ticked in this configuration?

    `dataset_assemble.only` is the stored form of the tick, inverted:
    True means "Use Pull Control" is OFF. With it off nothing is pulled
    and no week folder is written, so the pull's checks and sweeps have
    nothing to act on."""
    return not bool((config.get("dataset_assemble") or {}).get("only"))


def settings_owner(identity: tuple, *, ignore: str = "", sources=None) -> str:
    """A settings file that would fight this one over the same folders.

    Only when it reads DIFFERENT source folders. Two files with the same
    picks and the same destination are a snapshot of one configuration --
    they file identical files to identical places, and dedup makes the
    second a no-op. Refusing those blocked the ordinary habit of saving
    `..._v4` beside `..._v3`, which is not a collision at all.

    What is worth refusing is two configurations reading DIFFERENT
    folders into one set of week folders: their images merge with nothing
    recording which dataset each came from, and no filename says.
    """
    for row in list_settings_files().get("files") or []:
        name = row.get("name") or ""
        if not name or name == ignore:
            continue
        try:
            saved = read_settings_file(name)
        except (ValueError, OSError):
            continue
        if folder_identity(saved) != identity:
            continue
        # A file with Pull Control off writes no week folders, so it
        # cannot fight anybody over them.
        if not pull_in_use(saved):
            continue
        if sources is not None and _picked_sources(saved) == sources:
            continue                      # the same configuration, saved twice
        return name
    return ""


def read_settings_file(name: str) -> dict:
    """The config inside a settings file. Applies nothing."""
    path = _settings_path(name)
    if not os.path.exists(path):
        raise ValueError(f"No settings file called {name}.")
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    config = body.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError(f"{name}.json holds no settings.")
    return config


def save_settings_file(*, name: str = "pull_control", config: dict | None = None,
                       allow_clash: bool = False,
                       sections: dict | None = None,
                       scope: str = "") -> dict:
    """Writes the current settings to a file on the host.

    `scope` ("dp" or "at") writes only that tab's settings, to
    `<name>_DP` or `<name>_AT` -- see SCOPES. Blank keeps the old
    whole-page file.

    Written to a temporary file and renamed, so a crash midway leaves the
    previous settings intact rather than half a file that will not parse.

    `sections` adds further TOP-LEVEL sections beside `config` --
    `registry`, today. They sit outside `config` deliberately: the live
    config is the pull-control settings and nothing else, and
    `load_settings_file` writes `config` straight into it. A registry
    copy inside that dict would be written into the live configuration
    on every load, where nothing reads it and `write_config` would then
    carry it forever.

    Refuses to write a second file that owns the SAME roots and prefix as
    an existing one. Two configurations filing into one set of week
    folders merge two datasets into one with no record of the split, and
    the only fix afterwards is to re-pull both from scratch.
    """
    scope = _check_scope(scope)
    name = scoped_name(name, scope)
    path = _settings_path(name)
    whole = dict(config or read_config())
    body = scope_slice(whole, scope)

    # The refusal is a WARNING somebody can overrule, not a rule the file
    # format depends on: two configs filing into one set of week folders
    # is usually a mistake, and occasionally exactly what was meant --
    # saving a variant of a config you are iterating on, say. Overridden
    # only by an explicit tick, never by default.
    identity = folder_identity(body)
    # Only when THIS configuration pulls. With Pull Control off it writes
    # no week folders, so a clash over them is not a clash -- and the
    # warning was refusing saves of configurations that never pull.
    # A Multi-Auto Training file holds no folders at all, so it cannot
    # clash over them -- and `pull_in_use` reads a missing Assemble
    # section as "pulling".
    clash = None if allow_clash or scope == "at" or not pull_in_use(body) else settings_owner(
        identity, ignore=name, sources=_picked_sources(body))
    if clash:
        roots = " + ".join(part for part in identity[:2] if part) or "(no root)"
        raise ValueError(
            f"{clash} already writes {roots} with prefix {identity[2]!r}. "
            f"It reads different source folders, so the two would merge "
            f"different datasets into one set of week folders with no record of "
            f"which images came from which. Change the week prefix, or the root "
            f"folder, before saving this one."
        )
    os.makedirs(SETTINGS_DIR, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        document = {"saved_at": _now().isoformat(), "config": body}
        if scope:
            document["scope"] = scope
        # Unknown top-level keys are ignored by `load_settings_file`,
        # which reads `config` only -- so a file written with sections
        # still loads on a build that predates them.
        for key, value in (sections or {}).items():
            if key not in ("saved_at", "config"):
                document[key] = value
        json.dump(document, handle, indent=2)
    os.replace(temporary, path)
    # Recorded AFTER the write succeeds, so a refused save (a clashing
    # root) does not leave the page claiming to be in a file it never
    # wrote.
    try:
        write_config({f"settings_file_{scope}" if scope else "settings_file": name})
    except Exception:                                          # noqa: BLE001
        pass
    return {"name": name, "path": path, "bytes": os.path.getsize(path),
            "scope": scope}


def reset_to_defaults(scope: str = "") -> dict:
    """Puts the live settings back to the shipped defaults -- a blank
    template to start a new settings file from.

    A REPLACE, not a merge. `write_config` keeps anything the caller did
    not mention, which is right for every other writer and wrong here:
    a key that is stored but has no default -- `queue_after_pull`, say --
    would survive into what is supposed to be a fresh start, so the new
    file would quietly carry a setting nobody chose. Anything left out
    comes back as its default anyway, because `read_config` merges over
    DEFAULTS.

    The trigger is re-applied for the same reason a load does it: the
    defaults are `mode: off` with no source folders picked, so skipping
    it would leave a rule armed at AWS, firing against a config that no
    longer names anything to collect.

    Saved files on disk are NOT touched. This is "start a new one", not
    "throw the old ones away".

    `scope` resets ONE tab's settings and leaves the other's live: "New"
    on Data Preparation must not drop the detector Multi-Auto Training is
    set to, nor the reverse.
    """
    scope = _check_scope(scope)
    if scope:
        fresh = {k: DEFAULTS[k] for k in SCOPES[scope] if k in DEFAULTS}
        # Not in DEFAULTS, so a merge would keep them; reset explicitly.
        if scope == "dp":
            fresh["week_anchor"] = ""
        live_cards = read_config().get("card_state") or {}
        fresh["card_state"] = {k: v for k, v in live_cards.items()
                               if (k in DP_CARDS) != (scope == "dp")}
        fresh[f"settings_file_{scope}"] = ""
        merged = write_config(fresh)
        if scope == "at":
            return {"config": merged, "problems": []}
        problems: list = []
        try:
            problems = set_mode(
                mode=str(merged.get("mode") or "off"),
                schedule_expression=str(merged.get("schedule_expression") or ""),
            ).get("problems") or []
        except Exception as exc:                            # noqa: BLE001
            problems = [f"settings reset, but the trigger was not applied: {exc}"]
        return {"config": merged, "problems": problems}
    body = dict(DEFAULTS)
    ddb.put_item(
        TableName=TABLE_NAME,
        Item={
            "kind": {"S": "config"},
            "sort": {"S": CONFIG_SORT},
            "body": {"S": json.dumps(body)},
            "updated_at": {"S": _now().isoformat()},
        },
    )
    merged = read_config()
    problems: list = []
    try:
        applied = set_mode(
            mode=str(merged.get("mode") or "off"),
            schedule_expression=str(merged.get("schedule_expression") or ""),
        )
        problems = applied.get("problems") or []
    except Exception as exc:                                # noqa: BLE001
        # The settings are live either way; only the rules are not.
        problems = [f"settings reset, but the trigger was not applied: {exc}"]
    return {"config": merged, "problems": problems}


def read_settings_section(name: str, key: str) -> dict:
    """One top-level section of a settings file, or {} if it has none.

    Separate from `read_settings_file` so a caller after the registry
    does not have to know the document's shape, and so a file written
    before sections existed answers {} rather than raising.
    """
    try:
        body = read_settings_file(name)
    except (OSError, ValueError):
        return {}
    section = body.get(key)
    return section if isinstance(section, dict) else {}


def load_settings_file(*, name: str = "pull_control", apply: bool = True,
                       scope: str = "") -> dict:
    """Reads a settings file back and makes every part of it live.

    Three things have to happen, not one, or the card shows a setting the
    system is not actually using:

      1. the settings themselves;
      2. `sources`, REBUILT from the restored `picked` -- writing the
         config alone would restore the folder lists while leaving the
         sweep pointed at the folders from before, which is the worst
         kind of wrong because both halves look fine on their own;
      3. the trigger, re-applied to the EventBridge rules -- otherwise a
         file saved with the trigger off loads as "off" in the card while
         the rule stays armed and keeps firing.

    `scope` loads ONE tab's settings out of the file and leaves the rest
    of the live configuration alone -- so an old whole-page file loaded
    on Data Preparation brings back only the data side. A file belonging
    to the other tab is refused rather than half-applied.
    """
    scope = _check_scope(scope)
    if scope and scope_of(name) and scope_of(name) != scope:
        other = "Multi-Auto Training" if scope_of(name) == "at" else "Data Preparation"
        raise ValueError(f"{name} holds {other} settings; load it on that tab.")
    path = _settings_path(name)
    if not os.path.exists(path):
        raise ValueError(f"No settings file called {name}.")
    with open(path, encoding="utf-8") as handle:
        body = json.load(handle)
    config = body.get("config") or {}
    if not isinstance(config, dict) or not config:
        raise ValueError(f"{name}.json holds no settings.")
    config = scope_slice(config, scope)

    applied_trigger: dict = {}
    problems: list = []
    if apply:
        live = read_config()
        update = dict(config)
        if scope:
            # Merged, not replaced: the record also holds the other tab's
            # cards, and `write_config` replaces a top-level key whole.
            update["card_state"] = {**(live.get("card_state") or {}),
                                    **(config.get("card_state") or {})}
            update[f"settings_file_{scope}"] = name
        else:
            # The file's own name wins over any `settings_file` inside it: a
            # file copied or renamed on disk would otherwise load claiming to
            # be the one it was copied from.
            update["settings_file"] = name
        merged = write_config(update)

        if scope == "at":
            # Nothing to re-arm: the training clock is read on every tick,
            # and the pull's rules belong to the other tab.
            return {"name": name, "saved_at": body.get("saved_at", ""),
                    "config": config, "applied": True, "scope": scope,
                    "mode": str(merged.get("mode") or "off"),
                    "sources": len(merged.get("sources") or []), "problems": []}

        if merged.get("picked"):
            merged = write_config({"sources": sources_from_picked(merged)})

        try:
            applied_trigger = set_mode(
                mode=str(merged.get("mode") or "off"),
                schedule_expression=str(merged.get("schedule_expression") or ""),
            )
            problems = applied_trigger.get("problems") or []
        except Exception as exc:
            # The settings are live either way; only the rules are not.
            # Reported rather than raised, so a load does not look like a
            # total failure because EventBridge was unreachable.
            problems = [f"settings loaded, but the trigger was not applied: {exc}"]

    return {
        "name": name,
        "saved_at": body.get("saved_at", ""),
        "config": config,
        "applied": bool(apply),
        "scope": scope,
        "mode": str(config.get("mode") or "off"),
        "sources": len((config.get("picked") and
                        sources_from_picked({**DEFAULTS, **config})) or
                       config.get("sources") or []),
        "problems": problems,
    }


def list_settings_files() -> dict:
    """Every saved settings file, newest first."""
    try:
        names = [f[:-5] for f in os.listdir(SETTINGS_DIR) if f.endswith(".json")]
    except OSError:
        return {"directory": SETTINGS_DIR, "files": [], "writable": False}
    files = []
    for name in names:
        path = _settings_path(name)
        try:
            files.append({"name": name, "scope": scope_of(name),
                          "bytes": os.path.getsize(path),
                          "modified": _dt.datetime.fromtimestamp(
                              os.path.getmtime(path), _dt.timezone.utc).isoformat()})
        except OSError:
            continue
    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"directory": SETTINGS_DIR, "files": files,
            "writable": os.access(SETTINGS_DIR, os.W_OK),
            # What each file CLAIMS, so the card can warn before a second
            # one is pointed at the same place. See `settings_claims`.
            "claims": settings_claims(names)}


def settings_claims(names: list | None = None) -> list:
    """Which folder each saved settings file trains out of, and as what.

    Two files pointed at one root is not an error -- the same week
    folders CAN feed two Classes -- but it is almost never what somebody
    means, and nothing said so. The pull writes into that root, retention
    prunes it and the chain trains from it, so the second file silently
    inherits the first one's data and the first one's pulls reshape the
    second one's corpus.

    Read from the FILES, not from live config: the live one is whatever
    was loaded last, and the question is what every OTHER saved
    configuration already uses.
    """
    if names is None:
        try:
            names = [f[:-5] for f in os.listdir(SETTINGS_DIR) if f.endswith(".json")]
        except OSError:
            return []
    out = []
    for name in names:
        try:
            body = read_settings_file(name)
        except (OSError, ValueError):
            # A file that will not parse is not a claim on anything, and
            # failing the whole scan for one bad file would take the
            # warning away from every other.
            continue
        config = body.get("config") if isinstance(body.get("config"), dict) else body
        detector = (config.get("default_detector") or {}) if isinstance(config, dict) else {}
        assemble = (config.get("dataset_assemble") or {}) if isinstance(config, dict) else {}
        out.append({
            "name": name,
            "root": str((config or {}).get("root") or "").strip("/"),
            "model": str(detector.get("slug") or ""),
            # The Assemble card's folders are a claim too: two files
            # writing one data.yaml fight over the same file.
            "assembled": str((assemble.get("built") or {}).get("folder") or "").strip("/"),
        })
    return out


def week_contents(config: dict | None = None) -> dict:
    """Every week folder that exists, and how much is in each.

    Counted from S3 rather than from the week rows: the rows say what was
    DEFINED, and what matters when deciding whether to delete one is what
    is actually in it.
    """
    config = config or read_config()
    roots = set()
    for source in config.get("sources") or []:
        for field in ("destination", "eval_destination"):
            root = str(source.get(field) or "").strip("/")
            if root:
                roots.add(root)

    defined = {w["name"] for w in list_weeks()}
    into = destination_bucket(config)
    weeks: dict = {}
    for root in sorted(roots):
        # Markers included: a week folder that was created and never
        # filled is exactly the case the card has to be able to show,
        # and without them it looks like it was never created.
        for obj in _list_objects(root + "/", into, markers=True):
            rest = obj["Key"][len(root) + 1:]
            name = rest.split("/", 1)[0]
            if not name:
                continue
            row = weeks.setdefault(name, {"week": name, "images": 0, "labels": 0,
                                          "bytes": 0, "roots": set()})
            # The root is recorded whether this key is a file or a marker,
            # so `exists` follows the FOLDER rather than its contents.
            row["roots"].add(root)
            if obj["Key"].endswith("/"):
                continue
            row["bytes"] += obj.get("Size", 0)
            if "/images/" in obj["Key"]:
                row["images"] += 1
            elif "/labels/" in obj["Key"]:
                row["labels"] += 1

    out = []
    for name, row in weeks.items():
        row["roots"] = sorted(row["roots"])
        # A folder with no week row behind it is a leftover from a week
        # that was forgotten without its contents being removed.
        row["defined"] = name in defined
        out.append(row)
    for name in sorted(defined - set(weeks)):
        out.append({"week": name, "images": 0, "labels": 0, "bytes": 0,
                    "roots": [], "defined": True})

    def order(row):
        digits = "".join(ch for ch in row["week"] if ch.isdigit())
        return (int(digits) if digits else 0, row["week"])

    out.sort(key=order)
    return {"weeks": out, "roots": sorted(roots)}


def purge_week(*, name: str, config: dict | None = None) -> dict:
    """Deletes a week's FOLDERS as well as forgetting the week.

    Separate from `delete_week`, which only forgets the definition. This
    removes data, so it is never what a sweep does on its own -- only what
    someone presses a button for.
    """
    name = (name or "").strip().strip("/")
    if not name:
        raise ValueError("Which week?")
    config = config or read_config()

    roots = set()
    for source in config.get("sources") or []:
        for field in ("destination", "eval_destination"):
            root = str(source.get(field) or "").strip("/")
            if root:
                roots.add(root)

    into = destination_bucket(config)
    removed = 0
    for root in sorted(roots):
        batch: list = []
        for obj in _list_objects(f"{root}/{name}/", into):
            batch.append({"Key": obj["Key"]})
            if len(batch) == 1000:
                s3.delete_objects(Bucket=into, Delete={"Objects": batch})
                removed += len(batch)
                batch = []
        # The folder markers themselves end in "/" and are skipped by
        # _list_objects, so they are swept separately or the folder lingers
        # looking like an empty week that still exists.
        for marker in (f"{root}/{name}/",):
            batch.append({"Key": marker})
        for sub in ("images", "labels"):
            for split in ("train", "val"):
                batch.append({"Key": f"{root}/{name}/{sub}/{split}/"})
                batch.append({"Key": f"{root}/{name}/{sub}/"})
        if batch:
            s3.delete_objects(Bucket=into, Delete={"Objects": batch})
            removed += len(batch)

    for week in list_weeks():
        if week["name"] == name:
            delete_week(name=week["name"], starts_at=week["starts_at"])
    return {"week": name, "deleted": removed, "roots": sorted(roots)}



def check_pairing(*, bucket: str = "", image_prefixes=(), label_prefixes=()) -> dict:
    """Do the chosen image folders and label folders describe each other?

    An Ultralytics label is matched to its image by STEM -- `frame_07.jpg`
    belongs to `frame_07.txt`, wherever each of them lives. So folders can
    be picked independently, but only if the names line up, and nothing
    else in the pipeline checks that.

    The two kinds of mismatch are not equally serious, and saying so is
    the point of splitting them:

      * A label with no image is always a mistake. It describes nothing
        and the trainer never sees it, so whatever it was annotating is
        silently absent from the run.

      * An image with no label is a BACKGROUND example -- a frame with
        nothing to detect. Ultralytics accepts those deliberately and
        they improve precision. A few are healthy; the whole folder means
        the labels were not picked, which is the mistake this exists to
        catch.
    """
    bucket = (bucket or BUCKET).strip()

    def stems(prefixes, extensions):
        found: dict = {}
        for prefix in prefixes:
            prefix = str(prefix or "").strip().strip("/")
            if not prefix:
                continue
            for obj in _list_objects(prefix + "/", bucket):
                name = obj["Key"].rsplit("/", 1)[-1]
                if name.lower().endswith(tuple(extensions)):
                    found.setdefault(name.rsplit(".", 1)[0], obj["Key"])
        return found

    images = stems(image_prefixes, [e.lower() for e in DEFAULTS["image_extensions"]])
    labels = stems(label_prefixes, (".txt",))

    only_images = sorted(set(images) - set(labels))
    only_labels = sorted(set(labels) - set(images))
    paired = sorted(set(images) & set(labels))

    return {
        "images": len(images),
        "labels": len(labels),
        "paired": len(paired),
        "images_without_labels": len(only_images),
        "labels_without_images": len(only_labels),
        # A handful of names is enough to recognise the pattern -- a
        # different extension, a prefix, the wrong folder entirely.
        "sample_images_without_labels": only_images[:5],
        "sample_labels_without_images": only_labels[:5],
        "checked": bool(images or labels),
    }



PICK_LISTS = (
    ("train_images", "train"),
    ("train_labels", "train"),
    ("val_images", "val"),
    ("val_labels", "val"),
)


def sources_from_picked(config: dict) -> list:
    """Turns the four folder lists into the sources a sweep walks.

    One source per folder. The evaluation root is left blank on every one
    of them and the split is set explicitly, so a folder put in the
    training lists goes to training even when `val` appears in its path --
    the list it was put in is the answer.
    """
    picked = config.get("picked") or {}
    bucket = str(config.get("source_bucket") or "").strip()
    single = str(config.get("root") or "").strip().strip("/")
    if single:
        # Both splits file into it. `placement` then keeps train/ and val/
        # in the path, because the root alone no longer says which is
        # which -- see splits_per_root.
        roots = {"train": single, "val": single}
    else:
        roots = {
            "train": str(config.get("training_root") or "").strip("/"),
            "val": str(config.get("eval_root") or "").strip("/"),
        }

    out = []
    for key, split in PICK_LISTS:
        root = roots[split]
        if not root:
            continue
        for folder in picked.get(key) or []:
            folder = str(folder or "").strip().strip("/")
            if not folder:
                continue
            out.append({
                "prefix": folder,
                "bucket": bucket,
                "destination": root,
                "eval_destination": "",
                "split": split,
                # Always a copy. These are live datasets read from another
                # bucket that is granted read-only; a move could not work
                # and should not be offered.
                "action": "copy",
            })
    return out


def delete_settings_file(*, name: str) -> dict:
    """Removes a saved settings file. Only ever called from the card."""
    path = _settings_path(name)
    if not os.path.exists(path):
        raise ValueError(f"No settings file called {name}.")
    os.remove(path)
    return {"name": name, "deleted": True}



def source_sha() -> str:
    """The hash of THIS file, as Terraform computes it for the router.

    `filesha256` over the same bytes, so the two can be compared directly.
    The container and the Lambda run this one file; if their hashes differ,
    the router is filing with different rules than the card is previewing.
    """
    try:
        with open(__file__, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except OSError:
        return ""


def router_freshness() -> dict:
    """Is the deployed router running this same routing code?

    Silent drift is the failure being caught. The container deploys and
    the Lambda does not; each half is internally consistent, so nothing
    errors -- files simply start landing in a shape the card does not
    describe. That happened three times in one day before this existed.

    Never raises. A router that cannot be read reports `unknown`, because
    a status panel that breaks when an unrelated permission is missing is
    worse than one that admits it does not know.
    """
    mine = source_sha()
    try:
        config = boto3.client("lambda", region_name=REGION).get_function_configuration(
            FunctionName=ROUTER_FUNCTION,
        )
        theirs = (config.get("Environment", {}).get("Variables", {})
                  .get("ROUTER_SOURCE_SHA", ""))
        modified = config.get("LastModified", "")
    except Exception as exc:
        return {"state": "unknown", "detail": str(exc)[:200], "app_sha": mine[:12]}

    if not theirs:
        # Deployed before the hash existed, so it cannot be compared --
        # which is itself a sign it has not been updated recently.
        return {"state": "unknown", "detail": "the router publishes no source hash",
                "app_sha": mine[:12], "router_modified": modified}

    return {
        "state": "current" if theirs == mine else "stale",
        "app_sha": mine[:12],
        "router_sha": theirs[:12],
        "router_modified": modified,
        "detail": "" if theirs == mine else (
            "The router is running different routing code than this app. "
            "Unattended sweeps will file using the OLDER rules. Run: "
            "cd terraform/app && terraform apply "
            "-target=aws_lambda_function.pull_control_router"
        ),
    }



def week_at(moment: _dt.datetime, config: dict) -> dict:
    """Which week a moment falls in: name, index, and its bounds.

    Pure, and the only place the week arithmetic lives. The sweep uses it
    to decide where files go; the forecast uses it to say what the next
    six weeks will be. Two implementations of this would be two answers
    to "which week is it", which is the one thing that must not be in
    doubt.

        week 1  -- everything up to the cut-off, with no start
        week 2  -- cut-off              .. cut-off + length
        week 3  -- cut-off + length     .. cut-off + 2 x length

    With no cut-off set there is only ever week 1: nothing has said when
    the baseline stops, so nothing can be after it.
    """
    prefix = str(config.get("week_prefix") or "week_")
    length = _dt.timedelta(days=max(int(config.get("week_length_days") or DEFAULT_WEEK_DAYS), 1))

    cutoff_text = str(config.get("week_cutoff") or "").strip()
    cutoff = None
    if cutoff_text:
        try:
            cutoff = _parse_time(cutoff_text)
        except ValueError:
            cutoff = None

    if cutoff is None or moment < cutoff:
        # Week 1 has no start. `starts_at` is the cut-off minus one length
        # only so the row has a sortable key; `open_start` is what says it
        # actually reaches back forever.
        return {
            "index": 1,
            "name": f"{prefix}1",
            "starts_at": (cutoff - length).isoformat() if cutoff else moment.isoformat(),
            "ends_at": cutoff.isoformat() if cutoff else "",
            "open_start": True,
        }

    elapsed = int((moment - cutoff) // length)
    start = cutoff + elapsed * length
    return {
        "index": elapsed + 2,
        "name": f"{prefix}{elapsed + 2}",
        "starts_at": start.isoformat(),
        "ends_at": (start + length).isoformat(),
        "open_start": False,
    }


def active_week(config: dict, contents: dict, now: _dt.datetime | None = None) -> str:
    """The week the NEXT pull will actually file into.

    Not the week the clock is in, and that difference is the whole point.
    Under `file_time` a file goes to the week its OWN timestamp falls in,
    so what decides the answer is the dates of the files waiting to be
    pulled -- not today's date.

    Week 1 is the only week that collects retrospectively: it takes
    everything before the cut-off, however old. So while week 1 is still
    empty, the next pull's bulk lands there no matter how far past the
    cut-off today is. Reporting the clock's week instead said `week_2`
    while a pull would have put 1,458 files in week 1 and 34 in week 2.

    Once the backlog has been collected, every later file is dated inside
    a week that has already started, and the clock's week is the right
    answer again.
    """
    prefix = str(config.get("week_prefix") or "week_")
    here = week_at(now or _now(), config)

    if not bool(config.get("first_week_takes_all", True)):
        # No backlog is wanted, so nothing reaches back and the clock is
        # the only thing that decides.
        return here["name"]

    first = f"{prefix}1"
    if here["name"] == first:
        return first
    row = {r["week"]: r for r in contents.get("weeks") or []}.get(first)
    # Defined but empty is not collected: a week row with no folder
    # behind it means nothing has been filed there yet.
    collected = bool(row) and (int(row.get("images") or 0) or int(row.get("labels") or 0))
    return here["name"] if collected else first


def week_forecast(*, config: dict | None = None, ahead: int = 4,
                  overrides: dict | None = None) -> dict:
    """Every week: what exists, what is being filled, what is coming.

    Counts come from S3 rather than from the week rows, because what
    matters about a week is what is in it. A week that is defined but
    empty and a week that was never defined look the same in a listing,
    so both are reported with their state rather than merged.

    `overrides` re-dates the whole table under settings that have been
    chosen but not applied -- changing "new week every N days" in the
    card should show what that would mean before it is saved. Only the
    DATES move: the counts still come from the folders that exist, since
    changing the week length does not move a single file.

    That is worth being precise about, because it is the trap. A folder
    named week_3 held 8-15 Sep under seven-day weeks; under fourteen-day
    weeks the same folder is now labelled 8-22 Sep, and the files inside
    it did not change. The card warns when weeks already exist.
    """
    config = dict(config or read_config())
    if overrides:
        config.update({k: v for k, v in overrides.items() if v is not None})
    now = _now()
    current = week_at(now, config)

    contents = week_contents(config)
    by_name = {row["week"]: row for row in contents["weeks"]}

    rows = []
    seen = set()

    prefix = str(config.get("week_prefix") or "week_")

    # Everything that already exists, in order, whether or not the
    # arithmetic would still place anything there.
    for row in contents["weeks"]:
        # A week DEFINED under a prefix that is no longer in use, with no
        # folder behind it, is a leftover -- not a week. Renaming the
        # prefix leaves its old rows in the table, and drawing them puts
        # a `day_1` in a list of `meow_N` that nothing will ever fill.
        #
        # Only ever dropped when there is no folder: a week holding real
        # files keeps its row whatever it is called, because the files
        # are the thing that matters and hiding them would be a lie.
        if not row.get("roots") and not row["week"].startswith(prefix):
            continue
        bounds = {}
        digits = "".join(ch for ch in row["week"] if ch.isdigit())
        if digits:
            index = int(digits)
            # Reconstruct this week's bounds from its index, so a folder
            # created under an older cut-off still reports honestly.
            bounds = _bounds_for_index(index, config)
        rows.append({
            **row,
            **bounds,
            "state": "current" if row["week"] == current["name"] else "filled",
            # Whether a FOLDER is there, not whether a row is. Rows come
            # from two places: folders found in the bucket, which carry
            # the roots they were found under, and week definitions in
            # the table, which carry none. Calling the second kind
            # "present" made the state column claim a folder existed for
            # a week that had only ever been defined -- and a week
            # defined under an old prefix then sat there as `present`
            # with nothing behind it.
            "exists": bool(row.get("roots")),
        })
        seen.add(row["week"])

    # Any week between 1 and now that nothing was ever filed into. These
    # used to be created as empty folders so the gap stayed visible; the
    # gap is better shown here than written to S3, but it does have to be
    # shown -- a jump from week_2 to week_4 with no week_3 row reads as a
    # numbering bug rather than as a fortnight when nothing arrived.
    for index in range(1, current["index"]):
        name = f"{prefix}{index}"
        if name in seen:
            continue
        rows.append({
            "week": name, "images": 0, "labels": 0, "bytes": 0,
            "roots": [], "defined": False, "exists": False,
            **_bounds_for_index(index, config), "state": "filled",
        })
        seen.add(name)

    # The week being filled now, if nothing has landed in it yet.
    if current["name"] not in seen:
        rows.append({
            "week": current["name"], "images": 0, "labels": 0, "bytes": 0,
            "roots": [], "defined": False, "exists": False,
            "starts_at": current["starts_at"], "ends_at": current["ends_at"],
            "open_start": current["open_start"], "state": "current",
        })
        seen.add(current["name"])

    # And the ones after it, which do not exist yet by definition.
    for step in range(1, max(0, int(ahead)) + 1):
        index = current["index"] + step
        bounds = _bounds_for_index(index, config)
        name = f"{prefix}{index}"
        if name in seen:
            continue
        rows.append({
            "week": name, "images": 0, "labels": 0, "bytes": 0,
            "roots": [], "defined": False, "exists": False,
            **bounds, "state": "future",
        })
        seen.add(name)

    def order(row):
        digits = "".join(ch for ch in row["week"] if ch.isdigit())
        return (int(digits) if digits else 0, row["week"])

    rows.sort(key=order)

    # Every week closes, including week 1 -- its closing moment IS the
    # cut-off. Naming it on the row rather than leaving the card to infer
    # it from `ends_at` means one week's boundary is stated once, and the
    # same field answers it for week 1 and for week 9.
    for row in rows:
        row["closes_at"] = row.get("ends_at") or ""
        row["is_cutoff"] = bool(row.get("open_start"))
        # EMPTY means the folder is there and holds nothing -- something
        # you can look in, and delete. A week with no folder at all is
        # not empty, it is waiting, and calling both "empty" claimed
        # twenty folders existed that did not.
        if row["state"] == "filled" and not (row.get("images") or row.get("labels")):
            # EMPTY means the folder is there and holds nothing -- something
            # you can look in, and delete. MISSING means it was never
            # created. Calling both "empty" claimed twenty folders existed
            # that did not.
            row["state"] = "empty" if row.get("exists") else "missing"

    # A cheap way for the card to poll: if this is unchanged, nothing has
    # landed since the last look and there is nothing to redraw.
    stamp = hashlib.sha256(
        "|".join(f"{r['week']}:{r.get('images', 0)}:{r.get('labels', 0)}" for r in rows)
        .encode("utf-8")
    ).hexdigest()[:16]

    return {
        "weeks": rows,
        "current": current["name"],
        # Where the next pull will put files, which is not always the
        # week the clock is in -- see `active_week`.
        "active": active_week(config, contents, now),
        "cutoff": str(config.get("week_cutoff") or ""),
        "week_length_days": int(config.get("week_length_days") or DEFAULT_WEEK_DAYS),
        "roots": contents["roots"],
        "now": now.isoformat(),
        "stamp": stamp,
        "total_images": sum(int(r.get("images") or 0) for r in rows),
        "total_labels": sum(int(r.get("labels") or 0) for r in rows),
        "schedule": schedule_plan(config, now),
        # True when the dates shown are for settings that have not been
        # applied, so the card can say so rather than implying the weeks
        # already run to these boundaries.
        "previewing": bool(overrides),
        # How many week folders already exist. Re-dating them is the
        # consequence of changing the week length, and it is only worth
        # warning about when there is something to re-date.
        "existing_weeks": sum(1 for r in rows if r.get("exists")),
    }


def gap_weeks(config: dict | None = None, contents: dict | None = None) -> list:
    """Empty weeks that sit BETWEEN weeks holding files.

    Not every empty week. The stretch after the last collection is not a
    gap, it is simply the present -- and creating a folder for each week
    that passes would fill the bucket during any quiet period. A week
    between two that hold files is different: nothing arrived that week,
    and the folder records that as a fact of the collection rather than
    leaving a hole for someone to wonder about.
    """
    config = config or read_config()
    contents = contents or week_contents(config)
    prefix = str(config.get("week_prefix") or "week_")

    def index_of(name: str) -> int:
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0

    holding = sorted(
        index_of(row["week"]) for row in contents.get("weeks") or []
        if (row.get("images") or row.get("labels")) and row["week"].startswith(prefix)
    )
    if len(holding) < 2:
        # Nothing to be between.
        return []
    present = set(holding)
    return [f"{prefix}{i}" for i in range(holding[0] + 1, holding[-1])
            if i not in present]


def create_gap_folders(config: dict | None = None) -> dict:
    """Lays down marker folders for the gaps `gap_weeks` finds.

    S3 has no directories: a zero-byte key ending in `/` is the only
    thing there is to create. Only the splits the root actually receives
    get one -- `splits_per_root` knows which, and a `val/` folder under a
    root that never sees validation data would be the same clutter this
    deliberately avoids elsewhere.

    Idempotent: writing the same marker twice is the same marker.
    """
    config = config or read_config()
    names = gap_weeks(config)
    if not names:
        return {"created": [], "keys": 0}

    into = destination_bucket(config)
    images_sub = str(config.get("images_subpath") or "images").strip("/")
    labels_sub = str(config.get("labels_subpath") or "labels").strip("/")
    splits = splits_per_root(config)

    keys = 0
    for root, sides in sorted(splits.items()):
        if not root:
            continue
        for name in names:
            for sub in (images_sub, labels_sub):
                if not sub:
                    continue
                tails = sorted(sides) if len(sides) > 1 else [""]
                for tail in tails:
                    key = f"{root}/{name}/{sub}/{tail}/" if tail else f"{root}/{name}/{sub}/"
                    try:
                        s3.put_object(Bucket=into, Key=key, Body=b"")
                        keys += 1
                    except ClientError:
                        pass
    return {"created": names, "keys": keys}


def _bounds_for_index(index: int, config: dict) -> dict:
    """When week N runs, from its number alone."""
    length = _dt.timedelta(days=max(int(config.get("week_length_days") or DEFAULT_WEEK_DAYS), 1))
    cutoff_text = str(config.get("week_cutoff") or "").strip()
    if not cutoff_text:
        return {"starts_at": "", "ends_at": "", "open_start": index <= 1}
    try:
        cutoff = _parse_time(cutoff_text)
    except ValueError:
        return {"starts_at": "", "ends_at": "", "open_start": index <= 1}

    if index <= 1:
        return {"starts_at": "", "ends_at": cutoff.isoformat(), "open_start": True}
    start = cutoff + (index - 2) * length
    return {
        "starts_at": start.isoformat(),
        "ends_at": (start + length).isoformat(),
        "open_start": False,
    }


# --------------------------------------------------------------------------
# Triggers
#
# Two EventBridge rules exist; at most one is enabled. Which one is the
# "on a schedule, or when items arrive" part of this, and it is held on
# the rules themselves rather than only in the settings -- so turning
# routing off actually stops it, instead of leaving a rule firing into a
# function that has been told to do nothing.
# --------------------------------------------------------------------------

APP_NAME = os.environ.get("PULL_CONTROL_APP_NAME", "track2-app")
ARRIVAL_RULE = os.environ.get("PULL_CONTROL_ARRIVAL_RULE", f"{APP_NAME}-pull-control-arrival")
SCHEDULE_RULE = os.environ.get("PULL_CONTROL_SCHEDULE_RULE", f"{APP_NAME}-pull-control-schedule")
ROUTER_FUNCTION = os.environ.get("PULL_CONTROL_ROUTER", f"{APP_NAME}-pull-control-router")

_events = None


def events():
    """Lazily built: the Lambda never touches EventBridge."""
    global _events
    if _events is None:
        _events = boto3.client("events", region_name=REGION)
    return _events


def rule_state(name: str) -> dict:
    """One rule as the card shows it, or why it could not be read."""
    try:
        got = events().describe_rule(Name=name)
        return {
            "name": name,
            "exists": True,
            "enabled": got.get("State") == "ENABLED",
            "schedule": got.get("ScheduleExpression", ""),
        }
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        return {
            "name": name,
            "exists": False,
            "enabled": False,
            "schedule": "",
            # Not created yet is the expected state before the stack is
            # applied, and is worth saying plainly rather than showing as
            # an innocent "off".
            "error": "not created yet" if code == "ResourceNotFoundException" else str(exc),
        }


def trigger_status() -> dict:
    """What is actually armed, per trigger.

    The schedule comes from EventBridge SCHEDULER, not the classic rule of
    the same purpose. Both exist: the rule is the older mechanism and is
    left disabled, because `rate()` on a rule is anchored to whenever the
    rule was last written and its cron is UTC with no start date -- which
    cannot express "first pull at this moment, then every N days" in SGT.
    `set_mode` has armed the Scheduler for a long time.

    This function still read the RULE, so the card reported
    "schedule: disabled" while a Scheduler schedule was enabled and
    counting down. The setting was right, the sweep would have fired, and
    the one line on the page whose job is to say so was reading a
    mechanism nothing uses any more.

    Reported under the keys `rule_state` uses, so the card and the toolbar
    read it unchanged.
    """
    live = schedule_status()
    return {
        "arrival": rule_state(ARRIVAL_RULE),
        "schedule": {
            "name": SCHEDULE_NAME,
            "exists": bool(live.get("exists")),
            "enabled": bool(live.get("enabled")),
            "schedule": str(live.get("expression") or ""),
            # The DATE it next fires, not just the cadence. "rate(7 days)"
            # does not say which 7 days: the grid is counted from the week
            # cut-off, so setting a time of 15:43 on the 5th arms the 7th
            # at 15:43 and the card said only "armed" -- which reads as
            # "today" to everyone who has just pressed Apply.
            "next_run": (next_pull_times(count=1) or [""])[0],
            "error": str(live.get("error") or ""),
        },
    }


SCHEDULE_NAME = os.environ.get("PULL_CONTROL_SCHEDULE", f"{APP_NAME}-pull-control")

#: How long after a scheduled time of day it still counts as "now" rather
#: than "missed, see you tomorrow". Five minutes, because the thing being
#: forgiven is a person typing a time and pressing Apply: they pick the
#: current minute, or the one just gone while they were still reading the
#: form. Anything longer starts guessing at intent -- 09:00 set at 09:30
#: is a schedule for tomorrow morning, not a request to run immediately.
SCHEDULE_GRACE_SECONDS = 300

_scheduler = None


def scheduler():
    """EventBridge Scheduler, not EventBridge Rules.

    A classic rule cannot express "first pull at this moment, then every
    N days": `rate()` on a rule is anchored to whenever the rule was last
    written, and its cron is UTC with no start date. Scheduler has both a
    start date and a timezone, which is what a pull time you can actually
    choose requires.
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = boto3.client("scheduler", region_name=REGION)
    return _scheduler


#: Singapore is UTC+8 and has had no daylight saving since 1982, so a
#: fixed offset is exact rather than an approximation that drifts twice a
#: year. The card shows and accepts SGT, so replacing a time of day has
#: to happen in SGT too -- doing it in UTC would move an early-morning
#: pull onto the previous date.
SGT = _dt.timezone(_dt.timedelta(hours=8))


def parse_time_of_day(text) -> tuple | None:
    """"HH:MM" as `(hour, minute)`, or None if it is not a time."""
    text = str(text or "").strip()
    if not text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _at_time_of_day(moment: _dt.datetime, text) -> _dt.datetime:
    """The same Singapore day as `moment`, at "HH:MM".

    Returns `moment` untouched for a blank or unparseable time, so a
    half-typed box never silently reschedules anything. `set_mode`
    rejects an unparseable time outright rather than relying on that.
    """
    parsed = parse_time_of_day(text)
    if parsed is None:
        return moment
    hour, minute = parsed
    local = moment.astimezone(SGT).replace(
        hour=hour, minute=minute, second=0, microsecond=0)
    return local.astimezone(_dt.timezone.utc)


def pull_interval_days(config: dict | None = None) -> int:
    """Days between scheduled pulls. At least one."""
    config = config or read_config()
    try:
        return max(int(config.get("schedule_every_days") or 7), 1)
    except (TypeError, ValueError):
        return 7


def next_pull_times(config: dict | None = None, count: int = 5,
                    now: _dt.datetime | None = None) -> list:
    """The next few moments a scheduled pull would run.

    Computed here rather than read back from AWS so the card can show
    what a setting WOULD do before it is applied. `set_mode` sends the
    same arithmetic to Scheduler, so what is shown and what is armed are
    the same series.
    """
    config = config or read_config()
    now = now or _now()
    every = _dt.timedelta(days=pull_interval_days(config))

    # A cut-off is still REQUIRED -- `set_mode` refuses a schedule without
    # one, because the week folders are measured from it and a pull would
    # have nowhere to file. Returning times for a config that cannot be
    # armed would be a schedule nobody can have.
    cutoff_text = str(config.get("week_cutoff") or "").strip()
    if not cutoff_text:
        return []
    try:
        cutoff = _parse_time(cutoff_text)
    except ValueError:
        return []

    # The DATE is counted from NOW; only the TIME OF DAY comes from the
    # settings.
    #
    # It used to anchor the whole series to `week_cutoff + every` and roll
    # forward by whole intervals, which put every pull on a grid fixed by
    # a date chosen for a different purpose. Set "every 7 days" on the 5th
    # against a cut-off in March and the next pull was the 7th: the grid
    # did not happen to include today, and nothing said why. The cut-off
    # decides which WEEK a file is filed into; it has no business deciding
    # when the collecting happens, and those two questions were sharing
    # one answer.
    #
    # The consequence, stated because it was the old behaviour's whole
    # point: changing the time of day on a running schedule now re-anchors
    # it from today rather than keeping the original grid.
    #
    # A blank time still means the CUT-OFF's own hour -- one clock, not
    # two -- which is the part of the old design worth keeping.
    wanted = str(config.get("schedule_time") or "").strip()
    if not wanted:
        local = cutoff.astimezone(SGT)
        wanted = f"{local.hour:02d}:{local.minute:02d}"
    first = _at_time_of_day(now, wanted)

    # The interval is a whole number of days and Singapore has no daylight
    # saving, so every later pull keeps this exact wall-clock time.
    #
    # The GRACE is the whole point of this branch. Somebody types 12:37
    # and presses Apply, and the Apply lands at 12:37:00.027 -- twenty-
    # seven MILLISECONDS after the time they asked for. A bare
    # `first <= now` reads that as missed and defers the run a full day,
    # which is precisely what happened on 2026-10-06: StartDate came back
    # as the SEVENTH at 12:37 and the question was "I set it at 12:37,
    # why did it not fire?". Nothing in the UI could have explained it,
    # because from outside the two times are the same minute.
    #
    # So a time of day that has only just passed is treated as still
    # wanted, and scheduled for the next whole minute rather than
    # tomorrow. Beyond the grace it genuinely is tomorrow -- setting
    # 09:00 at 16:00 means tomorrow morning, not "in a moment".
    if first <= now:
        missed_by = (now - first).total_seconds()
        if missed_by <= SCHEDULE_GRACE_SECONDS:
            # Rounded up to the next minute: Scheduler rejects a start
            # date in the past, and "now" is in the past by the time the
            # call arrives.
            first = (now + _dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
        else:
            first = first + _dt.timedelta(days=1)

    # "Every N days FROM THIS DATE": the series sits on the grid that
    # starts at the chosen date, not on today. Rolled forward by whole
    # intervals to the first occurrence still ahead -- the same grace as
    # above, so pressing Apply in the very minute of a grid point still
    # catches it.
    anchor = schedule_anchor(config, wanted)
    if anchor is not None:
        first = anchor
        if first < now - _dt.timedelta(seconds=SCHEDULE_GRACE_SECONDS):
            behind = now - first
            steps = -(-behind // every)  # ceiling, in whole intervals
            first = first + steps * every
        if first <= now:
            first = (now + _dt.timedelta(minutes=1)).replace(second=0, microsecond=0)                 if (now - first).total_seconds() <= SCHEDULE_GRACE_SECONDS else first + every

    return [(first + step * every).isoformat() for step in range(max(int(count), 0))]


def schedule_anchor(config: dict, wanted: str = "") -> _dt.datetime | None:
    """The chosen start date at the schedule's time of day, in UTC.

    None unless "every N days from this date" is selected and the date
    parses -- in which case the series counts from today, as before.
    """
    if str(config.get("schedule_from") or "today") != "date":
        return None
    text = str(config.get("schedule_start_date") or "").strip()
    try:
        day = _dt.date.fromisoformat(text)
    except ValueError:
        return None
    parsed = parse_time_of_day(wanted or str(config.get("schedule_time") or "")) or (0, 0)
    local = _dt.datetime(day.year, day.month, day.day, parsed[0], parsed[1], tzinfo=SGT)
    return local.astimezone(_dt.timezone.utc)


def schedule_plan(config: dict | None = None, now: _dt.datetime | None = None) -> dict:
    """What the configured schedule means, in weeks rather than in cron.

    The useful question about a pull interval is not "when does it fire"
    but "which week folders will one firing fill" -- because an interval
    longer than a week fills several, and that only works if files are
    dated individually.
    """
    config = config or read_config()
    now = now or _now()
    every = pull_interval_days(config)
    length = max(int(config.get("week_length_days") or DEFAULT_WEEK_DAYS), 1)
    upcoming = next_pull_times(config, count=5, now=now)

    covers: list = []
    if upcoming:
        fires = _parse_time(upcoming[0])
        # One pull collects whatever appeared since the pull before it,
        # so it reaches back a whole interval.
        window_start = fires - _dt.timedelta(days=every)
        first_week = week_at(window_start, config)
        last_week = week_at(fires - _dt.timedelta(seconds=1), config)
        for index in range(first_week["index"], last_week["index"] + 1):
            bounds = _bounds_for_index(index, config)
            covers.append({
                "week": f"{str(config.get('week_prefix') or 'week_')}{index}",
                "index": index, **bounds,
            })

    # Per-FILE dating, whichever clock it reads. Only `scan` puts a whole
    # sweep in one week, and only `scan` therefore misfiles when a pull
    # spans several. Testing for `file_time` alone made capture_time look
    # like the batch mode it is the opposite of.
    dated_by = str(config.get("week_by") or "capture_time")
    per_file = dated_by in ("file_time", "capture_time")
    cutoff = str(config.get("week_cutoff") or "").strip()
    return {
        "every_days": every,
        "week_length_days": length,
        # What the series is counted from. The card shows it so the two
        # settings are visibly one clock, not two that happen to agree.
        "anchor": cutoff,
        # "" when pulls land on the cut-off's own time, which is the
        # default; otherwise the "HH:MM" they were moved to.
        "time_of_day": str(config.get("schedule_time") or "").strip(),
        "start": upcoming[0] if upcoming else "",
        # Without a cut-off there is nothing to count from, so a schedule
        # cannot be armed at all.
        "ready": bool(upcoming),
        "next_runs": upcoming,
        # More than one week per pull is fine, but only when files are
        # dated individually. Under `scan` they would all land in one.
        "weeks_per_pull": len(covers),
        "covers": covers,
        "week_by": dated_by,
        "spans_multiple_weeks": len(covers) > 1,
        "misfiling_risk": len(covers) > 1 and not per_file,
    }


def schedule_status() -> dict:
    """The live Scheduler schedule, if terraform has made it."""
    try:
        got = scheduler().get_schedule(Name=SCHEDULE_NAME)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code", "")
        if code in ("ResourceNotFoundException", "AccessDeniedException"):
            return {"exists": False, "enabled": False, "error": code}
        return {"exists": False, "enabled": False, "error": str(exc)}
    start = got.get("StartDate")
    return {
        "exists": True,
        "enabled": got.get("State") == "ENABLED",
        "expression": got.get("ScheduleExpression", ""),
        "timezone": got.get("ScheduleExpressionTimezone", ""),
        "start": start.isoformat() if start else "",
        "error": "",
    }


def set_mode(*, mode: str, schedule_expression: str = "",
             schedule_every_days: int | None = None,
             schedule_time: str | None = None,
             schedule_from: str | None = None,
             schedule_start_date: str | None = None) -> dict:
    """Switches routing off, onto arrivals, or onto a timer.

    Exactly one trigger ends up live. The others are turned off first:
    two at once means every arrival fires a sweep the timer is also
    running, which is worse than the momentary gap the other order would
    leave.

    The timer is an EventBridge **Scheduler** schedule, not the old
    `pull-control-schedule` rule -- that rule is disabled here on every
    call so a schedule set before this change cannot keep firing
    alongside the new one.
    """
    mode = (mode or "off").strip().lower()
    if mode not in {"off", "arrival", "schedule"}:
        raise ValueError("Mode must be off, arrival or schedule.")

    config = read_config()
    patch: dict = {"mode": mode}
    if schedule_every_days is not None:
        patch["schedule_every_days"] = max(int(schedule_every_days), 1)
    if schedule_time is not None:
        wanted = str(schedule_time).strip()
        # Rejected rather than ignored. A time that quietly fell back to
        # the cut-off's would be a schedule running at a different hour
        # than the one on screen, which is the worst of both.
        if wanted and parse_time_of_day(wanted) is None:
            raise ValueError(
                f"{wanted!r} is not a time -- use HH:MM, 24-hour, e.g. 02:00."
            )
        patch["schedule_time"] = wanted
    if schedule_from is not None:
        if schedule_from not in ("today", "date"):
            raise ValueError("Count the schedule from today or from a date.")
        patch["schedule_from"] = schedule_from
    if schedule_start_date is not None:
        day = str(schedule_start_date).strip()
        if day:
            try:
                _dt.date.fromisoformat(day)
            except ValueError:
                raise ValueError(f"{day!r} is not a date -- use YYYY-MM-DD.") from None
        patch["schedule_start_date"] = day
    if patch.get("schedule_from", config.get("schedule_from")) == "date"             and not str(patch.get("schedule_start_date", config.get("schedule_start_date")) or "").strip():
        raise ValueError("Pick the date the schedule counts from, or choose \"every N days\".")
    merged = {**config, **patch}

    # The cadence no longer counts from the cut-off -- it counts from
    # now -- but a pull still has to FILE what it collects, and the week
    # a file belongs to is measured from the cut-off. Without one there
    # are no week boundaries to file into, so the sweep would run and
    # have nowhere to put anything.
    if mode == "schedule" and not str(merged.get("week_cutoff") or "").strip():
        # ...unless the pull has nothing to file.
        #
        # "Use these folders only" in Assemble a dataset switches Pull
        # Control, Auto Training and Retention OFF -- the card names its
        # folders outright and no sweep files anything into week folders.
        # Demanding a week cut-off in that state is demanding
        # configuration for a mechanism the operator has explicitly
        # turned off, and it made scheduled TRAINING unreachable: the
        # chain is triggered by the pull schedule, and the schedule
        # could not be applied without a date for weeks nobody was
        # using.
        #
        # The refusal stays wherever the sweep would actually file: untick
        # "only" and the next Apply asks for the cut-off again.
        assemble_only = bool((merged.get("dataset_assemble") or {}).get("only"))
        if not assemble_only:
            raise ValueError(
                "Set a cut-off date first -- it is what the week folders are "
                "measured from, and a pull has nowhere to file into without it."
            )

    every = pull_interval_days(merged)
    expression = f"rate({every} day{'' if every == 1 else 's'})"
    if schedule_expression.strip():
        # An expression typed by hand wins, so anything Scheduler accepts
        # is still reachable without a field per cron component.
        expression = schedule_expression.strip()
    patch["schedule_expression"] = expression

    problems = []

    # The hole the waiver above leaves, said out loud rather than left to
    # be discovered. Without a cut-off every file falls into week 1 --
    # `_week_for` buckets anything before the cut-off there, and with no
    # cut-off that is everything. Harmless while nothing is being swept;
    # a corpus silently collapsed into one week the moment something is.
    if (mode == "schedule"
            and not str(merged.get("week_cutoff") or "").strip()
            and any((merged.get("picked") or {}).get(key)
                    for key in ("train_images", "train_labels",
                                "val_images", "val_labels"))):
        problems.append(
            "Scheduled with no cut-off date: source folders ARE picked, so "
            "every file this pull moves will land in week 1. Set a cut-off "
            "date, or untick the source folders."
        )

    # Arrivals, and the superseded rule, both off unless wanted.
    for name, wanted in ((ARRIVAL_RULE, mode == "arrival"), (SCHEDULE_RULE, False)):
        if wanted:
            continue
        try:
            events().disable_rule(Name=name)
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") != "ResourceNotFoundException":
                problems.append(f"{name}: {exc}")

    if mode == "arrival":
        try:
            events().enable_rule(Name=ARRIVAL_RULE)
        except ClientError as exc:
            problems.append(f"{ARRIVAL_RULE}: {exc}")

    # The schedule is updated whether or not it is the live mode, so the
    # times shown on the card are the times that are armed the moment it
    # is switched on.
    try:
        existing = scheduler().get_schedule(Name=SCHEDULE_NAME)
    except ClientError as exc:
        existing = None
        code = exc.response.get("Error", {}).get("Code", "")
        if mode == "schedule":
            problems.append(
                f"{SCHEDULE_NAME}: {code or exc} -- apply terraform/app to create it"
            )

    if existing is not None:
        kwargs = {
            "Name": SCHEDULE_NAME,
            "ScheduleExpression": expression,
            "ScheduleExpressionTimezone": existing.get(
                "ScheduleExpressionTimezone", "Asia/Singapore"),
            "FlexibleTimeWindow": existing["FlexibleTimeWindow"],
            "Target": existing["Target"],
            "State": "ENABLED" if mode == "schedule" else "DISABLED",
        }
        if existing.get("GroupName"):
            kwargs["GroupName"] = existing["GroupName"]

        # Scheduler rejects a start date in the past, and it would be a
        # lie anyway -- the first pull is the next one due, not one that
        # has been missed. next_pull_times does that rolling forward.
        upcoming = next_pull_times(merged, count=1)
        if upcoming:
            kwargs["StartDate"] = _parse_time(upcoming[0])

        try:
            scheduler().update_schedule(**kwargs)
        except ClientError as exc:
            problems.append(f"{SCHEDULE_NAME}: {exc}")

    # ONE schedule row, both clocks.
    #
    # This strip is the only cadence on screen, and it has to mean the
    # same thing whichever mechanism is in charge. When the training
    # clock owns it -- "Use these folders only", where no sweep runs --
    # the pull's `rate(N days)` decides nothing, so the same interval
    # and time are mirrored onto `retrain_schedule` and that clock is
    # armed or disarmed with the mode.
    #
    # Mirrored rather than READ from here at fire time: the clock is in
    # a different module with its own defaults, and reaching across to
    # the pull's keys would leave two meanings for `every_days`
    # depending on who asked.
    retrain = {**(config.get("retrain_schedule") or {})}
    every_days = int(patch.get("schedule_every_days")
                     or merged.get("schedule_every_days") or 1)
    at_time = str(patch.get("schedule_time") or merged.get("schedule_time") or "")
    retrain["every_days"] = max(1, every_days)
    if at_time:
        retrain["at"] = at_time
    # The same anchor for the training clock, so both clocks keep one grid.
    retrain["from_date"] = (str(merged.get("schedule_start_date") or "")
                            if merged.get("schedule_from") == "date" else "")
    # OFF is off for both. Leaving the training clock armed while the
    # strip reads "Off" is the card saying one thing and the system
    # doing another.
    retrain["enabled"] = mode == "schedule"
    # The training clock's grid. Inline import: retrain_schedule imports
    # this module. Re-anchored only when the clock is armed here or its
    # interval/time changed -- pressing Apply again with the same values
    # must not slide the grid, which is what let a weekly schedule fall
    # due every day.
    from . import retrain_schedule as _rs
    before = {**_rs.DEFAULTS, **(config.get("retrain_schedule") or {})}
    retrain = _rs.with_anchor(before, retrain)
    patch["retrain_schedule"] = retrain

    write_config(patch)

    # Arming is not running. The fresh anchor already puts the first slot
    # ahead of now; an operator-chosen start date can still have one a
    # few minutes behind, so claim it -- the same guard the retrain card's
    # own Save has.
    if retrain["enabled"] and not before.get("enabled"):
        try:
            slot = _rs.due_training_run({**merged, "retrain_schedule": retrain})
            if slot:
                claim_once("training_schedule", slot)
        except Exception as exc:                               # noqa: BLE001
            problems.append(f"Could not mark the slot that just passed as handled: {exc}")
    return {
        "mode": mode,
        "schedule_expression": expression,
        "schedule_every_days": every,
        "schedule_time": str(merged.get("schedule_time") or ""),
        "triggers": trigger_status(),
        "schedule": schedule_status(),
        "plan": schedule_plan(merged),
        "problems": problems,
    }


def status() -> dict:
    """Everything the card needs, in one round trip."""
    config = read_config()
    try:
        weeks = list_weeks()
        table_ready, table_error = True, ""
    except ClientError as exc:
        weeks, table_ready = [], False
        table_error = (
            "the settings table does not exist yet -- apply terraform/app"
            if exc.response.get("Error", {}).get("Code") == "ResourceNotFoundException"
            else str(exc)
        )
    triggers = trigger_status()
    return {
        "config": config,
        "weeks": weeks,
        "next_week_name": next_week_name(config, weeks),
        "runs": recent_runs() if table_ready else [],
        "triggers": triggers,
        "schedule": schedule_status(),
        "plan": schedule_plan(config),
        "ready": table_ready and triggers["arrival"]["exists"],
        "error": table_error,
        "router": router_freshness(),
    }
