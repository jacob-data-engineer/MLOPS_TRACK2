"""Exporting the best registered checkpoint of one Class -- by hand, or on a watch.

What goes out is the **.pt itself**, not a spreadsheet about it. "The best
version of the file" is a thing you deploy or hand to somebody; a CSV row
describing it is not, and the two were conflated while this card exported
metrics.

Three decisions shape everything below, and each was asked and answered
rather than assumed:

  * **Which version is best** is decided by ONE criterion, chosen in the
    card -- a metric, highest wins, or simply the newest version.
  * **There are no eligibility guards.** Approval status and validation
    count do not exclude a version. That is a deliberate choice, and it
    has teeth on this estate: `yolo11s-helmet-detector` v2, v3 and v4 all
    score 0.995 mAP50 measured on FIVE validation images, against v9's
    0.000 on 149. Ranked on mAP50 with no floor, a five-image score wins
    every time. So every result this module returns STATES the validation
    count and the approval status of what it picked. Not gating is a
    decision; hiding it would be a defect.
  * **Nothing is ever overwritten.** Each export is a new, timestamped
    object. A fixed name would be fetchable without a listing, but it
    would also mean an automatic export could silently replace the file
    somebody downstream was already using.

WHY THE WATCHER LIVES IN THE APP. Same reason `queue_after_pull` does: it
is a poll on the process's own 25s tick, and the thing it watches -- a new
version appearing in a model package group -- is written by SageMaker,
from a pipeline, with nothing to hook. One `list_model_packages` per tick
is the whole steady-state cost; the expensive part only runs on the tick
after a version actually appears.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time

from . import track2_shared as t2

#: Which metric decides, plus the one non-metric answer. `newest` is here
#: because it is a real answer to "which should go out" -- for a Class
#: trained on a growing corpus the newest model is the one trained on the
#: most data, whatever it scored against a validation set that also moved.
CRITERIA = ("mAP50", "mAP50-95", "precision", "recall", "f1", "accuracy", "newest")

#: What happens to the EARLIER exports of the same Class in the
#: destination folder, once a new one lands.
#:
#: Every mode writes a new timestamped object first. They differ only in
#: what they then do about the ones already there, which is a retention
#: decision and belongs to whoever owns the folder:
#:
#:   keep_all        leave them. The folder is the history.
#:   archive_oldest  move them into <folder>/archive/. Still there,
#:                   out of the way, and a consumer listing the folder
#:                   sees one file.
#:   keep_latest     delete them. One file, no history.
#:
#: `keep_latest` is the only irreversible one, and it is scoped -- see
#: `export_siblings`.
RETENTION_MODES = ("keep_all", "archive_oldest", "keep_latest")

#: Where `archive_oldest` puts them, under the destination folder.
ARCHIVE_FOLDER = "archive"

#: The armed watcher's own file. Underscore-prefixed on purpose: the
#: preset name rule (`SAFE_NAME` in the router) requires a letter or digit
#: first, so no preset can ever be called this, and the preset listing
#: already drops any file whose stem fails that rule. The watcher's state
#: therefore cannot appear in, or be clobbered by, the Load/Save dropdown.
AUTO_FILE = "_auto_export.json"

AUTO_DEFAULTS = {
    #: Off. An unattended job that writes to a shared bucket is asked
    #: for, not arrived at by loading a preset -- which is exactly why
    #: arming is its own call and is NOT part of the saved settings.
    "enabled": False,
    "group_name": "",
    "criterion": "mAP50",
    #: S3 only, and the arm call refuses anything else. Nothing running
    #: unattended in a container can download to somebody's laptop.
    "bucket": "",
    "prefix": "",
    #: What happens to earlier exports once a new one lands. Carried by
    #: the WATCHER, not only by the card: an unattended export that
    #: tidied differently from the manual one would leave a folder whose
    #: contents depended on which path last wrote to it.
    "retention": "keep_all",
    #: The highest version number already seen in that group. Set to the
    #: group's current newest AT ARMING TIME, so switching the watcher on
    #: does not immediately export for a version registered last week.
    "last_version": 0,
    #: The version the watcher last actually exported. A new version
    #: registering does not mean the WINNER changed -- v15 scoring 0.3
    #: against v4's 0.995 leaves v4 the best -- and re-exporting the same
    #: checkpoint every time a run finishes would pile up byte-identical
    #: timestamped copies of one file. So the trigger is "a new version
    #: appeared", and the write is "and it changed the answer".
    "last_exported_version": "",
    #: What happened last time, for the card to show. A background action
    #: with no visible outcome is one nobody trusts.
    "last_result": {},
    "armed_at": "",
}


def _settings_dir() -> str:
    """Where presets and the watcher's file live.

    Read off the ROUTER module at call time rather than imported at module
    scope: the tests monkeypatch `track2_metrics.EXPORT_SETTINGS_DIR` to a
    tmp_path, and a value copied at import would ignore them. The lazy
    import also keeps the dependency one-way -- the router imports this
    module, not the other way round.
    """
    from .routers import track2_metrics

    return track2_metrics.EXPORT_SETTINGS_DIR


# --- reading one group --------------------------------------------------

def registry_packages(group_name: str) -> list[tuple[int, str]]:
    """(version number, package ARN) for every version in a group, highest first.

    PAGINATED, unlike `model_registry_summary`, which passes no MaxResults
    and so takes whatever the service hands back. That is tolerable there
    -- it sorts newest-first and only ever shows the top few -- and not
    tolerable here, where a truncated list means "the best version" is
    the best of whichever versions happened to come back.

    The ARN comes off the summary rather than being rebuilt from the group
    name and the number: an ARN carries the partition, region and account,
    and reconstructing it is three chances to be subtly wrong about a
    string the registry already handed over.
    """
    found: list[tuple[int, str]] = []
    token = ""
    while True:
        kwargs = {"ModelPackageGroupName": group_name, "MaxResults": 100}
        if token:
            kwargs["NextToken"] = token
        try:
            page = t2.sm.list_model_packages(**kwargs)
        except Exception as exc:                                   # noqa: BLE001
            # A group that has not been created yet is "nothing
            # registered", the same reading `model_registry_summary`
            # takes. Anything else is a real failure and is raised.
            if "does not exist" in str(exc):
                return []
            raise
        for summary in page.get("ModelPackageSummaryList", []):
            arn = summary.get("ModelPackageArn", "")
            try:
                found.append((int(arn.rsplit("/", 1)[-1]), arn))
            except ValueError:
                continue
        token = page.get("NextToken") or ""
        if not token:
            break
    found.sort(key=lambda pair: pair[0], reverse=True)
    return found


def newest_version(group_name: str) -> int:
    """The highest version number in a group, or 0. The watcher's per-tick cost."""
    packages = registry_packages(group_name)
    return packages[0][0] if packages else 0


#: `group_versions` results, keyed on group name, with the monotonic time
#: they were read. See `max_age` below.
_versions_cache: dict[str, tuple[float, list]] = {}


def group_versions(group_name: str, *, max_age: float = 0.0) -> list[dict]:
    """Every version of ONE group, with its metrics -- rows shaped like
    `scan_all_model_metrics` produces, but for one group rather than the
    whole estate.

    Scoped deliberately. `scan_all_model_metrics` walks every production
    group, every open group and every detector's groups, describing each
    package and fetching each evaluation report. Picking the best version
    of one Class needs none of that, and this runs on a poll.

    `max_age` is opt-in and defaults to OFF, which is the important part.
    This costs one describe plus one S3 read per version -- 1.7s and ~28
    calls for the 14-version helmet group -- and the card calls it every
    time the criterion dropdown moves, where a few seconds of staleness
    is invisible and worth the saving. The WATCHER must never take it:
    it runs precisely because a new version just appeared, so a cached
    answer is the one answer it cannot use, and it would decide "no new
    winner" off a list that predates the thing that woke it.
    """
    if max_age > 0:
        cached = _versions_cache.get(group_name)
        if cached and time.monotonic() - cached[0] < max_age:
            return [dict(r) for r in cached[1]]

    generation, size, _rest = t2.parse_group_name(group_name)
    rows = []
    for number, arn in registry_packages(group_name):
        try:
            described = t2.sm.describe_model_package(ModelPackageName=arn)
        except Exception as exc:                                   # noqa: BLE001
            print(f"Could not describe {arn}: {exc}")
            continue
        metrics, description, _lineage = t2.fetch_metrics_and_lineage(arn)
        rows.append({
            "group_name": group_name,
            "generation": generation or "",
            "size": size or "",
            "version": str(number),
            "model_package_arn": arn,
            "status": described.get("ModelApprovalStatus", ""),
            "metrics": metrics,
            "description": description,
        })
    _versions_cache[group_name] = (time.monotonic(), [dict(r) for r in rows])
    return rows


# --- picking ------------------------------------------------------------

def metric_value(row: dict, criterion: str):
    """The number this criterion ranks on, or None if the version has none.

    None is NOT zero. A version with no evaluation report -- an imported
    checkpoint, or a run that died before it evaluated -- has not scored
    badly, it has not been measured, and ranking it as 0.0 would quietly
    put it last in a list where it does not belong at all.
    """
    if criterion == "newest":
        try:
            return int(row.get("version") or 0)
        except (TypeError, ValueError):
            return None
    metrics = row.get("metrics") or {}
    value = metrics.get(criterion)
    return value if isinstance(value, (int, float)) else None


def validation_count(row: dict):
    dataset = ((row.get("metrics") or {}).get("dataset") or {})
    count = dataset.get("validation_count")
    return count if isinstance(count, int) else None


def version_number(row: dict) -> int:
    try:
        return int(row.get("version") or 0)
    except (TypeError, ValueError):
        return 0


def pick_best(rows: list[dict], criterion: str) -> tuple[dict | None, str]:
    """(the winning row, why there is none).

    Highest value wins. A tie goes to the HIGHER VERSION NUMBER, because
    something has to break it and version order is the one ordering that
    is stable and matches how the numbers were handed out -- the
    alternative is a "best version" that changes between two identical
    calls.

    No eligibility filter. Status and validation count are reported by the
    caller, not used to exclude: that was asked and answered.
    """
    if criterion not in CRITERIA:
        return None, f"{criterion!r} is not something a version can be ranked by."
    if not rows:
        return None, "Nothing is registered in this group yet."
    ranked = [(metric_value(r, criterion), version_number(r), r) for r in rows]
    scored = [entry for entry in ranked if entry[0] is not None]
    if not scored:
        return None, (
            f"No version in this group has a recorded {criterion}. Usually "
            "imported checkpoints, or runs that failed before they "
            "evaluated -- not models that scored zero."
        )
    scored.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
    return scored[0][2], ""


def describe_pick(row: dict, criterion: str) -> dict:
    """The winner, stated in full -- including the two things that make a
    high score meaningless, which this module does not gate on."""
    count = validation_count(row)
    return {
        "version": row.get("version", ""),
        "model_package_arn": row.get("model_package_arn", ""),
        "group_name": row.get("group_name", ""),
        "status": row.get("status", ""),
        # A seed's weights detect something other than this Class -- it
        # is a starting point awaiting its first run, not a model of
        # this Class. `criterion="newest"` will pick one, so the fact
        # has to travel with the pick.
        "is_seed": t2.is_seed_description(row.get("description", "")),
        "criterion": criterion,
        "value": metric_value(row, criterion),
        "validation_images": count,
        "classes": (row.get("metrics") or {}).get("classes") or [],
        # Said out loud rather than left to be noticed. Below about fifty
        # images one picture moves mAP by whole percentage points, and
        # this estate has three versions at 0.995 on five.
        "thin_validation": count is not None and count < 50,
        "label": t2.registry_source_desc(
            row.get("model_package_arn", ""), row.get("size", "")),
    }


# --- exporting ----------------------------------------------------------

def export_filename(group_name: str, version: str, when=None) -> str:
    """`yolo11-s-helmet-v7-20261008-091514.pt`.

    Stamped to the second and therefore never colliding, which is the
    whole naming decision: an automatic export must not be able to replace
    a file something downstream already fetched. The stem is the same one
    `export_weights_to_folder` has always used, so exports from both paths
    sort together in a folder and read the same way.
    """
    when = when or _dt.datetime.now(_dt.timezone.utc)
    stamp = when.strftime("%Y%m%d-%H%M%S")
    return f"{t2.export_stem(group_name)}-v{version}-{stamp}.pt"


def export_siblings(bucket: str, prefix: str, group_name: str,
                    *, exclude: str = "") -> list[str]:
    """Earlier exports OF THIS CLASS sitting directly in that folder.

    This is the blast radius of `archive_oldest` and `keep_latest`, and it
    is deliberately narrow. "Delete everything in the folder keeping only
    the latest" is the obvious reading of what these modes do, and it is
    the wrong thing to implement: an export folder is a normal S3 prefix
    that may hold somebody else's data, a README, last quarter's reports.
    A retention rule on a checkpoint export has no business removing any
    of it.

    So four conditions, all of them required:

      * it ends in `.pt`
      * its name starts with this group's own export stem -- so exporting
        the helmet Class never touches the segment Class's files, even in
        the same folder
      * it sits DIRECTLY in the folder, not in a subfolder -- which also
        keeps `archive/` out of its own reach, or archiving would feed
        itself every run
      * it is not the object just written

    Anything else in that folder is somebody else's, and is left alone.
    """
    folder = (prefix or "").strip("/")
    base = f"{folder}/" if folder else ""
    stem = f"{t2.export_stem(group_name)}-v"
    found = []
    token = ""
    while True:
        kwargs = {"Bucket": bucket, "Prefix": base}
        if token:
            kwargs["ContinuationToken"] = token
        page = t2.s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key == exclude:
                continue
            name = key[len(base):]
            if "/" in name:
                continue                     # a subfolder, archive/ included
            if not name.endswith(".pt") or not name.startswith(stem):
                continue
            found.append(key)
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken") or ""
        if not token:
            break
    return sorted(found)


def export_counts(bucket: str, prefix: str, group_name: str) -> dict:
    """How many of this Model Name's .pt exports sit in the folder, and how
    many in its archive/ -- the card's running count (2026-10-26).

    Counted with `export_siblings`, the same rule the housekeeping acts
    on, so the number is exactly what "the rest" refers to: this Model
    Name's files only, directly in that folder (or directly in archive/).
    """
    folder = (prefix or "").strip("/")
    archive = f"{folder}/{ARCHIVE_FOLDER}" if folder else ARCHIVE_FOLDER
    keys = export_siblings(bucket, prefix, group_name)
    # Which version the folder holds now: the NEWEST file by the stamp in
    # its name (`-v21-20261026-202117.pt`). By stamp, not by sorting the
    # names, which would put v9 after v10.
    latest = None
    for key in keys:
        found = _EXPORT_NAME.search(key)
        if found and (latest is None or found.group(2) > latest["stamp"]):
            latest = {"file": key.rsplit("/", 1)[-1], "version": found.group(1),
                      "stamp": found.group(2)}
    return {"in_folder": len(keys),
            "archived": len(export_siblings(bucket, archive, group_name)),
            "latest": latest}


#: `<stem>-v<version>-<YYYYmmdd-HHMMSS>.pt`, as `export_filename` writes it.
_EXPORT_NAME = re.compile(r"-v([0-9]+)-([0-9]{8}-[0-9]{6})[.]pt$")


def apply_retention(bucket: str, prefix: str, group_name: str, *,
                    mode: str, keep_key: str) -> dict:
    """What happens to the earlier exports once the new one has landed.

    Runs AFTER the new object is written, never before. The other order
    leaves a window in which the folder holds nothing at all, and a
    consumer that reads it in that window gets an empty answer rather
    than an old one -- which is the worse failure, because an old
    checkpoint is still a checkpoint.
    """
    if mode not in RETENTION_MODES:
        raise ValueError(f"{mode!r} is not a way of handling earlier exports.")
    if mode == "keep_all":
        return {"mode": mode, "archived": [], "deleted": []}

    folder = (prefix or "").strip("/")
    siblings = export_siblings(bucket, prefix, group_name, exclude=keep_key)
    if not siblings:
        return {"mode": mode, "archived": [], "deleted": []}

    if mode == "archive_oldest":
        archived = []
        for key in siblings:
            name = key.rsplit("/", 1)[-1]
            target = f"{folder}/{ARCHIVE_FOLDER}/{name}" if folder else f"{ARCHIVE_FOLDER}/{name}"
            # Copy THEN delete, and only delete if the copy returned. A
            # delete that ran first, or ran regardless, turns "archive it"
            # into "lose it" the moment the copy is refused.
            t2.s3.copy_object(Bucket=bucket, Key=target,
                              CopySource={"Bucket": bucket, "Key": key})
            t2.s3.delete_object(Bucket=bucket, Key=key)
            archived.append(target)
        return {"mode": mode, "archived": archived, "deleted": []}

    for key in siblings:
        t2.s3.delete_object(Bucket=bucket, Key=key)
    return {"mode": mode, "archived": [], "deleted": siblings}


def export_checkpoint(row: dict, *, destination: str, bucket: str = "",
                      prefix: str = "", retention: str = "keep_all") -> dict:
    """Puts one version's checkpoint where it was asked to go.

    Both destinations go through `publish_versioned_weights`, which
    extracts model.pt out of the registered package's own tarball to a
    stable, immutable key and is a no-op once that key exists. So:

      * **S3** is then a server-side `copy_object` -- the checkpoint never
        travels through this container, which shares one small box with
        every page the app serves.
      * **This computer** is a presigned link, and the browser fetches
        straight from S3 for the same reason. A 19MB-110MB download
        streamed through here would stall every request in this
        single-worker process for its duration.
    """
    arn = row.get("model_package_arn", "")
    group_name = row.get("group_name", "")
    version = str(row.get("version") or "")
    if not arn:
        raise ValueError("That version has no model package to export.")

    uri = t2.publish_versioned_weights(arn)
    source_bucket, source_key = uri.removeprefix("s3://").split("/", 1)
    filename = export_filename(group_name, version)

    if destination == "s3":
        if not bucket:
            raise ValueError("No S3 bucket chosen.")
        folder = (prefix or "").strip("/")
        key = f"{folder}/{filename}" if folder else filename
        t2.s3.copy_object(
            Bucket=bucket, Key=key,
            CopySource={"Bucket": source_bucket, "Key": source_key},
        )
        # Only now. Tidying the folder before the new file is in it leaves
        # a window where it holds nothing, and a consumer reading it then
        # gets an empty answer rather than an old one -- the worse
        # failure, because an old checkpoint is still a checkpoint.
        housekeeping = apply_retention(bucket, prefix, group_name,
                                       mode=retention, keep_key=key)
        return {"destination": "s3", "written_to": f"s3://{bucket}/{key}",
                "filename": filename, "url": "", "retention": housekeeping}

    return {
        "destination": "local",
        "written_to": "",
        "filename": filename,
        # Named for the group and version it is, rather than arriving as
        # yet another model.pt in a downloads folder.
        "url": t2.s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": source_bucket, "Key": source_key,
                    "ResponseContentDisposition": f'attachment; filename="{filename}"'},
            ExpiresIn=900,
        ),
        # Nothing to tidy. A download adds no file to anybody's folder, so
        # there is no earlier export of it to archive or delete -- and a
        # retention mode that silently reached into S3 on a download
        # would be acting somewhere the operator did not ask it to.
        "retention": {"mode": "keep_all", "archived": [], "deleted": []},
    }


def run_export(group_name: str, criterion: str, *, destination: str,
               bucket: str = "", prefix: str = "",
               retention: str = "keep_all") -> dict:
    """Pick, then export. The one path both the button and the watcher take.

    One implementation on purpose: a manual export and an automatic one
    that could choose differently would make "the exported model" mean two
    things, and the difference would only show up months later in whatever
    was consuming the folder.
    """
    rows = group_versions(group_name)
    best, why = pick_best(rows, criterion)
    if best is None:
        raise ValueError(why)
    placed = export_checkpoint(best, destination=destination, bucket=bucket,
                               prefix=prefix, retention=retention)
    return {"picked": describe_pick(best, criterion),
            "considered": len(rows), **placed}


# --- the watcher --------------------------------------------------------

def _auto_path() -> str:
    return os.path.join(_settings_dir(), AUTO_FILE)


#: The watcher's ONE shared record (2026-10-26), in the pull-control
#: table beside the configuration. It was a file in each container, so the
#: watcher ran only while that container was up, and dev and production
#: were armed separately. Now the app containers AND the always-on queue
#: Lambda read the same record, so an export still happens with the app
#: off, and `claim_once` in `check` makes exactly one of them act.
AUTO_KIND = "metrics_auto_export"


def _shared_get():
    from . import pull_control as pc
    got = pc.ddb.get_item(
        TableName=pc.TABLE_NAME,
        Key={"kind": {"S": AUTO_KIND}, "sort": {"S": "v1"}},
        ConsistentRead=True,
    ).get("Item")
    return json.loads(got["body"]["S"]) if got and "body" in got else None


def _shared_put(plan: dict) -> None:
    from . import pull_control as pc
    pc.ddb.put_item(
        TableName=pc.TABLE_NAME,
        Item={"kind": {"S": AUTO_KIND}, "sort": {"S": "v1"},
              "body": {"S": json.dumps(plan)}},
    )


def _file_settings():
    """The container's old per-container file, read once to carry an armed
    watcher over to the shared record. Never present in the Lambda --
    nor importable there, as it reaches the FastAPI router for its path."""
    try:
        with open(_auto_path(), encoding="utf-8") as handle:
            stored = json.load(handle)
    except Exception:                                              # noqa: BLE001
        return None
    return stored if isinstance(stored, dict) else None


def auto_settings() -> dict:
    """What the watcher is armed with, or the defaults when it is not."""
    try:
        stored = _shared_get()
    except Exception as exc:                                       # noqa: BLE001
        print(f"Could not read the auto-export record: {exc}")
        stored = None
    if stored is None:
        # Nothing shared yet: carry this container's old file over, once.
        stored = _file_settings()
        if stored:
            try:
                _shared_put(stored)
            except Exception as exc:                               # noqa: BLE001
                print(f"Could not carry the auto-export file over: {exc}")
    if not isinstance(stored, dict):
        return dict(AUTO_DEFAULTS)
    return {**AUTO_DEFAULTS, **stored}


def write_auto(values: dict) -> dict:
    """Merges into the shared record and writes it whole."""
    plan = {**auto_settings(), **values}
    _shared_put(plan)
    return plan


def arm(*, group_name: str, criterion: str, bucket: str, prefix: str,
        retention: str = "keep_all") -> dict:
    """Switch the watcher on, marking the group's current newest version
    as already seen.

    Without that mark, arming a watcher on a group holding fourteen
    versions would immediately export v14 -- a version registered weeks
    ago, which is not "a new version registered". The same reasoning as
    `queue_after_pull`, which marks the newest finished pull as handled
    when it is switched on.
    """
    if criterion not in CRITERIA:
        raise ValueError(f"{criterion!r} is not something a version can be ranked by.")
    if retention not in RETENTION_MODES:
        raise ValueError(f"{retention!r} is not a way of handling earlier exports.")
    if not group_name:
        raise ValueError("Pick a Class, model and size first -- there is no group to watch.")
    if not bucket:
        # Stated rather than silently allowed: there is no browser
        # attached to an automatic export, so "this computer" has nowhere
        # to put the file. The card greys the switch for the same reason.
        raise ValueError(
            "An automatic export has to go to an S3 bucket. Nothing is "
            "running in a browser when it fires, so there is nowhere on "
            "your computer for it to land.")
    return write_auto({
        "enabled": True,
        "group_name": group_name,
        "criterion": criterion,
        "bucket": bucket,
        "prefix": prefix,
        "retention": retention,
        "last_version": newest_version(group_name),
        # Cleared, not carried. Re-arming on a DIFFERENT group with a
        # stale "already exported v4" would skip the first export into a
        # bucket holding nothing at all.
        "last_exported_version": "",
        "armed_at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "last_result": {},
    })


def disarm() -> dict:
    return write_auto({"enabled": False})


def pending(plan: dict | None = None) -> dict:
    """Whether the watcher would act right now, and why it would not.

    Separated from `check` so the card can say what is about to happen
    without doing it, and so a test can assert the decision without
    writing to anybody's bucket.
    """
    plan = plan if plan is not None else auto_settings()
    if not plan.get("enabled"):
        return {"act": False, "reason": "Not switched on.", "version": 0}
    group_name = str(plan.get("group_name") or "")
    if not group_name:
        return {"act": False, "reason": "No group to watch.", "version": 0}
    newest = newest_version(group_name)
    if not newest:
        return {"act": False, "reason": "Nothing registered in that group yet.",
                "version": 0}
    seen = int(plan.get("last_version") or 0)
    if newest <= seen:
        return {"act": False,
                "reason": f"No new version since v{seen}.", "version": newest}
    return {"act": True, "reason": "", "version": newest}


def check() -> dict:
    """Called on the app's poll tick. Exports at most one checkpoint.

    Every path returns rather than raises: this shares a tick with the
    training queue and the audit reconciler, and an S3 permission problem
    here must not stop either of those.

    The marker moves EVEN WHEN THE EXPORT FAILED. A bucket the role cannot
    write to will not become writable 25 seconds later, and not recording
    the version turns one misconfiguration into a failed S3 call every 25
    seconds until somebody notices.
    """
    plan = auto_settings()
    decision = pending(plan)
    if not decision["act"]:
        return {"exported": False, "reason": decision["reason"]}

    newest = decision["version"]
    # One actor per version. Both app containers and the queue Lambda run
    # this on their own ticks against the one shared record; the claim is a
    # conditional put, so exactly one of them exports.
    from . import pull_control as pc
    if not pc.claim_once("metrics_auto_export", f"{plan.get('group_name')}:v{newest}"):
        return {"exported": False, "reason": "Another instance is handling this version."}
    exported = ""
    result = {
        "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "trigger_version": newest,
        "ok": False,
        "detail": "",
    }
    try:
        rows = group_versions(str(plan["group_name"]))
        best, why = pick_best(rows, str(plan["criterion"]))
        if best is None:
            result["detail"] = why
        elif str(best.get("version")) == str(plan.get("last_exported_version") or ""):
            # The new version did not win. Nothing to write -- the file
            # already in the bucket IS the best version, and a second
            # timestamped copy of it would only make the folder harder to
            # read. Recorded as a real outcome rather than a silence.
            result.update({
                "ok": True,
                "skipped": True,
                "version": str(best.get("version")),
                "detail": f"v{newest} registered but v{best.get('version')} is still "
                          f"the best by {plan['criterion']} -- nothing re-exported.",
            })
        else:
            placed = export_checkpoint(
                best, destination="s3", bucket=str(plan["bucket"]),
                prefix=str(plan.get("prefix") or ""),
                retention=str(plan.get("retention") or "keep_all"))
            picked = describe_pick(best, str(plan["criterion"]))
            result.update({
                "ok": True,
                "skipped": False,
                "written_to": placed["written_to"],
                "version": picked["version"],
                "status": picked["status"],
                "value": picked["value"],
                "validation_images": picked["validation_images"],
                "thin_validation": picked["thin_validation"],
                "retention": placed.get("retention") or {},
                "detail": f"v{picked['version']} -> {placed['written_to']}",
            })
            exported = picked["version"]
    except Exception as exc:                                       # noqa: BLE001
        result["detail"] = str(exc)[:400]

    marker = {"last_version": newest, "last_result": result}
    if exported:
        marker["last_exported_version"] = exported
    try:
        write_auto(marker)
    except OSError as exc:
        print(f"Could not record the auto-export result: {exc}")
    return {"exported": bool(exported), "result": result}
