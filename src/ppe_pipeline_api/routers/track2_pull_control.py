"""Pull control: drop folders, week folders, and the two triggers.

Thin over `pull_control`. The routing rule itself lives there because the
Lambda runs that same file -- an endpoint that reimplemented "which week
does this belong to" would be a second answer to the question.
"""

from __future__ import annotations

import threading

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import golden_set as gs
from .. import pull_control as pc
from ..auth import require_session

# The prefix is set where the router is mounted, as every other router
# here does it -- declaring it in both places doubles it.
router = APIRouter()


class Source(BaseModel):
    prefix: str = ""
    destination: str = ""
    # The bucket the source is READ from. Blank means this app's own.
    # The escalation datasets live in safety-model-dataset-dev, granted
    # read-only, so nothing here can write back to them.
    bucket: str = ""
    # Where the `val` side goes. Set it and an already-split dataset fills
    # both roots without either side being re-split.
    eval_destination: str = ""
    # copy | move. Copy by default -- see _action_of.
    action: str = "copy"
    # train | val, or blank to infer from the path. Set when a folder was
    # picked by hand: the list it was put in is the answer, and a folder
    # must not be re-routed because `val` appears in its path.
    split: str = ""


class UnitsRequest(BaseModel):
    bucket: str = ""
    root: str = ""


class FoldersRequest(BaseModel):
    bucket: str = ""
    prefix: str = ""


class SettingsRequest(BaseModel):
    name: str = "pull_control"
    apply: bool = True
    # Saves past the "another file already writes there" refusal. Off
    # unless the page sends it, so the check still has to be read and
    # overruled rather than clicked through by accident.
    allow_clash: bool = False


class WeekNameRequest(BaseModel):
    name: str


class PairingRequest(BaseModel):
    bucket: str = ""
    image_prefixes: list[str] = []
    label_prefixes: list[str] = []


class PickedLists(BaseModel):
    train_images: list[str] = []
    train_labels: list[str] = []
    val_images: list[str] = []
    val_labels: list[str] = []


class ConfigRequest(BaseModel):
    # The Auto Training card's own selection -- the root it is pointed at
    # and the weeks ticked out of it. Stored server-side so it rides along
    # in the settings file: a selection that only exists in a browser tab
    # is one nobody can restore.
    auto_training: dict | None = None
    # {"slug": ..., "architecture": ...} -- what everything trains as,
    # chosen once in "Register a model and Classes". The unattended
    # retrain always reads this; the training form pre-selects from it
    # but may be overridden for one run.
    # Step 2's own choices: which curated folder to split, where the
    # split goes, and how.
    #
    # This was MISSING, and the Golden Set card was doing everything
    # right: it debounces a `golden_set` patch on every edit, the route
    # answered 200, and Pydantic dropped the field on the way in because
    # the model did not declare it. The card kept showing the values --
    # they were still in its own state -- so the only visible symptom was
    # that a saved settings file came back with `golden_set: {}` and the
    # folders had to be found again.
    golden_set: dict | None = None
    # Which cards are open. Restored with everything else so a loaded
    # file puts the page back as it was left.
    card_state: dict | None = None
    # Whether each run freezes a snapshot first; see pull_control.
    freeze_on_run: bool | None = None
    # The Assemble card's folders, detector, overwrite tick and its
    # "this is the only source" switch. Declared HERE or the POST drops
    # it without a word: Pydantic discards what the model does not
    # declare, so a card can look wired, answer 200, and save nothing.
    dataset_assemble: dict | None = None
    # The Import Checkpoint card's standing choices. See pull_control.
    import_checkpoint: dict | None = None
    default_detector: dict | None = None
    # {"epochs": .., "img_size": .., "batch": .., "seed": .., "patience":
    # .., "notes": ..} -- the shape of every run, chosen in step 1.
    default_training: dict | None = None
    # What the last automatic run built and froze; see pull_control.
    default_dataset: dict | None = None
    # What the pull chain last froze. Declared so a settings file
    # round-trips it; written by `run_chain`, not by the card.
    last_pull_dataset: dict | None = None
    # "existing" | "scratch" | "import" -- see pull_control.
    default_source: str | None = None
    # The model package ARN a run continues from; see pull_control.
    default_version: str | None = None
    # "live" | "snapshot" -- which KIND of dataset a run uses; see
    # pull_control.
    data_source: str | None = None
    # Which settings file is in effect; see pull_control.
    settings_file: str | None = None
    # scan | file_time | capture_time. See file_moment.
    week_by: str | None = None
    # The four folder lists. `sources` is rebuilt from these, so they are
    # what a saved setting has to carry.
    picked: PickedLists | None = None
    source_bucket: str | None = None
    destination_bucket: str | None = None
    # When week 1 ENDS. Week 2 starts here; see week_at.
    week_cutoff: str | None = None
    # The one root holding every week; train and val sit inside each.
    root: str | None = None
    # Superseded by `root`, still accepted so an older client or a saved
    # settings file round-trips.
    training_root: str | None = None
    eval_root: str | None = None
    mode: str | None = None
    schedule_expression: str | None = None
    sources: list[Source] | None = None
    images_subpath: str | None = None
    labels_subpath: str | None = None
    week_prefix: str | None = None
    week_length_days: int | None = None
    auto_create_weeks: bool | None = None
    first_week_takes_all: bool | None = None
    overwrite: bool | None = None
    schedule_every_days: int | None = None
    schedule_time: str | None = None
    # "today" | "date", and the date it counts from; see pull_control.
    schedule_from: str | None = None
    schedule_start_date: str | None = None


class WeekRequest(BaseModel):
    name: str
    starts_at: str
    make_folders: bool = True


class ModeRequest(BaseModel):
    mode: str
    # Left blank the interval below builds it. Sent, it wins, so any
    # expression Scheduler accepts is still reachable.
    schedule_expression: str = ""
    # Days between pulls. The series is counted from the week cut-off,
    # so there is no start date to send -- see DEFAULTS.
    schedule_every_days: int | None = None
    # "HH:MM" Singapore time, or "" to land on the cut-off's own time.
    # Moves the clock only; the dates still come from the cut-off.
    schedule_time: str | None = None
    # "today": every N days counted from the next occurrence of the time.
    # "date": every N days counted from `schedule_start_date` (YYYY-MM-DD).
    schedule_from: str | None = None
    schedule_start_date: str | None = None


class RunRequest(BaseModel):
    # A sweep that would run for fifteen minutes is a sweep nobody is
    # still watching. Capped so the card gets an answer.
    #
    # REQUIRED, with no default, although the page always sends 2000.
    # This is the one route that starts real work -- it MOVES files out
    # of the drop folders, and there is no second copy -- and it was the
    # only destructive write that would act on a completely empty body.
    # Every other one either requires a confirmation (`purge-all` wants
    # the exact object count), refuses an empty target (`export/weights`
    # answers 400), or falls back to something harmless. Requiring the
    # field costs the caller nothing and makes `POST {}` a 422 instead of
    # a sweep.
    limit: int


@router.get("")
def get_status(session=Depends(require_session)) -> dict:
    """Settings, weeks, trigger state and the last few sweeps."""
    try:
        return pc.status()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read pull control: {exc}")


@router.post("/config")
def save_config(payload: ConfigRequest, session=Depends(require_session)) -> dict:
    """Saves the settings. Anything not sent keeps its current value."""
    patch = payload.model_dump(exclude_none=True)
    if "sources" in patch:
        cleaned = []
        for source in patch["sources"]:
            prefix = str(source.get("prefix") or "").strip().strip("/")
            destination = str(source.get("destination") or "").strip().strip("/")
            evaluation = str(source.get("eval_destination") or "").strip().strip("/")
            action = "move" if str(source.get("action") or "").lower() == "move" else "copy"
            bucket = str(source.get("bucket") or "").strip()
            if bucket and action == "move":
                raise HTTPException(
                    status_code=400,
                    detail=f"{prefix} is in another bucket ({bucket}), which is granted "
                           f"read-only. Use copy.",
                )
            if not prefix or not destination:
                continue
            # A drop folder inside its own destination would re-file what
            # it just filed on every sweep, forever.
            for root in (destination, evaluation):
                if root and (prefix == root or prefix.startswith(root + "/")):
                    raise HTTPException(
                        status_code=400,
                        detail=f"The drop folder {prefix} is inside its own destination "
                               f"{root} -- routing it would never finish.",
                    )
            # A copy that wrote back under its own source would re-copy its
            # own output every sweep, growing without end.
            for root in (destination, evaluation):
                if root and root.startswith(prefix + "/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"The destination {root} is inside the folder being "
                               f"watched ({prefix}) -- each sweep would re-file its own output.",
                    )
            split = str(source.get("split") or "").strip().lower()
            if split not in ("", "train", "val"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Split must be train, val, or blank -- not {split!r}.",
                )
            cleaned.append({
                "prefix": prefix, "destination": destination,
                "eval_destination": evaluation, "action": action,
                "bucket": bucket, "split": split,
            })
        patch["sources"] = cleaned
    # A schedule is counted FROM the cut-off, and `set_mode` refuses
    # schedule mode without one -- so clearing it while a schedule is
    # armed leaves a rule enabled at AWS with nothing to count from, and
    # the next run quietly never happens. Refused rather than accepted,
    # because every report of "my settings cleared themselves" on this
    # page has been a card writing a blank it had not read yet.
    if "week_cutoff" in patch and not str(patch["week_cutoff"] or "").strip():
        live = pc.read_config()
        if str(live.get("mode") or "") == "schedule" and str(live.get("week_cutoff") or "").strip():
            raise HTTPException(
                status_code=409,
                detail="The cut-off date cannot be cleared while the pull is on a "
                       "schedule -- the schedule is counted from it, and without "
                       "one the next run never happens. Switch the trigger off "
                       "first if you mean to clear it.",
            )

    try:
        saved = pc.write_config(patch)
        # Rebuilt from the picks rather than sent by the card, so there is
        # one rule for turning folders into sources and the card cannot
        # disagree with the sweep about what was configured.
        if saved.get("picked"):
            saved = pc.write_config({"sources": pc.sources_from_picked(saved)})
        return {"config": saved}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not save: {exc}")


@router.post("/weeks")
def add_week(payload: WeekRequest, session=Depends(require_session)) -> dict:
    """Defines a week and lays out its folders under every destination."""
    try:
        week = pc.create_week(
            name=payload.name,
            starts_at=payload.starts_at,
            make_folders=payload.make_folders,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not create the week: {exc}")
    return {"week": week, "weeks": pc.list_weeks()}


@router.post("/weeks/delete")
def remove_week(payload: WeekRequest, session=Depends(require_session)) -> dict:
    """Forgets a week. Its folders and their contents are left in place."""
    try:
        pc.delete_week(name=payload.name, starts_at=payload.starts_at)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not delete the week: {exc}")
    return {"weeks": pc.list_weeks()}


class PurgeWeekRequest(BaseModel):
    """One week, plus proof the count was seen.

    `purge-all` has always required the exact object count and re-counted
    it server-side. This route -- which is the one actually reachable from
    a row in the table, so the one more likely to be pressed -- took a
    name and nothing else. The page asked for a confirmation; the API did
    not, so anything calling it directly deleted a week's data on one
    request. Now both purges answer to the same rule.
    """
    name: str
    confirm_objects: int


class PurgeWeeksRequest(BaseModel):
    #: Must be the exact object count the preview reported. Not a
    #: checkbox and not the word "yes": the number is the only thing
    #: that proves the person saw what they were about to destroy, and
    #: it changes as the corpus does, so it cannot be muscle-memory.
    confirm_objects: int
    #: Specific weeks, or all defined ones when empty.
    names: list[str] = []


@router.get("/weeks/purge-preview")
def purge_preview(name: str = "", session=Depends(require_session)) -> dict:
    """What deleting the week folders would destroy. Counted first and
    shown, because "delete the week folders" and "delete 1,496 files"
    are not the same sentence.

    `name` narrows it to ONE week, which is what the per-row purge needs:
    that button now has to send the exact count, and it cannot send a
    number nobody ever counted.
    """
    try:
        return pc.count_week_objects(names=[name] if name else None)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not count: {exc}")


@router.post("/weeks/purge-all")
def purge_weeks(payload: PurgeWeeksRequest, session=Depends(require_session)) -> dict:
    """PERMANENTLY delete every object under the week folders.

    There is no second copy: the pull MOVES files out of the drop
    folders rather than copying them. Nothing here is recoverable.

    Re-counted inside the request rather than trusting the number the
    client sends. Between the preview and the press a sweep may have
    landed files, and deleting more than was shown is exactly the
    surprise the confirmation exists to prevent.
    """
    try:
        now = pc.count_week_objects(names=payload.names or None)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not count: {exc}")

    if payload.confirm_objects != now["objects"]:
        raise HTTPException(
            status_code=409,
            detail=(f"The week folders now hold {now['objects']} objects, not "
                    f"{payload.confirm_objects}. Something changed since you "
                    "looked -- check the count and confirm again."),
        )
    if not now["objects"]:
        return {"deleted": 0, "errors": [], "message": "Nothing to delete."}

    wanted = list(payload.names or [])
    try:
        out = pc.purge_week_folders(names=wanted or None)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Delete failed: {exc}")

    # ...and forget the weeks, so the card actually empties. Deleting the
    # files alone left every row on screen reading "0 img -- empty",
    # which looks like the delete did not work. The single-week button
    # has always done both (`pc.purge_week`); this is the bulk of it.
    forgotten = 0
    for week in pc.list_weeks():
        if wanted and week["name"] not in wanted:
            continue
        try:
            pc.delete_week(name=week["name"], starts_at=week["starts_at"])
            forgotten += 1
        except Exception as exc:                               # noqa: BLE001
            out.setdefault("errors", []).append(f"{week['name']}: {exc}")
    out["weeks_forgotten"] = forgotten
    out["message"] = (f"Deleted {out['deleted']} object(s) and forgot "
                      f"{forgotten} week(s).")
    out["weeks"] = pc.list_weeks()
    return out


@router.post("/preview")
def preview(session=Depends(require_session)) -> dict:
    """What a sweep would move, without moving it."""
    try:
        return pc.plan_routing()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the drop folders: {exc}")


@router.post("/run")
def run_now(payload: RunRequest, session=Depends(require_session)) -> dict:
    """Starts a sweep and returns its id at once.

    Not synchronous any more. A sweep over a few thousand files takes
    minutes, and a request that blocks for minutes can report neither
    progress nor be interrupted -- both of which are the point. The
    caller polls `/run/{run_id}` and can `/run/{run_id}/stop`.
    """
    running = pc.active_run()
    if running and running.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail=f"A sweep is already running ({running['run_id']}). Stop it first, "
                   f"or wait -- two sweeps over the same folders would race each other "
                   f"into the same week.",
        )

    run_id = pc.new_run_id()
    limit = max(1, min(payload.limit, 20000))
    pc.write_run_state(run_id, status="running", trigger="manual", total=0, done=0,
                       moved=0, skipped=0, errors=0, stop_requested=False,
                       started_at=pc._now().isoformat())

    def work():
        try:
            pc.route_pending(limit=limit, trigger="manual", run_id=run_id)
        except Exception as exc:                       # noqa: BLE001
            # The thread has nowhere to raise to, so the failure goes
            # where the card is already looking.
            pc.write_run_state(run_id, status="failed", aborted=str(exc),
                               finished_at=pc._now().isoformat())

    # Daemon: a container being replaced should not be held open by a
    # sweep. The run then goes quiet and `active_run` reports it lost
    # rather than leaving a Stop button for something already gone.
    threading.Thread(target=work, name=f"pull-{run_id}", daemon=True).start()
    return {"run_id": run_id, "status": "running"}


@router.get("/run/{run_id}")
def run_progress(run_id: str, session=Depends(require_session)) -> dict:
    """How far a sweep has got."""
    state = pc.read_run_state(run_id)
    if not state:
        raise HTTPException(status_code=404, detail=f"No run called {run_id}.")
    return state


@router.get("/run")
def current_run(session=Depends(require_session)) -> dict:
    """The sweep running now, if any. Used to re-attach after a reload."""
    return {"run": pc.active_run()}


@router.get("/last-pull")
def last_pull(session=Depends(require_session)) -> dict:
    """When a sweep last FINISHED, and what it moved.

    Its own tiny endpoint because Auto Training polls it to know when to
    re-discover its week folders, and the full status response lists S3
    to build `weeks` -- far too heavy to ask for every twenty seconds.

    NOT under /run/: `GET /run/{run_id}` is registered above this, and
    the first matching route wins, so `/run/last-pull` would arrive as a
    run id. That exact collision already cost a live afternoon once.

    Reads `run_state`, not the event rows. `_record_run` skips a
    scheduled sweep that moved nothing, so a quiet or failed run would
    look like no run at all -- and "the schedule fired and found
    nothing" is precisely what somebody watching this wants to know.
    """
    for run in pc.recent_run_states(10):
        if run.get("finished_at"):
            return {
                "finished_at": str(run.get("finished_at") or ""),
                "run_id": str(run.get("run_id") or ""),
                "moved": int(run.get("moved") or 0),
                "status": str(run.get("status") or ""),
            }
    return {"finished_at": "", "run_id": "", "moved": 0, "status": ""}


@router.post("/run/{run_id}/stop")
def stop_run(run_id: str, session=Depends(require_session)) -> dict:
    """Asks a sweep to stop, and undo what it created.

    A flag, not a kill: the sweep is mid-copy somewhere and the only safe
    moment to stop is between files. It takes effect within
    `PROGRESS_EVERY` files, and works for the Lambda too, which nothing
    can kill from outside.
    """
    try:
        return pc.request_stop(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/pending")
def pending(session=Depends(require_session)) -> dict:
    """What is waiting to be pulled, and which week each file would reach.

    Read-only, and the same placement decisions a sweep makes -- so the
    number here is what the next pull will actually do, not an estimate.
    """
    try:
        return pc.pending_summary()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the sources: {exc}")


@router.post("/mode")
def set_mode(payload: ModeRequest, session=Depends(require_session)) -> dict:
    """Off, on arrival, or on a schedule."""
    try:
        return pc.set_mode(
            mode=payload.mode,
            schedule_expression=payload.schedule_expression,
            schedule_every_days=payload.schedule_every_days,
            schedule_time=payload.schedule_time,
            schedule_from=payload.schedule_from,
            schedule_start_date=payload.schedule_start_date,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not change the trigger: {exc}")


@router.post("/units")
def units(payload: UnitsRequest, session=Depends(require_session)) -> dict:
    """The datasets under a root, with their class lists and sizes.

    Read-only, and the same call answers which folders are datasets and
    whether their class lists agree closely enough to be combined.
    """
    try:
        return pc.discover_units(bucket=payload.bucket, root=payload.root)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {payload.root}: {exc}")


@router.get("/buckets")
def buckets(session=Depends(require_session)) -> dict:
    """Every bucket in the account, flagged readable or not."""
    try:
        return pc.list_buckets()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/folders")
def folders(payload: FoldersRequest, session=Depends(require_session)) -> dict:
    """One level of folders under a prefix, in any readable bucket."""
    try:
        return pc.list_folders(bucket=payload.bucket, prefix=payload.prefix)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class FolderDeleteRequest(BaseModel):
    bucket: str = ""
    prefix: str = ""
    # False: count and report only. True: delete.
    confirm: bool = False
    # Delete even though a setting still uses the folder.
    force: bool = False


@router.post("/folders/delete")
def delete_folder(payload: FolderDeleteRequest, session=Depends(require_session)) -> dict:
    """Delete a folder from the folder browser -- asked twice.

    Without `confirm` this only counts: what would go, and which settings
    still point at it. The dialog shows that, then sends `confirm`; a
    folder in use is refused (409) until it also sends `force`, and the
    refusal text names what uses it.
    """
    try:
        if not payload.confirm:
            return pc.folder_delete_check(bucket=payload.bucket, prefix=payload.prefix)
        return pc.delete_folder(bucket=payload.bucket, prefix=payload.prefix,
                                force=payload.force)
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class YamlBrowseRequest(BaseModel):
    bucket: str = ""
    prefix: str = ""


@router.post("/yaml-files")
def yaml_files(payload: YamlBrowseRequest, session=Depends(require_session)) -> dict:
    """Folders AND the .yaml files beside them, in any readable bucket.

    `/folders` returns folders only, which is right for picking a dataset
    root. Reading class names needs the file itself.
    """
    try:
        return pc.list_folders_and_yaml_files(bucket=payload.bucket, prefix=payload.prefix)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class YamlClassesRequest(BaseModel):
    bucket: str = ""
    key: str = ""


@router.post("/yaml-classes")
def yaml_classes(payload: YamlClassesRequest, session=Depends(require_session)) -> dict:
    """The class names one yaml declares, from any readable bucket.

    Parsed by the same `classes_from_yaml_text` the local-file route
    uses, so a yaml read from a laptop, from this app's bucket and from
    somebody else's bucket all mean the same thing.
    """
    if not payload.key:
        raise HTTPException(status_code=422, detail="Pick a .yaml file first.")
    try:
        body = pc.read_yaml_object(bucket=payload.bucket, key=payload.key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    try:
        from .. import track2_shared as t2

        classes = t2.classes_from_yaml_text(body)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not read {payload.key} as YAML: {exc}")
    return {
        "bucket": payload.bucket,
        "key": payload.key,
        "classes": {str(k): v for k, v in classes.items()},
    }


@router.get("/settings")
def list_settings(session=Depends(require_session)) -> dict:
    """Saved settings files on the host, newest first."""
    return pc.list_settings_files()


@router.post("/settings/new")
def new_settings(session=Depends(require_session)) -> dict:
    """Blank template: the live settings go back to the shipped defaults.

    Changes what the system is actually set to, not just what the page is
    showing -- every card reads the live config, so a "new" that only
    cleared the form would leave the old settings running underneath it.
    Saved files are untouched; this is where a new one is started from.
    """
    try:
        return pc.reset_to_defaults()
    except Exception as exc:                                # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not reset: {exc}")


# --- the class and model registry, carried in the settings file ----------------
#
# The detectors are the classes everything trains as. They lived only in
# this account's DynamoDB, so a settings file described a run it could not
# actually reproduce: it named a detector by slug and carried nothing that
# would let a machine without that detector build it.
#
# They ride in a section BESIDE `config`, not inside it -- see
# `save_settings_file`. The live registry stays the source of truth; the
# file carries a copy.

#: Imported inside the functions. `track2_shared` reaches for torch and
#: ultralytics on some paths, and this module is also read by tooling that
#: has no reason to pay for that.
def _registry_snapshot() -> dict:
    """Every detector this machine defines, as a file records them.

    The built-in is skipped: it ships inside the image, every machine
    already has it, and "restoring" it would mean writing a stored copy
    of something the code defines.
    """
    from .. import track2_shared as t2

    detectors = []
    for item in t2.list_detectors(fresh=True):
        if item.get("builtin"):
            continue
        detectors.append({
            "slug": str(item.get("slug") or ""),
            "label": str(item.get("label") or ""),
            # JSON has no integer keys; `_restore_registry` converts back.
            "classes": {str(k): str(v) for k, v
                        in (item.get("classes") or {}).items()},
        })
    return {"detectors": detectors}


def _restore_registry(section: dict) -> dict:
    """Create the detectors the file has that this machine does not.

    CREATE MISSING ONLY, and never update or delete. A load must not be
    able to change the classes of a detector a run is training against
    right now, nor remove one a registered model package is filed under
    -- the registry group name is derived from the detector, and the
    approval rule matches those names exactly.

    So the rule is: this machine's detectors win, every time. The file
    can add, and that is all.
    """
    from .. import track2_shared as t2

    try:
        have = {str(d.get("slug") or "") for d in t2.list_detectors(fresh=True)}
    except Exception as exc:                                   # noqa: BLE001
        return {"created": [], "kept": [], "problems": [f"could not read the registry: {exc}"]}

    created, kept, problems = [], [], []
    for item in section.get("detectors") or []:
        slug = str(item.get("slug") or "").strip()
        if not slug:
            continue
        if slug in have:
            kept.append(slug)
            continue
        try:
            classes = {int(k): str(v) for k, v in (item.get("classes") or {}).items()}
        except (TypeError, ValueError):
            problems.append(f"{slug}: its class list is not index -> name")
            continue
        if not classes:
            problems.append(f"{slug}: no classes, so it would train nothing")
            continue
        try:
            t2.upsert_detector(slug, str(item.get("label") or slug), classes)
            created.append(slug)
        except Exception as exc:                               # noqa: BLE001
            problems.append(f"{slug}: {exc}")
    return {"created": created, "kept": kept, "problems": problems}


@router.post("/settings/save")
def save_settings(payload: SettingsRequest, session=Depends(require_session)) -> dict:
    """Writes the live settings to a file on the EC2 host.

    The host, not the container: a file inside the container is destroyed
    by the next deploy, which is the one thing these are meant to survive.
    """
    try:
        # Best effort: a registry that cannot be read must not stop the
        # settings being saved. The file is still a complete record of
        # every other card.
        try:
            sections = {"registry": _registry_snapshot()}
        except Exception:                                      # noqa: BLE001
            sections = {}
        return pc.save_settings_file(name=payload.name,
                                     allow_clash=payload.allow_clash,
                                     sections=sections)
    except ValueError as exc:
        # 409 for the folder collision specifically, so the page can offer
        # the override without having to recognise the prose. Everything
        # else a save can refuse for stays a 400.
        raise HTTPException(
            status_code=409 if "already writes" in str(exc) else 400,
            detail=str(exc),
        )
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not write to {pc.SETTINGS_DIR}: {exc}. The host directory "
                   f"is mounted by deploy.sh; an older container will not have it.",
        )


@router.post("/settings/load")
def load_settings(payload: SettingsRequest, session=Depends(require_session)) -> dict:
    """Reads a settings file back and makes it live."""
    try:
        out = pc.load_settings_file(name=payload.name, apply=payload.apply)
        if payload.apply:
            # After the settings, not before: a detector is only useful
            # once the configuration that names it is live, and a failure
            # here must not leave the settings half-applied.
            out["registry"] = _restore_registry(
                pc.read_settings_section(payload.name, "registry"))
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not read it: {exc}")


@router.get("/weeks/contents")
def weeks_contents(session=Depends(require_session)) -> dict:
    """Every week folder that exists, and what is in it."""
    try:
        return pc.week_contents()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the weeks: {exc}")


@router.post("/weeks/purge")
def purge_week(payload: PurgeWeekRequest, session=Depends(require_session)) -> dict:
    """Deletes a week's folders AND forgets the week.

    This removes data. It is only ever reached by someone pressing the
    button for it; no sweep and no trigger calls it.

    Re-counted here rather than trusting the number the client sends --
    the same rule `purge-all` follows, and for the same reason: between
    looking and pressing, a sweep may have landed files, and deleting
    more than was shown is exactly the surprise the confirmation exists
    to prevent.
    """
    try:
        now = pc.count_week_objects(names=[payload.name])
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not count: {exc}")
    if payload.confirm_objects != now["objects"]:
        raise HTTPException(
            status_code=409,
            detail=(f"{payload.name} now holds {now['objects']} objects, not "
                    f"{payload.confirm_objects}. Something changed since you "
                    "looked -- check the count and confirm again."),
        )
    try:
        return pc.purge_week(name=payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not delete it: {exc}")


class QueuePullRequest(BaseModel):
    """Queue a sweep. Empty means "the settings as they stand now"."""
    #: A snapshot to run under. Omitted, the live config is snapshotted at
    #: queue time -- which is the point: what was on screen when you
    #: queued it is what runs, not whatever the card holds later.
    config: dict | None = None
    note: str = ""


@router.get("/queue")
def get_pull_queue(session=Depends(require_session)) -> dict:
    """Everything queued, oldest first, plus whatever is sweeping now."""
    running = pc.active_run()
    return {
        "entries": pc.list_pull_queue(),
        "running": running if running and running.get("status") == "running" else None,
    }


@router.post("/queue/add")
def add_to_pull_queue(payload: QueuePullRequest,
                      session=Depends(require_session)) -> dict:
    try:
        entry = pc.enqueue_pull(config=payload.config, note=payload.note,
                                added_by="card")
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not queue it: {exc}")
    return {"entry": entry, "queued": len(pc.list_pull_queue())}


@router.post("/queue/remove/{entry_id}")
def remove_from_pull_queue(entry_id: str, session=Depends(require_session)) -> dict:
    if not pc.remove_pull_queue_entry(entry_id):
        raise HTTPException(status_code=404,
                            detail="Not in the queue (already started or removed).")
    return {"removed": entry_id, "queued": len(pc.list_pull_queue())}


@router.post("/queue/clear")
def clear_the_pull_queue(session=Depends(require_session)) -> dict:
    return {"removed": pc.clear_pull_queue()}


@router.post("/queue/advance")
def advance_the_pull_queue(session=Depends(require_session)) -> dict:
    """Start the next one now, rather than waiting for a tick.

    Refuses while a sweep is running, same as every other caller -- this
    skips the wait, not the guard.
    """
    return pc.advance_pull_queue(trigger="queue")


@router.post("/pairing")
def pairing(payload: PairingRequest, session=Depends(require_session)) -> dict:
    """Whether the chosen images and labels actually describe each other.

    Read-only. Called as folders are picked, so the mismatch is visible
    before a sweep files thousands of files that do not pair.
    """
    try:
        return pc.check_pairing(
            bucket=payload.bucket,
            image_prefixes=payload.image_prefixes,
            label_prefixes=payload.label_prefixes,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not compare them: {exc}")


@router.post("/settings/delete")
def delete_settings(payload: SettingsRequest, session=Depends(require_session)) -> dict:
    """Removes a saved settings file from the host."""
    try:
        return pc.delete_settings_file(name=payload.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not remove it: {exc}")


@router.get("/forecast")
def forecast(
    week_length_days: int | None = None,
    week_cutoff: str | None = None,
    week_prefix: str | None = None,
    training_root: str | None = None,
    eval_root: str | None = None,
    settings: str | None = None,
    ahead: int = 4,
    session=Depends(require_session),
) -> dict:
    """Every week -- filled, being filled, and the next four to come.

    Read-only. The future weeks do not exist yet and are not created by
    asking about them, and neither override writes anything.

    The overrides re-date and re-name the table under settings chosen in
    the card but not yet applied, so changing "new week every N days" or
    the week prefix shows what it would mean before it is saved. They are
    passed straight to `week_forecast`, which is the one implementation
    of the arithmetic -- computing the preview in the browser instead
    would be a second answer to "when does week 5 close".

    A blank `week_prefix` falls back to the default rather than producing
    folders called `1/` and `2/`, which is what `week_forecast` does with
    a blank saved one.
    """
    overrides: dict = {}
    # `settings` names a saved file to VIEW. Everything in it is used as
    # the override, and nothing is applied -- looking at another dataset
    # must not re-point the next pull, least of all a scheduled one.
    if settings:
        try:
            saved = pc.read_settings_file(settings)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        for field in ("week_length_days", "week_cutoff", "week_prefix",
                      "training_root", "eval_root", "sources",
                      "images_subpath", "labels_subpath", "first_week_takes_all"):
            if saved.get(field) is not None:
                overrides[field] = saved[field]
    if week_length_days is not None:
        overrides["week_length_days"] = max(int(week_length_days), 1)
    if week_cutoff is not None:
        overrides["week_cutoff"] = week_cutoff
    if week_prefix:
        overrides["week_prefix"] = week_prefix
    if training_root:
        overrides["training_root"] = training_root
    if eval_root:
        overrides["eval_root"] = eval_root
    # week_contents reads the roots off `sources`, not off training_root,
    # so overriding the roots alone would count the OLD folders under the
    # new names. Rebuilt here so the two agree.
    if overrides.get("training_root") or overrides.get("eval_root"):
        merged = {**pc.read_config(), **overrides}
        overrides["sources"] = pc.sources_from_picked(merged)
    try:
        return pc.week_forecast(ahead=max(int(ahead), 0), overrides=overrides or None)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the weeks: {exc}")


@router.get("/schedule")
def schedule(session=Depends(require_session)) -> dict:
    """When the next pulls run, and which weeks each one will fill.

    Read-only, and it answers for the SAVED settings -- so it shows what
    is armed, not what is typed into the card but not yet applied.
    """
    try:
        return {"schedule": pc.schedule_status(), "plan": pc.schedule_plan()}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the schedule: {exc}")


# ---------------------------------------------------------------------------
# Splitting a curated folder into train and val
#
# The folder is one someone chose by hand -- a golden set, a pile of edge
# cases. Nothing here decides what belongs in it; it only divides it, and
# keeps every image with its own label while doing so.
# ---------------------------------------------------------------------------

class SplitRequest(BaseModel):
    source: str = ""
    #: Where the LABELS are, when they do not live under `source`.
    #:
    #: Empty keeps the two shapes this always accepted -- `<source>/
    #: images/ + labels/`, and a flat folder -- so an older caller and a
    #: saved settings file both behave exactly as before.
    labels_source: str = ""
    bucket: str = ""
    target_folder: str = ""
    target_bucket: str = ""
    # Off copies the folder whole, with no train/val level.
    split: bool = True
    train_percent: int = 80
    images_subpath: str = "images"
    labels_subpath: str = "labels"


@router.post("/split/preview")
def split_preview(payload: SplitRequest, session=Depends(require_session)) -> dict:
    """What the split would produce. Copies nothing."""
    try:
        return gs.plan(
            source=payload.source, labels_source=payload.labels_source,
            bucket=payload.bucket,
            target_folder=payload.target_folder, target_bucket=payload.target_bucket,
            split=payload.split, train_percent=payload.train_percent,
            images_subpath=payload.images_subpath, labels_subpath=payload.labels_subpath,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {payload.source}: {exc}")


@router.post("/split/build")
def split_build(payload: SplitRequest, session=Depends(require_session)) -> dict:
    """Copies the split into the target folder."""
    try:
        return gs.build(
            source=payload.source, labels_source=payload.labels_source,
            bucket=payload.bucket,
            target_folder=payload.target_folder, target_bucket=payload.target_bucket,
            split=payload.split, train_percent=payload.train_percent,
            images_subpath=payload.images_subpath, labels_subpath=payload.labels_subpath,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not build the split: {exc}")


@router.post("/weeks/fill-gaps")
def fill_gaps(session=Depends(require_session)) -> dict:
    """Creates a folder for every empty week BETWEEN weeks that hold files.

    Writes zero-byte markers, which is all an S3 "folder" is. Only the
    holes inside the collected range: the stretch after the last
    collection is the present, not a gap, and giving each passing week a
    folder would fill the bucket during a quiet period.

    Also happens at the end of a pull that moved something. This is the
    same thing on demand, for when you want the tree tidy now rather
    than after the next sweep.
    """
    try:
        return pc.create_gap_folders()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not create them: {exc}")
