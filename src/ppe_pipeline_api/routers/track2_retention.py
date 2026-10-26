"""Recency weighting: older weeks contribute fewer images.

Settings live beside pull control's, in the same table -- the week folders
this reads are the ones pull control writes, and two places to configure
one pipeline is one too many.
"""

from __future__ import annotations

import datetime as _dt

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import pull_control as pc
from .. import dataset_snapshots as ds
from .. import retained_lists as rl
from .. import retrain_schedule as rs
from .. import track2_shared as t2
from .. import retention as rt
from ..auth import require_session

from .. import dataset_yaml as dy

router = APIRouter()

CONFIG_KEY = "retention"


class RetentionConfig(BaseModel):
    enabled: bool | None = None
    weekly_rate: float | None = None
    curve: str | None = None
    #: How many of the newest weeks count in full before decay starts.
    full_weeks: int | None = None
    #: A curated folder that never decays and is always trained on.
    golden_root: str | None = None
    #: Which bucket that folder is in. Blank is this app's own.
    golden_bucket: str | None = None
    #: Whether that folder is used. Separate from the path, so the path
    #: survives being switched off.
    use_golden: bool | None = None
    #: Whether `floor` and `min_images` apply at all.
    use_floor: bool | None = None
    floor: float | None = None
    min_images: int | None = None
    root: str | None = None
    eval_root: str | None = None


class PreviewRequest(BaseModel):
    root: str = ""
    #: The settings as the card currently reads, not as last saved.
    #: Without this the bars answered with the SAVED policy while the
    #: controls above them said something else -- change the curve from
    #: compounding to linear and the bars did not move, which reads as
    #: the curve making no difference.
    config: RetentionConfig | None = None


class BuildRequest(BaseModel):
    root: str = ""
    eval_root: str = ""
    classes: dict = {}
    destination: str = ""


def _settings() -> dict:
    """The saved retention settings, defaults filled in."""
    saved = pc.read_config().get(CONFIG_KEY) or {}
    return {**rt.DEFAULTS, **saved}


def _with_prefix(settings: dict) -> dict:
    """`settings` plus Pull Control's week prefix, for READING weeks.

    Never saved: `save_settings` writes `_settings()` back, and the prefix
    belongs to Pull Control, not to retention.
    """
    return {**settings,
            "week_prefix": str(pc.read_config().get("week_prefix") or "week_")}


@router.get("")
def get_settings(session=Depends(require_session)) -> dict:
    """Settings, plus where pull control is filing weeks."""
    config = pc.read_config()
    saved = {**rt.DEFAULTS, **(config.get(CONFIG_KEY) or {})}
    # Suggested from pull control's own destinations, so the two cards do
    # not have to be kept in step by hand.
    sources = config.get("sources") or []
    return {
        "config": saved,
        # Pull Control's root FIRST: it is the one the chain trains from
        # (`retrain_schedule` reads `config["root"]`, never this card's
        # own). A stale saved root here showed a different folder's weeks
        # than the run would use.
        "suggested_root": str(config.get("root") or "").strip("/")
            or saved.get("root")
            or (str(sources[0].get("destination")) if sources else ""),
        "suggested_eval_root": saved.get("eval_root")
            or (str(sources[0].get("eval_destination") or "") if sources else ""),
        "curve": _curve(saved),
    }


#: Two years, weekly. Long enough that a slow decay is still visibly
#: going somewhere, which a one-year window at 1% is not.
CURVE_WEEKS = 104


def _curve(config: dict) -> list:
    """The decay as points, computed once here rather than twice.

    `fraction` is the curve; `effective` is what actually happens, which
    is the curve held up by the floor. Drawing only the curve would show a
    week decaying to nothing while the floor quietly keeps 10% of it --
    a chart that disagrees with the thing it claims to describe.
    """
    floor = max(0.0, min(float(config.get("floor") or 0.0), 1.0))
    points = []
    for age in range(CURVE_WEEKS + 1):
        fraction = rt.retention_fraction(age, config)
        points.append({
            "age_weeks": age,
            "fraction": round(fraction, 4),
            "effective": round(max(fraction, floor), 4),
        })
    return points


class CurveRequest(BaseModel):
    """The settings as they are on screen, saved or not."""
    weekly_rate: float = 0.01
    curve: str = "exponential"
    floor: float = 0.10
    #: Carried so the chart draws the same shape the grid does. Without
    #: it the chart would decay from week 1 while the grid held the
    #: newest weeks flat, and one of them would be lying.
    full_weeks: int = 1


@router.post("/curve")
def curve(payload: CurveRequest, session=Depends(require_session)) -> dict:
    """The decay for settings that have not been saved yet.

    The card redraws from this as the controls move. Recomputing it in
    TypeScript would be a second implementation of the formula, and the
    two would disagree the first time either changed.
    """
    return {
        "curve": _curve(payload.model_dump()),
        "weeks": CURVE_WEEKS,
    }


@router.post("/config")
def save_settings(payload: RetentionConfig, session=Depends(require_session)) -> dict:
    patch = payload.model_dump(exclude_none=True)
    if "weekly_rate" in patch and not 0.0 <= patch["weekly_rate"] <= 1.0:
        raise HTTPException(status_code=400, detail="The weekly rate is a fraction, 0 to 1.")
    if "curve" in patch and patch["curve"] not in ("exponential", "linear"):
        raise HTTPException(status_code=400, detail="Curve is exponential or linear.")
    if "full_weeks" in patch and int(patch["full_weeks"]) < 1:
        raise HTTPException(
            status_code=400,
            detail="At least one week counts in full -- that is the week just collected.",
        )
    merged = {**_settings(), **patch}
    try:
        pc.write_config({CONFIG_KEY: merged})
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not save: {exc}")
    return {"config": merged}


class RetrainScheduleRequest(BaseModel):
    """The unattended retrain's settings. Anything omitted keeps its
    current value, like every other patch on this app."""
    enabled: bool | None = None
    follow_pull: bool | None = None
    every_days: int | None = None
    at: str | None = None
    detector_slug: str | None = None
    architecture: str | None = None
    epochs: int | None = None
    img_size: int | None = None
    batch: int | None = None
    seed: int | None = None
    #: `base_version` is NOT settable here. It comes from step 1's own
    #: choice, read in `rs.settings()` -- a second control for it in the
    #: retention card was two places to say one thing, and they
    #: disagreed in production.


# `GET /base-versions` lived here, listing a Class's registered
# versions for the retention card's own "Continue from" dropdown. Both
# are gone: step 1 asks which version a run starts from, and asking it
# twice produced two answers -- step 1 said version 2 while the chain
# took the newest Approved one. The Models page's own version dropdown
# reads the registry it already lists.


@router.get("/retrain")
def get_retrain(session=Depends(require_session)) -> dict:
    """The schedule, and everything that would stop it firing.

    The preflight is returned whether or not the schedule is on, so
    "will this work on Sunday" is answerable on Saturday. A schedule
    whose failures are only visible afterwards reads as "the model is
    being retrained" right up until somebody checks.
    """
    config = pc.read_config()
    out = rs.preflight(config)
    out["next_runs"] = [m.isoformat() for m in rs.next_runs(out["settings"])]
    out["expression"] = rs.schedule_expression(out["settings"])
    # The EFFECTIVE answer, not the raw setting: "Use these folders only"
    # overrides `follow_pull`, and the card has to show which clock is
    # actually in charge. Computed here so the rule lives in one place
    # rather than being restated in TypeScript.
    out["follows_pull"] = rs.follows_pull(config)
    return out


@router.post("/retrain")
def save_retrain(payload: RetrainScheduleRequest,
                 session=Depends(require_session)) -> dict:
    patch = payload.model_dump(exclude_none=True)
    current = rs.settings()
    merged = {**current, **patch}
    if int(merged.get("every_days") or 0) < 1:
        raise HTTPException(
            status_code=400,
            detail="A retrain takes hours of GPU, so anything under a day "
                   "would overlap itself.",
        )
    try:
        pc.write_config({"retrain_schedule": merged})
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not save: {exc}")
    config = pc.read_config()

    # Switching it ON claims the slot that has already passed, so
    # enabling does not immediately fire for this morning.
    #
    # The clock looks BACKWARDS for the most recent occurrence, which is
    # what lets a container that was down catch up with one run. Enable
    # a daily 09:00 schedule at 09:30 and that same look-back finds
    # 09:00 today and starts a GPU run on the spot -- the operator
    # pressed Save, not Run. The same guard `queue_after_pull` has when
    # it is switched on, for the same reason.
    if patch.get("enabled") and not current.get("enabled"):
        slot = rs.due_training_run(config)
        if slot:
            pc.claim_once("training_schedule", slot)

    out = rs.preflight(config)
    out["next_runs"] = [m.isoformat() for m in rs.next_runs(out["settings"])]
    out["expression"] = rs.schedule_expression(out["settings"])
    out["follows_pull"] = rs.follows_pull(config)
    return out


#: The pull triggers that may chain. Anything else is a typo, and a typo
#: here is a chain that silently never fires.
CHAIN_TRIGGERS = ("schedule", "arrival", "manual")


class QueueAfterPullRequest(BaseModel):
    enabled: bool | None = None
    even_if_nothing_moved: bool | None = None
    # Declared, or the POST drops them without a word and the card looks
    # saved until the page is reloaded.
    #
    # `arrival` is the one to think about. It fires per upload burst --
    # on 2026-10-06 three fired in ten minutes, one of them moving a
    # single file -- so the two guards below are what make it a sane
    # choice rather than a way to spend a day of GPU on somebody
    # uploading photos one at a time.
    triggers: list[str] | None = None
    #: How many files a sweep must have moved to count as a batch.
    min_files: int | None = None
    #: Shortest gap between two chains, in minutes.
    cooldown_minutes: int | None = None


@router.get("/queue-after-pull")
def get_queue_after_pull(session=Depends(require_session)) -> dict:
    """The setting, plus what it would do right now."""
    from .. import queue_after_pull as qap

    config = pc.read_config()
    plan = qap.settings(config)
    decision = qap.pending(config)
    return {
        "settings": plan,
        # What the next tick would do. A background action with no way to
        # ask "and what will you do next" is one nobody trusts.
        "next": {"act": decision["act"], "reason": decision["reason"],
                 "run": decision["run"]},
    }


@router.post("/queue-after-pull")
def save_queue_after_pull(payload: QueueAfterPullRequest,
                          session=Depends(require_session)) -> dict:
    from .. import queue_after_pull as qap

    patch = payload.model_dump(exclude_none=True)

    if "triggers" in patch:
        wanted = [str(t).strip().lower() for t in patch["triggers"] if str(t).strip()]
        unknown = sorted(set(wanted) - set(CHAIN_TRIGGERS))
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"{', '.join(unknown)} is not a pull trigger. "
                       f"Choose from {', '.join(CHAIN_TRIGGERS)}.",
            )
        if not wanted:
            raise HTTPException(
                status_code=400,
                detail="Pick at least one trigger, or switch the chain off "
                       "instead -- an empty list reads as 'on' while doing "
                       "nothing, which is the one state nobody can debug.",
            )
        patch["triggers"] = wanted

    if "min_files" in patch and int(patch["min_files"]) < 1:
        raise HTTPException(status_code=400,
                            detail="A batch is at least one file.")
    if "cooldown_minutes" in patch and int(patch["cooldown_minutes"]) < 0:
        raise HTTPException(status_code=400,
                            detail="A cooldown cannot be negative.")

    current = qap.settings()
    merged = {**current, **patch}
    # Switching it ON marks the newest finished pull as already handled,
    # so turning it on does not immediately queue a run for a sweep that
    # happened last week. Enabling a feature must not be a way to start
    # GPU work by surprise.
    if patch.get("enabled") and not current.get("enabled"):
        # Against the triggers being SAVED, not the ones that were live a
        # moment ago: enabling `arrival` for the first time must mark the
        # newest arrival pull as handled, or switching it on immediately
        # chains the sweep that happened before anybody chose it.
        runs = qap._finished_pulls(1, plan=merged)
        merged["last_run_id"] = str(runs[0].get("run_id") or "") if runs else ""
    try:
        pc.write_config({qap.KEY: merged})
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not save: {exc}")
    config = pc.read_config()
    decision = qap.pending(config)
    return {"settings": qap.settings(config),
            "next": {"act": decision["act"], "reason": decision["reason"],
                     "run": decision["run"]}}


class RetrainRunRequest(BaseModel):
    """How the chain hands over at the end.

    `start` submits the run directly; `queue` adds it to the queue and
    releases the queue, which is what the schedule does. The training
    card uses `queue` so a composed run lands in the same list as every
    other run on that card, rather than appearing only as a job.
    """
    submit: str = "start"


@router.post("/retrain/run")
def run_retrain(body: RetrainRunRequest | None = None,
                session=Depends(require_session)) -> dict:
    """Run the chain now: rebuild, freeze, submit.

    The same function the schedule calls. A manual trigger that took a
    different path would make "it works when I press it" say nothing
    about whether the schedule works.

    This is ALSO what the training card's button calls when the run is
    composed from more than one source. The card's own queue entry
    names one folder, and one folder cannot carry a golden set that
    lives under another root -- only the chain builds that yaml.
    """
    submit = (body.submit if body else "start") or "start"
    if submit not in ("start", "queue"):
        raise HTTPException(status_code=400,
                            detail=f"submit must be 'start' or 'queue', not {submit!r}.")
    try:
        return rs.run_chain(trigger="manual", submit=submit)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not run it: {exc}")


def _pull_status() -> dict:
    """The sweep running now, or the last one that finished.

    Lives on the chain-status response rather than its own poll because
    the bar asking "what is happening" has to cover a pull that runs
    WITHOUT a chain behind it. A scheduled sweep is the one nobody is
    watching, and until this it was invisible anywhere except inside the
    Pull Control card -- which only looks while it is expanded, so a
    collapsed card meant a pull could run start to finish with nothing
    on screen saying so.

    `chains` is the other half of the answer: a pull that moved 1,600
    files and is followed by nothing looks identical to one that is
    about to retrain, and only this says which.
    """
    from .. import queue_after_pull as qap

    live = None
    try:
        live = pc.active_run()
    except Exception:                                       # noqa: BLE001
        live = None

    run = live
    if run is None:
        try:
            for got in pc.recent_run_states(10):
                if got.get("finished_at"):
                    run = got
                    break
        except Exception:                                   # noqa: BLE001
            run = None

    try:
        chains = bool(qap.settings().get("enabled"))
    except Exception:                                       # noqa: BLE001
        chains = False

    if not run:
        return {"run_id": "", "status": "", "trigger": "", "done": 0,
                "total": 0, "moved": 0, "skipped": 0, "errors": 0,
                "started_at": "", "finished_at": "", "chains": chains}
    return {
        "run_id": str(run.get("run_id") or ""),
        "status": str(run.get("status") or ""),
        "trigger": str(run.get("trigger") or ""),
        "done": int(run.get("done") or 0),
        "total": int(run.get("total") or 0),
        "moved": int(run.get("moved") or 0),
        "skipped": int(run.get("skipped") or 0),
        "errors": int(run.get("errors") or 0),
        "started_at": str(run.get("started_at") or ""),
        "finished_at": str(run.get("finished_at") or ""),
        "chains": chains,
    }


def _cpu_status() -> dict:
    """The processing half of a run, for the bar beside the GPU one.

    Between the end of TrainYolo and the end of the execution there is a
    stretch -- seven or eight minutes -- where the queue says a run is
    going and the GPU is genuinely free, because evaluation runs on an
    ml.m5.xlarge. Reported separately so the two readings stop looking
    like a contradiction.
    """
    try:
        jobs = t2.get_cpu_resource_usage()
    except Exception as exc:                                # noqa: BLE001
        return {"known": False, "busy": False, "step": "", "jobs": [],
                "error": str(exc)}
    ours = [job for job in jobs if job.get("own_pipeline")]
    step = next((str(job.get("step") or "") for job in ours if job.get("step")), "")
    return {"known": True, "busy": bool(jobs), "ours": bool(ours),
            "step": step, "jobs": jobs, "error": ""}


@router.get("/chain-status")
def chain_status(session=Depends(require_session)) -> dict:
    """The unattended chain's stages, and whether the GPU is free.

    One call because a status bar wants one poll, not two: the stages say
    what is happening and the GPU says whether the last of them can even
    start. The account has a single training slot, so "queued" and
    "running" are different answers to "when will this finish" only if
    you know what is holding it.

    Always returns the FULL stage list, pending ones included -- a bar
    that only lists steps already taken cannot show what is still to come.
    """
    gpu: list = []
    try:
        gpu = t2.get_gpu_resource_usage()
    except Exception as exc:                                # noqa: BLE001
        # A bar is not worth erroring a page over; say it is unknown.
        return {**rs.chain_progress(), "pull": _pull_status(),
                "gpu": {"known": False, "busy": False,
                        "jobs": [], "error": str(exc)},
                "cpu": _cpu_status()}
    return {
        **rs.chain_progress(),
        "pull": _pull_status(),
        "cpu": _cpu_status(),
        "gpu": {
            "known": True,
            # BUSY means "a run cannot start", not "a job object exists".
            #
            # A managed-spot job draws on a different quota and cannot
            # exhaust ours, and a job still waiting on capacity holds no
            # instance at all. Counting either made the bar read "GPU in
            # use -- another job" while the GPU was idle and a run would
            # have started fine -- which is how a spot smoke test stuck
            # in a 45-minute capacity retry loop blocked the queue.
            "busy": any(not j.get("spot") for j in gpu),
            # `own_pipeline` matters: a job this app started is progress,
            # one it did not is somebody else holding the only slot.
            "ours": any(j.get("own_pipeline") for j in gpu),
            "jobs": gpu,
            "error": "",
        },
    }


class SnapshotRequest(BaseModel):
    root: str = ""
    note: str = ""


@router.post("/snapshots")
def take_snapshot(payload: SnapshotRequest, session=Depends(require_session)) -> dict:
    """Freeze the current definition of a root so a run can be repeated.

    Writes two small objects and one row. No image is copied: the
    manifest is a list of names, and the images stay where the pull put
    them.
    """
    settings = _settings()
    root = (payload.root or settings.get("root") or "").strip("/")
    if not root:
        raise HTTPException(status_code=400, detail="Pick the folder that holds the weeks.")
    try:
        return ds.create(root=root, note=payload.note)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not snapshot {root}: {exc}")


#: The chain names its snapshot `f"{trigger} retrain"`, so the trigger is
#: the first word. `manual` is somebody pressing Run it now, which is not
#: an auto pull and so is not offered as one.
AUTO_PULL_TRIGGERS = ("after-pull", "schedule")


@router.get("/last-pull-dataset")
def last_pull_dataset(session=Depends(require_session)) -> dict:
    """The dataset the auto pull last generated, if there is one.

    Two places to look, and the second is why this is a route rather
    than a config read:

      1. `last_pull_dataset`, written by `run_chain`. Authoritative, and
         empty until a chain runs after this field was added.
      2. The snapshot store. Every chain run froze a snapshot and tagged
         it with its trigger, so the record survives even for runs that
         predate the config field -- which is every run so far.

    Without the second, this option would sit greyed until the next
    pull, while sixteen snapshots from previous pulls sat in the table.
    """
    recorded = pc.read_config().get("last_pull_dataset") or {}
    if recorded.get("folder"):
        return {"dataset": {**recorded, "found_in": "config"}}

    try:
        rows = ds.listing(limit=50)
    except Exception as exc:                                   # noqa: BLE001
        return {"dataset": None, "problem": f"Could not read the snapshots: {exc}"}

    for row in rows:
        note = str(row.get("note") or "").strip().lower()
        if not note.startswith(AUTO_PULL_TRIGGERS):
            continue
        return {"dataset": {
            "folder": row.get("root"),
            "snapshot_id": row.get("snapshot_id"),
            "files": row.get("file_count"),
            "at": row.get("created_at"),
            "set_by": note.split()[0],
            "found_in": "snapshots",
        }}
    return {"dataset": None}


@router.get("/snapshots")
def list_snapshots(root: str = "", limit: int = 50,
                   session=Depends(require_session)) -> dict:
    rows = ds.listing(root=root.strip("/"), limit=limit)
    # Reported rather than inferred from an empty list: "nothing frozen
    # yet" and "the table does not exist" send you to different places.
    return {"snapshots": rows, "ready": bool(rows) or ds.table_ready()}


@router.get("/snapshots/{snapshot_id}")
def read_snapshot(snapshot_id: str, session=Depends(require_session)) -> dict:
    try:
        return ds.get(snapshot_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/snapshots/{snapshot_id}/drift")
def snapshot_drift(snapshot_id: str, session=Depends(require_session)) -> dict:
    """What has moved under the snapshot since it was taken.

    A re-run WITH the manifest is unaffected either way -- that is what
    the manifest is for. This answers whether a re-run without it would
    have differed.
    """
    try:
        return ds.drift(snapshot_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not compare: {exc}")


@router.delete("/snapshots/{snapshot_id}")
def delete_snapshot(snapshot_id: str, session=Depends(require_session)) -> dict:
    """Throw away a frozen dataset.

    The images are untouched -- a snapshot is a list of names, not a copy
    of the corpus -- so what is lost is the ability to repeat that exact
    run, not the data it selected.
    """
    try:
        return ds.remove(snapshot_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not delete: {exc}")


@router.post("/snapshots/{snapshot_id}/restore")
def restore_snapshot(snapshot_id: str, session=Depends(require_session)) -> dict:
    """Put that snapshot's yaml back in the one slot the trainer reads.

    There is exactly one per root: `train.py` requires `data.yaml` at the
    root of the prefix it downloads, and only that parent holds every
    week folder the yaml names. Nothing is lost -- every snapshot keeps
    its own copy.
    """
    try:
        restored = ds.restore(snapshot_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not restore: {exc}")

    # Record WHICH definition is now live. Without this the restore moved
    # the yaml in the bucket and nothing else knew: the training card
    # reads `default_dataset`, so it went on reporting the snapshot the
    # last chain froze -- "frozen as 6afd7515dbf38be7" over a slot now
    # holding a different one. Loading a snapshot looked like it had done
    # nothing, and worse, the card named the wrong dataset while the run
    # would have used the right one.
    try:
        pc.write_config({"default_dataset": {
            "folder": restored["root"],
            "snapshot_id": restored["snapshot_id"],
            "files": restored.get("selected"),
            "set_by": "snapshot",
            # When the SNAPSHOT was taken, carried separately from `at`
            # below -- which is when it was put back. Showing the restore
            # time made every loaded snapshot read as "created just now",
            # which is the one thing it is not.
            "snapshot_created_at": restored.get("created_at") or "",
            "snapshot_note": restored.get("note") or "",
            # Whether the run will read a pinned file list or the week
            # folders. Recorded rather than left to the click that caused
            # it, because the card is read long after: a note set in
            # React state is gone on the next reload, and "which images
            # will this train on" is not a question that should expire.
            "pinned": bool(restored.get("pinned_from_manifest")
                           or restored.get("names_from") == "list"),
            "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        }})
    except Exception as exc:                                   # noqa: BLE001
        # The yaml IS restored; only the record of it failed. Saying so
        # beats failing a restore that actually happened.
        restored["selection_error"] = str(exc)
    return restored


@router.post("/snapshots/{snapshot_id}/train")
def train_on_snapshot(snapshot_id: str, session=Depends(require_session)) -> dict:
    """Restore that snapshot and START a run on it.

    Restoring puts the frozen definition back in the live slot; it does
    not train. Doing the two separately is right for somebody comparing
    datasets, and wrong for the far commoner "run this one again" -- who
    then has to find the training card, confirm the folder it is already
    pointed at, and press Start. Those are three chances to pick up a
    different dataset between restoring one and training on it.

    Goes through `run_chain`, the same path the schedule and Run it now
    use, so a run started here is assembled by the code every other run
    is assembled by. `submit="queue"` because the account has one GPU:
    the entry queues behind whatever is going rather than failing with
    SAGEMAKER_RESOURCE_LIMIT.
    """
    restored = restore_snapshot(snapshot_id, session=session)

    # `restore_snapshot` already recorded the selection; it is a hard
    # failure here rather than a note, because the chain is about to read
    # `default_dataset` and would otherwise assemble whatever the card
    # last had.
    if restored.get("selection_error"):
        raise HTTPException(
            status_code=400,
            detail=f"Restored, but could not select it: {restored['selection_error']}")

    try:
        outcome = rs.run_chain(trigger=f"snapshot {snapshot_id}", submit="queue")
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Restored, but could not start it: {exc}")
    return {"restore": restored, "run": outcome}


class MatrixRequest(BaseModel):
    root: str = ""
    #: Weeks beyond the ones that exist, so the rows carry on into the
    #: retrains that have not happened yet.
    ahead: int = 4
    #: Try the settings as the card currently reads, not as last saved.
    #: The grid is a what-if before it is a report.
    config: RetentionConfig | None = None


@router.post("/matrix")
def matrix(payload: MatrixRequest, session=Depends(require_session)) -> dict:
    """Every retrain against every week of data, as one grid.

    Rows are the week a retrain is deployed, columns the week the data was
    collected, the cell how much of that week it trains on.

    The preview answers "what would a retrain use today". This answers
    "what will a retrain use in two months, and is the oldest week still
    in it by then" -- which is the question that decides whether a
    retention policy is the right one, and the preview cannot show it.

    Reads only the week NAMES, so the future rows need no folders to
    exist. If the root cannot be listed the grid is still drawn from the
    count alone rather than failing: a policy is worth looking at before
    any data has been collected under it.
    """
    settings = _settings()
    if payload.config is not None:
        settings = {**settings, **payload.config.model_dump(exclude_none=True)}
    root = (payload.root or settings.get("root") or "").strip("/")

    names: list[str] = []
    sizes: dict = {}
    if root:
        try:
            # `read_weeks` already returns the stems, so the SIZES cost
            # nothing extra -- and they are what lets the grid apply the
            # minimum-image floor as well as the percentage one. Without
            # them a policy whose image minimum is the binding half is
            # understated.
            for found in rt.read_weeks(
                    root, week_prefix=_with_prefix({})["week_prefix"]):
                names.append(found["name"])
                sizes[found["name"]] = len(found.get("stems") or [])
        except Exception:                                      # noqa: BLE001
            names, sizes = [], {}
    if not names:
        # Nothing to list yet. Draw the shape of the policy anyway, over a
        # window wide enough that the decay is visibly going somewhere.
        names = [f"week_{n}" for n in range(1, 9)]

    golden_available = 0
    if settings.get("use_golden"):
        try:
            golden_available = int(
                rt.golden_summary(settings).get("available") or 0)
        except Exception:                                      # noqa: BLE001
            golden_available = 0

    ahead = max(0, min(int(payload.ahead or 0), 26))
    out = rt.schedule_matrix(names, settings, ahead=ahead, sizes=sizes,
                             golden_available=golden_available)
    out["root"] = root
    return out


@router.post("/preview")
def preview(payload: PreviewRequest, session=Depends(require_session)) -> dict:
    """Per week: what it holds, what survives the decay, what does not.

    Reads only. Nothing is copied and nothing is removed -- the week
    folders are never written by this feature at all.
    """
    settings = _settings()
    if payload.config is not None:
        settings = {**settings, **payload.config.model_dump(exclude_none=True)}
    root = (payload.root or settings.get("root") or "").strip("/")
    if not root:
        raise HTTPException(status_code=400, detail="Pick the folder that holds the weeks.")
    try:
        return rt.preview(root=root, config=_with_prefix(settings))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read {root}: {exc}")


@router.post("/build")
def build(payload: BuildRequest, session=Depends(require_session)) -> dict:
    """Assembles a training set with the older weeks thinned.

    The evaluation side is taken whole. An evaluation set that shrinks
    week by week makes each run's numbers incomparable with the last, and
    an improvement nobody can measure is not one worth shipping.
    """
    settings = _settings()
    root = (payload.root or settings.get("root") or "").strip("/")
    if not root:
        raise HTTPException(status_code=400, detail="Pick the folder that holds the weeks.")
    if not payload.classes:
        raise HTTPException(status_code=400, detail="Give the dataset a class list.")
    try:
        return rt.build(
            root=root,
            eval_root=(payload.eval_root or settings.get("eval_root") or "").strip("/"),
            classes={str(k): str(v) for k, v in payload.classes.items()},
            config=_with_prefix(settings),
            destination=payload.destination,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not build: {exc}")


# ---------------------------------------------------------------------------
# data.yaml over the week folders
#
# Points a trainer at the weeks where they already are. Nothing is
# copied: an Ultralytics `train:` takes a list of image directories, so
# consolidating them into one folder would duplicate the whole corpus to
# say something the yaml can say in six lines.
# ---------------------------------------------------------------------------

class DatasetYamlRequest(BaseModel):
    #: Write `train:`/`val:` as FILE LISTS holding exactly what the
    #: retention policy chose, rather than as the week folders whole.
    #: Without this the policy reaches nothing: a folder list trains on
    #: every image in it whatever the decay says.
    apply_retention: bool = False
    # The one root holding every week, with train/ and val/ inside each.
    root: str = ""
    # Superseded by `root`: two roots, one per split. Still accepted so a
    # saved configuration from before the merge still builds.
    training_root: str = ""
    eval_root: str = ""
    bucket: str = ""
    images_subpath: str = "images"
    labels_subpath: str = "labels"
    # Folders to look for an existing data.yaml in, for the class names.
    # Left empty, the pull control sources are used -- which is where the
    # labels came from, so it is where their indices are defined.
    class_sources: list[str] = []
    # Set only to override what was found. Indices must match the label
    # files; see find_classes.
    names: dict[str, str] | None = None


class DatasetYamlSaveRequest(DatasetYamlRequest):
    #: Write `train:`/`val:` as FILE LISTS holding exactly what the
    #: retention policy chose, rather than as the week folders whole.
    #: Without this the policy reaches nothing: a folder list trains on
    #: every image in it whatever the decay says.
    apply_retention: bool = False
    key: str = ""


def _class_sources(payload: DatasetYamlRequest) -> list:
    """Where to look for class names, defaulting to the pull sources."""
    if payload.class_sources:
        return payload.class_sources
    config = pc.read_config()
    picked = config.get("picked") or {}
    found: list = []
    for key in ("train_images", "train_labels", "val_images", "val_labels"):
        for folder in picked.get(key) or []:
            if folder and folder not in found:
                found.append(folder)
    return found


def _retained_entries(payload, config: dict) -> tuple:
    """What `train:`/`val:` should say when the policy is being applied.

    Named without writing anything, so the preview shows the form the
    saved yaml will take. The lists themselves are written on save.
    """
    root = (payload.root or config.get("root") or "").strip("/")
    if not root or not getattr(payload, "apply_retention", False):
        return "", ""
    # Relative to the root, and the lists sit AT the root -- see
    # retained_lists.FOLDER for why they have to.
    prefix = f"{rl.FOLDER}/" if rl.FOLDER else ""
    return f"{prefix}{rl.TRAIN_NAME}", f"{prefix}{rl.VAL_NAME}"


@router.post("/dataset-yaml")
def dataset_yaml(payload: DatasetYamlRequest, session=Depends(require_session)) -> dict:
    """What the data.yaml would be. Writes nothing."""
    config = pc.read_config()
    train_entry, val_entry = _retained_entries(payload, config)
    try:
        out = dy.build(
            root=payload.root or config.get("root") or "",
            training_root=payload.training_root or config.get("training_root") or "",
            eval_root=payload.eval_root or config.get("eval_root") or "",
            bucket=payload.bucket,
            source_bucket=config.get("source_bucket") or "",
            images_subpath=payload.images_subpath,
            labels_subpath=payload.labels_subpath,
            class_sources=_class_sources(payload),
            names={int(k): v for k, v in (payload.names or {}).items()},
            retained_train=train_entry,
            retained_val=val_entry,
        )
        if train_entry:
            # What the policy would actually select, so the preview can
            # say "587 of 599" rather than leaving the reader to trust
            # that the list will contain something sensible.
            chosen = rl.select(
                root=(payload.root or config.get("root") or "").strip("/"),
                config=_with_prefix(_settings()), bucket=payload.bucket,
                images_subpath=payload.images_subpath,
            )
            out["retained"] = {
                "train": len(chosen["train"]),
                "val": len(chosen["val"]),
                "train_available": chosen["train_available"],
                "val_available": chosen["val_available"],
                "weeks": chosen["weeks"],
            }
        return out
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read the weeks: {exc}")


@router.post("/dataset-yaml/save")
def save_dataset_yaml(payload: DatasetYamlSaveRequest,
                      session=Depends(require_session)) -> dict:
    """Writes the yaml to S3, and the file lists it points at.

    Rebuilt here rather than taking the text the card is showing: a yaml
    saved from a stale preview would point at week folders that have
    moved on since it was drawn.
    """
    config = pc.read_config()
    train_entry, val_entry = _retained_entries(payload, config)

    # The lists FIRST, then the yaml that points at them. The other order
    # leaves a window where the yaml names a file that is not there yet,
    # and a run started in that window fails on a missing split rather
    # than on anything a person did.
    lists = None
    if train_entry:
        try:
            lists = rl.write(
                root=(payload.root or config.get("root") or "").strip("/"),
                config=_with_prefix(_settings()), bucket=payload.bucket,
                images_subpath=payload.images_subpath,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:                               # noqa: BLE001
            raise HTTPException(status_code=400,
                                detail=f"Could not write the file lists: {exc}")
        if not lists["train"]:
            raise HTTPException(
                status_code=400,
                detail="The retention policy selected no training images at all.",
            )

    try:
        built = dy.build(
            root=payload.root or config.get("root") or "",
            training_root=payload.training_root or config.get("training_root") or "",
            eval_root=payload.eval_root or config.get("eval_root") or "",
            bucket=payload.bucket,
            source_bucket=config.get("source_bucket") or "",
            images_subpath=payload.images_subpath,
            labels_subpath=payload.labels_subpath,
            class_sources=_class_sources(payload),
            names={int(k): v for k, v in (payload.names or {}).items()},
            retained_train=train_entry,
            retained_val=val_entry,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not built["ready"]:
        blocking = "; ".join(p["text"] for p in built["problems"]
                             if p["level"] == "blocking")
        raise HTTPException(
            status_code=400,
            detail=f"Not writing a dataset that cannot train: {blocking}",
        )
    try:
        written = dy.save(key=payload.key or built["write_to"],
                          text=built["yaml"], bucket=payload.bucket)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not write it: {exc}")
    # Writing the live yaml is the act that UN-RESTORES a snapshot: the
    # slot a restore wrote into has just been overwritten with the week
    # folders. The record has to follow, or the config goes on claiming
    # a frozen dataset is live -- which is not only a wrong line on the
    # card. `_stamp_snapshot` keys on exactly that record, so a run
    # queued afterwards would carry "frozen snapshot <id>" as its
    # provenance while training on the live corpus.
    chosen = pc.read_config().get("default_dataset") or {}
    if str(chosen.get("set_by") or "") == "snapshot":
        try:
            pc.write_config({"default_dataset": {
                "folder": str(chosen.get("folder") or ""),
                "files": built.get("train_images", 0) + built.get("val_images", 0),
                "set_by": "rebuilt",
                "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            }})
        except Exception as exc:                               # noqa: BLE001
            print(f"Could not clear the snapshot marker after a rebuild: {exc}")

    out = {**built, "written": written}
    if lists:
        out["retained"] = {
            "train": len(lists["train"]),
            "val": len(lists["val"]),
            "train_available": lists["train_available"],
            "val_available": lists["val_available"],
            "weeks": lists["weeks"],
            "written": lists["written"],
        }
    return out


# --- the Scheduled page ---------------------------------------------------------

def scheduled_events(config: dict, *, days: int = 42,
                     now: _dt.datetime | None = None) -> dict:
    """Every run the clocks will start in the next `days`, and from what.

    Read from the SAME rules the queue runner acts on, not restated:

      * a scheduled PULL fires at `pc.next_pull_times` while Pull Control
        is in use and its mode is "schedule";
      * a TRAINING run fires either on its own clock (`rs.next_runs`,
        when the retrain schedule is on and does not follow the pull) or
        straight after each scheduled pull (`queue_after_pull` on, with
        "schedule" among its triggers).

    Arrival-triggered training cannot be put on a calendar -- it fires
    when files land -- so it is reported as a note, not as events.
    """
    from .. import queue_after_pull as qap

    now = now or _dt.datetime.now(_dt.timezone.utc)
    end = now + _dt.timedelta(days=max(1, min(int(days), 92)))
    picks = rs.source_selection(config)
    sources = [label for key, label in (("pull", "Pull"), ("golden", "Golden set"),
                                        ("folders", "Assembled folders"))
               if picks.get(key)]
    events: list = []
    notes: list = []

    pull_on = pc.pull_in_use(config) and str(config.get("mode") or "off") == "schedule"
    pulls: list = []
    if pull_on:
        every = pc.pull_interval_days(config)
        count = int((end - now).days / every) + 2
        # ISO strings from `next_pull_times`; parsed so they compare and
        # sort with the training clock's datetimes.
        pulls = []
        for raw in pc.next_pull_times(config, count=count, now=now):
            t = raw if isinstance(raw, _dt.datetime) else _dt.datetime.fromisoformat(str(raw))
            if t.tzinfo is None:
                t = t.replace(tzinfo=_dt.timezone.utc)
            if t <= end:
                pulls.append(t.astimezone(_dt.timezone.utc))
        for t in pulls:
            events.append({"at": t.isoformat(), "kind": "pull", "sources": ["Pull"],
                           "label": "Scheduled pull"})
    elif pc.pull_in_use(config) and str(config.get("mode") or "off") == "arrival":
        notes.append("Pull Control pulls when files arrive, so its pulls are not on a calendar.")

    chain = qap.settings(config)
    triggers = chain.get("triggers") or ["schedule"]
    plan = rs.settings(config)
    if plan.get("enabled") and not rs.follows_pull(config):
        every = max(1, int(plan.get("every_days") or 1))
        count = int((end - now).days / every) + 2
        for t in rs.next_runs(plan, start=now, count=count):
            if t <= end:
                events.append({"at": t.astimezone(_dt.timezone.utc).isoformat(), "kind": "train", "sources": sources,
                               "label": "Scheduled training run"})
    if chain.get("enabled") and "schedule" in triggers:
        for t in pulls:
            events.append({"at": t.isoformat(), "kind": "train", "sources": sources,
                           "label": "Training after the scheduled pull"})
    if chain.get("enabled") and "arrival" in triggers:
        notes.append(f"Training also starts after an arrival pull that moves at least "
                     f"{int(chain.get('min_files') or 1)} file(s) -- not on a calendar, "
                     "because it fires when files land.")

    events.sort(key=lambda e: (e["at"], e["kind"] != "pull"))
    return {
        "settings_file": str(config.get("settings_file") or ""),
        "now": now.isoformat(),
        "until": end.isoformat(),
        "next": next((e for e in events if e["kind"] == "train"), None),
        "next_pull": next((e for e in events if e["kind"] == "pull"), None),
        "sources": sources,
        "events": events,
        "notes": notes,
    }


def scheduled_for_files(names: list, *, days: int = 42,
                        now: _dt.datetime | None = None) -> dict:
    """The LIVE schedule, plus what each named saved file would schedule.

    Only one configuration is loaded at a time, and only its clocks fire.
    Every other saved file is read without being applied (`read_settings_file`
    applies nothing) and shown as what it WOULD run if loaded -- each event
    carries `file` and `live` so the page can say which is which.

    The live configuration stands in for the file it was loaded from:
    it may have been changed since that file was saved, and it is what
    actually runs.
    """
    live_config = pc.read_config()
    live_name = str(live_config.get("settings_file") or "")
    out = scheduled_events(live_config, days=days, now=now)
    blocks = [{"name": live_name or "(live settings)", "live": True, "error": "",
               "sources": out["sources"], "next": out["next"], "notes": out["notes"]}]
    for ev in out["events"]:
        ev.update(file=blocks[0]["name"], live=True)
    events = list(out["events"])

    for name in names:
        if not name or name == live_name:
            continue
        try:
            got = scheduled_events(pc.read_settings_file(name), days=days, now=now)
        except Exception as exc:                                  # noqa: BLE001
            blocks.append({"name": name, "live": False, "error": str(exc),
                           "sources": [], "next": None, "notes": []})
            continue
        for ev in got["events"]:
            ev.update(file=name, live=False)
        events += got["events"]
        blocks.append({"name": name, "live": False, "error": "", "sources": got["sources"],
                       "next": got["next"], "notes": got["notes"]})

    events.sort(key=lambda e: (e["at"], not e["live"], e["file"], e["kind"] != "pull"))
    out["events"] = events
    out["files"] = blocks
    try:
        out["available"] = [f["name"] for f in pc.list_settings_files().get("files") or []]
    except Exception:                                             # noqa: BLE001
        out["available"] = []
    return out


@router.get("/scheduled")
def get_scheduled(days: int = 42, files: str = "",
                  session=Depends(require_session)) -> dict:
    """What the Scheduled page draws: the file in effect, the next run,
    and every scheduled pull and training run in the window.

    `files`: saved settings files to show beside the live one, comma
    separated, or `*` for all of them.
    """
    wanted = [n.strip() for n in files.split(",") if n.strip()]
    if wanted == ["*"]:
        try:
            wanted = [f["name"] for f in pc.list_settings_files().get("files") or []]
        except Exception:                                         # noqa: BLE001
            wanted = []
    return scheduled_for_files(wanted, days=days)
