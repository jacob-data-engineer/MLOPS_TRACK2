"""Track2 model registry evaluation metrics -- read-only, backs
`/track2/metrics`. Bar-chart rendering moved entirely to the frontend
(the old app rendered it as a hand-rolled inline SVG server-side)."""

from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import os
import re

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import metrics_export as mx
from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()

#: Where the export card's own saved settings live.
#:
#: A SUBDIRECTORY of the same host volume the pull-control settings use
#: -- `/settings` is `/opt/track2-app/settings/<container>` on the EC2
#: box, so these survive a deploy replacing the container and a
#: development container cannot overwrite production's. Its own folder
#: because an export preset and a pull configuration have nothing to do
#: with each other: sharing one list would put "metrics for the weekly
#: report" next to "helmet pull, every Monday" in the same dropdown, and
#: Load on the wrong one would silently rearm a schedule.
EXPORT_SETTINGS_DIR = os.environ.get(
    "METRICS_EXPORT_SETTINGS_DIR",
    os.path.join(os.environ.get("PULL_CONTROL_SETTINGS_DIR", "/settings"), "export"),
)

#: A file name has to survive being a path segment. Anything else is a
#: way to write outside the settings folder.
SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$")


def _settings_path(name: str) -> str:
    name = (name or "").strip()
    if not SAFE_NAME.match(name):
        raise ValueError(
            "A name may hold letters, numbers, spaces, dots, dashes and "
            "underscores, and must start with a letter or a number.")
    return os.path.join(EXPORT_SETTINGS_DIR, f"{name}.json")

#: Every metric a version can carry, in the order a reader expects them.
#: The export's columns come from here rather than from whatever the
#: caller sent, so two exports of the same metrics always have the same
#: column order and diff cleanly.
METRIC_KEYS = ("mAP50", "mAP50-95", "precision", "recall", "f1", "accuracy")


@router.get("")
def get_metrics(session=Depends(require_session)) -> dict:
    return {"rows": t2.scan_all_model_metrics()}


class MetricsExportRequest(BaseModel):
    #: Which versions. Empty means every version the registry holds --
    #: the whole table rather than a comparison.
    model_package_arns: list[str] = []
    #: Which metrics. Empty means all of them.
    metrics: list[str] = []
    #: Where to put it. No bucket means "hand it back for a download".
    bucket: str = ""
    prefix: str = ""


def build_metrics_csv(arns: list[str], metrics: list[str]) -> tuple[str, int]:
    """The CSV, and how many versions are in it.

    ONE builder for both destinations. A download and an S3 copy of the
    same selection have to be byte-identical, or "I exported it" stops
    meaning one thing -- and the obvious shortcut, letting the browser
    build the download and the server build the S3 copy, is exactly how
    two formats of one export come to exist.

    Built from the registry rather than from anything the caller sent:
    the numbers in the file are the numbers the registry holds, not the
    numbers a page happened to be showing.
    """
    wanted = [k for k in METRIC_KEYS if not metrics or k in metrics]
    if not wanted:
        raise ValueError("Pick at least one metric to export.")

    rows = t2.scan_all_model_metrics()
    if arns:
        keep = set(arns)
        rows = [r for r in rows if r.get("model_package_arn") in keep]
    if not rows:
        raise ValueError("No versions matched the selection.")

    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow([
        "group", "registry", "generation", "size", "version", "status",
        *wanted,
        # The columns that stop a number being read out of context. A
        # 0.995 measured on five images is not a better score than a
        # 0.400 measured on two hundred, and a spreadsheet sorted on
        # mAP50 alone says it is.
        "validation_images", "classes", "dataset_version", "model_package_arn",
    ])
    for row in rows:
        m = row.get("metrics") or {}
        dataset = m.get("dataset") or {}
        writer.writerow([
            row.get("group_name", ""), row.get("registry", ""),
            row.get("generation", ""), row.get("size", ""),
            row.get("version", ""), row.get("status", ""),
            # Blank, not 0 -- a version that was never evaluated is not
            # one that scored zero, and a spreadsheet cannot tell the
            # difference once a 0 is written.
            *[("" if m.get(k) is None else m.get(k)) for k in wanted],
            dataset.get("validation_count", ""),
            " ".join(m.get("classes") or []),
            dataset.get("dataset_version", ""),
            row.get("model_package_arn", ""),
        ])
    return out.getvalue(), len(rows)


@router.post("/export")
def export_metrics(payload: MetricsExportRequest,
                   session=Depends(require_session)) -> dict:
    """Build the CSV, and either hand it back or write it to S3.

    One endpoint for both so the two can never drift apart.
    """
    try:
        text, count = build_metrics_csv(payload.model_package_arns, payload.metrics)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"track2-metrics-{stamp}.csv"

    if not payload.bucket:
        return {"filename": filename, "csv": text, "versions": count, "written_to": ""}

    key = f"{payload.prefix.strip('/')}/{filename}" if payload.prefix.strip("/") else filename
    try:
        t2.s3.put_object(
            Bucket=payload.bucket, Key=key,
            Body=text.encode("utf-8"), ContentType="text/csv",
        )
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Could not write s3://{payload.bucket}/{key}: {exc}",
        )
    return {"filename": filename, "csv": "", "versions": count,
            "written_to": f"s3://{payload.bucket}/{key}"}


# --- the export card's own saved settings ------------------------------
#
# Deliberately NOT the pull-control settings file. That one holds a whole
# retraining configuration -- sources, cut-off, schedule, detector -- and
# loading it rearms a schedule. An export preset is which Class, which
# criterion and where the checkpoint goes. Putting both in one dropdown
# would make "Load" a button whose consequences depend on which name you
# picked.
#
# NOTE what is NOT in here: the automatic export's armed state. Arming a
# watcher is its own call (`/export/auto/arm`), so loading a preset named
# after something else cannot quietly start writing to a shared bucket on
# a timer. Same reasoning as the pull schedule, which is armed in its card
# rather than by whichever settings file happened to be loaded.

class ExportSettings(BaseModel):
    #: The three dropdowns, stored so the card restores the selection a
    #: preset was saved against -- not just the group it resolved to.
    #: "Class helmet, YOLO26, size M" is what somebody chose; the group
    #: name is what that means.
    registry: str = ""
    generation: str = ""
    size: str = ""
    #: What those three resolve to, and the ONLY field the server acts
    #: on. A version belongs to a registry group, so the group is the
    #: real address; the three above are how it was arrived at.
    group_name: str = ""
    #: Which metric decides the best version, or "newest". Part of what
    #: the export IS: the same Class exported by mAP50 and by recall are
    #: two different files.
    criterion: str = "mAP50"
    #: "local" | "s3" -- where it goes, and mutually exclusive by
    #: construction rather than by a check. The same checkpoint
    #: downloaded to a laptop and written to a shared bucket are two
    #: different jobs, and a preset that restored everything but that
    #: would reproduce the wrong one.
    destination: str = "local"
    bucket: str = ""
    prefix: str = ""
    #: What happens to earlier exports of this Class in that folder:
    #: keep_all | archive_oldest | keep_latest. A retention decision, so
    #: it belongs in the preset -- restoring where an export goes without
    #: restoring what it does to what is already there would reproduce
    #: half the job.
    retention: str = "keep_all"


class ExportSettingsRequest(BaseModel):
    name: str = ""
    settings: ExportSettings | None = None


@router.get("/export/settings")
def list_export_settings(session=Depends(require_session)) -> dict:
    """Saved export presets on the EC2 host, newest first."""
    try:
        names = [f[:-5] for f in os.listdir(EXPORT_SETTINGS_DIR) if f.endswith(".json")]
    except OSError:
        # Missing is not an error: nothing has been saved yet, and the
        # card should offer Save rather than report a fault.
        return {"directory": EXPORT_SETTINGS_DIR, "files": [],
                "writable": os.access(os.path.dirname(EXPORT_SETTINGS_DIR) or "/", os.W_OK)}
    files = []
    for name in names:
        try:
            path = _settings_path(name)
            files.append({
                "name": name,
                "bytes": os.path.getsize(path),
                "modified": _dt.datetime.fromtimestamp(
                    os.path.getmtime(path), _dt.timezone.utc).isoformat(),
            })
        except (OSError, ValueError):
            continue
    files.sort(key=lambda f: f["modified"], reverse=True)
    return {"directory": EXPORT_SETTINGS_DIR, "files": files,
            "writable": os.access(EXPORT_SETTINGS_DIR, os.W_OK)}


@router.post("/export/settings/save")
def save_export_settings(payload: ExportSettingsRequest,
                         session=Depends(require_session)) -> dict:
    """Writes a preset to the host.

    The host, not the container: a file inside the container is destroyed
    by the next deploy, and a settings file that does not survive a deploy
    is not a settings file.
    """
    if payload.settings is None:
        raise HTTPException(status_code=422, detail="Nothing to save.")
    try:
        path = _settings_path(payload.name)
        os.makedirs(EXPORT_SETTINGS_DIR, exist_ok=True)
        # Written whole then moved: a half-written preset read back by the
        # next page load is worse than no preset.
        temporary = path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"saved_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
                       "settings": payload.settings.model_dump()}, handle, indent=2)
        os.replace(temporary, path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not save: {exc}")
    return {"name": payload.name, "path": path, "bytes": os.path.getsize(path)}


@router.post("/export/settings/load")
def load_export_settings(payload: ExportSettingsRequest,
                         session=Depends(require_session)) -> dict:
    """Reads a preset back. Nothing is exported by loading one."""
    try:
        with open(_settings_path(payload.name), encoding="utf-8") as handle:
            body = json.load(handle)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No preset named {payload.name!r}.")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not read it: {exc}")
    # Defaults filled in, so a preset written before a field existed loads
    # as the default rather than as undefined.
    return {"name": payload.name, "saved_at": body.get("saved_at", ""),
            "settings": ExportSettings(**(body.get("settings") or {})).model_dump()}


@router.post("/export/settings/delete")
def delete_export_settings(payload: ExportSettingsRequest,
                           session=Depends(require_session)) -> dict:
    try:
        os.remove(_settings_path(payload.name))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"No preset named {payload.name!r}.")
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not delete it: {exc}")
    return {"name": payload.name, "deleted": True}


# --- exporting the best checkpoint of one Class ------------------------
#
# What goes out is the .pt itself, not a spreadsheet about it. The picking,
# the naming and the watcher all live in `metrics_export`; these routes are
# the thin edge of it, so the button and the unattended watcher cannot
# choose differently.


class BestVersionRequest(BaseModel):
    group_name: str = ""
    criterion: str = "mAP50"


class WeightsExportRequest(BestVersionRequest):
    destination: str = "local"
    bucket: str = ""
    prefix: str = ""
    #: What happens to EARLIER exports of this Class once the new one
    #: lands: keep_all | archive_oldest | keep_latest.
    retention: str = "keep_all"


@router.get("/export/groups")
def exportable_groups(session=Depends(require_session)) -> dict:
    """Every registry group a model could land in, whether or not one has.

    The export card's three dropdowns are built from THIS, not from the
    versions that happen to exist. Built from what exists, a Class that
    has only ever trained at YOLO11-S offers YOLO11 and S and nothing
    else -- so there is no way to set up an export, or arm a watcher, for
    the group a YOLO26-X run will land in next week. Which is precisely
    when you would want it armed: the first version to appear is then
    exported the moment it registers.

    Enumerated, never composed by the browser. Rebuilding
    "yolo{gen}{size}-{slug}-detector" client-side is how the Models card
    once told somebody a YOLO26 run would register into
    `yolo11s-helmet-detector` while the run itself used the right name.
    """
    # The open-taxonomy registry is left OUT of this list on request.
    # It is not a Class: `custom` is where a run with no declared Class
    # lands, so it reads as a fifth Class in a dropdown labelled Class
    # while being a different kind of thing entirely. Those groups stay
    # fully visible on the comparison chart above, which enumerates from
    # the same `exportable_groups()` -- removed from a picker, not from
    # the platform, because a registered model that appears on no page is
    # how a YOLO26 open run was once trained, Approved and invisible.
    return {"groups": [g for g in t2.exportable_groups()
                       if g["registry"] != "custom"]}


@router.post("/export/preview")
def preview_best_version(payload: BestVersionRequest,
                         session=Depends(require_session)) -> dict:
    """Which version this criterion picks, without exporting it.

    Server-side, although the page already holds every row and could rank
    them itself. Two implementations of "best" -- one that decides what
    the card SAYS and one that decides what the button DOES -- is how a
    card comes to promise v4 and ship v9, and the difference would only
    show up in whatever was consuming the folder months later.
    """
    if not payload.group_name:
        raise HTTPException(status_code=422, detail="No registry group chosen.")
    try:
        # Briefly cached. One of these is a describe plus an S3 read per
        # version -- 1.7s for the 14-version helmet group -- and the card
        # calls it every time the criterion dropdown moves. Twenty
        # seconds of staleness is invisible here and saves ~28 AWS calls
        # per flick. The WATCHER deliberately takes no cache at all.
        rows = mx.group_versions(payload.group_name, max_age=20)
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not read {payload.group_name}: {exc}")
    best, why = mx.pick_best(rows, payload.criterion)
    return {
        "group_name": payload.group_name,
        "criterion": payload.criterion,
        "considered": len(rows),
        "best": mx.describe_pick(best, payload.criterion) if best else None,
        "reason": why,
    }


@router.post("/export/weights")
def export_best_weights(payload: WeightsExportRequest,
                        session=Depends(require_session)) -> dict:
    """Export the best version's checkpoint, now.

    The bytes never pass through this container either way -- S3 is a
    server-side copy, a download is a presigned link the browser follows
    straight to S3. A 19MB-110MB checkpoint streamed through here would
    stall every request in this single-worker process for its duration.
    """
    if payload.destination not in ("local", "s3"):
        raise HTTPException(status_code=422,
                            detail=f"{payload.destination!r} is not a destination.")
    try:
        return mx.run_export(
            payload.group_name, payload.criterion,
            destination=payload.destination,
            bucket=payload.bucket, prefix=payload.prefix,
            retention=payload.retention,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"The export failed: {exc}")


class FolderContentsRequest(BaseModel):
    group_name: str = ""
    bucket: str = ""
    prefix: str = ""


@router.post("/export/folder")
def export_folder_contents(payload: FolderContentsRequest,
                           session=Depends(require_session)) -> dict:
    """The earlier exports of this Class already in that folder.

    Here so the two housekeeping modes are legible BEFORE they run.
    "Keep only the latest" deletes objects in a bucket somebody chose,
    and a button whose blast radius you have to infer from its label is
    one nobody should press. This is the exact list those modes would
    act on -- the same `export_siblings` they call, not a second
    approximation of it.
    """
    if not payload.group_name or not payload.bucket:
        return {"files": [], "folder": ""}
    folder = (payload.prefix or "").strip("/")
    try:
        keys = mx.export_siblings(payload.bucket, payload.prefix, payload.group_name)
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not read that folder: {exc}")
    return {
        "folder": f"s3://{payload.bucket}/{folder}" if folder else f"s3://{payload.bucket}",
        "files": [k.rsplit("/", 1)[-1] for k in keys],
    }


class ArmAutoExportRequest(BestVersionRequest):
    bucket: str = ""
    prefix: str = ""
    retention: str = "keep_all"


@router.get("/export/auto")
def get_auto_export(session=Depends(require_session)) -> dict:
    """What the watcher is armed with, and whether it would fire now.

    `pending` is read-only and costs one `list_model_packages`, so the
    card can state what is about to happen rather than leaving a
    background job to be taken on trust.
    """
    plan = mx.auto_settings()
    try:
        decision = mx.pending(plan)
    except Exception as exc:                                       # noqa: BLE001
        # A registry that cannot be read is not a watcher that is off.
        decision = {"act": False, "reason": f"Could not read the registry: {exc}",
                    "version": 0}
    # The running count of .pt files where it writes -- and in archive/.
    # None when there is nowhere to count yet, or the folder cannot be read.
    counts = None
    if plan.get("bucket") and plan.get("group_name"):
        try:
            counts = mx.export_counts(plan["bucket"], plan.get("prefix") or "",
                                      plan["group_name"])
        except Exception as exc:                                   # noqa: BLE001
            print(f"Could not count exports: {exc}")
    # The version in the folder, scored by the measure the watcher ranks
    # on -- and the group's best by that measure now, beside it. Cached for
    # a minute: the card polls every 30s, and this is a describe and an S3
    # read per version. The WATCHER never takes the cache (see
    # group_versions); only this display does.
    if counts is not None:
        try:
            criterion = str(plan.get("criterion") or "")
            rows = mx.group_versions(plan["group_name"], max_age=60)
            keep = ("version", "value", "validation_images", "thin_validation")
            latest = counts.get("latest")
            if latest:
                row = next((r for r in rows
                            if str(r.get("version")) == str(latest["version"])), None)
                if row:
                    picked = mx.describe_pick(row, criterion)
                    latest.update({k: picked[k] for k in keep if k != "version"})
            best, _why = mx.pick_best(rows, criterion)
            counts["best"] = ({k: mx.describe_pick(best, criterion)[k] for k in keep}
                              if best else None)
            counts["criterion"] = criterion
        except Exception as exc:                                   # noqa: BLE001
            print(f"Could not score the exported version: {exc}")
    return {"settings": plan, "pending": decision, "counts": counts}


@router.post("/export/auto/arm")
def arm_auto_export(payload: ArmAutoExportRequest,
                    session=Depends(require_session)) -> dict:
    """Switch the watcher on for one group.

    Marks the group's current newest version as already seen, so arming
    on a group holding fourteen versions does not immediately export v14
    -- a version registered weeks ago is not "a new version registered".
    """
    try:
        return {"settings": mx.arm(
            group_name=payload.group_name, criterion=payload.criterion,
            bucket=payload.bucket, prefix=payload.prefix,
            retention=payload.retention)}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not arm it: {exc}")


@router.post("/export/auto/disarm")
def disarm_auto_export(session=Depends(require_session)) -> dict:
    try:
        return {"settings": mx.disarm()}
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"Could not switch it off: {exc}")
