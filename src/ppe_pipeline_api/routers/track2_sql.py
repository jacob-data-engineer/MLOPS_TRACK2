"""The SQL tab's routes: which databases there are, their tables, and a
read-only query answered as rows plus the images in them."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from queue import Queue, Empty
from threading import Thread
import json

from ..auth import require_session
from ..data import sql_explorer as sx
from ..data import sql_settings
from ..data import sql_sensor
from ..data import sql_training_export
from typing import Literal

router = APIRouter()


@router.get("/sensor")
def get_sensor(session=Depends(require_session)):
    return sql_sensor.status()


@router.post("/sensor")
def save_sensor(payload: sql_sensor.SensorConfig, session=Depends(require_session)):
    try:
        return sql_sensor.save(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/sensor/check")
def check_sensor(session=Depends(require_session)):
    try:
        if not sql_sensor.start(manual=True):
            raise HTTPException(status_code=409, detail="A sensor check is already running.")
        return sql_sensor.status()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@router.post("/sensor/disable")
def disable_sensor(session=Depends(require_session)):
    try:
        config = sql_sensor.SensorConfig.model_validate(sql_sensor.status()["config"])
        config.enabled = False
        return sql_sensor.save(config)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


class SqlSettingsRequest(BaseModel):
    name: str
    settings: sql_settings.SqlSettings | None = None


def _settings_action(action, *args):
    try:
        return action(*args)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Settings file not found.")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except OSError:
        raise HTTPException(status_code=400, detail="Could not access the EC2 settings folder.")


@router.get("/settings")
def list_sql_settings(session=Depends(require_session)) -> dict:
    return _settings_action(sql_settings.list_files)


@router.post("/settings/save")
def save_sql_settings(payload: SqlSettingsRequest, session=Depends(require_session)) -> dict:
    if payload.settings is None:
        raise HTTPException(status_code=422, detail="Nothing to save.")
    return _settings_action(sql_settings.save_with_schedule, payload.name, payload.settings)


@router.post("/settings/load")
def load_sql_settings(payload: SqlSettingsRequest, session=Depends(require_session)) -> dict:
    return _settings_action(sql_settings.load, payload.name)


@router.post("/settings/delete")
def delete_sql_settings(payload: SqlSettingsRequest, session=Depends(require_session)) -> dict:
    return _settings_action(sql_settings.delete, payload.name)


class SchemaRequest(BaseModel):
    database: str


class QueryRequest(BaseModel):
    database: str
    sql: str
    # Narrow the images to one bucket and a folder inside it. Optional.
    bucket: str = ""
    folder: str = ""


class PullRequest(QueryRequest):
    class_ids: dict[str, int] | None = None
    export_mode: Literal["detection", "classification"] | None = None
    class_names: list[str] | None = None
    # Where the images go, as images/{train,val} + labels/{train,val}.
    dest_bucket: str
    dest_folder: str
    # Share of the images that go to train; the rest to val. 100 = all train.
    train_percent: int = 100
    # Write a YOLO label for each image (from the database's boxes when the
    # query returns them) and a data.yaml. Off: images only.
    with_labels: bool = True


def run_pull(payload, progress=None):
    options = dict(bucket=payload.bucket.strip(), folder=payload.folder.strip(),
                   dest_bucket=payload.dest_bucket, dest_folder=payload.dest_folder,
                   train_percent=payload.train_percent, progress=progress)
    if payload.export_mode and payload.with_labels:
        return sql_training_export.export(payload.database, payload.sql, **options,
                                         mode=payload.export_mode, names=payload.class_names)
    if sql_training_export.exists(payload.dest_bucket, payload.dest_folder.strip().strip("/") + "/export-contract.json"):
        raise ValueError("This destination is a labelled training export. Keep labels enabled or choose another folder.")
    return sx.pull(payload.database, payload.sql, **options, with_labels=payload.with_labels, class_ids=payload.class_ids)


class PresetRequest(BaseModel):
    # A class (hazard type) from /classes, and which of its frames.
    hazard: str
    database: str = "violation-review"
    scope: str = "resolved"


class PreviewRequest(BaseModel):
    uris: list[str]


@router.post("/previews")
def previews(payload: PreviewRequest, session=Depends(require_session)) -> dict:
    """Signed thumbnail links for up to one screenful of images."""
    return {"previews": sx.previews(payload.uris)}


@router.post("/classes")
def classes(payload: SchemaRequest, session=Depends(require_session)) -> dict:
    """The classes the review database has frames for, with counts."""
    try:
        return {"classes": sx.classes(payload.database)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/preset")
def preset(payload: PresetRequest, session=Depends(require_session)) -> dict:
    """The ready-made human-verified query for one class, as SQL the page
    puts in the editor -- so it can be read, changed and pulled from."""
    try:
        return {"sql": sx.preset_sql(payload.hazard, payload.scope,
                                   crop_verdicts=sx.has_crop_verdicts(payload.database))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/readiness")
def readiness(payload: SchemaRequest, session=Depends(require_session)) -> dict:
    """Where every judged frame stops on its way to retraining, per class,
    read live -- for the Readiness flow card. One read-only query."""
    try:
        return sx.readiness(payload.database)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/databases")
def databases(session=Depends(require_session)) -> dict:
    return {"databases": sx.databases()}


@router.post("/databases/{key}/start")
def start_database(key: str, session=Depends(require_session)) -> dict:
    """Start a stopped database. Takes several minutes to become available."""
    try:
        return sx.set_running(key, True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/databases/{key}/stop")
def stop_database(key: str, session=Depends(require_session)) -> dict:
    """Stop a running database, so it costs nothing while unused."""
    try:
        return sx.set_running(key, False)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/schema")
def schema(payload: SchemaRequest, session=Depends(require_session)) -> dict:
    try:
        return {"tables": sx.schema(payload.database)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/query")
def query(payload: QueryRequest, session=Depends(require_session)) -> dict:
    try:
        return sx.run(payload.database, payload.sql,
                      bucket=payload.bucket.strip(), folder=payload.folder.strip())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/pull")
def pull(payload: PullRequest, session=Depends(require_session)) -> dict:
    """Run the query and copy the images it names into a training folder.
    Copies only: nothing at the source or the destination is deleted."""
    try:
        return run_pull(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/pull/stream")
def pull_stream(payload: PullRequest, session=Depends(require_session)):
    """Stream real copy progress; completion follows the metadata writes."""
    events = Queue()

    def worker():
        try:
            result = run_pull(payload, progress=lambda value: events.put({"type": "progress", **value}))
            events.put({"type": "complete", "result": result})
        except Exception as exc:
            events.put({"type": "error", "detail": str(exc)})

    def stream():
        Thread(target=worker, daemon=True).start()
        while True:
            try:
                event = events.get(timeout=10)
            except Empty:
                yield "\n"
                continue
            yield json.dumps(event) + "\n"
            if event["type"] in ("complete", "error"):
                break

    return StreamingResponse(stream(), media_type="application/x-ndjson",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
