"""The SQL tab's routes: which databases there are, their tables, and a
read-only query answered as rows plus the images in them."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..auth import require_session
from ..data import sql_explorer as sx

router = APIRouter()


class SchemaRequest(BaseModel):
    database: str


class QueryRequest(BaseModel):
    database: str
    sql: str
    # Narrow the images to one bucket and a folder inside it. Optional.
    bucket: str = ""
    folder: str = ""


class PullRequest(QueryRequest):
    # Where the images go, as images/{train,val} + labels/{train,val}.
    dest_bucket: str
    dest_folder: str
    # Share of the images that go to train; the rest to val. 100 = all train.
    train_percent: int = 100
    # Write a YOLO label for each image (from the database's boxes when the
    # query returns them) and a data.yaml. Off: images only.
    with_labels: bool = True


class PresetRequest(BaseModel):
    # A class (hazard type) from /classes, and which of its frames.
    hazard: str
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
        return {"sql": sx.preset_sql(payload.hazard, payload.scope)}
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
        return sx.pull(payload.database, payload.sql,
                       bucket=payload.bucket.strip(), folder=payload.folder.strip(),
                       dest_bucket=payload.dest_bucket, dest_folder=payload.dest_folder,
                       train_percent=payload.train_percent,
                       with_labels=payload.with_labels)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
