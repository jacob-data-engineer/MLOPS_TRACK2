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


@router.get("/databases")
def databases(session=Depends(require_session)) -> dict:
    return {"databases": sx.databases()}


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
                       train_percent=payload.train_percent)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
