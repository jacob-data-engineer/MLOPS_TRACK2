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
