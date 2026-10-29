"""The central data catalog's routes (2026-10-29): is it reachable, what is
in it, and create its tables. Step 1 of the central store; filing,
versions, recreation and export build on these tables."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from ..auth import require_session
from ..data import catalog

router = APIRouter()


@router.get("/status")
def status(session=Depends(require_session)) -> dict:
    return catalog.status()


@router.post("/schema")
def schema(session=Depends(require_session)) -> dict:
    """Create any missing table. Idempotent; nothing is ever dropped."""
    try:
        return catalog.ensure_schema()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not reach the catalog: {type(exc).__name__}: {exc}")
