"""Track2 training lineage -- read-only, backs `/track2/lineage`."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_lineage(session=Depends(require_session)) -> dict:
    return {"records": t2.scan_all_lineage()}
