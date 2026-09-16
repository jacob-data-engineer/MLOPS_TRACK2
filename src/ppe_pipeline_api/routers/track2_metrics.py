"""Track2 model registry evaluation metrics -- read-only, backs
`/track2/metrics`. Bar-chart rendering moved entirely to the frontend
(the old app rendered it as a hand-rolled inline SVG server-side)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_metrics(session=Depends(require_session)) -> dict:
    return {"rows": t2.scan_all_model_metrics()}
