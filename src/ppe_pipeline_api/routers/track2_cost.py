"""Track2 SageMaker spend -- per-day totals split by usage type, so the
standing cost of the hosting endpoint is visible next to the one-off cost
of training jobs and Studio sessions.

Read on demand only. Cost Explorer bills $0.01 per request, so there is
deliberately no polling here and no call on page render -- the UI fetches
once and then only when someone presses Refresh. Backs `/track2/cost`."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_cost(
    days: int = t2.COST_DEFAULT_DAYS,
    refresh: bool = False,
    session=Depends(require_session),
) -> dict:
    """Cached for COST_CACHE_TTL_SECONDS by default -- the Deploy page reads
    today's endpoint cost on every visit, and Cost Explorer charges per
    request. refresh=true is what the Cost tab's own button sends."""
    return t2.sagemaker_cost_breakdown(days=days, refresh=refresh)
