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


@router.get("/today")
def cost_today(session=Depends(require_session)) -> dict:
    """What today's runs have cost, measured from the jobs themselves.

    Separate from `/track2/cost`, which reads Cost Explorer: CE is the
    authority and lags roughly a day, so on the afternoon of a day spent
    training it answers $0.00. A cost figure that reads "free" while the
    GPU is warm is the one answer it must never give.

    Cheap enough to call on a page load -- two list calls and a describe
    per job, no Cost Explorer charge.
    """
    return t2.training_cost_today()
