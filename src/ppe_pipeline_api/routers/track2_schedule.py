"""Track2 training schedule -- the single global retrain EventBridge rule
(manual/daily/weekly/monthly) plus a manual "train now" trigger. Backs
`/track2/schedule`.

Automated Training Queues (persistent, individually-scheduled recipes,
distinct from this one global schedule) are added/edited/deleted via
`track2_models`'s endpoints -- this page only needs to list them, which it
gets from this router's GET response, same data `track2_models` also
independently fetches (this mirrors the old app's own comment on why
these aren't shared/cached across tabs)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_schedule(session=Depends(require_session)) -> dict:
    return {
        "schedule": t2.current_schedule(),
        "training_progress": t2.get_training_progress(),
        "automated_queues": t2.scan_automated_queues(),
    }


@router.post("/train-now")
def train_now(session=Depends(require_session)) -> dict:
    already_running = t2.training_already_running_reason(t2.get_training_progress())
    if already_running:
        raise HTTPException(status_code=409, detail=already_running)
    response = t2.sm.start_pipeline_execution(PipelineName=t2.PIPELINE_NAME)
    arn = response["PipelineExecutionArn"]
    return {"message": f"Started: {arn.rsplit('/', 1)[-1]}"}


class ScheduleRequest(BaseModel):
    mode: str  # "manual" | "daily" | "weekly" | "monthly"
    daily_hour: int | None = None
    weekly_day: str | None = None
    weekly_hour: int | None = None
    monthly_day: int | None = None
    monthly_hour: int | None = None


def _form_dict(payload: ScheduleRequest) -> dict:
    return {k: str(v) for k, v in payload.model_dump().items() if v is not None}


@router.post("")
def set_schedule(payload: ScheduleRequest, session=Depends(require_session)) -> dict:
    if payload.mode == "manual":
        t2.events.disable_rule(Name=t2.RULE_NAME)
        return {"message": "Schedule disabled - manual trigger only"}

    try:
        expr, summary = t2.cron_and_summary_from_schedule_form(payload.mode, _form_dict(payload))
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    t2.events.put_rule(Name=t2.RULE_NAME, ScheduleExpression=expr, State="ENABLED")
    return {"message": f"Schedule updated: {summary}", "expression": expr}
