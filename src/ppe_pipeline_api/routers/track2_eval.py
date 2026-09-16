"""Track2 evaluation queue -- run a registry version OR your own S3 .pt
weights against a chosen dataset folder (YOLO-format images/+labels/) and
get back real detection metrics (mAP50, mAP50-95, precision, recall,
accuracy), persisted in DynamoDB so results survive a process restart.
Backs `/track2/eval`.

Deploys the requested weights before evaluating (the same path a real
deploy already uses), so an eval run temporarily occupies that size's
Inference Component -- see track2_shared.py's module docstring on this
subsystem for why."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_eval(session=Depends(require_session)) -> dict:
    return {
        "runs": t2.list_eval_runs(),
        "queue": t2.eval_queue_snapshot(),
        "registries": t2.serialize_registries_brief(),
        "generation_sizes": t2.GENERATION_SIZES,
        "size_labels": t2.SIZE_LABELS,
    }


class QueueEvalRunRequest(BaseModel):
    size: str
    data_folder: str
    weights_kind: str  # "registry" | "s3"
    model_package_arn: str | None = None
    s3_folder: str | None = None


@router.post("/add")
def add_eval_run(payload: QueueEvalRunRequest, session=Depends(require_session)) -> dict:
    if payload.size not in t2.REGISTRY_SIZES:
        raise HTTPException(status_code=404, detail=f"Unknown size: {payload.size!r}")
    try:
        run = t2.queue_eval_run(
            size=payload.size,
            data_folder=payload.data_folder,
            weights_kind=payload.weights_kind,
            model_package_arn=payload.model_package_arn,
            s3_folder=payload.s3_folder,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"message": f"Queued eval of {run['weights_label']} against {run['data_folder']}", "run": run}


@router.post("/remove/{run_id}")
def remove_eval_run(run_id: str, session=Depends(require_session)) -> dict:
    if not t2.remove_eval_run(run_id):
        raise HTTPException(status_code=404, detail="Run not found, or already started/finished.")
    return {"removed": True}


@router.post("/start")
def start_eval_queue(session=Depends(require_session)) -> dict:
    if not t2.start_eval_queue():
        raise HTTPException(status_code=400, detail="Nothing queued to start.")
    return {"message": "Eval queue started."}


@router.post("/stop")
def stop_eval_queue(session=Depends(require_session)) -> dict:
    t2.stop_eval_queue()
    return {"message": "Eval queue stopped -- the run in progress, if any, will finish."}
