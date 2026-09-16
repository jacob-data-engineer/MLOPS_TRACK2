"""Track2 training control -- dataset/weights S3 browsing, GPU status,
training progress, the ad-hoc training queue, and persistent Automated
Training Queues (EventBridge-scheduled recipes). Backs
`/track2/models` (labelled "Track2 Training" in nav to avoid colliding
with the existing PPE `/models` page).

This is the largest single piece ported from the old review app's Models
tab -- see `track2_shared.py` for the underlying business logic; this
router is mostly just parameter parsing + response shaping."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_models(
    # None (the param genuinely omitted) means "give me the nice landing
    # default"; "" (the param explicitly present but empty) means "the
    # bucket root" -- these must stay distinguishable. Collapsing both to
    # "" via `browse_prefix or DATASETS_S3_PREFIX` made navigating to the
    # bucket root silently snap back to the default subfolder instead,
    # since an empty string is falsy in that expression regardless of
    # which of the two cases it actually was.
    browse_prefix: str | None = None,
    weights_browse_prefix: str = "",
    session=Depends(require_session),
) -> dict:
    dataset_prefix = t2.DATASETS_S3_PREFIX if browse_prefix is None else browse_prefix
    registries = t2.all_model_registries()
    training_progress = t2.get_training_progress()
    pending_message = t2.pending_submission_overlay(training_progress)
    return {
        "registries": t2.serialize_registries_brief(),
        "latest_approved": t2.latest_approved_lookup(registries),
        "generation_sizes": t2.GENERATION_SIZES,
        "generation_labels": t2.GENERATION_LABELS,
        "size_labels": t2.SIZE_LABELS,
        "default_dataset": t2.DEFAULT_TRAIN_DATASET,
        "training_progress": training_progress,
        "pending_message": pending_message,
        "gpu_jobs": t2.get_gpu_resource_usage(),
        "queue": t2.queue_snapshot(),
        "automated_queues": t2.scan_automated_queues(),
        "dataset_browser": {
            "prefix": dataset_prefix,
            "folders": t2.list_s3_folders(dataset_prefix),
        },
        "weights_browser": {
            "prefix": weights_browse_prefix,
            "folders": t2.list_s3_folders(weights_browse_prefix),
        },
    }


class TrainingParams(BaseModel):
    generation: str
    size: str
    epochs: int = 50
    img_size: int = 640
    batch: int = 16
    seed: int = 42
    dataset_folder: str
    base_model_folder: str = ""
    notes: str = ""


@router.post("/train/dataset")
def start_training(payload: TrainingParams, session=Depends(require_session)) -> dict:
    already_running = t2.training_already_running_reason(t2.get_training_progress())
    if already_running:
        raise HTTPException(status_code=409, detail=already_running)
    try:
        build_id = t2.start_training_run(
            payload.generation, payload.size, payload.epochs, payload.dataset_folder,
            payload.base_model_folder, payload.notes[:2048], payload.img_size, payload.batch, payload.seed,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Training trigger failed: {exc}")
    return {"message": f"Training started (YOLO{payload.generation}-{payload.size}, {payload.epochs} epochs, {payload.dataset_folder}): {build_id}", "build_id": build_id}


@router.post("/train/stop")
def stop_training(session=Depends(require_session)) -> dict:
    """Stops the currently Executing pipeline execution, if any. A stopped
    execution's steps end in a Stopped/Stopping state -- SageMaker does not
    support resuming one, a fresh run has to be started instead."""
    progress = t2.get_training_progress()
    if not progress or progress["status"] != "Executing":
        raise HTTPException(status_code=409, detail="No training run is currently executing.")
    try:
        t2.sm.stop_pipeline_execution(PipelineExecutionArn=progress["execution_arn"])
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Stop failed: {exc}")
    return {"message": f"Stopping run {progress['execution_arn'].rsplit('/', 1)[-1]}..."}


@router.post("/queue/add")
def add_to_queue(payload: TrainingParams, session=Depends(require_session)) -> dict:
    try:
        epochs, img_size, batch, seed = t2.validate_training_params(
            payload.epochs, payload.dataset_folder, payload.img_size, payload.batch, payload.seed,
        )
        entry = {
            "id": uuid.uuid4().hex[:12],
            "generation": payload.generation,
            "size": payload.size,
            "epochs": epochs,
            "img_size": img_size,
            "batch": batch,
            "seed": seed,
            "dataset_folder": payload.dataset_folder,
            "base_model_folder": payload.base_model_folder,
            "notes": payload.notes[:2048],
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        t2.queue_add(entry)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Add to queue failed: {exc}")
    return {"message": f"Added to queue (YOLO{payload.generation}-{payload.size}, {epochs} epochs, {payload.dataset_folder})", "entry": entry}


@router.post("/queue/remove/{entry_id}")
def remove_from_queue(entry_id: str, session=Depends(require_session)) -> dict:
    if not t2.queue_remove(entry_id):
        raise HTTPException(status_code=404, detail="Queue entry not found (already started or removed)")
    return {"removed": True}


@router.post("/queue/start")
def start_queue(session=Depends(require_session)) -> dict:
    if not t2.queue_start():
        raise HTTPException(status_code=400, detail="Queue is empty -- add at least one run first.")
    return {"message": "Queue started"}


@router.post("/queue/stop")
def stop_queue(session=Depends(require_session)) -> dict:
    t2.queue_stop()
    return {"message": "Queue stopped -- any run already in progress will finish; remaining entries stay queued."}


class AutomatedQueueRequest(BaseModel):
    generation: str
    size: str
    epochs: int = 50
    img_size: int = 640
    batch: int = 16
    seed: int = 42
    dataset_folder: str = ""
    base_model_folder: str = ""
    notes: str = ""
    mode: str = "weekly"  # "daily" | "weekly" | "monthly"
    daily_hour: int | None = None
    weekly_day: str | None = None
    weekly_hour: int | None = None
    monthly_day: int | None = None
    monthly_hour: int | None = None


def _schedule_form(payload: AutomatedQueueRequest) -> dict:
    return {k: str(v) for k, v in payload.model_dump().items() if v is not None}


@router.post("/automated-queue/add")
def add_automated_queue(payload: AutomatedQueueRequest, session=Depends(require_session)) -> dict:
    try:
        epochs, img_size, batch, seed = t2.validate_training_params(
            payload.epochs, payload.dataset_folder, payload.img_size, payload.batch, payload.seed,
        )
        form = _schedule_form(payload)
        schedule_expression, schedule_summary = t2.cron_and_summary_from_schedule_form(payload.mode, form)
        base_model_s3_uri = t2.resolve_base_model_s3_uri(payload.base_model_folder) if payload.base_model_folder else None
        now = datetime.now(timezone.utc).isoformat()
        queue = {
            "id": uuid.uuid4().hex[:12],
            "generation": payload.generation, "size": payload.size,
            "epochs": epochs, "img_size": img_size, "batch": batch, "seed": seed,
            "dataset_folder": payload.dataset_folder, "base_model_folder": payload.base_model_folder,
            "base_model_s3_uri": base_model_s3_uri, "notes": payload.notes[:2048],
            "schedule_mode": payload.mode,
            "schedule_hour": form.get(f"{payload.mode}_hour"),
            "schedule_day": payload.weekly_day if payload.mode == "weekly" else None,
            "schedule_dom": payload.monthly_day if payload.mode == "monthly" else None,
            "schedule_expression": schedule_expression,
            "schedule_summary": schedule_summary,
            "created_at": now, "updated_at": now,
        }
        t2.automated_queues_table().put_item(Item=queue)
        t2.put_automated_queue_rule(queue)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Add automated queue failed: {exc}")
    return {"message": f"Automated queue added (YOLO{payload.generation}-{payload.size}, {schedule_summary})", "queue": queue}


@router.post("/automated-queue/update/{queue_id}")
def update_automated_queue(queue_id: str, payload: AutomatedQueueRequest, session=Depends(require_session)) -> dict:
    try:
        existing = t2.automated_queues_table().get_item(Key={"id": queue_id}).get("Item")
        if not existing:
            raise ValueError(f"Automated queue {queue_id} not found (already deleted?).")
        epochs, img_size, batch, seed = t2.validate_training_params(
            payload.epochs, payload.dataset_folder, payload.img_size, payload.batch, payload.seed,
        )
        form = _schedule_form(payload)
        schedule_expression, schedule_summary = t2.cron_and_summary_from_schedule_form(payload.mode, form)
        base_model_s3_uri = t2.resolve_base_model_s3_uri(payload.base_model_folder) if payload.base_model_folder else None
        queue = {
            **existing,
            "generation": payload.generation, "size": payload.size,
            "epochs": epochs, "img_size": img_size, "batch": batch, "seed": seed,
            "dataset_folder": payload.dataset_folder, "base_model_folder": payload.base_model_folder,
            "base_model_s3_uri": base_model_s3_uri, "notes": payload.notes[:2048],
            "schedule_mode": payload.mode,
            "schedule_hour": form.get(f"{payload.mode}_hour"),
            "schedule_day": payload.weekly_day if payload.mode == "weekly" else None,
            "schedule_dom": payload.monthly_day if payload.mode == "monthly" else None,
            "schedule_expression": schedule_expression,
            "schedule_summary": schedule_summary,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        t2.automated_queues_table().put_item(Item=queue)
        # put_rule/put_targets are upserts by name -- this replaces the
        # existing rule's schedule/payload in place.
        t2.put_automated_queue_rule(queue)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Update automated queue failed: {exc}")
    return {"message": f"Automated queue updated (YOLO{payload.generation}-{payload.size}, {schedule_summary})", "queue": queue}


@router.post("/automated-queue/delete/{queue_id}")
def delete_automated_queue(queue_id: str, session=Depends(require_session)) -> dict:
    t2.delete_automated_queue_rule(queue_id)
    t2.automated_queues_table().delete_item(Key={"id": queue_id})
    return {"deleted": True}
