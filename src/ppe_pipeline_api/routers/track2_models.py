"""Track2 training control -- dataset/evaluation/weights S3 browsing, GPU
status, training progress, the ad-hoc training queue, and the per-run
training audit. Backs `/track2/models` (labelled "Track2 Training" in nav
to avoid colliding with the existing PPE `/models` page).

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
    # Same None-vs-"" distinction as browse_prefix above: None means "give
    # me the landing default", "" means the operator navigated to the
    # bucket root and must stay there.
    eval_browse_prefix: str | None = None,
    session=Depends(require_session),
) -> dict:
    dataset_prefix = t2.DATASETS_S3_PREFIX if browse_prefix is None else browse_prefix
    eval_prefix = t2.DATASETS_S3_PREFIX if eval_browse_prefix is None else eval_browse_prefix
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
        "dataset_browser": {
            "prefix": dataset_prefix,
            "folders": t2.list_s3_folders(dataset_prefix),
        },
        "weights_browser": {
            "prefix": weights_browse_prefix,
            "folders": t2.list_s3_folders(weights_browse_prefix),
        },
        "eval_browser": {
            "prefix": eval_prefix,
            "folders": t2.list_s3_folders(eval_prefix),
        },
    }


@router.get("/audit")
def training_audit(limit: int = 50, session=Depends(require_session)) -> dict:
    """Full record of every submitted run: the exact folders and file names
    on both sides, every parameter, the notes, which model version came out
    of it, and when it started and ended."""
    return {"audits": t2.scan_training_audits(limit=min(max(limit, 1), 200))}


@router.post("/audit/{audit_id}/delete")
def delete_training_audit(audit_id: str, session=Depends(require_session)) -> dict:
    """Removes a run from the history AND deletes the model version it
    produced -- the two are one record, so they go together."""
    try:
        return t2.delete_training_audit(audit_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found (already deleted?).")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class TrainingParams(BaseModel):
    generation: str
    size: str
    epochs: int = 50
    img_size: int = 640
    batch: int = 16
    seed: int = 42
    dataset_folder: str
    # Optional: unset, the pipeline evaluates on the training dataset, which
    # is what it has always done.
    eval_dataset_folder: str = ""
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
            payload.eval_dataset_folder, source="manual",
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
            payload.eval_dataset_folder,
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
            "eval_dataset_folder": payload.eval_dataset_folder,
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

