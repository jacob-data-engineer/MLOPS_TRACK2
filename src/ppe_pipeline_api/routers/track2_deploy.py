"""Track2 model deploy -- pushes an Approved registry version (or your
own S3 weights) onto the ONE shared active Inference Component, then
optionally batch-infers a folder of images with it. Deliberately only
one model is ever live at a time (by explicit product decision):
deploying replaces whatever was there, regardless of size. Deploying
itself always goes through the inference queue below now (no separate
one-shot deploy endpoint, and no manual spin-up endpoint either -- every
deploy already requests CopyCount: 1, so starting a queued job is the
only action needed to bring a scaled-to-zero endpoint back up). Backs
`/track2/deploy`."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_deploy(browse_prefix: str = "", session=Depends(require_session)) -> dict:
    return {
        "registries": t2.serialize_registries_brief(),
        "deployed": t2.describe_deployed_inference_component(),
        # None when the endpoint has been torn down -- its normal resting
        # state between runs, which the UI shows as "not running, not
        # billing" rather than as an error.
        "endpoint": t2.endpoint_status(),
        "folders": t2.list_s3_folders(browse_prefix),
        "weight_files": t2.list_s3_weight_files(browse_prefix),
        "inference_queue": t2.list_inference_queue(),
        "inference_queue_status": t2.inference_queue_snapshot(),
    }


@router.get("/lineage")
def get_run_lineage(output_folder: str, run_id: str | None = None, session=Depends(require_session)) -> dict:
    """What produced one run's results: the source folder and the exact
    files read, the model size/version behind them, and when inference
    ran. Recorded in DynamoDB keyed by run id, with the output folder's
    own lineage.json as the fallback for runs that predate the table."""
    try:
        lineage = t2.read_run_lineage(run_id, output_folder)
    except ValueError as exc:
        # The folder is not a folder. Nothing here raises a global
        # ValueError handler, so without this the 400 it should be
        # becomes a 500 with a traceback -- which reads as the service
        # being broken rather than the argument being wrong.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if lineage is None:
        raise HTTPException(
            status_code=404,
            detail=f"No lineage recorded for {output_folder!r} -- it predates lineage being written, "
                   "or the run failed before inference started.",
        )
    return lineage


@router.post("/endpoint/delete")
def delete_endpoint(session=Depends(require_session)) -> dict:
    """Manual teardown. Queues already do this for themselves once they
    drain; this is for the case where the endpoint was left up (a job that
    died mid-run, a deploy that never settled) and is quietly billing."""
    return t2.delete_endpoint()


@router.get("/browse")
def browse_folders(prefix: str = "", session=Depends(require_session)) -> dict:
    """Just the folder-browser payload. `GET /track2/deploy` above also
    rebuilds the registry summary and the deployed-component state, which
    together cost ~30 serial AWS calls (4 list_model_packages, 12
    describe_model_package, their lineage lookups, 3 more for the live
    component) -- none of which a browser click needs. Navigating used to
    go through it anyway, so every click into a folder waited on all of
    that to list one prefix."""
    folders, weight_files = t2.list_s3_folders_and_weight_files(prefix)
    return {"folders": folders, "weight_files": weight_files}


class QueueInferenceRunRequest(BaseModel):
    size: str
    weights_kind: str  # "registry" | "s3"
    images_folder: str
    output_folder: str
    model_package_arn: str | None = None
    # For weights_kind="s3": exactly one of these two.
    weights_folder: str | None = None
    weights_key: str | None = None
    # For weights_kind="s3" only -- a free-text description of the
    # checkpoint, since it has no metadata of its own the way a
    # registered version does.
    notes: str | None = None
    # Applies regardless of weights_kind -- a free-text note about this
    # particular run, not about the checkpoint.
    comment: str | None = None


@router.post("/inference-queue/add")
def add_inference_queue_item(payload: QueueInferenceRunRequest, session=Depends(require_session)) -> dict:
    try:
        item = t2.queue_inference_run(
            size=payload.size,
            weights_kind=payload.weights_kind,
            images_folder=payload.images_folder,
            output_folder=payload.output_folder,
            model_package_arn=payload.model_package_arn,
            weights_folder=payload.weights_folder,
            weights_key=payload.weights_key,
            notes=payload.notes,
            comment=payload.comment,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # If the queue is already running, this item belongs behind whatever is
    # in flight -- so make sure the worker is still alive to reach it. The
    # loop exits the moment it finds nothing queued, which it can do in the
    # gap between the running job finishing and this add landing. Without
    # this the item silently waits for someone to press Start, and worse,
    # the drain will already have torn the endpoint down, so it pays a full
    # cold start for a queue that never actually went idle.
    # start_inference_queue is idempotent and restarts a dead thread.
    if t2.inference_queue_snapshot()["started"]:
        t2.start_inference_queue()

    return {"message": f"Queued {item['weights_label']} -> {item['images_folder']}", "item": item}


@router.post("/inference-queue/remove/{item_id}")
def remove_inference_queue_item(item_id: str, session=Depends(require_session)) -> dict:
    """Despite the URL (kept stable for the frontend), this deletes the
    record outright regardless of status -- queued, done, or failed --
    refusing only one still actively deploying/running. See
    delete_inference_queue_item."""
    if not t2.delete_inference_queue_item(item_id):
        raise HTTPException(status_code=404, detail="Item not found, or still deploying/running.")
    return {"removed": True}


@router.post("/inference-queue/start")
def start_inference_queue(session=Depends(require_session)) -> dict:
    if not t2.start_inference_queue():
        raise HTTPException(status_code=400, detail="Nothing queued to start.")
    return {"message": "Inference queue started."}


@router.post("/inference-queue/stop")
def stop_inference_queue(session=Depends(require_session)) -> dict:
    t2.stop_inference_queue()
    return {"message": "Inference queue stopped -- the run in progress, if any, will finish."}
