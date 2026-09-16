"""Track2 "Export Weights" -- pulls an Approved registry package's
model.pt out to an arbitrary S3 folder as `selected.pt`. Backs
`/track2/export`."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("")
def get_export(browse_prefix: str = "", session=Depends(require_session)) -> dict:
    return {
        "registries": t2.serialize_registries_brief(),
        "folders": t2.list_s3_folders(browse_prefix),
    }


class ExportRequest(BaseModel):
    model_package_arn: str
    destination_folder: str


@router.post("")
def export_weights(payload: ExportRequest, session=Depends(require_session)) -> dict:
    if not payload.model_package_arn:
        raise HTTPException(status_code=422, detail="No model selected.")
    try:
        new_key, archived_key = t2.export_weights_to_folder(payload.model_package_arn, payload.destination_folder)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Export failed: {exc}")
    message = f"Exported to s3://{t2.BUCKET}/{new_key}"
    if archived_key:
        message += f" (previous weights archived to s3://{t2.BUCKET}/{archived_key})"
    return {"message": message, "new_key": new_key, "archived_key": archived_key}
