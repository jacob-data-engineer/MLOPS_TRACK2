"""The central data catalog's routes (2026-10-29): is it reachable, what is
in it, and create its tables. Step 1 of the central store; filing,
versions, recreation and export build on these tables."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from ..auth import require_session
from ..data import catalog, catalog_sync, catalog_view

router = APIRouter()


@router.get("/status")
def status(session=Depends(require_session)) -> dict:
    return catalog.status()


@router.post("/schema")
def schema(session=Depends(require_session)) -> dict:
    """Create any missing table. Idempotent; nothing is ever dropped."""
    try:
        return catalog.ensure_schema()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not reach the catalog: {type(exc).__name__}: {exc}")


@router.post("/sync")
def sync(session=Depends(require_session)) -> dict:
    """Copy the records from DynamoDB now, in the background. The tick
    does this every few minutes anyway; this is for the first, full copy
    and for not waiting."""
    if not catalog.configured():
        raise HTTPException(status_code=400, detail="The catalog is not set up on this container.")
    started = catalog_sync.start_background()
    return {"started": started,
            "detail": "Copying in the background." if started else "A sync is already running."}


# --- the Data Catalog page: browse runs and versions, and export them --------

def _guard(work):
    try:
        return work()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"The catalog did not answer: {type(exc).__name__}: {exc}")


@router.get("/runs")
def runs(limit: int = 100, offset: int = 0, session=Depends(require_session)) -> dict:
    return _guard(lambda: catalog_view.runs(limit, offset))


@router.get("/versions")
def versions(limit: int = 100, offset: int = 0, session=Depends(require_session)) -> dict:
    return _guard(lambda: catalog_view.versions(limit, offset))


@router.get("/runs/{run_id}/files")
def run_files(run_id: str, split: str = "", role: str = "", kind: str = "image",
              limit: int = 100, offset: int = 0, session=Depends(require_session)) -> dict:
    return _guard(lambda: catalog_view.run_files(run_id, split=split, role=role, kind=kind,
                                                 limit=limit, offset=offset))


@router.get("/versions/{version_id}/files")
def version_files(version_id: str, split: str = "", limit: int = 100, offset: int = 0,
                  session=Depends(require_session)) -> dict:
    return _guard(lambda: catalog_view.version_files(version_id, split=split,
                                                     limit=limit, offset=offset))


@router.get("/runs/{run_id}/files.csv", response_class=PlainTextResponse)
def run_files_csv(run_id: str, session=Depends(require_session)) -> PlainTextResponse:
    body = _guard(lambda: catalog_view.file_list_csv("run", run_id))
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="run-{run_id}-files.csv"'})


@router.get("/versions/{version_id}/files.csv", response_class=PlainTextResponse)
def version_files_csv(version_id: str, session=Depends(require_session)) -> PlainTextResponse:
    body = _guard(lambda: catalog_view.file_list_csv("version", version_id))
    return PlainTextResponse(body, media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="{version_id}-files.csv"'})


class ExportRequest(BaseModel):
    kind: str          # "run" or "version"
    id: str
    destination: str = "local"
    bucket: str = ""
    prefix: str = ""
    format: str | None = None


class WeightsExportRequest(BaseModel):
    destination: str = "local"
    bucket: str = ""
    prefix: str = ""


@router.post("/runs/{run_id}/weights/export")
def export_run_weights(run_id: str, payload: WeightsExportRequest, session=Depends(require_session)) -> dict:
    return _guard(lambda: catalog_view.export_run_weights(
        run_id, destination=payload.destination, bucket=payload.bucket, prefix=payload.prefix))


@router.post("/exports")
def start_export(payload: ExportRequest, session=Depends(require_session)) -> dict:
    """Build a zip of a run or a dataset version, in the background."""
    return _guard(lambda: catalog_view.start_export(
        payload.kind, payload.id, destination=payload.destination,
        bucket=payload.bucket, prefix=payload.prefix, format=payload.format))


@router.get("/exports")
def list_exports(session=Depends(require_session)) -> dict:
    return {"exports": _guard(catalog_view.exports)}


@router.get("/exports/{export_id}/download")
def export_download(export_id: str, session=Depends(require_session)) -> dict:
    """A one-hour link to the zip."""
    return {"url": _guard(lambda: catalog_view.download_url(export_id))}


@router.post("/exports/{export_id}/stop")
def export_stop(export_id: str, session=Depends(require_session)) -> dict:
    return _guard(lambda: catalog_view.stop_export(export_id))
