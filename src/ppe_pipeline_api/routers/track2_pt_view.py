"""Authenticated, read-only checkpoint inspection."""
import io
import tarfile
from fastapi import APIRouter, Depends, UploadFile, HTTPException
from pydantic import BaseModel
from ..auth import require_session
from ..data import pt_view
from ..training import track2_shared as t2

router = APIRouter()


@router.get('/registered')
def list_registered(session=Depends(require_session)):
    """List versions directly, without downloading metrics or model artifacts."""
    rows = []
    token = ''
    try:
        while True:
            kwargs = {'ModelPackageType': 'Versioned', 'MaxResults': 100,
                      'SortBy': 'CreationTime', 'SortOrder': 'Descending'}
            if token:
                kwargs['NextToken'] = token
            page = t2.sm.list_model_packages(**kwargs)
            for item in page.get('ModelPackageSummaryList', []):
                arn = item['ModelPackageArn']
                group, version = arn.rsplit('/', 2)[-2:]
                rows.append({'arn': arn, 'label': f'{group} · v{version}',
                             'status': item.get('ModelPackageStatus', '')})
            token = page.get('NextToken')
            if not token:
                return {'models': rows}
    except Exception as exc:
        raise HTTPException(502, f'Could not list registered models: {exc}') from exc


@router.post('/upload')
def upload(file: UploadFile, session=Depends(require_session)):
    try:
        return pt_view.inspect(file.file, file.filename or '')
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    finally:
        file.file.close()


class Registered(BaseModel):
    arn: str


@router.post('/registered')
def registered(payload: Registered, session=Depends(require_session)):
    try:
        return pt_view.inspect_registered(payload.arn)
    except Exception as exc:
        raise HTTPException(422, f'Could not inspect registered model: {exc}') from exc
