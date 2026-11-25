"""Authenticated baseline registry and same-validation comparisons."""
from fastapi import APIRouter, Depends, HTTPException, File, Form, UploadFile, Path
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field

from ..auth import require_session
from ..training import comparison as c
from .. import data

router = APIRouter()


class BaselineRequest(BaseModel):
    source_arn: str
    label: str = Field(min_length=1, max_length=200)


class ComparisonRequest(BaseModel):
    selection: dict | None = None


@router.post('/baselines/upload')
def upload(file: UploadFile = File(...), label: str = Form(...), session=Depends(require_session)):
    try:
        return c.upload_baseline(file.file, file.filename or '', label)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get('/baselines')
def baselines(session=Depends(require_session)):
    return {'name': 'Baseline Model', 'group': c.group_name(), 'versions': c.baseline_versions()}


@router.post('/baselines')
def register(payload: BaselineRequest, session=Depends(require_session)):
    try:
        return c.register_baseline(payload.source_arn, payload.label)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.delete('/baselines/{version}')
def delete_baseline(version: int = Path(ge=1), session=Depends(require_session)):
    try:
        return c.delete_baseline(version)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except ClientError as exc:
        raise HTTPException(409, 'AWS could not delete this baseline version: ' + exc.response['Error'].get('Message', exc.response['Error']['Code'])) from exc


@router.get('/saved-selection/{version_id}')
def saved_selection(version_id: str, session=Depends(require_session)):
    try:
        return {'selection': data.saved_comparison_selection(version_id)}
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@router.get('')
def comparisons(session=Depends(require_session)):
    from ..data.catalog_view import run_numbers
    audits = c.t2.scan_training_audits(limit=200)
    try:
        numbers = run_numbers([a['id'] for a in audits])
    except Exception:
        import logging
        logging.getLogger(__name__).warning('Could not load catalog run numbers', exc_info=True)
        numbers = {}
    return {'runs': [{'id': a['id'], 'label': a.get('detector_slug') or a['id'],
                     'run_number': numbers.get(a['id']),
                     'submitted_at': a.get('submitted_at', ''), 'training_status': a.get('status'),
                     'status': c.status_for(a), 'comparison': a.get('comparison'),
                     'can_compare': a.get('status') == 'Succeeded'}
                    for a in audits]}


@router.post('/{run_id}/run')
def run(run_id: str, payload: ComparisonRequest, session=Depends(require_session)):
    try:
        result = c.request(run_id, payload.selection)
        if result.get('started') is False:
            raise HTTPException(409, 'The comparison worker is busy. Try again in a moment.')
        return result
    except ValueError as exc:
        raise HTTPException(422, str(exc))
