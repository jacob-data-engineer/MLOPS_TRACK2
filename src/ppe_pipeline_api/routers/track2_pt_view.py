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
        package = t2.sm.describe_model_package(ModelPackageName=payload.arn)
        uri = package['InferenceSpecification']['Containers'][0]['ModelDataUrl']
        if not uri.startswith('s3://'):
            raise ValueError('This registered model has no supported S3 artifact.')
        bucket, key = uri[5:].split('/', 1)
        obj = t2.s3.get_object(Bucket=bucket, Key=key)
        body = obj['Body']
        try:
            if obj.get('ContentLength', 0) > pt_view.MAX_FILE:
                raise ValueError('Model artifact exceeds the 512 MB inspection limit.')
            content = body.read(pt_view.MAX_FILE + 1)
        finally:
            body.close()
        if len(content) > pt_view.MAX_FILE:
            raise ValueError('Model artifact exceeds the 512 MB inspection limit.')
        if key.lower().endswith('.pt'):
            result = pt_view.inspect(io.BytesIO(content), key.rsplit('/', 1)[-1])
        else:
            with tarfile.open(fileobj=io.BytesIO(content), mode='r:*') as archive:
                members = [m for m in archive if m.isfile() and m.name.lower().endswith('.pt')]
                preferred = [m for m in members if m.name.rsplit('/', 1)[-1] == 'best.pt']
                member = preferred[0] if len(preferred) == 1 else members[0] if len(members) == 1 else None
                if member is None or member.size > pt_view.MAX_FILE:
                    raise ValueError('No unique checkpoint found, or checkpoint exceeds 512 MB.')
                with archive.extractfile(member) as stream:
                    result = pt_view.inspect(io.BytesIO(stream.read(pt_view.MAX_FILE + 1)), member.name)
        result['registry'] = {'arn': payload.arn, 'artifact': uri, 'status': package.get('ModelApprovalStatus'),
                              'metadata': package.get('CustomerMetadataProperties', {})}
        return result
    except Exception as exc:
        raise HTTPException(422, f'Could not inspect registered model: {exc}') from exc
