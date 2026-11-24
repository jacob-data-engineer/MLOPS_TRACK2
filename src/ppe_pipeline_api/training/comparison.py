"""Versioned baseline registry and durable, endpoint-free run comparisons."""
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from botocore.exceptions import ClientError, ConnectionError as BotoConnectionError, HTTPClientError

from .. import data, record_store
from . import track2_shared as t2


def group_name():
    # Outside production's approval/deployment event rule, isolated by environment.
    scope = os.getenv('CATALOG_SCHEMA', 'production').replace('_', '-')
    return f'yolo-baseline-{scope}-detector'


def baseline_versions():
    items, token = [], None
    while True:
        args = {'ModelPackageGroupName': group_name(), 'SortBy': 'CreationTime', 'SortOrder': 'Descending'}
        if token:
            args['NextToken'] = token
        try:
            page = t2.sm.list_model_packages(**args)
        except ClientError as exc:
            if exc.response['Error']['Code'] == 'ValidationException' and 'exist' in str(exc).lower():
                return []
            raise
        for p in page.get('ModelPackageSummaryList', []):
            items.append({'arn': p['ModelPackageArn'], 'version': p['ModelPackageVersion'],
                          'label': p.get('ModelPackageDescription', '')})
        token = page.get('NextToken')
        if not token:
            return items


def register_baseline(source_arn, label):
    label = str(label or '').strip()
    if not label or len(label) > 200:
        raise ValueError('Give the baseline a name of 1–200 characters.')
    _arn(source_arn)
    description = t2.sm.describe_model_package(ModelPackageName=source_arn)
    containers = description.get('InferenceSpecification', {}).get('Containers', [])
    if len(containers) != 1 or not containers[0].get('ModelDataUrl'):
        raise ValueError('Choose a registered single-model checkpoint.')
    # Preserve the artifact as well as its registry identity. Deleting the source
    # version later must not silently change this baseline.
    artifact = data.preserve_training_base(containers[0]['ModelDataUrl'])
    if not t2.ensure_model_package_group(group_name(), 'Baseline Model: comparison only, never auto-deployed.'):
        raise ValueError('The Baseline Model registry could not be created.')
    spec = dict(description['InferenceSpecification'])
    spec['Containers'] = [{**containers[0], 'ModelDataUrl': artifact['uri']}]
    created = t2.sm.create_model_package(ModelPackageGroupName=group_name(),
        ModelPackageDescription=label, InferenceSpecification=spec,
        ModelApprovalStatus='PendingManualApproval',
        CustomerMetadataProperties={'model_role': 'comparison-baseline', 'source_package': source_arn})
    return {'arn': created['ModelPackageArn']}


def _arn(value):
    if not isinstance(value, str) or not re.fullmatch(r'arn:aws:sagemaker:[a-z0-9-]+:\d{12}:model-package/[A-Za-z0-9-]+/\d+', value):
        raise ValueError('Select an exact registered model version.')
    return value


def normalize(value):
    if value is None or value == {}:
        return {'enabled': False}
    if not isinstance(value, dict) or not isinstance(value.get('enabled'), bool):
        raise ValueError('Comparison must be enabled or disabled.')
    if set(value) - {'enabled', 'baseline_arn', 'other_arns'}:
        raise ValueError('Unsupported comparison setting.')
    if not value['enabled']:
        return {'enabled': False}
    baseline = _arn(value.get('baseline_arn'))
    if baseline.split('/')[-2] != group_name():
        raise ValueError('Select a version from the Baseline Model registry.')
    others = value.get('other_arns', [])
    if not isinstance(others, list) or len(others) > 4:
        raise ValueError('Select at most four other trained models.')
    others = list(dict.fromkeys(_arn(a) for a in others if a != baseline))
    return {'enabled': True, 'baseline_arn': baseline, 'other_arns': others}


def pin_selection(value):
    selection = normalize(value)
    if not selection['enabled']:
        return selection
    models = []
    for arn in [selection['baseline_arn'], *selection['other_arns']]:
        uri = t2.publish_versioned_weights(arn)
        pinned = data.preserve_training_base(uri)
        bucket, key = pinned['uri'][5:].split('/', 1)
        etag = t2.s3.head_object(Bucket=bucket, Key=key)['ETag']
        models.append({'id': arn, 'uri': pinned['uri'], 'etag': etag,
                       'role': 'baseline' if arn == selection['baseline_arn'] else 'other',
                       'label': arn.split('model-package/')[-1]})
    return {**selection, 'models': models}


def selection_for(audit):
    return ((audit.get('reproducibility') or {}).get('entry') or {}).get('comparison') or {'enabled': False}


def status_for(audit):
    if audit.get('comparison'):
        return audit['comparison'].get('status', 'Pending')
    return 'Pending' if selection_for(audit).get('enabled') else 'No comparisons'


def _save(audit_id, comparison):
    t2.training_audit_table().update_item(Key={'id': audit_id},
        UpdateExpression='SET comparison = :c',
        ExpressionAttributeValues={':c': t2._decimalize(comparison)})


def _read(run_id):
    audit = t2.training_audit_table().get_item(Key={'id': run_id}).get('Item')
    if not audit or audit.get('record_type', 'training_audit') != 'training_audit':
        raise ValueError('Training run not found.')
    return audit


@record_store.exclusive('baseline_comparison')
def request(run_id, selection=None):
    data.require_training_sql()
    audit = _read(run_id)
    if audit.get('status') != 'Succeeded':
        raise ValueError('Only a successfully completed training run can be compared.')
    previous = audit.get('comparison') or {}
    if previous.get('status') in ('Pending', 'Running'):
        raise ValueError('This run already has a pending or running comparison.')
    pinned = pin_selection(selection) if selection is not None else previous.get('selection') or selection_for(audit)
    if not pinned.get('enabled'):
        raise ValueError('Select a baseline model first.')
    history = list(previous.get('history') or [])
    if previous:
        history.append({k: previous[k] for k in ('id', 'status', 'result_uri', 'manifest_uri', 'error') if k in previous})
    state = {'id': uuid.uuid4().hex, 'status': 'Pending', 'selection': pinned, 'history': history}
    _save(run_id, state)
    return state


def processing_spec(audit):
    pipeline = (audit.get('reproducibility') or {}).get('pipeline') or {}
    doc = json.loads(pipeline.get('definition') or '{}')
    train = next((s['Arguments'] for s in doc.get('Steps', []) if s.get('Type') == 'Training'), {})
    image = train.get('AlgorithmSpecification', {}).get('TrainingImage', '')
    role = train.get('RoleArn', '')
    if not isinstance(image, str) or '@sha256:' not in image or not isinstance(role, str) or not role:
        raise ValueError('This run has no pinned training image and execution role for comparison.')
    return image, role


def _start(audit, state):
    validation = data.validation_manifest(audit.get('dataset_version_id') or '')
    candidate = (audit.get('model') or {}).get('model_package_arn')
    if not candidate:
        raise ValueError('The completed run has no registered checkpoint.')
    image, role = processing_spec(audit)
    uri = t2.publish_versioned_weights(candidate)
    bucket, key = uri[5:].split('/', 1)
    candidate_model = {'id': candidate, 'uri': uri,
        'etag': t2.s3.head_object(Bucket=bucket, Key=key)['ETag'],
        'role': 'trained', 'label': 'Selected run: ' + candidate.split('model-package/')[-1]}
    settings = {'imgsz': int((audit.get('parameters') or {}).get('img_size') or 640),
                'ap_conf': 0.001, 'confidence': 0.25, 'nms_iou': 0.7, 'max_det': 300}
    manifest = {'validation': validation, 'models': [candidate_model, *state['selection']['models']],
                'settings': settings, 'image': image}
    prefix = f'comparisons/{os.getenv("CATALOG_SCHEMA", "production")}/{state["id"]}'
    manifest_key, script_key = prefix + '/input/manifest.json', prefix + '/code/comparison_runner.py'
    t2.s3.put_object(Bucket=t2.BUCKET, Key=manifest_key,
                     Body=json.dumps(record_store.native_numbers(manifest)).encode())
    t2.s3.put_object(Bucket=t2.BUCKET, Key=script_key,
                     Body=Path(__file__).with_name('comparison_runner.py').read_bytes())
    job = 'track2-compare-' + state['id']
    state.update(job_name=job, result_uri=f's3://{t2.BUCKET}/{prefix}/output/result.json',
                 manifest_uri=f's3://{t2.BUCKET}/{manifest_key}', image=image)
    # Persist BEFORE submitting. A restart uses the deterministic AWS name to
    # recover an accepted request instead of billing a second processing job.
    _save(audit['id'], state)
    inputs = []
    for name, key, path in [('manifest', manifest_key, 'input'), ('code', script_key, 'code')]:
        inputs.append({'InputName': name, 'S3Input': {'S3Uri': f's3://{t2.BUCKET}/{key}',
            'LocalPath': '/opt/ml/processing/' + path, 'S3DataType': 'S3Prefix', 'S3InputMode': 'File'}})
    t2.sm.create_processing_job(ProcessingJobName=job, RoleArn=role,
        AppSpecification={'ImageUri': image, 'ContainerEntrypoint': ['python3', '/opt/ml/processing/code/comparison_runner.py']},
        ProcessingResources={'ClusterConfig': {'InstanceCount': 1, 'InstanceType': 'ml.m5.xlarge', 'VolumeSizeInGB': 50}},
        StoppingCondition={'MaxRuntimeInSeconds': 21600}, ProcessingInputs=inputs,
        ProcessingOutputConfig={'Outputs': [{'OutputName': 'comparison',
            'S3Output': {'S3Uri': f's3://{t2.BUCKET}/{prefix}/output',
                         'LocalPath': '/opt/ml/processing/output', 'S3UploadMode': 'EndOfJob'}}]})
    state['status'] = 'Running'
    _save(audit['id'], state)


def advance(audit):
    state = record_store.native_numbers(audit.get('comparison') or {})
    if not state:
        selected = selection_for(audit)
        if not selected.get('enabled'):
            return
        if audit.get('status') not in t2.TERMINAL_EXECUTION_STATUSES:
            return
        state = {'id': uuid.uuid4().hex, 'status': 'Pending', 'selection': selected}
        _save(audit['id'], state)
    if state.get('status') not in ('Pending', 'Running'):
        return
    try:
        if audit.get('status') != 'Succeeded':
            raise ValueError('Training did not finish successfully; no checkpoint to compare.')
        job = None
        if state.get('job_name'):
            try:
                job = t2.sm.describe_processing_job(ProcessingJobName=state['job_name'])
            except ClientError as exc:
                if exc.response['Error']['Code'] != 'ValidationException' or 'exist' not in str(exc).lower():
                    raise
        if not job:
            _start(audit, state)
            return
        status = job['ProcessingJobStatus']
        if status == 'Completed':
            bucket, key = state['result_uri'][5:].split('/', 1)
            result = json.loads(t2.s3.get_object(Bucket=bucket, Key=key)['Body'].read())
            if result.get('status') not in ('Completed', 'Failed', 'Unavailable'):
                raise ValueError('The comparison report has an invalid status.')
            state.update(status=result['status'], result=result, error=result.get('error', ''),
                         ended_at=datetime.now(timezone.utc).isoformat())
        elif status in ('Failed', 'Stopped'):
            state.update(status='Failed', error=job.get('FailureReason') or f'Processing job {status.lower()}.')
        else:
            state['status'] = 'Running'
        _save(audit['id'], state)
    except ValueError as exc:
        state.update(status='Unavailable', error=str(exc))
        _save(audit['id'], state)
    except Exception as exc:
        # On ambiguous submission/network failures keep Pending so the next
        # tick describes the saved job name before trying to submit again.
        state['status'] = 'Failed'
        if isinstance(exc, (BotoConnectionError, HTTPClientError)):
            state['status'] = 'Pending' if not state.get('job_name') else 'Running'
        if isinstance(exc, ClientError):
            code = exc.response['Error']['Code']
            if code in ('NoSuchKey', '404', 'PreconditionFailed', '412'):
                state['status'] = 'Unavailable'
            elif code in ('ThrottlingException', 'Throttling', 'ServiceUnavailable', 'InternalFailure', 'ResourceInUse') or (
                    code == 'ValidationException' and state.get('job_name') and 'already exist' in str(exc).lower()):
                state['status'] = 'Pending' if not state.get('job_name') else 'Running'
        state['error'] = str(exc)[:1500]
        _save(audit['id'], state)


@record_store.exclusive('baseline_comparison')
def tick():
    if not record_store.enabled():
        return
    # Scan all recorded runs, including historical runs manually queued today.
    for audit in t2.scan_training_audits(limit=1_000_000):
        advance(audit)
