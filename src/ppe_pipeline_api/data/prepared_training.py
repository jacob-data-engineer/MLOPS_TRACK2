"""Preserve the already-built decay dataset before it can enter a training queue."""
from __future__ import annotations

import copy
import datetime as dt
import json
import uuid
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import yaml

from .. import record_store, storage
from . import catalog_metadata, combined_store as cs, dataset_versions as dv, dataset_yaml as dy


def require_sql():
    if not record_store.enabled():
        raise ValueError("Repeatable training requires the PostgreSQL record backend.")
    record_store.verify_ready()


def _objects(bucket, prefix):
    return {o['Key']: o for page in storage.s3.get_paginator('list_objects_v2').paginate(
        Bucket=bucket, Prefix=prefix.rstrip('/') + '/') for o in page.get('Contents', [])}


def _put(bucket, key, body):
    storage.s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch='*')


def verify_members(version):
    from .catalog_view import _rows
    counts = {r['split']: int(r['files']) for r in _rows(
        'SELECT split, count(*) AS files FROM dataset_members WHERE version_id = %s AND included GROUP BY split',
        (version['version_id'],))}
    if counts.get('train', 0) != version['train_images'] or counts.get('val', 0) != version['val_images']:
        raise ValueError('The SQL dataset file list is incomplete; training has not been queued.')


@cs.decay_locked
def freeze(config: dict, *, trigger='manual', target_classes=None, progress=None) -> dict:
    """No combine, golden rebuild, retention selection or writes to the decay folder.

    Copies are version-specific and never reused by another preparation. Labels
    are normalized by the source YAML's explicit mapping, so exported datasets
    have the same class meanings as training, including mixed source taxonomies.
    """
    require_sql()
    bucket, store, decay = cs.location(config)
    marker_key = f'{decay}/{cs.MARKER}'
    marker_obj = storage.s3.get_object(Bucket=bucket, Key=marker_key)
    marker = json.loads(marker_obj['Body'].read())
    if marker.get('status') != 'built' or marker.get('failed'):
        raise ValueError('Complete Combine data sources in Data Preparation before training.')
    objects = _objects(bucket, decay)
    if progress:
        progress('Reading prepared dataset', 0, 0)
    splits = {'train': [], 'val': []}
    images = []
    for key, obj in sorted(objects.items()):
        if not cs._is_image(key) or '/images/' not in key:
            continue
        rel = key[len(decay) + 1:]
        side = 'val' if '/images/val/' in key else 'train'
        directory = key.rsplit('/images/', 1)[0] + '/images'
        tail = key[len(directory) + 1:]
        if tail.startswith(('train/', 'val/', 'test/')):
            if tail.startswith('test/'):
                continue
            directory += '/' + tail.split('/')[0]
        entry = {'images': directory, 'labels': ''}
        if entry not in splits[side]:
            splits[side].append(entry)
        images.append((key, rel, side, obj))
    if not splits['train'] or not splits['val']:
        raise ValueError('The prepared decayed dataset must contain training and validation images.')
    classes = (config.get('dataset_assemble') or {}).get('label_classes') or {}
    built = dy.build_in_place(splits=splits, classes=classes, bucket=bucket,
                              overwrite=True, label_scan=0, drop_undeclared=False)
    doc = yaml.safe_load(built['text'])
    prefixes = built['download_prefix'].split(',')
    local = dy.local_layout(prefixes)
    remaps = doc.get('x_label_remap') or {}
    names = {int(k): str(v) for k, v in doc['names'].items()}
    dataset_names = dict(names)
    selection = None
    if target_classes:
        names = {int(k): str(v) for k, v in target_classes.items()}
        missing = set(names.values()) - set(dataset_names.values())
        if missing or sorted(names) != list(range(len(names))):
            raise ValueError(f'Model classes do not match the prepared dataset: {sorted(missing)}.')
        by_name = {name: index for index, name in names.items()}
        selection = {i: by_name[name] for i, name in dataset_names.items() if name in by_name}
    folder = f'_training_inputs/{uuid.uuid4().hex}/dataset'
    progress_lock = threading.Lock()
    completed, last_report = 0, 0.0

    def preserve(item):
        nonlocal completed, last_report
        key, relative, side, obj = item
        target = f'{folder}/{relative}'
        storage.s3.copy_object(Bucket=bucket, Key=target,
            CopySource={'Bucket': bucket, 'Key': key}, CopySourceIfMatch=obj['ETag'])
        label = storage.label_path_for(key)
        label_obj = objects.get(label)
        destination_label = storage.label_path_for(target)
        provenance = {'image': f's3://{bucket}/{key}', 'image_etag': obj['ETag'],
                      'stored_image': f's3://{bucket}/{target}', 'split': side,
                      'label': f's3://{bucket}/{label}' if label_obj else '',
                      'stored_label': f's3://{bucket}/{destination_label}',
                      'label_etag': label_obj['ETag'] if label_obj else ''}
        text = ''
        if label_obj:
            body = storage.s3.get_object(Bucket=bucket, Key=label, IfMatch=label_obj['ETag'])
            text = body['Body'].read().decode('utf-8')
        # Find a source-root remap, including nested label directories.
        local_label = local(bucket, label or '').rstrip('/')
        matching = [(root, mapping) for root, mapping in remaps.items()
                    if local_label.startswith(root.rstrip('/') + '/')]
        mapping = max(matching, key=lambda x: len(x[0]))[1] if matching else {}
        normalized = []
        for line in text.splitlines():
            if not line.strip():
                continue
            parts = line.split()
            source_id = int(parts[0])
            target_id = int(mapping.get(source_id, mapping.get(str(source_id), source_id)))
            if target_id not in dataset_names:
                raise ValueError(f'{label}: class {source_id} has no declared dataset class.')
            if selection is not None:
                if target_id not in selection:
                    continue
                target_id = selection[target_id]
            normalized.append(' '.join([str(target_id), *parts[1:]]))
        _put(bucket, destination_label, ('\n'.join(normalized) + ('\n' if normalized else '')).encode())
        if progress:
            with progress_lock:
                completed += 1
                now = time.monotonic()
                if completed == len(images) or now - last_report >= 2:
                    progress('Preserving images and labels', completed, len(images))
                    last_report = now
        return provenance

    with ThreadPoolExecutor(max_workers=16) as pool:
        provenance = list(pool.map(preserve, images))
    # A concurrent rebuild changes this marker before changing any files.
    after = storage.s3.head_object(Bucket=bucket, Key=marker_key)
    if after['ETag'] != marker_obj['ETag']:
        raise ValueError('Data Preparation changed while preserving this dataset. Retry after it finishes.')
    for key, obj in objects.items():
        if key.endswith(('.yaml', '.yml')):
            if storage.s3.head_object(Bucket=bucket, Key=key)['ETag'] != obj['ETag']:
                raise ValueError('Source class metadata changed during preservation. Please retry.')
    frozen_doc = {'path': f's3://{bucket}/{folder}', 'nc': len(names), 'names': names}
    for side in splits:
        frozen_doc[side] = [entry['images'][len(decay) + 1:] for entry in splits[side]]
    frozen_text = yaml.safe_dump(frozen_doc, sort_keys=False)
    _put(bucket, f'{folder}/data.yaml', frozen_text.encode())
    if progress:
        progress('Saving dataset records to SQL', len(images), len(images))
    version = dv.create(home=bucket, yaml_text=frozen_text,
        prefixes=[f's3://{bucket}/{folder}/'], trigger=trigger,
        sources=[f's3://{bucket}/{decay}'], preserve_files=False)
    verify_members(version)
    preparation = {k: copy.deepcopy(config.get(k)) for k in
        ('training_data_preparation', 'dataset_assemble', 'central_store', 'retention', 'golden_set', 'auto_training')}
    dv.mark(version['version_id'], strict=True, immutable_files=True, prepared_decay=f's3://{bucket}/{decay}',
            preparation=catalog_metadata.clean(preparation), preparation_marker=marker,
            preparation_marker_etag=marker_obj['ETag'],
            file_provenance=provenance, source_yaml=built['text'], frozen_yaml=frozen_text,
            dataset_classes=dataset_names, training_classes=names)
    return dv.get(version['version_id'])


def record_submission(version_id: str, *, entry: dict, pipeline: dict, parameters: dict) -> dict:
    require_sql()
    # Returned to the mandatory SQL audit write before the GPU is started.
    version = dv.get(version_id)
    if not version.get('immutable_files'):
        raise ValueError('This run does not have a preserved training dataset.')
    verify_members(version)
    return {'dataset_version_id': version_id, 'entry': copy.deepcopy(entry),
            'pipeline': pipeline, 'pipeline_parameters': parameters,
            'captured_at': dt.datetime.now(dt.timezone.utc).isoformat()}


def preserve_base(uri: str) -> dict:
    require_sql()
    bucket, key = dv._uri(uri)
    obj = storage.s3.head_object(Bucket=bucket, Key=key)
    home = storage.BUCKET
    target = f'_training_inputs/{uuid.uuid4().hex}/base/{key.rsplit("/", 1)[-1]}'
    storage.s3.copy_object(Bucket=home, Key=target, CopySource={'Bucket': bucket, 'Key': key},
                          CopySourceIfMatch=obj['ETag'])
    return {'uri': f's3://{home}/{target}', 'key': target, 'source_uri': uri,
            'etag': obj['ETag'], 'source_version_id': obj.get('VersionId', '')}


def repeat_entry(run_id: str) -> dict:
    from . import catalog_view
    require_sql()
    rows = catalog_view._rows('SELECT record FROM training_runs WHERE run_id = %s', (run_id,))
    if not rows:
        raise ValueError('Training run not found.')
    replay = (rows[0]['record'] or {}).get('reproducibility') or {}
    template = replay.get('entry')
    if not template:
        raise ValueError('This older run has no complete replay recipe. Queue a new run from the prepared dataset first.')
    return copy.deepcopy(template)


@cs.decay_locked
def cleanup_finished():
    """Delete only the opted-in run's unchanged working copy, never its snapshot."""
    from botocore.exceptions import ClientError
    from .catalog_view import _rows
    if not record_store.enabled():
        return ''
    rows = _rows("SELECT DISTINCT v.id FROM training_runs r JOIN dataset_versions v ON v.id=r.version_id "
                 "WHERE r.status IN ('Succeeded','Failed','Stopped') AND r.execution_arn IS NOT NULL "
                 "AND r.record #>> '{reproducibility,entry,delete_decay_after_run}' = 'true' "
                 "AND NOT (v.record ? 'decay_cleanup') LIMIT 50")
    messages = []
    for row in rows:
        version = dv.get(row['id'])
        if not version.get('immutable_files'):
            continue
        bucket, store, decay = cs.location(version['preparation'])
        if version.get('prepared_decay') != f's3://{bucket}/{decay}' or decay == store:
            raise ValueError('The recorded decay location does not match its preparation settings.')
        marker_key = f'{decay}/{cs.MARKER}'
        try:
            current = storage.s3.get_object(Bucket=bucket, Key=marker_key)
        except ClientError as exc:
            if exc.response['Error']['Code'] not in ('NoSuchKey', '404', 'NotFound'):
                raise
            state = 'already absent'
        else:
            marker = json.loads(current['Body'].read())
            if (marker != version.get('preparation_marker') or marker.get('status') != 'built'
                    or (version.get('preparation_marker_etag') and current['ETag'] != version['preparation_marker_etag'])):
                state = 'skipped: preparation has changed'
            else:
                verify_members(version)
                keys = [k for k in _objects(bucket, decay) if k != marker_key]
                for start in range(0, len(keys), 1000):
                    result = storage.s3.delete_objects(Bucket=bucket, Delete={
                        'Objects': [{'Key': k} for k in keys[start:start + 1000]], 'Quiet': True})
                    if result.get('Errors'):
                        raise RuntimeError('Some decayed files could not be deleted; cleanup will retry.')
                storage.s3.delete_object(Bucket=bucket, Key=marker_key)
                state = f'deleted {len(keys)} working files'
        dv.mark(version['version_id'], strict=True, decay_cleanup={
            'state': state, 'at': dt.datetime.now(dt.timezone.utc).isoformat(),
            'uri': version['prepared_decay']})
        messages.append(f"{version['version_id']}: {state}")
    return '; '.join(messages)
