"""New runs from SQL-pinned data, independent of today's preparation settings."""
import uuid
from concurrent.futures import ThreadPoolExecutor

import yaml

from .. import storage
from . import central_recipe as cr, dataset_versions as dv
from .combined_store import decay_locked
from .prepared_training import require_sql, verify_members


def _classes(value):
    names = {int(k): str(v) for k, v in value.items()}
    if not names or sorted(names) != list(range(len(names))) or len(set(names.values())) != len(names):
        raise ValueError('The saved or selected class mapping is ambiguous.')
    return names


def check(version_id, target_classes):
    require_sql()
    version = dv.get(version_id)
    if not version.get('central_recipe'):
        raise ValueError('This older dataset has no permanent central-store recipe. Use newly prepared data to create a reusable SQL dataset.')
    source = _classes(version['training_classes'])
    target = _classes(target_classes)
    if set(source.values()) != set(target.values()):
        raise ValueError('Saved dataset and model must have the same class names. Subset training is not supported here.')
    cr.verify_central(version, check_cancel=True)
    verify_members(version)
    return {'version_id': version_id, 'created_at': version.get('created_at', ''),
            'run_id': version.get('run_id', ''), 'train_images': version['train_images'],
            'val_images': version['val_images'], 'classes': list(source.values()),
            'store_uri': version['central_recipe']['store_uri'], 'remapped': source != target}


@decay_locked
def prepare(version_id, *, target_classes, trigger='manual', progress=None):
    if progress:
        progress('Checking saved central-store files', 0, 0)
    check(version_id, target_classes)
    source = dv.get(version_id)
    names = _classes(target_classes)
    by_name = {name: index for index, name in names.items()}
    mapping = {index: by_name[name] for index, name in _classes(source['training_classes']).items()}
    bucket, store = cr._uri(source['central_recipe']['store_uri'])
    folder = f'_training_inputs/{uuid.uuid4().hex}/dataset'
    recipe = {'schema': 1, 'store_uri': source['central_recipe']['store_uri'], 'bucket': bucket, 'folder': folder}

    def one(item):
        from ..chain import training_control
        training_control.checkpoint_fast()
        index, member = item
        side = member['split']
        if side not in ('train', 'val'):
            raise ValueError('Invalid saved dataset split.')
        image_bucket, image_key = cr._uri(member['central_image'])
        image_key_out = f'{folder}/images/{side}/{index}.{image_key.rsplit(".", 1)[-1]}'
        label_key = f'{folder}/labels/{side}/{index}.txt'
        storage.s3.copy_object(Bucket=bucket, Key=image_key_out,
            CopySource={'Bucket': image_bucket, 'Key': image_key}, CopySourceIfMatch=member['central_image_etag'])
        label_bucket, label_source = cr._uri(member['central_label'])
        obj = storage.s3.get_object(Bucket=label_bucket, Key=label_source, IfMatch=member['central_label_etag'])
        lines = []
        for line in obj['Body'].read().decode('utf-8').splitlines():
            if line.strip():
                parts = line.split()
                if int(parts[0]) not in mapping:
                    raise ValueError('A saved label contains an undeclared class ID.')
                lines.append(' '.join([str(mapping[int(parts[0])]), *parts[1:]]))
        body = ('\n'.join(lines) + ('\n' if lines else '')).encode()
        storage.s3.put_object(Bucket=bucket, Key=label_key, Body=body)
        central_label, etag = cr.keep_label(bucket, store, body)
        return {**member, 'stored_image': f's3://{bucket}/{image_key_out}',
                'stored_label': f's3://{bucket}/{label_key}',
                'central_label': central_label, 'central_label_etag': etag}

    members = source['file_provenance']
    provenance = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for member in pool.map(one, enumerate(members)):
            provenance.append(member)
            if progress and (len(provenance) % 100 == 0 or len(provenance) == len(members)):
                progress('Rebuilding saved dataset', len(provenance), len(members))
    text = yaml.safe_dump({'path': f's3://{bucket}/{folder}', 'train': 'images/train',
                          'val': 'images/val', 'nc': len(names), 'names': names}, sort_keys=False)
    storage.s3.put_object(Bucket=bucket, Key=folder+'/data.yaml', Body=text.encode())
    if progress:
        progress('Saving dataset records to SQL', len(members), len(members))
    version = dv.create(home=bucket, yaml_text=text, prefixes=[f's3://{bucket}/{folder}/'],
                        trigger=trigger, sources=source.get('sources', []), preserve_files=False, central_recipe=recipe)
    verify_members(version)
    cr.record_members(version['version_id'], provenance)
    dv.mark(version['version_id'], strict=True, immutable_files=True,
            source_dataset_version_id=version_id, file_provenance=provenance,
            source_yaml=source['frozen_yaml'], frozen_yaml=text, dataset_classes=names,
            training_classes=names, central_recipe=recipe, temporary_cleaned=False)
    return dv.get(version['version_id'])
