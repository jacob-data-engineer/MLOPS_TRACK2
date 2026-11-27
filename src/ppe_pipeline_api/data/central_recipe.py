"""Permanent central-store inputs and disposable, reproducible training folders."""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor

from botocore.exceptions import ClientError

from .. import storage
from .combined_store import decay_locked


def _head(bucket, key):
    try:
        return storage.s3.head_object(Bucket=bucket, Key=key)
    except ClientError as exc:
        if exc.response['Error']['Code'] in ('404', 'NoSuchKey', 'NotFound'):
            return None
        raise


def keep_image(bucket, store, source, etag):
    # Include source identity as ETags are not guaranteed to be content hashes.
    identity = hashlib.sha256(f'{bucket}/{source}\n{etag}'.encode()).hexdigest()
    extension = source.rsplit('.', 1)[-1].lower()
    key = f'{store}/objects/replay/{identity}.{extension}'
    head = _head(bucket, key)
    if head is None:
        storage.s3.copy_object(Bucket=bucket, Key=key,
            CopySource={'Bucket': bucket, 'Key': source}, CopySourceIfMatch=etag)
        head = storage.s3.head_object(Bucket=bucket, Key=key)
    return f's3://{bucket}/{key}', head['ETag']


def keep_label(bucket, store, body):
    key = f'{store}/objects/replay/{hashlib.sha256(body).hexdigest()}.txt'
    if _head(bucket, key) is None:
        try:
            storage.s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch='*')
        except ClientError as exc:
            if exc.response['Error']['Code'] not in ('PreconditionFailed', '412'):
                raise
    return f's3://{bucket}/{key}', storage.s3.head_object(Bucket=bucket, Key=key)['ETag']


def record_members(version_id, members):
    from . import catalog
    conn = catalog.connect()
    try:
        cur = conn.cursor()
        cur.executemany('UPDATE dataset_members SET stored_image_uri=%s, stored_label_uri=%s, image_etag=%s '
                       'WHERE version_id=%s AND image_uri=%s',
            [(m['central_image'], m['central_label'], m['central_image_etag'], version_id, m['stored_image']) for m in members])
        if cur.rowcount != len(members):
            raise ValueError('The central-store SQL file references are incomplete.')
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _target(recipe):
    bucket, folder = recipe['bucket'], recipe['folder']
    if not re.fullmatch(r'_training_inputs/[0-9a-f]{32}/dataset', folder):
        raise ValueError('Invalid temporary training dataset location.')
    return bucket, folder


def _uri(uri):
    if not uri.startswith('s3://'):
        raise ValueError('Invalid recorded central-store reference.')
    return uri[5:].split('/', 1)


def verify_central(version, *, check_cancel=False):
    """Never discard the temporary copy if its permanent inputs were lost."""
    recipe = version['central_recipe']
    bucket, folder = _target(recipe)
    prefix = recipe['store_uri'].rstrip('/') + '/objects/replay/'
    members = version['file_provenance']
    if len(members) != version['train_images'] + version['val_images']:
        raise ValueError('The saved central-store file recipe is incomplete.')
    def one(member):
        from ..chain import training_control
        if check_cancel: training_control.checkpoint_fast()
        for kind in ('image', 'label'):
            uri = member[f'central_{kind}']
            if not uri.startswith(prefix) or uri.startswith(f's3://{bucket}/{folder}/'):
                raise ValueError('A permanent file is outside its recorded central store.')
            source_bucket, source_key = _uri(uri)
            head = _head(source_bucket, source_key)
            if not head or head['ETag'] != member[f'central_{kind}_etag']:
                raise ValueError(f'The preserved {kind} is missing or changed: {uri}')
    with ThreadPoolExecutor(max_workers=16) as pool:
        # Do not submit the entire dataset: executor shutdown would wait for
        # thousands of HEADs even after the first missing file was found.
        for start in range(0, len(members), 16):
            list(pool.map(one, members[start:start + 16]))


@decay_locked
def restore(version):
    """Use only the saved recipe, never the currently selected central store."""
    recipe = version.get('central_recipe')
    if not recipe:
        return
    bucket, folder = _target(recipe)
    members = version['file_provenance']
    if len(members) != version['train_images'] + version['val_images']:
        raise ValueError('The saved central-store file recipe is incomplete.')
    verify_central(version)
    from . import dataset_versions as dv
    dv.mark(version['version_id'], strict=True, temporary_cleaned=False)

    def one(member):
        from ..chain import training_control
        training_control.checkpoint_fast()
        for kind in ('image', 'label'):
            source_bucket, source_key = _uri(member[f'central_{kind}'])
            expected = member[f'central_{kind}_etag']
            head = _head(source_bucket, source_key)
            if not head or head['ETag'] != expected:
                raise ValueError(f'The preserved {kind} is missing or changed: {member[f"central_{kind}"]}')
            target_bucket, target_key = _uri(member[f'stored_{kind}'])
            if target_bucket != bucket or not target_key.startswith(folder + '/') or '..' in target_key.split('/'):
                raise ValueError('A saved temporary file is outside its dataset folder.')
            present = _head(bucket, target_key)
            if present and present['ETag'] == expected:
                continue
            storage.s3.copy_object(Bucket=bucket, Key=target_key,
                CopySource={'Bucket': source_bucket, 'Key': source_key}, CopySourceIfMatch=expected)
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, members))
    storage.s3.put_object(Bucket=bucket, Key=folder+'/data.yaml', Body=version['frozen_yaml'].encode())


def prepare_entry(entry):
    from . import dataset_versions as dv
    version = dv.get(entry['dataset_version_id'])
    if version.get('central_recipe'):
        restore(version)
    return entry


def cleanup_finished():
    """Called only once no execution or queued run can still read a dataset."""
    from . import catalog_view, dataset_versions as dv
    from .prepared_training import _objects
    rows = catalog_view._rows("SELECT DISTINCT v.id FROM training_runs r JOIN dataset_versions v ON v.id=r.version_id "
        "WHERE r.status IN ('Succeeded','Failed','Stopped') AND r.execution_arn IS NOT NULL "
        "AND v.record ? 'central_recipe' AND COALESCE(v.record->>'temporary_cleaned','false') <> 'true'")
    cleaned = []
    for row in rows:
        version = dv.get(row['id'])
        owner = version.get('experiment_owner')
        if owner:
            from ..chain import training_variants
            experiment = training_variants.read(owner)
            if experiment['state'] not in ('Completed', 'Completed with failures', 'Failed', 'Cancelled', 'Comparison failed'):
                continue
        recipe = version.get('central_recipe')
        if not recipe:
            continue
        bucket, folder = _target(recipe)
        keys = list(_objects(bucket, folder))
        if keys:
            verify_central(version)
        for offset in range(0, len(keys), 1000):
            result = storage.s3.delete_objects(Bucket=bucket, Delete={
                'Objects': [{'Key': key} for key in keys[offset:offset+1000]], 'Quiet': True})
            if result.get('Errors'):
                raise RuntimeError('Temporary dataset cleanup was incomplete; it will retry.')
        if keys:
            cleaned.append(f"{row['id']}: removed {len(keys)} temporary files; central-store files retained")
        dv.mark(row['id'], strict=True, temporary_cleaned=True)
    return '; '.join(cleaned)
