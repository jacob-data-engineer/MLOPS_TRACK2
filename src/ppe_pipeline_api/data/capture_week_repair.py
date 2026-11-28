"""Explicit, auditable regrouping of SQL-exported pairs after a cutoff change.

Run only with pull/training workers quiescent. Source exports and preserved
training datasets are never deleted. Old weekly copies are backed up before
removal; the caller persists the plan and per-pair results for recovery.
"""
from pathlib import PurePosixPath

from .. import pull_control as pc


def plan(images, *, source_prefix, root, config, existing):
    """Use the SQL export's stable names/splits, not upload timestamps."""
    from . import sql_explorer as sx
    source_prefix, root = source_prefix.strip('/'), root.strip('/')
    if not source_prefix or not root or source_prefix == root:
        raise ValueError('Distinct nonempty source and weekly root required')
    rows, undated = [], []
    for image, side, name in sx.copy_plan(images, config.get('train_percent', 80)):
        moment = sx.capture_moment({'captured_at': image.get('captured_at')}, image['key'])
        if moment is None:
            undated.append(image['uri'])
            continue
        week = pc.week_at(moment, config)
        image_key, label_key = sx.target_keys(source_prefix, side, name, '')
        targets = [f"{root}/{week['name']}/images/{side}/{name}",
                   f"{root}/{week['name']}/labels/{side}/{PurePosixPath(name).stem}.txt"]
        old = []
        for target in targets:
            suffix = '/'.join(target.split('/')[-3:])
            matches = [key for key in existing if key.startswith(root + '/')
                       and key.endswith('/' + suffix)
                       and key[len(root)+1:].split('/')[0].startswith(config.get('week_prefix') or 'week_')]
            old.append([key for key in matches if key != target])
        rows.append(dict(source=[image_key, label_key], target=targets, old=old,
                         captured_at=moment.isoformat(), week=week))
    return rows, undated


def relocate_pair(s3, bucket, row, backup_prefix):
    """Copy and verify the entire pair before removing either old object."""
    from . import sql_explorer as sx
    if not backup_prefix.strip('/') or backup_prefix.startswith(row['target'][0].split('/')[0] + '/'):
        raise ValueError('Backup must be outside weekly root')
    source_heads = [s3.head_object(Bucket=bucket, Key=k) for k in row['source']]
    # Preserve unrelated metadata and HTTP attributes while stamping both files.
    for key in row['source']:
        sx._stamp_existing_capture(bucket, key, row['captured_at'])
    old_heads = {}
    for keys in row['old']:
        for key in keys:
            h = s3.head_object(Bucket=bucket, Key=key)
            backup = backup_prefix.rstrip('/') + '/' + key
            s3.copy_object(Bucket=bucket, Key=backup,
                           CopySource={'Bucket': bucket, 'Key': key}, CopySourceIfMatch=h['ETag'])
            check = s3.head_object(Bucket=bucket, Key=backup)
            if (check['ETag'], check['ContentLength']) != (h['ETag'], h['ContentLength']):
                raise ValueError('Backup verification failed: ' + key)
            old_heads[key] = h
    for source, target, h in zip(row['source'], row['target'], source_heads):
        s3.copy_object(Bucket=bucket, Key=target,
                       CopySource={'Bucket': bucket, 'Key': source}, CopySourceIfMatch=h['ETag'])
        check = s3.head_object(Bucket=bucket, Key=target)
        if ((check['ETag'], check['ContentLength']) != (h['ETag'], h['ContentLength'])
                or check.get('Metadata', {}).get('captured_at') != row['captured_at']):
            raise ValueError('Destination verification failed: ' + target)
    # Both verified before either is removed. A retry is safe after partial copy.
    for key, h in old_heads.items():
        current = s3.head_object(Bucket=bucket, Key=key)
        if current['ETag'] != h['ETag'] or current['LastModified'] != h['LastModified']:
            raise ValueError('Old file changed during repair: ' + key)
    for key, h in old_heads.items():
        s3.delete_object(Bucket=bucket, Key=key, IfMatch=h['ETag'])
    return {'week': row['week']['name'], 'removed': len(old_heads), 'verified': 2}
