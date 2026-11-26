"""Explicit source class mappings, applied only to rebuilt combined training copies."""
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor


def source_key(path, bucket):
    path = str(path or '').strip().strip('/')
    if path.startswith('s3://'):
        bucket, _, path = path[5:].partition('/')
    root = re.split(r'/(?:images|labels)(?:/|$)', path, maxsplit=1)[0]
    return f's3://{bucket}/{root}'.rstrip('/')


def validate(records, names):
    if not isinstance(records, dict):
        raise ValueError('Class mappings must be an object.')
    names = {str(k): str(v) for k, v in names.items()}
    for source, record in records.items():
        if not source.startswith('s3://') or not isinstance(record, dict):
            raise ValueError('Invalid source class mapping.')
        if record.get('target_classes') != names:
            raise ValueError('Dataset classes changed. Review and save the source mappings again.')
        ids = record.get('ids')
        if not isinstance(ids, dict) or not ids:
            raise ValueError(f'{source}: map at least one source ID.')
        for key, dest in ids.items():
            if not str(key).isdigit() or (dest is not None and str(dest) not in names):
                raise ValueError(f'{source}: invalid class ID or target {key} -> {dest}.')


def rewrite(text, ids):
    out = []
    counts = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        try:
            values = [float(v) for v in parts]
            ident = values[0]
            if len(values) < 5 or not all(math.isfinite(v) for v in values) or ident < 0 or not ident.is_integer():
                raise ValueError()
        except (ValueError, IndexError, OverflowError):
            raise ValueError('Malformed label line; class mapping stopped.')
        key = str(int(ident))
        if key not in ids or ids[key] is None:
            raise ValueError(f'Class ID {key} is used in labels but is unmapped or marked unused.')
        out.append(' '.join([str(ids[key]), *parts[1:]]))
        counts[key] = counts.get(key, 0) + 1
    return ('\n'.join(out) + ('\n' if out else '')).encode(), counts


def preparation_maps(config, decayed, train, val):
    from .combined_store import slug_of
    records = config.get('source_class_mappings') or {}
    if not records:
        return {}
    names = (config.get('dataset_assemble') or {}).get('label_classes') or {}
    validate(records, names)
    targets = {}
    roots = {entry['images'].split('/images', 1)[0] for entry in train + val}
    def add(path, bucket, destination):
        record = records.get(source_key(path, bucket))
        if record and destination in roots:
            targets[destination] = record
    a = config.get('dataset_assemble') or {}
    entries = (a.get('train_sources') or []) + (a.get('eval_sources') or [])
    if not entries:
        f = a.get('folders') or {}
        entries = [{'images': f.get('train_images')}, {'images': f.get('eval_images')}]
    for entry in entries:
        path = entry.get('images')
        if path:
            add(path, config.get('source_bucket') or 'track2-s3', f'{decayed}/folders/{slug_of(path)}')
    g = config.get('golden_set') or {}
    add(g.get('source'), g.get('bucket') or 'track2-s3', f'{decayed}/golden')
    week_roots = [root for root in roots if root.startswith(f'{decayed}/weeks/')]
    if week_roots:
        keys = {source_key(p, config.get('source_bucket') or 'track2-s3')
                for paths in (config.get('picked') or {}).values() for p in paths if p}
        selected = [records.get(key) for key in keys]
        if any(selected):
            if not all(selected) or len({json.dumps(r['ids'], sort_keys=True) for r in selected}) != 1:
                raise ValueError('Pulled sources share week folders. Save the same class-ID mapping for every pulled source before processing.')
            for root in week_roots:
                targets[root] = selected[0]
    return targets


def apply(config, bucket, decayed, pairs, train, val, *, s3, checkpoint, progress):
    targets = preparation_maps(config, decayed, train, val)
    if not targets:
        return
    labels = [(src_bucket, src, dest, root, record)
              for src_bucket, src, dest in pairs for root, record in targets.items()
              if dest.startswith(root + '/labels/') and dest.lower().endswith('.txt')]
    def prepare(item):
        checkpoint()
        src_bucket, src, dest, root, record = item
        body = s3.get_object(Bucket=src_bucket, Key=src)['Body'].read().decode('utf-8')
        try:
            rewritten, counts = rewrite(body, record['ids'])
        except ValueError as exc:
            raise ValueError(f'{src}: {exc}') from exc
        return dest, rewritten, root, counts
    # Validate all used IDs before writing any rewritten labels or metadata.
    with ThreadPoolExecutor(max_workers=8) as pool:
        prepared = list(pool.map(prepare, labels))
    for i, (dest, body, _, _) in enumerate(prepared):
        checkpoint()
        s3.put_object(Bucket=bucket, Key=dest, Body=body, ContentType='text/plain')
        progress('class_mapping', i + 1, len(prepared))
    for root, record in targets.items():
        checkpoint()
        metadata = {'names': record['target_classes'], 'nc': len(record['target_classes'])}
        # JSON is valid YAML. Both metadata files now describe the rewritten copy.
        for name in ('data.yaml', 'data.source.yaml'):
            s3.put_object(Bucket=bucket, Key=f'{root}/{name}', Body=json.dumps(metadata).encode(), ContentType='application/yaml')
        s3.put_object(Bucket=bucket, Key=f'{root}/class_mapping.json',
                      Body=json.dumps(record, sort_keys=True).encode(), ContentType='application/json')
