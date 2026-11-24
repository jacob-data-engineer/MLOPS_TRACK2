"""Read the exact, permanent validation recipe; never fall back to today's data."""
import hashlib
import json

from . import dataset_versions as dv


def saved_selection(version_id):
    version = dv.get(version_id)
    selected = (version.get('entry_template') or {}).get('comparison') or {'enabled': False}
    return {k: v for k, v in selected.items() if k in ('enabled', 'baseline_arn', 'other_arns')}


def validation_manifest(version_id):
    version = dv.get(version_id)
    if not version.get('immutable_files') or not version.get('central_recipe'):
        raise ValueError('This run has no preserved central-store validation snapshot.')
    members = [m for m in version.get('file_provenance', []) if m.get('split') == 'val']
    if not members or len(members) != int(version.get('val_images', 0)):
        raise ValueError('The saved validation image and label recipe is incomplete.')
    names = {int(k): str(v) for k, v in version.get('training_classes', {}).items()}
    if sorted(names) != list(range(len(names))) or not names or len(set(names.values())) != len(names):
        raise ValueError('The saved validation class mapping is ambiguous.')
    prefix = version['central_recipe']['store_uri'].rstrip('/') + '/objects/replay/'
    files = []
    for member in members:
        row = {}
        for kind in ('image', 'label'):
            uri, etag = member.get(f'central_{kind}', ''), member.get(f'central_{kind}_etag', '')
            if not uri.startswith(prefix) or not etag:
                raise ValueError('A validation file has no permanent versioned reference.')
            row[kind], row[kind + '_etag'] = uri, etag
        files.append(row)
    identity = {'names': names, 'files': files}
    return {**identity, 'dataset_version_id': version_id,
            'snapshot_id': hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()}
