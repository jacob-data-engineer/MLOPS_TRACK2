"""Export only members of one experiment with its immutable shared dataset."""
def context(ident, arn):
    from ..chain import training_variants as variants
    from . import catalog_view as cv
    job = variants.public(variants.read(ident))
    dataset = job.get('working_dataset_version_id') or job.get('dataset_version_id')
    if not dataset:
        raise ValueError('This experiment has no preserved dataset yet.')
    member = next((v for v in job['variants'] if v.get('model_package_arn') == arn), None)
    if arn == job['baseline_arn']:
        uri = (job.get('baseline') or {}).get('uri', '')
        label = 'baseline'
    elif member and member['state'] == 'Succeeded':
        rows = cv._rows('SELECT record FROM training_runs WHERE run_id=%s', (member['run_id'],))
        uri = (rows[0]['record'] if rows else {}).get('weights_uri', '')
        label = 'variant-'+str(member['index']+1)
    else:
        raise ValueError('Select the baseline or a successful model from this experiment.')
    return {'job': job, 'dataset': dataset, 'member': member, 'weights_uri': uri,
            'label': f"run-{job.get('run_number') or ident}-{label}", 'arn': arn}


def metadata(ident, arn):
    from . import catalog_metadata as cm
    c = context(ident, arn)
    result = cm.describe('version', c['dataset'])
    result['label'] = c['label']
    result['checkpoint_available'] = bool(c['weights_uri'])
    result['groups']['Experiment'] = cm.clean({
        'id': ident, 'run_number': c['job'].get('run_number'),
        'dataset_version_id': c['dataset'], 'selected_model': arn,
        'role': 'variant' if c['member'] else 'baseline',
        'variant': c['member'], 'comparison': c['job'].get('comparison'),
        'baseline_arn': c['job']['baseline_arn'],
        'notice': 'Dataset belongs to this experiment; it is not the baseline original training dataset.'})
    if c['member']:
        trained = cm.describe('run', c['member']['run_id'])
        result['groups']['Selected model training'] = trained['groups']
    return result
