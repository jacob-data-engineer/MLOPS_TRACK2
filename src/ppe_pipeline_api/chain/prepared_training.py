"""Queue the prepared decay dataset without changing the preparation."""
import datetime as dt
import uuid

from .. import data, pull_control as pc, training


def queue_prepared(config=None, *, trigger='manual', submit='enqueue', overrides=None):
    from . import retrain_schedule as rs
    config = config if config is not None else pc.read_config()
    plan = rs.settings(config)
    allowed = {'epochs', 'img_size', 'batch', 'seed', 'patience', 'hyperparams',
               'detector_slug', 'architecture', 'notes', 'base_version', 'delete_decay_after_run'}
    if overrides:
        if set(overrides) - allowed:
            raise ValueError('Unsupported training setting.')
        plan.update(overrides)
    if not isinstance(plan.get('delete_decay_after_run', False), bool):
        raise ValueError('Delete decayed dataset must be on or off.')
    if submit not in ('enqueue', 'queue', 'start'):
        raise ValueError('Choose enqueue, queue or start.')
    plan['hyperparams'] = training.normalize_hyperparams(plan.get('hyperparams'))
    for name in ('epochs', 'img_size', 'batch', 'seed', 'patience'):
        plan[name] = int(plan.get(name) or 0)
    if plan['epochs'] < 1 or plan['img_size'] < 32 or plan['batch'] == 0 or plan['batch'] < -1 or plan['patience'] < 0:
        raise ValueError('Invalid epochs, image size, batch size or patience.')
    data.require_training_sql()
    detector = training.get_detector(str(plan['detector_slug']))
    architecture = str(plan['architecture'])
    packages = training.model_registry_summary(training.detector_group_name(detector, architecture)).get('packages') or []
    pinned = str(plan.get('base_version') or '').strip()
    scratch = pinned.lower() == 'scratch'
    chosen = None
    if scratch:
        refusal = training.scratch_refusal(str(plan['detector_slug']), architecture)
        if refusal:
            raise ValueError(refusal)
    elif pinned:
        chosen = rs.find_pinned_version(packages, pinned)
        if not chosen:
            raise ValueError(f'The selected starting model {pinned} is not available.')
    else:
        chosen = next((p for p in packages if p.get('ModelApprovalStatus') == 'Approved'), None)
        chosen = chosen or (packages[0] if packages else None)
        if not chosen:
            raise ValueError('Select a starting model or Train from scratch before queueing.')
    source_arn = chosen['ModelPackageArn'] if chosen else ''
    uri, source = training.resolve_base_model_source(scratch=scratch, scratch_architecture=architecture,
        base_model_key='', base_model_folder='', base_model_package_arn=source_arn)
    base = data.preserve_training_base(uri)
    version = data.freeze_prepared_training(config, trigger=trigger, target_classes=detector['classes'])
    generation, size = training.split_architecture(architecture)
    entry = {**data.version_run_fields(version), 'id': uuid.uuid4().hex[:12],
        'generation': generation, 'size': size, 'architecture': architecture,
        'detector_slug': str(plan['detector_slug']), 'custom_classes': True,
        'base_model_key': base['key'], 'base_model_package_arn': '',
        'base_model_scratch': False, 'base_model_folder': '',
        'epochs': int(plan['epochs']), 'img_size': int(plan['img_size']),
        'batch': int(plan['batch']), 'seed': int(plan['seed']),
        'patience': int(plan.get('patience') or 0), 'hyperparams': plan.get('hyperparams') or {},
        'notes': str(plan.get('notes') or ''), 'queued_by': trigger,
        'added_at': dt.datetime.now(dt.timezone.utc).isoformat(),
        'reproducible': True, 'detector_definition': detector,
        'delete_decay_after_run': plan.get('delete_decay_after_run', False),
        'base_provenance': {**base, 'package_arn': source_arn, 'source': source},
        'drop_undeclared_classes': False}
    # The full recipe is committed to SQL before making the entry runnable.
    template = {k: v for k, v in entry.items() if k not in ('id', 'added_at')}
    data.mark_version(version['version_id'], strict=True, state='queued', entry_id=entry['id'], entry_template=template)
    training.queue_add(entry)
    if submit in ('queue', 'start'):
        training.queue_start()
    return {'ok': True, 'steps': [{'step': 'preserve', 'ok': True,
        'detail': f"Preserved {version['train_images']} training and {version['val_images']} validation images"},
        {'step': 'queue', 'ok': True, 'detail': f"Queued {entry['id']} using the prepared decay dataset"}],
        'dataset_version_id': version['version_id'], 'entry_id': entry['id']}


def repeat_run(run_id):
    template = data.repeat_training_entry(run_id)
    entry = {**template, 'id': uuid.uuid4().hex[:12], 'queued_by': f'repeat {run_id}',
             'added_at': dt.datetime.now(dt.timezone.utc).isoformat()}
    training.queue_add(entry)
    return {'entry_id': entry['id']}
