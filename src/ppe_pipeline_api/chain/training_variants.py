"""Durable variant batches: fixed inputs, atomic queue publication, shared evaluation."""
import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
import uuid
import logging
from botocore.exceptions import ClientError
from .. import data, pull_control as pc, record_store
from ..training import track2_shared as t2, comparison as cmp
from ..data import catalog_view, dataset_versions
from . import training_control as control

KIND = 'training_variant_run'
DRAFT_KIND = 'training_variant_draft'

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def table():
    return record_store.Table(pc.TABLE_NAME)

def read(ident):
    item = table().get_item(Key={'kind': KIND, 'sort': ident}).get('Item')
    if not item:
        raise ValueError('Variant batch not found.')
    return record_store.native_numbers(item)

def save(item):
    item['updated_at'] = now()
    table().put_item(Item=item)

def all_records(kind=KIND):
    rows=[]
    kwargs={'KeyConditionExpression':'kind = :kind','ExpressionAttributeValues':{':kind':kind},'ScanIndexForward':False}
    while True:
        page=table().query(**kwargs);rows.extend(page['Items'])
        if not page.get('LastEvaluatedKey'):
            return sorted(record_store.native_numbers(rows),key=lambda r:r['created_at'],reverse=True)
        kwargs['ExclusiveStartKey']=page['LastEvaluatedKey']

def save_draft(payload):
    """Save configurations only. No queue, S3 preparation or AWS job calls."""
    data.require_training_sql()
    context={k:payload[k] for k in ('detector_slug','architecture','baseline_arn','dataset_version_id')}
    mode=payload.get('data_mode','saved')
    if mode not in ('saved','prepared'):
        raise ValueError('Choose saved or freshly prepared data.')
    if mode=='prepared':
        context['data_mode']=mode
        context['dataset_version_id']=''
    if not all(context[k] for k in ('detector_slug','architecture','baseline_arn')) or (mode=='saved' and not context['dataset_version_id']):
        raise ValueError('Choose a model, registered baseline and saved dataset first.')
    cmp._arn(context['baseline_arn'])
    ident=hashlib.sha256(json.dumps(context,sort_keys=True).encode()).hexdigest()
    variants=[]; names=set(); ids=set()
    for row in payload['variants']:
        name=row['name'].strip()
        if not name or name.casefold() in names or row['id'] in ids:
            raise ValueError('Each variant needs a unique name and ID.')
        names.add(name.casefold());ids.add(row['id'])
        variants.append({**row,'name':name,'settings':settings(row['settings'])})
    key={'kind':DRAFT_KIND,'sort':ident}
    tab=table()
    with contextlib.closing(record_store.connection()) as conn:
        cur=conn.cursor();tab._lock(cur,tab.key(key))
        previous=tab._read(cur,tab.key(key))
        revision=int(previous.get('revision',0))
        if revision!=payload['revision']:
            raise ValueError('These variants changed in another window. Refresh saved variants before saving again.')
        draft={**key,**context,'batch_name':str(payload.get('batch_name','')).strip(),'id':ident,'variants':variants,'revision':revision+1,
               'created_at':previous.get('created_at',now()),'updated_at':now()}
        tab._save(cur,draft);conn.commit()
    return draft

def settings(raw):
    allowed = {'epochs','img_size','batch','seed','patience','hyperparams','notes'}
    if set(raw)-allowed:
        raise ValueError('Unsupported training parameter.')
    out = {'epochs':20,'img_size':640,'batch':16,'seed':42,'patience':0, **raw}
    for k in ('epochs','img_size','batch','seed','patience'):
        if isinstance(out[k], bool) or not isinstance(out[k], (int, float)) or not float(out[k]).is_integer():
            raise ValueError(f'{k} must be an integer.')
        out[k] = int(out[k])
    if not 1 <= out['epochs'] <= 10000 or not 32 <= out['img_size'] <= 4096 or out['batch'] == 0 or out['batch'] < -1 or out['patience'] < 0 or not 0 <= out['seed'] <= 2**32-1:
        raise ValueError('Invalid epochs, image size, batch or patience.')
    if 'notes' in out and not isinstance(out['notes'], str):
        raise ValueError('Training notes must be text.')
    out['hyperparams'] = t2.normalize_hyperparams(out.get('hyperparams') or {})
    return out

def create(payload, *, preparation_config=None, trigger="manual"):
    data.require_training_sql()
    control.checkpoint()
    ident = str(uuid.UUID(payload['id']))
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    existing = table().get_item(Key={'kind':KIND,'sort':ident}).get('Item')
    if existing:
        if existing['request_digest'] != digest:
            raise ValueError('This submission ID belongs to a different variant batch.')
        return public(record_store.native_numbers(existing))
    detector = t2.get_detector(payload['detector_slug'])
    generation, size = t2.split_architecture(payload['architecture'])
    if not generation or not size:
        raise ValueError('Choose a valid architecture.')
    # Validate the exact baseline version before storing a runnable experiment.
    baseline = cmp._arn(payload['baseline_arn'])
    # A registered seed can belong to another model name (including production).
    # Match the architecture, not the destination registry name.
    if baseline.split('/')[-2].split('-')[0] != f'yolo{generation}{size}':
        raise ValueError('Select a baseline with the chosen model generation and size.')
    package = t2.sm.describe_model_package(ModelPackageName=baseline)
    if package.get('ModelPackageStatus') != 'Completed':
        raise ValueError('The baseline model is not ready.')
    mode=payload.get('data_mode','saved')
    if mode not in ('saved','prepared'):
        raise ValueError('Choose saved or freshly prepared data.')
    check = data.check_saved_training(payload['dataset_version_id'], detector['classes']) if mode=='saved' else {}
    config=copy.deepcopy(preparation_config if preparation_config is not None else pc.read_config()) if mode=='prepared' else None
    shared = settings(payload['shared'])
    variants=[]
    for index, row in enumerate(payload['variants']):
        overrides = row.get('overrides') or {}
        effective = settings({**shared, **overrides, 'hyperparams': {
            **shared['hyperparams'], **(overrides.get('hyperparams') or {})}})
        variants.append({'index':index,'name':row['name'], 'overrides':overrides,
                         'settings':effective,'state':'Pending','attempt':0,'history':[]})
    job = {'kind':KIND,'sort':ident,'id':ident,'name':payload['name'], 'request_digest':digest,
           'created_at':now(),'updated_at':now(),'state':'Pending','detector':detector,
           'run_number':catalog_view.reserve_experiment_number(ident),
           'architecture':payload['architecture'],'baseline_arn':baseline,
           'data_mode':mode,'preparation_config':config,'trigger':trigger,
           'dataset_version_id':payload['dataset_version_id'] if mode=='saved' else '', 'dataset':check,'shared':shared,
           'evaluation':payload['evaluation'],'variants':variants,'comparison_attempt':0}
    try:
        table().put_item(Item=job,ConditionExpression='attribute_not_exists(#k)',ExpressionAttributeNames={'#k':'kind'})
    except ClientError as exc:
        if exc.response['Error']['Code'] != 'ConditionalCheckFailedException':
            raise
        existing = read(ident)
        if existing['request_digest'] != digest:
            raise ValueError('This submission ID belongs to another variant batch.')
        job=existing
    return public(job)

def public(job):
    value=copy.deepcopy(job)
    value.pop('request_digest',None)
    ids=[v['run_id'] for v in value['variants'] if v.get('run_id')]
    numbers=catalog_view.run_numbers(ids)
    value['run_number']=value.get('run_number') or max(numbers.values(),default=None)
    for v in value['variants']:
        v['training_job_number']=numbers.get(v.get('run_id'))
        v['run_number']=value['run_number']
    variants_by_arn={v['model_package_arn']:v for v in value['variants'] if v.get('model_package_arn')}
    for row in ((value.get('comparison') or {}).get('result') or {}).get('rows',[]):
        variant=variants_by_arn.get(row.get('id'))
        if variant:
            row['label']=variant['name'] + (f" · Run {variant['run_number']}" if variant.get('run_number') else '')
    return value

def entry_id(job, variant):
    return uuid.uuid5(uuid.NAMESPACE_URL,f"experiment:{job['id']}:{variant['index']}:{variant['attempt']}").hex

def pin(arn, role, label):
    base=data.preserve_training_base(t2.publish_versioned_weights(arn))
    bucket,key=base['uri'][5:].split('/',1)
    return {**base,'id':arn,'etag':t2.s3.head_object(Bucket=bucket,Key=key)['ETag'], 'role':role,'label':label}

def publish(job, variant, entry):
    """Queue switch, entry and experiment state commit together, or not at all."""
    control.checkpoint(job.get('_stop_generation'))
    queue=record_store.Table(os.getenv('TRAINING_AUDIT_TABLE','track2-yolo-training-audit'))
    variant.update(state='Queued',run_id='run-'+entry['id'],entry_id=entry['id'],dataset_version_id=entry['dataset_version_id'])
    job.update(state='Training',stage=f"Queued {variant['name']}",updated_at=now())
    with contextlib.closing(record_store.connection()) as conn:
        cur=conn.cursor()
        for tab,row in ((queue,{**entry,'record_type':'queue_entry'}),
                        (queue,{'id':'queue-state','record_type':'queue_state','started':True,'at':now()}),
                        (table(),job)):
            tab._lock(cur,tab.key(row));tab._save(cur,row)
        conn.commit()

@record_store.exclusive('prepared_training_jobs')
def prepare_variant(job, variant):
    job['_stop_generation']=control.checkpoint()
    snapshot=t2.queue_snapshot()
    if snapshot.get('pending') or snapshot.get('running'):
        job['stage']='Waiting for the training queue';save(job);return
    variant['state']='Preparing';job['state']='Preparing';save(job)
    def progress(stage, completed, total):
        control.checkpoint(job['_stop_generation'])
        job['stage']=f"{variant['name']}: {stage}" + (f' ({completed}/{total})' if total else '')
        save(job)
    if job.get('trigger')=='schedule' and not job.get('pull_done'):
        config=job['preparation_config']
        if pc.pull_in_use(config):
            active=pc.active_run()
            if active and active.get('status')=='running' and active.get('run_id')!=job.get('pull_run_id'):
                progress('Waiting for current data pull to finish',0,0);return
            progress('Pulling fresh data for scheduled experiment',0,0)
            if not job.get('pull_run_id'):
                job['pull_run_id']=pc.new_run_id();save(job)
            result=pc.route_pending(config=config,trigger='training_schedule',run_id=job['pull_run_id'])
            if result.get('errors') or result.get('aborted'):
                raise ValueError('Scheduled experiment data pull failed; no variant was queued.')
            if result.get('limit_reached'):
                progress('Continuing scheduled experiment data pull',0,0);return
        job['pull_done']=True;save(job)
    if job.get('data_mode')=='prepared' and not job.get('dataset_version_id'):
        from . import retrain_schedule as rs
        progress('Preparing shared fresh dataset for all variants',0,0)
        rs.combine_and_decay(job['preparation_config'],for_run=True,progress=progress)
        shared=data.freeze_prepared_training(job['preparation_config'],trigger='variant_batch',
            target_classes=job['detector']['classes'],progress=progress)
        job['dataset_version_id']=shared['version_id']
        job['dataset']=shared
        job['working_dataset_version_id']=shared['version_id']
        save(job)  # Persist shared SQL recipe before any variant is queued.
    working_id=job.get('working_dataset_version_id')
    if working_id:
        version=dataset_versions.get(working_id)
        progress('Reusing shared training input files',0,0)
    else:
        version=data.prepare_saved_training(job['dataset_version_id'],target_classes=job['detector']['classes'],trigger='experiment',progress=progress)
        job['working_dataset_version_id']=version['version_id']
        save(job)  # Survives worker restart before queue publication.
    # Keep the working folder between variants and through final comparison.
    # Queue preparation still verifies/restores missing files from the pinned recipe.
    data.mark_version(version['version_id'],strict=True,experiment_owner=job['id'])
    generation,size=t2.split_architecture(job['architecture'])
    entry={**data.version_run_fields(version),**variant['settings'], 'id':entry_id(job,variant),
           'generation':generation,'size':size,'architecture':job['architecture'],
           'detector_slug':job['detector']['slug'],'custom_classes':True,
           'base_model_key':job['baseline']['key'],'base_model_package_arn':'',
           'base_model_scratch':False,'base_model_folder':'','detector_definition':job['detector'],
           'base_provenance':{**job['baseline'],'package_arn':job['baseline_arn']},
           'notes':f"Variant batch {job['name']} / {variant['name']}" + (' — '+str(variant['settings']['notes']) if variant['settings'].get('notes') else ''), 'queued_by':'experiment',
           'added_at':now(),'reproducible':True,'drop_undeclared_classes':False,
           'comparison':{'enabled':False},'experiment_id':job['id']}
    data.mark_version(version['version_id'],strict=True,state='queued',entry_id=entry['id'],
        entry_template={k:v for k,v in entry.items() if k not in ('id','added_at')})
    publish(job,variant,entry)

def compare(job, successful):
    control.checkpoint()
    audit_table=t2.training_audit_table()
    ident='experiment-comparison-'+job['id']+'-'+str(job['comparison_attempt'])
    audit=audit_table.get_item(Key={'id':ident}).get('Item')
    if not audit:
        anchor=cmp._read(successful[0]['run_id'])
        models=[{k:job['baseline'][k] for k in ('id','uri','etag','role','label')}]
        for variant in successful[1:]:
            pinned=pin(variant['model_package_arn'],'other',variant['name'])
            models.append({k:pinned[k] for k in ('id','uri','etag','role','label')})
        audit={**anchor,'id':ident,'record_type':'experiment_comparison',
               'dataset_version_id':job['dataset_version_id'],
               'comparison':{'id':uuid.uuid5(uuid.NAMESPACE_URL,ident).hex,'status':'Pending',
                   'evaluation':job['evaluation'],'selection':{'enabled':True,'models':models},'experiment_id':job['id']}}
        audit_table.put_item(Item=audit,ConditionExpression='attribute_not_exists(id)')
    cmp.advance(record_store.native_numbers(audit))
    state=record_store.native_numbers(audit_table.get_item(Key={'id':ident})['Item']['comparison'])
    job['comparison']=state
    if state['status']=='Completed':
        job['state']='Completed' if len(successful)==len(job['variants']) else 'Completed with failures'
    elif state['status'] in ('Failed','Unavailable'):
        job['state']='Comparison failed'
    else:
        job['state']='Comparing'
    save(job)

def advance(job):
    if not job.get('baseline'):
        job['baseline']=pin(job['baseline_arn'],'baseline','Baseline: '+job['baseline_arn'].split('/')[-2]+'/'+job['baseline_arn'].split('/')[-1])
        save(job)
    for variant in job['variants']:
        if variant['state'] in ('Pending','Preparing'):
            prepare_variant(job,variant);return
        if variant['state']=='Queued':
            audit=t2.training_audit_table().get_item(Key={'id':variant['run_id']}).get('Item')
            if audit and audit.get('status') in t2.TERMINAL_EXECUTION_STATUSES:
                variant['state']=audit['status']
                variant['model_package_arn']=(audit.get('model') or {}).get('model_package_arn','')
                if variant['state']=='Succeeded' and not variant['model_package_arn']:
                    variant.update(state='Failed',error='Training completed without a registered model.')
                variant['error']=variant.get('error') or audit.get('failure_reason','')
                save(job)
            else:
                variant['training_status']=(audit or {}).get('status','Queued')
                snapshot=t2.queue_snapshot()
                progress=snapshot.get('submission_progress') or {}
                if progress.get('entry_id')==variant['entry_id'] and progress.get('resource_wait'):
                    variant['training_status']='Waiting for resources'
                if not audit and not any(e['id']==variant['entry_id'] for e in snapshot.get('pending',[])) and not snapshot.get('running'):
                    variant.update(state='Failed',error='Queue entry was removed before training started. Retry this variant to submit a new attempt.')
                job['stage']=f"{variant['name']}: {variant.get('training_status','Queued')}";save(job);return
    successful=[v for v in job['variants'] if v['state']=='Succeeded']
    if successful:
        compare(job,successful)
    else:
        job.update(state='Failed',error='No successful variants to compare.');save(job)

ACTIVE_STATES = ('Pending', 'Preparing', 'Training', 'Comparing')
TRANSIENT_MESSAGES = ('the read operation timed out', 'read timeout', 'read timed out',
                      'connection timed out', 'connection reset', 'connection refused',
                      'server closed the connection', 'could not connect to the endpoint')


def transient_error(exc):
    seen=set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, (TimeoutError, ConnectionError)):
            return True
        if isinstance(exc, ClientError):
            return exc.response['Error']['Code'] in ('Throttling', 'ThrottlingException',
                'ServiceUnavailable', 'InternalFailure', 'RequestTimeout', 'RequestTimeoutException')
        if isinstance(exc, ValueError):
            return False
        if any(text in str(exc).lower() for text in TRANSIENT_MESSAGES):
            return True
        exc=exc.__cause__ or exc.__context__
    return False


def recoverable(job):
    return job['state'] in ACTIVE_STATES or (
        job['state']=='Needs attention' and
        any(text in str(job.get('error','')).lower() for text in TRANSIENT_MESSAGES))


def retry_due(job):
    return not job.get('retry_at') or dt.datetime.fromisoformat(job['retry_at']) <= dt.datetime.now(dt.timezone.utc)


def advance_current(ident):
    """Reconcile durable state after transient failures, never replay a stale submission."""
    def owned():
        job=read(ident)
        if not recoverable(job) or not retry_due(job):
            return
        if job['state']=='Needs attention':
            # Legacy timeout failures must resume reconciliation, not start a new attempt.
            job['state']='Comparing' if job.get('comparison') else 'Training' if any(
                v['state']=='Queued' for v in job['variants']) else 'Preparing'
            save(job)
        try:
            control.checkpoint()
            advance(job)
            if job.get('retry_count') or job.get('error') or job.get('retry_at'):
                current=read(ident)
                if current['state']!='Cancelled':
                    current.update(error='',retry_count=0)
                    current.pop('retry_at',None)
                    save(current)
        except control.Stopped as exc:
            current=read(ident)
            current.update(state='Cancelled',stage=str(exc));save(current)
        except Exception as exc:
            logging.getLogger(__name__).exception('Experiment %s reconciliation failed',ident)
            # A lost response can follow a successful SQL commit. Re-read before writing
            # error metadata so queued entries / completed runs are never overwritten.
            current=read(ident)
            if current['state'] not in ACTIVE_STATES:
                return
            if transient_error(exc):
                attempts=int(current.get('retry_count',0))+1
                delay=min(60,5 * 2**min(attempts-1,4))
                current.update(retry_count=attempts,
                    retry_at=(dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=delay)).isoformat(),
                    error=f'Temporary connection error; retrying automatically in {delay}s: {exc}'[:1500])
            else:
                current.update(state='Needs attention',error=str(exc)[:1500])
            save(current)
    return record_store.exclusive('training_variant:'+ident)(owned)()


@record_store.exclusive('training_variants')
def tick():
    if control.state().get('halted'): return
    if not record_store.enabled():
        return
    for job in reversed(all_records()):
        if not recoverable(job) or not retry_due(job):
            continue
        try:
            advance_current(job['id'])
        except Exception:
            # If even reading/saving failed, leave SQL state intact for the next tick.
            logging.getLogger(__name__).exception('Experiment %s worker will retry',job['id'])
        break

@record_store.exclusive('training_variants')
def retry(ident, index=None):
    job=read(ident)
    job['run_number']=public(job).get('run_number') or catalog_view.reserve_experiment_number(ident)
    if job['state']=='Cancelled':
        raise ValueError('This experiment was removed. Submit a new experiment to run these variants.')
    if job['state'] in ('Pending','Preparing','Training','Comparing'):
        raise ValueError('Wait until the current variant batch work finishes before retrying.')
    if index is not None:
        variant=next((v for v in job['variants'] if v['index']==index),None)
        if not variant or variant['state']=='Succeeded' or variant['state']=='Queued':
            raise ValueError('Only failed or unsubmitted variants can be retried.')
        variant['history'].append({k:v for k,v in variant.items() if k!='history'})
        variant.update(state='Pending',attempt=variant['attempt']+1,error='')
        for k in ('run_id','entry_id','model_package_arn','dataset_version_id'):
            variant.pop(k,None)
    if job.get('comparison'):
        job.setdefault('comparison_history',[]).append(job.pop('comparison'))
        job['comparison_attempt']+=1
    job.update(state='Pending',error='');save(job)
    return public(job)


def remove_pending(ident):
    return record_store.exclusive('training_variant:'+ident)(lambda: _remove_pending(ident))()

def _remove_pending(ident):
    """Lock only this experiment, so other preparation cannot block removal."""
    job=read(ident)
    if job['state']=='Cancelled':
        return public(job)
    if job['state']!='Pending' or any(v['state']!='Pending' or v.get('run_id') for v in job['variants']):
        raise ValueError('Only experiments that have not started can be removed. Preparing and running experiments cannot be removed.')
    job.update(state='Cancelled',stage='Removed from queue',cancelled_at=now())
    for variant in job['variants']:
        variant['state']='Cancelled'
    save(job)
    return public(job)


def scheduled_payload(config, ident):
    """Read only the explicitly saved active experiment; never infer from unrelated drafts."""
    pages=config.get('page_state') or {}
    prefix='/track2/multi-auto-training/'
    if (pages.get(prefix+'training-mode') or {}).get('trainingMode')!='experiments':
        return None
    selected=pages.get(prefix+'scheduled-experiment') or {}
    if not selected.get('enabled') or not all(selected.get(k) for k in ('detector_slug','architecture','baseline_arn')):
        raise ValueError('Save the experiment model, baseline and selected variants before scheduling.')
    variants=[v for v in selected.get('variants',[]) if v.get('selected')]
    names=[str(v.get('name','')).strip() for v in variants]
    if not 1<=len(variants)<=20 or not all(names) or len({n.casefold() for n in names})!=len(names):
        raise ValueError('Save between 1 and 20 selected variants with unique names before scheduling.')
    snapshots=[settings(v['settings']) for v in variants]
    return {'id':ident,'name':f"Scheduled {selected['detector_slug']} variants {ident[:8]}",
            **{k:selected[k] for k in ('detector_slug','architecture','baseline_arn')},
            'data_mode':'prepared','dataset_version_id':'','shared':snapshots[0],
            'evaluation':{'imgsz':snapshots[0]['img_size'],'confidence':.25,'ap_conf':.001,'nms_iou':.7,'max_det':300},
            'variants':[{'name':name,'overrides':snapshot} for name,snapshot in zip(names,snapshots)]}
