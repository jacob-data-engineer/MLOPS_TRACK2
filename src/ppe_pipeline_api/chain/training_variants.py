"""Durable variant batches: fixed inputs, atomic queue publication, shared evaluation."""
import contextlib
import copy
import datetime as dt
import hashlib
import json
import os
import uuid
from botocore.exceptions import ClientError
from .. import data, pull_control as pc, record_store
from ..training import track2_shared as t2, comparison as cmp
from ..data import catalog_view

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
        draft={**key,**context,'id':ident,'variants':variants,'revision':revision+1,
               'created_at':previous.get('created_at',now()),'updated_at':now()}
        tab._save(cur,draft);conn.commit()
    return draft

def settings(raw):
    allowed = {'epochs','img_size','batch','seed','patience','hyperparams'}
    if set(raw)-allowed:
        raise ValueError('Unsupported training parameter.')
    out = {'epochs':20,'img_size':640,'batch':16,'seed':42,'patience':0, **raw}
    for k in ('epochs','img_size','batch','seed','patience'):
        if isinstance(out[k], bool) or not isinstance(out[k], (int, float)) or not float(out[k]).is_integer():
            raise ValueError(f'{k} must be an integer.')
        out[k] = int(out[k])
    if not 1 <= out['epochs'] <= 10000 or not 32 <= out['img_size'] <= 4096 or out['batch'] == 0 or out['batch'] < -1 or out['patience'] < 0 or not 0 <= out['seed'] <= 2**32-1:
        raise ValueError('Invalid epochs, image size, batch or patience.')
    out['hyperparams'] = t2.normalize_hyperparams(out.get('hyperparams') or {})
    return out

def create(payload):
    data.require_training_sql()
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
    if baseline.split('/')[-2] != t2.detector_group_name(detector, payload['architecture']):
        raise ValueError('Select a baseline from the chosen model name and architecture.')
    package = t2.sm.describe_model_package(ModelPackageName=baseline)
    if package.get('ModelPackageStatus') != 'Completed':
        raise ValueError('The baseline model is not ready.')
    mode=payload.get('data_mode','saved')
    if mode not in ('saved','prepared'):
        raise ValueError('Choose saved or freshly prepared data.')
    check = data.check_saved_training(payload['dataset_version_id'], detector['classes']) if mode=='saved' else {}
    config=copy.deepcopy(pc.read_config()) if mode=='prepared' else None
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
           'architecture':payload['architecture'],'baseline_arn':baseline,
           'data_mode':mode,'preparation_config':config,
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
    for v in value['variants']:
        v['run_number']=numbers.get(v.get('run_id'))
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
    snapshot=t2.queue_snapshot()
    if snapshot.get('pending') or snapshot.get('running'):
        job['stage']='Waiting for the training queue';save(job);return
    variant['state']='Preparing';job['state']='Preparing';save(job)
    def progress(stage, completed, total):
        job['stage']=f"{variant['name']}: {stage}" + (f' ({completed}/{total})' if total else '')
        save(job)
    if job.get('data_mode')=='prepared' and not job.get('dataset_version_id'):
        from . import retrain_schedule as rs
        progress('Preparing shared fresh dataset for all variants',0,0)
        rs.combine_and_decay(job['preparation_config'],for_run=True,progress=progress)
        shared=data.freeze_prepared_training(job['preparation_config'],trigger='variant_batch',
            target_classes=job['detector']['classes'],progress=progress)
        job['dataset_version_id']=shared['version_id']
        job['dataset']=shared
        save(job)  # Persist shared SQL recipe before any variant is queued.
    version=data.prepare_saved_training(job['dataset_version_id'],target_classes=job['detector']['classes'],trigger='experiment',progress=progress)
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

@record_store.exclusive('training_variants')
def tick():
    if not record_store.enabled():
        return
    for job in reversed(all_records()):
        if job['state'] not in ('Pending','Preparing','Training','Comparing'):
            continue
        try:
            advance(job)
        except Exception as exc:
            job.update(state='Needs attention',error=str(exc)[:1500]);save(job)
        # One experiment per tick keeps preparation and AWS calls bounded.
        break

@record_store.exclusive('training_variants')
def retry(ident, index=None):
    job=read(ident)
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
