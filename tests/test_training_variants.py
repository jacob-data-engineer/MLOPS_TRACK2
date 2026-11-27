import copy
import uuid
import pytest
from botocore.exceptions import ClientError
from ppe_pipeline_api.chain import training_variants as e

ARN='arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/yolo26s-helmet-detector/9'
class Table:
    def __init__(self): self.rows={}
    def key(self,row):return row.get('id') if 'kind' not in row else row['sort']
    def get_item(self,Key):return {'Item':copy.deepcopy(self.rows.get(self.key(Key)))} if self.key(Key) in self.rows else {}
    def put_item(self,Item,**kw):
        key=self.key(Item)
        if kw.get('ConditionExpression') and key in self.rows:
            raise ClientError({'Error':{'Code':'ConditionalCheckFailedException'}},'PutItem')
        self.rows[key]=copy.deepcopy(Item)
    def query(self,**kw):return {'Items':list(copy.deepcopy(self.rows).values())}

@pytest.fixture
def store(monkeypatch):
    from ppe_pipeline_api.data import pt_view
    monkeypatch.setattr(pt_view,'inspect_registered',lambda arn:{'classes':[{'id':0,'name':'helmet'}]})
    table,audits=Table(),Table()
    monkeypatch.setattr(e,'table',lambda:table)
    monkeypatch.setattr(e.t2,'training_audit_table',lambda:audits)
    monkeypatch.setattr(e.data,'require_training_sql',lambda:None)
    monkeypatch.setattr(e.data,'check_saved_training',lambda *a:{'train_images':10,'val_images':3})
    monkeypatch.setattr(e.t2,'get_detector',lambda slug:{'slug':slug,'classes':{'0':'helmet'}})
    monkeypatch.setattr(e.t2,'detector_group_name',lambda *a:'yolo26s-helmet-detector')
    monkeypatch.setattr(e.t2.sm,'describe_model_package',lambda **kw:{'ModelPackageStatus':'Completed'})
    monkeypatch.setattr(e.catalog_view,'reserve_experiment_number',lambda ident:101)
    monkeypatch.setattr(e.catalog_view,'run_numbers',lambda ids:{i:n+1 for n,i in enumerate(ids)})
    monkeypatch.setattr(e.record_store,'enabled',lambda:False)
    monkeypatch.setattr(e.control,'state',lambda:{'halted':False,'generation':''})
    monkeypatch.setattr(e.dataset_versions,'get',lambda ident:{'version_id':ident})
    return table,audits

def payload(count=2):
    return {'id':str(uuid.uuid4()),'name':'Learning rate','detector_slug':'helmet','architecture':'yolo26s',
            'baseline_arn':ARN,'dataset_version_id':'fixed-dataset','shared':{'epochs':20,'seed':42,'hyperparams':{'optimizer':'AdamW','lr0':.01}},
            'evaluation':{'imgsz':640,'confidence':.25,'ap_conf':.001,'nms_iou':.7,'max_det':300},
            'variants':[{'name':f'Variant {i}','overrides':{'hyperparams':{'lr0':.005}}} for i in range(count)]}

def job(store,count=2):
    p=payload(count);e.create(p);return e.read(p['id'])

def test_creation_is_idempotent_and_captures_settings(store):
    p=payload();a=e.create(p);b=e.create(p)
    assert a['id']==b['id'] and len(store[0].rows)==1
    assert a['variants'][0]['settings']['hyperparams']=={'optimizer':'AdamW','lr0':.005}
    assert a['variants'][0]['settings']['seed']==42
    p['name']='Changed'
    with pytest.raises(ValueError,match='different'):e.create(p)

def test_registered_seed_from_another_model_name_is_allowed(store):
    p=payload();p['baseline_arn']=ARN.replace('helmet','person')
    p['shared']['notes']='Compare learning rates'
    result=e.create(p)
    assert result['baseline_arn']==p['baseline_arn']
    assert result['variants'][0]['settings']['notes']=='Compare learning rates'

@pytest.mark.parametrize('architecture',['yolo11s','yolo26m'])
def test_incompatible_baseline_architecture_rejected(store,architecture):
    p=payload();p['baseline_arn']=ARN.replace('yolo26s',architecture)
    with pytest.raises(ValueError,match='generation and size'):e.create(p)
    assert not store[0].rows


@pytest.mark.parametrize('bad',[{'epochs':0},{'epochs':None},{'batch':True},{'seed':-1},{'img_size':10000},{'batch':0},{'img_size':1},{'unknown':True},{'hyperparams':{'epochs':5}}])
def test_invalid_settings_rejected(bad):
    with pytest.raises(ValueError):e.settings(bad)

def test_preparation_uses_same_pinned_inputs_and_unique_entries(store,monkeypatch):
    j=job(store);j['baseline']={'key':'pinned/best.pt','uri':'s3://bucket/pinned/best.pt'}
    used=[];entries=[]
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{})
    monkeypatch.setattr(e.data,'prepare_saved_training',lambda ident,**kw:used.append((ident,kw['target_classes'])) or {'version_id':'replayed'})
    monkeypatch.setattr(e.data,'version_run_fields',lambda v:{'dataset_version_id':v['version_id']})
    monkeypatch.setattr(e.data,'mark_version',lambda *a,**kw:None)
    monkeypatch.setattr(e,'publish',lambda j,v,entry:entries.append(entry))
    for v in j['variants']:e.prepare_variant(j,v)
    assert used==[('fixed-dataset',{'0':'helmet'})]
    assert {x['dataset_version_id'] for x in entries}=={'replayed'}
    assert {x['base_model_key'] for x in entries}=={'pinned/best.pt'}
    assert len({x['id'] for x in entries})==2
    assert entries[0]['id']==e.entry_id(j,j['variants'][0])

def test_queued_variant_is_not_prepared_twice_after_restart(store,monkeypatch):
    j=job(store);j['baseline']={};j['variants'][0].update(state='Queued',run_id='run-a',entry_id='a')
    monkeypatch.setattr(e,'pin',lambda *a:{'uri':'s3://b/base'})
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{'pending':[{'id':'a'}]})
    monkeypatch.setattr(e,'prepare_variant',lambda *a:pytest.fail('duplicate preparation'))
    e.advance(j)
    assert e.read(j['id'])['variants'][0]['state']=='Queued'

def test_comparison_contains_all_successful_variants_and_is_restart_safe(store,monkeypatch):
    j=job(store,7);j['baseline']={'id':ARN,'uri':'s3://b/base','etag':'fixed','role':'baseline','label':'Base'}
    for i,v in enumerate(j['variants']):
        v.update(state='Succeeded',run_id=f'run-{i}',model_package_arn=f'{ARN[:-1]}{i+10}')
    anchor={'id':'run-0','status':'Succeeded','model':{'model_package_arn':j['variants'][0]['model_package_arn']},'reproducibility':{}}
    monkeypatch.setattr(e.cmp,'_read',lambda ident:anchor)
    pins=[]
    monkeypatch.setattr(e,'pin',lambda arn,role,label:pins.append(arn) or {'id':arn,'uri':'s3://b/'+label,'etag':'fixed','role':role,'label':label})
    monkeypatch.setattr(e.cmp,'advance',lambda a:None)
    e.compare(j,j['variants']);e.compare(e.read(j['id']),j['variants'])
    assert len(pins)==6 # not duplicated on restart
    synthetic=next(iter(store[1].rows.values()))
    assert synthetic['record_type']=='experiment_comparison'
    assert synthetic['dataset_version_id']=='fixed-dataset'
    assert len(synthetic['comparison']['selection']['models'])==7 # baseline + six others; anchor is added by runner
    assert synthetic['comparison']['evaluation']==j['evaluation']

def test_retry_failed_variant_preserves_successes(store):
    j=job(store);j['state']='Completed with failures'
    j['variants'][0].update(state='Succeeded',run_id='run-ok')
    j['variants'][1].update(state='Failed',run_id='run-failed')
    old=e.entry_id(j,j['variants'][1]);e.save(j)
    result=e.retry(j['id'],1)
    assert result['variants'][0]['run_id']=='run-ok'
    assert result['variants'][1]['state']=='Pending'
    assert e.entry_id(result,result['variants'][1])!=old
    assert result['variants'][1]['history'][0]['run_id']=='run-failed'

def test_retry_cannot_duplicate_active_training(store):
    j=job(store);j['state']='Training';e.save(j)
    with pytest.raises(ValueError,match='finishes'):e.retry(j['id'],0)

def test_publish_commits_entry_switch_and_experiment_together(store,monkeypatch):
    j=job(store);calls=[]
    class Connection:
        def cursor(self):return self
        def commit(self):calls.append('commit')
        def close(self):calls.append('close')
    monkeypatch.setattr(e.record_store,'connection',Connection)
    monkeypatch.setattr(Table,'_lock',lambda *a:None,raising=False)
    monkeypatch.setattr(Table,'_save',lambda self,cur,row:calls.append(copy.deepcopy(row)),raising=False)
    monkeypatch.setattr(e.record_store,'Table',lambda name:store[1])
    e.publish(j,j['variants'][0],{'id':'fixed-id','dataset_version_id':'replay'})
    assert [x['record_type'] for x in calls[:2]]==['queue_entry','queue_state']
    assert calls[2]['variants'][0]['run_id']=='run-fixed-id'
    assert calls[2]['variants'][0]['state']=='Queued'
    assert calls[3:]==['commit','close']

def test_publish_does_not_commit_partial_transaction(store,monkeypatch):
    j=job(store);commits=[]
    class Connection:
        def cursor(self):return self
        def commit(self):commits.append(True)
        def close(self):pass
    monkeypatch.setattr(e.record_store,'connection',Connection)
    monkeypatch.setattr(Table,'_lock',lambda *a:None,raising=False)
    def fail(*args):raise RuntimeError('SQL unavailable')
    monkeypatch.setattr(Table,'_save',fail,raising=False)
    monkeypatch.setattr(e.record_store,'Table',lambda name:store[1])
    with pytest.raises(RuntimeError,match='SQL unavailable'):
        e.publish(j,j['variants'][0],{'id':'fixed-id','dataset_version_id':'replay'})
    assert not commits

def test_api_bounds_and_authentication(store):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ppe_pipeline_api.routers import track2_training_variants as router
    app=FastAPI();app.include_router(router.router,prefix='/track2/training-variants')
    client=TestClient(app)
    assert client.get('/track2/training-variants').status_code in (401,403)
    app.dependency_overrides[router.require_session]=lambda:{}
    p=payload();p['evaluation']['imgsz']=0
    assert client.post('/track2/training-variants',json=p).status_code==422
    p=payload();assert client.post('/track2/training-variants',json=p).status_code==200
    assert len(client.get('/track2/training-variants').json()['runs'])==1

@pytest.fixture
def draft_store(store,monkeypatch):
    class Connection:
        def cursor(self):return self
        def commit(self):pass
        def close(self):pass
    monkeypatch.setattr(e.record_store,'connection',Connection)
    monkeypatch.setattr(Table,'_lock',lambda *a:None,raising=False)
    monkeypatch.setattr(Table,'_read',lambda self,cur,key:copy.deepcopy(self.rows.get(key,{})),raising=False)
    monkeypatch.setattr(Table,'_save',lambda self,cur,row:self.put_item(Item=row),raising=False)
    monkeypatch.setattr(e,'pin',lambda *a:pytest.fail('Saving a draft must not touch AWS'))
    monkeypatch.setattr(e,'publish',lambda *a:pytest.fail('Saving a draft must not queue training'))
    return store

def draft_payload():
    p=payload();return {k:p[k] for k in ('detector_slug','architecture','baseline_arn','dataset_version_id')} | {
        'revision':0,'variants':[{'id':str(uuid.uuid4()),'name':'Reference settings','selected':True,
                                'settings':{'epochs':20,'hyperparams':{'lr0':.01}}}]}

def test_drafts_persist_complete_independent_snapshots_without_training(draft_store):
    p=draft_payload();saved=e.save_draft(p)
    assert saved['revision']==1 and saved['variants'][0]['settings']['batch']==16
    p['variants'][0]['settings']['epochs']=200
    assert next(iter(draft_store[0].rows.values()))['variants'][0]['settings']['epochs']==20
    assert draft_store[1].rows=={}

def test_stale_draft_cannot_overwrite_another_window(draft_store):
    p=draft_payload();e.save_draft(p)
    with pytest.raises(ValueError,match='another window'):e.save_draft(p)
    p['revision']=1;p['variants'][0]['settings']['epochs']=30
    assert e.save_draft(p)['revision']==2

def test_draft_context_isolated_and_names_unique(draft_store):
    p=draft_payload();first=e.save_draft(p)
    p['baseline_arn']=ARN[:-1]+'10'
    second=e.save_draft(p)
    assert first['id']!=second['id'] and len(draft_store[0].rows)==2
    p['revision']=1;p['variants'].append({**p['variants'][0],'id':str(uuid.uuid4()),'name':'reference settings'})
    with pytest.raises(ValueError,match='unique'):e.save_draft(p)

def test_fresh_draft_saves_without_dataset_or_preparation(draft_store,monkeypatch):
    p=draft_payload();p.update(data_mode='prepared',dataset_version_id='')
    monkeypatch.setattr(e.data,'freeze_prepared_training',lambda *a,**k:pytest.fail('draft must not prepare'))
    saved=e.save_draft(p)
    assert saved['data_mode']=='prepared' and saved['dataset_version_id']==''


def test_fresh_batch_prepares_once_and_replays_after_restart(store,monkeypatch):
    from ppe_pipeline_api.chain import retrain_schedule as rs
    config={'source':'original'}
    monkeypatch.setattr(e.pc,'read_config',lambda:config)
    p=payload();p.update(data_mode='prepared',dataset_version_id='')
    e.create(p);config['source']='changed later'
    j=e.read(p['id']);assert j['preparation_config']=={'source':'original'}
    j['baseline']={'key':'pinned.pt'}
    calls=[];used=[]
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{})
    monkeypatch.setattr(rs,'combine_and_decay',lambda c,**kw:calls.append(('prepare',copy.deepcopy(c))))
    monkeypatch.setattr(e.data,'freeze_prepared_training',lambda c,**kw:calls.append(('freeze',c)) or {'version_id':'fresh-shared'})
    monkeypatch.setattr(e.data,'prepare_saved_training',lambda ident,**kw:used.append(ident) or {'version_id':'replay'})
    monkeypatch.setattr(e.data,'version_run_fields',lambda v:{'dataset_version_id':v['version_id']})
    monkeypatch.setattr(e.data,'mark_version',lambda *a,**kw:None)
    monkeypatch.setattr(e,'publish',lambda j,v,entry:e.save(j))
    e.prepare_variant(j,j['variants'][0])
    j=e.read(p['id'])
    e.prepare_variant(j,j['variants'][1])
    assert [x[0] for x in calls]==['prepare','freeze']
    assert used==[]
    assert j['working_dataset_version_id']=='fresh-shared'
    assert j['dataset_version_id']=='fresh-shared'


def test_batch_name_is_saved_without_starting_work(draft_store):
    p=draft_payload();p['batch_name']='My batch'
    saved=e.save_draft(p)
    assert saved['batch_name']=='My batch'
    assert saved['variants'][0]['selected']
    assert draft_store[1].rows=={}

def test_worker_trains_variants_in_order_then_compares_with_original_base(store, monkeypatch):
    j = job(store)
    j['baseline'] = {'id': ARN, 'key': 'fixed/base.pt', 'uri': 's3://bucket/fixed/base.pt', 'etag': 'fixed', 'role': 'baseline', 'label': 'Base'}
    actions = []
    def prepare(current, variant):
        actions.append(('train', variant['index'], current['baseline']['id'], current['dataset_version_id']))
        variant.update(state='Queued', run_id=f"run-{variant['index']}", entry_id=f"entry-{variant['index']}")
        e.save(current)
    def compare(current, successful):
        actions.append(('compare', [v['index'] for v in successful], current['baseline']['id'], current['dataset_version_id']))
        current['state'] = 'Completed'
        e.save(current)
    monkeypatch.setattr(e, 'prepare_variant', prepare)
    monkeypatch.setattr(e, 'compare', compare)
    monkeypatch.setattr(e.t2, 'queue_snapshot', lambda: {'running': {'id': 'active'}})
    e.advance(j)
    e.advance(e.read(j['id']))  # Still running: no next variant and no comparison.
    assert len(actions) == 1
    for index in range(2):
        store[1].put_item(Item={'id': f'run-{index}', 'status': 'Succeeded', 'model': {'model_package_arn': ARN[:-1] + str(index + 10)}})
        e.advance(e.read(j['id']))
    assert actions == [('train', 0, ARN, 'fixed-dataset'), ('train', 1, ARN, 'fixed-dataset'), ('compare', [0, 1], ARN, 'fixed-dataset')]
    assert e.read(j['id'])['state'] == 'Completed'


def test_remove_pending_preserves_history_and_prevents_execution(store,monkeypatch):
    j=job(store)
    removed=e.remove_pending(j['id'])
    assert removed['state']=='Cancelled'
    assert all(v['state']=='Cancelled' for v in removed['variants'])
    assert e.remove_pending(j['id'])['state']=='Cancelled'
    with pytest.raises(ValueError,match='removed'):e.retry(j['id'])
    monkeypatch.setattr(e.record_store,'enabled',lambda:True)
    monkeypatch.setattr(e,'worker_candidates',e.all_records)
    monkeypatch.setattr(e,'advance',lambda j:pytest.fail('Cancelled experiment executed'))
    e.tick.__wrapped__()

@pytest.mark.parametrize('state',['Preparing','Training','Comparing','Completed','Needs attention'])
def test_remove_rejects_started_experiments(store,state):
    j=job(store);j['state']=state;e.save(j)
    with pytest.raises(ValueError,match='not started'):e.remove_pending(j['id'])
    assert e.read(j['id'])['state']==state


def test_removal_does_not_acquire_global_worker_lock(store,monkeypatch):
    j=job(store);locks=[]
    def exclusive(name):
        locks.append(name)
        assert name!='training_variants'
        return lambda fn:fn
    monkeypatch.setattr(e.record_store,'exclusive',exclusive)
    assert e.remove_pending(j['id'])['state']=='Cancelled'
    assert locks==['training_variant:'+j['id']]
    monkeypatch.setattr(e,'advance',lambda j:pytest.fail('Stale listing revived removed job'))
    e.advance_current(j['id'])


def test_running_other_experiment_does_not_block_pending_removal(store,monkeypatch):
    import threading
    a=job(store);b=job(store)
    locks={};entered=threading.Event();release=threading.Event()
    def exclusive(name):
        lock=locks.setdefault(name,threading.Lock())
        def decorate(fn):
            def wrapped(*args,**kwargs):
                if not lock.acquire(blocking=False):return {'started':False}
                try:return fn(*args,**kwargs)
                finally:lock.release()
            return wrapped
        return decorate
    monkeypatch.setattr(e.record_store,'exclusive',exclusive)
    monkeypatch.setattr(e,'advance',lambda j:(entered.set(),release.wait(5)))
    thread=threading.Thread(target=e.advance_current,args=(a['id'],));thread.start()
    try:
        assert entered.wait(2)
        assert e.remove_pending(b['id'])['state']=='Cancelled'
        assert e.remove_pending(a['id'])=={'started':False}
    finally:release.set();thread.join(5)



def scheduled_config():
    prefix='/track2/multi-auto-training/'
    return {'page_state':{prefix+'training-mode':{'trainingMode':'experiments'},
            prefix+'scheduled-experiment':{'enabled':True,'detector_slug':'helmet','architecture':'yolo26s','baseline_arn':ARN,
              'variants':[{'name':'One epoch','selected':True,'settings':{'epochs':1}},
                          {'name':'Fifty epochs','selected':True,'settings':{'epochs':50}},
                          {'name':'Not chosen','selected':False,'settings':{'epochs':100}}]}}}


def test_schedule_uses_explicit_saved_selection_and_fresh_data():
    config=scheduled_config();p=e.scheduled_payload(config,str(uuid.uuid4()))
    assert [v['overrides']['epochs'] for v in p['variants']]==[1,50]
    assert p['baseline_arn']==ARN and p['data_mode']=='prepared' and p['dataset_version_id']==''
    config['page_state']['/track2/multi-auto-training/scheduled-experiment']['variants'][0]['settings']['epochs']=999
    assert p['variants'][0]['overrides']['epochs']==1
    assert e.scheduled_payload({},'id') is None


def test_schedule_refuses_missing_variant_configuration():
    config=scheduled_config();del config['page_state']['/track2/multi-auto-training/scheduled-experiment']
    with pytest.raises(ValueError,match='Save the experiment'):e.scheduled_payload(config,'id')


def test_scheduled_variants_pull_once_then_share_dataset(store,monkeypatch):
    from ppe_pipeline_api.chain import retrain_schedule as rs
    config=scheduled_config();p=e.scheduled_payload(config,str(uuid.uuid4()))
    e.create(p,preparation_config=config,trigger='schedule')
    j=e.read(p['id']);j['baseline']={'key':'base.pt'}
    calls=[]
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{})
    monkeypatch.setattr(e.pc,'pull_in_use',lambda c:True)
    monkeypatch.setattr(e.pc,'active_run',lambda:None)
    monkeypatch.setattr(e.pc,'new_run_id',lambda:'scheduled-pull')
    monkeypatch.setattr(e.pc,'route_pending',lambda **kw:calls.append('pull') or {})
    monkeypatch.setattr(rs,'combine_and_decay',lambda *a,**kw:calls.append('prepare'))
    monkeypatch.setattr(e.data,'freeze_prepared_training',lambda *a,**kw:calls.append('freeze') or {'version_id':'shared'})
    monkeypatch.setattr(e.data,'prepare_saved_training',lambda ident,**kw:calls.append(('replay',ident)) or {'version_id':ident})
    monkeypatch.setattr(e.data,'version_run_fields',lambda v:{'dataset_version_id':v['version_id']})
    monkeypatch.setattr(e.data,'mark_version',lambda *a,**kw:None)
    monkeypatch.setattr(e,'publish',lambda j,v,entry:e.save(j))
    e.prepare_variant(j,j['variants'][0])
    restored=e.read(p['id']);e.prepare_variant(restored,restored['variants'][1])
    assert calls==['pull','prepare','freeze']
    assert e.create(p,preparation_config=config,trigger='schedule')['id']==p['id']
    assert len(store[0].rows)==1


@pytest.mark.parametrize('result',[{'errors':['SQL unavailable']},{'aborted':True},{'limit_reached':True}])
def test_scheduled_pull_must_finish_before_preparation(store,monkeypatch,result):
    from ppe_pipeline_api.chain import retrain_schedule as rs
    p=e.scheduled_payload(scheduled_config(),str(uuid.uuid4()))
    e.create(p,preparation_config=scheduled_config(),trigger='schedule');j=e.read(p['id'])
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{})
    monkeypatch.setattr(e.pc,'pull_in_use',lambda c:True)
    monkeypatch.setattr(e.pc,'active_run',lambda:None)
    monkeypatch.setattr(e.pc,'new_run_id',lambda:'pull')
    monkeypatch.setattr(e.pc,'route_pending',lambda **kw:result)
    monkeypatch.setattr(rs,'combine_and_decay',lambda *a,**kw:pytest.fail('Premature preparation'))
    if result.get('limit_reached'):e.prepare_variant(j,j['variants'][0])
    else:
        with pytest.raises(ValueError,match='pull failed'):e.prepare_variant(j,j['variants'][0])
    assert not e.read(p['id']).get('pull_done')

def test_transient_timeout_retries_with_backoff_without_losing_queue(store,monkeypatch):
    j=job(store);j['state']='Training';j['variants'][0].update(state='Queued',run_id='existing-run');e.save(j)
    calls=[]
    def fail(current):calls.append(current['id']);raise TimeoutError('The read operation timed out')
    monkeypatch.setattr(e,'advance',fail)
    e.advance_current(j['id']);saved=e.read(j['id'])
    assert saved['state']=='Training' and saved['retry_count']==1
    assert saved['variants'][0]['run_id']=='existing-run'
    assert 'retrying automatically' in saved['error']
    e.advance_current(j['id']);assert len(calls)==1
    saved['retry_at']='2000-01-01T00:00:00+00:00';e.save(saved)
    monkeypatch.setattr(e,'advance',lambda current:None)
    e.advance_current(j['id']);saved=e.read(j['id'])
    assert saved['error']=='' and saved['retry_count']==0 and 'retry_at' not in saved


def test_legacy_timeout_reconciles_success_then_reuses_dataset_for_next_variant(store,monkeypatch):
    j=job(store);j.update(state='Needs attention',error='The read operation timed out',
        baseline={'key':'pinned.pt'},working_dataset_version_id='shared-working')
    j['variants'][0].update(state='Queued',run_id='run-74',entry_id='existing')
    e.save(j)
    store[1].put_item(Item={'id':'run-74','status':'Succeeded','model':{'model_package_arn':ARN}})
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{})
    monkeypatch.setattr(e.data,'prepare_saved_training',lambda *a,**kw:pytest.fail('must not rebuild shared inputs'))
    monkeypatch.setattr(e.data,'version_run_fields',lambda version:{'dataset_version_id':version['version_id']})
    monkeypatch.setattr(e.data,'mark_version',lambda *a,**kw:None)
    entries=[]
    def publish(current,variant,entry):
        entries.append(entry)
        variant.update(state='Queued',run_id='run-next',entry_id=entry['id'])
        current['state']='Training';e.save(current)
    monkeypatch.setattr(e,'publish',publish)
    e.advance_current(j['id'])
    saved=e.read(j['id'])
    assert saved['variants'][0]['state']=='Succeeded'
    assert saved['variants'][0]['run_id']=='run-74'
    assert len(entries)==1 and entries[0]['dataset_version_id']=='shared-working'
    assert saved['variants'][1]['state']=='Queued' and saved['error']==''


def test_lost_commit_response_does_not_restore_stale_pending_variant(store,monkeypatch):
    j=job(store)
    def committed_then_timeout(current):
        saved=e.read(current['id']);saved['state']='Training'
        saved['variants'][0].update(state='Queued',run_id='committed-run');e.save(saved)
        raise TimeoutError('lost response')
    monkeypatch.setattr(e,'advance',committed_then_timeout)
    e.advance_current(j['id'])
    assert e.read(j['id'])['variants'][0]['run_id']=='committed-run'


def test_permanent_error_still_requires_attention(store,monkeypatch):
    j=job(store)
    monkeypatch.setattr(e,'advance',lambda j:(_ for _ in ()).throw(ValueError('Class names do not match')))
    e.advance_current(j['id']);saved=e.read(j['id'])
    assert saved['state']=='Needs attention' and not e.recoverable(saved)


def test_failed_error_save_does_not_write_stale_snapshot_from_tick(store,monkeypatch):
    j=job(store)
    monkeypatch.setattr(e.record_store,'enabled',lambda:True)
    monkeypatch.setattr(e,'worker_candidates',e.all_records)
    monkeypatch.setattr(e,'advance_current',lambda ident:(_ for _ in ()).throw(TimeoutError('SQL unavailable')))
    monkeypatch.setattr(e,'save',lambda item:pytest.fail('outer tick must not overwrite newer state'))
    e.tick.__wrapped__()
    assert e.read(j['id'])['state']=='Pending'

def test_wrapped_database_socket_timeout_is_retryable():
    wrapped=RuntimeError('network error')
    wrapped.__cause__=TimeoutError('socket expired')
    assert e.transient_error(wrapped)


def test_timeout_does_not_revive_cancelled_experiment(store,monkeypatch):
    j=job(store)
    def cancelled_then_timeout(current):
        latest=e.read(current['id']);latest['state']='Cancelled';e.save(latest)
        raise TimeoutError('late response')
    monkeypatch.setattr(e,'advance',cancelled_then_timeout)
    e.advance_current(j['id'])
    assert e.read(j['id'])['state']=='Cancelled'

def test_parent_run_number_groups_all_variants(store):
    j=job(store)
    for i,v in enumerate(j['variants']):v['run_id']=f'child-{i}'
    got=e.public(j)
    assert got['run_number']==101
    assert [v['run_number'] for v in got['variants']]==[101,101]
    assert [v['training_job_number'] for v in got['variants']]==[1,2]


def test_legacy_experiment_uses_latest_existing_number_without_renumbering(store):
    j=job(store);j.pop('run_number')
    for i,v in enumerate(j['variants']):v['run_id']=f'child-{i}'
    got=e.public(j)
    assert got['run_number']==2
    assert [v['run_number'] for v in got['variants']]==[2,2]

def test_list_summaries_omits_manifests_but_preserves_full_records(store, monkeypatch):
    p=payload(); e.create(p)
    saved=e.read(p['id']); saved['dataset']={'file_provenance':['large manifest']}; saved['preparation_config']={'private':'settings'}
    saved['variants'][0]['run_id']='child';e.save(saved)
    calls=[]
    monkeypatch.setattr(e.catalog_view,'run_numbers',lambda ids:calls.append(ids) or {'child':77})
    result=e.list_summaries()
    assert len(calls)==1
    assert 'dataset' not in result[0] and 'preparation_config' not in result[0]
    assert result[0]['variants'][0]['training_job_number']==77
    assert e.read(p['id'])['dataset']['file_provenance']==['large manifest']
    assert 'dataset' in e.public(e.read(p['id']))

def test_empty_queue_during_handoff_does_not_fail_or_submit_again(store,monkeypatch):
    j=job(store);j['baseline']={'uri':'pinned'}
    j['variants'][0].update(state='Queued',run_id='run-a',entry_id='a')
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{'pending':[],'running':None})
    monkeypatch.setattr(e,'prepare_variant',lambda *a:pytest.fail('Advanced past unresolved submission'))
    e.advance(j)
    saved=e.read(j['id'])['variants'][0]
    assert saved['state']=='Queued'
    assert saved['training_status']=='Waiting for training record'


def test_legacy_false_failure_reconciles_success_without_retraining(store,monkeypatch):
    j=job(store);j['baseline']={'uri':'pinned'}
    j['variants'][0].update(state='Failed',run_id='run-a',entry_id='a',error=e.MISSING_QUEUE_ERROR)
    j['variants'][1].update(state='Queued',run_id='run-b',entry_id='b')
    store[1].put_item(Item={'id':'run-a','status':'Succeeded','model':{'model_package_arn':ARN}})
    store[1].put_item(Item={'id':'run-b','status':'Executing'})
    monkeypatch.setattr(e.t2,'queue_snapshot',lambda:{})
    monkeypatch.setattr(e,'prepare_variant',lambda *a:pytest.fail('Duplicate submission'))
    e.advance(j)
    saved=e.read(j['id'])['variants'][0]
    assert saved['state']=='Succeeded' and saved['error']==''
    assert saved['model_package_arn']==ARN


def test_reconciliation_preserves_real_failures_and_missing_evidence(store):
    j=job(store)
    j['variants'][0].update(state='Failed',run_id='run-a',error='CUDA out of memory')
    j['variants'][1].update(state='Failed',run_id='run-b',error=e.MISSING_QUEUE_ERROR)
    store[1].put_item(Item={'id':'run-a','status':'Succeeded','model':{'model_package_arn':ARN}})
    assert not e.reconcile_queue_failure(j)
    assert all(v['state']=='Failed' for v in j['variants'])

def test_unchanged_poll_does_not_rewrite_experiment(store,monkeypatch):
    j=job(store)
    writes=[]
    original=store[0].put_item
    monkeypatch.setattr(store[0],'put_item',lambda **kw:(writes.append(kw),original(**kw))[1])
    e.save(j)
    assert not writes
    j['variants'][0]['state']='Queued'
    e.save(j)
    assert len(writes)==1
    e.save(j)
    assert len(writes)==1
    assert not any('fingerprint' in k for k in writes[0]['Item'])


def test_failed_save_keeps_changes_retryable(store,monkeypatch):
    j=job(store);j['stage']='Changed'
    original=store[0].put_item
    monkeypatch.setattr(store[0],'put_item',lambda **kw:(_ for _ in ()).throw(TimeoutError()))
    with pytest.raises(TimeoutError):e.save(j)
    monkeypatch.setattr(store[0],'put_item',original)
    e.save(j)
    assert e.read(j['id'])['stage']=='Changed'


def test_worker_candidates_load_only_schedule_fields(store,monkeypatch):
    from types import SimpleNamespace
    queries=[]
    cur=SimpleNamespace(execute=lambda sql,args:queries.append((sql,args)),fetchall=lambda:[(e.record_store.typed({'id':'a','state':'Training','retry_at':'','error':''}),)])
    monkeypatch.setattr(e.record_store,'enabled',lambda:True)
    monkeypatch.setattr(e.record_store,'connection',lambda:SimpleNamespace(cursor=lambda:cur,close=lambda:None))
    monkeypatch.setattr(e,'table',lambda:SimpleNamespace(sql='records_test'))
    assert e.worker_candidates()==[{'id':'a','state':'Training','retry_at':'','error':''}]
    assert 'jsonb_build_object' in queries[0][0]
    assert "payload->'dataset'" not in queries[0][0]

def test_create_rejects_actual_checkpoint_class_mismatch_before_queueing(store,monkeypatch):
    from ppe_pipeline_api.data import pt_view
    monkeypatch.setattr(pt_view,'inspect_registered',lambda arn:{'classes':[{'id':0,'name':'person'}]})
    with pytest.raises(ValueError,match='Missing from baseline: helmet.*Extra baseline classes: person'):
        e.create(payload())
    assert not store[0].rows


def test_baseline_class_check_allows_reordered_names(monkeypatch):
    from ppe_pipeline_api.data import pt_view
    monkeypatch.setattr(pt_view,'inspect_registered',lambda arn:{'classes':[{'id':0,'name':'person'},{'id':1,'name':'helmet'}]})
    e.validate_baseline_classes(ARN,{'0':'helmet','1':'person'})


def test_create_rejects_unverifiable_classes_before_queueing(store,monkeypatch):
    from ppe_pipeline_api.data import pt_view
    monkeypatch.setattr(pt_view,'inspect_registered',lambda arn:(_ for _ in ()).throw(ValueError('unreadable checkpoint')))
    with pytest.raises(ValueError,match='could not be verified'):e.create(payload())
    assert not store[0].rows
