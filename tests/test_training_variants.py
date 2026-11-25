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
    table,audits=Table(),Table()
    monkeypatch.setattr(e,'table',lambda:table)
    monkeypatch.setattr(e.t2,'training_audit_table',lambda:audits)
    monkeypatch.setattr(e.data,'require_training_sql',lambda:None)
    monkeypatch.setattr(e.data,'check_saved_training',lambda *a:{'train_images':10,'val_images':3})
    monkeypatch.setattr(e.t2,'get_detector',lambda slug:{'slug':slug,'classes':{'0':'helmet'}})
    monkeypatch.setattr(e.t2,'detector_group_name',lambda *a:'yolo26s-helmet-detector')
    monkeypatch.setattr(e.t2.sm,'describe_model_package',lambda **kw:{'ModelPackageStatus':'Completed'})
    monkeypatch.setattr(e.catalog_view,'run_numbers',lambda ids:{i:n+1 for n,i in enumerate(ids)})
    monkeypatch.setattr(e.record_store,'enabled',lambda:False)
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
    assert used==[('fixed-dataset',{'0':'helmet'})]*2
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
    assert used==['fresh-shared','fresh-shared']
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
    monkeypatch.setattr(e,'advance',lambda j:pytest.fail('Cancelled experiment executed'))
    e.tick.__wrapped__()

@pytest.mark.parametrize('state',['Preparing','Training','Comparing','Completed','Needs attention'])
def test_remove_rejects_started_experiments(store,state):
    j=job(store);j['state']=state;e.save(j)
    with pytest.raises(ValueError,match='not started'):e.remove_pending(j['id'])
    assert e.read(j['id'])['state']==state
