import copy
import datetime as dt
import pytest
from ppe_pipeline_api.training import export_schedules as es

ARN = 'arn:aws:sagemaker:ap-southeast-1:123:model-package/'


@pytest.fixture
def estate(monkeypatch):
    rows = {
        'helmet-s': [{'group_name':'helmet-s','version':'1','model_package_arn':ARN+'helmet-s/1','metrics':{'recall':.7}},
                     {'group_name':'helmet-s','version':'2','model_package_arn':ARN+'helmet-s/2','metrics':{'recall':.95}}],
        'helmet-l': [{'group_name':'helmet-l','version':'1','model_package_arn':ARN+'helmet-l/1','metrics':{'recall':.9}}],
        'vest-s': [{'group_name':'vest-s','version':'1','model_package_arn':ARN+'vest-s/1','metrics':{'recall':.99}}],
    }
    monkeypatch.setattr(es.mx,'group_versions',lambda group,**kw:copy.deepcopy(rows[group]))
    monkeypatch.setattr(es.t2,'exportable_groups',lambda:[{'group_name':g,'registry':g.split('-')[0]} for g in rows])
    monkeypatch.setattr(es.mx,'describe_pick',lambda row,criterion:dict(row,criterion=criterion))
    return rows


def plan(scope='comparison'):
    return dict(scope=scope,criterion='recall',model_package_arns=[ARN+'helmet-s/1',ARN+'helmet-l/1'],
                model_name='helmet',destination='s3',bucket='test-exports',prefix='models',enabled=True,
                time_sgt='09:00',start_date='2026-11-20',every_days=1)


def test_compared_scope_never_selects_an_unselected_better_model(estate):
    best,_,count=es.winner(plan())
    assert count==2 and best['model_package_arn']==ARN+'helmet-l/1'


def test_named_scope_includes_all_sizes_but_only_that_name(estate):
    best,_,count=es.winner(plan('model_name'))
    assert count==3 and best['model_package_arn']==ARN+'helmet-s/2'


def test_missing_selection_refuses_instead_of_silently_narrowing(estate):
    payload=plan();payload['model_package_arns'].append(ARN+'helmet-s/99')
    with pytest.raises(ValueError,match='unavailable'):es.winner(payload)


def test_unmeasured_models_do_not_win(estate):
    estate['helmet-l'][0]['metrics']={}
    best,_,_=es.winner(plan())
    assert best['model_package_arn']==ARN+'helmet-s/1'


def test_preview_identity_change_blocks_manual_export(estate,monkeypatch):
    monkeypatch.setattr(es.mx,'export_checkpoint',lambda *a,**k:pytest.fail('must not export'))
    with pytest.raises(ValueError,match='changed'):es.export_now(plan(),ARN+'helmet-s/1')


def test_schedule_time_is_singapore_and_preserves_day_interval():
    current=dt.datetime(2026,11,19,2,tzinfo=dt.timezone.utc) # 10am Singapore
    assert es.next_time('2026-11-19','09:00',2,current)=='2026-11-21T01:00:00+00:00'
    assert es.next_time('2026-11-20','09:00',1,current)=='2026-11-20T01:00:00+00:00'
    for time,days in [('24:00',1),('09:00',0),('09:00',366)]:
        with pytest.raises(ValueError):es.next_time('',time,days,current)


@pytest.fixture
def store(monkeypatch):
    data={}
    monkeypatch.setattr(es,'_get',lambda scope:copy.deepcopy(data.get(scope,{'scope':scope,'enabled':False})))
    def put(value,previous=None):
        if previous is not None and data.get(value['scope'])!=previous:return False
        data[value['scope']]=copy.deepcopy(value);return True
    monkeypatch.setattr(es,'_put',put)
    monkeypatch.setattr(es,'now',lambda:dt.datetime(2026,11,19,2,tzinfo=dt.timezone.utc))
    return data


def test_schedules_are_independent_and_save_does_not_export(estate,store,monkeypatch):
    monkeypatch.setattr(es,'export_now',lambda *a:pytest.fail('saving must not export'))
    es.save('comparison',plan());es.save('model_name',plan('model_name'))
    assert store['comparison']['model_package_arns']==plan()['model_package_arns']
    assert store['model_name']['model_package_arns']==[]
    es.save('comparison',{'enabled':False})
    assert not store['comparison']['enabled'] and store['model_name']['enabled']
    assert es.check()==[]


def test_due_slot_claimed_once_even_with_another_worker(estate,store,monkeypatch):
    es.save('comparison',plan());store['comparison']['next_run_at']='2026-11-15T01:00:00+00:00'
    calls=[]
    def export(payload):
        calls.append(payload);assert es.check()==[]
        return {'written_to':'s3://test-exports/winner.pt'}
    monkeypatch.setattr(es,'export_now',export)
    assert es.check()[0]['state']=='ready'
    assert len(calls)==1
    assert store['comparison']['next_run_at']=='2026-11-20T01:00:00+00:00'
    assert es.check()==[]


def test_failed_export_records_result_without_retrying_each_tick(estate,store,monkeypatch):
    es.save('comparison',plan());store['comparison']['next_run_at']='2026-11-19T01:00:00+00:00'
    def fail(*a):raise RuntimeError('checkpoint unavailable')
    monkeypatch.setattr(es,'export_now',fail)
    assert es.check()[0]['state']=='failed'
    assert 'checkpoint unavailable' in store['comparison']['last_result']['detail']
    assert es.check()==[]


def test_inflight_result_cannot_overwrite_stopped_schedule(estate,store,monkeypatch):
    es.save('comparison',plan());store['comparison']['next_run_at']='2026-11-19T01:00:00+00:00'
    def export(*a):
        es.save('comparison',{'enabled':False});return {'written_to':'s3://test-exports/winner.pt'}
    monkeypatch.setattr(es,'export_now',export)
    es.check()
    assert not store['comparison']['enabled']


def test_real_store_writes_use_conditional_body(monkeypatch):
    from ppe_pipeline_api import pull_control as pc
    calls=[]
    class DDB:
        def put_item(self,**kwargs):calls.append(kwargs)
    monkeypatch.setattr(pc,'ddb',DDB())
    before={'scope':'comparison','enabled':True}
    es._put({**before,'enabled':False},before)
    assert calls[0]['ConditionExpression']=='#body = :previous'
