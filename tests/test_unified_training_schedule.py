import datetime as dt
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from ppe_pipeline_api import record_store
from ppe_pipeline_api.chain import retrain_schedule as rs, sql_worker, queue_after_pull
from ppe_pipeline_api.routers import track2_retention as router


@pytest.fixture
def sql_mode(monkeypatch):
    monkeypatch.setattr(record_store, 'enabled', lambda: True)


def test_old_pull_clock_is_read_as_one_training_clock(sql_mode):
    config={'mode':'schedule','schedule_every_days':3,'schedule_time':'14:50',
            'schedule_from':'date','schedule_start_date':'2026-11-20','retrain_schedule':{'trigger':'pull','enabled':True}}
    plan=rs.settings(config)
    assert rs.training_trigger(config)=='clock' and not rs.follows_pull(config)
    assert (plan['every_days'],plan['at'],plan['from_date'])==(3,'14:50','2026-11-20')
    config['mode']='off'
    assert rs.training_trigger(config)=='off' and not rs.settings(config)['enabled']


def test_independent_pull_clock_cannot_enqueue(sql_mode,monkeypatch):
    monkeypatch.setattr(sql_worker.pc,'claim_once',lambda *a:False)
    monkeypatch.setattr(sql_worker.pc,'enqueue_pull',lambda **kw:pytest.fail('independent pull still active'))
    monkeypatch.setattr(sql_worker.pc,'read_config',lambda:{'mode':'schedule','sql_next_pull_at':'2020-01-01T00:00:00Z'})
    sql_worker.tick.__wrapped__()
    assert not queue_after_pull.after_sql_pull('old',{'copied':100})['queued']


def test_schedule_date_and_time_save_reload_without_immediate_run(sql_mode,monkeypatch):
    config={'retrain_schedule':{'trigger':'off','enabled':False,'at':'02:00','every_days':1}}
    monkeypatch.setattr(rs.pc,'read_config',lambda:config)
    monkeypatch.setattr(rs.pc,'write_config',lambda changes:config.update(changes))
    monkeypatch.setattr(rs,'preflight',lambda c:{'settings':rs.settings(c)})
    monkeypatch.setattr(rs,'_now',lambda:dt.datetime(2026,11,17,6,55,tzinfo=dt.timezone.utc))
    claims=[]
    monkeypatch.setattr(rs.pc,'claim_once',lambda *a:claims.append(a) or True)
    result=router.save_retrain(router.RetrainScheduleRequest(trigger='clock',every_days=3,at='14:50',from_date='2026-11-17'))
    assert result['settings']['from_date']=='2026-11-17'
    assert result['next_runs'][0]=='2026-11-20T06:50:00+00:00'
    assert config['mode']=='off' and config['sql_next_pull_at']==''
    assert claims == [('training_schedule','2026-11-17T06:50:00+00:00')]
    assert router.get_retrain()['settings']['at']=='14:50'


@pytest.mark.parametrize('patch',[{'at':'25:00'},{'from_date':'2026-02-30'}])
def test_invalid_schedule_rejected(sql_mode,monkeypatch,patch):
    monkeypatch.setattr(rs.pc,'read_config',lambda:{})
    with pytest.raises(HTTPException) as error:
        router.save_retrain(router.RetrainScheduleRequest(**patch))
    assert error.value.status_code==400
