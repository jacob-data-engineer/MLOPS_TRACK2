import copy
from unittest.mock import Mock
import inspect
from botocore.exceptions import ClientError
import pytest
from ppe_pipeline_api.chain import training_control as c, training_variants as v, preparation_jobs as p
from ppe_pipeline_api.training import track2_shared as t, comparison

@pytest.fixture
def control(monkeypatch):
    row={'halted':False,'generation':'old'}
    class Table:
        def get_item(self,**kw):return {'Item':copy.deepcopy(row)}
        def put_item(self,Item,**kw):
            if kw.get('ConditionExpression') and row['generation'] != kw['ExpressionAttributeValues'][':generation']:
                raise ClientError({'Error':{'Code':'ConditionalCheckFailedException'}},'PutItem')
            row.clear();row.update(copy.deepcopy(Item))
    monkeypatch.setattr(c.record_store,'enabled',lambda:True)
    monkeypatch.setattr(c.record_store,'Table',lambda *a:Table())
    monkeypatch.setattr(t,'queue_stop',Mock())
    monkeypatch.setattr(t,'queue_snapshot',lambda:{'pending':[]})
    monkeypatch.setattr(v,'all_records',lambda *a:[])
    monkeypatch.setattr(c,'audits',lambda:[])
    return row

def test_stop_is_durable_and_old_preparation_cannot_publish_after_resume(control,monkeypatch):
    token=c.checkpoint()
    result=c.stop()
    assert result['halted'] and control['halted']
    with pytest.raises(c.Stopped):c.checkpoint()
    control['halted']=False
    with pytest.raises(c.Stopped):c.checkpoint(token)
    with pytest.raises(c.Stopped):v.publish({'_stop_generation':token},{},{})
    with pytest.raises(c.Stopped):p.publish({'_stop_generation':token},{})

def test_sweep_stops_only_recorded_executions_and_comparisons(control,monkeypatch):
    control['halted']=True
    monkeypatch.setattr(c,'audits',lambda:[
      {'id':'run-a','status':'Executing','execution_arn':'owned-execution'},
      {'id':'run-b','status':'Succeeded','execution_arn':'finished','comparison':{'status':'Running','job_name':'owned-comparison'}},
      {'id':'run-c','status':'Succeeded','comparison':{'status':'Completed'}}])
    stop=Mock();processing=Mock()
    monkeypatch.setattr(t.sm,'describe_pipeline_execution',lambda **kw:{'PipelineExecutionStatus':'Executing'})
    monkeypatch.setattr(t.sm,'stop_pipeline_execution',stop)
    monkeypatch.setattr(t.sm,'describe_processing_job',lambda **kw:{'ProcessingJobStatus':'InProgress'})
    monkeypatch.setattr(t.sm,'stop_processing_job',processing)
    result=c.sweep()
    stop.assert_called_once_with(PipelineExecutionArn='owned-execution')
    processing.assert_called_once_with(ProcessingJobName='owned-comparison')
    assert result['pending'] and not result['problems']

def test_aws_stop_failure_is_reported_and_pause_remains(control,monkeypatch):
    control['halted']=True
    monkeypatch.setattr(c,'audits',lambda:[{'id':'run','status':'Executing','execution_arn':'owned'}])
    monkeypatch.setattr(t.sm,'describe_pipeline_execution',Mock(side_effect=RuntimeError('denied')))
    assert 'denied' in c.sweep()['problems'][0]
    assert control['halted']

def test_cancels_pending_work_without_removing_completed_results(control,monkeypatch):
    control['halted']=True
    jobs=[{'id':'pending','state':'Pending','variants':[{'state':'Pending'},{'state':'Succeeded','model':'kept'}]}, {'id':'done','state':'Completed'}]
    monkeypatch.setattr(v,'all_records',lambda kind:jobs if kind==v.KIND else [])
    saved=[];monkeypatch.setattr(v,'save',lambda j:saved.append(copy.deepcopy(j)))
    c.sweep()
    assert len(saved)==1 and saved[0]['state']=='Cancelled'
    assert saved[0]['variants'][1]['model']=='kept'

def test_fast_checkpoint_limits_database_polling(control,monkeypatch):
    monkeypatch.setattr(c,'_checked_at',0)
    now=[10.0];monkeypatch.setattr(c.time,'monotonic',lambda:now[0])
    c.checkpoint_fast();control['halted']=True
    now[0]=11.1
    with pytest.raises(c.Stopped):c.checkpoint_fast()


def test_automation_reenables_after_stop_completes(control,monkeypatch):
    control['halted']=True
    monkeypatch.setattr(c,'resume',inspect.unwrap(c.resume))
    result=c.auto_resume()
    assert result['halted'] is False and control['halted'] is False
    assert control['generation']=='old'

@pytest.mark.parametrize('result',[{'pending':True,'problems':[]},{'pending':False,'problems':['AWS refused stop']}])
def test_automation_stays_paused_until_aws_stops(control,monkeypatch,result):
    control['halted']=True
    monkeypatch.setattr(c,'resume',inspect.unwrap(c.resume))
    monkeypatch.setattr(c,'sweep',lambda:result)
    c.auto_resume()
    assert control['halted']

def test_new_stop_cannot_be_overwritten_by_auto_resume(control,monkeypatch):
    control['halted']=True
    monkeypatch.setattr(c,'resume',inspect.unwrap(c.resume))
    def sweep():
        control['generation']='new-stop'
        return {'pending':False,'problems':[]}
    monkeypatch.setattr(c,'sweep',sweep)
    c.auto_resume()
    assert control['halted'] and control['generation']=='new-stop'

def test_busy_preparation_lock_does_not_enable_automation(control,monkeypatch):
    control['halted']=True
    monkeypatch.setattr(c,'resume',lambda:{'started':False,'reason':'busy'})
    c.auto_resume()
    assert control['halted']
