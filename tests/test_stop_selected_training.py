from types import SimpleNamespace
import pytest
from fastapi import HTTPException
from ppe_pipeline_api.routers import track2_models as m

@pytest.mark.parametrize('state',['Executing','Stopping','Succeeded'])
def test_stops_only_the_selected_record(monkeypatch,state):
 calls=[]
 monkeypatch.setattr(m.t2,'training_audit_table',lambda:SimpleNamespace(get_item=lambda **kw:{'Item':{'execution_arn':'selected-execution'}}))
 monkeypatch.setattr(m.t2.sm,'describe_pipeline_execution',lambda **kw:{'PipelineExecutionStatus':state})
 monkeypatch.setattr(m.t2.sm,'stop_pipeline_execution',lambda **kw:calls.append(kw))
 if state=='Succeeded':
  with pytest.raises(HTTPException) as exc:m.stop_selected_training(m.StopSelectedRunRequest(run_id='run-a'),session={})
  assert exc.value.status_code==409
 else:m.stop_selected_training(m.StopSelectedRunRequest(run_id='run-a'),session={})
 assert calls==([{'PipelineExecutionArn':'selected-execution'}] if state=='Executing' else [])

@pytest.mark.parametrize('audit',[None,{}])
def test_no_fallback_to_other_training(monkeypatch,audit):
 monkeypatch.setattr(m.t2,'training_audit_table',lambda:SimpleNamespace(get_item=lambda **kw:{'Item':audit}))
 monkeypatch.setattr(m.t2.sm,'stop_pipeline_execution',lambda **kw:pytest.fail('Stopped unrelated execution'))
 with pytest.raises(HTTPException):m.stop_selected_training(m.StopSelectedRunRequest(run_id='missing'),session={})
