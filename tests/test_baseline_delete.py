import pytest
from botocore.exceptions import ClientError
from ppe_pipeline_api.training import comparison as c


def test_delete_only_registered_baseline_preserves_files_and_audits(monkeypatch):
    arn=f'arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/{c.group_name()}/3'
    monkeypatch.setattr(c,'baseline_versions',lambda:[{'version':3,'arn':arn,'label':'Baseline'}])
    calls=[]
    monkeypatch.setattr(c.t2.sm,'delete_model_package',lambda **kw:calls.append(kw))
    monkeypatch.setattr(c.t2.s3,'delete_object',lambda **kw:pytest.fail('Must retain files'))
    monkeypatch.setattr(c.t2.s3,'delete_objects',lambda **kw:pytest.fail('Must retain files'))
    monkeypatch.setattr(c.t2,'training_audit_table',lambda:pytest.fail('Must retain results'))
    assert c.delete_baseline(3)['deleted']
    assert calls==[{'ModelPackageName':arn}]


def test_delete_refuses_training_registry_and_invalid_version(monkeypatch):
    monkeypatch.setattr(c,'baseline_versions',lambda:[{'version':3,'arn':'arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/yolo26s-training-detector/3'}])
    monkeypatch.setattr(c.t2.sm,'delete_model_package',lambda **kw:pytest.fail('Wrong registry'))
    with pytest.raises(ValueError,match='Baseline Model registry'):c.delete_baseline(3)
    for value in (0,-1,True,'3'):
        with pytest.raises(ValueError):c.delete_baseline(value)


def test_delete_missing_is_idempotent_and_aws_failure_propagates(monkeypatch):
    monkeypatch.setattr(c,'baseline_versions',lambda:[])
    assert c.delete_baseline(3)['already_absent']
    arn=f'arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/{c.group_name()}/3'
    monkeypatch.setattr(c,'baseline_versions',lambda:[{'version':3,'arn':arn}])
    def fail(**kw):raise ClientError({'Error':{'Code':'AccessDeniedException','Message':'Denied'}},'DeleteModelPackage')
    monkeypatch.setattr(c.t2.sm,'delete_model_package',fail)
    with pytest.raises(ClientError):c.delete_baseline(3)


def test_delete_api_requires_auth_and_valid_version(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ppe_pipeline_api.routers import track2_comparisons as r
    app=FastAPI();app.include_router(r.router,prefix='/track2/comparisons');client=TestClient(app)
    calls=[];monkeypatch.setattr(c,'delete_baseline',lambda version:calls.append(version) or {'deleted':True})
    assert client.delete('/track2/comparisons/baselines/3').status_code==401
    assert calls==[]
    app.dependency_overrides[r.require_session]=lambda:{}
    assert client.delete('/track2/comparisons/baselines/0').status_code==422
    assert client.delete('/track2/comparisons/baselines/3').status_code==200
    assert calls==[3]
