import pytest
from botocore.exceptions import ClientError
from ppe_pipeline_api.training import resource_wait as rw, submission_progress as sp


@pytest.mark.parametrize('code,expected', [('ResourceLimitExceeded', True), ('InsufficientCapacityError', True), ('AccessDenied', False), ('ValidationException', False)])
def test_error_classification(code, expected):
    assert rw.is_resource_shortage(ClientError({'Error': {'Code': code, 'Message': code}}, 'StartTrainingJob')) is expected


def test_waiting_only_for_actual_resource_waits():
    idle = {'known': True, 'busy': False, 'jobs': []}
    pending = {'started': True, 'pending': [{'id': 'a'}]}
    assert not rw.describe(pending, idle, idle)['waiting']
    busy = {**idle, 'busy': True}
    assert rw.describe(pending, busy, idle)['waiting']
    assert not rw.describe({**pending, 'started': False}, busy, idle)['waiting']
    assert not rw.describe({}, busy, idle)['waiting']
    allocation = {**idle, 'jobs': [{'name': 'ours', 'own_pipeline': True, 'started': False}]}
    assert rw.describe({}, allocation, idle)['waiting']
    assert 'CPU' in rw.describe({}, idle, allocation)['detail']
    assert not rw.describe({}, {**allocation, 'known': False}, idle)['waiting']
    assert not rw.describe({}, {**idle, 'jobs': [{'name': 'foreign', 'own_pipeline': False, 'started': False}]}, idle)['waiting']
    assert rw.describe({**pending, 'submission_progress': {'resource_wait': True}}, idle, idle)['waiting']


def test_wait_reason_persisted_and_cleared_on_retry(monkeypatch):
    rows = {}
    class Table:
        def put_item(self, Item):
            rows[Item['id']] = Item
        def get_item(self, Key):
            return {'Item': rows.get(Key['id'])}
    monkeypatch.setattr(sp, 'table', Table)
    monkeypatch.setattr(sp.record_store, 'enabled', lambda: True)
    sp.write('run-a', 'Waiting for resources', resource_wait=True, detail='Quota reached')
    assert sp.for_queue([{'id': 'a'}])['detail'] == 'Quota reached'
    sp.write('run-a', 'Checking resources')
    assert sp.for_queue([{'id': 'a'}])['resource_wait'] is False


def test_live_status_distinguishes_idle_capacity_wait_and_access_failure(monkeypatch):
    from ppe_pipeline_api.routers import track2_retention as route
    monkeypatch.setattr(route.rs, 'chain_progress', lambda: {})
    monkeypatch.setattr(route, '_pull_status', lambda: {})
    monkeypatch.setattr(route.t2, 'queue_snapshot', lambda: {})
    monkeypatch.setattr(route.t2.sm, 'list_training_jobs', lambda **kw: {'TrainingJobSummaries': []})
    monkeypatch.setattr(route.t2.sm, 'list_processing_jobs', lambda **kw: {'ProcessingJobSummaries': []})
    assert route.chain_status()['resources']['waiting'] is False
    monkeypatch.setattr(route.t2.sm, 'list_processing_jobs', lambda **kw: {'ProcessingJobSummaries': [{'ProcessingJobName': 'track2-compare-a'}]})
    monkeypatch.setattr(route.t2.sm, 'describe_processing_job', lambda **kw: {'ProcessingResources': {'ClusterConfig': {'InstanceType': 'ml.m5.xlarge'}}})
    assert route.chain_status()['resources']['waiting'] is True
    def denied(**kw):
        raise RuntimeError('Access denied')
    monkeypatch.setattr(route.t2.sm, 'list_training_jobs', denied)
    assert route.chain_status()['gpu']['known'] is False
