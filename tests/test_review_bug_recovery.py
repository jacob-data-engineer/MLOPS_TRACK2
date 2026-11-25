import copy
import threading
from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from ppe_pipeline_api import record_store
from ppe_pipeline_api.chain import preparation_jobs as jobs, retrain_schedule as schedule
from ppe_pipeline_api.data import catalog_view as cv


@pytest.mark.parametrize('failure', ['before_write', 'lost_response'])
def test_scheduled_submission_retries_without_duplicate_jobs(monkeypatch, failure):
    rows = {}
    calls = []

    class Table:
        def get_item(self, Key):
            return {'Item': copy.deepcopy(rows[(Key['kind'], Key['sort'])])} if (Key['kind'], Key['sort']) in rows else {}

        def put_item(self, Item, **kw):
            calls.append(1)
            key = (Item['kind'], Item['sort'])
            if len(calls) == 1 and failure == 'before_write':
                raise RuntimeError('temporary database outage')
            if key in rows:
                raise ClientError({'Error': {'Code': 'ConditionalCheckFailedException'}}, 'PutItem')
            assert kw['ConditionExpression'] == 'attribute_not_exists(#k)'
            rows[key] = copy.deepcopy(Item)
            if len(calls) == 1:
                raise RuntimeError('response lost after commit')

    table = Table()
    monkeypatch.setattr(record_store, 'enabled', lambda: True)
    monkeypatch.setattr(record_store, 'Table', lambda name: table)
    monkeypatch.setattr(jobs, 'table', lambda: table)
    monkeypatch.setattr(jobs.data, 'require_training_sql', lambda: None)
    monkeypatch.setattr(schedule, 'due_training_run', lambda c: '2026-11-25T02:00:00+00:00')
    monkeypatch.setattr(schedule, 'automatic', lambda c: c)
    monkeypatch.setattr(schedule.pc, 'claim_once', lambda *a: pytest.fail('Claim must be atomic with the job'))
    assert not schedule.check_training_schedule({})['started']
    assert schedule.check_training_schedule({})['started']
    row = next(iter(rows.values()))
    row['state'] = 'running'
    row['percent'] = 50
    schedule.check_training_schedule({})
    assert len(rows) == 1 and row['state'] == 'running' and row['percent'] == 50


def test_saving_schedule_does_not_run_just_passed_slot(monkeypatch):
    monkeypatch.setattr(record_store, 'enabled', lambda: True)
    monkeypatch.setattr(record_store, 'Table', lambda name: SimpleNamespace(get_item=lambda **kw: {'Item': {'kind': 'claim#training_schedule'}}))
    monkeypatch.setattr(schedule, 'due_training_run', lambda c: 'slot')
    monkeypatch.setattr(jobs, 'submit', lambda **kw: pytest.fail('Saved past slot must stay suppressed'))
    assert not schedule.check_training_schedule({})['started']


def test_reconcile_only_expires_stale_unfinished_exports(monkeypatch):
    statements, events = [], []
    cursor = SimpleNamespace(execute=lambda sql, params: statements.append((sql, params)), rowcount=2)
    conn = SimpleNamespace(cursor=lambda: cursor, commit=lambda: events.append('commit'), close=lambda: events.append('close'))
    monkeypatch.setattr(cv.catalog, 'ensure_schema_once', lambda: None)
    monkeypatch.setattr(cv.catalog, 'connect', lambda: conn)
    assert cv.reconcile_exports() == 2
    sql, params = statements[0]
    assert "state IN ('building','stopping')" in sql
    assert "COALESCE(updated_at, created_at) < now() - interval '10 minutes'" in sql
    assert 'retry' in params[0] and 'DELETE' not in sql
    assert events == ['commit', 'close']


def test_export_owner_heartbeat_stops_when_build_finishes(monkeypatch):
    beat = threading.Event()
    real_event = threading.Event
    class FastEvent:
        def __init__(self): self.event = real_event()
        def wait(self, timeout): return self.event.wait(0.01)
        def set(self): self.event.set()
    # Only replace the owner's event factory; Thread itself also uses Event.
    class ThreadFactory:
        Event = FastEvent
        Thread = threading.Thread
    monkeypatch.setattr(cv, 'threading', ThreadFactory)
    updates = []
    def finish(ident, sql, params):
        updates.append((ident, sql, params))
        beat.set()
    monkeypatch.setattr(cv, '_finish', finish)
    monkeypatch.setattr(cv, '_build', lambda *a: beat.wait(2) or pytest.fail('No owner heartbeat'))
    cv._run_export('export-one')
    assert updates[0][0] == 'export-one'
    assert "state IN ('building','stopping')" in updates[0][1]
    assert not any(t.name == 'export-heartbeat-export-one' for t in threading.enumerate())
