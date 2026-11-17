import copy
from decimal import Decimal
from types import SimpleNamespace

import pytest

from ppe_pipeline_api.chain import preparation_jobs as jobs, prepared_training, retrain_schedule


class Table:
    def __init__(self):
        self.rows = {}
    def put_item(self, Item):
        self.rows[Item['id']] = copy.deepcopy(Item)
    def query(self, **kw):
        rows = list(self.rows.values())
        if kw.get('FilterExpression'):
            tree = jobs.record_store.expression_tree(kw['FilterExpression'], {}, {})
            rows = [r for r in rows if jobs.record_store.matches(tree, r)]
        return {'Items': sorted(rows, key=lambda j:j['sort'], reverse=not kw['ScanIndexForward'])[:kw['Limit']]}


@pytest.fixture
def world(monkeypatch):
    table = Table()
    monkeypatch.setattr(jobs, 'table', lambda: table)
    monkeypatch.setattr(jobs.data, 'require_training_sql', lambda: None)
    monkeypatch.setattr(jobs.pc, 'read_config', lambda: {'central_store': {'folder': 'saved'}})
    return table


def test_submit_is_durable_and_does_not_do_preparation(world, monkeypatch):
    monkeypatch.setattr(prepared_training, 'queue_prepared', lambda **kw: pytest.fail('HTTP request ran preparation'))
    response = jobs.submit(overrides={'epochs': 25})
    assert response['job']['state'] == 'pending'
    row = world.rows[response['job']['id']]
    assert row['config']['central_store']['folder'] == 'saved'
    assert row['overrides'] == {'epochs': 25}
    assert 'config' not in jobs.latest()


def test_replacement_worker_resumes_running_job(world, monkeypatch):
    response = jobs.submit(submit='queue')
    row = world.rows[response['job']['id']]
    row['state'] = 'running'
    calls = []
    def prepare(config, **kw):
        assert config == row['config']
        kw['progress']('Preserving images and labels', 4, 8)
        assert jobs.latest()['percent'] == 47
        kw['publish']({'id': kw['entry_id'], 'dataset_version_id': 'version'})
    def publish(job, entry):
        calls.append(entry)
        job.update(state='completed', percent=100)
        world.put_item(Item=job)
    monkeypatch.setattr(prepared_training, 'queue_prepared', prepare)
    monkeypatch.setattr(jobs, 'publish', publish)
    jobs.tick.__wrapped__()
    jobs.tick.__wrapped__()
    assert len(calls) == 1 and calls[0]['id'] == row['id']
    assert jobs.latest()['state'] == 'completed'


def test_failure_is_saved_and_visible_after_reload(world, monkeypatch):
    jobs.submit()
    def fail(*a, **kw):
        raise ValueError('No prepared dataset')
    monkeypatch.setattr(prepared_training, 'queue_prepared', fail)
    jobs.tick.__wrapped__()
    assert jobs.latest()['state'] == 'failed'
    assert jobs.latest()['error'] == 'No prepared dataset'


def test_persisted_settings_restore_nested_numbers(world, monkeypatch):
    response = jobs.submit(overrides={'hyperparams': {'bgr': Decimal('0.0'), 'lr0': Decimal('0.01')}})
    got = jobs.jobs(active=True)[0]
    assert got['id'] == response['job']['id']
    values = got['overrides']['hyperparams']
    assert type(values['bgr']) is int and type(values['lr0']) is float
    from ppe_pipeline_api.training import track2_shared as t
    restored = t._undecimalize({'id':'queued', 'record_type':'queue_entry',
                               'hyperparams': {'bgr': Decimal('0.0'), 'lr0': Decimal('0.01')}})
    assert t.normalize_hyperparams(restored['hyperparams']) == {'bgr': 0, 'lr0': 0.01}


def test_queue_and_finished_status_share_one_transaction(monkeypatch):
    writes, events = [], []
    class TxTable:
        def __init__(self, name): self.name = name
        def key(self, row): return row.get('id'), ''
        def _lock(self, cur, key): pass
        def _save(self, cur, row): writes.append((self.name, copy.deepcopy(row)))
    conn = SimpleNamespace(cursor=lambda: object(), commit=lambda:events.append('commit'), close=lambda:None)
    monkeypatch.setattr(jobs.record_store, 'connection', lambda: conn)
    monkeypatch.setattr(jobs.record_store, 'Table', TxTable)
    job = {'id':'job','submit':'queue'}
    jobs.publish(job, {'id':'job','dataset_version_id':'version'})
    assert [r['record_type'] for _,r in writes[:2]] == ['queue_entry','queue_state']
    assert writes[2][1]['state'] == 'completed'
    assert events == ['commit']


def test_sql_schedule_hands_off_to_same_durable_job(monkeypatch):
    monkeypatch.setattr(jobs.record_store, 'enabled', lambda: True)
    monkeypatch.setattr(retrain_schedule, 'settings', lambda config: {})
    got=[]
    monkeypatch.setattr(jobs, 'submit', lambda config, **kw: got.append(kw) or {'ok':True})
    assert retrain_schedule._run_chain_body(config={}, trigger='schedule', submit='queue')['ok']
    assert got == [{'trigger':'schedule','submit':'queue'}]


@pytest.mark.parametrize('pull_error', [False, True])
def test_schedule_pulls_then_rebuilds_before_preserving(world, monkeypatch, pull_error):
    jobs.submit(config={'dataset_assemble':{'only':False}}, trigger='schedule', submit='queue')
    calls=[]
    monkeypatch.setattr(jobs.pc, 'active_run', lambda: None)
    monkeypatch.setattr(jobs.pc, 'new_run_id', lambda: '20261117T090000-scheduled')
    monkeypatch.setattr(jobs.pc, 'route_pending', lambda **kw: calls.append('pull') or {'errors':int(pull_error)})
    monkeypatch.setattr(retrain_schedule, 'combine_and_decay', lambda *a, **kw: calls.append('rebuild'))
    monkeypatch.setattr(prepared_training, 'queue_prepared', lambda *a, **kw: calls.append('preserve'))
    jobs.tick.__wrapped__()
    assert jobs.jobs()[0]['pull_run_id'] == '20261117T090000-scheduled'
    assert calls == (['pull'] if pull_error else ['pull','rebuild','preserve'])
    if pull_error:
        assert jobs.latest()['state'] == 'failed'
    else:
        # Restart after completed pull/rebuild: resume preservation without pulling again.
        jobs.tick.__wrapped__()
        assert calls == ['pull','rebuild','preserve','preserve']


def test_schedule_waits_for_existing_pull(world, monkeypatch):
    jobs.submit(trigger='schedule')
    monkeypatch.setattr(jobs.pc,'active_run',lambda:{'status':'running','run_id':'other'})
    monkeypatch.setattr(jobs.pc,'route_pending',lambda **kw:pytest.fail('overlapping pull'))
    monkeypatch.setattr(prepared_training,'queue_prepared',lambda *a,**kw:pytest.fail('training before pull finished'))
    jobs.tick.__wrapped__()
    assert jobs.latest()['stage'] == 'Waiting for the current data pull to finish'
