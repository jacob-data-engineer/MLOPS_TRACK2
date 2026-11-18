import json
from types import SimpleNamespace

import pytest
from ppe_pipeline_api import record_store as rs
from ppe_pipeline_api.training import submission_progress as progress


def test_bulk_files_use_four_statements_and_last_duplicate_wins(monkeypatch):
    calls = []
    conn = SimpleNamespace(cursor=lambda:conn, execute=lambda sql,args:calls.append((sql,args)),
                           commit=lambda:calls.append(('COMMIT',None)), close=lambda:None)
    monkeypatch.setattr(rs, 'connection', lambda:conn)
    table = rs.Table('track2-yolo-training-files')
    batch = rs.Batch(table)
    for i in range(100):
        batch.put_item(Item={'audit_id':'test-run','file_key':f'train#image#{i:06}',
                             'kind':'image','name':str(i),'s3_uri':f's3://test/{i}'})
    batch.put_item(Item={'audit_id':'test-run','file_key':'train#image#000000','kind':'image','name':'changed','s3_uri':'s3://test/changed'})
    batch.flush()
    assert len(calls) == 5 and calls[-1][0] == 'COMMIT'
    payloads = json.loads(calls[1][1][0])
    assert len(payloads) == 100
    assert rs.plain(payloads[0]['payload'])['name'] == 'changed'
    projection = json.loads(calls[3][1][0])
    assert projection[0]['s3_uri'] == 's3://test/changed'
    assert not batch.pending


def test_failed_projection_never_commits_and_keeps_batch_for_retry(monkeypatch):
    from ppe_pipeline_api.data import sql_projection
    commits = []
    conn = SimpleNamespace(cursor=lambda:conn,execute=lambda *args:None,
                           commit=lambda:commits.append(True),close=lambda:None)
    monkeypatch.setattr(rs,'connection',lambda:conn)
    monkeypatch.setattr(sql_projection,'project_files',lambda *args:(_ for _ in ()).throw(RuntimeError('projection failed')))
    batch=rs.Batch(rs.Table('track2-yolo-training-files'))
    batch.put_item(Item={'audit_id':'run','file_key':'f'})
    with pytest.raises(RuntimeError): batch.flush()
    assert not commits and len(batch.pending)==1


def test_progress_is_durable_and_only_selected_for_pending_entries(monkeypatch):
    rows={}
    table=SimpleNamespace(put_item=lambda Item:rows.update({Item['id']:Item}),
                          get_item=lambda Key:{'Item':rows[Key['id']]} if Key['id'] in rows else {})
    monkeypatch.setattr(rs,'enabled',lambda:True)
    monkeypatch.setattr(progress,'table',lambda:table)
    progress.write('run-a','Recording files',500,24824)
    assert progress.for_queue([{'id':'a'}])['completed']==500
    assert progress.for_queue([{'id':'b'}]) is None
