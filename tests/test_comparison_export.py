import pytest
from ppe_pipeline_api.data import catalog_view as cv
from ppe_pipeline_api.training import metrics_export as mx


def prepare(monkeypatch, record=None):
    monkeypatch.setattr(cv, '_destination', lambda *a: ('out', 'exports'))
    monkeypatch.setattr(cv.catalog, 'ensure_schema_once', lambda: None)
    monkeypatch.setattr(cv, '_plan', lambda *a: ([('images/train/a.jpg', 's3://in/a.jpg', {'role': 'train'})], {0: 'helmet'}, 'run-1'))
    monkeypatch.setattr(cv, '_rows', lambda *a: [{'model_package_arn': 'arn:chosen', 'record': record or {}}])


def test_wrong_run_cannot_export_a_different_model(monkeypatch):
    prepare(monkeypatch)
    with pytest.raises(ValueError, match='does not match'):
        cv.start_export('run', 'r', model_package_arn='arn:other')


def test_requested_checkpoint_cannot_be_silently_omitted(monkeypatch):
    prepare(monkeypatch)
    with pytest.raises(ValueError, match='No .pt'):
        cv.start_export('run', 'r', include_model=True)


def test_missing_checkpoint_stops_before_background_export(monkeypatch):
    prepare(monkeypatch, {'weights_uri': 's3://in/model.pt'})
    class S3:
        def head_object(self, **kwargs): raise RuntimeError('missing')
    monkeypatch.setattr(cv.storage, 's3', S3())
    with pytest.raises(ValueError, match='missing or inaccessible'):
        cv.start_export('run', 'r', include_model=True)


def test_model_and_dataset_options_work_without_metadata(monkeypatch):
    prepare(monkeypatch, {'weights_uri': 's3://in/model.pt'})
    class S3:
        def head_object(self, **kwargs): return {}
    monkeypatch.setattr(cv.storage, 's3', S3())
    class DB:
        def cursor(self): return self
        def execute(self, *args): pass
        def commit(self): pass
        def close(self): pass
    monkeypatch.setattr(cv.catalog, 'connect', DB)
    captured = []
    class Thread:
        def __init__(self, **kwargs): captured.append(kwargs['args'])
        def start(self): cv._building.release()
    monkeypatch.setattr(cv.threading, 'Thread', Thread)
    cv.start_export('run', 'r', include_images=False, include_model=True, include_metadata=False, model_package_arn='arn:chosen')
    entries = captured[0][4]
    assert [path for path, _, _ in entries if path] == ['models/run-1.pt']
    assert entries[-1][1] == 's3://in/model.pt'
    assert captured[0][-1] is None


def test_manual_export_stays_pinned_when_a_better_model_appears(monkeypatch):
    rows = [{'model_package_arn': 'arn:chosen', 'version': '1', 'metrics': {'mAP50': .5}},
            {'model_package_arn': 'arn:new', 'version': '2', 'metrics': {'mAP50': .9}}]
    monkeypatch.setattr(mx, 'group_versions', lambda *a: rows)
    exported = []
    monkeypatch.setattr(mx, 'export_checkpoint', lambda row, **kw: exported.append(row) or {})
    monkeypatch.setattr(mx, 'describe_pick', lambda row, metric: row)
    mx.run_export('group', 'mAP50', destination='local', model_package_arn='arn:chosen')
    assert exported == [rows[0]]
    with pytest.raises(ValueError):
        mx.run_export('group', 'mAP50', destination='local', model_package_arn='arn:gone')
