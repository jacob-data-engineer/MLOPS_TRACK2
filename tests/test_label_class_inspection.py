from io import BytesIO
from types import SimpleNamespace
from ppe_pipeline_api.data import label_class_inspection as mod


def setup(monkeypatch, bodies):
    monkeypatch.setattr(mod.pc, '_list_objects', lambda prefix, bucket: [{'Key': k} for k in bodies])
    def get(Bucket, Key):
        if bodies[Key] is None:
            raise OSError('unreadable')
        return {'Body': BytesIO(bodies[Key])}
    monkeypatch.setattr(mod.pc, 's3', SimpleNamespace(get_object=get))


def test_ids_counts_and_empty_background(monkeypatch):
    setup(monkeypatch, {'labels/a.txt': b'0 .5 .5 .2 .2\n2 .4 .4 .1 .1\n2 .2 .2 .1 .1', 'labels/b.txt': b''})
    result = mod.inspect_labels('bucket', 'labels')
    assert result['counts'] == {'0': 1, '2': 2}
    assert result['complete'] and result['scanned'] == 2


def test_invalid_and_unreadable_not_verified(monkeypatch):
    setup(monkeypatch, {'labels/a.txt': b'1.5 .5 .5 .2 .2\nnan .5 .5 .2 .2\n-1 .5 .5 .2 .2\n2', 'labels/b.txt': None})
    result = mod.inspect_labels('bucket', 'labels')
    assert result['counts'] == {}
    assert result['invalid_lines'] == 4 and result['unreadable'] == 1
    assert not result['complete']


def test_sampling_reports_partial(monkeypatch):
    setup(monkeypatch, {f'labels/{i}.txt': b'0 .5 .5 .2 .2' for i in range(5)})
    result = mod.inspect_labels('bucket', 'labels', limit=2)
    assert result['total'] == 5 and result['scanned'] == 2 and not result['complete']


def test_missing_folder(monkeypatch):
    setup(monkeypatch, {})
    result = mod.inspect_labels('bucket', 'labels')
    assert result['total'] == 0 and result['counts'] == {}
