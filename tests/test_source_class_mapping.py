from io import BytesIO
from types import SimpleNamespace
import pytest
from ppe_pipeline_api.data import source_class_mapping as m

NAMES = {'0': 'helmet', '1': 'person'}
RECORD = {'target_classes': NAMES, 'ids': {'0': '1', '2': '0', '1': None}}


def config():
    return {'dataset_assemble': {'label_classes': NAMES},
            'golden_set': {'source': 'gold/images', 'bucket': 'source'},
            'source_class_mappings': {'s3://source/gold': RECORD}}


def test_rewrite_keeps_coordinates_and_empty_background():
    body, counts = m.rewrite('0 .1 .2 .3 .4\n2 .2 .3 .4 .5\n', RECORD['ids'])
    assert body == b'1 .1 .2 .3 .4\n0 .2 .3 .4 .5\n'
    assert counts == {'0': 1, '2': 1}
    assert m.rewrite('', RECORD['ids'])[0] == b''


@pytest.mark.parametrize('text', ['1 .1 .2 .3 .4', '9 .1 .2 .3 .4', '1.2 .1 .2 .3 .4', 'nan .1 .2 .3 .4', '0 .1'])
def test_unmapped_unused_and_malformed_stop(text):
    with pytest.raises(ValueError):
        m.rewrite(text, RECORD['ids'])


def test_stale_target_names_rejected():
    with pytest.raises(ValueError, match='classes changed'):
        m.validate({'s3://b/f': RECORD}, {'0': 'person'})


def test_mixed_pull_maps_rejected():
    c = config()
    c['picked'] = {'train_images': ['a/images', 'b/images']}
    c['source_bucket'] = 'source'
    c['source_class_mappings']['s3://source/a'] = RECORD
    with pytest.raises(ValueError, match='every pulled source'):
        m.preparation_maps(c, 'store_decay', [{'images': 'store_decay/weeks/week_1/images/train'}], [])


def test_only_prepared_copies_written_and_originals_read():
    reads, writes = [], []
    def get(**kw):
        reads.append(kw)
        return {'Body': BytesIO(b'0 .1 .2 .3 .4\n'), 'Metadata': {'captured_at': '2026-09-03T12:00:00+00:00'}}
    s3 = SimpleNamespace(get_object=get, put_object=lambda **kw: writes.append(kw))
    args = dict(config=config(), bucket='dest', decayed='store_decay',
        pairs=[('dest', 'store/golden/labels/train/a.txt', 'store_decay/golden/labels/train/a.txt')],
        train=[{'images': 'store_decay/golden/images/train'}], val=[], s3=s3,
        checkpoint=lambda: None, progress=lambda *a: None)
    m.apply(**args)
    assert reads[0]['Key'] == 'store/golden/labels/train/a.txt'
    assert all(w['Key'].startswith('store_decay/') for w in writes)
    assert writes[0]['Body'].startswith(b'1 ')
    assert writes[0]['Metadata']['captured_at'] == '2026-09-03T12:00:00+00:00'
    assert any(w['Key'].endswith('class_mapping.json') for w in writes)
    m.apply(**args)
    assert writes[4]['Body'].startswith(b'1 ')  # no cumulative remapping


def test_full_label_validation_before_any_rewrite():
    writes = []
    s3 = SimpleNamespace(get_object=lambda **kw: {'Body': BytesIO(b'9 .1 .2 .3 .4')},
                         put_object=lambda **kw: writes.append(kw))
    with pytest.raises(ValueError, match='Class ID 9'):
        m.apply(config(), 'dest', 'store_decay',
          [('dest', 'store/golden/labels/a.txt', 'store_decay/golden/labels/a.txt')],
          [{'images': 'store_decay/golden/images'}], [], s3=s3,
          checkpoint=lambda: None, progress=lambda *a: None)
    assert writes == []


def test_mappings_in_saved_and_training_settings():
    from ppe_pipeline_api import pull_control as pc
    assert 'source_class_mappings' in pc.SCOPES['dp']
    assert 'source_class_mappings' in pc.TRAINING_DATA_KEYS
