"""Saved SQL data can seed a different run without changing the original recipe."""
import copy

import pytest

from test_prepared_training import prepared
from ppe_pipeline_api import data
from ppe_pipeline_api.data import prepared_training as pt, saved_training as saved, dataset_versions as dv
from ppe_pipeline_api.chain import prepared_training as chain


@pytest.fixture
def source(prepared, monkeypatch):
    config, s3, table = prepared
    monkeypatch.setattr(saved, 'require_sql', lambda: None)
    monkeypatch.setattr(saved, 'verify_members', lambda v: None)
    version = pt.freeze(config)
    return version, s3, table


def test_saved_data_remaps_classes_after_working_files_are_gone(source, monkeypatch, tmp_path):
    original, s3, table = source
    before = copy.deepcopy(original)
    for key in list(s3.files):
        if key[1].startswith(('combined_decay/', original['central_recipe']['folder'])):
            del s3.files[key]
    monkeypatch.setattr(pt.cs, 'location', lambda *a: pytest.fail('read current store'))
    result = saved.prepare(original['version_id'], target_classes={0:'person',1:'helmet'})
    assert result['version_id'] != original['version_id']
    assert result['central_recipe']['folder'] != original['central_recipe']['folder']
    assert result['central_recipe']['store_uri'] == 's3://home/combined'
    assert result['source_dataset_version_id'] == original['version_id']
    assert result['train_images'] == result['val_images'] == 1
    assert result['classes'] == ['person', 'helmet']
    for member in result['file_provenance']:
        body = s3.files[dv._uri(member['stored_label'])]
        assert body.startswith(b'1 ' if member['split'] == 'train' else b'0 ')
        assert s3.files[dv._uri(member['central_label'])] == body
    assert dv.get(original['version_id']) == before
    from test_dataset_versions import _download
    import train
    local = _download(monkeypatch, s3, tmp_path, result['train_data_uris'], result['data_yaml_uri'])
    report = train.preflight_dataset(local)
    assert report['train']['images'] == report['val']['images'] == 1


@pytest.mark.parametrize('problem', ['class-set', 'missing', 'changed', 'legacy'])
def test_bad_saved_inputs_do_not_create_a_new_version(source, problem):
    version, s3, table = source
    target = {0:'helmet',1:'person'}
    if problem == 'class-set':
        target = {0:'helmet'}
    elif problem == 'legacy':
        del table.rows[version['version_id']]['central_recipe']
    else:
        key = dv._uri(version['file_provenance'][0]['central_image'])
        if problem == 'missing': del s3.files[key]
        else: s3.files[key] = b'changed'
    count = len(table.rows)
    with pytest.raises(ValueError):
        saved.prepare(version['version_id'], target_classes=target)
    assert len(table.rows) == count


def test_saved_queue_uses_current_model_settings_and_no_decay_cleanup(monkeypatch):
    from ppe_pipeline_api.chain import retrain_schedule as rs
    monkeypatch.setattr(rs, 'settings', lambda config: {})
    monkeypatch.setattr(data, 'require_training_sql', lambda: None)
    t = chain.training
    monkeypatch.setattr(t, 'get_detector', lambda slug: {'classes': {0:'helmet'}, 'slug':slug})
    monkeypatch.setattr(t, 'detector_group_name', lambda *a:'group')
    monkeypatch.setattr(t, 'model_registry_summary', lambda *a:{'packages':[]})
    monkeypatch.setattr(t, 'scratch_refusal', lambda *a:'')
    monkeypatch.setattr(t, 'resolve_base_model_source', lambda **kw:('s3://base/current.pt','scratch'))
    monkeypatch.setattr(data, 'preserve_training_base', lambda uri:{'key':'current.pt'})
    monkeypatch.setattr(data, 'freeze_prepared_training', lambda *a,**kw:pytest.fail('used current decay'))
    calls = []
    monkeypatch.setattr(data, 'prepare_saved_training', lambda ident,**kw:(calls.append((ident,kw)) or
        {'version_id':'new','train_images':2,'val_images':1}))
    monkeypatch.setattr(data, 'version_run_fields', lambda version:{'dataset_version_id':version['version_id']})
    monkeypatch.setattr(data, 'mark_version', lambda *a,**kw:None)
    entries = []
    comparison = {'enabled':True,'baseline_arn':'baseline-v1','other_arns':[]}
    pinned = {**comparison,'models':[{'uri':'s3://base/pinned.pt','etag':'original'}]}
    monkeypatch.setattr(t,'pin_comparison_selection',lambda value:pinned if value == comparison else pytest.fail('comparison selection lost'))
    chain.queue_prepared({}, overrides={'dataset_version_id':'old','detector_slug':'helmet',
        'architecture':'yolo11s','base_version':'scratch','epochs':7,'img_size':640,'batch':2,
        'seed':12,'patience':4,'hyperparams':{'lr0':0.002},'delete_decay_after_run':True,
        'comparison':comparison}, publish=entries.append)
    assert calls[0][0] == 'old'
    entry = entries[0]
    assert entry['comparison'] == pinned
    assert entry['epochs'] == 7 and entry['hyperparams']['lr0'] == .002
    assert entry['base_model_key'] == 'current.pt' and entry['dataset_version_id'] == 'new'
    assert entry['delete_decay_after_run'] is False and 'pipeline' not in entry
