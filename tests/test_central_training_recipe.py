import copy

import pytest

from ppe_pipeline_api.data import central_recipe as cr, prepared_training as pt, dataset_versions as dv, catalog_view
from test_prepared_training import prepared  # noqa: F401


def test_repeat_survives_cleanup_and_a_different_store_and_class_mapping(prepared, monkeypatch):
    config, s3, _ = prepared
    first = pt.freeze(config, target_classes={0: 'person'})
    expected = {m['stored_label']: s3.files[dv._uri(m['stored_label'])] for m in first['file_provenance']}
    permanent = {k: v for k, v in s3.files.items() if '/objects/replay/' in k[1]}
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'id': first['version_id']}])
    cr.cleanup_finished()
    assert all(dv._uri(m['stored_image']) not in s3.files for m in first['file_provenance'])
    assert all(s3.files[k] == v for k, v in permanent.items())
    assert dv.get(first['version_id'])['temporary_cleaned']
    # Another project selects another store and class order after this run.
    changed = copy.deepcopy(config)
    changed['central_store']['folder'] = 'different-project'
    changed['dataset_assemble']['label_classes'] = {'0': 'person', '1': 'helmet'}
    monkeypatch.setattr(pt.cs.pc, 'read_config', lambda: changed)
    monkeypatch.setattr(pt.cs, 'location', lambda *a: ('elsewhere', 'new-store', 'new-decay'))
    cr.prepare_entry({'dataset_version_id': first['version_id']})
    assert all(s3.files[dv._uri(uri)] == body for uri, body in expected.items())
    assert dv.get(first['version_id'])['training_classes'] == {'0': 'person'} or dv.get(first['version_id'])['training_classes'] == {0: 'person'}
    assert not dv.get(first['version_id'])['temporary_cleaned']


def test_classes_get_their_own_normalized_labels_while_images_are_reused(prepared):
    config, s3, _ = prepared
    first = pt.freeze(config, target_classes={0: 'person'})
    second = pt.freeze(config, target_classes={0: 'helmet'})
    for a, b in zip(first['file_provenance'], second['file_provenance']):
        assert a['central_image'] == b['central_image']
        assert a['central_label'] != b['central_label']
    assert first['central_recipe']['store_uri'] == 's3://home/combined'


@pytest.mark.parametrize('changed', [False, True])
def test_missing_or_changed_permanent_files_block_repeat_and_cleanup(prepared, monkeypatch, changed):
    config, s3, _ = prepared
    version = pt.freeze(config)
    key = dv._uri(version['file_provenance'][0]['central_image'])
    if changed:
        s3.files[key] = b'changed'
    else:
        del s3.files[key]
    with pytest.raises(ValueError, match='missing or changed'):
        cr.restore(version)
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'id': version['version_id']}])
    with pytest.raises(ValueError, match='missing or changed'):
        cr.cleanup_finished()
    assert all(dv._uri(m['stored_image']) in s3.files for m in version['file_provenance'])


def test_cleanup_refuses_a_non_temporary_prefix(prepared, monkeypatch):
    config, s3, _ = prepared
    version = pt.freeze(config)
    recipe = {**version['central_recipe'], 'folder': 'combined'}
    dv.mark(version['version_id'], central_recipe=recipe)
    before = dict(s3.files)
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'id': version['version_id']}])
    with pytest.raises(ValueError, match='Invalid temporary'):
        cr.cleanup_finished()
    assert s3.files == before


def test_export_reads_recorded_central_files_after_temporary_dataset_is_deleted(prepared, monkeypatch):
    config, s3, _ = prepared
    version = pt.freeze(config)
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'id': version['version_id']}])
    cr.cleanup_finished()
    for member in version['file_provenance']:
        for kind in ('image', 'label'):
            body, note = catalog_view._frozen(member[f'central_{kind}'], member[f'stored_{kind}'], member[f'central_{kind}_etag'])
            assert body is not None and not note
    member = version['file_provenance'][0]
    s3.files[dv._uri(member['central_label'])] = b'changed label'
    body, note = catalog_view._frozen(member['central_label'], member['stored_label'], member['central_label_etag'])
    assert body is None and 'changed' in note


@pytest.mark.parametrize('state', ['Pending', 'Preparing', 'Training', 'Comparing', 'Needs attention'])
def test_experiment_working_files_retained_until_terminal(prepared, monkeypatch, state):
    from ppe_pipeline_api.chain import training_variants
    config, s3, _ = prepared
    version = pt.freeze(config)
    dv.mark(version['version_id'], experiment_owner='experiment-1')
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'id': version['version_id']}])
    monkeypatch.setattr(training_variants, 'read', lambda ident: {'state': state})
    before = dict(s3.files)
    cr.cleanup_finished()
    assert s3.files == before
    monkeypatch.setattr(training_variants, 'read', lambda ident: {'state': 'Completed'})
    cr.cleanup_finished()
    assert dv.get(version['version_id'])['temporary_cleaned']
    # Retrying after cleanup reconstructs exactly the same shared working folder.
    cr.prepare_entry({'dataset_version_id': version['version_id']})
    assert all(dv._uri(m['stored_image']) in s3.files for m in version['file_provenance'])
