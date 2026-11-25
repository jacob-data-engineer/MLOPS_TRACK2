"""Prepared inputs survive rebuilds, retain class meaning and fail closed."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest
import yaml

from ppe_pipeline_api import data, pull_control as pc, storage
from ppe_pipeline_api.data import prepared_training as pt, dataset_versions as dv
from ppe_pipeline_api.chain import prepared_training as chain
from ppe_pipeline_api.training.reproducibility import pin_pipeline
from test_dataset_versions import FakeTable, _download
from test_review_f1_f4_f5_dataset_contract import FakeS3


class ConditionalS3(FakeS3):
    def delete_objects(self, Bucket, Delete):
        for obj in Delete['Objects']:
            self.files.pop((Bucket, obj['Key']), None)
        return {}

    def delete_object(self, Bucket, Key):
        self.files.pop((Bucket, Key), None)

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.files:
            from botocore.exceptions import ClientError
            raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
        body = self.files[Bucket, Key]
        return {'ETag': '"' + hashlib.sha256(body).hexdigest() + '"', 'ContentLength': len(body)}

    def get_object(self, Bucket, Key, IfMatch=None):
        head = self.head_object(Bucket, Key)
        if IfMatch and IfMatch != head['ETag']:
            raise ValueError('Changed object')
        return {**head, **super().get_object(Bucket, Key)}

    def copy_object(self, Bucket, Key, CopySource, CopySourceIfMatch):
        source = self.get_object(**CopySource, IfMatch=CopySourceIfMatch)
        self.put_object(Bucket=Bucket, Key=Key, Body=source['Body'].read())

    def get_paginator(self, name):
        original = super().get_paginator(name)
        def paginate(**kwargs):
            for page in original.paginate(**kwargs):
                for obj in page['Contents']:
                    obj.update(self.head_object(kwargs['Bucket'], obj['Key']))
                yield page
        return SimpleNamespace(paginate=paginate)


@pytest.fixture
def prepared(monkeypatch):
    config = {'central_store': {'folder': 'combined', 'bucket': 'home'},
              'dataset_assemble': {'label_classes': {'0': 'helmet', '1': 'person'}}}
    objects = {('home', 'combined_decay/_decay.json'): json.dumps({'status': 'built', 'failed': 0})}
    for source, split, names in [('a', 'train', ['helmet', 'person']), ('b', 'val', ['person', 'helmet'])]:
        root = f'combined_decay/{source}'
        objects['home', f'{root}/images/{split}/{source}.jpg'] = b'image-' + source.encode()
        objects['home', f'{root}/labels/{split}/{source}.txt'] = '0 .5 .5 .2 .2\n'
        objects['home', f'{root}/data.yaml'] = yaml.safe_dump({'names': dict(enumerate(names))})
    s3 = ConditionalS3(objects)
    table = FakeTable()
    monkeypatch.setattr(storage, 's3', s3)
    monkeypatch.setattr(storage, 'BUCKET', 'home')
    monkeypatch.setattr(pc, 's3', s3)
    monkeypatch.setattr(pc, 'destination_bucket', lambda config=None: 'home')
    monkeypatch.setattr(pc, '_list_objects', lambda prefix, bucket='', **kw: [
        {'Key': k, 'Size': len(v)} for (b, k), v in sorted(s3.files.items())
        if b == (bucket or 'home') and k.startswith(prefix)])
    monkeypatch.setattr(dv, '_table', lambda: table)
    monkeypatch.setattr(pt, 'require_sql', lambda: None)
    monkeypatch.setattr(pt, 'verify_members', lambda version: None)
    monkeypatch.setattr(pt.central_recipe, 'record_members', lambda *args: None)
    return config, s3, table


def test_preserved_files_train_after_prepared_source_is_deleted(prepared, monkeypatch, tmp_path):
    config, s3, table = prepared
    before = copy.deepcopy(s3.files)
    version = pt.freeze(config)
    assert version['immutable_files'] is True
    assert version['train_images'] == version['val_images'] == 1
    assert all(s3.files[k] == v for k, v in before.items())
    provenance = version['file_provenance']
    val = next(p for p in provenance if p['split'] == 'val')
    bucket, key = dv._uri(val['stored_label'])
    assert s3.files[bucket, key].startswith(b'1 '), 'source class 0 person must become target class 1'
    for key in before:
        del s3.files[key]
    local = _download(monkeypatch, s3, tmp_path, version['train_data_uris'], version['data_yaml_uri'])
    import train
    report = train.preflight_dataset(local)
    assert report['train']['images'] == report['val']['images'] == 1
    assert table.rows[version['version_id']]['file_provenance'] == provenance


def test_rebuild_during_copy_refuses_version(prepared, monkeypatch):
    config, s3, table = prepared
    original = s3.copy_object
    def change(**kwargs):
        original(**kwargs)
        s3.files['home', 'combined_decay/_decay.json'] = b'{"status":"building"}'
    monkeypatch.setattr(s3, 'copy_object', change)
    with pytest.raises(ValueError, match='changed while preserving'):
        pt.freeze(config)
    assert not table.rows


def test_model_class_selection_keeps_images_and_normalizes_export_labels(prepared):
    config, s3, _ = prepared
    version = pt.freeze(config, target_classes={0: 'person'})
    assert version['classes'] == ['person']
    assert version['train_images'] == version['val_images'] == 1
    for item in version['file_provenance']:
        content = s3.files[dv._uri(item['stored_label'])]
        assert content == (b'' if item['split'] == 'train' else b'0 .5 .5 .2 .2\n')


def test_failed_preparation_is_not_trainable(prepared):
    config, s3, table = prepared
    s3.files['home', 'combined_decay/_decay.json'] = b'{"status":"built","failed":1}'
    with pytest.raises(ValueError, match='Complete Combine'):
        pt.freeze(config)
    assert not table.rows


def test_sql_is_mandatory(monkeypatch):
    monkeypatch.setattr(pt.record_store, 'enabled', lambda: False)
    with pytest.raises(ValueError, match='PostgreSQL'):
        pt.require_sql()


def test_missing_sql_file_manifest_blocks_training(monkeypatch):
    from ppe_pipeline_api.data import catalog_view
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'split': 'train', 'files': 1}])
    with pytest.raises(ValueError, match='SQL dataset file list is incomplete'):
        pt.verify_members({'version_id': 'v', 'train_images': 1, 'val_images': 1})


def test_repeated_pipeline_reuses_inputs_and_pins_version():
    old = {'TrainingDatasetUri': 's3://b/preserved', 'Epochs': '25', 'RunId': 'old'}
    definition = json.dumps({'Parameters': [{'Name': k} for k in old]})
    calls = []
    sm = SimpleNamespace(describe_pipeline=lambda **kw: (calls.append(kw) or {'PipelineDefinition': definition}))
    saved = {'name': 'original', 'version_id': 7, 'definition': definition, 'recorded_parameters': old}
    pipeline, sent = pin_pipeline(sm, 'changed', saved, {**old, 'Epochs': '99', 'RunId': 'new'})
    assert sent == {**old, 'RunId': 'new'}
    assert calls == [{'PipelineName': 'original', 'PipelineVersionId': 7}]
    assert pipeline['version_id'] == 7


def test_first_run_pins_mutable_evaluation_dependencies(monkeypatch):
    from ppe_pipeline_api.training import reproducibility as rp
    monkeypatch.setenv('APP_BUILD_TAG', 'deployed-build')
    original = {'Parameters': [{'Name': 'RunId'}], 'Steps': [{'Arguments': {
        'AppSpecification': {'ImageUri': '123.dkr.ecr.ap-southeast-1.amazonaws.com/track2-yolo-train:latest'},
        'ProcessingInputs': [{'S3Input': {'S3Uri': 's3://b/code/evaluate.py'}}]}}]}
    monkeypatch.setattr(rp.boto3, 'client', lambda *a, **kw: SimpleNamespace(
        describe_images=lambda **kw: {'imageDetails': [{'imageDigest': 'sha256:abc'}]}))
    monkeypatch.setattr(data, 'preserve_training_base', lambda uri: {'uri': 's3://b/_training_inputs/code/evaluate.py'})
    updated = []
    sm = SimpleNamespace(
        list_pipeline_versions=lambda **kw: {'PipelineVersionSummaries': [{'PipelineVersionId': 4}]},
        describe_pipeline=lambda **kw: {'PipelineDefinition': json.dumps(original)},
        update_pipeline=lambda **kw: (updated.append(kw) or {'PipelineVersionId': 5}))
    pipeline, params = rp.pin_pipeline(sm, 'training', None, {'RunId': 'new'})
    assert pipeline['version_id'] == 5
    assert pipeline['app_build'] == 'deployed-build'
    assert '@sha256:abc' in pipeline['definition']
    assert 's3://b/_training_inputs/code/evaluate.py' in pipeline['definition']
    assert len(pipeline['dependencies']) == 2 and len(updated) == 1


def test_repeat_queues_saved_entry_without_repreparing(monkeypatch):
    template = {'dataset_version_id': 'frozen', 'epochs': 25, 'base_model_key': 'saved/model.pt',
                'reproducible': True, 'pipeline': {'version_id': 7}}
    monkeypatch.setattr(data, 'repeat_training_entry', lambda ident: copy.deepcopy(template))
    queued = []
    monkeypatch.setattr(chain.training, 'queue_add', queued.append)
    result = chain.repeat_run('original-run')
    assert result['entry_id'] == queued[0]['id']
    assert all(queued[0][k] == v for k, v in template.items())


def test_reproducible_export_uses_complete_version_members(monkeypatch):
    from ppe_pipeline_api.data import catalog_view as cv
    monkeypatch.setattr(cv.catalog, 'ensure_schema_once', lambda: None)
    def rows(sql, params):
        if 'FROM training_runs' in sql:
            return [{'run_number': 5, 'version_id': 'frozen', 'record': {'reproducibility': {'entry': {}}}}]
        if 'FROM dataset_versions' in sql:
            return [{'version_number': 3, 'class_names': ['helmet']}]
        if 'FROM dataset_members' in sql:
            return [{'image_uri': 's3://b/preserved/images/train/a.jpg', 'label_uri': 's3://b/preserved/labels/train/a.txt',
                     'stored_image_uri': '', 'stored_label_uri': '', 'split': 'train', 'included': True,
                     'reason': '', 'decay_weight': 1, 'week': ''}]
        raise AssertionError(sql)
    monkeypatch.setattr(cv, '_rows', rows)
    entries, names, title = cv._plan('run', 'run')
    assert len(entries) == 2 and names == ['helmet'] and title == 'run-5'
    assert entries[1][0] == 'labels/train/a.txt'


@pytest.mark.parametrize('sql_fails', [True, False])
def test_sql_audit_must_succeed_before_gpu_submission(monkeypatch, sql_fails):
    from ppe_pipeline_api.training import track2_shared as t
    from ppe_pipeline_api.training import reproducibility
    calls = []
    monkeypatch.setattr(t, 'normalize_hyperparams', lambda x: {})
    monkeypatch.setattr(t, 'validate_training_params', lambda *a, **kw: (1, 640, 1, 42))
    monkeypatch.setattr(t, 'ensure_model_package_group', lambda *a: None)
    monkeypatch.setattr(t, 'resolve_base_model_source', lambda **kw: ('s3://home/preserved/model.pt', 'file'))
    monkeypatch.setattr(data, 'require_training_sql', lambda: None)
    monkeypatch.setattr(data, 'record_training_submission', lambda *a, **kw: {'dataset_version_id': 'frozen', **kw})
    monkeypatch.setattr(reproducibility, 'pin_pipeline', lambda sm, name, saved, sent: ({'name': name, 'version_id': 7}, sent))
    def audit(**kw):
        assert kw['strict'] and kw['reproducibility']['dataset_version_id'] == 'frozen'
        calls.append('sql')
        if sql_fails:
            raise RuntimeError('SQL unavailable')
        return {'id': 'audit'}
    monkeypatch.setattr(t, 'write_training_audit', audit)
    def start(**kw):
        assert calls == ['sql'] and kw['PipelineVersionId'] == 7
        calls.append('gpu')
        return {'PipelineExecutionArn': 'execution'}
    monkeypatch.setattr(t, 'sm', SimpleNamespace(start_pipeline_execution=start))
    monkeypatch.setattr(t, 'training_audit_table', lambda: SimpleNamespace(update_item=lambda **kw: None))
    persisted = []
    monkeypatch.setattr(t, '_persist_queue_entry', lambda entry: persisted.append(copy.deepcopy(entry)))
    kwargs = dict(dataset_folder='preserved', eval_dataset_folder='', epochs=1, img_size=640,
                  batch=1, seed=42, notes='', architecture='yolo11s', base_model_key='preserved/model.pt',
                  replay_entry={'id': 'durable-job', 'dataset_version_id': 'frozen', 'reproducible': True})
    if sql_fails:
        with pytest.raises(RuntimeError, match='SQL unavailable'):
            t.start_custom_training_run(**kwargs)
        assert calls == ['sql']
    else:
        first = t.start_custom_training_run(**kwargs)
        assert calls == ['sql', 'gpu']
        calls.clear()
        second = t.start_custom_training_run(**kwargs)
        assert first['run_id'] == second['run_id']
        assert len(persisted) == 2 and persisted[0]['pipeline'] == persisted[1]['pipeline']


def test_retried_audit_never_overwrites_pending_queue_entry(monkeypatch):
    from ppe_pipeline_api.training import track2_shared as t
    rows = {'durable-job': {'id': 'durable-job', 'record_type': 'queue_entry'}}
    monkeypatch.setattr(t, 'training_audit_table', lambda: SimpleNamespace(
        put_item=lambda Item: rows.update({Item['id']: Item})))
    monkeypatch.setattr(t, '_run_artifacts', lambda *a: {})
    monkeypatch.setattr(t, '_dataset_audit_block', lambda *a: {})
    monkeypatch.setattr(t, '_evaluates_on_its_training_images', lambda *a: False)
    for _ in range(2):
        audit = t.write_training_audit(build_id='', generation='26', size='s', epochs=1,
            img_size=640, batch=16, seed=42, dataset_folder='preserved',
            eval_dataset_folder='', base_model_folder='', notes='', strict=True,
            reproducibility={'dataset_version_id':'frozen', 'entry': {'id':'durable-job'}})
        assert audit['id'] == 'run-durable-job'
        assert rows['durable-job']['record_type'] == 'queue_entry'
    assert len(rows) == 2


def test_queue_sql_failure_does_not_leave_runnable_entry(monkeypatch):
    from ppe_pipeline_api.training import track2_shared as t
    monkeypatch.setattr(t, '_training_queue', [])
    def fail(**kw):
        raise RuntimeError('SQL unavailable')
    monkeypatch.setattr(t, 'training_audit_table', lambda: SimpleNamespace(put_item=fail))
    with pytest.raises(RuntimeError, match='SQL unavailable'):
        t.queue_add({'id': 'new', 'reproducible': True})
    assert t._training_queue == []


@pytest.mark.parametrize('rebuilt', [False, True])
def test_cleanup_only_deletes_the_original_working_copy(prepared, monkeypatch, rebuilt):
    monkeypatch.setattr(pt.central_recipe, 'cleanup_finished', lambda: '')
    from ppe_pipeline_api.data import catalog_view
    config, s3, _ = prepared
    version = pt.freeze(config)
    preserved = {k: v for k, v in s3.files.items() if k[1].startswith(('_training_inputs/', '_datasets/'))}
    s3.files['home', 'combined/keep.jpg'] = b'central store'
    if rebuilt:
        s3.files['home', 'combined_decay/_decay.json'] = b'{"status":"built","built_at":"new"}'
    def rows(sql, params=()):
        assert "r.status IN ('Succeeded','Failed','Stopped')" in sql
        assert 'delete_decay_after_run' not in sql
        assert "v.record ? 'prepared_decay'" in sql
        assert "v.record ? 'central_recipe'" in sql
        assert "r.execution_arn IS NOT NULL" in sql
        return [{'id': version['version_id']}]
    monkeypatch.setattr(catalog_view, '_rows', rows)
    monkeypatch.setattr(pt.record_store, 'enabled', lambda: True)
    result = pt.cleanup_finished.__wrapped__()
    assert ('skipped' if rebuilt else 'deleted') in result
    assert all(s3.files[k] == v for k, v in preserved.items())
    assert s3.files['home', 'combined/keep.jpg'] == b'central store'
    assert any(k.startswith('combined_decay/') for b, k in s3.files) == rebuilt
    assert dv.get(version['version_id'])['decay_cleanup']['state'] in result


def test_cleanup_delete_failure_keeps_marker_for_retry(prepared, monkeypatch):
    monkeypatch.setattr(pt.central_recipe, 'cleanup_finished', lambda: '')
    from ppe_pipeline_api.data import catalog_view
    config, s3, _ = prepared
    version = pt.freeze(config)
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [{'id': version['version_id']}])
    monkeypatch.setattr(pt.record_store, 'enabled', lambda: True)
    monkeypatch.setattr(s3, 'delete_objects', lambda **kw: {'Errors': [{'Code': 'AccessDenied'}]})
    with pytest.raises(RuntimeError, match='cleanup will retry'):
        pt.cleanup_finished.__wrapped__()
    assert ('home', 'combined_decay/_decay.json') in s3.files
    assert 'decay_cleanup' not in dv.get(version['version_id'])


def test_cleanup_does_nothing_without_completed_opted_in_runs(prepared, monkeypatch):
    from ppe_pipeline_api.data import catalog_view
    _, s3, _ = prepared
    original = dict(s3.files)
    monkeypatch.setattr(catalog_view, '_rows', lambda *a: [])
    monkeypatch.setattr(pt.record_store, 'enabled', lambda: True)
    assert pt.cleanup_finished.__wrapped__() == ''
    assert s3.files == original


def test_cleanup_cannot_overlap_preparation_or_preservation(monkeypatch):
    from ppe_pipeline_api.data import combined_store as cs
    monkeypatch.setattr(cs.record_store, 'enabled', lambda: True)
    cursor = SimpleNamespace(execute=lambda *a: None, fetchone=lambda: [False])
    monkeypatch.setattr(cs.record_store, 'connection', lambda: SimpleNamespace(cursor=lambda: cursor, close=lambda: None))
    called = []
    with pytest.raises(ValueError, match='being prepared or preserved'):
        cs.decay_locked(lambda: called.append(True))()
    assert not called
