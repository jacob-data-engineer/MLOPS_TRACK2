"""Saved validation identity, durable retry and baseline isolation contracts."""
import copy
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from botocore.exceptions import ClientError

from ppe_pipeline_api import data, record_store
from ppe_pipeline_api.data import comparison as dc
from ppe_pipeline_api.training import comparison as c
from ppe_pipeline_api.training.comparison_runner import class_mapping, counts, download


def arn(group='other-detector', version=1):
    return f'arn:aws:sagemaker:ap-southeast-1:123456789012:model-package/{group}/{version}'


def selected():
    return {'enabled': True, 'baseline_arn': arn(c.group_name()), 'other_arns': [arn()]}


def version():
    return {'immutable_files': True, 'central_recipe': {'store_uri': 's3://b/store'},
            'val_images': 1, 'training_classes': {'0': 'helmet', '1': 'person'},
            'file_provenance': [{'split': 'val', 'central_image': 's3://b/store/objects/replay/a.jpg',
                'central_image_etag': 'image-v1', 'central_label': 's3://b/store/objects/replay/a.txt',
                'central_label_etag': 'label-v1'}, {'split': 'train'}]}


def test_manifest_uses_only_saved_validation_and_changes_when_labels_change(monkeypatch):
    v = version()
    monkeypatch.setattr(dc.dv, 'get', lambda ident: v)
    original = dc.validation_manifest('saved')
    assert len(original['files']) == 1 and original['dataset_version_id'] == 'saved'
    v['file_provenance'][0]['central_label_etag'] = 'label-v2'
    assert dc.validation_manifest('saved')['snapshot_id'] != original['snapshot_id']


@pytest.mark.parametrize('change', [lambda v: v.update(immutable_files=False),
    lambda v: v.update(val_images=2), lambda v: v.update(training_classes={'1': 'helmet'}),
    lambda v: v['file_provenance'][0].update(central_image='s3://b/current/a.jpg')])
def test_incomplete_historical_data_never_falls_back(monkeypatch, change):
    v = version(); change(v)
    monkeypatch.setattr(dc.dv, 'get', lambda ident: v)
    with pytest.raises(ValueError): dc.validation_manifest('old')


def test_class_mapping_aligns_names_and_rejects_subsets():
    assert class_mapping({0:'helmet', 1:'person'}, {0:'person', 1:'helmet'}) == {0:1, 1:0}
    with pytest.raises(ValueError): class_mapping({0:'helmet'}, {0:'person'})
    with pytest.raises(ValueError): class_mapping({0:'helmet'}, {0:'helmet', 1:'person'})


def test_error_counts_include_wrong_class_and_background():
    matrix = np.array([[3, 2, 1], [1, 4, 2], [2, 1, 0]])
    rows = counts(matrix, {0: 'helmet', 1: 'person'})
    assert (rows[0]['tp'], rows[0]['fp'], rows[0]['fn']) == (3, 3, 3)
    assert (rows[1]['tp'], rows[1]['fp'], rows[1]['fn']) == (4, 3, 3)
    assert rows[0]['precision'] == rows[0]['recall'] == 0.5


def test_download_requires_recorded_object_identity(tmp_path):
    calls = []
    s3 = SimpleNamespace(get_object=lambda **kw: calls.append(kw) or {
        'Body': SimpleNamespace(iter_chunks=lambda **kw: iter([b'exact bytes']))})
    download(s3, 's3://b/f.jpg', 'old-etag', tmp_path / 'image')
    assert calls == [{'Bucket':'b', 'Key':'f.jpg', 'IfMatch':'old-etag'}]


def test_selection_pins_baseline_registry_and_limits_other_models():
    assert c.normalize(None) == {'enabled':False}
    assert c.normalize(selected()) == selected()
    with pytest.raises(ValueError): c.normalize({**selected(), 'baseline_arn': arn()})
    with pytest.raises(ValueError): c.normalize({**selected(), 'other_arns': [arn(version=i) for i in range(5)]})
    with pytest.raises(ValueError): c.normalize({'enabled': 'true'})


def test_no_comparisons_is_no_work_and_no_zero_scores(monkeypatch):
    monkeypatch.setattr(c, '_save', lambda *a: pytest.fail('No comparison should be created'))
    assert c.status_for({'status':'Succeeded'}) == 'No comparisons'
    c.advance({'status':'Succeeded'})


@pytest.fixture
def state(monkeypatch):
    saved = []
    monkeypatch.setattr(c, '_save', lambda ident, value: saved.append(copy.deepcopy(value)))
    return saved


def test_training_failure_marks_comparison_unavailable_without_changing_training(state):
    audit = {'id':'run', 'status':'Failed', 'reproducibility': {'entry': {'comparison': selected()}}}
    c.advance(audit)
    assert state[-1]['status'] == 'Unavailable' and audit['status'] == 'Failed'


def test_restart_recovers_existing_job_without_resubmitting(monkeypatch, state):
    monkeypatch.setattr(c.t2, 'sm', SimpleNamespace(describe_processing_job=lambda **kw: {'ProcessingJobStatus':'InProgress'}))
    monkeypatch.setattr(c, '_start', lambda *a: pytest.fail('Already submitted'))
    c.advance({'id':'run', 'status':'Succeeded', 'comparison': {'id':'same', 'status':'Pending', 'job_name':'existing'}})
    assert state[-1]['status'] == 'Running'


def test_finished_comparison_keeps_scores_out_of_original_training_metrics(monkeypatch, state):
    report = {'status':'Completed', 'rows':[{'mAP50-95':0.7}]}
    monkeypatch.setattr(c.t2, 'sm', SimpleNamespace(describe_processing_job=lambda **kw: {'ProcessingJobStatus':'Completed'}))
    monkeypatch.setattr(c.t2, 's3', SimpleNamespace(get_object=lambda **kw: {'Body':io.BytesIO(json.dumps(report).encode())}))
    audit = {'id':'run', 'status':'Succeeded', 'metrics': {'mAP50-95':0.5},
             'comparison': {'id':'same', 'status':'Running', 'job_name':'existing', 'result_uri':'s3://b/result'}}
    c.advance(audit)
    assert state[-1]['result'] == report and audit['metrics']['mAP50-95'] == 0.5


def test_missing_snapshot_becomes_unavailable(monkeypatch, state):
    monkeypatch.setattr(data, 'validation_manifest', lambda v: (_ for _ in ()).throw(ValueError('Snapshot missing')))
    c.advance({'id':'run', 'status':'Succeeded', 'comparison': {'id':'id', 'status':'Pending', 'selection':selected()}})
    assert state[-1]['status'] == 'Unavailable' and state[-1]['error'] == 'Snapshot missing'


def test_access_denied_is_failed_not_endless_pending(monkeypatch, state):
    monkeypatch.setattr(c, '_start', lambda *a: (_ for _ in ()).throw(
        ClientError({'Error':{'Code':'AccessDeniedException','Message':'Denied'}}, 'CreateProcessingJob')))
    c.advance({'id':'run','status':'Succeeded','comparison':{'status':'Pending'}})
    assert state[-1]['status'] == 'Failed'


def test_retry_keeps_saved_models_and_prior_result_reference(monkeypatch, state):
    monkeypatch.setattr(record_store, 'enabled', lambda: False)
    monkeypatch.setattr(data, 'require_training_sql', lambda: None)
    pinned = {**selected(), 'models':[{'uri':'s3://b/preserved.pt','etag':'v1'}]}
    monkeypatch.setattr(c, '_read', lambda ident: {'status':'Succeeded','comparison': {
        'id':'old', 'status':'Failed','selection':pinned,'result_uri':'s3://b/old-result'}})
    result = c.request('run')
    assert result['selection'] == pinned and result['id'] != 'old'
    assert result['history'][0]['result_uri'] == 's3://b/old-result'


def test_repeat_training_keeps_pinned_comparison(monkeypatch):
    from ppe_pipeline_api.chain import prepared_training as chain
    template = {'comparison':{**selected(), 'models':[{'uri':'s3://b/fixed.pt','etag':'v1'}]}}
    monkeypatch.setattr(data,'repeat_training_entry',lambda ident:copy.deepcopy(template))
    queued = []
    monkeypatch.setattr(chain.training,'queue_add',queued.append)
    chain.repeat_run('source')
    assert queued[0]['comparison'] == template['comparison']


def test_saved_selection_returns_only_public_configuration(monkeypatch):
    monkeypatch.setattr(dc.dv,'get',lambda ident:{'entry_template':{'comparison':{**selected(),'models':[{'uri':'secret'}]}}})
    assert dc.saved_selection('saved') == selected()


def test_processing_spec_requires_saved_digest():
    with pytest.raises(ValueError): c.processing_spec({})
    spec = {'Steps':[{'Type':'Training','Arguments':{
        'AlgorithmSpecification':{'TrainingImage':'image@sha256:fixed'},'RoleArn':'role'}}]}
    assert c.processing_spec({'reproducibility':{'pipeline':{'definition':json.dumps(spec)}}}) == ('image@sha256:fixed','role')


def test_processing_submission_records_identity_before_aws_and_never_deploys(monkeypatch, state):
    calls = []
    monkeypatch.setattr(data,'validation_manifest',lambda ident:{'files':[{}],'names':{0:'helmet'},'snapshot_id':'fixed','dataset_version_id':ident})
    monkeypatch.setattr(c,'processing_spec',lambda audit:('image@sha256:fixed','role'))
    monkeypatch.setattr(c.t2,'publish_versioned_weights',lambda ident:'s3://b/model.pt')
    monkeypatch.setattr(c.t2,'s3',SimpleNamespace(head_object=lambda **kw:{'ETag':'fixed'},put_object=lambda **kw:None))
    def create(**kwargs):
        assert state[-1]['job_name'] == kwargs['ProcessingJobName']
        calls.append(kwargs)
    monkeypatch.setattr(c.t2,'sm',SimpleNamespace(create_processing_job=create))
    c._start({'id':'run','dataset_version_id':'saved','parameters':{'img_size':320},'model':{'model_package_arn':arn()}},
             {'id':'abc','status':'Pending','selection':{'models':[]}})
    assert calls[0]['AppSpecification']['ImageUri'] == 'image@sha256:fixed'
    assert calls[0]['ProcessingJobName'] == 'track2-compare-abc'
    assert state[-1]['status'] == 'Running'


def test_register_baseline_copies_artifact_without_approval_or_deployment(monkeypatch):
    created = []
    monkeypatch.setattr(data,'preserve_training_base',lambda uri:{'uri':'s3://b/preserved/model.tar.gz'})
    monkeypatch.setattr(c.t2,'ensure_model_package_group',lambda *a:True)
    monkeypatch.setattr(c.t2,'sm',SimpleNamespace(
        describe_model_package=lambda **kw:{'InferenceSpecification':{'Containers':[{'Image':'serve','ModelDataUrl':'s3://b/original.tar.gz'}]}},
        create_model_package=lambda **kw:created.append(kw) or {'ModelPackageArn':arn(c.group_name())}))
    c.register_baseline(arn(),'Reference')
    assert created[0]['ModelPackageGroupName'] == c.group_name()
    assert created[0]['ModelApprovalStatus'] == 'PendingManualApproval'
    assert created[0]['InferenceSpecification']['Containers'][0]['ModelDataUrl'] == 's3://b/preserved/model.tar.gz'


def test_local_yolo_evaluator_smoke_without_training_or_network(tmp_path):
    """Real CPU validation of two identical random checkpoints must give zero deltas."""
    import ultralytics
    import yaml
    from pathlib import Path
    from PIL import Image
    from ppe_pipeline_api.training.comparison_runner import evaluate

    cfg = yaml.safe_load((Path(ultralytics.__file__).parent / 'cfg/models/11/yolo11.yaml').read_text())
    cfg.update(nc=1, scale='n')
    config = tmp_path / 'test-model.yaml'
    config.write_text(yaml.safe_dump(cfg))
    model = ultralytics.YOLO(str(config), task='detect')
    model.model.names = {0:'helmet'}
    checkpoint = tmp_path / 'model.pt'
    model.save(str(checkpoint))
    picture = io.BytesIO()
    Image.new('RGB',(64,64),'white').save(picture,format='JPEG')
    objects = {'image.jpg':picture.getvalue(),'label.txt':b'0 0.5 0.5 0.2 0.2\n','model.pt':checkpoint.read_bytes()}
    def get_object(Bucket,Key,IfMatch):
        assert Bucket == 'fake' and IfMatch == 'fixed'
        return {'Body':SimpleNamespace(iter_chunks=lambda **kw:iter([objects[Key]]))}
    manifest = {'validation':{'files':[{'image':'s3://fake/image.jpg','image_etag':'fixed',
        'label':'s3://fake/label.txt','label_etag':'fixed'}],'names':{0:'helmet'},'snapshot_id':'same','dataset_version_id':'saved'},
        'settings':{'imgsz':64,'ap_conf':0.001,'confidence':0.25,'nms_iou':0.7,'max_det':300},
        'models':[{'id':role,'role':role,'label':role,'uri':'s3://fake/model.pt','etag':'fixed'} for role in ('trained','baseline')]}
    result = evaluate(manifest,tmp_path/'evaluation',SimpleNamespace(get_object=get_object))
    assert len(result['rows']) == 2 and result['validation_images'] == 1
    assert result['rows'][0]['weights_sha256'] == result['rows'][1]['weights_sha256']
    assert all(delta == 0 for delta in result['rows'][0]['delta_pp'].values())
    assert result['settings']['classes'] is None and result['settings']['augment'] is False


def test_comparison_routes_require_auth_and_return_busy_conflict(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from ppe_pipeline_api.routers import track2_comparisons as router
    app = FastAPI()
    app.include_router(router.router,prefix='/track2/comparisons')
    client = TestClient(app)
    assert client.get('/track2/comparisons').status_code == 401
    app.dependency_overrides[router.require_session] = lambda: {'user':'test'}
    monkeypatch.setattr(c,'request',lambda *args:{'started':False})
    assert client.post('/track2/comparisons/run-1/run',json={}).status_code == 409
    monkeypatch.setattr(c.t2,'scan_training_audits',lambda **kw:[{'id':'old','status':'Succeeded'}])
    response = client.get('/track2/comparisons')
    assert response.status_code == 200 and response.json()['runs'][0]['status'] == 'No comparisons'
