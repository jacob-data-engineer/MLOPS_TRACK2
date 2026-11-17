import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ppe_pipeline_api.training.reproducibility import evaluation_inputs, pin_pipeline


def reader():
    source = Path(__file__).resolve().parents[1] / 'model_training/custom_training/evaluate_any.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'extra_val_uris')
    scope = {'json': json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), 'exec'), scope)
    return scope['extra_val_uris']


def test_long_preserved_locations_round_trip_without_truncation():
    paths = [f's3://bucket/_training_inputs/{"a"*32}/dataset/{source}/{kind}/val/'
             for source in ['folders/Testing_Dataset__object_detection_dataset', 'golden']
             for kind in ['images', 'labels']]
    objects = {}
    s3 = SimpleNamespace(
        put_object=lambda **kw: objects.update({(kw['Bucket'], kw['Key']):kw['Body']}),
        get_object=lambda **kw: {'Body': io.BytesIO(objects[kw['Bucket'], kw['Key']])})
    value = evaluation_inputs(s3, 'bucket', 'custom-run', ','.join(paths))
    assert len(','.join(paths)) > 256 and len(value) <= 256
    assert reader()(value, s3) == paths


def test_short_legacy_input_does_not_require_manifest():
    value = 's3://bucket/images/val/,s3://bucket/labels/val/'
    assert evaluation_inputs(None, 'bucket', 'run', value) == value
    assert reader()(value, None) == value.split(',')


def test_invalid_manifest_is_rejected():
    s3 = SimpleNamespace(get_object=lambda **kw: {'Body':io.BytesIO(b'{"uris":"not a list"}')})
    with pytest.raises(ValueError, match='Invalid evaluation input manifest'):
        reader()('s3://bucket/evaluation-inputs.json', s3)


def test_old_pinned_long_input_is_blocked_before_submission():
    definition = json.dumps({'Parameters':[{'Name':'EvalExtraUris','DefaultValue':''}],
        'Steps':[{'Name':'EvaluateYolo','Type':'Processing','Arguments':{
            'Environment':{'EVAL_EXTRA_URIS':{'Get':'Parameters.EvalExtraUris'}}}}]})
    sm = SimpleNamespace(describe_pipeline=lambda **kw: {'PipelineDefinition':definition})
    saved = {'name':'test', 'version_id':19, 'definition':definition,
             'recorded_parameters':{'EvalExtraUris':'x'*453}}
    with pytest.raises(ValueError, match='453 characters; AWS allows 256'):
        pin_pipeline(sm, 'test', saved, {'EvalExtraUris':'s3://bucket/evaluation-inputs.json'})
