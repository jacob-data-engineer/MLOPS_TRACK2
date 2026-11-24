import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('supervisor', Path(__file__).parents[1] / 'scripts/watch_training_export.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_waits_for_both_training_and_comparison():
    assert module.readiness({'status': 'Executing'}) == 'Waiting for training'
    assert module.readiness({'status': 'Succeeded', 'comparison': {'status': 'Running'}}) == 'Waiting for comparison'
    assert module.readiness({'status': 'Succeeded', 'comparison': {'status': 'Completed'}}) == 'Exporting'


@pytest.mark.parametrize('audit', [
    {'status': 'Failed'}, {'status': 'Stopped'},
    {'status': 'Succeeded', 'comparison': {'status': 'Failed'}},
    {'status': 'Succeeded', 'comparison': {'status': 'Unavailable'}},
])
def test_failure_does_not_export(audit):
    with pytest.raises(ValueError):
        module.readiness(audit)


def test_recovery_only_matches_own_destination():
    rows = [{'id': 'other', 'destination_uri': 's3://bucket/jobs/ab/run.zip'},
            {'id': 'mine', 'destination_uri': 's3://bucket/jobs/a/run.zip'}]
    assert module.matching_exports(rows, 'bucket', 'jobs/a') == [rows[1]]


def test_state_survives_restart(tmp_path):
    import json
    path = tmp_path / 'job.json'
    module.save(path, {'run_id': 'run-123', 'stage': 'Waiting for training'})
    assert json.loads(path.read_text())['run_id'] == 'run-123'
    assert not path.with_suffix('.tmp').exists()


@pytest.mark.parametrize('result,expected', [('ready', 'Completed'), ('partial', 'Failed')])
def test_supervisor_stays_alive_until_export_finishes(tmp_path, monkeypatch, result, expected):
    import json
    import sys
    from types import SimpleNamespace as NS

    calls, uploads, rows = [], [], []
    audit = {'status': 'Succeeded', 'comparison': {'status': 'Completed'},
             'model': {'model_package_arn': 'exact-new-model'}}

    def start(kind, ident, **options):
        calls.append((kind, ident, options))
        rows.append({'id': 'export1', 'state': 'building',
                     'destination_uri': 's3://bucket/workflow/run.zip'})
        return rows[0].copy()

    def sleep(_):
        rows[0]['state'] = result
        rows[0]['missing'] = 0 if result == 'ready' else 1

    monkeypatch.setitem(sys.modules, 'fcntl', NS(flock=lambda *a: None, LOCK_EX=1, LOCK_NB=2))
    monkeypatch.setitem(sys.modules, 'ppe_pipeline_api', NS(
        record_store=NS(verify_ready=lambda: None, native_numbers=lambda x: x),
        storage=NS(s3=NS(put_object=lambda **kw: uploads.append(kw)))))
    monkeypatch.setitem(sys.modules, 'ppe_pipeline_api.training', NS(comparison=NS(_read=lambda _: audit)))
    monkeypatch.setitem(sys.modules, 'ppe_pipeline_api.chain', NS(preparation_jobs=NS()))
    monkeypatch.setitem(sys.modules, 'ppe_pipeline_api.data', NS(
        catalog_view=NS(exports=lambda **kw: rows, start_export=start),
        catalog_metadata=NS(clean=lambda x: x)))
    monkeypatch.setattr(module.time, 'sleep', sleep)
    path = tmp_path / 'job.json'
    module.save(path, {'run_id': 'run1', 'bucket': 'bucket', 'prefix': 'workflow'})
    module.supervise(path)
    assert json.loads(path.read_text())['stage'] == expected
    assert len(calls) == 1
    assert calls[0][2]['model_package_arn'] == 'exact-new-model'
    assert all(calls[0][2][key] for key in ('include_model', 'include_images', 'include_metadata'))
    assert len(uploads) == (1 if expected == 'Completed' else 0)
