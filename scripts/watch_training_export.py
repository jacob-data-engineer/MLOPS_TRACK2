"""One-run dev export supervisor. Run in a persistent, restartable container.

Usage: python watch_training_export.py /state/job.json
The JSON supplies run_id, bucket and a unique prefix. Mount /state persistently.
Training/preparation/comparison stay owned by the normal Track2 worker. This
process owns the export thread and must stay alive until it completes. It never
submits training and never chooses a different model on failure.
"""
import json
import os
import sys
import time
from pathlib import Path


def readiness(audit):
    status = audit.get('status')
    if status in ('Failed', 'Stopped', 'Cancelled', 'Canceled'):
        raise ValueError(f'Training ended: {status}')
    if status != 'Succeeded':
        return 'Waiting for training'
    comparison = audit.get('comparison') or {}
    if comparison.get('status') in ('Failed', 'Stopped', 'Unavailable', 'Cancelled'):
        raise ValueError(f'Comparison ended: {comparison.get("status")}')
    if comparison.get('status') != 'Completed':
        return 'Waiting for comparison'
    return 'Exporting'


def save(path, state):
    state['updated_at'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, default=str), encoding='utf-8')
    os.replace(temporary, path)


def matching_exports(exports, bucket, prefix):
    root = f's3://{bucket}/{prefix.strip("/")}/'
    return [e for e in exports if str(e.get('destination_uri', '')).startswith(root)]


def supervise(path):
    # Imports delayed so the decision rules can be tested without AWS/SQL.
    import fcntl
    from ppe_pipeline_api import record_store, storage
    from ppe_pipeline_api.training import comparison
    from ppe_pipeline_api.chain import preparation_jobs
    from ppe_pipeline_api.data import catalog_view, catalog_metadata

    lock = path.with_suffix('.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state = json.loads(path.read_text(encoding='utf-8'))
    if state.get('stage') in ('Completed', 'Failed'):
        return
    record_store.verify_ready()
    owned_export = None
    while True:
        try:
            try:
                audit = comparison._read(state['run_id'])
            except ValueError:
                if state.get('preparation_key'):
                    job = preparation_jobs.table().get_item(Key=state['preparation_key']).get('Item', {})
                    if job.get('state') in ('failed', 'cancelled'):
                        raise ValueError(f'Preparation failed: {job.get("error", "")}')
                state['stage'] = 'Waiting for preparation and run record'
                save(path, state)
                time.sleep(15)
                continue
            state['stage'] = readiness(audit)
            save(path, state)
            if state['stage'] != 'Exporting':
                time.sleep(15)
                continue
            exports = matching_exports(catalog_view.exports(run_id=state['run_id']),
                                       state['bucket'], state['prefix'])
            export = exports[0] if exports else None
            if export and export['state'] == 'building' and owned_export != export['id']:
                # The filesystem lock proves the prior supervisor is gone. Its
                # daemon thread died with it; record failure, never claim success.
                catalog_view._finish(export['id'],
                    "UPDATE exports SET state = 'failed', finished_at = now(), "
                    "error = %s, updated_at = now() WHERE id = %s AND state = 'building'",
                    ('Export supervisor restarted; output requires review', export['id']))
                raise ValueError('Export interrupted by supervisor restart; review before retrying')
            if export is None:
                export = catalog_view.start_export('run', state['run_id'],
                    destination='s3', bucket=state['bucket'], prefix=state['prefix'],
                    format='zip', include_model=True, include_images=True,
                    include_metadata=True, metadata_groups=None,
                    model_package_arn=audit['model']['model_package_arn'])
                owned_export = export['id']
            state['export_id'] = export['id']
            state['export'] = export
            save(path, state)
            if export['state'] == 'ready':
                if export.get('missing'):
                    raise ValueError('Export has missing files')
                # Keep the complete comparison audit alongside the full ZIP.
                report = catalog_metadata.clean(record_store.native_numbers(audit['comparison']))
                key = state['prefix'].strip('/') + '/comparison.json'
                storage.s3.put_object(Bucket=state['bucket'], Key=key,
                    Body=json.dumps(report, indent=2, default=str).encode('utf-8'),
                    ContentType='application/json')
                state.update(stage='Completed', comparison_uri=f's3://{state["bucket"]}/{key}')
                save(path, state)
                print(json.dumps(state, default=str), flush=True)
                return
            if export['state'] != 'building':
                raise ValueError(f'Export ended: {export["state"]}: {export.get("error", "")}')
            time.sleep(15)
        except ValueError as exc:
            state.update(stage='Failed', error=str(exc))
            save(path, state)
            print(json.dumps(state, default=str), flush=True)
            return
        except Exception as exc:
            # Transient AWS/SQL faults retry in this process, preserving the
            # export thread. A browser closing cannot terminate this process.
            state['last_retry_error'] = str(exc)
            save(path, state)
            time.sleep(30)


if __name__ == '__main__':
    supervise(Path(sys.argv[1]))
