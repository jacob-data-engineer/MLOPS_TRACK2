"""Durable, development-only emergency stop for the training workflow."""
import uuid
import time
import threading
from .. import record_store, pull_control as pc

KEY = {'kind': 'training_control', 'sort': 'stop_all'}

class Stopped(RuntimeError):
    pass

def state():
    if not record_store.enabled():
        return {'halted': False, 'generation': ''}
    return record_store.Table(pc.TABLE_NAME).get_item(Key=KEY).get('Item') or {**KEY, 'halted': False, 'generation': ''}

def checkpoint(generation=None):
    current = state()
    if current.get('halted') or (generation is not None and generation != current.get('generation', '')):
        raise Stopped('Stopped by Stop All. Completed results and preserved data are retained.')
    return current.get('generation', '')

def audits():
    from ..training import track2_shared as t
    rows=[];kwargs={}
    while True:
        page=t.training_audit_table().scan(**kwargs)
        rows.extend(r for r in page.get('Items',[]) if r.get('record_type','training_audit') in ('training_audit','experiment_comparison'))
        if not page.get('LastEvaluatedKey'): return rows
        kwargs['ExclusiveStartKey']=page['LastEvaluatedKey']

def sweep():
    """Retry scoped AWS stops while halted; never select account-wide jobs."""
    from . import training_variants as variants, preparation_jobs
    from ..training import track2_shared as t, comparison
    if not state().get('halted'):
        return {'problems': [], 'pending': False}
    problems=[]; pending=False
    t.queue_stop()
    for entry in t.queue_snapshot().get('pending', []):
        t.queue_remove(entry['id'])
    for kind in (variants.KIND, preparation_jobs.KIND):
        for job in variants.all_records(kind):
            active = job.get('state') in ('Pending','Preparing','Training','Comparing','Needs attention','pending','running')
            if not active:
                continue
            if job.get('pull_run_id'):
                try: pc.request_stop(job['pull_run_id'])
                except Exception as exc: problems.append(str(exc))
            job.update(state='Cancelled' if kind==variants.KIND else 'cancelled', stage='Stopped by Stop All')
            for variant in job.get('variants', []):
                if variant['state'] in ('Pending','Preparing','Queued'):
                    variant.update(state='Stopped',training_status='Stopping')
            variants.save(job)
    for audit in audits():
        try:
            arn=audit.get('execution_arn')
            if arn and audit.get('status') not in t.TERMINAL_EXECUTION_STATUSES:
                status=t.sm.describe_pipeline_execution(PipelineExecutionArn=arn)['PipelineExecutionStatus']
                if status in ('Executing','Stopping'):
                    pending=True
                    if status=='Executing': t.sm.stop_pipeline_execution(PipelineExecutionArn=arn)
            build=audit.get('build_id')
            if build and audit.get('status') not in t.TERMINAL_EXECUTION_STATUSES:
                builds=t.codebuild.batch_get_builds(ids=[build]).get('builds',[])
                if builds and builds[0]['buildStatus']=='IN_PROGRESS':
                    pending=True;t.codebuild.stop_build(id=build)
            value=audit.get('comparison') or {}
            if value.get('status') in ('Pending','Running'):
                name=value.get('job_name')
                if name:
                    status=t.sm.describe_processing_job(ProcessingJobName=name)['ProcessingJobStatus']
                    if status in ('InProgress','Stopping'):
                        pending=True
                        if status=='InProgress':t.sm.stop_processing_job(ProcessingJobName=name)
                        continue
                comparison._save(audit['id'],{**value,'status':'Stopped','error':'Stopped by Stop All'})
        except Exception as exc:
            problems.append(f"{audit['id']}: {exc}")
    return {'problems':problems,'pending':pending}

def stop():
    if not record_store.enabled():
        raise ValueError('Stop All is available only in the isolated SQL development environment.')
    value={**KEY,'halted':True,'generation':uuid.uuid4().hex}
    record_store.Table(pc.TABLE_NAME).put_item(Item=value)  # Stop new work before any AWS calls.
    result=sweep()
    return {**value,**result,'message':'Stop All requested. Preparation stops at its next checkpoint; AWS jobs may take longer. Automation remains paused.'}

@record_store.exclusive('prepared_training_jobs')
@record_store.exclusive('training_queue')
@record_store.exclusive('baseline_comparison')
def resume():
    if not record_store.enabled():
        raise ValueError('Training control is available only in SQL development.')
    result=sweep()
    if result['problems'] or result['pending']:
        raise ValueError('Stopping is not finished. Retry after active jobs stop. '+ '; '.join(result['problems']))
    value={**state(),'halted':False}
    record_store.Table(pc.TABLE_NAME).put_item(Item=value)
    return {**value,'message':'Automation enabled. Cancelled work is not restarted; submit a new run when ready.'}


_check_lock = threading.Lock()
_checked_at = 0.0
_cached_halted = False

def checkpoint_fast():
    """Bound SQL polling to once per second across copy threads."""
    global _checked_at, _cached_halted
    if not record_store.enabled(): return
    with _check_lock:
        if time.monotonic() - _checked_at > 1:
            _cached_halted = state().get('halted', False)
            _checked_at = time.monotonic()
        if _cached_halted: raise Stopped('Stopped by Stop All')
