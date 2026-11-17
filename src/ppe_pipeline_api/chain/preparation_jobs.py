"""Durable preparation jobs; the browser submits and polls, the worker runs them."""
import contextlib
import datetime as dt
import os
import uuid
from boto3.dynamodb.conditions import Attr

from .. import data, pull_control as pc, record_store

KIND = 'prepared_training_job'


def table():
    return record_store.Table(pc.TABLE_NAME)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def public(job):
    return {k: job[k] for k in ('id', 'state', 'stage', 'completed', 'total', 'percent',
            'created_at', 'updated_at', 'error', 'entry_id', 'dataset_version_id') if k in job}


def jobs(*, active=False):
    query = dict(KeyConditionExpression='kind = :kind',
                 ExpressionAttributeValues={':kind': KIND}, ScanIndexForward=active, Limit=1)
    if active:
        query['FilterExpression'] = Attr('state').eq('pending') | Attr('state').eq('running')
    return record_store.native_numbers(table().query(**query)['Items'])


def latest():
    found = jobs(active=True) or jobs()
    return public(found[0]) if found else None


def submit(config=None, *, trigger='manual', submit='enqueue', overrides=None):
    data.require_training_sql()
    if submit not in ('enqueue', 'queue', 'start'):
        raise ValueError('Choose enqueue, queue or start.')
    stamp = now()
    ident = uuid.uuid4().hex
    job = {'kind': KIND, 'sort': stamp + '#' + ident, 'id': ident, 'state': 'pending',
           'stage': 'Waiting for background worker', 'percent': 0, 'completed': 0, 'total': 0,
           'created_at': stamp, 'updated_at': stamp, 'config': config if config is not None else pc.read_config(),
           'trigger': trigger, 'submit': submit, 'overrides': overrides or {}}
    table().put_item(Item=job)
    return {'ok': True, 'job': public(job), 'steps': [{'step': 'background', 'ok': True,
            'detail': 'Preparation saved; the background worker will preserve and queue the dataset.'}]}


def publish(job, entry):
    """Commit queue entry, queue switch and finished job in ONE transaction."""
    queue = record_store.Table(os.getenv('TRAINING_AUDIT_TABLE', 'track2-yolo-training-audit'))
    job.update(state='completed', stage='Added to training queue', percent=100,
               entry_id=entry['id'], dataset_version_id=entry['dataset_version_id'], updated_at=now())
    with contextlib.closing(record_store.connection()) as conn:
        cur = conn.cursor()
        queue._lock(cur, queue.key(entry))
        queue._save(cur, {**entry, 'record_type': 'queue_entry'})
        if job['submit'] in ('queue', 'start'):
            state = {'id': 'queue-state', 'record_type': 'queue_state', 'started': True, 'at': now()}
            queue._lock(cur, queue.key(state))
            queue._save(cur, state)
        table()._lock(cur, table().key(job))
        table()._save(cur, job)
        conn.commit()


@record_store.exclusive('prepared_training_jobs')
def tick():
    from .prepared_training import queue_prepared
    active = jobs(active=True)
    if not active:
        return
    job = active[0]
    def progress(stage, completed, total):
        percent = (10 + int(75 * completed / total)) if total else 5
        if stage == 'Saving dataset records to SQL':
            percent = 95
        job.update(state='running', stage=stage, completed=completed, total=total,
                   percent=percent, updated_at=now())
        table().put_item(Item=job)
    try:
        progress('Starting or resuming preparation', 0, 0)
        queue_prepared(job['config'], trigger=job['trigger'], submit=job['submit'],
                       overrides=job['overrides'], progress=progress, entry_id=job['id'],
                       publish=lambda entry: publish(job, entry))
    except Exception as exc:
        # A killed process leaves 'running', which the replacement worker resumes.
        job.update(state='failed', stage='Preparation failed', error=str(exc), updated_at=now())
        table().put_item(Item=job)
