"""Durable pre-training progress, visible across worker and web processes."""
import datetime as dt
import os
from .. import record_store


def table():
    return record_store.Table(os.getenv('TRAINING_AUDIT_TABLE', 'track2-yolo-training-audit'))


def write(audit_id, stage, completed=0, total=0):
    if not record_store.enabled():
        return
    table().put_item(Item={'id': audit_id + ':recording', 'record_type':'training_recording',
        'audit_id':audit_id, 'stage':stage, 'completed':completed, 'total':total,
        'updated_at':dt.datetime.now(dt.timezone.utc).isoformat()})


def for_queue(entries):
    if not record_store.enabled():
        return None
    for entry in entries:
        row = table().get_item(Key={'id':'run-' + entry['id'] + ':recording'}).get('Item')
        if row:
            return {**record_store.native_numbers(row), 'entry_id':entry['id']}
    return None
