"""Development SQL scheduling without touching production EventBridge rules."""
import json
import os

from .. import pull_control as pc, record_store


def import_outbox():
    """Import immutable worker messages once; never overwrite reviewed state."""
    from .. import storage
    bucket = storage.BUCKET
    prefix = f"sql-outbox/{os.environ['CATALOG_SCHEMA']}/"
    mappings = {s.target: s for s in record_store.inventory().values()
                if s.target in ('training_lineage', 'review_queue')}
    count = 0
    for page in storage.s3.get_paginator('list_objects_v2').paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key, etag = obj['Key'], obj['ETag']
            target = key[len(prefix):].split('/', 1)[0]
            if target not in mappings or not key.endswith('.json'):
                continue
            conn = record_store.connection()
            try:
                cur = conn.cursor()
                cur.execute('SELECT 1 FROM sql_worker_receipts WHERE object_key=%s AND etag=%s', (key,etag))
                if cur.fetchone():
                    continue
                item = json.loads(storage.s3.get_object(Bucket=bucket,Key=key,IfMatch=etag)['Body'].read())
                source = mappings[target]
                table = record_store.Table(source.table)
                table._lock(cur, table.key(item))
                if not table._read(cur, table.key(item)):
                    table._save(cur, item)
                cur.execute('INSERT INTO sql_worker_receipts(object_key,etag) VALUES(%s,%s) ON CONFLICT DO NOTHING', (key,etag))
                conn.commit()
                count += 1
            finally:
                conn.close()
    return count


@record_store.exclusive("sql_schedule")
def tick():
    if not record_store.enabled():
        return
    now = pc._now()
    if pc.claim_once("sql_outbox", now.strftime("%Y-%m-%dT%H:%M")):
        import_outbox()
    if pc.claim_once("sql_expiry", now.strftime("%Y-%m-%dT%H")):
        conn = record_store.connection()
        try:
            cur = conn.cursor()
            cur.execute("DELETE FROM records_pull_control WHERE expires_at<=extract(epoch from now())")
            conn.commit()
        finally:
            conn.close()
    # Automatic pulls belong to the durable scheduled training job.
