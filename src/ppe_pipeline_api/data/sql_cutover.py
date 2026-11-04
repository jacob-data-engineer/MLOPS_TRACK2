"""Initialize an isolated development SQL backend from verified migration batches.

Never re-seeds a database that has been initialized: subsequent SQL writes
must not be overwritten by an old Dynamo snapshot. No production mutation.
"""
from __future__ import annotations

import argparse
import json
import os
import re

from . import catalog
from .dynamo_migration import sources


def ensure_record_tables(cur):
    cur.execute("CREATE TABLE IF NOT EXISTS sql_worker_receipts (object_key text NOT NULL,etag text NOT NULL,"
                "imported_at timestamptz NOT NULL DEFAULT now(),PRIMARY KEY(object_key,etag))")
    for source in sources():
        table = "records_" + source.target
        cur.execute(f"CREATE TABLE IF NOT EXISTS {table} (pk text NOT NULL,sk text NOT NULL DEFAULT '',"
                    "payload jsonb NOT NULL,record_type text NOT NULL DEFAULT '',file_name text NOT NULL DEFAULT '',"
                    "expires_at numeric,PRIMARY KEY(pk,sk))")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_type ON {table}(record_type,pk,sk)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_name ON {table}(file_name,pk,sk)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {table}_expires ON {table}(expires_at) WHERE expires_at IS NOT NULL")


def initialize(schema="track2_dev"):
    if not re.fullmatch(r"track2_dev(?:_[a-z0-9_]+)?", schema):
        raise ValueError("Development schema required")
    os.environ["CATALOG_SCHEMA"] = schema
    conn = catalog.connect(timeout=180)
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        cur.execute(f"CREATE TABLE IF NOT EXISTS {schema}.backend_migrations "
                    "(name text PRIMARY KEY, completed_at timestamptz NOT NULL DEFAULT now(), detail jsonb)")
        conn.commit()
        cur.execute("SELECT 1 FROM backend_migrations WHERE name='dynamo_to_sql'")
        if cur.fetchone():
            return {"schema": schema, "already_initialized": True}
    finally:
        conn.close()
    catalog.ensure_schema(report=False)
    conn = catalog.connect(timeout=180)
    try:
        cur = conn.cursor()
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (schema + ':bootstrap',))
        cur.execute("SELECT 1 FROM backend_migrations WHERE name='dynamo_to_sql'")
        if cur.fetchone():
            return {"schema": schema, "already_initialized": True}
        ensure_record_tables(cur)
        counts = {}
        for source in sources():
            cur.execute("SELECT id,copied FROM migration_stage.batches WHERE source_table=%s "
                        "AND state='verified' ORDER BY started_at DESC LIMIT 1", (source.table,))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"No verified import for {source.table}")
            batch, expected = row
            table = "records_" + source.target
            # Operational tables are initialized before the import.
            pk = source.keys[0]
            sk = f"payload->'{source.keys[1]}'->>'S'" if len(source.keys) > 1 else "''"
            cur.execute(f"INSERT INTO {table}(pk,sk,payload,record_type,file_name,expires_at) "
                        f"SELECT payload->'{pk}'->>'S',{sk},payload,record_type,"
                        "coalesce(payload->'name'->>'S',''),(payload->'expires_at'->>'N')::numeric "
                        f"FROM migration_stage.{source.target} WHERE batch_id=%s", (batch,))
            cur.execute(f"SELECT count(*) FROM {table}")
            count = int(cur.fetchone()[0])
            if count != expected:
                raise RuntimeError(f"Copy count mismatch for {source.target}: {count} != {expected}")
            counts[source.target] = count
        # Preserve established catalog numbers and immutable manifest members.
        for table in ("files", "file_sources", "labels", "dataset_versions", "dataset_members",
                      "training_runs", "run_files", "exports"):
            cur.execute(f"INSERT INTO {table} SELECT * FROM public.{table}")
        for table, column in (("file_sources", "id"), ("labels", "id"),
                              ("dataset_versions", "version_number"), ("training_runs", "run_number")):
            cur.execute(f"SELECT setval(pg_get_serial_sequence(%s,%s),greatest(coalesce(max({column}),0),1),"
                        f"count(*)>0) FROM {table}", (f"{schema}.{table}", column))
        # Preserve imported operational state in staging, but do not replay it
        # in a separate environment or claim production work as development.
        cur.execute("DELETE FROM records_training_audits WHERE record_type IN "
                    "('queue_entry','queue_state','training_submission')")
        cur.execute("DELETE FROM records_inference_queue WHERE payload->'status'->>'S' IN ('queued','running')")
        cur.execute("DELETE FROM records_pull_control WHERE pk LIKE 'claim#%' OR pk IN ('pull_queue','metrics_auto_export')")
        cur.execute("SELECT payload FROM records_pull_control WHERE pk='config' AND sk='v1'")
        row = cur.fetchone()
        if row:
            from ..record_store import plain, encode
            item = plain(row[0]); config = json.loads(item.get("body", "{}"))
            config["mode"] = "off"
            config["retrain_schedule"] = {**config.get("retrain_schedule", {}), "trigger": "off", "enabled": False, "follow_pull": False}
            config["queue_after_pull"] = {**config.get("queue_after_pull", {}), "enabled": False}
            config["dataset_assemble"] = {**config.get("dataset_assemble", {}), "run": "manual"}
            item["body"] = json.dumps(config)
            cur.execute("UPDATE records_pull_control SET payload=%s::jsonb WHERE pk='config' AND sk='v1'", (encode(item),))
        cur.execute("INSERT INTO backend_migrations(name,detail) VALUES ('dynamo_to_sql',%s::jsonb)", (json.dumps(counts),))
        conn.commit()
        return {"schema": schema, "copied": counts, "automatic_jobs": "off until enabled in development"}
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default="track2_dev")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(initialize(args.schema) if args.apply else {"plan": "initialize development SQL", "schema": args.schema}))
