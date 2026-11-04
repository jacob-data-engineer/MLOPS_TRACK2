"""Lossless, verified staging import for the DynamoDB-to-Postgres cutover.

No application backend is switched by this command. Each import is isolated
in its own batch; failed or changing-source batches must never be activated.
The source is read only. DynamoDB's typed values preserve decimals, sets,
binary data and unknown fields, rather than coercing them into floats.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import uuid
from dataclasses import dataclass

import boto3

from . import catalog
from ..storage import REGION


@dataclass(frozen=True)
class Source:
    table: str
    target: str
    keys: tuple[str, ...]


def sources() -> list[Source]:
    return [
        Source(os.getenv("TRAINING_AUDIT_TABLE", "track2-yolo-training-audit"), "training_audits", ("id",)),
        Source(os.getenv("TRAINING_FILES_TABLE", "track2-yolo-training-files"), "training_files", ("audit_id", "file_key")),
        Source(os.getenv("DATASET_SNAPSHOT_TABLE", "track2-yolo-dataset-snapshots"), "dataset_definitions", ("snapshot_id",)),
        Source(os.getenv("PULL_CONTROL_TABLE", "track2-pull-control"), "pull_control", ("kind", "sort")),
        Source("track2-dynamoDB", "training_lineage", ("id",)),
        Source("track2-yolo-inference-lineage", "inference_lineage", ("id",)),
        Source("track2-yolo-eval-runs", "evaluation_runs", ("id",)),
        Source("track2-yolo-inference-queue", "inference_queue", ("id",)),
        Source(os.getenv("REVIEW_QUEUE_TABLE") or "track2-yolo-review-queue", "review_queue", ("review_id",)),
    ]


def _typed(value: dict) -> dict:
    if len(value) != 1:
        raise ValueError("Invalid DynamoDB attribute")
    kind, body = next(iter(value.items()))
    if kind == "B":
        body = base64.b64encode(body).decode("ascii")
    elif kind == "BS":
        body = sorted(base64.b64encode(v).decode("ascii") for v in body)
    elif kind in ("NS", "SS"):
        body = sorted(body)
    elif kind == "M":
        body = {k: _typed(v) for k, v in body.items()}
    elif kind == "L":
        body = [_typed(v) for v in body]
    return {kind: body}


def canonical(item: dict) -> str:
    return json.dumps({k: _typed(v) for k, v in item.items()},
                      sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def record(source: Source, item: dict) -> tuple:
    key = canonical({k: item[k] for k in source.keys})
    payload = canonical(item)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    category = item.get("record_type", item.get("kind", {})).get("S", "")
    return key, category, payload, digest


def ensure_tables(conn) -> None:
    cur = conn.cursor()
    cur.execute("CREATE SCHEMA IF NOT EXISTS migration_stage")
    cur.execute("""CREATE TABLE IF NOT EXISTS migration_stage.batches (
        id text PRIMARY KEY, source_table text NOT NULL, target_table text NOT NULL,
        state text NOT NULL CHECK (state IN ('copying','verifying','verified','changed','failed')),
        started_at timestamptz NOT NULL DEFAULT now(), finished_at timestamptz,
        copied bigint NOT NULL DEFAULT 0, checked bigint NOT NULL DEFAULT 0,
        mismatches bigint NOT NULL DEFAULT 0, error text)""")
    for source in sources():
        # Identifiers are fixed in this module, never supplied by a caller.
        cur.execute(f"""CREATE TABLE IF NOT EXISTS migration_stage.{source.target} (
            batch_id text NOT NULL REFERENCES migration_stage.batches(id),
            source_key jsonb NOT NULL, record_type text NOT NULL,
            payload jsonb NOT NULL, sha256 char(64) NOT NULL,
            PRIMARY KEY (batch_id, source_key))""")
        cur.execute(f"CREATE INDEX IF NOT EXISTS {source.target}_type "
                    f"ON migration_stage.{source.target} (batch_id, record_type)")
    conn.commit()


def pages(client, source: Source):
    args = {"TableName": source.table, "ConsistentRead": True, "Limit": 250}
    while True:
        page = client.scan(**args)
        yield page.get("Items", [])
        if not page.get("LastEvaluatedKey"):
            return
        args["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def import_source(conn, client, source: Source) -> dict:
    if source not in sources():
        raise ValueError("Source is not in the migration inventory")
    batch = uuid.uuid4().hex
    cur = conn.cursor()
    cur.execute("INSERT INTO migration_stage.batches (id,source_table,target_table,state) "
                "VALUES (%s,%s,%s,'copying')", (batch, source.table, source.target))
    conn.commit()
    copied = checked = mismatches = 0
    try:
        for items in pages(client, source):
            rows = [(batch, *record(source, item)) for item in items]
            if rows:
                placeholders = ",".join(["(%s,%s::jsonb,%s,%s::jsonb,%s)"] * len(rows))
                cur.execute(f"INSERT INTO migration_stage.{source.target} "
                            "(batch_id,source_key,record_type,payload,sha256) VALUES " + placeholders,
                            tuple(v for row in rows for v in row))
            copied += len(rows)
            cur.execute("UPDATE migration_stage.batches SET copied=%s WHERE id=%s", (copied, batch))
            conn.commit()
        cur.execute("UPDATE migration_stage.batches SET state='verifying' WHERE id=%s", (batch,))
        conn.commit()
        # A fresh source scan catches changed, new and removed records. It is
        # not a global snapshot: writers must be paused for final cutover.
        for items in pages(client, source):
            rows = [record(source, item) for item in items]
            if rows:
                slots = ",".join(["(%s::jsonb,%s,%s::jsonb)"] * len(rows))
                cur.execute(f"SELECT count(*) FROM (VALUES {slots}) AS s(k,h,p) "
                            f"LEFT JOIN migration_stage.{source.target} t "
                            "ON t.batch_id=%s AND t.source_key=s.k "
                            "WHERE t.sha256 IS DISTINCT FROM s.h OR t.payload IS DISTINCT FROM s.p",
                            tuple(v for row in rows for v in (row[0], row[3], row[2])) + (batch,))
                mismatches += int(cur.fetchone()[0])
            checked += len(rows)
            conn.commit()
        cur.execute(f"SELECT count(*) FROM migration_stage.{source.target} WHERE batch_id=%s", (batch,))
        stored = int(cur.fetchone()[0])
        state = "verified" if copied == checked == stored and mismatches == 0 else "changed"
        cur.execute("UPDATE migration_stage.batches SET state=%s,checked=%s,mismatches=%s,"
                    "finished_at=now() WHERE id=%s", (state, checked, mismatches, batch))
        conn.commit()
        return {"batch": batch, "source": source.table, "target": source.target,
                "state": state, "copied": copied, "checked": checked, "mismatches": mismatches}
    except Exception as exc:
        conn.rollback()
        cur = conn.cursor()
        cur.execute("UPDATE migration_stage.batches SET state='failed',finished_at=now(),error=%s "
                    "WHERE id=%s", (type(exc).__name__, batch))
        conn.commit()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Create SQL staging tables and copy records")
    parser.add_argument("--schema-only", action="store_true", help="Create tables without copying")
    parser.add_argument("--source", action="append", choices=[s.target for s in sources()],
                        help="Import only these sources (repeatable); defaults to all")
    args = parser.parse_args()
    selected = [s for s in sources() if not args.source or s.target in args.source]
    if not args.apply:
        print(json.dumps({"mode": "plan", "sources": [s.__dict__ for s in selected]}, indent=2))
        return 0
    conn = catalog.connect()
    try:
        ensure_tables(conn)
        if args.schema_only:
            print(json.dumps({"schema": "migration_stage", "tables": 10}))
            return 0
        client = boto3.client("dynamodb", region_name=REGION)
        ok = True
        for source in selected:
            try:
                result = import_source(conn, client, source)
            except Exception as exc:
                # One denied/unavailable source must not prevent independent
                # tables from being copied. No exception becomes success.
                code = getattr(exc, "response", {}).get("Error", {}).get("Code", type(exc).__name__)
                result = {"source": source.table, "state": "failed", "error": code}
            print(json.dumps(result), flush=True)
            ok = ok and result["state"] == "verified"
        return 0 if ok else 1
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
