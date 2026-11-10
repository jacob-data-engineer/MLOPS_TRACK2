"""Mutable dataset definitions referencing the existing catalog assets and versions.

The old soft_* tables are deliberately left intact as historical records. New
membership uses files/labels and freezes into dataset_versions/dataset_members.
"""
from __future__ import annotations

import json
import uuid

from . import catalog

DDL = [
    """CREATE TABLE IF NOT EXISTS dataset_selections (
        id text PRIMARY KEY, created_at timestamptz NOT NULL DEFAULT now(),
        updated_at timestamptz NOT NULL DEFAULT now(), name text NOT NULL DEFAULT '',
        config jsonb NOT NULL DEFAULT '{}'::jsonb, next_at timestamptz,
        last_result jsonb NOT NULL DEFAULT '{}'::jsonb, collection integer NOT NULL DEFAULT 0)""",
    """CREATE TABLE IF NOT EXISTS dataset_selection_members (
        selection_id text NOT NULL REFERENCES dataset_selections(id), source_uri text NOT NULL,
        sha256 char(64) NOT NULL REFERENCES files(sha256), label_id bigint NOT NULL REFERENCES labels(id),
        source_id text NOT NULL, kind text NOT NULL, source_etag text NOT NULL,
        split text NOT NULL CHECK(split IN ('train','val')), week text NOT NULL,
        active boolean NOT NULL DEFAULT true, reason text NOT NULL DEFAULT '',
        PRIMARY KEY(selection_id,source_uri))""",
    """CREATE TABLE IF NOT EXISTS dataset_selection_splits (
        selection_id text NOT NULL REFERENCES dataset_selections(id), source_id text NOT NULL,
        policy text NOT NULL, PRIMARY KEY(selection_id,source_id))""",
    """CREATE TABLE IF NOT EXISTS dataset_selection_versions (
        selection_id text NOT NULL REFERENCES dataset_selections(id),
        version_id text NOT NULL REFERENCES dataset_versions(id), content_key text NOT NULL,
        created_at timestamptz NOT NULL DEFAULT now(), record jsonb NOT NULL,
        PRIMARY KEY(selection_id,content_key))""",
    """CREATE TABLE IF NOT EXISTS dataset_selection_jobs (
        id text PRIMARY KEY, selection_id text NOT NULL REFERENCES dataset_selections(id),
        kind text NOT NULL, config jsonb NOT NULL, status text NOT NULL DEFAULT 'queued',
        created_at timestamptz NOT NULL DEFAULT now(), updated_at timestamptz NOT NULL DEFAULT now(),
        done integer NOT NULL DEFAULT 0, total integer NOT NULL DEFAULT 0,
        result jsonb NOT NULL DEFAULT '{}'::jsonb, error text NOT NULL DEFAULT '')""",
    "CREATE UNIQUE INDEX IF NOT EXISTS dataset_selection_active_job ON dataset_selection_jobs(selection_id) WHERE status IN ('queued','running')",
]


def connection():
    catalog.ensure_schema_once()
    conn = catalog.connect()
    try:
        cur = conn.cursor()
        # DDL can be reached by the ticker and an HTTP request at first use.
        cur.execute("SELECT pg_advisory_xact_lock(hashtext('dataset_selection_schema'))")
        for statement in DDL:
            cur.execute(statement)
        conn.commit()
        return conn
    except Exception:
        conn.close()
        raise


def rows(cur):
    names = [x[0] for x in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def ensure(cur, sid):
    cur.execute("INSERT INTO dataset_selections(id) VALUES(%s) ON CONFLICT DO NOTHING", (sid,))


def create(source_id=""):
    sid = "sel-"+uuid.uuid4().hex
    conn = connection()
    try:
        cur = conn.cursor()
        ensure(cur, sid)
        if source_id:
            cur.execute("""INSERT INTO dataset_selection_members
                SELECT %s,source_uri,sha256,label_id,source_id,kind,source_etag,split,week,active,reason
                FROM dataset_selection_members WHERE selection_id=%s""", (sid, source_id))
            cur.execute("INSERT INTO dataset_selection_splits SELECT %s,source_id,policy FROM dataset_selection_splits WHERE selection_id=%s", (sid, source_id))
            cur.execute("UPDATE dataset_selections SET collection=coalesce((SELECT collection FROM dataset_selections WHERE id=%s),0) WHERE id=%s", (source_id,sid))
        conn.commit()
    finally:
        conn.close()
    return sid


def member_rows(cur, sid, active=True):
    cur.execute("""SELECT m.*, m.sha256 AS asset_id,m.label_id AS annotation_id,
        f.store_bucket,f.store_key,f.bytes,n.yolo,n.class_names AS classes,n.source_uri AS label_uri
        FROM dataset_selection_members m JOIN files f ON f.sha256=m.sha256
        JOIN labels n ON n.id=m.label_id WHERE m.selection_id=%s"""+
        (" AND m.active" if active else "")+" ORDER BY m.week,m.source_uri", (sid,))
    result = rows(cur)
    for row in result:
        row["stored_uri"] = f"s3://{row['store_bucket']}/{row['store_key']}"
    return result


def annotation(cur, item):
    """One immutable label revision per actual image content and taxonomy."""
    sha = item["asset_id"]
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (sha,))
    cur.execute("SELECT id,yolo,class_names,version FROM labels WHERE sha256=%s ORDER BY version", (sha,))
    previous = cur.fetchall()
    for lid, yolo, classes, version in previous:
        if yolo == item["yolo"] and classes == item["classes"]:
            return lid
    version = max((r[3] for r in previous), default=0)+1
    cur.execute("""INSERT INTO labels(sha256,version,class_names,yolo,origin,source_uri)
        VALUES(%s,%s,%s::jsonb,%s,'import',%s) RETURNING id""",
        (sha, version, json.dumps(item["classes"]), item["yolo"], item["label_uri"]))
    return cur.fetchone()[0]


def state(sid):
    conn = connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT * FROM dataset_selections WHERE id=%s", (sid,))
        records = rows(cur)
        return records[0] if records else {}
    finally:
        conn.close()


def result(sid, value):
    conn = connection()
    try:
        cur = conn.cursor()
        ensure(cur, sid)
        cur.execute("UPDATE dataset_selections SET last_result=%s::jsonb WHERE id=%s", (json.dumps(value,default=str),sid))
        conn.commit()
    finally:
        conn.close()


def versions(sid):
    conn = connection()
    try:
        cur = conn.cursor()
        cur.execute("""SELECT v.version_id,v.created_at,d.state,d.train_images,d.val_images,
            d.version_number FROM dataset_selection_versions v JOIN dataset_versions d ON d.id=v.version_id
            WHERE v.selection_id=%s ORDER BY v.created_at DESC LIMIT 30""", (sid,))
        return rows(cur)
    finally:
        conn.close()
