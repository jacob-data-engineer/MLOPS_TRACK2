"""Keep catalog/export tables current in the same SQL transaction as records."""
import json


def project_files(cur, items):
    """Bulk projection in the caller's transaction; retries replace identical keys."""
    cur.execute("INSERT INTO training_runs (run_id,status,files_synced) "
                "SELECT DISTINCT value,'Recording',true FROM jsonb_array_elements_text(%s::jsonb) "
                "ON CONFLICT (run_id) DO NOTHING", (json.dumps(sorted({i['audit_id'] for i in items})),))
    rows = [dict(run_id=i['audit_id'], file_key=i['file_key'], s3_uri=i.get('s3_uri',''),
                 role=i.get('role'), split=i.get('split'), kind=i.get('kind'), folder=i.get('folder')) for i in items]
    cur.execute('INSERT INTO run_files (run_id,file_key,s3_uri,role,split,kind,folder) '
                'SELECT run_id,file_key,s3_uri,role,split,kind,folder FROM jsonb_to_recordset(%s::jsonb) '
                'AS r(run_id text,file_key text,s3_uri text,role text,split text,kind text,folder text) '
                'ORDER BY run_id,file_key ON CONFLICT (run_id,file_key) DO UPDATE SET '
                's3_uri=EXCLUDED.s3_uri,role=EXCLUDED.role,split=EXCLUDED.split,kind=EXCLUDED.kind,folder=EXCLUDED.folder',
                (json.dumps(rows),))


def project(cur, target, item):
    from . import catalog_sync as cs
    if target == "training_audits" and item.get("record_type", "training_audit") == "training_audit":
        r = cs._plain(item)
        version = ((r.get("train_dataset") or {}).get("dataset_version_id") or r.get("dataset_version_id")
                   or (r.get("snapshot") or {}).get("snapshot_id"))
        cur.execute("INSERT INTO training_runs (run_id,version_id,submitted_at,started_at,ended_at,"
                    "status,detector,architecture,execution_arn,model_package_arn,record,files_synced) "
                    "VALUES (%s,%s,%s::timestamptz,coalesce(%s::timestamptz,now()),%s::timestamptz,"
                    "%s,%s,%s,%s,%s,%s::jsonb,true) ON CONFLICT (run_id) DO UPDATE SET "
                    "version_id=EXCLUDED.version_id,submitted_at=EXCLUDED.submitted_at,"
                    "started_at=EXCLUDED.started_at,ended_at=EXCLUDED.ended_at,status=EXCLUDED.status,"
                    "detector=EXCLUDED.detector,architecture=EXCLUDED.architecture,"
                    "execution_arn=EXCLUDED.execution_arn,model_package_arn=EXCLUDED.model_package_arn,"
                    "record=EXCLUDED.record,files_synced=true",
                    (r["id"], version, cs._when(r.get("submitted_at")), cs._when(r.get("started_at") or r.get("submitted_at")),
                     cs._when(r.get("ended_at")), r.get("status"), r.get("detector_slug"), r.get("architecture"),
                     r.get("execution_arn"), (r.get("model") or {}).get("model_package_arn"), json.dumps(cs._slim(r))))
    elif target == "training_files":
        # Files can be recorded while their audit document is being built.
        cur.execute("INSERT INTO training_runs (run_id,status,files_synced) SELECT %s,'Recording',true "
                    "WHERE NOT EXISTS (SELECT 1 FROM training_runs WHERE run_id=%s) ON CONFLICT DO NOTHING",
                    (item["audit_id"], item["audit_id"]))
        cur.execute("INSERT INTO run_files (run_id,file_key,s3_uri,role,split,kind,folder) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (run_id,file_key) DO UPDATE SET "
                    "s3_uri=EXCLUDED.s3_uri,role=EXCLUDED.role,split=EXCLUDED.split,kind=EXCLUDED.kind,folder=EXCLUDED.folder",
                    (item["audit_id"], item["file_key"], item.get("s3_uri", ""), item.get("role"),
                     item.get("split"), item.get("kind"), item.get("folder")))
    elif target == "dataset_definitions":
        row = cs._plain(item)
        ident = row.get("version_id") or row.get("snapshot_id")
        # The manifest is immutable. Its members are materialized at the same
        # time as the definition rather than waiting for a five-minute sync.
        cs._one_version(cur, row, ident, {})


def remove(cur, target, item):
    if not item:
        return
    if target == "training_files":
        cur.execute("DELETE FROM run_files WHERE run_id=%s AND file_key=%s", (item["audit_id"], item["file_key"]))
    elif target == "training_audits" and item.get("record_type", "training_audit") == "training_audit":
        cur.execute("DELETE FROM run_files WHERE run_id=%s", (item["id"],))
        cur.execute("DELETE FROM training_runs WHERE run_id=%s", (item["id"],))
    elif target == "dataset_definitions":
        ident = item.get("version_id") or item["snapshot_id"]
        cur.execute("DELETE FROM dataset_members WHERE version_id=%s", (ident,))
        cur.execute("DELETE FROM dataset_versions WHERE id=%s", (ident,))
