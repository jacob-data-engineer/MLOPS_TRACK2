"""Repair incomplete version audits from their saved manifests, with S3 backups.

Run inside the app container: python -m ppe_pipeline_api.data.repair_run_files
Add --apply after reviewing the dry-run report. Source datasets are never changed.
"""
import argparse
import collections
import datetime
import json
from concurrent.futures import ThreadPoolExecutor

import boto3
from boto3.dynamodb.conditions import Key
from boto3.dynamodb.types import TypeSerializer

from .. import storage
from ..audit_files import version_file_rows
from . import catalog, catalog_sync, catalog_view


def plan(run: dict) -> tuple[list[dict], dict]:
    record = run["version_record"]
    if isinstance(record, str):
        record = json.loads(record)
    uri = record.get("manifest_uri")
    if not uri or not uri.startswith("s3://"):
        raise ValueError("No saved version manifest; refusing to guess file addresses")
    bucket, key = uri[5:].split("/", 1)
    manifest = json.loads(storage.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    if manifest.get("snapshot_id") == run["version_id"] and "files" in manifest:
        members = catalog_sync.members_of(record)
        manifest = {split: [image for image, _, side, _ in members if side == split]
                    for split in ("train", "val")}
    elif manifest.get("version_id") != run["version_id"]:
        raise ValueError("Manifest version does not match run")
    expected = version_file_rows(run["run_id"], manifest, storage.BUCKET)
    counts = collections.Counter((r["role"], r["kind"]) for r in expected)
    current = catalog_view._rows(
        "SELECT role, count(*) AS n FROM run_files WHERE run_id = %s AND kind = 'image' GROUP BY role",
        (run["run_id"],))
    if {r["role"]: r["n"] for r in current} != {role: counts[(role, "image")] for role in ("train", "eval") if counts[(role, "image")]}:
        raise ValueError("Manifest image counts do not match audit; needs manual investigation")
    return expected, {"run": run["run_number"], "images": sum(r["kind"] == "image" for r in expected),
                      "labels": sum(r["kind"] == "label" for r in expected), "version": run["version_id"]}


def repair(run: dict, backup_prefix: str, apply: bool) -> dict:
    expected, report = plan(run)
    if not apply:
        return report
    rid = run["run_id"]
    dynamo = boto3.resource("dynamodb", region_name=storage.REGION)
    files = dynamo.Table(catalog_sync.FILES_TABLE)
    audits = dynamo.Table(catalog_sync.AUDIT_TABLE)
    old, kwargs = [], {"KeyConditionExpression": Key("audit_id").eq(rid)}
    while True:
        page = files.query(**kwargs)
        old.extend(page.get("Items", []))
        if "LastEvaluatedKey" not in page:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    audit = audits.get_item(Key={"id": rid}, ConsistentRead=True).get("Item")
    if not audit:
        raise ValueError("Original training audit was not found")
    serializer = TypeSerializer()
    backup = {"run_id": rid, "version_id": run["version_id"],
              "dynamo_files": [serializer.serialize(r) for r in old],
              "dynamo_audit": serializer.serialize(audit),
              "sql_files": catalog_view._rows("SELECT * FROM run_files WHERE run_id = %s", (rid,)),
              "sql_run": catalog_view._rows("SELECT * FROM training_runs WHERE run_id = %s", (rid,))}
    backup_key = f"{backup_prefix}/run-{run['run_number']}-{rid}.json"
    storage.s3.put_object(Bucket=storage.BUCKET, Key=backup_key,
                          Body=json.dumps(backup, default=str).encode(), ContentType="application/json")
    keys = {r["file_key"] for r in expected}
    with files.batch_writer() as batch:
        for row in expected:
            batch.put_item(Item=row)
        for row in old:
            if row["file_key"] not in keys:
                batch.delete_item(Key={"audit_id": rid, "file_key": row["file_key"]})
    # Verify full DynamoDB records before replacing the SQL copy.
    actual, kwargs = [], {"KeyConditionExpression": Key("audit_id").eq(rid), "ConsistentRead": True}
    while True:
        page = files.query(**kwargs)
        actual.extend(page.get("Items", []))
        if "LastEvaluatedKey" not in page:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    wanted = {r["file_key"]: r for r in expected}
    if len(actual) != len(expected) or any(r != wanted.get(r["file_key"]) for r in actual):
        raise RuntimeError("DynamoDB verification failed; backup retained, SQL not replaced")
    for role, block in (("train", "train_dataset"), ("eval", "eval_dataset")):
        count = sum(r["role"] == role and r["kind"] == "image" for r in expected)
        audits.update_item(Key={"id": rid}, UpdateExpression="SET #b.label_count = :n, #b.file_rows = :rows",
                           ExpressionAttributeNames={"#b": block},
                           ExpressionAttributeValues={":n": count, ":rows": count * 2})
    conn = catalog.connect()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM run_files WHERE run_id = %s", (rid,))
        catalog_sync._insert(cur, "INSERT INTO run_files (run_id,file_key,s3_uri,role,split,kind,folder)",
            [(rid, r["file_key"], r["s3_uri"], r["role"], r["split"], r["kind"], r["folder"]) for r in expected])
        cur.execute("UPDATE training_runs SET files_synced = true WHERE run_id = %s", (rid,))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    verified = catalog_view._rows("SELECT count(*) AS n FROM run_files WHERE run_id = %s AND s3_uri <> ''", (rid,))[0]["n"]
    if verified != len(expected):
        raise RuntimeError("SQL repair count mismatch")
    return {**report, "verified_records": verified, "backup": f"s3://{storage.BUCKET}/{backup_key}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    runs = catalog_view._rows(
        "SELECT r.run_number,r.run_id,r.version_id,v.record AS version_record FROM training_runs r"
        " JOIN dataset_versions v ON v.id=r.version_id WHERE EXISTS"
        " (SELECT 1 FROM run_files f WHERE f.run_id=r.run_id AND f.kind='image' AND f.s3_uri='')"
        " ORDER BY r.run_number")
    prefix = "_audit_repairs/" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    def work(run):
        try:
            result = repair(run, prefix, args.apply)
        except Exception as exc:
            result = {"run": run["run_number"], "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result), flush=True)
        return result
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(work, runs))
    if any("error" in r for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
