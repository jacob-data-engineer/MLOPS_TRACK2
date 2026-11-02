"""Copy the RECORDS from DynamoDB into the catalog (2026-10-29).

Records -- what happened, written once and then read -- move to SQL; live
operational state (settings, claims, queues, pull progress) stays in
DynamoDB, where the Lambdas reach it. This is stages 1 and 2 of that move:

  1. every new or changed record is copied into the catalog on the app's
     tick (every SYNC_MINUTES), and
  2. the first sync copies the whole history -- every snapshot and dataset
     version with its files, every training run with the files it used.

DynamoDB stays the source the app reads until production runs this code;
switching reads over is stage 3. So the copy is one way, idempotent
(upserts, ON CONFLICT DO NOTHING) and NEVER removes anything: a snapshot
deleted from DynamoDB stays in the catalog, marked `deleted`.

A sync rather than a hook at each write: a run's record is written from
about ten places (queued, started, cost, metrics, model ...), and the
catalog must never be able to break one of them by being slow or down.

Numbers: rows are copied oldest first, so `version_number` and
`run_number` follow the order things happened in.
"""

from __future__ import annotations

import datetime as _dt
import decimal
import json
import os
import threading

import boto3

from .. import storage
from ..storage import IMAGE_EXTENSIONS, REGION
from . import catalog
from . import dataset_snapshots as ds

AUDIT_TABLE = os.environ.get("TRAINING_AUDIT_TABLE", "track2-yolo-training-audit")
FILES_TABLE = os.environ.get("TRAINING_FILES_TABLE", "track2-yolo-training-files")
#: How often the tick copies what is new.
SYNC_MINUTES = 5
#: Rows per INSERT. pg8000 allows 65,535 parameters per statement.
BATCH = 2000
#: A run whose record will not change again; its files are copied once.
FINISHED = ("Succeeded", "Failed", "Stopped", "Stopping")
#: Lists longer than this inside a record are replaced by their length:
#: a run's record carries every training file name, and those are copied
#: row by row into `run_files` instead.
LIST_CAP = 200

_running = threading.Lock()
#: The schema is checked (and migrated) once per process, not every tick.
_schema_checked = False


def _plain(value):
    """A DynamoDB value as ordinary JSON-able Python."""
    if isinstance(value, decimal.Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, set):
        return sorted(_plain(v) for v in value)
    return value


def _slim(value):
    """The record without its long file lists (kept as counts)."""
    if isinstance(value, dict):
        return {k: _slim(v) for k, v in value.items()}
    if isinstance(value, list):
        return {"_count": len(value)} if len(value) > LIST_CAP else [_slim(v) for v in value]
    return value


def _when(value) -> str | None:
    """An ISO timestamp Postgres will accept, or None. Checked here rather
    than left to `::timestamptz`: one bad value from an old writer would
    otherwise fail the whole pass on every tick."""
    text = str(value or "").strip()
    if not text:
        return None
    try:
        _dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return text


def _int(value) -> int:
    try:
        return int(float(value or 0))
    except (TypeError, ValueError):
        return 0


def _scan(table) -> list:
    rows, kwargs = [], {}
    while True:
        page = table.scan(**kwargs)
        rows += page.get("Items") or []
        if "LastEvaluatedKey" not in page:
            return [_plain(r) for r in rows]
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]


def _label_for(image_uri: str) -> str:
    return storage.label_path_for(image_uri)


def members_of(row: dict) -> list[tuple]:
    """(image_uri, label_uri, split, week) for every image a snapshot or
    dataset version holds, read from its manifest in S3."""
    uri = row.get("manifest_uri") or (
        f"s3://{row.get('bucket')}/{row.get('manifest_key')}" if row.get("manifest_key") else "")
    if not uri.startswith("s3://"):
        return []
    bucket, key = uri[5:].split("/", 1)
    body = json.loads(storage.s3.get_object(Bucket=bucket, Key=key)["Body"].read())
    out = []
    if "train" in body or "val" in body:                 # a dataset version
        for split in ("train", "val"):
            for entry in body.get(split) or []:
                image = entry if str(entry).startswith("s3://") else f"s3://{entry}"
                out.append((image, _label_for(image), split, ""))
    else:                                                # a snapshot
        # An EMPTY prefix is deliberate: a composed dataset's names are
        # already full keys. Only a manifest with no prefix at all falls
        # back to the root.
        prefix = str(body["prefix"] if "prefix" in body
                     else (body.get("root") or row.get("root") or "")).strip("/")
        for rel in body.get("files") or []:
            rel = str(rel)
            if not rel.lower().endswith(IMAGE_EXTENSIONS):
                continue
            image = f"s3://{row.get('bucket') or bucket}/{prefix + '/' if prefix else ''}{rel}"
            split = "val" if "/val/" in f"/{rel}" else "train"
            out.append((image, _label_for(image), split, rel.split("/", 1)[0]))
    return out


def _insert(cur, head: str, rows: list, tail: str = "ON CONFLICT DO NOTHING") -> int:
    """Insert in batches; returns how many rows were actually NEW."""
    if not rows:
        return 0
    width = len(rows[0])
    row_sql = "(" + ", ".join(["%s"] * width) + ")"
    added = 0
    for start in range(0, len(rows), BATCH):
        chunk = rows[start:start + BATCH]
        cur.execute(f"{head} VALUES {', '.join([row_sql] * len(chunk))} {tail}",
                    [v for r in chunk for v in r])
        added += max(0, cur.rowcount or 0)
    return added


def _guarded(cur, counts: dict, what: str, ident: str, work) -> None:
    """One row's work inside a savepoint: a row that fails is recorded and
    skipped, instead of rolling back -- and failing again -- every other
    row on every tick."""
    cur.execute("SAVEPOINT row_work")
    try:
        work()
        cur.execute("RELEASE SAVEPOINT row_work")
    except Exception as exc:                                       # noqa: BLE001
        cur.execute("ROLLBACK TO SAVEPOINT row_work")
        counts.setdefault("skipped", []).append(f"{what} {ident}: {type(exc).__name__}: {exc}"[:300])


def _sync_versions(cur, counts: dict) -> set:
    rows = sorted(_scan(ds._table()), key=lambda r: str(r.get("created_at") or ""))
    seen = set()
    for row in rows:
        vid = str(row.get("snapshot_id") or "")
        if not vid:
            continue
        seen.add(vid)
        _guarded(cur, counts, "version", vid,
                 lambda row=row, vid=vid: _one_version(cur, row, vid, counts))
    counts["versions"] = len(seen)
    # Gone from DynamoDB is not gone from the record.
    cur.execute("SELECT id FROM dataset_versions WHERE coalesce(state, '') <> 'deleted'")
    for (vid,) in cur.fetchall():
        if vid not in seen:
            cur.execute("UPDATE dataset_versions SET state = 'deleted' WHERE id = %s", (vid,))
            counts["marked_deleted"] = counts.get("marked_deleted", 0) + 1
    return seen


def _one_version(cur, row: dict, vid: str, counts: dict) -> None:
    kind = "version" if row.get("record_type") == "dataset_version" else "snapshot"
    names = row.get("names") or row.get("classes")
    cur.execute(
        "INSERT INTO dataset_versions (id, created_at, purpose, class_names, settings, note,"
        " kind, state, run_id, train_images, val_images, record)"
        " VALUES (%s, coalesce(%s::timestamptz, now()), 'training', %s::jsonb, '{}'::jsonb,"
        " %s, %s, %s, %s, %s, %s, %s::jsonb)"
        " ON CONFLICT (id) DO UPDATE SET state = EXCLUDED.state, run_id = EXCLUDED.run_id,"
        " train_images = EXCLUDED.train_images, val_images = EXCLUDED.val_images,"
        " record = EXCLUDED.record",
        (vid, _when(row.get("created_at")), json.dumps(names) if names else None,
         str(row.get("note") or "") or None, kind, str(row.get("state") or "") or None,
         str(row.get("run_id") or "") or None,
         _int(row.get("train_images")), _int(row.get("val_images")),
         json.dumps(_slim(row))))
    # The manifest is read ONCE: recorded as read even when it held no
    # images, so an empty or label-only one is not fetched every tick.
    cur.execute("SELECT members_synced FROM dataset_versions WHERE id = %s", (vid,))
    if cur.fetchone()[0]:
        return
    try:
        members = members_of(row)
    except Exception as exc:                                       # noqa: BLE001
        counts.setdefault("manifests_unreadable", []).append(f"{vid}: {exc}")
        return
    if kind == "snapshot" and members:
        _repair_composed(cur, row, vid, members, counts)
    added = _insert(cur, "INSERT INTO dataset_members (version_id, image_uri, label_uri, split,"
                         " week, included)",
                    [(vid, image, label or None, split, week or None, True)
                     for image, label, split, week in members])
    cur.execute("UPDATE dataset_versions SET members_synced = true WHERE id = %s", (vid,))
    counts["members"] = counts.get("members", 0) + added


def _repair_composed(cur, row: dict, vid: str, members: list, counts: dict) -> None:
    """Correct, in place, members copied before 2026-11-02 with the root
    put in front of a composed snapshot's already-full keys. The rows are
    updated, not deleted: same files, right addresses."""
    root = str(row.get("root") or "").strip("/")
    bucket = str(row.get("bucket") or "")
    if not root or not bucket:
        return
    wrong, right = f"s3://{bucket}/{root}/", f"s3://{bucket}/"
    if members[0][0].startswith(wrong):
        return                                   # not composed: nothing was wrong
    cur.execute(
        "UPDATE dataset_members SET image_uri = %s || substring(image_uri from %s::int),"
        " label_uri = CASE WHEN label_uri LIKE %s THEN %s || substring(label_uri from %s::int)"
        " ELSE label_uri END"
        " WHERE version_id = %s AND image_uri LIKE %s",
        (right, len(wrong) + 1, wrong + "%", right, len(wrong) + 1, vid, wrong + "%"))
    if cur.rowcount:
        counts["repaired"] = counts.get("repaired", 0) + cur.rowcount


def _sync_runs(cur, versions: set, counts: dict) -> None:
    audit = boto3.resource("dynamodb", region_name=REGION).Table(AUDIT_TABLE)
    files = boto3.resource("dynamodb", region_name=REGION).Table(FILES_TABLE)
    runs = [r for r in _scan(audit) if r.get("record_type") == "training_audit"]
    runs.sort(key=lambda r: str(r.get("submitted_at") or r.get("started_at") or ""))
    for run in runs:
        rid = str(run.get("id") or "")
        if not rid:
            continue
        _guarded(cur, counts, "run", rid,
                 lambda run=run, rid=rid: _one_run(cur, files, run, rid, versions, counts))
    counts["runs"] = len(runs)


#: A run still not finished after this long is not going to be; its files
#: are copied one last time and not re-read every tick.
SETTLED = _dt.timedelta(days=2)


def _one_run(cur, files, run: dict, rid: str, versions: set, counts: dict) -> None:
    # The dataset VERSION it trained on, which the audit keeps inside the
    # dataset blocks; the snapshot only when there was no version.
    version = str((run.get("train_dataset") or {}).get("dataset_version_id")
                  or run.get("dataset_version_id")
                  or (run.get("snapshot") or {}).get("snapshot_id") or "")
    model = run.get("model") or {}
    cur.execute(
        "INSERT INTO training_runs (run_id, version_id, started_at, detector, execution_arn,"
        " model_package_arn, detail, status, submitted_at, ended_at, architecture, record)"
        " VALUES (%s, %s, coalesce(%s::timestamptz, %s::timestamptz, now()), %s, %s, %s,"
        " '{}'::jsonb, %s, %s::timestamptz, %s::timestamptz, %s, %s::jsonb)"
        " ON CONFLICT (run_id) DO UPDATE SET version_id = EXCLUDED.version_id,"
        " status = EXCLUDED.status, ended_at = EXCLUDED.ended_at,"
        " model_package_arn = EXCLUDED.model_package_arn, record = EXCLUDED.record",
        (rid, version if version in versions else None,
         _when(run.get("started_at")), _when(run.get("submitted_at")),
         str(run.get("detector_slug") or "") or None,
         str(run.get("execution_arn") or "") or None,
         str(model.get("model_package_arn") or "") or None,
         str(run.get("status") or "") or None, _when(run.get("submitted_at")),
         _when(run.get("ended_at")), str(run.get("architecture") or "") or None,
         json.dumps(_slim(run))))
    cur.execute("SELECT files_synced FROM training_runs WHERE run_id = %s", (rid,))
    if cur.fetchone()[0]:
        return
    rows, kwargs = [], {"KeyConditionExpression": "audit_id = :a",
                        "ExpressionAttributeValues": {":a": rid}}
    while True:
        page = files.query(**kwargs)
        rows += page.get("Items") or []
        if "LastEvaluatedKey" not in page:
            break
        kwargs["ExclusiveStartKey"] = page["LastEvaluatedKey"]
    added = _insert(cur, "INSERT INTO run_files (run_id, file_key, s3_uri, role, split, kind,"
                         " folder)",
                    [(rid, str(f.get("file_key")), str(f.get("s3_uri") or ""),
                      str(f.get("role") or "") or None, str(f.get("split") or "") or None,
                      str(f.get("kind") or "") or None, str(f.get("folder") or "") or None)
                     for f in rows if f.get("file_key")])
    counts["run_files"] = counts.get("run_files", 0) + added
    # Copied for good once a run will not change again: finished, or
    # submitted long enough ago that it never will. Until then each sync
    # tops it up (existing rows are skipped).
    when = _when(run.get("submitted_at") or run.get("started_at"))
    settled = False
    if when:
        moment = _dt.datetime.fromisoformat(when.replace("Z", "+00:00"))
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=_dt.timezone.utc)
        settled = _dt.datetime.now(_dt.timezone.utc) - moment > SETTLED
    if str(run.get("status") or "") in FINISHED or settled:
        cur.execute("UPDATE training_runs SET files_synced = true WHERE run_id = %s", (rid,))


def sync() -> dict:
    """One full pass. Returns what it copied. Raises only if the catalog
    cannot be reached at all; anything after that is recorded in
    `sync_runs`."""
    if not _running.acquire(blocking=False):
        return {"started": False, "reason": "A sync is already running."}
    try:
        global _schema_checked
        if not _schema_checked:
            catalog.ensure_schema(report=False)
            _schema_checked = True
        conn = catalog.connect()
        cur = conn.cursor()
        cur.execute("INSERT INTO sync_runs DEFAULT VALUES RETURNING id")
        sync_id = cur.fetchone()[0]
        conn.commit()
        counts: dict = {}
        try:
            versions = _sync_versions(cur, counts)
            conn.commit()
            _sync_runs(cur, versions, counts)
            conn.commit()
            cur.execute("UPDATE sync_runs SET finished_at = now(), ok = true, counts = %s::jsonb"
                        " WHERE id = %s", (json.dumps(counts), sync_id))
            conn.commit()
            return {"started": True, "ok": True, **counts}
        except Exception as exc:                                   # noqa: BLE001
            conn.rollback()
            cur.execute("UPDATE sync_runs SET finished_at = now(), ok = false, counts = %s::jsonb,"
                        " error = %s WHERE id = %s",
                        (json.dumps(counts), f"{type(exc).__name__}: {exc}", sync_id))
            conn.commit()
            return {"started": True, "ok": False, "error": str(exc), **counts}
        finally:
            conn.close()
    finally:
        _running.release()


def start_background() -> bool:
    """Run `sync` on its own thread, so neither a request nor the tick
    waits for it. False when one is already running."""
    if _running.locked():
        return False
    threading.Thread(target=_safe_sync, name="catalog-sync", daemon=True).start()
    return True


def _safe_sync() -> None:
    try:
        out = sync()
        print(json.dumps({"catalog_sync": {k: v for k, v in out.items()
                                           if k != "manifests_unreadable"}}, default=str))
    except Exception as exc:                                       # noqa: BLE001
        print(f"Catalog sync failed: {type(exc).__name__}: {exc}")


def due(now: _dt.datetime | None = None) -> str:
    """The current SYNC_MINUTES slot, for `claim_once`: the stable and the
    development container both tick, and one sync per slot is enough."""
    moment = now or _dt.datetime.now(_dt.timezone.utc)
    slot = moment.replace(second=0, microsecond=0,
                          minute=moment.minute - moment.minute % SYNC_MINUTES)
    return slot.isoformat()
