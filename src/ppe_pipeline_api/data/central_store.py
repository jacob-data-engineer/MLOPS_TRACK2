"""Keep every file a dataset version uses, for good (UAT-01, 2026-11-04).

A dataset version records WHERE its images were when it was frozen. Those
folders are shared and change -- the golden set is re-split, a week is
re-pulled -- so an export of an older version came back with files
missing: Version 46 lost 40 of its 100 images that way.

So, when a version is frozen, every image and its label are copied into
the CENTRAL DATA STORE (the bucket and folder chosen on Data Preparation's
"Central data store" card), and the copy is recorded:

  <store>/objects/<etag><ext>      one object per distinct file content

Named by the source's S3 ETag, so a file used by many versions is stored
once, and copying the same file again is a no-op. The copy is server-side
(`copy_object`): nothing is downloaded. Copies are never deleted.

Recorded twice, so either one survives on its own:
  * `_datasets/<id>/preserved.json` beside the version -- source -> copy;
  * in the catalog, on each member row (`stored_image_uri`,
    `stored_label_uri`, `image_etag`) and on the version
    (`preserved_at`, `preserved_files`, `preserve_missing`).

Exports and re-training read the stored copy whenever the original has
gone. Versions frozen before this existed are not back-filled (out of
scope by the operator's decision): what is gone from those stays gone.
"""

from __future__ import annotations

import datetime as _dt
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from botocore.exceptions import ClientError

from .. import pull_control as pc
from .. import storage

#: Copies in flight at once.
WORKERS = 16
#: Where copies go when no central store has been chosen.
DEFAULT_FOLDER = "central-store"

_running: set[str] = set()
_lock = threading.Lock()


def location() -> tuple[str, str]:
    """(bucket, folder) of the central data store."""
    store = pc.read_config().get("central_store") or {}
    bucket = str(store.get("bucket") or storage.BUCKET or "track2-s3")
    folder = str(store.get("folder") or "").strip("/") or DEFAULT_FOLDER
    return bucket, folder


def _split(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def _etag(bucket: str, key: str) -> str | None:
    """The object's ETag; None only when it is not there. Any other failure
    (throttling, a denied read) raises: it is a copy to retry, not a file
    that is gone (R3)."""
    try:
        return storage.s3.head_object(Bucket=bucket, Key=key)["ETag"].strip('"')
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def _keep(uri: str, bucket: str, folder: str) -> tuple[str | None, str | None]:
    """Copy one object into the store; (stored uri, etag), or (None, None)
    when the source is not there to copy."""
    if not uri or not uri.startswith("s3://"):
        return None, None
    src_bucket, src_key = _split(uri)
    etag = _etag(src_bucket, src_key)
    if not etag:
        return None, None
    leaf = src_key.rsplit("/", 1)[-1]
    ext = "." + leaf.rsplit(".", 1)[1].lower() if "." in leaf else ".jpg"
    target = f"{folder}/objects/{etag}{ext}"
    if _etag(bucket, target) is None:
        storage.s3.copy_object(Bucket=bucket, Key=target,
                               CopySource={"Bucket": src_bucket, "Key": src_key})
    return f"s3://{bucket}/{target}", etag


def preserve(version: dict) -> dict:
    """Copy every file of one frozen version into the central store and
    record it. Safe to repeat: existing copies are reused."""
    from . import catalog, catalog_sync

    vid = str(version.get("version_id") or version.get("snapshot_id") or "")
    catalog.ensure_schema_once()
    bucket, folder = location()
    members = catalog_sync.members_of(version)          # (image, label, split, week)

    def one(member):
        image, label, split, _ = member
        row = {"image": image, "label": label or "", "split": split, "etag": "",
               "stored_image": "", "stored_label": "", "error": ""}
        try:
            stored_image, etag = _keep(image, bucket, folder)
            row.update(stored_image=stored_image or "", etag=etag or "")
            if label:
                row["stored_label"] = _keep(label, bucket, folder)[0] or ""
        except Exception as exc:                                   # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]
        return row

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(one, members))

    # Images AND labels (R3, independent UAT 2026-11-04): counting images
    # only reported "missing: 0" for a version whose label was never kept.
    #   images_missing / labels_missing -- not there at freeze; recorded,
    #     nothing to retry. A label absent at freeze means the image was
    #     trained as unlabeled (background), which is what the export says.
    #   failed -- a copy that errored; the version stays un-preserved so
    #     the tick's backstop (`pending`) tries it again.
    done = [r for r in rows if not r["error"]]
    kept = sum(1 for r in done if r["stored_image"])
    labels_expected = sum(1 for r in done if r["label"] and r["stored_image"])
    labels_kept = sum(1 for r in done if r["label"] and r["stored_image"] and r["stored_label"])
    counts = {"images_kept": kept, "images_missing": len(done) - kept,
              "labels_expected": labels_expected, "labels_kept": labels_kept,
              "labels_missing": labels_expected - labels_kept,
              "failed": len(rows) - len(done)}
    missing = counts["images_missing"] + counts["labels_missing"]
    record = {"version_id": vid, "store": f"s3://{bucket}/{folder}/objects/",
              "at": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
              "files": len(rows), "kept": kept, "missing": missing, **counts,
              "complete": not missing and not counts["failed"], "members": rows}

    # Beside the version, so the record outlives any database.
    manifest_uri = str(version.get("manifest_uri") or "")
    if manifest_uri.startswith("s3://"):
        m_bucket, m_key = _split(manifest_uri)
        storage.s3.put_object(Bucket=m_bucket, Key=m_key.rsplit("/", 1)[0] + "/preserved.json",
                              Body=json.dumps(record, indent=1).encode(),
                              ContentType="application/json")

    if catalog.configured():
        conn = catalog.connect()
        try:
            cur = conn.cursor()
            for r in done:
                cur.execute(
                    "UPDATE dataset_members SET stored_image_uri = %s, stored_label_uri = %s,"
                    " image_etag = %s WHERE version_id = %s AND image_uri = %s",
                    (r["stored_image"] or None, r["stored_label"] or None, r["etag"] or None,
                     vid, r["image"]))
            if counts["failed"]:
                # Not preserved yet: `pending` picks it up again.
                cur.execute("UPDATE dataset_versions SET preserve_error = %s WHERE id = %s",
                            (f"{counts['failed']} of {len(rows)} copies failed; retried on the"
                             f" next tick. First: {next(r['error'] for r in rows if r['error'])}"[:500],
                             vid))
            else:
                cur.execute("UPDATE dataset_versions SET preserved_at = now(), preserved_files = %s,"
                            " preserve_missing = %s, preserve_error = NULL WHERE id = %s",
                            (kept, missing, vid))
            conn.commit()
        finally:
            conn.close()
    return {k: v for k, v in record.items() if k != "members"}


def preserve_in_background(version: dict) -> bool:
    """Start `preserve` on its own thread, once per version at a time, so
    queueing a run never waits for the copies."""
    vid = str(version.get("version_id") or "")
    with _lock:
        if not vid or vid in _running:
            return False
        _running.add(vid)

    def work():
        try:
            out = preserve(version)
            print(json.dumps({"central_store": "preserved", **out}))
        except Exception as exc:                                   # noqa: BLE001
            print(f"Central store: could not preserve {vid}: {type(exc).__name__}: {exc}")
            _record_error(vid, f"{type(exc).__name__}: {exc}")
        finally:
            with _lock:
                _running.discard(vid)

    threading.Thread(target=work, name=f"preserve-{vid}", daemon=True).start()
    return True


def _record_error(vid: str, error: str) -> None:
    from . import catalog
    if not catalog.configured():
        return
    try:
        conn = catalog.connect()
        try:
            conn.cursor().execute("UPDATE dataset_versions SET preserve_error = %s WHERE id = %s",
                                  (error[:500], vid))
            conn.commit()
        finally:
            conn.close()
    except Exception:                                              # noqa: BLE001
        pass


#: Versions frozen before this date are not back-filled (operator's
#: decision, UAT report 2026-11-04): only what is frozen from now on.
SINCE = "2026-11-04"


def pending(limit: int = 5) -> list[dict]:
    """Versions frozen since SINCE that have no preserved copy yet -- a
    thread that died with its container, say. For the tick's backstop."""
    from . import catalog
    if not catalog.configured():
        return []
    catalog.ensure_schema_once()
    conn = catalog.connect()
    try:
        cur = conn.cursor()
        cur.execute("SELECT id, record FROM dataset_versions WHERE kind = 'version'"
                    " AND preserved_at IS NULL AND created_at >= %s::timestamptz"
                    " ORDER BY created_at LIMIT %s", (SINCE, limit))
        out = []
        for vid, record in cur.fetchall():
            row = record if isinstance(record, dict) else json.loads(record or "{}")
            out.append({**row, "version_id": vid})
        return out
    finally:
        conn.close()


def catch_up() -> int:
    """Start preserving any version the backstop finds. Returns how many."""
    return sum(1 for v in pending() if preserve_in_background(v))
