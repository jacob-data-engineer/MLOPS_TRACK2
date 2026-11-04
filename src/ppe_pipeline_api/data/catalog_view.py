"""Reading the catalog, and exporting from it (2026-10-29).

What the Data Catalog page shows: every training run (by run number) and
every dataset version and snapshot (by version number), the files each
used or holds, and exports of either as a zip.

An EXPORT is built in the background, inside AWS: every file is fetched
from S3 and streamed straight into a zip that is itself written to S3 in
64 MB parts -- a training set is gigabytes, too big for a request or for
the browser to assemble. The zip carries the training layout
(`images/<split>/`, `labels/<split>/`), a `data.yaml` when the class
names are known, and `manifest.csv` naming every file's source. A file
that can no longer be read is listed in the manifest as missing rather
than failing the whole export.
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import re
import threading
import time
import uuid
import zipfile
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor

from .. import pull_control as pc
from .. import storage
from . import catalog, catalog_metadata

PAGE = 100
#: Exports building at once. Each streams gigabytes; two is plenty.
_building = threading.BoundedSemaphore(2)
#: Bytes held before a part is sent. S3 parts must be >= 5 MB.
PART = 64 * 1024 * 1024
#: Files fetched in parallel while a zip is written.
FETCHERS = 16


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    conn = catalog.connect()
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        names = [d[0] for d in cur.description]
        return [dict(zip(names, row)) for row in cur.fetchall()]
    finally:
        conn.close()


def _iso(rows: list[dict]) -> list[dict]:
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (_dt.datetime, _dt.date)):
                row[k] = v.isoformat()
    return rows


# --- browsing ---------------------------------------------------------------

def run_numbers(run_ids: list[str]) -> dict[str, int]:
    """Stable catalog numbers for history entries, independent of display order."""
    if not run_ids or not catalog.configured():
        return {}
    placeholders = ", ".join(["%s"] * len(run_ids))
    return {row["run_id"]: row["run_number"] for row in _rows(
        f"SELECT run_id, run_number FROM training_runs WHERE run_id IN ({placeholders})",
        tuple(run_ids))}

def runs(limit: int = PAGE, offset: int = 0) -> dict:
    total = _rows("SELECT count(*) AS n FROM training_runs")[0]["n"]
    rows = _rows(
        "SELECT r.run_number, r.run_id, r.submitted_at, r.ended_at, r.status, r.detector,"
        " r.architecture, r.version_id, v.version_number, v.kind AS version_kind,"
        " r.model_package_arn,"
        " (SELECT count(*) FROM run_files f WHERE f.run_id = r.run_id AND f.kind = 'image')"
        "   AS images"
        " FROM training_runs r LEFT JOIN dataset_versions v ON v.id = r.version_id"
        " ORDER BY r.run_number DESC LIMIT %s OFFSET %s",
        (min(int(limit), 500), max(0, int(offset))))
    return {"total": total, "runs": _iso(rows)}


def versions(limit: int = PAGE, offset: int = 0) -> dict:
    total = _rows("SELECT count(*) AS n FROM dataset_versions")[0]["n"]
    rows = _rows(
        "SELECT v.version_number, v.id, v.kind, v.created_at, v.state, v.train_images,"
        " v.val_images, v.class_names,"
        " (SELECT count(*) FROM dataset_members m WHERE m.version_id = v.id) AS members,"
        " (SELECT array_agg(r.run_number ORDER BY r.run_number) FROM training_runs r"
        "   WHERE r.version_id = v.id) AS runs"
        " FROM dataset_versions v ORDER BY v.version_number DESC LIMIT %s OFFSET %s",
        (min(int(limit), 500), max(0, int(offset))))
    return {"total": total, "versions": _iso(rows)}


def _run_filter(run_id: str, split: str, role: str, kind: str) -> tuple[str, tuple]:
    where, params = ["run_id = %s"], [run_id]
    for column, value in (("split", split), ("role", role), ("kind", kind)):
        if value:
            where.append(f"{column} = %s")
            params.append(value)
    return " AND ".join(where), tuple(params)


def run_files(run_id: str, *, split: str = "", role: str = "", kind: str = "image",
              limit: int = PAGE, offset: int = 0) -> dict:
    where, params = _run_filter(run_id, split, role, kind)
    total = _rows(f"SELECT count(*) AS n FROM run_files WHERE {where}", params)[0]["n"]
    rows = _rows(f"SELECT s3_uri AS uri, role, split, kind, folder FROM run_files WHERE {where}"
                 " ORDER BY file_key LIMIT %s OFFSET %s",
                 params + (min(int(limit), 500), max(0, int(offset))))
    return {"total": total, "files": rows}


def version_files(version_id: str, *, split: str = "", limit: int = PAGE,
                  offset: int = 0) -> dict:
    where, params = "version_id = %s", (version_id,)
    if split:
        where, params = where + " AND split = %s", params + (split,)
    total = _rows(f"SELECT count(*) AS n FROM dataset_members WHERE {where}", params)[0]["n"]
    rows = _rows("SELECT image_uri AS uri, label_uri, split, included, reason, decay_weight,"
                 f" week FROM dataset_members WHERE {where}"
                 " ORDER BY image_uri LIMIT %s OFFSET %s",
                 params + (min(int(limit), 500), max(0, int(offset))))
    return {"total": total, "files": rows}


def file_list_csv(kind: str, ident: str) -> str:
    """Every file of a run or a version, as CSV -- the list without the
    bytes, for a spreadsheet or a script."""
    out = io.StringIO()
    writer = csv.writer(out)
    if kind == "run":
        writer.writerow(["s3_uri", "role", "split", "kind", "folder"])
        for r in _rows("SELECT s3_uri, role, split, kind, folder FROM run_files"
                       " WHERE run_id = %s ORDER BY file_key", (ident,)):
            writer.writerow([r["s3_uri"], r["role"], r["split"], r["kind"], r["folder"]])
    else:
        writer.writerow(["image_uri", "label_uri", "split", "included", "reason",
                         "decay_weight", "week"])
        for r in _rows("SELECT image_uri, label_uri, split, included, reason, decay_weight,"
                       " week FROM dataset_members WHERE version_id = %s ORDER BY image_uri",
                       (ident,)):
            writer.writerow([r["image_uri"], r["label_uri"], r["split"], r["included"],
                             r["reason"], r["decay_weight"], r["week"]])
    return out.getvalue()


# --- export ------------------------------------------------------------------

class _MultipartSink:
    """A write-only file that is an S3 multipart upload. No seek or tell:
    `zipfile` then writes a streaming zip (data descriptors), which is
    exactly what an upload in one pass needs."""

    def __init__(self, bucket: str, key: str):
        self.bucket, self.key = bucket, key
        self.upload = storage.s3.create_multipart_upload(
            Bucket=bucket, Key=key, ContentType="application/zip")["UploadId"]
        self.parts, self.buffer, self.size = [], bytearray(), 0

    def write(self, data) -> int:
        self.buffer += data
        self.size += len(data)
        if len(self.buffer) >= PART:
            self._send()
        return len(data)

    def flush(self) -> None:
        pass

    def _send(self) -> None:
        number = len(self.parts) + 1
        etag = storage.s3.upload_part(Bucket=self.bucket, Key=self.key, UploadId=self.upload,
                                      PartNumber=number, Body=bytes(self.buffer))["ETag"]
        self.parts.append({"PartNumber": number, "ETag": etag})
        self.buffer = bytearray()

    def finish(self) -> None:
        if self.buffer or not self.parts:
            self._send()
        storage.s3.complete_multipart_upload(Bucket=self.bucket, Key=self.key,
                                             UploadId=self.upload,
                                             MultipartUpload={"Parts": self.parts})

    def abort(self) -> None:
        try:
            storage.s3.abort_multipart_upload(Bucket=self.bucket, Key=self.key,
                                              UploadId=self.upload)
        except Exception:                                          # noqa: BLE001
            pass


def _split_uri(uri: str) -> tuple[str, str]:
    bucket, _, key = uri[5:].partition("/")
    return bucket, key


def _fetch(uri: str) -> bytes | None:
    if not uri or not uri.startswith("s3://"):
        return None
    bucket, key = _split_uri(uri)
    try:
        return storage.s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "NotFound", "404"):
            return None
        raise


def _plan(kind: str, ident: str) -> tuple[list[tuple[str, str, dict]], dict | None, str]:
    """[(zip path, source uri, manifest row)], the class names, a label."""
    entries, taken = [], set()

    def unique(path: str, uri: str) -> str:
        """A path nothing else in the zip uses. A renamed image and its
        label get the SAME suffix -- taken from the image's address -- so
        they still pair after renaming."""
        if path not in taken:
            taken.add(path)
            return path
        image_uri = uri
        if "/labels/" in "/" + uri:
            head, _, tail = ("/" + uri).rpartition("/labels/")
            image_uri = f"{head}/images/{tail}"[1:]
        pair_key = image_uri.rsplit(".", 1)[0]
        stem, dot, ext = path.rpartition(".")
        path = f"{stem}_{hashlib.sha1(pair_key.encode()).hexdigest()[:8]}{dot}{ext}"
        taken.add(path)
        return path

    if kind == "run":
        run = _rows("SELECT run_number, version_id FROM training_runs WHERE run_id = %s", (ident,))
        if not run:
            raise ValueError(f"No run {ident}.")
        names = None
        if run[0]["version_id"]:
            got = _rows("SELECT class_names FROM dataset_versions WHERE id = %s",
                        (run[0]["version_id"],))
            names = got[0]["class_names"] if got else None
        for f in _rows("SELECT s3_uri, role, split, kind FROM run_files WHERE run_id = %s"
                       " ORDER BY file_key", (ident,)):
            if not str(f["s3_uri"] or "").startswith("s3://"):
                raise ValueError("This run has incomplete file addresses. Repair its audit records before exporting.")
            leaf = f["s3_uri"].rsplit("/", 1)[-1]
            side = "eval/" if f["role"] == "eval" else ""
            sub = "images" if f["kind"] == "image" else "labels"
            path = unique(f"{side}{sub}/{f['split'] or 'train'}/{leaf}", f["s3_uri"])
            entries.append((path, f["s3_uri"], {"path": path, "source": f["s3_uri"],
                                                "role": f["role"], "split": f["split"]}))
        return entries, names, f"run-{run[0]['run_number']}"

    version = _rows("SELECT version_number, class_names FROM dataset_versions WHERE id = %s",
                    (ident,))
    if not version:
        raise ValueError(f"No dataset version {ident}.")
    catalog.ensure_schema_once()
    for m in _rows("SELECT image_uri, label_uri, split, included, reason, decay_weight, week,"
                   " stored_image_uri, stored_label_uri"
                   " FROM dataset_members WHERE version_id = %s ORDER BY image_uri", (ident,)):
        row = {"split": m["split"], "included": m["included"], "reason": m["reason"],
               "decay_weight": m["decay_weight"], "week": m["week"],
               "stored": m["stored_image_uri"] or ""}
        if not m["included"]:
            # Recorded in the manifest, not packed: it was not trained on.
            entries.append(("", m["image_uri"], {**row, "path": "", "source": m["image_uri"]}))
            continue
        leaf = m["image_uri"].rsplit("/", 1)[-1]
        path = unique(f"images/{m['split'] or 'train'}/{leaf}", m["image_uri"])
        entries.append((path, m["image_uri"], {**row, "path": path, "source": m["image_uri"]}))
        if m["label_uri"]:
            label_path = path.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
            entries.append((label_path, m["label_uri"],
                            {**row, "path": label_path, "source": m["label_uri"],
                             "stored": m["stored_label_uri"] or ""}))
    return entries, version[0]["class_names"], f"version-{version[0]['version_number']}"


def _numbered(names) -> dict:
    """Class names as {index: name}, whichever way they were recorded:
    `{"0": "helmet"}`, `["helmet", ...]` (index = position), or either of
    those as JSON text. {} when there are none."""
    if isinstance(names, str):
        try:
            names = json.loads(names)
        except ValueError:
            return {}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        try:
            return {int(k): str(v) for k, v in names.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def _export_bucket() -> tuple[str, str]:
    """Into the central store when one is set, else this app's bucket."""
    store = (pc.read_config().get("central_store") or {})
    bucket = str(store.get("bucket") or storage.BUCKET or "track2-s3")
    folder = str(store.get("folder") or "").strip("/")
    return bucket, f"{folder}/_exports" if folder else "_exports"


class _ExportStopped(Exception):
    pass


_stop_events: dict[str, threading.Event] = {}


def stop_export(export_id: str) -> dict:
    conn = catalog.connect()
    try:
        cur = conn.cursor()
        cur.execute("UPDATE exports SET state = 'stopping' WHERE id = %s AND state = 'building'", (export_id,))
        cur.execute("SELECT state FROM exports WHERE id = %s", (export_id,))
        row = cur.fetchone()
        conn.commit()
    finally:
        conn.close()
    if not row:
        raise ValueError("Export not found.")
    if row[0] == "stopping" and export_id in _stop_events:
        _stop_events[export_id].set()
    return {"id": export_id, "state": row[0]}


class _ObjectWriter:
    """Write actual files into a unique S3 export folder, in the ZIP layout."""

    def __init__(self, bucket: str, prefix: str):
        self.bucket, self.prefix, self.size = bucket, prefix, 0
        self.keys = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def writestr(self, path, body):
        if isinstance(path, zipfile.ZipInfo):
            path = path.filename
        if isinstance(body, str):
            body = body.encode("utf-8")
        storage.s3.put_object(Bucket=self.bucket, Key=f"{self.prefix}/{path}", Body=body)
        self.keys.append(f"{self.prefix}/{path}")
        self.size += len(body)


def _destination(destination: str, bucket: str, prefix: str) -> tuple[str, str]:
    if destination not in ("local", "s3"):
        raise ValueError("Choose local PC or S3 as the export destination.")
    if destination == "local":
        return _export_bucket()
    bucket, prefix = bucket.strip(), prefix.strip().strip("/")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]", bucket):
        raise ValueError("Enter an S3 bucket name, without s3:// or a folder path.")
    if any(part in (".", "..") for part in prefix.split("/")) or "\\" in prefix or any(ord(c) < 32 for c in prefix):
        raise ValueError("Enter a valid S3 folder prefix, without dot segments or backslashes.")
    return bucket, prefix


def _build(export_id: str, kind: str, ident: str, label: str, entries: list, names,
           destination: str = "local", bucket: str = "", prefix: str = "", format: str | None = None,
           metadata: dict | None = None) -> None:
    sink = None
    output = None
    completed = False
    event = _stop_events.setdefault(export_id, threading.Event())
    checked = 0.0

    def check(force=False):
        nonlocal checked
        if event.is_set():
            raise _ExportStopped()
        if force or time.monotonic() - checked >= 1:
            rows = _rows("SELECT state FROM exports WHERE id = %s", (export_id,))
            if not rows or rows[0]["state"] != "building":
                event.set()
                raise _ExportStopped()
            checked = time.monotonic()

    def fetch(entry):
        if event.is_set():
            raise _ExportStopped()
        # The original first; the copy kept in the central store at freeze
        # when the original has gone (UAT-01).
        body = _fetch(entry[1])
        if body is None and entry[2].get("stored"):
            body = _fetch(entry[2]["stored"])
        return body

    try:
        check(True)
        if not bucket:
            bucket, prefix = _export_bucket()
        folder = "/".join(p for p in (prefix, f"{label}-{export_id}") if p)
        key = f"{folder}/{label}.zip"
        format = format or ("zip" if destination == "local" else "folder")
        if format == "zip":
            sink = _MultipartSink(bucket, key)
            output = zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True)
        else:
            output = _ObjectWriter(bucket, folder)
        manifest = io.StringIO()
        writer = csv.DictWriter(manifest, fieldnames=["path", "source", "role", "split",
                                                      "included", "reason", "decay_weight",
                                                      "week", "missing", "bytes", "sha256"],
                                extrasaction="ignore")
        writer.writeheader()
        # Include unselected/excluded inventory before enriching the checkpoint.
        for path, uri, row in entries:
            if not path:
                writer.writerow(row)
        packed, missing = 0, 0
        with output as zf:
            wanted = [e for e in entries if e[0]]
            with ThreadPoolExecutor(max_workers=FETCHERS) as pool:
                for start in range(0, len(wanted), FETCHERS * 4):
                    check()
                    chunk = wanted[start:start + FETCHERS * 4]
                    for (path, uri, row), body in zip(chunk, pool.map(fetch, chunk)):
                        check()
                        if body is None:
                            missing += 1
                            writer.writerow({**row, "missing": True})
                            continue
                        if metadata and path == f"models/{label}.pt":
                            check(True)
                            body = catalog_metadata.embed(body, metadata, manifest.getvalue() if "File provenance" in metadata["groups"] else None)
                        zf.writestr(zipfile.ZipInfo(path, date_time=(2026, 1, 1, 0, 0, 0)), body)
                        packed += 1
                        writer.writerow({**row, "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()})
            numbered = _numbered(names)
            check(True)
            if numbered:
                lines = "\n".join(f"  {k}: {v}" for k, v in sorted(numbered.items()))
                val_path = "eval/images/val" if any(p.startswith("eval/images/val/") for p, _, _ in entries) else "images/val"
                zf.writestr("data.yaml", f"path: .\ntrain: images/train\nval: {val_path}\n"
                                         f"names:\n{lines}\n")
            zf.writestr("manifest.csv", manifest.getvalue())
            if metadata:
                check(True)
                metadata["export_result"] = {"files_written": packed, "missing_files": missing,
                                             "completed_at": _dt.datetime.now(_dt.timezone.utc).isoformat()}
                zf.writestr("metadata.json", catalog_metadata.dumps(metadata))
                if "Evaluation" in metadata["groups"]:
                    zf.writestr("metrics.json", catalog_metadata.dumps(metadata["groups"]["Evaluation"]))
                if "Training history" in metadata["groups"]:
                    zf.writestr("training-history.json", catalog_metadata.dumps(metadata["groups"]["Training history"]))
                if not numbered:
                    zf.writestr("data.yaml", "path: .\ntrain: images/train\nval: images/val\nnames: {}\n# Class mapping was not recorded.\n")
                zf.writestr("README.txt", "Track2 export\nmetadata.json: recorded run/dataset information; unavailable fields are listed.\n"
                            "manifest.csv: source addresses, relative paths, missing files, byte sizes and SHA-256 checksums.\n"
                            "models/*.pt: a copy of the model with extra/track2_metadata.json and extra/track2_manifest.csv embedded.\n"
                            "The embedded manifest describes dataset files; the outer manifest includes the enriched model checksum.\n"
                            "Open metadata.json, the enriched .pt or this ZIP in the Data Catalog information viewer.\n"
                            "Only files selected for export and still available are included. Original model files are unchanged.\n")
        if sink:
            check(True)
            sink.finish()
            completed = True
        check(True)
        state = "partial" if missing else "ready"
        saved = _finish(export_id, "UPDATE exports SET state = %s, finished_at = now(), zip_uri = %s,"
                           " bytes = %s, files = %s, missing = %s, destination_uri = %s WHERE id = %s AND state = 'building'",
                (state, f"s3://{bucket}/{key}" if sink else None,
                 sink.size if sink else output.size, packed, missing,
                 f"s3://{bucket}/{key}" if sink else f"s3://{bucket}/{folder}/", export_id))
        if saved is False:
            raise _ExportStopped()
    except _ExportStopped:
        try:
            if sink:
                if completed:
                    storage.s3.delete_object(Bucket=bucket, Key=key)
                else:
                    storage.s3.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=sink.upload)
            elif isinstance(output, _ObjectWriter):
                for written_key in output.keys:
                    storage.s3.delete_object(Bucket=bucket, Key=written_key)
            _finish(export_id, "UPDATE exports SET state = 'cancelled', finished_at = now(), files = 0, bytes = 0, zip_uri = NULL WHERE id = %s", (export_id,))
        except Exception as exc:
            _finish(export_id, "UPDATE exports SET state = 'failed', finished_at = now(), error = %s WHERE id = %s", (f"Stopped; cleanup failed: {exc}", export_id))
    except Exception as exc:                                       # noqa: BLE001
        if sink:
            sink.abort()
        _finish(export_id, "UPDATE exports SET state = 'failed', finished_at = now(), error = %s"
                           " WHERE id = %s", (f"{type(exc).__name__}: {exc}", export_id))
    finally:
        _stop_events.pop(export_id, None)
        _building.release()


def _finish(export_id: str, sql: str, params: tuple) -> bool | None:
    """Record how an export ended, on a FRESH connection: one held idle for
    a 40-minute build can be gone by the end, and the row would then say
    'building' for ever. Retried once; printed if it still fails."""
    for attempt in (1, 2):
        try:
            conn = catalog.connect()
            try:
                cur = conn.cursor()
                cur.execute(sql, params)
                conn.commit()
                return cur.rowcount > 0
            finally:
                conn.close()
        except Exception as exc:                                   # noqa: BLE001
            if attempt == 2:
                print(f"Export {export_id}: could not record its result: {exc}")


def start_export(kind: str, ident: str, *, destination: str = "local",
                 bucket: str = "", prefix: str = "", format: str | None = None,
                 include_metadata: bool = False, include_images: bool = True, include_model: bool = False,
                 metadata_groups: list[str] | None = None) -> dict:
    """Start building a zip of a run or a dataset version. Returns at once;
    `exports()` shows its progress."""
    if kind not in ("run", "version"):
        raise ValueError("Export a run or a dataset version.")
    bucket, prefix = _destination(destination, bucket, prefix)
    format = format or ("zip" if destination == "local" else "folder")
    if format not in ("zip", "folder") or (destination == "local" and format != "zip"):
        raise ValueError("Choose ZIP for a local download, or ZIP / folder for S3.")
    entries, names, label = _plan(kind, ident)
    metadata = catalog_metadata.describe(kind, ident) if include_metadata else None
    if metadata:
        metadata["export_options"] = {"images": include_images, "model": include_model,
                                      "destination": destination, "format": format,
                                      "checkpoint_embedding": "JSON and CSV stored inside the .pt archive; model tensors are unchanged"}
        if not include_images:
            entries = [("", uri, {**row, "included": False, "reason": "Not selected for export"}) for _, uri, row in entries]
        if include_model and kind == "run":
            selected = metadata["records"]["selected"]["record"]
            if isinstance(selected, str):
                selected = json.loads(selected)
            uri = str(selected.get("weights_uri") or "")
            if uri:
                path = f"models/{label}.pt"
                entries.append((path, uri, {"path": path, "source": uri, "role": "model", "included": True}))
            else:
                metadata["notes"].append("No published checkpoint was recorded for this run; no .pt is included.")
    if metadata and metadata_groups is not None:
        options = metadata["export_options"]
        metadata = catalog_metadata.select_groups(metadata, metadata_groups)
        metadata["export_options"] = options
    if not metadata and not any(path for path, _, _ in entries):
        raise ValueError("Nothing to export: no files are recorded for it.")
    if not _building.acquire(blocking=False):
        raise ValueError("Two exports are already building; try again when one finishes.")
    export_id = uuid.uuid4().hex[:12]
    if metadata:
        metadata["export_id"] = export_id
    try:
        conn = catalog.connect()
        try:
            conn.cursor().execute(
                "INSERT INTO exports (id, version_id, run_id, label, state, files, destination, destination_uri, format)"
                " VALUES (%s, %s, %s, %s, 'building', %s, %s, %s, %s)",
                (export_id, ident if kind == "version" else None,
                 ident if kind == "run" else None, label,
                 sum(1 for path, _, _ in entries if path), destination,
                 f"s3://{bucket}/" + "/".join(p for p in (prefix, f"{label}-{export_id}") if p) + "/", format))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        _building.release()
        raise
    threading.Thread(target=_build, args=(export_id, kind, ident, label, entries, names, destination, bucket, prefix, format, metadata),
                     name=f"export-{export_id}", daemon=True).start()
    return {"id": export_id, "label": label, "state": "building"}


def exports(limit: int = 50) -> list[dict]:
    return _iso(_rows("SELECT id, label, version_id, run_id, state, created_at, finished_at,"
                      " bytes, files, missing, error, destination, destination_uri, format FROM exports"
                      " ORDER BY created_at DESC LIMIT %s", (min(int(limit), 200),)))


def download_url(export_id: str) -> str:
    rows = _rows("SELECT zip_uri, label, state FROM exports WHERE id = %s", (export_id,))
    if not rows or rows[0]["state"] not in ("ready", "partial") or not rows[0]["zip_uri"]:
        raise ValueError("That export is not ready.")
    bucket, key = _split_uri(rows[0]["zip_uri"])
    return storage.s3.generate_presigned_url(
        "get_object", ExpiresIn=3600,
        Params={"Bucket": bucket, "Key": key,
                "ResponseContentDisposition": f'attachment; filename="{rows[0]["label"]}.zip"'})


def export_run_weights(run_id: str, *, destination: str = "local", bucket: str = "", prefix: str = "",
                       metadata_groups: list[str] | None = None) -> dict:
    """Export this run's published checkpoint, without loading a PyTorch pickle."""
    rows = _rows("SELECT run_number, record FROM training_runs WHERE run_id = %s", (run_id,))
    if not rows:
        raise ValueError("Run not found.")
    record = rows[0]["record"] or {}
    if isinstance(record, str):
        record = json.loads(record)
    uri = str(record.get("weights_uri") or "")
    if not uri.startswith("s3://") or not uri.lower().endswith(".pt"):
        raise ValueError("No .pt checkpoint has been published for this run yet.")
    source_bucket, source_key = _split_uri(uri)
    storage.s3.head_object(Bucket=source_bucket, Key=source_key)
    filename = f"run-{rows[0]['run_number']}.pt"
    if metadata_groups:
        metadata = catalog_metadata.select_groups(catalog_metadata.describe("run", run_id), metadata_groups)
        manifest = None
        if "File provenance" in metadata_groups:
            entries, _, _ = _plan("run", run_id)
            out = io.StringIO()
            writer = csv.DictWriter(out, fieldnames=["path", "source", "role", "split", "included", "reason"], extrasaction="ignore")
            writer.writeheader()
            for _, _, row in entries:
                writer.writerow(row)
            manifest = out.getvalue()
            metadata["groups"]["File provenance"]["checksums"] = "Not calculated: dataset files are not fetched for a model-only export."
        body = _fetch(uri)
        if body is None:
            raise ValueError("The published checkpoint file is missing.")
        body = catalog_metadata.embed(body, metadata, manifest)
        bucket, prefix = _destination(destination, bucket, prefix)
        key = "/".join(p for p in (prefix, f"run-{rows[0]['run_number']}-{uuid.uuid4().hex[:12]}", filename) if p)
        storage.s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/octet-stream")
        if destination == "local":
            return {"url": storage.s3.generate_presigned_url("get_object", ExpiresIn=3600,
                    Params={"Bucket": bucket, "Key": key, "ResponseContentDisposition": f'attachment; filename="{filename}"'})}
        return {"uri": f"s3://{bucket}/{key}"}
    if destination == "local":
        return {"url": storage.s3.generate_presigned_url(
            "get_object", ExpiresIn=3600, Params={"Bucket": source_bucket, "Key": source_key,
            "ResponseContentDisposition": f'attachment; filename="{filename}"'})}
    bucket, prefix = _destination(destination, bucket, prefix)
    key = "/".join(p for p in (prefix, f"run-{rows[0]['run_number']}-{uuid.uuid4().hex[:12]}", filename) if p)
    storage.s3.copy({"Bucket": source_bucket, "Key": source_key}, bucket, key)
    return {"uri": f"s3://{bucket}/{key}"}


def summary_json(value) -> str:
    return json.dumps(value, default=str)


def export_metadata(kind: str, ident: str, *, destination="local", bucket="", prefix="", metadata_groups=None):
    metadata = catalog_metadata.describe(kind, ident)
    if metadata_groups is not None:
        metadata = catalog_metadata.select_groups(metadata, metadata_groups)
    bucket, prefix = _destination(destination, bucket, prefix)
    filename = metadata["label"] + "-metadata.json"
    key = "/".join(p for p in (prefix, metadata["label"] + "-" + uuid.uuid4().hex[:12], filename) if p)
    storage.s3.put_object(Bucket=bucket, Key=key, Body=catalog_metadata.dumps(metadata).encode("utf-8"), ContentType="application/json")
    if destination == "s3":
        return {"uri": f"s3://{bucket}/{key}"}
    return {"url": storage.s3.generate_presigned_url("get_object", ExpiresIn=3600, Params={"Bucket": bucket, "Key": key,
                  "ResponseContentDisposition": f'attachment; filename="{filename}"'})}
