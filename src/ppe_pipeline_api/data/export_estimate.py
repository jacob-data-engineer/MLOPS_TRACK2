"""Read-only package storage estimates; never downloads or starts exports."""
from concurrent.futures import ThreadPoolExecutor
import json
import math

from . import catalog_view as cv, catalog_metadata
from .. import storage


def estimate(kind, ident, *, destination="local", bucket="", prefix="", format=None,
             include_images=True, include_model=False, include_metadata=False,
             metadata_groups=None, model_package_arn=""):
    if kind not in ("run", "version"):
        raise ValueError("Export a run or a dataset version.")
    format = format or ("zip" if destination == "local" else "folder")
    if format not in ("zip", "folder"):
        raise ValueError("Choose ZIP or folder.")
    entries, names, label = cv._plan(kind, ident)
    record = {}
    if kind == "run":
        saved = cv._rows("SELECT model_package_arn, record FROM training_runs WHERE run_id = %s", (ident,))
        if not saved or (model_package_arn and saved[0].get("model_package_arn") != model_package_arn):
            raise ValueError("The selected model does not match this training run.")
        record = saved[0].get("record") or {}
        if isinstance(record, str):
            record = json.loads(record)
    wanted = list(entries) if include_images else []
    if include_model:
        uri = record.get("weights_uri", "")
        if not uri.startswith("s3://") or not uri.lower().endswith(".pt"):
            raise ValueError("No .pt checkpoint has been published for this run.")
        wanted.append((f"models/{label}.pt", uri, {"role": "model"}))
    wanted = [e for e in wanted if e[0]]

    def head(uri):
        try:
            b, k = cv._split_uri(uri)
            return storage.s3.head_object(Bucket=b, Key=k)
        except Exception:
            return None

    def size(key):
        uri, stored, expected = key
        obj = head(stored or uri)
        if stored and expected:
            if obj is None or obj.get("ETag", "").strip('"') != expected.strip('"'):
                return None
        elif stored and obj is None:
            obj = head(uri)
            frozen = stored.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if obj is None or obj.get("ETag", "").strip('"') != frozen:
                return None
        return int(obj["ContentLength"]) if obj is not None and "ContentLength" in obj else None

    keys = [(uri, row.get("stored", ""), row.get("stored_etag", "")) for _, uri, row in wanted]
    unique = list(dict.fromkeys(keys))
    with ThreadPoolExecutor(max_workers=16) as pool:
        sizes = dict(zip(unique, pool.map(size, unique)))
    known = sum(sizes[key] or 0 for key in keys)
    unknown = sum(sizes[key] is None for key in keys)
    model_bytes = sum(sizes[key] or 0 for entry, key in zip(wanted, keys) if entry[2].get("role") == "model")
    # Budget generated manifests, YAML, recipe, metadata and ZIP headers.
    # This is an estimate, not a claim of exact compressed/archive size.
    manifest = 1024 + sum(len(json.dumps(row, default=str).encode()) + len(uri.encode()) + len(path.encode()) + 256 for path, uri, row in entries)
    generated = manifest + len(json.dumps(names, default=str).encode()) + 8192
    generated += len(catalog_metadata.dumps(catalog_metadata.clean(record.get("reproducibility") or {})).encode())
    if include_metadata:
        metadata = catalog_metadata.describe(kind, ident)
        if metadata_groups is not None:
            metadata = catalog_metadata.select_groups(metadata, metadata_groups)
        metadata_bytes = len(catalog_metadata.dumps(metadata).encode()) + 8192
        generated += metadata_bytes
        for group in ("Evaluation", "Training history"):
            if group in metadata.get("groups", {}):
                generated += len(catalog_metadata.dumps(metadata["groups"][group]).encode())
        if include_model:
            generated += metadata_bytes + (manifest if "File provenance" in metadata.get("groups", {}) else 0)
    if format == "zip":
        generated += sum(256 + 2 * len(path.encode()) for path, _, _ in wanted) + 4096
    estimated = known + generated
    return {"source_bytes": known, "model_bytes": model_bytes, "generated_bytes_estimate": generated,
            "estimated_bytes": estimated if not unknown else None,
            "recommended_bytes": math.ceil(estimated * 1.1) if not unknown else None,
            "files": len(wanted), "unknown_files": unknown, "format": format,
            "note": "Source sizes checked in S3. Generated files are estimated; suggested space includes 10% headroom. ZIP exports use no compression. Sizes can change before export."}
