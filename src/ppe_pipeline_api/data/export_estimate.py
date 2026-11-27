"""Read-only package storage estimates; never downloads or starts exports."""
from concurrent.futures import ThreadPoolExecutor
import json
import math
from botocore.exceptions import ClientError

from . import catalog_view as cv, catalog_metadata
from .. import storage


def estimate(kind, ident, *, destination="local", bucket="", prefix="", format=None,
             include_images=True, include_model=False, include_metadata=False,
             metadata_groups=None, model_package_arn="", model_package_arns=None):
    if kind not in ("run", "version", "experiment"):
        raise ValueError("Export a run or a dataset version.")
    format = format or ("zip" if destination == "local" else "folder")
    if format not in ("zip", "folder"):
        raise ValueError("Choose ZIP or folder.")
    entries, names, label = cv._plan(kind, ident)
    record = {}
    if kind == 'experiment':
        from . import experiment_export
        experiment = experiment_export.bundle(ident, model_package_arns) if model_package_arns else experiment_export.context(ident, model_package_arn)
        record = {'weights_uri': experiment['weights_uri']}
        label = experiment['label']
    if kind == "run":
        saved = cv._rows("SELECT model_package_arn, record FROM training_runs WHERE run_id = %s", (ident,))
        if not saved or (model_package_arn and saved[0].get("model_package_arn") != model_package_arn):
            raise ValueError("The selected model does not match this training run.")
        record = saved[0].get("record") or {}
        if isinstance(record, str):
            record = json.loads(record)
    wanted = list(entries) if include_images else []
    if include_model:
        checkpoints = experiment.get('members', [experiment]) if kind == 'experiment' else [{'label': label, 'weights_uri': record.get('weights_uri', '')}]
        for checkpoint in checkpoints:
            uri = checkpoint['weights_uri']
            if not uri.startswith('s3://') or not uri.lower().endswith('.pt'):
                raise ValueError('No .pt checkpoint has been published for this run.')
            wanted.append((f"models/{checkpoint['label']}.pt", uri, {'role': 'model'}))
    wanted = [e for e in wanted if e[0]]

    def head(uri):
        try:
            b, k = cv._split_uri(uri)
            return storage.s3.head_object(Bucket=b, Key=k)
        except ClientError as exc:
            if exc.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
                return None
            raise

    def size(key):
        uri, stored, expected = key
        try:
            obj = head(stored or uri)
            if stored and expected:
                if obj is None:
                    return None, 'missing'
                if obj.get('ETag', '').strip('"') != expected.strip('"'):
                    return None, 'changed'
            elif stored and obj is None:
                obj = head(uri)
                frozen = stored.rsplit('/', 1)[-1].rsplit('.', 1)[0]
                if obj is None:
                    return None, 'missing'
                if obj.get('ETag', '').strip('"') != frozen:
                    return None, 'changed'
            if obj is None:
                return None, 'missing'
            return (int(obj['ContentLength']), '') if 'ContentLength' in obj else (None, 'check_failed')
        except Exception:
            # Access/network errors are not evidence that a file is missing.
            return None, 'check_failed'

    keys = [(uri, row.get("stored", ""), row.get("stored_etag", "")) for _, uri, row in wanted]
    unique = list(dict.fromkeys(keys))
    with ThreadPoolExecutor(max_workers=16) as pool:
        checks = dict(zip(unique, pool.map(size, unique)))
    sizes = {key: result[0] for key, result in checks.items()}
    unavailable = [{"path": path, "uri": row.get("stored") or uri, "reason": checks[key][1],
                    "role": row.get("role", "data")}
                   for (path, uri, row), key in zip(wanted, keys) if checks[key][1]]
    known = sum(sizes[key] or 0 for key in keys)
    unknown = sum(sizes[key] is None for key in keys)
    model_bytes = sum(sizes[key] or 0 for entry, key in zip(wanted, keys) if entry[2].get("role") == "model")
    # Budget generated manifests, YAML, recipe, metadata and ZIP headers.
    # This is an estimate, not a claim of exact compressed/archive size.
    manifest = 1024 + sum(len(json.dumps(row, default=str).encode()) + len(uri.encode()) + len(path.encode()) + 256 for path, uri, row in entries)
    generated = manifest + len(json.dumps(names, default=str).encode()) + 8192
    generated += len(catalog_metadata.dumps(catalog_metadata.clean(record.get("reproducibility") or {})).encode())
    if include_metadata:
        metadata = (experiment_export.bundle_metadata(ident, model_package_arns) if model_package_arns else experiment_export.metadata(ident, model_package_arn)) if kind == 'experiment' else catalog_metadata.describe(kind, ident)
        if metadata_groups is not None:
            models = metadata.get('models')
            metadata = catalog_metadata.select_groups(metadata, metadata_groups)
            if models:
                metadata['models'] = {key: catalog_metadata.select_groups(value, metadata_groups) for key, value in models.items()}
        metadata_bytes = len(catalog_metadata.dumps(metadata).encode()) + 8192
        generated += metadata_bytes
        for group in ("Evaluation", "Training history"):
            if group in metadata.get("groups", {}):
                generated += len(catalog_metadata.dumps(metadata["groups"][group]).encode())
        if include_model:
            generated += metadata_bytes * max(1, len(model_package_arns or [])) + (manifest if "File provenance" in metadata.get("groups", {}) else 0)
    if format == "zip":
        generated += sum(256 + 2 * len(path.encode()) for path, _, _ in wanted) + 4096
    estimated = known + generated
    return {"source_bytes": known, "model_bytes": model_bytes, "generated_bytes_estimate": generated,
            "estimated_bytes": estimated if not unknown else None,
            "recommended_bytes": math.ceil(estimated * 1.1) if not unknown else None,
            "files": len(wanted), "unknown_files": unknown, "format": format,
            "unavailable_files": unavailable,
            "missing_files": sum(item["reason"] == "missing" for item in unavailable),
            "changed_files": sum(item["reason"] == "changed" for item in unavailable),
            "check_failed_files": sum(item["reason"] == "check_failed" for item in unavailable),
            "note": "Source sizes checked in S3. Generated files are estimated; suggested space includes 10% headroom. ZIP exports use no compression. Sizes can change before export."}
