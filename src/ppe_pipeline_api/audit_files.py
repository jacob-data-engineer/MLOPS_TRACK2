"""Canonical per-file audit records from a frozen dataset-version manifest."""
from .storage import label_path_for


def version_file_rows(audit_id: str, manifest: dict, default_bucket: str) -> list[dict]:
    rows = []
    for split, role in (("train", "train"), ("val", "eval")):
        for index, entry in enumerate(manifest.get(split) or []):
            uri = str(entry)
            if not uri.startswith("s3://"):
                uri = "s3://" + uri
            bucket, sep, key = uri[5:].partition("/")
            if not sep or not bucket or not key:
                raise ValueError(f"Invalid image address in dataset manifest: {entry}")
            label = label_path_for(uri)
            if not label:
                raise ValueError(f"Cannot resolve annotation address for {uri}")
            for kind, address in (("image", uri), ("label", label)):
                object_bucket, _, object_key = address[5:].partition("/")
                rows.append({
                    "audit_id": audit_id, "file_key": f"{role}#{kind}#{index:06d}",
                    "role": role, "kind": kind, "split": split, "s3_uri": address,
                    "name": object_key if object_bucket == default_bucket else f"{object_bucket}/{object_key}",
                    "folder": object_key.rsplit("/", 1)[0],
                    "file_name": object_key.rsplit("/", 1)[-1],
                })
    return rows
