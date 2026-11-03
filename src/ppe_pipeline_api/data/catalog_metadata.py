"""Readable catalog metadata and pickle-free checkpoint enrichment/inspection."""
from __future__ import annotations

import datetime as dt
import io
import json
import re
import zipfile

SCHEMA = "track2.metadata.v1"
MAX_JSON = 8 * 1024 * 1024
MAX_MANIFEST = 16 * 1024 * 1024


def clean(value):
    """Do not publish credentials from saved configuration records."""
    if isinstance(value, dict):
        return {str(k): ("[redacted]" if re.search(
            r"password|secret|token|credential|access.?key|authorization|private.?key", str(k), re.I)
            else clean(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, str) and re.search(r"[?&](?:X-Amz-|Signature=|token=)", value, re.I):
        return value.split("?", 1)[0] + "?[redacted]"
    return value


def dumps(value):
    return json.dumps(value, indent=2, default=str, ensure_ascii=False)


def describe(kind: str, ident: str) -> dict:
    from .catalog_view import _rows
    if kind not in ("run", "version"):
        raise ValueError("Select a run or dataset version.")
    rows = _rows("SELECT * FROM training_runs WHERE run_id = %s" if kind == "run"
                 else "SELECT * FROM dataset_versions WHERE id = %s", (ident,))
    if not rows:
        raise ValueError("Selected record was not found.")
    row = rows[0]
    record = row.get("record") or {}
    if isinstance(record, str):
        record = json.loads(record)
    dataset = row if kind == "version" else None
    if kind == "run" and row.get("version_id"):
        found = _rows("SELECT * FROM dataset_versions WHERE id = %s", (row["version_id"],))
        dataset = found[0] if found else None
    number = row.get("run_number" if kind == "run" else "version_number")
    label = f"{kind}-{number}"

    def pick(*keys):
        return {k: record[k] for k in keys if record.get(k) is not None}

    def columns(*keys):
        return {k: row[k] for k in keys if row.get(k) is not None}

    groups = {
        "Identity": {"kind": kind, "id": ident, "number": number, **pick("detector_slug", "source", "notes")},
        "Timestamps": columns("created_at", "submitted_at", "started_at", "ended_at"),
        "Model and classes": {**pick("architecture", "generation", "size", "model", "training_options", "weights_uri"),
                              **({"class_names": dataset.get("class_names")} if dataset and dataset.get("class_names") else {})},
        "Starting checkpoint": pick("base_model", "base_model_s3_uri"),
        "Dataset and preparation": {**pick("train_dataset", "eval_dataset", "snapshot", "eval_is_train_dataset"),
                                    **({"dataset": dataset} if dataset else {})},
        "Training settings": pick("parameters", "hyperparams", "actual", "training_options"),
        "Training history": pick("history", "training_history", "epochs_completed", "best_epoch"),
        "Evaluation": pick("metrics", "evaluation", "per_class_metrics", "confusion_matrix"),
        "Inference settings": pick("inference_settings", "confidence_threshold", "iou_threshold"),
        "Software and code": pick("software", "versions", "git_commit", "container_image", "build_id"),
        "Infrastructure": {**columns("execution_arn"), **pick("training_job_name", "instance_type", "gpu")},
        "Registry and deployment": {**columns("model_package_arn"), **pick("model", "deployment")},
        "Governance and notes": pick("owner", "licence", "license", "intended_use", "limitations", "notes"),
        "File provenance": {"inventory": "manifest.csv", "checksums": "SHA-256 calculated for files actually exported",
                            "note": "Full file inventory is in the manifest; saved audit lists may be truncated."},
    }
    return clean({"schema": SCHEMA, "label": label, "kind": kind, "id": ident,
                  "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                  "source": "catalog database", "checkpoint_available": bool(record.get("weights_uri")),
                  "groups": groups, "unavailable": [k for k, v in groups.items() if not v],
                  "records": {"selected": row, "dataset": dataset},
                  "notes": ["Only recorded information is included; unavailable fields are not inferred.",
                            "Credentials are redacted. Registry status is a recorded snapshot."]})


def select_groups(metadata: dict, selected: list[str]) -> dict:
    unknown = set(selected) - set(metadata["groups"])
    if unknown:
        raise ValueError("Unknown metadata categories: " + ", ".join(sorted(unknown)))
    # Raw records would reintroduce categories the user explicitly unchecked.
    return {"schema": SCHEMA, "label": metadata["label"], "captured_at": metadata["captured_at"],
            "groups": {k: v for k, v in metadata["groups"].items() if k in selected},
            "selected_groups": selected, "unavailable": [k for k in metadata["unavailable"] if k in selected],
            "notes": ["Only selected catalog categories are added. Existing model checkpoint contents are preserved."]}


def embed(checkpoint: bytes, metadata: dict, manifest: str | None) -> bytes:
    """Append JSON/CSV under the checkpoint root; never deserialize model code."""
    stream = io.BytesIO(checkpoint)
    if not zipfile.is_zipfile(stream):
        raise ValueError("This checkpoint uses a legacy format; metadata embedding requires a ZIP-format .pt.")
    with zipfile.ZipFile(stream, "a", compression=zipfile.ZIP_STORED) as archive:
        roots = [n[:-len("data.pkl")] for n in archive.namelist() if n.endswith("/data.pkl")]
        if len(roots) != 1:
            raise ValueError("Unsupported checkpoint layout: expected one model data.pkl.")
        path = roots[0] + "extra/track2_metadata.json"
        if path in archive.namelist():
            raise ValueError("This checkpoint is already enriched; export from the original model.")
        archive.writestr(path, dumps(metadata))
        if manifest is not None:
            archive.writestr(roots[0] + "extra/track2_manifest.csv", manifest)
    return stream.getvalue()


def inspect_file(file, filename: str) -> dict:
    """Read only bounded JSON/CSV from uploads, never pickle or torch.load."""
    file.seek(0)
    if filename.lower().endswith(".json"):
        body = file.read(MAX_JSON + 1)
        if len(body) > MAX_JSON:
            raise ValueError("Metadata JSON exceeds 8 MB.")
        value = json.loads(body)
    else:
        with zipfile.ZipFile(file) as archive:
            names = archive.namelist()
            paths = [n for n in names if n == "metadata.json" or n.endswith("/extra/track2_metadata.json")]
            if len(paths) != 1:
                raise ValueError("No unique Track2 metadata found. Choose an enriched .pt, metadata.json or an exported ZIP.")
            info = archive.getinfo(paths[0])
            if info.file_size > MAX_JSON:
                raise ValueError("Embedded metadata exceeds 8 MB.")
            value = json.loads(archive.read(info))
            manifests = [n for n in names if n == "manifest.csv" or n.endswith("/extra/track2_manifest.csv")]
            if len(manifests) == 1:
                info = archive.getinfo(manifests[0])
                if info.file_size > MAX_MANIFEST:
                    raise ValueError("Embedded manifest exceeds the 16 MB viewer limit.")
                value["file_manifest"] = archive.read(info).decode("utf-8")
    if not isinstance(value, dict) or value.get("schema") != SCHEMA or not isinstance(value.get("groups"), dict):
        raise ValueError("Not a supported Track2 metadata document.")
    return clean(value)
