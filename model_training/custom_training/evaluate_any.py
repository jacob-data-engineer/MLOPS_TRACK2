"""Detection evaluation for a model whose classes come from its dataset.

The full-frame replay evaluator this is derived from is deliberately locked
to one taxonomy -- it rejects any checkpoint whose names are not
{0: person, 1: helmet}, requires a "hero" dataset card, and reports
per-class figures for exactly those two labels. That lock is correct
there: it is what stops a wrong-taxonomy model reaching the production
replay lineage.

This evaluator is for the other case, a model trained on whatever classes
its dataset declares. It therefore reads the taxonomy instead of asserting
one -- but still checks the checkpoint and the dataset agree with each
other, which is the failure that actually produces silently mislabelled
predictions.

Baseline comparison is optional here. A custom taxonomy has no production
predecessor to compare against on its first run, and demanding one would
make that run impossible rather than merely unbaselined.
"""
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def detection(model_path, data_yaml, name, expected_names=None):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = {int(k): v for k, v in model.names.items()}
    if expected_names is not None and names != expected_names:
        # Not a taxonomy policy -- a consistency check. A checkpoint whose
        # class order differs from the data it is scored against produces
        # numbers that look fine and are attributed to the wrong labels.
        raise ValueError(
            f"{name} checkpoint classes {names} do not match the dataset's {expected_names}"
        )
    result = model.val(data=str(data_yaml), split="val", imgsz=640, device="cpu",
                       project="/opt/ml/processing/validation", name=name, plots=False)
    precision, recall = float(result.box.mp), float(result.box.mr)
    metrics = {"precision": precision, "recall": recall, "f1": _f1(precision, recall),
               "mAP50": float(result.box.map50), "mAP50-95": float(result.box.map)}
    # Every class the dataset declares, so one absent from the validation
    # split reads as "not measured" rather than being silently missing.
    per_class = {label: None for label in names.values()}
    for index, class_id in enumerate(result.box.ap_class_index):
        p, r, ap50, ap = map(float, result.box.class_result(index))
        per_class[names[int(class_id)]] = {"precision": p, "recall": r, "f1": _f1(p, r),
                                           "mAP50": ap50, "mAP50-95": ap}
    return {"metrics": metrics, "per_class": per_class, "classes": names}


def download_baseline(uri, root):
    """Supports both registered SageMaker archives and bare checkpoints."""
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    target = root / "baseline.pt"
    client = boto3.client("s3")
    if not key.endswith(".tar.gz"):
        client.download_file(bucket, key, str(target))
        return target
    archive_path = root / "baseline.tar.gz"
    client.download_file(bucket, key, str(archive_path))
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive.getmembers() if m.name in {"model.pt", "./model.pt"}]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("baseline archive must contain one regular model.pt")
        # Read only the checkpoint; never extract executable code or paths.
        with archive.extractfile(members[0]) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    return target


def main():
    root = Path("/opt/ml/processing")
    model_dir, data = root / "model", root / "data"
    with tarfile.open(model_dir / "model.tar.gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not (model_dir / member.name).resolve().is_relative_to(model_dir):
                raise ValueError("unsafe model archive")
        archive.extractall(model_dir)

    manifest = yaml.safe_load((data / "data.yaml").read_text())
    dataset_names = {int(k): v for k, v in (manifest.get("names") or {}).items()}
    if not dataset_names:
        raise ValueError("data.yaml declares no class names")
    # Ultralytics resolves the split paths relative to `path`.
    manifest["path"] = str(data)
    (data / "data.yaml").write_text(yaml.safe_dump(manifest))

    card_path = data / "dataset-card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    card_classes = {int(k): v for k, v in (card.get("classes") or {}).items()}
    if card_classes and card_classes != dataset_names:
        raise ValueError(f"dataset-card.json classes {card_classes} disagree with data.yaml {dataset_names}")

    candidate = detection(model_dir / "model.pt", data / "data.yaml", "candidate", dataset_names)

    # Optional: a first model for a new taxonomy has no predecessor, and
    # an empty value is how the pipeline says so.
    baseline_uri = (os.environ.get("BASE_MODEL_URI") or "").strip()
    baseline = None
    if baseline_uri:
        try:
            baseline = detection(download_baseline(baseline_uri, root), data / "data.yaml",
                                 "baseline", dataset_names)
        except Exception as exc:
            # A baseline from a different taxonomy cannot be compared, and
            # that is worth recording rather than failing the run over --
            # the candidate's own numbers are still valid.
            print(f"Baseline not comparable, reporting candidate only: {exc}")
            baseline = {"error": str(exc)}

    report = {
        "schema_version": "1.0.0",
        "evaluation_kind": "custom_detection_validation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_version": card.get("dataset_version"),
        "dataset_uri": card.get("source_uri"),
        "classes": dataset_names,
        "baseline_uri": baseline_uri or None,
        "training_execution_id": os.environ.get("RUN_NAME"),
        "validation_count": sum(1 for x in card.get("samples", []) if x.get("split") == "val") or None,
        **candidate,
        "baseline": baseline,
    }
    if baseline and "metrics" in baseline:
        report["metric_differences"] = {
            k: v - baseline["metrics"][k] for k, v in candidate["metrics"].items()
        }
    # This pipeline validates detection only. Saying so explicitly keeps a
    # custom model from being read as production-approved because nothing
    # said otherwise; approval stays a human step in the registry.
    report["gate_result"] = {
        "passed": False, "status": "not_applicable",
        "reason": "Custom-taxonomy training: detection validation only, no replay gate defined.",
    }

    output = root / "output"
    output.mkdir(exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
