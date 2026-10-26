"""SageMaker full-frame detection evaluation, with version-specific evidence.

Object detection validation is NOT a substitute for alarm/VLM replay evaluation.
Absent independent replay evidence always leaves the promotion gate blocked.

THIS FILE IS THE PRODUCTION EVALUATOR. The pipeline runs it from a
content-addressed S3 key (replay/fullframe/code/<sha256>/evaluate.py), not
from the repo, so editing it changes nothing until it is uploaded under its
new hash and the pipeline definition is re-pointed at that key. Note that
model_training/training-container/evaluate.py is a DIFFERENT, older file
that nothing runs.

Takes a standard Ultralytics dataset -- data.yaml plus images/{train,val}
and labels/{train,val}. A dataset-card.json is used when present and is no
longer required, which is the change from the previous version.

What the card used to be the only source of, and where each part comes from
now:

  the taxonomy       read from data.yaml, which is the file Ultralytics
                     itself trains and validates against. The card was
                     only ever a second copy of the same assertion, so
                     requiring it protected nothing that data.yaml cannot
                     say -- while making every ordinary YOLO folder
                     unusable. The gate itself is unchanged and still
                     refuses anything that is not {0: person, 1: helmet},
                     because this pipeline's registry group has consumers
                     that read index 0 as person; a six-class model in
                     that chain is mislabelled at every step without
                     failing anywhere.

  dataset_version    the card's, when there is one. Otherwise a sha256
                     over every image and label file's contents, so the
                     metric is still pinned to exact data and two runs on
                     the same folder produce the same version.

  snapshot_uri       the card's, when there is one. Otherwise null --
                     recorded as genuinely absent rather than invented,
                     for the reason prepare_any_dataset.py gives at
                     length: provenance that claims more than it knows is
                     worse than provenance that admits the gap.

  validation_count   counted off disk, which is what it always described.
"""
import hashlib
import json
import os
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

REQUIRED_CLASSES = {0: "person", 1: "helmet"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def detection(model_path, data_yaml, name):
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    if model.names != REQUIRED_CLASSES:
        raise ValueError(f"{name} checkpoint class manifest mismatch: {model.names}")
    result = model.val(data=str(data_yaml), split="val", imgsz=640, device="cpu",
                       project="/opt/ml/processing/validation", name=name, plots=False)
    p, r = float(result.box.mp), float(result.box.mr)
    metrics = {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0,
               "mAP50": float(result.box.map50), "mAP50-95": float(result.box.map)}
    classes = {label: None for label in ("person", "helmet")}
    for index, class_id in enumerate(result.box.ap_class_index):
        p, r, ap50, ap = map(float, result.box.class_result(index))
        classes[model.names[int(class_id)]] = {"precision": p, "recall": r,
            "f1": 2*p*r/(p+r) if p+r else 0.0, "mAP50": ap50, "mAP50-95": ap}
    return {"metrics": metrics, "per_class": classes}


def download_baseline(uri, root):
    """Support both registered SageMaker archives and legacy bare checkpoints."""
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    target = root / "baseline.pt"
    if not key.endswith(".tar.gz"):
        boto3.client("s3").download_file(bucket, key, str(target))
        return target
    archive_path = root / "baseline.tar.gz"
    boto3.client("s3").download_file(bucket, key, str(archive_path))
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive.getmembers() if m.name in {"model.pt", "./model.pt"}]
        if len(members) != 1 or not members[0].isfile():
            raise ValueError("production archive must contain one regular model.pt")
        # Read only the checkpoint; never extract executable code or archive paths.
        import shutil
        with archive.extractfile(members[0]) as source, target.open("wb") as output:
            shutil.copyfileobj(source, output)
    return target


def label_dir_for(images):
    """The labels directory paired with an images directory.

    Both layouts put them as siblings: images/train -> labels/train, and
    train/images -> train/labels. Swapping the one "images" component is
    what they have in common.
    """
    parts = list(images.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            parts[index] = "labels"
            break
    return type(images)(*parts)


def split_dirs(data, manifest):
    """Where each split's images actually live.

    Hard-coding images/<split> is wrong for a Roboflow-style export, whose
    data.yaml says `train: ../train/images`. Such a dataset trains fine and
    then fails here, so the manifest's own paths are tried first and the
    conventional layout is the fallback.
    """
    found = {}
    for split in ("train", "val"):
        candidates = []
        declared = manifest.get(split) or manifest.get("valid" if split == "val" else split)
        if isinstance(declared, str):
            # Relative to the dataset root, with any leading ../ stripped:
            # the root here IS the folder data.yaml sits in.
            candidates.append(data / declared.lstrip("./").replace("../", ""))
        candidates.append(data / "images" / split)
        candidates.append(data / split / "images")
        for candidate in candidates:
            if candidate.is_dir():
                found[split] = candidate
                break
    return found


def dataset_files(data, manifest=None):
    """Every image and its label, relative to the dataset root, sorted.

    Sorted so the fingerprint below is stable: S3 listing order and local
    walk order are not the same, and a version that changed with the
    weather would be worse than none.
    """
    found = []
    resolved = split_dirs(data, manifest or {})
    for split in ("train", "val"):
        images = resolved.get(split)
        if images is None:
            continue
        for path in sorted(images.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                label = label_dir_for(images) / path.relative_to(images).with_suffix(".txt")
                found.append((split, path, label if label.is_file() else None))
    return found


def fingerprint(files):
    """sha256 over the contents of every image and label.

    Contents, not names or sizes: the point is that the same version can
    only mean the same pixels and the same boxes.
    """
    digest = hashlib.sha256()
    for split, image, label in files:
        digest.update(split.encode())
        digest.update(image.name.encode())
        digest.update(hashlib.sha256(image.read_bytes()).hexdigest().encode())
        # A missing label file is a legitimate negative example, and has to
        # hash differently from an empty one that happens to exist.
        digest.update(hashlib.sha256(label.read_bytes()).hexdigest().encode() if label else b"-")
    return digest.hexdigest()


def main():
    root = Path("/opt/ml/processing")
    model_dir, data = root / "model", root / "data"
    with tarfile.open(model_dir / "model.tar.gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not (model_dir / member.name).resolve().is_relative_to(model_dir):
                raise ValueError("unsafe model archive")
        archive.extractall(model_dir)

    manifest = yaml.safe_load((data / "data.yaml").read_text())
    declared = manifest.get("names") or {}
    if isinstance(declared, (list, tuple)):
        declared = {index: value for index, value in enumerate(declared)}
    names = {int(k): v for k, v in declared.items()}
    if names != REQUIRED_CLASSES:
        raise ValueError(
            f"this pipeline evaluates {REQUIRED_CLASSES} only, and data.yaml declares {names}. "
            "Train a different taxonomy through the custom pipeline "
            "(track2-yolo-custom-training-pipeline), whose registry group has no consumer "
            "that assumes what index 0 means."
        )
    # Ultralytics resolves the split paths relative to `path`.
    manifest["path"] = str(data)
    (data / "data.yaml").write_text(yaml.safe_dump(manifest))

    card_path = data / "dataset-card.json"
    card = json.loads(card_path.read_text()) if card_path.exists() else {}
    card_classes = {int(k): v for k, v in (card.get("classes") or {}).items()}
    if card_classes and card_classes != names:
        raise ValueError(f"dataset-card.json classes {card_classes} disagree with data.yaml {names}")

    files = dataset_files(data, manifest)
    if not files:
        raise ValueError("no images found under images/train or images/val")

    baseline_uri = os.environ["BASE_MODEL_URI"]
    baseline_path = download_baseline(baseline_uri, root)
    baseline = detection(baseline_path, data / "data.yaml", "baseline")
    candidate = detection(model_dir / "model.pt", data / "data.yaml", "candidate")
    report = {"schema_version": "1.1.0", "evaluation_kind": "fullframe_detection_validation",
              "generated_at": datetime.now(timezone.utc).isoformat(),
              "dataset_version": card.get("dataset_version") or ("sha256-" + fingerprint(files)[:32]),
              "snapshot_uri": card.get("snapshot_uri"),
              # Which of the two the numbers above are actually pinned to,
              # so a reader never has to guess whether this run had a
              # prepared snapshot behind it.
              "dataset_provenance": "dataset-card" if card else "file-contents",
              "baseline_uri": baseline_uri, "baseline_role": "production_replay_at_run_start",
              "training_execution_id": os.environ["RUN_NAME"],
              "validation_count": sum(1 for split, _image, _label in files if split == "val"),
              **candidate, "baseline": baseline,
              "metric_differences": {k: v-baseline["metrics"][k] for k, v in candidate["metrics"].items()},
              "alarm_counts": None,
              "gate_result": {"passed": False, "status": "blocked",
                  "reason": "Independent four-week alarm/VLM replay evaluation and scene-leakage evidence not yet available. Detection validation alone cannot pass the promotion gate."}}
    output = root / "output"
    output.mkdir(exist_ok=True)
    (output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, allow_nan=False))


if __name__ == "__main__":
    main()
