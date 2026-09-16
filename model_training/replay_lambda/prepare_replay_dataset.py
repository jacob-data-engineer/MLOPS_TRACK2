#!/usr/bin/env python3
"""Create a new derived YOLO dataset from a downloaded immutable snapshot.

Usage: python scripts/prepare_replay_dataset.py SNAPSHOT OUTPUT --model hero
OUTPUT must not exist. Source files are never modified or hard-linked.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from replay_policy import MODEL_CLASSES, SOURCE_CLASSES, digest, remap_labels


def prepare(snapshot, output, model):
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(snapshot) or snapshot.is_relative_to(output):
        raise ValueError("output must be new and outside the source snapshot")
    marker = json.loads((snapshot / "_SUCCESS.json").read_text())
    if marker.get("status") != "complete" or not marker.get("source_fingerprint") or not marker.get("snapshot"):
        raise ValueError("complete snapshot marker with fingerprint and source URI required")
    prefix = "escalation_dataset_full_frame" if model == "hero" else "escalation_dataset"
    root = snapshot / "track2" / prefix
    # The producer writes an integer-keyed names map. Parse only that explicit
    # contract; reject arbitrary YAML rather than accepting unexpected aliases.
    names, in_names = {}, False
    for line in (root / "data.yaml").read_text().splitlines():
        if line.strip() == "names:":
            in_names = True
        elif in_names and line.startswith(" ") and line.strip():
            key, value = line.strip().split(":", 1)
            names[int(key)] = value.strip().strip("\"'")
        elif line and not line.startswith(" "):
            in_names = False
    if names != SOURCE_CLASSES:
        raise ValueError("source six-class manifest mismatch")
    records, hashes = [], {}
    for split in ("train", "val"):
        images = root / "images" / split
        labels = root / "labels" / split
        if not images.is_dir() or not labels.is_dir():
            raise ValueError(f"missing {split} directories")
        image_map = {}
        for path in sorted(images.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            if path.stem in image_map:
                raise ValueError("ambiguous image stem")
            image_map[path.stem] = path
        label_map = {p.stem: p for p in labels.glob("*.txt")}
        if image_map.keys() != label_map.keys() or not image_map:
            raise ValueError(f"empty or unpaired {split} images/labels")
        for stem, path in image_map.items():
            label = label_map[stem]
            if path.is_symlink() or label.is_symlink():
                raise ValueError("snapshot files must not be symlinks")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha in hashes and hashes[sha] != split:
                raise ValueError("identical image appears in train and val")
            hashes[sha] = split
            source = label.read_text()
            derived = remap_labels(source, model)
            records.append({"split": split, "image": path.name, "sha256": sha,
                            "source_label_sha256": hashlib.sha256(source.encode()).hexdigest(),
                            "label": derived})
    card = {"schema_version": "1.0.0", "adapter_version": "ppe-remap-v1", "model": model,
            "snapshot_uri": marker["snapshot"], "source_fingerprint": marker["source_fingerprint"],
            "source_classes": SOURCE_CLASSES, "classes": MODEL_CLASSES[model], "samples": records,
            "validation": {"paired": True, "exact_train_val_leakage": False,
                           "scene_leakage_checked": False}}
    card["dataset_version"] = "ds-" + digest(card)
    # Validate the entire source before creating output. READY is written last;
    # interrupted outputs are never eligible for training.
    output.mkdir(parents=True)
    for record in records:
        split, name = record["split"], record["image"]
        dest_image = output / "images" / split / name
        dest_label = output / "labels" / split / (Path(name).stem + ".txt")
        dest_image.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "images" / split / name, dest_image)
        if hashlib.sha256(dest_image.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("source changed while copying")
        dest_label.write_text(record["label"])
    yaml_names = "\n".join(f"  {key}: {value}" for key, value in MODEL_CLASSES[model].items())
    (output / "data.yaml").write_text(f"names:\n{yaml_names}\ntrain: images/train\nval: images/val\n")
    (output / "dataset-card.json").write_text(json.dumps(card, indent=2)+"\n")
    (output / "_READY.json").write_text(json.dumps({"dataset_version": card["dataset_version"]})+"\n")
    return card


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", choices=MODEL_CLASSES, required=True)
    args = parser.parse_args()
    print(prepare(args.snapshot, args.output, args.model)["dataset_version"])
