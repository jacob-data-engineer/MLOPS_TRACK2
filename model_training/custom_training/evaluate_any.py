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
import hashlib
import json
import os
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _f1(precision, recall):
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


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


def drop_undeclared_class_boxes(data, declared_names):
    """Remove, from this container's copy of the evaluation set, every box
    whose class `data.yaml` does not name -- the same rule the trainer
    applies, for the same reason: Ultralytics discards the WHOLE image for
    one such box, and an evaluation set with none left is a crash thirty
    minutes in. Only files under a `labels` component are touched; S3 is
    never written."""
    declared = set(int(k) for k in declared_names)
    files_changed = 0
    dropped = {}
    for path in Path(data).rglob("*.txt"):
        if "labels" not in path.relative_to(data).parts:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        keep = []
        changed = False
        for line in lines:
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                keep.append(line)
                continue
            if cls in declared:
                keep.append(line)
            else:
                dropped[cls] = dropped.get(cls, 0) + 1
                changed = True
        if changed:
            files_changed += 1
            path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
    return {"files_changed": files_changed, "dropped": dict(sorted(dropped.items()))}


def split_dirs(data, manifest):
    """Where each split's images actually live -- one directory or SEVERAL.

    Hard-coding images/<split> is wrong for a Roboflow-style export, whose
    data.yaml says `train: ../train/images`. Such a dataset trains fine and
    then fails here, so the manifest's own paths are tried first and the
    conventional layout is the fallback.

    A split may also be a LIST, which is how the retrain chain names the
    week folders it keeps:

        train:
        - week_1/images/train
        - week_2/images/train

    Ultralytics accepts that and trains on the union of them, so evaluation
    has to read it the same way. Taking only the first entry would score a
    model on one week of a corpus it was trained on in full; checking only
    for a `str` -- which is what this did -- found nothing at all, and every
    scheduled retrain died here with "no images found under images/train or
    images/val" AFTER a successful seven-minute training step. Training
    passing and evaluation failing on the same dataset is the shape of the
    bug: the two were reading the same file by different rules.
    """
    found = {}
    for split in ("train", "val"):
        dirs = [entry for entry in _declared_paths(data, manifest, split) if entry.is_dir()]
        # Only when the manifest named nothing that exists -- and nothing
        # that is a LIST FILE either, which `split_list_files` reads. A
        # yaml that resolved to SOME real folders is the answer: a chain
        # whose oldest week has been retired still names it, and silently
        # swapping in images/<split> there would evaluate a different
        # dataset from the one that was trained on.
        if not dirs and not _declared_list_files(data, manifest, split):
            for fallback in (data / "images" / split, data / split / "images"):
                if fallback.is_dir():
                    dirs = [fallback]
                    break
        if dirs:
            found[split] = dirs
    return found


def _declared_paths(data, manifest, split):
    """Every path a split names, resolved against the dataset root."""
    declared = manifest.get(split) or manifest.get("valid" if split == "val" else split)
    entries = declared if isinstance(declared, (list, tuple)) else [declared]
    out = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.strip():
            continue
        # Relative to the dataset root, with any leading ../ stripped:
        # the root here IS the folder data.yaml sits in.
        out.append(data / entry.strip().lstrip("./").replace("../", ""))
    return out


def _declared_list_files(data, manifest, split):
    """The split entries that are TEXT LISTS rather than directories.

    Ultralytics reads a split entry that is a file as a list of image
    paths, one per line -- that is the whole mechanism the retention
    policy and the snapshot restore rely on:

        train: train.txt

    This function exists because evaluation did NOT read it that way. It
    only ever looked for directories, so the moment a run trained on a
    pinned list the evaluator reported "no images found under
    images/train or images/val" -- after a successful training step, on a
    dataset Ultralytics had just read without complaint. Training and
    evaluation disagreeing about one file is the same shape of bug as the
    week-folder case above, one layer along.
    """
    return [entry for entry in _declared_paths(data, manifest, split) if entry.is_file()]


def _images_in_list(list_file):
    """The images a list names, as paths that exist.

    `get_img_files` expands a leading `./` against THE LIST FILE'S OWN
    directory, not against `path:`. Resolving it any other way is how a
    list at the dataset root reading `./week_1/...` came to be looked for
    under `_retained/week_1/...`.
    """
    try:
        lines = list_file.read_text(encoding="utf-8").splitlines()
    except Exception:                                          # noqa: BLE001
        return []
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        path = Path(line)
        if not path.is_absolute():
            path = list_file.parent / line.lstrip("./")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            out.append(path)
    return out


def dataset_files(data, manifest=None):
    """Every image and its label, sorted so a fingerprint over them is stable.

    Sorted over the WHOLE split rather than within each directory: with
    several week folders per split, the fingerprint must not depend on the
    order the manifest happens to list them in, or re-ordering the yaml
    would look like new data.
    """
    manifest = manifest or {}
    found = []
    resolved = split_dirs(data, manifest)
    for split in ("train", "val"):
        rows = []
        for images in resolved.get(split) or []:
            for path in images.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    label = label_dir_for(images) / path.relative_to(images).with_suffix(".txt")
                    rows.append((path, label if label.is_file() else None))
        # And the splits named as LIST FILES. Labels are still found per
        # image by swapping /images/ for /labels/ in that image's own
        # path, so a list may cross week folders freely -- which is the
        # property both the retention policy and a pinned snapshot rest
        # on.
        for list_file in _declared_list_files(data, manifest, split):
            for path in _images_in_list(list_file):
                label = label_dir_for(path.parent) / (path.stem + ".txt")
                rows.append((path, label if label.is_file() else None))
        for path, label in sorted(rows, key=lambda row: str(row[0])):
            found.append((split, path, label))
    return found


def fingerprint(files):
    """sha256 over the contents of every image and label.

    Gives a cardless folder a real dataset_version: the same pixels and the
    same boxes produce the same version, which is the whole job the card's
    own version was doing. Contents rather than names, so a rename is not
    mistaken for new data and new data is never mistaken for a rerun.
    """
    digest = hashlib.sha256()
    for split, image, label in files:
        digest.update(split.encode())
        digest.update(image.name.encode())
        digest.update(hashlib.sha256(image.read_bytes()).hexdigest().encode())
        # A missing label is a legitimate negative example, and has to hash
        # differently from an empty one that happens to exist.
        digest.update(hashlib.sha256(label.read_bytes()).hexdigest().encode() if label else b"-")
    return digest.hexdigest()


def detection(model_path, data_yaml, name, expected_names=None,
              classes=None, single_cls=False, filtered_name=""):
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    names = {int(k): v for k, v in model.names.items()}
    if single_cls:
        # The run narrowed a many-class dataset to ONE class at load
        # time: `classes=[i]` drops every other box and `single_cls=True`
        # renumbers the survivors to 0, so Ultralytics builds a one-output
        # head and calls the class "item". The checkpoint therefore CANNOT
        # equal the dataset's six names, and demanding that it does is why
        # every run of this shape died here after a successful training
        # step -- the evaluator was never told the filter had been applied.
        if len(names) != 1:
            raise ValueError(
                f"{name} checkpoint has {len(names)} classes {names}, but the run "
                f"was single-class. The checkpoint does not match how it was trained."
            )
    elif expected_names is not None and names != expected_names:
        # Not a taxonomy policy -- a consistency check. A checkpoint whose
        # class order differs from the data it is scored against produces
        # numbers that look fine and are attributed to the wrong labels.
        raise ValueError(
            f"{name} checkpoint classes {names} do not match the dataset's {expected_names}"
        )
    # The SAME narrowing the trainer used, or the model is scored against
    # boxes it was never taught to predict: `classes` filters, then
    # `single_cls` renumbers, in that order inside Ultralytics' own
    # dataset loading.
    extra = {}
    if classes:
        extra["classes"] = list(classes)
    if single_cls:
        extra["single_cls"] = True
    result = model.val(data=str(data_yaml), split="val", imgsz=640, device="cpu",
                       project="/opt/ml/processing/validation", name=name, plots=False,
                       **extra)
    precision, recall = float(result.box.mp), float(result.box.mr)
    metrics = {"precision": precision, "recall": recall, "f1": _f1(precision, recall),
               "mAP50": float(result.box.map50), "mAP50-95": float(result.box.map)}
    # Every class the dataset declares, so one absent from the validation
    # split reads as "not measured" rather than being silently missing.
    # Ultralytics calls a single_cls run's one class "item". Report it
    # under the name somebody would recognise -- the dataset's name for
    # the class that was kept -- or the metrics table reads "item" and
    # says nothing about which detector this is.
    if single_cls and filtered_name:
        names = {0: filtered_name}
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


def build_filtered_dataset(dataset_root, data_yaml_path, *, drop_undeclared=False,
                           out_root=None):
    """A TEMPORARY, YOLO-ready copy of the dataset with clean labels.

    The downloaded files are never modified. For every image a split names,
    the image is linked (symlink, else hardlink, else copy) into `out_root`
    at the same relative path, and its label is written there FILTERED:

      * x_label_remap -- a source that numbered its classes differently is
        translated to the declared numbering first (by name, from the
        builder);
      * classes the yaml does not name are DROPPED when `drop_undeclared`,
        and are an error otherwise -- never silently trained;
      * the declared classes are renumbered to 0..n-1 if they were not
        consecutive, and the temporary yaml uses the same numbering;
      * malformed lines -- wrong field count, non-numbers, coordinates
        outside 0-1 -- are an ERROR, reported apart from excluded classes.

    An image left with no boxes keeps an empty label (a background image);
    an image with no label file stays label-less, as Ultralytics treats it.
    Path-list (.txt) split entries are copied verbatim: their lines are
    relative and the tree is mirrored, so they resolve the same way.
    A fresh `out_root` each call, so no stale `.cache` can survive between
    runs with different classes. Returns (new_yaml_path, report).
    """
    import os as _os
    import shutil as _shutil
    import yaml as _yaml

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
    with open(data_yaml_path) as f:
        doc = _yaml.safe_load(f) or {}
    root = str(doc.get("path") or _os.path.dirname(data_yaml_path))
    if not _os.path.isabs(root) or not _os.path.isdir(root):
        root = dataset_root
    raw = doc.get("names") or {}
    if isinstance(raw, (list, tuple)):
        raw = dict(enumerate(raw))
    names = {int(k): str(v) for k, v in raw.items()}
    if not names:
        raise SystemExit("DATASET FILTER FAILED: data.yaml declares no class names.")
    nc = doc.get("nc")
    if nc is not None and int(nc) != len(names):
        raise SystemExit(f"DATASET FILTER FAILED: data.yaml says nc: {nc} but names "
                         f"lists {len(names)} classes.")
    order = sorted(names)
    consecutive = {old: new for new, old in enumerate(order)}
    remaps = {}
    for rel, mapping in (doc.get("x_label_remap") or {}).items():
        remaps[str(rel).strip("/")] = {int(k): int(v) for k, v in mapping.items()}

    out_root = out_root or (dataset_root.rstrip("/\\") + "_filtered")
    if _os.path.exists(out_root):
        _shutil.rmtree(out_root)
    _os.makedirs(out_root)

    def link(src, dst):
        _os.makedirs(_os.path.dirname(dst), exist_ok=True)
        for how in (_os.symlink, _os.link, _shutil.copy2):
            try:
                how(src, dst)
                return
            except (OSError, NotImplementedError):
                continue
        raise SystemExit(f"DATASET FILTER FAILED: could not place {src} in {out_root}.")

    def label_rel(image_rel):
        parts = image_rel.replace(_os.sep, "/").split("/")
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                return _os.path.splitext("/".join(parts))[0] + ".txt"
        return ""

    def remap_for(lrel):
        for prefix, mapping in remaps.items():
            if lrel == prefix or lrel.startswith(prefix + "/"):
                return mapping
        return None

    rep = {"label_files": 0, "kept": 0, "removed": 0, "removed_by_class": {},
           "now_empty": 0, "no_label_file": 0, "images": 0}
    malformed, undeclared_hits = [], {}
    seen = set()

    def take(image_abs):
        rel = _os.path.relpath(image_abs, root)
        if rel in seen:
            return
        seen.add(rel)
        rep["images"] += 1
        link(image_abs, _os.path.join(out_root, rel))
        lrel = label_rel(rel)
        if not lrel:
            return
        src_label = _os.path.join(root, lrel)
        if not _os.path.isfile(src_label):
            rep["no_label_file"] += 1
            return
        rep["label_files"] += 1
        mapping = remap_for(lrel.replace(_os.sep, "/"))
        kept = []
        with open(src_label, errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                parts = line.split()
                if not parts:
                    continue
                try:
                    vals = [float(x) for x in parts]
                except ValueError:
                    malformed.append(f"{lrel}:{n} non-numeric: {line.strip()[:60]}")
                    continue
                cls, coords = vals[0], vals[1:]
                if cls != int(cls) or cls < 0:
                    malformed.append(f"{lrel}:{n} class id {parts[0]!r} is not a whole number")
                    continue
                if not (len(coords) == 4 or (len(coords) >= 6 and len(coords) % 2 == 0)):
                    malformed.append(f"{lrel}:{n} has {len(parts)} fields; a box needs 5")
                    continue
                if any(c < -0.01 or c > 1.01 for c in coords):
                    malformed.append(f"{lrel}:{n} coordinates outside 0-1: {line.strip()[:60]}")
                    continue
                cls = int(cls)
                if mapping is not None:
                    cls = mapping.get(cls, cls)
                if cls not in names:
                    if cls >= 0:
                        undeclared_hits[cls] = undeclared_hits.get(cls, 0) + 1
                    rep["removed"] += 1
                    rep["removed_by_class"][cls] = rep["removed_by_class"].get(cls, 0) + 1
                    continue
                kept.append(" ".join([str(consecutive[cls])] + parts[1:]))
                rep["kept"] += 1
        if not kept:
            rep["now_empty"] += 1
        dst = _os.path.join(out_root, lrel)
        _os.makedirs(_os.path.dirname(dst), exist_ok=True)
        with open(dst, "w") as fh:
            fh.write("\n".join(kept) + ("\n" if kept else ""))

    for split in ("train", "val", "test"):
        entries = doc.get(split) or []
        for entry in (entries if isinstance(entries, list) else [entries]):
            path = entry if _os.path.isabs(str(entry)) else _os.path.join(root, str(entry))
            if str(entry).lower().endswith(".txt") and _os.path.isfile(path):
                link_target = _os.path.join(out_root, _os.path.relpath(path, root))
                _os.makedirs(_os.path.dirname(link_target), exist_ok=True)
                _shutil.copy2(path, link_target)
                base = _os.path.dirname(path)
                with open(path) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            img = _os.path.normpath(_os.path.join(base, line[2:]) if line.startswith("./") else line)
                            if _os.path.isfile(img):
                                take(img)
                continue
            if _os.path.isdir(path):
                for r, _, fs in _os.walk(path):
                    for fname in fs:
                        if fname.lower().endswith(exts):
                            take(_os.path.join(r, fname))

    problems = []
    if malformed:
        problems.append(f"{len(malformed)} malformed label line(s), e.g. " + "; ".join(malformed[:5]))
    if undeclared_hits and not drop_undeclared:
        problems.append("labels use class(es) the yaml does not name: "
                        + ", ".join(f"{c} ({n} boxes)" for c, n in sorted(undeclared_hits.items()))
                        + ". Tick 'Filter undeclared classes' or name them in Label classes.")
    if problems:
        _shutil.rmtree(out_root, ignore_errors=True)
        raise SystemExit("DATASET FILTER FAILED -- nothing was trained:\n  - " + "\n  - ".join(problems))

    new_doc = {k: v for k, v in doc.items() if k != "x_label_remap"}
    new_doc["path"] = out_root
    new_doc["names"] = {consecutive[k]: names[k] for k in order}
    new_doc["nc"] = len(names)
    new_yaml = _os.path.join(out_root, "data.yaml")
    with open(new_yaml, "w") as fh:
        _yaml.safe_dump(new_doc, fh, sort_keys=False)
    rep["removed_by_class"] = dict(sorted(rep["removed_by_class"].items()))
    print("LABEL_FILTER: " + ", ".join([
        f"label files {rep['label_files']}",
        f"annotations kept {rep['kept']}",
        f"removed {rep['removed']} (by source class {rep['removed_by_class'] or 'none'})",
        f"images left with no boxes {rep['now_empty']}",
        f"images without a label file {rep['no_label_file']}",
        f"classes {len(names)}" + ("" if list(order) == list(range(len(order))) else f" renumbered from {order}"),
        "originals unchanged"]), flush=True)
    return new_yaml, rep


def fetch_extra_val(data, uris, primary_bucket=""):
    """Download val folders that live outside the one ProcessingInput
    prefix, to `data/<their full key>` -- which is how the yaml names
    them. Only the folders listed: each images/val and its labels."""
    client = boto3.client("s3")
    fetched = 0
    for uri in [u.strip() for u in (uris or "").split(",") if u.strip()]:
        bucket, prefix = uri.removeprefix("s3://").split("/", 1)
        for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents") or []:
                if obj["Key"].endswith("/"):
                    continue
                # Another bucket's files under <bucket>/, as the yaml names them.
                target = Path(data) / (obj["Key"] if not primary_bucket or bucket == primary_bucket
                                       else f"{bucket}/{obj['Key']}")
                target.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(bucket, obj["Key"], str(target))
                fetched += 1
    print(f"EVAL_EXTRA: fetched {fetched} objects from {uris}", flush=True)


def main():
    root = Path("/opt/ml/processing")
    model_dir, data = root / "model", root / "data"
    original = yaml.safe_load((data / "data.yaml").read_text()) or {}
    primary_bucket = str(original.get("path") or "").removeprefix("s3://").split("/", 1)[0]
    if (os.environ.get("EVAL_EXTRA_URIS") or "").strip():
        fetch_extra_val(data, os.environ["EVAL_EXTRA_URIS"], primary_bucket)
    # The same temporary filtered copy the trainer builds -- remap, filter
    # when ticked, renumber, malformed lines refused -- so evaluation reads
    # exactly what training read. The processing input is not modified.
    drop = (os.environ.get("DROP_UNDECLARED_CLASSES") or "").strip().lower() in ("1", "true", "yes")
    new_yaml, _ = build_filtered_dataset(str(data), str(data / "data.yaml"),
                                         drop_undeclared=drop, out_root="/tmp/eval_filtered")
    data = Path(new_yaml).parent
    with tarfile.open(model_dir / "model.tar.gz") as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk() or not (model_dir / member.name).resolve().is_relative_to(model_dir):
                raise ValueError("unsafe model archive")
        archive.extractall(model_dir)

    manifest = yaml.safe_load((data / "data.yaml").read_text())
    declared = manifest.get("names") or {}
    if isinstance(declared, (list, tuple)):
        declared = {index: value for index, value in enumerate(declared)}
    dataset_names = {int(k): v for k, v in declared.items()}
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

    files = dataset_files(data, manifest)
    if not files:
        raise ValueError("no images found under images/train or images/val")

    # How the run narrowed the dataset, passed through from the pipeline.
    # Absent means "trained on everything the yaml declares", which is
    # what every run did before class filtering existed.
    classes = [int(part) for part in (os.environ.get("CLASSES") or "").split(",")
               if part.strip().lstrip("-").isdigit()]
    single_cls = (os.environ.get("SINGLE_CLS") or "").strip().lower() in ("1", "true", "yes")
    filtered_name = ""
    if single_cls and len(classes) == 1:
        filtered_name = str(dataset_names.get(classes[0]) or "")

    candidate = detection(model_dir / "model.pt", data / "data.yaml", "candidate",
                          dataset_names, classes=classes, single_cls=single_cls,
                          filtered_name=filtered_name)

    # Optional: a first model for a new taxonomy has no predecessor, and
    # an empty value is how the pipeline says so.
    baseline_uri = (os.environ.get("BASE_MODEL_URI") or "").strip()
    baseline = None
    if baseline_uri:
        try:
            baseline = detection(download_baseline(baseline_uri, root), data / "data.yaml",
                                 "baseline", dataset_names, classes=classes,
                                 single_cls=single_cls, filtered_name=filtered_name)
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
        "dataset_version": card.get("dataset_version") or ("sha256-" + fingerprint(files)[:32]),
        "dataset_uri": card.get("source_uri"),
        # Which of the two the version above came from, so a reader never
        # has to guess whether this run had a prepared card behind it.
        "dataset_provenance": "dataset-card" if card else "file-contents",
        "classes": dataset_names,
        # What was actually trained, when the run narrowed the dataset.
        # Without this a one-class model reads as a six-class one that
        # happens to score 0 on five of them.
        "trained_classes": ({0: filtered_name} if single_cls and filtered_name
                            else {i: dataset_names[i] for i in classes if i in dataset_names}
                            or dataset_names),
        "class_filter": {"classes": classes, "single_cls": single_cls},
        # WHERE the numbers come from. The chain evaluates on the training
        # corpus's own val split unless a separate eval dataset was given,
        # and that split is also the one Ultralytics picked `best.pt` on --
        # so the score is optimistic by an unmeasured amount. Recording it
        # is the difference between a known limitation and a wrong number.
        "eval_split_source": (
            "separate evaluation dataset"
            if (os.environ.get("EVAL_IS_SEPARATE") or "").strip().lower() in ("1", "true", "yes")
            else "the training corpus's own val split -- also used to select best.pt, "
                 "so these metrics are selection-biased"
        ),
        "baseline_uri": baseline_uri or None,
        "training_execution_id": os.environ.get("RUN_NAME"),
        # Counted off disk, which is what it always described, and the only
        # source available when there is no card.
        "validation_count": sum(1 for split, _i, _l in files if split == "val"),
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
