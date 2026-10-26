"""SageMaker script-mode entrypoint: fine-tunes a YOLO checkpoint on the
Track2 demo dataset (real hazard images, synthetic boxes — see
../src/seed_training_data.py for why the boxes are synthetic today).

Follows the standard training-toolkit contract: hyperparameters arrive as
CLI args, and the final artifact must land in SM_MODEL_DIR so SageMaker
tars it up as model.tar.gz — that still happens, unchanged, for the
pipeline's RegisterModel step. The training dataset, however, is NOT
delivered via a SageMaker-managed input channel (SM_CHANNEL_*) anymore --
this script downloads it itself (see download_dataset_with_progress)
specifically so a real "N images loaded" count can be printed and turned
into CloudWatch metrics (pipeline_definition.py's Estimator has
metric_definitions regexes matching this script's own
"DATASET_TOTAL_IMAGES: N" / "DATASET_IMAGES: n/N" print lines, the same
technique already used for train:epoch). SageMaker's own automatic
channel download gives no per-file progress at all -- it's one opaque
"Downloading" phase bundled with the container image pull, confirmed via
a real DescribeTrainingJob (SecondaryStatusTransitions has no percentage
field).

The base checkpoint to fine-tune from comes from exactly one of two
sources, never a bare Ultralytics name (e.g. "yolov8n.pt") pulled fresh
from Ultralytics' own hub:
  --base-model-package-arn   the latest Approved package in whichever
                              group TRAIN_SIZE/TRAIN_GENERATION selects
                              (config.py's model_package_group_name) --
                              the original, registry-only path, still the
                              default when both exist for a
                              generation/size.
  --base-model-s3-uri         a direct S3 object (a .tar.gz or a bare
                              .pt) -- the escape hatch for a
                              generation/size that has nothing in the
                              Model Registry yet (e.g. a freshly
                              uploaded checkpoint via the review-app's
                              "Upload Another YOLO Model" section,
                              which never registers anything). Resolved
                              by review_app.py's /train/dataset route
                              from a folder into an exact object key
                              before this script ever sees it.
pipeline_definition.py picks whichever source was actually given and
raises before any Estimator/TrainingStep is built if neither resolves to
something real -- so by the time this script runs, whichever ARN/URI it
received is already known to be valid. The Approved check is repeated
here too for the registry path, in case this script is ever invoked
directly outside that pipeline.

Deliberately does NOT publish weights anywhere else in S3 on its own --
every training run produces a candidate (via RegisterModel), but weights
only need to be reachable at a stable location once a human has actually
approved that candidate. That gate is infra/lambda_src/
dump_approved_weights.py, EventBridge-triggered on ModelApprovalStatus
changing to Approved, which lands weights at
latest_approved_model_weights/model.pt (with its own archiving of
whatever was there before). A rejected or still-pending candidate's
weights stay reachable only via its own model.tar.gz
(ModelDataUrl in the registry) -- never copied to a convenience location.
"""

import argparse
import base64
import json
import os
import shutil
import tarfile
import tempfile
import time

import boto3
import yaml
from ultralytics import YOLO

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=int(os.environ.get("EPOCHS", 20)))
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    # Early stopping: epochs without an improvement in fitness before the
    # run stops on its own. Ultralytics defaults to 100, which on a 50-epoch
    # run means it never fires -- so the default here is 0, meaning "let
    # Ultralytics decide", and only a value the operator actually set is
    # passed through.
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--base-model-package-arn", type=str, default=None)
    parser.add_argument("--base-model-s3-uri", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-data-s3-uri", type=str,
                         default=os.environ.get("TRAIN_DATA_S3_URI"),
                         help="s3:// URI of the YOLO dataset root (contains data.yaml, images/, labels/) -- "
                              "downloaded by this script itself, not a SageMaker input channel (see module docstring).")
    parser.add_argument("--data-yaml-s3-uri", type=str,
                        default=os.environ.get("DATA_YAML_S3_URI", ""),
                        help="s3:// URI of the data.yaml to train with. Only needed "
                             "when --train-data-s3-uri names SEVERAL prefixes: there "
                             "is then no single dataset root to find one at. With one "
                             "prefix the yaml rides along in the download as before.")
    parser.add_argument("--classes", type=str, default=os.environ.get("CLASSES"),
                        help="Comma-separated class indices to train on, as they are "
                             "numbered in the dataset's own data.yaml -- e.g. '0' for "
                             "helmet in a six-class site dataset. Ultralytics drops "
                             "every box of any other class at load time, so the images "
                             "and labels in S3 are never rewritten and no filtered copy "
                             "is made. Blank trains on all of them.")
    parser.add_argument("--drop-undeclared-classes", type=str,
                        default=os.environ.get("DROP_UNDECLARED_CLASSES", ""),
                        help="'true' to rewrite the DOWNLOADED label files so that "
                             "boxes whose class the data.yaml does not name are "
                             "removed before Ultralytics reads them. Ultralytics "
                             "otherwise discards the whole image for one such box. "
                             "The files in S3 are never touched; only this "
                             "container's copy changes, and the run logs how many "
                             "boxes of which class were dropped.")
    parser.add_argument("--single-cls", type=str, default=os.environ.get("SINGLE_CLS", ""),
                        help="'true' to collapse whatever survives --classes into one "
                             "class and size the head to a single output. Together the "
                             "two turn a six-class site dataset into a one-class "
                             "detector without touching a file: --classes picks the "
                             "boxes, --single-cls renumbers them to 0 and sets nc=1. "
                             "Note Ultralytics also renames the class to 'item' when "
                             "this is on; the registered model takes its name from the "
                             "detector, not from here.")
    parser.add_argument("--dataset-manifest-s3-uri", type=str,
                        default=os.environ.get("DATASET_MANIFEST_S3_URI"),
                        help="s3:// URI of a snapshot manifest.json listing every image and "
                             "label the run must use, as paths relative to the dataset root. "
                             "Given one, this script downloads EXACTLY those keys instead of "
                             "everything under the prefix -- which is what makes a run "
                             "repeatable, because a data.yaml can only name DIRECTORIES and "
                             "those keep filling. Omit it and the whole prefix is taken, as "
                             "before.")
    parser.add_argument("--hyperparams", type=str,
                        default=os.environ.get("TRAIN_HYPERPARAMS"),
                        help="JSON object of extra Ultralytics training arguments, "
                             "e.g. {\"optimizer\": \"AdamW\", \"lr0\": 0.001}. "
                             "Validated against Ultralytics' own default config, so "
                             "a key it does not know is REFUSED rather than ignored. "
                             "One blob rather than one flag per parameter: this image "
                             "then never needs rebuilding to expose another knob.")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "ap-southeast-1"))
    # UNKNOWN HYPERPARAMETERS ARE IGNORED, NOT FATAL.
    #
    # SageMaker turns every pipeline hyperparameter into `--key value`,
    # so `parse_args` made this image refuse to start the moment the
    # pipeline learned a parameter it did not know. That is an ordering
    # trap with no safe direction: add the parameter first and every run
    # dies at startup on the old image; ship the image first and the app
    # cannot send the parameter yet. Both halves of a two-part change
    # were therefore releases that had to land in exactly one order,
    # across two stacks, with a broken pipeline in between.
    #
    # Ignoring them removes that. A new parameter reaching an old image
    # is a no-op, which is what "the image does not support this yet"
    # should mean.
    #
    # PRINTED, never silent: this also swallows a misspelled flag, and a
    # run that trained on the default because somebody typed
    # `--img-sizes` deserves a line in the log saying so.
    args, unrecognised = parser.parse_known_args()
    if unrecognised:
        print(f"Ignoring arguments this image does not know: {unrecognised}",
              flush=True)
    if bool(args.base_model_package_arn) == bool(args.base_model_s3_uri):
        parser.error("Pass exactly one of --base-model-package-arn or --base-model-s3-uri, not both/neither.")
    # SageMaker hyperparameters arrive as strings, always. Parsed here so
    # a malformed one fails at submit rather than an hour into the run.
    if args.classes:
        try:
            args.classes = [int(part) for part in str(args.classes).split(",") if part.strip()]
        except ValueError:
            parser.error(f"--classes must be comma-separated integers, got {args.classes!r}.")
    else:
        args.classes = None
    args.single_cls = str(args.single_cls).strip().lower() in ("1", "true", "yes")
    args.drop_undeclared_classes = (
        str(args.drop_undeclared_classes).strip().lower() in ("1", "true", "yes"))
    if not args.train_data_s3_uri:
        parser.error("--train-data-s3-uri (or TRAIN_DATA_S3_URI) is required.")
    return args


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    without_scheme = uri[len("s3://") :]
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def resolve_base_model_checkpoint(base_model_package_arn: str, region: str, download_dir: str) -> str:
    """Pulls the base checkpoint from the SageMaker Model Registry by ARN --
    the only supported source, no fallback. Raises if the package isn't
    Approved, or if the extracted tarball doesn't contain model.pt."""
    sm = boto3.client("sagemaker", region_name=region)
    desc = sm.describe_model_package(ModelPackageName=base_model_package_arn)

    if desc["ModelApprovalStatus"] != "Approved":
        raise RuntimeError(
            f"Refusing to fine-tune from '{base_model_package_arn}': ModelApprovalStatus is "
            f"{desc['ModelApprovalStatus']!r}, not Approved. Approve a base model in the "
            "Model Registry before training."
        )

    model_data_url = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    print(f"Resolved base model package '{base_model_package_arn}' -> {model_data_url}")

    bucket, key = _parse_s3_uri(model_data_url)
    tarball_path = os.path.join(download_dir, "base_model.tar.gz")
    s3 = boto3.client("s3", region_name=region)
    s3.download_file(bucket, key, tarball_path)

    extract_dir = os.path.join(download_dir, "base_model_extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    checkpoint_path = os.path.join(extract_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Extracted '{model_data_url}' but found no model.pt at its root "
            f"(searched {extract_dir}) -- this model package wasn't packaged in the "
            "shape this pipeline expects."
        )
    return checkpoint_path


def resolve_base_model_checkpoint_from_s3(base_model_s3_uri: str, region: str, download_dir: str) -> str:
    """The escape hatch for a generation/size with nothing Approved in the
    registry yet: base_model_s3_uri points directly at either a
    model.tar.gz (extracted, same shape check as the registry path) or a
    bare .pt file (used as-is, no extraction). No approval gate here --
    review_app.py's upload path never puts anything in the registry to
    gate on in the first place."""
    bucket, key = _parse_s3_uri(base_model_s3_uri)
    s3 = boto3.client("s3", region_name=region)

    if key.endswith(".tar.gz"):
        tarball_path = os.path.join(download_dir, "base_model.tar.gz")
        s3.download_file(bucket, key, tarball_path)
        extract_dir = os.path.join(download_dir, "base_model_extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        checkpoint_path = os.path.join(extract_dir, "model.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"Extracted '{base_model_s3_uri}' but found no model.pt at its root (searched {extract_dir})."
            )
        return checkpoint_path

    checkpoint_path = os.path.join(download_dir, "base_model.pt")
    s3.download_file(bucket, key, checkpoint_path)
    return checkpoint_path


def read_dataset_manifest(manifest_s3_uri: str, region: str) -> set:
    """The set of paths a snapshot pinned, relative to the dataset root.

    Returned as a set because the only question asked of it is "is this
    downloaded key in the run", once per object.
    """
    bucket, key = _parse_s3_uri(manifest_s3_uri)
    s3 = boto3.client("s3", region_name=region)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    manifest = json.loads(body.decode("utf-8"))
    files = {str(name).lstrip("/") for name in manifest.get("files") or []}
    if not files:
        raise ValueError(f"{manifest_s3_uri} lists no files.")
    print(f"Dataset manifest {manifest.get('snapshot_id', '?')}: {len(files)} files pinned",
          flush=True)
    return files


def _split_uris(value: str) -> list:
    """One URI or several, comma separated.

    `TRAIN_DATA_S3_URI` carried exactly one prefix for the whole life of
    this image, so a single value has to keep meaning what it meant --
    see the ONE PREFIX / SEVERAL note in the downloader below.
    """
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def drop_undeclared_class_boxes(dataset_root: str, data_yaml_path: str) -> dict:
    """Remove, from this container's copy, every box whose class the yaml
    does not name.

    Ultralytics' `verify_image_label` asserts `max_cls < num_cls` and marks
    the WHOLE image corrupt when it fails -- so one gloves box in a file
    discards the helmet boxes beside it. On a six-class site dataset
    trained as one class, that discarded 6,760 of 6,762 images without a
    single error. This rewrites the label files that were just downloaded
    so only declared classes remain; an image left with no boxes becomes a
    background example, which is what it should be.

    Only files under a `labels` path component are touched, so a path list
    such as retention's `train.txt` at the root is left alone -- its first
    column is a filename, not a class.

    Local only. Nothing is uploaded; S3 keeps every box.
    """
    with open(data_yaml_path) as f:
        names = (yaml.safe_load(f) or {}).get("names") or {}
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    declared = {int(k) for k in names}

    files_seen = files_changed = 0
    dropped: dict = {}
    for root, _dirs, filenames in os.walk(dataset_root):
        parts = os.path.relpath(root, dataset_root).replace(os.sep, "/").split("/")
        if "labels" not in parts:
            continue
        for filename in filenames:
            if not filename.lower().endswith(".txt"):
                continue
            path = os.path.join(root, filename)
            files_seen += 1
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
            keep = []
            changed = False
            for line in lines:
                parts_ = line.split()
                if not parts_:
                    continue
                try:
                    cls = int(float(parts_[0]))
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
                with open(path, "w", encoding="utf-8") as f:
                    f.write("\n".join(keep) + ("\n" if keep else ""))
    return {"files_seen": files_seen, "files_changed": files_changed,
            "dropped": dict(sorted(dropped.items()))}


def apply_label_remap(dataset_root: str, data_yaml_path: str) -> dict:
    """Translate class IDs in the DOWNLOADED copies of label files whose
    source numbered its classes differently. The map is the yaml's
    `x_label_remap: {labels dir: {src: dst}}`; dst -1 drops the box.
    Nothing is uploaded -- S3 keeps the original numbering."""
    with open(data_yaml_path) as f:
        block = (yaml.safe_load(f) or {}).get("x_label_remap") or {}
    changed = 0
    for rel, mapping in block.items():
        mapping = {int(k): int(v) for k, v in mapping.items()}
        base = os.path.join(dataset_root, str(rel).replace("/", os.sep))
        for r, _, fs in os.walk(base):
            for n in fs:
                if not n.lower().endswith(".txt"):
                    continue
                path = os.path.join(r, n)
                out = []
                with open(path, errors="replace") as fh:
                    for line in fh:
                        parts = line.split()
                        if not parts:
                            continue
                        try:
                            dst = mapping.get(int(float(parts[0])), int(float(parts[0])))
                        except ValueError:
                            out.append(line.rstrip("\n"))
                            continue
                        if dst >= 0:
                            out.append(" ".join([str(dst)] + parts[1:]))
                with open(path, "w") as fh:
                    fh.write("\n".join(out) + ("\n" if out else ""))
                changed += 1
    if block:
        print(f"LABEL_REMAP: {changed} label files translated from {len(block)} source(s); S3 unchanged", flush=True)
    return {"files": changed, "sources": len(block)}


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


def preflight_dataset(data_yaml_path: str, *, classes_filter=None) -> dict:
    """Check the downloaded dataset BEFORE any GPU time is spent.

    Every split entry resolves to something on disk; every image folder
    holds images; labels are found where Ultralytics will look (the last
    `/images/` swapped for `/labels/`); every class ID is declared; and no
    image is on both the train and val side. Problems are collected and
    raised together, so one run names everything that is wrong.
    """
    with open(data_yaml_path) as f:
        doc = yaml.safe_load(f) or {}
    root = doc.get("path") or os.path.dirname(data_yaml_path)
    names = doc.get("names") or {}
    nc = len(names)
    errors, report = [], {}
    images_by_split = {}

    def resolve(entry):
        return entry if os.path.isabs(entry) else os.path.join(root, entry)

    def labels_for(image_path):
        parts = image_path.replace(os.sep, "/").split("/")
        for i in range(len(parts) - 1, -1, -1):
            if parts[i] == "images":
                parts[i] = "labels"
                return os.path.splitext("/".join(parts))[0] + ".txt"
        return ""

    for split in ("train", "val"):
        entries = doc.get(split) or []
        entries = entries if isinstance(entries, list) else [entries]
        found = []
        for entry in entries:
            path = resolve(str(entry))
            if path.lower().endswith(".txt") and os.path.isfile(path):
                base = os.path.dirname(path)
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            found.append(os.path.normpath(os.path.join(base, line[2:]) if line.startswith("./") else line))
                continue
            if not os.path.isdir(path):
                errors.append(f"{split}: '{entry}' does not exist in the container (looked at {path}).")
                continue
            here = [os.path.join(r, n) for r, _, fs in os.walk(path) for n in fs
                    if n.lower().endswith(IMAGE_EXTENSIONS)]
            if not here:
                errors.append(f"{split}: '{entry}' holds no images.")
            found += [os.path.normpath(x) for x in here]
        images_by_split[split] = set(found)
        labelled = 0
        max_cls = -1
        for img in found:
            lab = labels_for(img)
            if lab and os.path.isfile(lab):
                labelled += 1
                with open(lab, errors="replace") as f:
                    for line in f:
                        parts = line.split()
                        if parts:
                            try:
                                max_cls = max(max_cls, int(float(parts[0])))
                            except ValueError:
                                pass
        if found and not labelled:
            errors.append(f"{split}: none of its {len(found)} images has a label file where "
                          "Ultralytics looks (the last /images/ swapped for /labels/).")
        if max_cls >= nc and not classes_filter:
            errors.append(f"{split}: labels use class {max_cls} but the yaml declares {nc} "
                          f"class(es), 0-{nc - 1}.")
        report[split] = {"images": len(found), "labelled": labelled, "max_class": max_cls}

    both = images_by_split.get("train", set()) & images_by_split.get("val", set())
    if both:
        errors.append(f"{len(both)} image(s) are in both train and val, e.g. {sorted(both)[0]}.")
    if errors:
        raise SystemExit("DATASET PREFLIGHT FAILED -- nothing was trained:\n  - "
                         + "\n  - ".join(errors))
    print(f"DATASET_PREFLIGHT: ok {report}", flush=True)
    return report


def download_dataset_with_progress(train_data_s3_uri: str, region: str, dest_dir: str,
                                   manifest_files: set | None = None,
                                   data_yaml_s3_uri: str = "") -> str:
    """Downloads every object under train_data_s3_uri (a YOLO dataset root:
    data.yaml + images/ + labels/) into dest_dir, printing
    "DATASET_TOTAL_IMAGES: N" once and "DATASET_IMAGES: n/N" after each
    image file finishes -- pipeline_definition.py's Estimator has
    metric_definitions regexes matching these exact lines, turning them
    into real "dataset:total_images"/"dataset:images_downloaded"
    CloudWatch metrics the review app polls for a live "N / M images
    loaded" progress bar (same technique as train:epoch's regex on
    Ultralytics' own console output). Non-image files (labels, data.yaml)
    are still downloaded, just not counted -- images dominate both the
    file count and the bytes transferred for a YOLO dataset, so counting
    only them is what "how many images have loaded" actually means to
    someone watching this run, and avoids a label file bumping the
    counter for something that isn't an image. Returns the path to the
    downloaded data.yaml, after rewriting its "path" key to dest_dir so
    its relative train/val entries (e.g. "images/train") resolve
    correctly regardless of where this container happens to place it."""
    # ONE PREFIX, OR SEVERAL.
    #
    # With one, every key is written to dest_dir with that prefix
    # STRIPPED -- `images/train/a.jpg`. Every data.yaml ever generated by
    # this system says `images/train`, so that has to keep working
    # exactly as it did.
    #
    # With several, the prefixes are unrelated roots -- a dataset under
    # `Testing_Dataset/`, a golden set under `golden_set/`, week folders
    # under `retraining/` -- and stripping would collide them into one
    # tree. So each key keeps its FULL path under dest_dir, and the yaml
    # names folders by that same path. Self-describing: no index or slot
    # convention to agree on between the two sides.
    uris = _split_uris(train_data_s3_uri)
    if not uris:
        raise ValueError("--train-data-s3-uri is empty.")
    many = len(uris) > 1
    # Several BUCKETS: the same key could exist in two, so every file is
    # laid out at <bucket>/<key>, which is how the yaml names it.
    multi_bucket = len({_parse_s3_uri(u)[0] for u in uris}) > 1
    s3 = boto3.client("s3", region_name=region)

    # Kept for the messages below, which named one prefix when there was
    # only ever one.
    bucket, prefix = _parse_s3_uri(uris[0])
    prefix = prefix.rstrip("/") + "/"

    objects = []
    total_bytes = 0
    skipped = 0
    paginator = s3.get_paginator("list_objects_v2")
    for uri in uris:
        one_bucket, one_prefix = _parse_s3_uri(uri)
        one_prefix = one_prefix.rstrip("/") + "/"
        for page in paginator.paginate(Bucket=one_bucket, Prefix=one_prefix):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith("/"):
                    continue
                # A pinned run takes only what the snapshot named.
                # Filtering HERE rather than deleting after the fact
                # means the run does not pay to transfer images it will
                # not train on -- a week folder that has doubled since
                # the snapshot would otherwise double the download for
                # nothing.
                #
                # data.yaml is always kept: it is the thing the trainer
                # opens, and it is not an image the manifest would list.
                rel = obj["Key"][len(one_prefix):]
                if manifest_files is not None:
                    if rel != "data.yaml" and rel not in manifest_files:
                        skipped += 1
                        continue
                # Where it lands. One prefix strips it; several keep the
                # full key, so unrelated roots cannot collide.
                objects.append({
                    "Bucket": one_bucket,
                    "Key": obj["Key"],
                    "Size": obj["Size"],
                    "Local": (f"{one_bucket}/{obj['Key']}" if multi_bucket
                              else obj["Key"] if many else rel),
                })
                total_bytes += obj["Size"]

    if not objects:
        where = ", ".join(uris)
        raise FileNotFoundError(f"No objects found under {where}")

    if manifest_files is not None:
        # Named but absent. Refusing is the point of a pinned run: a
        # manifest that silently trains on 900 of its 1,000 images is
        # worse than one that fails, because the run looks finished.
        present = {obj["Local"] for obj in objects}
        missing = sorted(manifest_files - present)
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} of {len(manifest_files)} files named by the dataset "
                f"manifest are not under s3://{bucket}/{prefix} -- this run would not "
                f"reproduce the one it was pinned to. First few: {missing[:5]}"
            )
        print(f"Manifest matched: {len(present) - 1} files kept, {skipped} skipped",
              flush=True)

    total_images = sum(1 for obj in objects if obj["Key"].lower().endswith(IMAGE_EXTENSIONS))
    print(f"Downloading dataset: {len(objects)} files ({total_images} images), {total_bytes / 1e6:.1f} MB total", flush=True)

    # SageMaker's metric_definitions CloudWatch Logs metric filter takes a
    # few seconds to attach after the training job starts -- confirmed via
    # a real run where train:epoch (which prints continuously for minutes)
    # got real datapoints while a fast-finishing download's own progress
    # metric got ZERO, despite the exact matching lines being present in
    # the raw CloudWatch Logs the whole time. This delay buys the filter
    # time to attach before any of this function's own scraped lines are
    # emitted, so the review app's progress bar actually has data to show
    # instead of silently staying empty for fast downloads.
    time.sleep(20)

    # Printed only after the sleep above, same reasoning -- this is the
    # progress bar's denominator (dataset:total_images), so it needs the
    # metric filter already attached too, not just the per-image lines
    # below.
    print(f"DATASET_TOTAL_IMAGES: {total_images}", flush=True)

    images_downloaded = 0
    for obj in objects:
        key = obj["Key"]
        dest_path = os.path.join(dest_dir, obj["Local"].replace("/", os.sep))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        s3.download_file(obj["Bucket"], key, dest_path)
        if key.lower().endswith(IMAGE_EXTENSIONS):
            images_downloaded += 1
            print(f"DATASET_IMAGES: {images_downloaded}/{total_images}", flush=True)

    data_yaml_path = os.path.join(dest_dir, "data.yaml")
    # WHERE THE YAML COMES FROM.
    #
    # With one prefix it rides along in the download, at the root, as it
    # always has. With several there is no single root to find it at --
    # each source keeps its full key path, and none of them is "the"
    # dataset -- so the run is told explicitly. The yaml is a per-run
    # artifact in either case; this only changes where it is fetched
    # from.
    if data_yaml_s3_uri:
        yaml_bucket, yaml_key = _parse_s3_uri(data_yaml_s3_uri)
        s3.download_file(yaml_bucket, yaml_key, data_yaml_path)
        print(f"data.yaml taken from {data_yaml_s3_uri}", flush=True)
    if not os.path.isfile(data_yaml_path):
        where = data_yaml_s3_uri or ", ".join(uris)
        raise FileNotFoundError(
            f"{data_yaml_path} not found after downloading {where} -- expected a "
            "YOLO data.yaml (train/val paths + class names) at the dataset root, "
            "or --data-yaml-s3-uri naming one."
        )
    with open(data_yaml_path) as f:
        data_yaml_contents = yaml.safe_load(f)
    # Every split written out ABSOLUTE, not as a relative entry resolved
    # against `path`. Both forms train identically; only this one leaves
    # the yaml saying, on its own, which directory each split was read
    # from. A relative "images/train" is only meaningful next to the
    # `path` it happened to be resolved against, and that pairing is not
    # recoverable from the run afterwards -- which is exactly the question
    # an audit asks. `path` is kept as well, for anything that reads it.
    data_yaml_contents["path"] = dest_dir
    for split_key in ("train", "val", "test"):
        entry = data_yaml_contents.get(split_key)
        if isinstance(entry, str) and entry and not os.path.isabs(entry):
            data_yaml_contents[split_key] = os.path.join(
                dest_dir, entry.replace("/", os.sep),
            )
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_contents, f, sort_keys=False)

    # The resolved layout, into the training log. dest_dir is inside a
    # TemporaryDirectory and is gone the moment the container exits, so
    # the S3 root is printed beside it: that is the durable identity of
    # what was read, and the only one still meaningful after the run.
    print(f"DATASET_SOURCE_S3_URI: s3://{bucket}/{prefix}", flush=True)
    print(f"DATASET_LOCAL_ROOT: {dest_dir}", flush=True)
    for split_key in ("train", "val", "test"):
        if data_yaml_contents.get(split_key):
            print(f"DATASET_SPLIT_PATH {split_key}: {data_yaml_contents[split_key]}", flush=True)

    return data_yaml_path


#: Ours, not the operator's. These come from the dataset, the form and
#: the pipeline, and letting a blob override them would mean the audit row
#: and the run disagreeing about what was trained.
RESERVED_HYPERPARAMS = {
    "data", "model", "epochs", "imgsz", "batch", "seed", "project", "name",
    "exist_ok", "resume", "device", "workers", "save_dir", "classes",
    "single_cls", "patience",
}


def _validated_hyperparams(raw):
    """Parse the blob, keep what Ultralytics knows, refuse what it does not.

    Returns {} for an empty value, so a run that sets nothing behaves
    exactly as it did before this existed.
    """
    if not raw or not str(raw).strip():
        return {}
    text = str(raw).strip()
    # Base64 OR plain JSON. The app sends base64 because this value travels
    # as a SageMaker hyperparameter, which becomes a command-line argument
    # -- and a JSON object full of quotes, braces and spaces through that
    # layer is the kind of quoting that breaks in ways that look like the
    # parameter being ignored. Base64 has none of those characters.
    #
    # Plain JSON is still accepted so this can be driven by hand from the
    # console or a local run without encoding anything first.
    # JSON first, then base64. Checking for a leading "{" instead sent
    # `[1, 2]` -- valid JSON, wrong shape -- down the base64 path, and the
    # operator got "not base64" about something that was plainly JSON.
    # Try to read it as what it looks like, and only then as an encoding.
    given = None
    try:
        given = json.loads(text)
    except ValueError:
        try:
            text = base64.b64decode(text, validate=True).decode("utf-8")
        except Exception as exc:                                   # noqa: BLE001
            raise SystemExit(
                f"TRAIN_HYPERPARAMS is neither JSON nor base64-encoded JSON: {exc}")
        try:
            given = json.loads(text)
        except ValueError as exc:
            raise SystemExit(
                f"TRAIN_HYPERPARAMS decoded, but is not valid JSON: {exc}")
    if not isinstance(given, dict):
        raise SystemExit("TRAIN_HYPERPARAMS must be a JSON object.")

    from ultralytics.cfg import DEFAULT_CFG_DICT

    unknown = sorted(k for k in given if k not in DEFAULT_CFG_DICT)
    if unknown:
        raise SystemExit(
            f"TRAIN_HYPERPARAMS: Ultralytics {_ultralytics_version()} has no "
            f"{', '.join(unknown)}. Check the spelling -- a key it does not know "
            "would be dropped in silence and the run would train on defaults."
        )
    reserved = sorted(k for k in given if k in RESERVED_HYPERPARAMS)
    if reserved:
        raise SystemExit(
            f"TRAIN_HYPERPARAMS: {', '.join(reserved)} is decided by the pipeline, "
            "not here. Setting it would make the audit row and the run disagree."
        )

    kept = {k: v for k, v in given.items()}
    if kept:
        print("HYPERPARAMS: " + ", ".join(f"{k}={v}" for k, v in sorted(kept.items())),
              flush=True)
    return kept


def _ultralytics_version():
    try:
        import ultralytics

        return ultralytics.__version__
    except Exception:                                              # noqa: BLE001
        return "(unknown)"


def main():
    args = parse_args()

    with tempfile.TemporaryDirectory() as work_dir:
        if args.base_model_package_arn:
            base_checkpoint_path = resolve_base_model_checkpoint(args.base_model_package_arn, args.region, work_dir)
        else:
            base_checkpoint_path = resolve_base_model_checkpoint_from_s3(args.base_model_s3_uri, args.region, work_dir)
        model = YOLO(base_checkpoint_path)  # loaded into memory here; the extracted file isn't needed after this

        dataset_dir = os.path.join(work_dir, "dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        # A pinned run takes exactly the files its snapshot named; an
        # unpinned one takes the prefix, as before. The difference is
        # whether this run can be repeated later, after the week folders
        # have carried on filling.
        manifest_files = (
            read_dataset_manifest(args.dataset_manifest_s3_uri, args.region)
            if args.dataset_manifest_s3_uri else None
        )
        data_yaml = download_dataset_with_progress(
            args.train_data_s3_uri, args.region, dataset_dir, manifest_files,
            data_yaml_s3_uri=args.data_yaml_s3_uri or "")

        # Boxes the yaml cannot name, dropped from THIS copy before
        # Ultralytics builds its cache -- which is where it would otherwise
        # discard the whole image. Said in the log, per class, because a
        # filter that works silently is indistinguishable from data that
        # was never there.
        # A TEMPORARY filtered dataset beside the download: images linked,
        # labels translated (remap), filtered (when the box is ticked),
        # renumbered 0..n-1 and checked for malformed lines. The download
        # itself is never modified, and training reads only the copy.
        data_yaml, _filtered = build_filtered_dataset(
            dataset_dir, data_yaml, drop_undeclared=args.drop_undeclared_classes)

        # Validated on the filesystem the run will actually read, after the
        # filter, before the GPU. A dataset that cannot train stops here
        # with every problem named rather than an hour in.
        preflight_dataset(data_yaml, classes_filter=args.classes)

        # ALWAYS passed, 0 included.
        #
        # 0 means "never stop early" -- which is what the card promises
        # and what makes two runs comparable. Ultralytics implements it:
        #
        #     self.patience = patience or float("inf")
        #       -- EarlyStopping.__init__, verified on 8.4.130
        #
        # This used to withhold 0 on the belief that it meant "stop at
        # the first epoch that does not improve". It does not; it means
        # no limit. Withholding it handed the run Ultralytics' OWN
        # default of 100 instead, so "Patience 0" silently meant "stop
        # after 100 epochs without improvement". Invisible on a 50-epoch
        # run, which cannot reach 100 -- and live on anything longer.
        extra = {"patience": args.patience}
        print(f"PATIENCE: {args.patience or '0 (never stop early)'}")

        # Everything else the operator set, validated against Ultralytics'
        # own config rather than a list copied into this file.
        #
        # WHY A BLOB. One flag per parameter would mean rebuilding and
        # redeploying this image every time somebody wants another knob --
        # and this image is NOT built by deploy.sh, it has its own
        # CodeBuild project. A validated blob moves that cost to zero: the
        # app sends what it has, this checks it against the installed
        # Ultralytics, and a version bump that adds a parameter exposes it
        # with no change here at all.
        #
        # REFUSED, not ignored. A typo like "optimiser" would otherwise be
        # dropped in silence and the run would train on defaults while the
        # audit recorded the operator's intent -- two different runs with
        # one record. So an unknown key fails the job before the GPU
        # starts, which costs seconds instead of an hour.
        extra.update(_validated_hyperparams(args.hyperparams))

        # Class filtering, done by Ultralytics at load time rather than by
        # rewriting the dataset in S3.
        #
        #   classes=[0]      -> BaseDataset.update_labels drops every box
        #                       of any other class
        #   single_cls=True  -> renumbers what survives to 0, and the
        #                       trainer sets names={0: "item"} and nc=1,
        #                       so the head is built with one output
        #
        # That turns a six-class site dataset into a one-class detector
        # without touching a file. The alternative was deriving a filtered
        # COPY of every image and label, which is a second corpus to store,
        # keep in step and explain -- and which could not read a dataset
        # whose yaml names week folders in place.
        if args.classes:
            extra["classes"] = args.classes
            print(f"CLASSES: {args.classes}")
        if args.single_cls:
            extra["single_cls"] = True
            # Said plainly because Ultralytics renames the class to "item"
            # and the run's own logs will show that rather than the name
            # anybody chose.
            print("SINGLE_CLS: true (Ultralytics will log the class as 'item'; "
                  "the registered model takes its name from the detector)")

        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            seed=args.seed,
            project=args.model_dir,
            name="run",
            exist_ok=True,
            **extra,
        )

    best_weights = os.path.join(args.model_dir, "run", "weights", "best.pt")
    model_pt_path = os.path.join(args.model_dir, "model.pt")
    if os.path.exists(best_weights):
        shutil.copy(best_weights, model_pt_path)

    # Bundles this training image's own inference.py into the artifact's
    # code/ dir, so whatever SageMaker tars up as model.tar.gz for the
    # RegisterModel step already carries the custom handler. Confirmed
    # live: baking inference.py into the *serving* image alone
    # (Dockerfile.serve's COPY + SAGEMAKER_PROGRAM env var) is NOT
    # reliably picked up by the sagemaker-pytorch-inference-toolkit --
    # every model deployed that way fell back to its default handler
    # (which requires a TorchScript .pt and rejects a plain Ultralytics
    # checkpoint) regardless. Bundling the handler inside the model
    # tarball itself, at code/inference.py, is what AWS's own tooling
    # actually honors.
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy(os.path.join(os.path.dirname(__file__), "inference.py"), os.path.join(code_dir, "inference.py"))

    print(f"Training complete. Results: {results}")


if __name__ == "__main__":
    main()
