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
    parser.add_argument("--base-model-package-arn", type=str, default=None)
    parser.add_argument("--base-model-s3-uri", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train-data-s3-uri", type=str,
                         default=os.environ.get("TRAIN_DATA_S3_URI"),
                         help="s3:// URI of the YOLO dataset root (contains data.yaml, images/, labels/) -- "
                              "downloaded by this script itself, not a SageMaker input channel (see module docstring).")
    parser.add_argument("--region", type=str, default=os.environ.get("AWS_REGION", "ap-southeast-1"))
    args = parser.parse_args()
    if bool(args.base_model_package_arn) == bool(args.base_model_s3_uri):
        parser.error("Pass exactly one of --base-model-package-arn or --base-model-s3-uri, not both/neither.")
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


def download_dataset_with_progress(train_data_s3_uri: str, region: str, dest_dir: str) -> str:
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
    bucket, prefix = _parse_s3_uri(train_data_s3_uri)
    prefix = prefix.rstrip("/") + "/"
    s3 = boto3.client("s3", region_name=region)

    objects = []
    total_bytes = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith("/"):
                continue
            objects.append(obj)
            total_bytes += obj["Size"]

    if not objects:
        raise FileNotFoundError(f"No objects found under s3://{bucket}/{prefix}")

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
        rel_path = key[len(prefix):]
        dest_path = os.path.join(dest_dir, rel_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        s3.download_file(bucket, key, dest_path)
        if key.lower().endswith(IMAGE_EXTENSIONS):
            images_downloaded += 1
            print(f"DATASET_IMAGES: {images_downloaded}/{total_images}", flush=True)

    data_yaml_path = os.path.join(dest_dir, "data.yaml")
    if not os.path.isfile(data_yaml_path):
        raise FileNotFoundError(
            f"{data_yaml_path} not found after downloading s3://{bucket}/{prefix} -- expected a "
            "YOLO data.yaml (train/val paths + class names) at the dataset root."
        )
    with open(data_yaml_path) as f:
        data_yaml_contents = yaml.safe_load(f)
    data_yaml_contents["path"] = dest_dir
    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data_yaml_contents, f)

    return data_yaml_path


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
        data_yaml = download_dataset_with_progress(args.train_data_s3_uri, args.region, dataset_dir)

        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            seed=args.seed,
            project=args.model_dir,
            name="run",
            exist_ok=True,
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
