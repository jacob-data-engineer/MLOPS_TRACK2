"""Shared config, boto3 clients, and business logic for every `track2_*`
router — the YOLO/Track2 training-pipeline control surface (review queue,
model registry, training runs, scheduling, deploy, inference, lineage,
notifications) that used to live in the standalone stdlib app at
`terraform/review-app/review_app.py`. That app had no authentication and
ran on its own public EC2 instance; this migration ports its business
logic into JP_UI's FastAPI app (gated by the existing session auth) and
its React shell (matching the rest of the UI's look), dropping only the
old app's hand-rolled HTML rendering and manual query-string routing.

Config is read directly from `os.environ`, matching this API's existing
no-central-Settings convention (see `auth.py`).
"""

from __future__ import annotations

import json
import os
import re
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import boto3
import botocore.config
import botocore.exceptions

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
TABLE_NAME = os.environ.get("REVIEW_QUEUE_TABLE", "")
BUCKET = os.environ.get("PIPELINE_BUCKET", "")
RULE_NAME = os.environ.get("RETRAIN_RULE_NAME", "track2-yolo-weekly-retrain-schedule")
PIPELINE_NAME = os.environ.get("TRAINING_PIPELINE_NAME", "track2-yolo-training-pipeline")

# Where real training datasets actually live -- NOT a bare "datasets/"
# prefix, which nothing has ever been uploaded to.
DATASETS_S3_PREFIX = "data/track2-datasets/sagemaker_training/track2"
DEFAULT_TRAIN_DATASET = f"{DATASETS_S3_PREFIX}/escalation_dataset"

# Kept in sync with model_training/training-container/train.py's own
# IMAGE_EXTENSIONS by convention (not imported -- separate deploy unit).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

# YOLO11 only, by explicit product decision -- other generations/Nano are
# deliberately excluded, not just "not yet added". Must match
# model_training/src/config.py's GENERATION_SIZES exactly.
GENERATION_SIZES = {"11": ("s", "m", "l", "x")}
GENERATION_LABELS = {"11": "YOLO11"}
SIZE_LABELS = {"s": "Small", "m": "Medium", "l": "Large", "x": "Extra-Large"}
YOLO_GENERATIONS = tuple(GENERATION_SIZES)

# The one shared, always-on endpoint hosting all four sizes as separate
# Inference Components -- see model_training/src/deploy_inference_component.py.
MULTI_SIZE_ENDPOINT_NAME = os.environ.get("MULTI_SIZE_ENDPOINT_NAME", "track2-yolo-multi-size-endpoint")
MULTI_SIZE_ENDPOINT_CONFIG_NAME = os.environ.get("MULTI_SIZE_ENDPOINT_CONFIG_NAME", "track2-yolo-multi-size-config")
MULTI_SIZE_VARIANT_NAME = os.environ.get("MULTI_SIZE_VARIANT_NAME", "AllSizes")
# Matches model_training/src/config.py's multi_size_instance_type -- the
# endpoint is CPU-only by design (GPU_INSTANCE_TYPE below is for training
# jobs, which contend for the account's single GPU slot).
MULTI_SIZE_INSTANCE_TYPE = os.environ.get("MULTI_SIZE_INSTANCE_TYPE", "ml.m5.2xlarge")
SAGEMAKER_EXECUTION_ROLE_ARN = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", "")
SERVING_IMAGE_URI = os.environ.get("SERVING_IMAGE_URI", "")
RUN_PIPELINE_CODEBUILD_PROJECT = os.environ.get("RUN_PIPELINE_CODEBUILD_PROJECT", "track2-yolo-run-pipeline")

GPU_INSTANCE_TYPE = "ml.g4dn.xlarge"

COMPUTE_REQUIREMENTS = {
    "s": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 2048},
    "m": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 3072},
    "l": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 4096},
    "x": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 6144},
}
# The four real serving sizes -- one Model Registry section per size, not
# just "s" alone.
REGISTRY_SIZES = tuple(COMPUTE_REQUIREMENTS)

# Pre-existing, reused table (hash key "id") -- see pipeline_definition.py's
# DYNAMODB_LINEAGE_TABLE / _write_run_metadata and
# register_baseline_model.py's _write_baseline_lineage, which write here.
LINEAGE_TABLE_NAME = "track2-dynamoDB"

MAX_QUEUE_SIZE = 10

DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
DAY_LABELS = {"SUN": "Sunday", "MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday",
              "THU": "Thursday", "FRI": "Friday", "SAT": "Saturday"}

# boto3 clients are documented thread-safe -- created once and shared
# across every request.
s3 = boto3.client("s3", region_name=REGION)
codebuild = boto3.client("codebuild", region_name=REGION)
events = boto3.client("events", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)
cloudwatch = boto3.client("cloudwatch", region_name=REGION)
sns = boto3.client("sns", region_name=REGION)
# Extended read_timeout: a cold Inference Component can legitimately take
# minutes to respond (ContainerStartupHealthCheckTimeoutInSeconds=600 in
# deploy_inference_component.py) -- boto3's 60s default falsely reports
# "Inference failed" for a model that was only ever slow to wake up.
# Retries off so a slow-but-eventually-successful cold start isn't retried
# from zero partway through.
sm_runtime = boto3.client(
    "sagemaker-runtime", region_name=REGION,
    config=botocore.config.Config(read_timeout=600, connect_timeout=10, retries={"max_attempts": 1}),
)
# Registers/updates the scale-to-zero policy on each size's Inference
# Component -- see ensure_scale_to_zero below.
autoscaling = boto3.client("application-autoscaling", region_name=REGION)
# Cost Explorer is a global service whose API only answers in us-east-1 --
# pointing it at REGION returns an endpoint error, not regional costs. The
# data it returns still covers every region; usage types carry their own
# region prefix ("APS1-Host:ml.m5.2xlarge").
cost_explorer = boto3.client("ce", region_name="us-east-1")

# boto3 *resources* (what .Table() returns) are explicitly not documented
# thread-safe, unlike clients above -- one boto3.resource per thread via
# threading.local (credential-refresh races under concurrency otherwise).
_thread_local = threading.local()


def _dynamodb():
    if not hasattr(_thread_local, "dynamodb"):
        _thread_local.dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return _thread_local.dynamodb


def table():
    return _dynamodb().Table(TABLE_NAME)


def lineage_table():
    return _dynamodb().Table(LINEAGE_TABLE_NAME)



# Resolved lazily (not at import time) so importing this module never
# makes a network call on its own -- tests can mock boto3 first.
_account_id_cache: str | None = None


def account_id() -> str:
    global _account_id_cache
    if _account_id_cache is None:
        _account_id_cache = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
    return _account_id_cache



def reviewer_notifications_topic_arn() -> str:
    # The one SNS topic every human-facing notification already goes
    # through (training pipeline status, drift-triggered review needed,
    # weights approved) -- see terraform/infra/monitoring.tf.
    return f"arn:aws:sns:{REGION}:{account_id()}:track2-yolo-reviewer-notifications"


# --- in-memory state --------------------------------------------------------
# Both of these are module-level, in-memory only, and lost on a process
# restart -- an accepted limitation matching this API's own existing
# single-worker/in-memory-session-store constraint (see session_store.py).

# Ad-hoc Training Queue: FIFO list of pending runs, advanced by a
# background poll loop (see queue_poll_loop, started from main.py's
# lifespan) since nothing else in this process runs on a schedule.
_queue_lock = threading.Lock()
_training_queue: list = []
_queue_running: dict | None = None
_queue_started = False
_queue_completed_count = 0
_queue_last_error = ""

# Bridges the real gap between "a codebuild.start_build call succeeded"
# and "a new SageMaker Pipeline Execution actually exists for
# get_training_progress to report on" -- run_pipeline.py has to check out
# source and run inside CodeBuild first, which takes real time, during
# which the UI would otherwise keep showing the PREVIOUS run's already-
# terminal status.
_pending_submission_lock = threading.Lock()
_pending_submission: dict | None = None  # {"build_id": str, "triggered_at": datetime} or None

TRACK2_SOURCE_TAG_KEY = "track2-source"
TRACK2_VERSION_TAG_KEY = "track2-version"


# --- schedule helpers --------------------------------------------------------

def sgt_to_utc_hour(local_hour: int) -> tuple[int, int]:
    """Returns (utc_hour, day_shift) -- day_shift is -1 if the SGT->UTC
    conversion (SGT is UTC+8, no DST) rolls back to the previous day."""
    utc_hour = local_hour - 8
    if utc_hour < 0:
        return utc_hour + 24, -1
    return utc_hour, 0


def cron_and_summary_from_schedule_form(mode: str, form: dict) -> tuple[str, str]:
    """Shared by the single global schedule and every Automated Training
    Queue -- same Daily/Weekly/Monthly fields, same SGT->UTC conversion,
    turned into both a real EventBridge cron(...) expression and a human-
    readable summary. `form` values are plain (not list-wrapped, unlike the
    old app's `parse_qs` form). Raises ValueError for "manual" or an
    unrecognized mode."""
    if mode == "daily":
        hour = int(form["daily_hour"])
        utc_hour, _ = sgt_to_utc_hour(hour)
        return f"cron(0 {utc_hour} * * ? *)", f"Daily at {hour:02d}:00 SGT"
    if mode == "weekly":
        day = form["weekly_day"]
        hour = int(form["weekly_hour"])
        utc_hour, shift = sgt_to_utc_hour(hour)
        cron_day = DAYS[(DAYS.index(day) - 1) % 7] if shift else day
        return f"cron(0 {utc_hour} ? * {cron_day} *)", f"Weekly on {DAY_LABELS[day]} at {hour:02d}:00 SGT"
    if mode == "monthly":
        dom = int(form["monthly_day"])
        hour = int(form["monthly_hour"])
        utc_hour, shift = sgt_to_utc_hour(hour)
        cron_dom = (dom - 1 if dom > 1 else 28) if shift else dom
        return f"cron(0 {utc_hour} {cron_dom} * ? *)", f"Monthly on day {dom} at {hour:02d}:00 SGT"
    raise ValueError(f"Unknown schedule mode: {mode!r}")


def current_schedule() -> dict:
    try:
        rule = events.describe_rule(Name=RULE_NAME)
        return {"enabled": rule["State"] == "ENABLED", "expression": rule.get("ScheduleExpression", "")}
    except events.exceptions.ResourceNotFoundException:
        return {"enabled": False, "expression": ""}


# --- review queue + model registry -------------------------------------------

def review_queue_counts() -> dict:
    items = table().scan()["Items"]
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for item in items:
        status = item.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts


def fetch_lineage(lineage_id: str) -> dict | None:
    try:
        item = lineage_table().get_item(Key={"id": lineage_id}).get("Item")
        return dict(item) if item else None
    except Exception as exc:
        print(f"Could not get lineage {lineage_id}: {exc}")
        return None


def scan_all_lineage() -> list:
    """Every record in the lineage table (training_run + baseline), newest
    first by timestamp_utc."""
    items = []
    kwargs: dict = {}
    while True:
        response = lineage_table().scan(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key
    items.sort(key=lambda item: str(item.get("timestamp_utc") or ""), reverse=True)
    return items


def model_registry_summary(group_name: str) -> dict:
    try:
        packages = sm.list_model_packages(
            ModelPackageGroupName=group_name, SortBy="CreationTime", SortOrder="Descending",
        )["ModelPackageSummaryList"]
    except botocore.exceptions.ClientError as exc:
        # A size whose group hasn't been created yet -- "nothing
        # registered" rather than erroring the whole page out.
        if "does not exist" not in str(exc):
            raise
        packages = []
    approved = sum(1 for p in packages if p["ModelApprovalStatus"] == "Approved")
    # list_model_packages' summary has no description/metadata -- only
    # fetched for the first 8 (what the Review tab actually shows).
    for pkg in packages[:8]:
        try:
            desc = sm.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
            pkg["Description"] = desc.get("ModelPackageDescription") or ""
            lineage_id = desc.get("CustomerMetadataProperties", {}).get("lineage_id")
            pkg["Lineage"] = fetch_lineage(lineage_id) if lineage_id else None
        except Exception as exc:
            print(f"Could not describe {pkg['ModelPackageArn']}: {exc}")
            pkg["Description"] = ""
            pkg["Lineage"] = None
    return {"total": len(packages), "approved": approved, "packages": packages}


def all_model_registries() -> dict:
    """One summary per real serving size, keyed by group name."""
    return {f"yolo11{size}-detector": model_registry_summary(f"yolo11{size}-detector") for size in REGISTRY_SIZES}


def serialize_registries_brief() -> dict:
    """`all_model_registries()`, JSON-serializable and trimmed to what the
    Deploy/Export pages need (no lineage/description) -- shared so both
    routers list registry versions identically."""
    registries = all_model_registries()
    return {
        group: {
            "total": summary["total"],
            "approved": summary["approved"],
            "packages": [
                {
                    "model_package_arn": p["ModelPackageArn"],
                    "version": p["ModelPackageArn"].rsplit("/", 1)[-1],
                    "approval_status": p["ModelApprovalStatus"],
                }
                for p in summary["packages"]
            ],
        }
        for group, summary in registries.items()
    }


def latest_approved_lookup(registries: dict) -> dict:
    """{generation: {size: {"group", "version"} | None}} -- whichever
    Model Package the "no .pt file available" checkbox would resolve to
    (mirrors pipeline_definition.py's _resolve_latest_approved_base_model
    exactly: latest-by-CreationTime Approved package in that group)."""
    result: dict = {}
    for generation, sizes in GENERATION_SIZES.items():
        result[generation] = {}
        for size in sizes:
            group_name = f"yolo{generation}{size}-detector"
            packages = registries.get(group_name, {}).get("packages", [])
            approved = next((p for p in packages if p["ModelApprovalStatus"] == "Approved"), None)
            result[generation][size] = (
                {"group": group_name, "version": approved["ModelPackageArn"].rsplit("/", 1)[-1]}
                if approved else None
            )
    return result


def _read_evaluation_metrics(data: dict) -> dict:
    """The detection numbers out of an evaluation.json, whichever shape it
    is in.

    The legacy evaluator wrote them flat at the top level. The full-frame
    evaluator that actually runs now nests them under "metrics" and reports
    f1 where the old one reported accuracy -- read flat-only, every
    full-frame run looked like it had no metrics at all. Missing keys stay
    None rather than 0: "not measured" and "measured as zero" are not the
    same thing on a model card.
    """
    source = data.get("metrics") if isinstance(data.get("metrics"), dict) else data
    return {
        "mAP50": source.get("mAP50"),
        "mAP50-95": source.get("mAP50-95"),
        "precision": source.get("precision"),
        "recall": source.get("recall"),
        "f1": source.get("f1"),
        "accuracy": source.get("accuracy"),
    }


def fetch_metrics_and_lineage(model_package_arn: str) -> tuple[dict | None, str, dict | None]:
    """One describe_model_package call -> (metrics-or-None, description,
    lineage-or-None). Metrics: the 5 flat numbers (mAP50, mAP50-95,
    precision, recall, accuracy) from evaluation.json recorded on
    ModelMetrics at registration time. Baseline registrations attach no
    ModelMetrics at all -- None for those."""
    try:
        desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    except Exception as exc:
        print(f"Could not describe model package {model_package_arn}: {exc}")
        return None, "", None

    description = desc.get("ModelPackageDescription") or ""
    lineage_id = desc.get("CustomerMetadataProperties", {}).get("lineage_id")
    lineage = fetch_lineage(lineage_id) if lineage_id else None

    metrics = None
    s3_uri = desc.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {}).get("S3Uri")
    if s3_uri and s3_uri.startswith("s3://"):
        try:
            bucket, _, key = s3_uri[len("s3://"):].partition("/")
            # pipeline_definition.py's evaluation_s3_uri already ends in
            # "/" -- the S3Uri that comes back is literally
            # ".../evaluation-output//evaluation.json" (double slash)
            # while the real object sits at the single-slash key. S3 keys
            # aren't URL-normalized, so the double slash 404s verbatim.
            key = re.sub(r"/+", "/", key)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            metrics = _read_evaluation_metrics(json.loads(body))
        except Exception as exc:
            print(f"Could not fetch evaluation metrics for {model_package_arn}: {exc}")

    return metrics, description, lineage


def scan_all_model_metrics() -> list:
    """One entry per registered model package version across every size --
    each row also carries training notes and full lineage."""
    rows = []
    for size in REGISTRY_SIZES:
        group_name = f"yolo11{size}-detector"
        summary = model_registry_summary(group_name)
        for pkg in summary["packages"]:
            metrics, description, lineage = fetch_metrics_and_lineage(pkg["ModelPackageArn"])
            rows.append({
                "group_name": group_name,
                "size": size,
                "version": pkg["ModelPackageArn"].rsplit("/", 1)[-1],
                "model_package_arn": pkg["ModelPackageArn"],
                "status": pkg["ModelApprovalStatus"],
                "metrics": metrics,
                "description": description,
                "lineage": lineage,
            })
    # Group order matches REGISTRY_SIZES; within a group, best mAP50-95
    # first. Rows with no metrics (baselines, fetch failures) sort last.
    size_order = {size: i for i, size in enumerate(REGISTRY_SIZES)}
    rows.sort(key=lambda r: (
        size_order[r["size"]],
        -(r["metrics"]["mAP50-95"] if r["metrics"] and isinstance(r["metrics"].get("mAP50-95"), (int, float)) else -1),
    ))
    return rows


# --- S3 browsing / weights ----------------------------------------------------

def list_s3_folders(prefix: str) -> list:
    """"Folders" directly under prefix (S3 has none really -- this is
    list_objects_v2's CommonPrefixes emulation via Delimiter)."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"][len(prefix):].rstrip("/") for p in response.get("CommonPrefixes", [])]


def list_s3_images(prefix: str) -> list:
    """Image files (matching IMAGE_EXTENSIONS) directly under prefix."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    names = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key == prefix:
            continue
        name = key[len(prefix):]
        if name.lower().endswith(IMAGE_EXTENSIONS):
            names.append(name)
    return names


WEIGHT_FILE_EXTENSIONS = (".pt", ".tar.gz")


def list_s3_weight_files(prefix: str) -> list:
    """Checkpoint files (.pt/.tar.gz) directly under prefix -- lets a
    picker show and select one specific file rather than assuming a
    fixed "model.pt" filename inside whatever folder was chosen."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    names = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key == prefix:
            continue
        name = key[len(prefix):]
        if name.lower().endswith(WEIGHT_FILE_EXTENSIONS):
            names.append(name)
    return names


def list_s3_folders_and_weight_files(prefix: str) -> tuple[list, list]:
    """Both of the two functions above, from ONE list_objects_v2 round
    trip -- they issue byte-identical Bucket/Prefix/Delimiter requests, so
    calling them in sequence (as the folder browser did on every click)
    paid for the same S3 response twice."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    folders = [p["Prefix"][len(prefix):].rstrip("/") for p in response.get("CommonPrefixes", [])]
    weight_files = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key == prefix:
            continue
        name = key[len(prefix):]
        if name.lower().endswith(WEIGHT_FILE_EXTENSIONS):
            weight_files.append(name)
    return folders, weight_files


def resolve_base_model_s3_uri(folder: str) -> str:
    """Resolves a weights-browser folder selection to the exact object key
    train.py's --base-model-s3-uri expects (accepts either suffix directly)."""
    for candidate in ("model.tar.gz", "model.pt"):
        key = f"{folder}/{candidate}"
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            return f"s3://{BUCKET}/{key}"
        except botocore.exceptions.ClientError:
            continue
    raise FileNotFoundError(f"Neither model.tar.gz nor model.pt found under s3://{BUCKET}/{folder}/")


# Exact copy of model_training/training-container/inference.py's serving
# entrypoint (SageMaker's model_fn/input_fn/predict_fn/output_fn contract).
# Confirmed live (via CloudWatch container logs) that baking this into the
# serving image alone -- COPY inference.py + ENV SAGEMAKER_PROGRAM=
# inference.py in Dockerfile.serve -- is NOT enough: the PyTorch inference
# toolkit still fell back to its default handler (which requires a
# TorchScript .pt and rejects a plain checkpoint) regardless of that
# image-level config, tried both as the Model resource's Environment and
# as an inline InferenceComponent Container spec. AWS's own known-issue
# writeups for this exact symptom ("custom inference.py ignored, falls
# back to default handler") agree: the toolkit only reliably picks it up
# from inside the model tarball itself, at code/inference.py -- so every
# tarball this app builds now carries its own copy, rather than relying
# on the serving image to already have it.
_INFERENCE_PY_SOURCE = '''\
"""Serving entrypoint for the SageMaker PyTorch Inference container
(SageMaker's model_fn/input_fn/predict_fn/output_fn contract).

Output shape ({"detections": [{"confidence": ..., "class": ..., "box":
[...]}]}) is a fixed contract confidence_monitor.py depends on.
"""

import io
import json

from PIL import Image
from ultralytics import YOLO

_MODEL = None


def model_fn(model_dir):
    global _MODEL
    _MODEL = YOLO(f"{model_dir}/model.pt")
    return _MODEL


def input_fn(request_body, content_type):
    if content_type != "image/jpeg":
        raise ValueError(f"Unsupported content type: {content_type}")
    return Image.open(io.BytesIO(request_body)).convert("RGB")


def predict_fn(image, model):
    results = model.predict(image, verbose=False)[0]
    detections = []
    for box in results.boxes:
        detections.append(
            {
                "confidence": float(box.conf[0]),
                "class": int(box.cls[0]),
                "box": [float(v) for v in box.xyxyn[0]],
            }
        )
    return {"detections": detections}


def output_fn(prediction, accept):
    return json.dumps(prediction), "application/json"
'''


def _tar_checkpoint(source_key: str, dest_key: str) -> str:
    """Downloads a bare .pt checkpoint from source_key, tars it up under
    the filename inference.py's model_fn expects (alongside a code/
    inference.py copy -- see _INFERENCE_PY_SOURCE above for why), and
    uploads the result to dest_key. Returns dest_key."""
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = os.path.join(tmp, "model.pt")
        s3.download_file(BUCKET, source_key, checkpoint_path)
        code_dir = os.path.join(tmp, "code")
        os.makedirs(code_dir, exist_ok=True)
        inference_py_path = os.path.join(code_dir, "inference.py")
        with open(inference_py_path, "w", encoding="utf-8") as f:
            f.write(_INFERENCE_PY_SOURCE)
        tarball_path = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(checkpoint_path, arcname="model.pt")
            tar.add(inference_py_path, arcname="code/inference.py")
        s3.upload_file(tarball_path, BUCKET, dest_key)
    return dest_key


def _tarball_has_inference_code(bucket: str, key: str) -> bool:
    """True if the model.tar.gz at bucket/key already bundles
    code/inference.py -- checked before trusting an existing tarball as
    given, since one built outside this app's own repackaging helpers
    (a Model Registry package, or a .tar.gz someone else assembled and
    dropped in S3) can easily be missing it, and exhibits the exact same
    crash-loop as _repackage_registry_model_if_needed's docstring
    describes if it is."""
    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "check.tar.gz")
        s3.download_file(bucket, key, local_path)
        with tarfile.open(local_path, "r:gz") as tar:
            return any(m.name == "code/inference.py" for m in tar.getmembers())


def _repackage_tarball_with_inference_code(source_bucket: str, source_key: str, dest_bucket: str, dest_key: str) -> None:
    """Extracts the top-level model.pt out of the tarball at
    source_bucket/source_key -- dropping anything else in it (training-
    run artifacts, an existing code/ dir that's missing inference.py,
    etc.), since serving only ever needed model.pt -- and re-tars it
    with a fresh code/inference.py alongside, uploaded to
    dest_bucket/dest_key."""
    with tempfile.TemporaryDirectory() as tmp:
        source_path = os.path.join(tmp, "source.tar.gz")
        s3.download_file(source_bucket, source_key, source_path)
        with tarfile.open(source_path, "r:gz") as tar:
            checkpoint = tar.extractfile("model.pt")
            if checkpoint is None:
                raise FileNotFoundError(f"s3://{source_bucket}/{source_key} has no top-level model.pt to repackage.")
            checkpoint_path = os.path.join(tmp, "model.pt")
            with open(checkpoint_path, "wb") as f:
                f.write(checkpoint.read())

        code_dir = os.path.join(tmp, "code")
        os.makedirs(code_dir, exist_ok=True)
        inference_py_path = os.path.join(code_dir, "inference.py")
        with open(inference_py_path, "w", encoding="utf-8") as f:
            f.write(_INFERENCE_PY_SOURCE)
        tarball_path = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(checkpoint_path, arcname="model.pt")
            tar.add(inference_py_path, arcname="code/inference.py")
        s3.upload_file(tarball_path, dest_bucket, dest_key)


def _repackage_registry_model_if_needed(model_data_url: str, size: str, version: str) -> str:
    """A Model Registry package's model.tar.gz is whatever the training
    pipeline produced: a top-level model.pt plus training-run artifacts
    (run/*.jpg, run/weights/*.pt, ...) -- never a code/inference.py.
    Without it, the PyTorch serving container silently falls back to its
    default handler (torch.jit.load, which rejects a plain YOLO
    checkpoint with "Please ensure model is saved using torchscript") and
    every real invocation crash-loops until it times out -- confirmed
    live via the Inference Component's own CloudWatch logs. A no-op
    (returns model_data_url unchanged) if code/inference.py is already
    present, e.g. a training pipeline that's since been fixed to bundle
    it itself."""
    if not model_data_url.startswith("s3://"):
        raise ValueError(f"Not an s3:// URL: {model_data_url!r}")
    src_bucket, _, src_key = model_data_url[len("s3://"):].partition("/")
    if _tarball_has_inference_code(src_bucket, src_key):
        return model_data_url
    dest_key = f"registry-repackaged/{size}/v{version}/model.tar.gz"
    _repackage_tarball_with_inference_code(src_bucket, src_key, BUCKET, dest_key)
    return f"s3://{BUCKET}/{dest_key}"


def repackage_raw_checkpoint_if_needed(weights_s3_prefix: str) -> str:
    """weights_s3_prefix (e.g. "weights/m/2026-09-07") points at a folder
    holding either an already-packaged model.tar.gz or a bare model.pt
    (tarred up under the filename inference.py's model_fn expects). An
    existing model.tar.gz is checked for code/inference.py rather than
    trusted outright -- one assembled outside this app can easily be
    missing it (see _repackage_registry_model_if_needed) -- and, if
    missing, repackaged into a sibling file rather than overwritten in
    place, so the folder's own original tarball is never silently
    replaced. Returns the resulting model.tar.gz's S3 key."""
    prefix = weights_s3_prefix.rstrip("/") + "/"
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    keys = {obj["Key"].rsplit("/", 1)[-1] for obj in listing.get("Contents", [])}

    if "model.tar.gz" in keys:
        existing_key = f"{prefix}model.tar.gz"
        if _tarball_has_inference_code(BUCKET, existing_key):
            return existing_key
        dest_key = f"{prefix}model.repackaged.tar.gz"
        _repackage_tarball_with_inference_code(BUCKET, existing_key, BUCKET, dest_key)
        return dest_key
    if "model.pt" not in keys:
        raise FileNotFoundError(f"s3://{BUCKET}/{prefix} has neither model.tar.gz nor model.pt.")
    return _tar_checkpoint(f"{prefix}model.pt", f"{prefix}model.tar.gz")


def repackage_raw_checkpoint_file(key: str) -> str:
    """Like repackage_raw_checkpoint_if_needed, but for a specific file
    picked directly (any filename, not just a folder assumed to contain
    exactly "model.pt"/"model.tar.gz"). key must end in .pt or .tar.gz.
    The generated tarball (for a bare .pt) is named after the source
    file's own stem, not a fixed "model.tar.gz", so multiple different
    .pt files picked from the same folder over time don't clobber each
    other's repackaged copy. A .tar.gz missing code/inference.py is
    likewise repackaged into a sibling file rather than overwritten in
    place -- same reasoning as repackage_raw_checkpoint_if_needed."""
    if key.endswith(".tar.gz"):
        if _tarball_has_inference_code(BUCKET, key):
            return key
        dest_key = key[: -len(".tar.gz")] + ".repackaged.tar.gz"
        _repackage_tarball_with_inference_code(BUCKET, key, BUCKET, dest_key)
        return dest_key
    if not key.endswith(".pt"):
        raise ValueError(f"{key!r} is not a .pt or .tar.gz file.")
    folder, _, filename = key.rpartition("/")
    stem = filename[: -len(".pt")]
    dest_key = f"{folder}/{stem}.tar.gz" if folder else f"{stem}.tar.gz"
    return _tar_checkpoint(key, dest_key)


def export_weights_to_folder(model_package_arn: str, destination_folder: str) -> tuple[str, str | None]:
    """Downloads/extracts model.pt from a registered package's own
    model.tar.gz and uploads it to
    <destination_folder>/yolo11-<size>-v<version>-<YYYYMMDD>.pt -- self-
    describing so a folder holding several exports stays legible without
    opening each one, unlike the old app's fixed "selected.pt" convention
    (every export overwrote the same name; only ever meaningful as "the
    current selection", not as a record of what was exported and when).
    If that exact name already exists (re-exporting the same package on
    the same day), the existing file is archived first (server-side copy
    to <destination_folder>/archived/<name>_<timestamp>.pt) rather than
    silently clobbered. Returns (new_key, archived_key_or_None)."""
    destination_folder = destination_folder.strip("/")
    if not destination_folder:
        raise ValueError("No destination folder selected.")
    desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    if desc["ModelApprovalStatus"] != "Approved":
        raise RuntimeError(f"{model_package_arn} is {desc['ModelApprovalStatus']!r}, not Approved.")
    model_data_url = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    bucket, _, key = model_data_url[len("s3://"):].partition("/")

    # yolo11{size}-detector/{version} -- the same group-naming convention
    # used everywhere else in this file (REGISTRY_SIZES, model_registry_summary).
    group_name, version = model_package_arn.rsplit("/", 2)[-2:]
    size = group_name.removeprefix("yolo11").removesuffix("-detector")
    export_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_name = f"yolo11-{size}-v{version}-{export_date}.pt"

    new_key = f"{destination_folder}/{file_name}"
    archived_key = None
    try:
        s3.head_object(Bucket=BUCKET, Key=new_key)
        exists = True
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
        exists = False
    if exists:
        archived_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        archived_key = f"{destination_folder}/archived/{file_name.removesuffix('.pt')}_{archived_timestamp}.pt"
        s3.copy_object(Bucket=BUCKET, Key=archived_key, CopySource={"Bucket": BUCKET, "Key": new_key})

    with tempfile.TemporaryDirectory() as tmp:
        tarball_path = os.path.join(tmp, "model.tar.gz")
        s3.download_file(bucket, key, tarball_path)
        extract_dir = os.path.join(tmp, "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        checkpoint_path = os.path.join(extract_dir, "model.pt")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Extracted {model_data_url} but found no model.pt at its root.")
        s3.upload_file(checkpoint_path, BUCKET, new_key)

    return new_key, archived_key


# --- deploy / inference -------------------------------------------------------

def _create_model_tagged(model_name: str, primary_container: dict, source_desc: str, version: str | None = None) -> None:
    """create_model, tagged with the friendly source label (and, for a
    registry deploy, the bare registry version number as its own tag --
    so it can be shown distinctly from the free-text source label, e.g.
    next to the raw Model resource name) so both survive an app restart
    (see _read_model_tags). Tagging on create needs its own IAM action
    (sagemaker:AddTags) beyond CreateModel itself -- if that hasn't been
    granted yet, falls back to creating the model without tags rather
    than failing the whole deploy over a label."""
    tags = [{"Key": TRACK2_SOURCE_TAG_KEY, "Value": source_desc}]
    if version is not None:
        tags.append({"Key": TRACK2_VERSION_TAG_KEY, "Value": version})
    try:
        sm.create_model(
            ModelName=model_name, PrimaryContainer=primary_container, ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
            Tags=tags,
        )
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") != "AccessDeniedException" or "AddTags" not in str(exc):
            raise
        print(f"Deploying {model_name} without a source tag -- sagemaker:AddTags not yet granted: {exc}")
        sm.create_model(
            ModelName=model_name, PrimaryContainer=primary_container, ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
        )


ACTIVE_INFERENCE_COMPONENT_NAME = "track2-yolo-active"


# --- endpoint lifecycle -------------------------------------------------
#
# The endpoint is created on demand and torn down once nothing needs it,
# rather than left standing between runs. A real-time endpoint bills its
# instance for every hour it EXISTS, whether or not anything invokes it,
# and the scale-to-zero policy further down does NOT change that: zeroing
# the Inference Component's copy count frees the compute slot inside the
# instance but leaves the instance itself running. Confirmed against the
# actual bill -- days on which scale-in fired and zero invocations were
# served still billed a full 24 instance-hours. Deleting the endpoint is
# the only thing that stops the meter, so that is what happens once a
# queue drains.

ENDPOINT_READY_TIMEOUT_SECONDS = 1800
ENDPOINT_READY_POLL_SECONDS = 15


def endpoint_status() -> dict | None:
    """The shared endpoint's state, or None when it doesn't exist -- which
    is now its normal resting state between runs, not an error."""
    try:
        desc = sm.describe_endpoint(EndpointName=MULTI_SIZE_ENDPOINT_NAME)
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code == "ResourceNotFound" or (code == "ValidationException" and "Could not find" in str(exc)):
            return None
        if code in ("AccessDeniedException", "AccessDenied", "UnrecognizedClientException"):
            # This runs on every GET /track2/deploy, so a missing
            # DescribeEndpoint grant would otherwise 500 the entire page.
            # Report it as unknown rather than as None: None is rendered as
            # "not running, nothing being billed", and claiming that while
            # unable to look is exactly the wrong way to be wrong about a
            # resource that bills by the hour.
            print(f"Cannot read endpoint state ({code}) -- reporting Unknown: {exc}")
            return {
                "status": "Unknown",
                "instance_type": MULTI_SIZE_INSTANCE_TYPE,
                "instance_count": None,
                "created_at": None,
            }
        raise
    variant = (desc.get("ProductionVariants") or [{}])[0]
    created = desc.get("CreationTime")
    # Used by the idle sweep as the floor for "how long has this been sitting
    # here": an endpoint that was just created or just changed has not been
    # idle, whatever the queues say.
    last_modified = desc.get("LastModifiedTime") or created
    return {
        "status": desc.get("EndpointStatus"),
        "last_modified_at": last_modified.isoformat() if last_modified else None,
        # describe_endpoint doesn't echo the instance type back on the
        # variant (only counts), and fetching the config just for it would
        # be a second call on a hot path -- this is what we create with.
        "instance_type": MULTI_SIZE_INSTANCE_TYPE,
        "instance_count": variant.get("CurrentInstanceCount"),
        "created_at": created.isoformat() if created else None,
    }


def _wait_for_endpoint_gone() -> None:
    deadline = time.monotonic() + ENDPOINT_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if endpoint_status() is None:
            return
        time.sleep(ENDPOINT_READY_POLL_SECONDS)
    raise RuntimeError(f"Timed out waiting for endpoint {MULTI_SIZE_ENDPOINT_NAME} to finish deleting.")


def _wait_for_endpoint_in_service() -> None:
    deadline = time.monotonic() + ENDPOINT_READY_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        state = endpoint_status()
        if state is None:
            raise RuntimeError(f"Endpoint {MULTI_SIZE_ENDPOINT_NAME} disappeared while waiting for it.")
        if state["status"] == "InService":
            return
        if state["status"] in ("Failed", "RollingBack"):
            raise RuntimeError(f"Endpoint {MULTI_SIZE_ENDPOINT_NAME} is {state['status']}, not InService.")
        time.sleep(ENDPOINT_READY_POLL_SECONDS)
    raise RuntimeError(f"Timed out waiting for endpoint {MULTI_SIZE_ENDPOINT_NAME} to become InService.")


def ensure_endpoint_exists() -> bool:
    """Creates the shared Inference-Component-enabled endpoint (and its
    config) if it isn't there, then blocks until it reports InService.
    Idempotent; returns True only if this call actually created it.

    Every deploy runs through here, so a torn-down endpoint comes back by
    itself the moment a queued job needs it. The cost is the creation wait
    (minutes), which is exactly why teardown waits for a queue to drain
    completely instead of happening after each individual job."""
    existing = endpoint_status()
    if existing and existing["status"] == "InService":
        return False

    if existing and existing["status"] == "Deleting":
        # A teardown is still finishing -- from the queue that just
        # drained, or another worker. Waiting for InService here would
        # block until timeout on an endpoint that is on its way out, so
        # wait for it to actually go and then build a fresh one.
        _wait_for_endpoint_gone()
        existing = None

    created = False
    if existing is None:
        if not endpoint_config_exists():
            _create_endpoint_config()
        sm.create_endpoint(
            EndpointName=MULTI_SIZE_ENDPOINT_NAME,
            EndpointConfigName=MULTI_SIZE_ENDPOINT_CONFIG_NAME,
        )
        created = True

    _wait_for_endpoint_in_service()
    return created


def endpoint_config_exists() -> bool:
    """Teardown deletes the endpoint but deliberately keeps its config, so
    on the next spin-up the config is normally already there."""
    try:
        sm.describe_endpoint_config(EndpointConfigName=MULTI_SIZE_ENDPOINT_CONFIG_NAME)
        return True
    except botocore.exceptions.ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("ResourceNotFound", "ValidationException"):
            return False
        raise


def _create_endpoint_config() -> None:
    try:
        sm.create_endpoint_config(
            EndpointConfigName=MULTI_SIZE_ENDPOINT_CONFIG_NAME,
            # Required specifically for IC-enabled configs (a variant with
            # no ModelName) -- see the same call in
            # model_training/src/deploy_inference_component.py.
            ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
            ProductionVariants=[
                {
                    "VariantName": MULTI_SIZE_VARIANT_NAME,
                    "InstanceType": MULTI_SIZE_INSTANCE_TYPE,
                    "InitialInstanceCount": 1,
                }
            ],
        )
    except botocore.exceptions.ClientError as exc:
        # Only reachable by losing a race with something else creating the
        # config between the existence check and here. "already exist"
        # rather than "already exists": the real message is "Cannot create
        # already EXISTING endpoint configuration", which an "already
        # exists" test silently fails to match -- that mismatch is exactly
        # what made the first spin-up after a teardown die instead of
        # reusing the kept config.
        if "already exist" not in str(exc).lower():
            raise


# --- cost reporting -----------------------------------------------------

COST_SERVICE_NAME = "Amazon SageMaker"
COST_DEFAULT_DAYS = 14
COST_MAX_DAYS = 90


COST_CACHE_TTL_SECONDS = 900
_cost_cache: dict[int, tuple[float, dict]] = {}
_cost_cache_lock = threading.Lock()


def sagemaker_cost_breakdown(days: int = COST_DEFAULT_DAYS, *, refresh: bool = False) -> dict:
    """Per-day SageMaker spend for the last `days` days, split by usage
    type (hosting vs training vs Studio vs storage), newest day first.

    Deliberately fetched on demand rather than polled: Cost Explorer bills
    $0.01 per request, so this runs when someone asks for it, not on a
    timer or on every page render.

    Cost Explorer only answers for whole days and its End is exclusive, so
    "today" is partial and usually lags a few hours behind -- the most
    recent day being lower than the one before it is normal, not a drop in
    spend."""
    days = max(1, min(days, COST_MAX_DAYS))

    # Cost Explorer bills $0.01 per request, and the Deploy page shows
    # today's endpoint cost while polling every few seconds -- uncached
    # that is real money for a number that only moves hourly anyway (the
    # data itself lags several hours). Refresh bypasses it, so the Cost
    # tab's button still forces a genuinely fresh read.
    if not refresh:
        with _cost_cache_lock:
            cached = _cost_cache.get(days)
        if cached and (time.time() - cached[0]) < COST_CACHE_TTL_SECONDS:
            return cached[1]

    today = datetime.now(timezone.utc).date()
    # +1 so today itself is included, since End is exclusive.
    end = today + timedelta(days=1)
    start = end - timedelta(days=days + 1)

    response = cost_explorer.get_cost_and_usage(
        TimePeriod={"Start": start.isoformat(), "End": end.isoformat()},
        Granularity="DAILY",
        Metrics=["UnblendedCost"],
        Filter={"Dimensions": {"Key": "SERVICE", "Values": [COST_SERVICE_NAME]}},
        GroupBy=[{"Type": "DIMENSION", "Key": "USAGE_TYPE"}],
    )

    currency = "USD"
    result_days = []
    totals_by_usage_type: dict[str, float] = {}

    for period in response.get("ResultsByTime", []):
        items = []
        day_total = 0.0
        for group in period.get("Groups", []):
            cost = group["Metrics"]["UnblendedCost"]
            amount = float(cost["Amount"])
            currency = cost.get("Unit") or currency
            if amount == 0:
                continue
            usage_type = group["Keys"][0]
            items.append({"usage_type": usage_type, "amount": amount})
            day_total += amount
            totals_by_usage_type[usage_type] = totals_by_usage_type.get(usage_type, 0.0) + amount

        items.sort(key=lambda i: i["amount"], reverse=True)
        result_days.append({
            "date": period["TimePeriod"]["Start"],
            "total": day_total,
            "items": items,
        })

    result_days.reverse()  # newest first -- what you look at first
    by_usage_type = sorted(
        ({"usage_type": k, "amount": v} for k, v in totals_by_usage_type.items()),
        key=lambda i: i["amount"],
        reverse=True,
    )

    result = {
        "currency": currency,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "days": result_days,
        "by_usage_type": by_usage_type,
        "total": sum(d["total"] for d in result_days),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    with _cost_cache_lock:
        _cost_cache[days] = (time.time(), result)
    return result


# Statuses that mean "a worker thread in this process owns this job right
# now". They are only ever true while the process that set them is alive,
# which is what makes them recoverable on startup.
IN_FLIGHT_STATUSES = ("deploying", "running")


def recover_interrupted_jobs() -> list:
    """Fails any job left mid-flight by a process that is no longer here.

    Called once at startup. The workers driving these jobs are daemon
    threads holding their state in memory, so a container restart -- a
    deploy, a crash, an EC2 stop -- leaves the row saying "deploying"
    forever with nothing on its way to finishing it. That is not only a
    wrong history entry: endpoint_teardown_blocker treats such a row as
    unfinished work, so the endpoint can never be torn down again and
    bills indefinitely. One such row kept the endpoint up for over a day.

    Safe to run at startup specifically because nothing else can own these
    rows yet: the worker threads are started by this same process, after
    this runs.
    """
    recovered = []
    now = datetime.now(timezone.utc).isoformat()
    message = "Interrupted -- the app restarted while this job was in progress."
    for item in list_inference_queue():
        if item.get("status") not in IN_FLIGHT_STATUSES:
            continue
        try:
            _update_inference_queue_item(
                item["id"], status="failed", completed_at=now, error=message,
            )
            recovered.append(f"inference:{item['id']}")
        except Exception as exc:
            print(f"Could not recover interrupted inference job {item.get('id')}: {exc}")
    for run in list_eval_runs():
        if run.get("status") not in IN_FLIGHT_STATUSES:
            continue
        try:
            eval_runs_table().update_item(
                Key={"id": run["id"]},
                UpdateExpression="SET #status = :status, #completed_at = :completed_at, #error = :error",
                ExpressionAttributeNames={"#status": "status", "#completed_at": "completed_at", "#error": "error"},
                ExpressionAttributeValues={":status": "failed", ":completed_at": now, ":error": message},
            )
            recovered.append(f"eval:{run['id']}")
        except Exception as exc:
            print(f"Could not recover interrupted eval run {run.get('id')}: {exc}")
    if recovered:
        print(f"Recovered {len(recovered)} interrupted job(s): {', '.join(recovered)}")
    return recovered


# How long the endpoint may sit with nothing needing it before the idle
# sweep removes it. Long enough that someone who spun it up deliberately
# and is about to queue work does not lose it under them; short enough
# that a missed teardown costs cents rather than a day's hosting.
ENDPOINT_IDLE_TEARDOWN_SECONDS = 1800


def _endpoint_idle_seconds(status: dict) -> float | None:
    """Seconds since anything last needed the endpoint, or None if that
    cannot be established (in which case the sweep leaves it alone).

    Measured from the most recent of: the endpoint's own last change, and
    the completion of the last job on either queue. Read from DynamoDB and
    SageMaker rather than process memory, so a restart does not reset the
    clock and hide an endpoint that has been idle for hours.
    """
    latest = status.get("last_modified_at") or status.get("created_at")
    if not latest:
        return None
    marks = [str(latest)]
    for item in list_inference_queue():
        if item.get("completed_at"):
            marks.append(str(item["completed_at"]))
    for run in list_eval_runs():
        if run.get("completed_at"):
            marks.append(str(run["completed_at"]))
    try:
        newest = max(datetime.fromisoformat(str(m).replace("Z", "+00:00")) for m in marks)
    except ValueError as exc:
        print(f"Could not read an idle timestamp: {exc}")
        return None
    if newest.tzinfo is None:
        newest = newest.replace(tzinfo=timezone.utc)
    return (datetime.now(timezone.utc) - newest).total_seconds()


def sweep_idle_endpoint() -> str | None:
    """Removes the endpoint when nothing has needed it for a while.

    The teardown at the end of each queue drain is the normal path; this
    is the safety net for when that never runs -- the process died between
    the last job and the teardown, or the job that raised the endpoint was
    itself interrupted. Without it, a single missed teardown bills until
    someone notices.

    Returns a short description if it deleted the endpoint, else None.
    """
    status = endpoint_status()
    if not status or status.get("status") != "InService":
        return None
    blocker = endpoint_teardown_blocker()
    if blocker:
        return None
    idle = _endpoint_idle_seconds(status)
    if idle is None or idle < ENDPOINT_IDLE_TEARDOWN_SECONDS:
        return None
    delete_endpoint()
    minutes = int(idle // 60)
    print(f"Idle sweep: deleted the endpoint after {minutes} minutes with nothing needing it.")
    return f"deleted after {minutes} minutes idle"


def endpoint_teardown_blocker() -> str | None:
    """Why the endpoint must stay up right now, or None if it's safe to
    delete. Both queues share the one endpoint, so either one still having
    work is enough to keep it."""
    if is_running_inference():
        return "an inference call is still in flight"
    if _inference_queue_started or _eval_queue_started:
        return "a queue is still running"
    for item in list_inference_queue():
        if item["status"] in ("queued", "deploying", "running"):
            return "the inference queue still has unfinished jobs"
    for run in list_eval_runs():
        if run["status"] in ("queued", "deploying", "running"):
            return "the eval queue still has unfinished runs"
    return None


INFERENCE_COMPONENT_DELETE_TIMEOUT_SECONDS = 600
INFERENCE_COMPONENT_DELETE_POLL_SECONDS = 10


def delete_active_inference_component() -> bool:
    """Deletes the active Inference Component and blocks until it is gone.
    Returns False if there wasn't one.

    Required before deleting the endpoint: SageMaker rejects DeleteEndpoint
    outright while a component is still attached ("Cannot delete endpoint
    with inference component associated"), and the deletion is async, so
    firing it and immediately calling DeleteEndpoint loses the same race."""
    try:
        sm.delete_inference_component(InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME)
    except sm.exceptions.ResourceNotFound:
        return False
    except botocore.exceptions.ClientError as exc:
        message = str(exc)
        if exc.response.get("Error", {}).get("Code") != "ValidationException":
            raise
        if "Could not find" in message:
            return False
        if "DELETE_IN_PROGRESS" not in message:
            raise
        # Someone got here first -- a queue draining concurrently, or a
        # previous teardown still finishing. Falling through to the wait
        # below is the right answer either way: the caller only needs the
        # component gone before it deletes the endpoint, not to have been
        # the one that removed it.
        print(f"{ACTIVE_INFERENCE_COMPONENT_NAME} already deleting -- waiting for it.")

    deadline = time.monotonic() + INFERENCE_COMPONENT_DELETE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if describe_deployed_inference_component() is None:
            return True
        time.sleep(INFERENCE_COMPONENT_DELETE_POLL_SECONDS)
    raise RuntimeError(
        f"Timed out waiting for inference component {ACTIVE_INFERENCE_COMPONENT_NAME} to delete."
    )


def delete_endpoint(*, force: bool = False) -> dict:
    """Deletes the endpoint, stopping the instance-hour billing. Keeps the
    endpoint config so ensure_endpoint_exists can rebuild from it.

    Refuses while either queue still has work unless force=True, since
    deleting mid-run kills every in-flight invocation."""
    if not force:
        blocker = endpoint_teardown_blocker()
        if blocker:
            return {"deleted": False, "reason": blocker}

    if endpoint_status() is None:
        return {"deleted": False, "reason": "the endpoint doesn't exist"}

    # Has to go first, and has to have finished -- see the helper.
    delete_active_inference_component()
    sm.delete_endpoint(EndpointName=MULTI_SIZE_ENDPOINT_NAME)
    return {"deleted": True, "reason": None}


def delete_endpoint_if_idle() -> None:
    """Best-effort teardown hook for the end of a worker loop -- a failure
    here must never surface as a failed job, since the work itself already
    finished. Worst case the endpoint stays up and the next run reuses it."""
    try:
        result = delete_endpoint()
        if result["deleted"]:
            print(f"Queue drained -- deleted endpoint {MULTI_SIZE_ENDPOINT_NAME} to stop instance billing.")
        else:
            print(f"Left endpoint {MULTI_SIZE_ENDPOINT_NAME} up: {result['reason']}.")
    except Exception as exc:
        print(f"Could not tear down endpoint {MULTI_SIZE_ENDPOINT_NAME}: {exc}")


def deploy_inference_component(
    size: str, *,
    model_package_arn: str | None = None,
    weights_s3_prefix: str | None = None,
    weights_s3_key: str | None = None,
) -> str:
    """Creates a Model from whichever source was given and creates/updates
    the ONE shared active Inference Component to point at it -- there is
    deliberately only ever one model live at a time (by explicit product
    decision: 4 independent always-on slots read as confusing next to a
    single "what's live right now" answer), so deploying any size replaces
    whatever was there before, regardless of what size that was.
    weights_s3_prefix (a folder assumed to hold model.pt/model.tar.gz) and
    weights_s3_key (an exact .pt/.tar.gz file, any name) are alternatives
    -- exactly one of model_package_arn/weights_s3_prefix/weights_s3_key
    should be given. Returns a short description of what was deployed."""
    # The endpoint is torn down whenever no queue needs it, so it may well
    # not exist right now -- bring it back first and wait for InService,
    # since an Inference Component can't be created against an endpoint
    # that isn't up. No-op (one describe) in the common case where a
    # previous job in the same queue run already brought it up.
    ensure_endpoint_exists()

    # Suspend scale-IN for the duration of this deploy, before touching
    # anything else -- confirmed live: a queued job's own deploy call
    # failed with "Cannot update in-progress inference component" because
    # the scale-to-zero policy's own scale-in mutation (30 idle minutes
    # elapsing while a cold start with zero invocations was still
    # underway) collided with it. Resumed by a background thread below
    # once this specific deploy actually settles, not here -- a fresh
    # deploy can legitimately take up to ~10 minutes with no traffic yet,
    # which is exactly the situation scale-in would otherwise "correctly"
    # act on.
    #
    # Only when a component already exists, though. The scalable target is
    # registered AGAINST the component, so on the first deploy after a
    # teardown -- when the component was deleted along with the endpoint --
    # there is nothing to register and RegisterScalableTarget fails the
    # whole job with "Could not find inference component". There is also
    # nothing to suspend in that case: a component that does not exist yet
    # has no scale-in to race with. The target gets registered after the
    # deploy instead, by _resume_scale_in_once_ready below.
    if describe_deployed_inference_component() is not None:
        ensure_scale_to_zero(suspend_scale_in=True)

    suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"track2-yolo-{size}-model-{suffix}"

    if model_package_arn:
        desc = sm.describe_model_package(ModelPackageName=model_package_arn)
        if desc["ModelApprovalStatus"] != "Approved":
            raise RuntimeError(f"{model_package_arn} is {desc['ModelApprovalStatus']!r}, not Approved.")
        # PrimaryContainer={"ModelPackageName": ...} is rejected on this
        # IC-enabled endpoint -- confirmed live: "Model package cannot mix
        # with SageMaker or bring-your-own images, and only one model
        # package should be used for endpoints with Inference Components
        # enabled." Resolving Image/ModelDataUrl directly sidesteps it.
        container = desc["InferenceSpecification"]["Containers"][0]
        version = model_package_arn.rsplit("/", 1)[-1]
        source_desc = f"YOLO11-{size.upper()} v{version}"
        model_data_url = _repackage_registry_model_if_needed(container["ModelDataUrl"], size, version)
        _create_model_tagged(
            model_name, {"Image": container["Image"], "ModelDataUrl": model_data_url}, source_desc, version,
        )
    elif weights_s3_key:
        artifact_key = repackage_raw_checkpoint_file(weights_s3_key)
        source_desc = f"S3: {weights_s3_key}"
        _create_model_tagged(
            model_name, {"Image": SERVING_IMAGE_URI, "ModelDataUrl": f"s3://{BUCKET}/{artifact_key}"}, source_desc,
        )
    else:
        artifact_key = repackage_raw_checkpoint_if_needed(weights_s3_prefix)
        source_desc = f"S3: {weights_s3_prefix}"
        _create_model_tagged(
            model_name, {"Image": SERVING_IMAGE_URI, "ModelDataUrl": f"s3://{BUCKET}/{artifact_key}"}, source_desc,
        )

    specification = {
        "ModelName": model_name,
        "ComputeResourceRequirements": COMPUTE_REQUIREMENTS[size],
        "StartupParameters": {
            "ModelDataDownloadTimeoutInSeconds": 600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        },
    }

    try:
        sm.describe_inference_component(InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME)
        exists = True
    except sm.exceptions.ResourceNotFound:
        exists = False
    except botocore.exceptions.ClientError as exc:
        # Confirmed live: a not-found InferenceComponentName raises
        # ValidationException ("Could not find inference component ..."),
        # not ResourceNotFound.
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find" in str(exc):
            exists = False
        else:
            raise

    if exists:
        # RuntimeConfig=CopyCount:1 here too, not just Specification --
        # otherwise a deploy that lands on a component already scaled to
        # zero (idle 30+ min before) loads the new model but leaves it at
        # 0 running copies, which then makes wait_for_inference_component_ready
        # (used by both the eval and inference queues right after
        # deploying) time out waiting for a copy count that was never
        # going to rise on its own.
        sm.update_inference_component(
            InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME,
            Specification=specification,
            RuntimeConfig={"CopyCount": 1},
        )
    else:
        sm.create_inference_component(
            EndpointName=MULTI_SIZE_ENDPOINT_NAME,
            VariantName=MULTI_SIZE_VARIANT_NAME,
            InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME,
            Specification=specification,
            RuntimeConfig={"CopyCount": 1},
        )

    # Resuming happens once THIS deploy actually settles (or gives up
    # waiting), not immediately -- on its own daemon thread, so
    # deploy_inference_component itself stays exactly as fast as it's
    # always been for the "just click Deploy" path, which doesn't
    # otherwise wait for InService at all.
    threading.Thread(target=_resume_scale_in_once_ready, daemon=True).start()

    return f"{size} <- {source_desc}"


def _resume_scale_in_once_ready() -> None:
    try:
        wait_for_inference_component_ready()
    except Exception as exc:
        print(f"Resuming scale-in after a deploy that never became ready: {exc}")
    finally:
        ensure_scale_to_zero(suspend_scale_in=False)


SCALE_IN_COOLDOWN_SECONDS = 1800  # 30 minutes idle before scaling in to zero copies


def ensure_scale_to_zero(*, suspend_scale_in: bool = False) -> None:
    """Registers (or, called again, just re-applies) a scale-to-zero
    policy on the active Inference Component: 0-1 copies, scaling in to 0
    after SCALE_IN_COOLDOWN_SECONDS with no invocations. Idempotent --
    both calls are upserts by resource id/policy name, so calling this on
    every deploy is exactly as safe as calling it once. Scaling back OUT
    from zero on a live request isn't configured here (that needs a
    second CloudWatch-alarm-driven policy this app doesn't set up) --
    queuing a deploy is the manual equivalent, since every deploy
    explicitly requests CopyCount: 1 regardless of the count beforehand.
    suspend_scale_in=True additionally suspends dynamic scale-IN actions
    on this target (still fully registered -- just paused) -- see
    deploy_inference_component's own use of this for why."""
    resource_id = f"inference-component/{ACTIVE_INFERENCE_COMPONENT_NAME}"
    try:
        autoscaling.register_scalable_target(
            ServiceNamespace="sagemaker",
            ResourceId=resource_id,
            ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
            MinCapacity=0,
            MaxCapacity=1,
            SuspendedState={"DynamicScalingInSuspended": suspend_scale_in},
        )
    except botocore.exceptions.ClientError as exc:
        # The component can legitimately be gone by the time this runs:
        # _resume_scale_in_once_ready calls it minutes later, on its own
        # thread, and a teardown in between deletes the component. Nothing
        # to scale then, and failing here would surface as a failed deploy
        # for work that had already finished.
        if "Could not find inference component" in str(exc):
            print(f"No inference component to autoscale ({resource_id}) -- skipping.")
            return
        raise
    autoscaling.put_scaling_policy(
        PolicyName="track2-yolo-active-scale-to-zero",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:inference-component:DesiredCopyCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": 5.0,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerInferenceComponentInvocationsPerCopy",
            },
            "ScaleInCooldown": SCALE_IN_COOLDOWN_SECONDS,
            "ScaleOutCooldown": 60,
        },
    )



def _read_model_tags(model_name: str | None) -> dict[str, str]:
    """Reads the tags set at deploy time (see _create_model_tagged) back
    off the SageMaker Model resource itself -- rather than tracking them
    in an in-memory dict, which wouldn't survive an app restart."""
    if not model_name:
        return {}
    try:
        model_arn = sm.describe_model(ModelName=model_name)["ModelArn"]
        tags = sm.list_tags(ResourceArn=model_arn).get("Tags", [])
    except botocore.exceptions.ClientError:
        return {}
    return {t["Key"]: t["Value"] for t in tags}


_MODEL_NAME_SIZE_RE = re.compile(r"^track2-yolo-(s|m|l|x)-model-")


def _size_from_model_name(model_name: str | None) -> str | None:
    """The active Model resource's own name still encodes which size was
    deployed (track2-yolo-<size>-model-<timestamp>) even though the
    Inference Component itself is now size-agnostically named -- reusing
    that instead of tracking size separately anywhere."""
    if not model_name:
        return None
    match = _MODEL_NAME_SIZE_RE.match(model_name)
    return match.group(1) if match else None


def describe_deployed_inference_component() -> dict | None:
    """The one active Inference Component, or None if nothing's ever been
    deployed."""
    try:
        desc = sm.describe_inference_component(InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME)
    except sm.exceptions.ResourceNotFound:
        return None
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find" in str(exc):
            return None
        raise
    spec = desc.get("Specification", {})
    runtime_config = desc.get("RuntimeConfig", {})
    model_name = spec.get("ModelName")
    tags = _read_model_tags(model_name)
    return {
        "active_size": _size_from_model_name(model_name),
        "status": desc.get("InferenceComponentStatus"),
        "model_name": model_name,
        "last_modified": desc.get("LastModifiedTime").isoformat() if desc.get("LastModifiedTime") else None,
        "source": tags.get(TRACK2_SOURCE_TAG_KEY),
        # Bare registry version number ("12"), None for an S3-sourced
        # deploy (there's no registry version to point at).
        "version": tags.get(TRACK2_VERSION_TAG_KEY),
        "current_copy_count": runtime_config.get("CurrentCopyCount"),
        "desired_copy_count": runtime_config.get("DesiredCopyCount"),
        "running_inference": is_running_inference(),
    }


# In-flight invoke_inference calls -- lets the status panel show "running
# inference" for real, rather than just deploy status. Only reflects
# calls made through this app (the eval queue's own inference loop being
# the main source now that ad-hoc single-image inference was removed),
# not any other traffic hitting the endpoint from elsewhere. A single
# counter, not one per size, since there's only one active model at a
# time to be running inference on.
_inference_in_flight_lock = threading.Lock()
_inference_in_flight = 0


def is_running_inference() -> bool:
    with _inference_in_flight_lock:
        return _inference_in_flight > 0


def invoke_inference(image_bytes: bytes) -> dict:
    """Real-time prediction from whatever's currently deployed to the
    active Inference Component. Returns {"detections": [{"confidence",
    "class", "box": [x1,y1,x2,y2]}]} with box coordinates already
    normalized 0.0-1.0. Raises on any failure (nothing deployed yet,
    cold-starting endpoint, malformed image)."""
    global _inference_in_flight
    with _inference_in_flight_lock:
        _inference_in_flight += 1
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=MULTI_SIZE_ENDPOINT_NAME,
            InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME,
            ContentType="image/jpeg",
            Body=image_bytes,
        )
        return json.loads(response["Body"].read())
    finally:
        with _inference_in_flight_lock:
            _inference_in_flight -= 1


def _write_lineage_file(
    images_folder: str, output_folder: str, files: list[str], job: dict, inference_started_at: str, inference_completed_at: str,
) -> None:
    """One lineage.json per batch run, written into output_folder -- a
    self-contained record of exactly what produced the results sitting
    next to it (which files went in, which model/version ran them, where
    both live in S3, and when the run itself started/ended), so the
    folder is legible on its own without cross-referencing the in-memory
    (restart-losable) queue history. inference_started_at/
    inference_completed_at bound just the image loop below, not the
    deploy-and-wait that precedes it in _run_inference_queue_job -- the
    queue item's own started_at/completed_at cover that wider span."""
    version = (
        job["model_package_arn"].rsplit("/", 1)[-1]
        if job.get("weights_kind") == "registry" and job.get("model_package_arn")
        else None
    )
    lineage = {
        "bucket": BUCKET,
        "images_folder": images_folder,
        "output_folder": output_folder,
        "files": files,
        "model": {
            "size": job.get("size"),
            "weights_kind": job.get("weights_kind"),
            "weights_label": job.get("weights_label"),
            "version": version,
            "model_package_arn": job.get("model_package_arn"),
            "weights_folder": job.get("weights_folder"),
            "weights_key": job.get("weights_key"),
            "notes": job.get("notes"),
        },
        "inference_started_at": inference_started_at,
        "inference_completed_at": inference_completed_at,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    body = json.dumps(lineage, indent=2)
    s3.put_object(Bucket=BUCKET, Key=f"{output_folder}/lineage.json", Body=body.encode("utf-8"), ContentType="application/json")

    # Also recorded in DynamoDB, keyed by the run's own id. The S3 copy
    # travels with the results and stays right if they are moved; the
    # table is what makes lineage answerable across runs without listing
    # folders, and survives the output being deleted. Best-effort: the
    # inference work is already done and written by this point, so a
    # bookkeeping failure must not turn a finished run into a failed one.
    try:
        record = {"id": job["id"], **lineage}
        inference_lineage_table().put_item(Item=record)
    except Exception as exc:
        print(f"Could not record lineage for run {job.get('id')}: {exc}")


INFERENCE_LINEAGE_TABLE_NAME = "track2-yolo-inference-lineage"


def inference_lineage_table():
    return _dynamodb().Table(INFERENCE_LINEAGE_TABLE_NAME)


def read_run_lineage(run_id: str | None, output_folder: str) -> dict | None:
    """What produced one run's results: DynamoDB first, keyed by the run's
    own id, falling back to the lineage.json in the output folder.

    Both are written, and the fallback is what keeps runs from before the
    table existed readable -- and covers output produced outside the queue
    entirely, which has a folder but no run id."""
    if run_id:
        try:
            item = inference_lineage_table().get_item(Key={"id": run_id}).get("Item")
            if item:
                return dict(item)
        except Exception as exc:
            # Fall through to S3 rather than fail: the folder's own copy is
            # just as authoritative, and a table problem should degrade to
            # the slower answer, not to no answer.
            print(f"Lineage lookup failed for run {run_id}, falling back to S3: {exc}")

    key = f"{output_folder.strip('/')}/lineage.json"
    try:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except s3.exceptions.NoSuchKey:
        return None
    except botocore.exceptions.ClientError as exc:
        # Only "it isn't there" is a None. An AccessDenied answered the
        # same way would read as "this run has no lineage", which is a
        # different and much more misleading statement.
        if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return None
        raise
    return json.loads(body)


def run_batch_inference(images_folder: str, output_folder: str, *, job: dict | None = None) -> dict:
    """Runs the currently deployed model against every image directly
    under images_folder, writing each image's detections to
    <output_folder>/<image_stem>.result.json, plus one lineage.json (see
    _write_lineage_file) if job (the originating queue item) is given.
    Synchronous (unlike the eval queue's own image loop) -- meant for the
    kind of folder a person picks by hand to sanity-check a deploy, not a
    dataset-sized batch; a per-image failure is recorded in "errors"
    rather than aborting the rest of the folder."""
    images_folder = images_folder.strip("/")
    output_folder = output_folder.strip("/")
    if not images_folder:
        raise ValueError("No image folder selected.")
    if not output_folder:
        raise ValueError("No output folder selected.")
    image_names = list_s3_images(images_folder)
    if not image_names:
        raise ValueError(f"No images found under s3://{BUCKET}/{images_folder}/")

    inference_started_at = datetime.now(timezone.utc).isoformat()
    processed = 0
    errors = []
    for name in image_names:
        image_key = f"{images_folder}/{name}"
        try:
            image_bytes = s3.get_object(Bucket=BUCKET, Key=image_key)["Body"].read()
            result = invoke_inference(image_bytes)
            stem = name.rsplit(".", 1)[0] if "." in name else name
            output_key = f"{output_folder}/{stem}.result.json"
            body = json.dumps({
                "image_key": image_key,
                "detections": result.get("detections", []),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }, indent=2)
            s3.put_object(Bucket=BUCKET, Key=output_key, Body=body.encode("utf-8"), ContentType="application/json")
            processed += 1
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    inference_completed_at = datetime.now(timezone.utc).isoformat()

    if job is not None:
        _write_lineage_file(images_folder, output_folder, image_names, job, inference_started_at, inference_completed_at)

    return {"processed": processed, "total": len(image_names), "output_folder": output_folder, "errors": errors}


# --- training progress / GPU status -------------------------------------------

def _get_metric_max(training_job_name: str, metric_name: str) -> int | None:
    """Latest Maximum datapoint for one CloudWatch metric under the
    standard training-job namespace, or None if nothing has landed yet."""
    try:
        now = datetime.now(timezone.utc)
        response = cloudwatch.get_metric_statistics(
            Namespace="/aws/sagemaker/TrainingJobs",
            MetricName=metric_name,
            Dimensions=[{"Name": "TrainingJobName", "Value": training_job_name}],
            StartTime=now - timedelta(hours=6),
            EndTime=now,
            Period=60,
            Statistics=["Maximum"],
        )
        datapoints = response.get("Datapoints", [])
        if not datapoints:
            return None
        return int(max(dp["Maximum"] for dp in datapoints))
    except Exception as exc:
        print(f"Could not get {metric_name} metric for {training_job_name}: {exc}")
        return None


def get_epoch_progress(training_job_name: str) -> dict | None:
    """{"current", "total"} or None if the job's own "epochs"
    hyperparameter isn't available yet or no train:epoch datapoint has
    landed yet (CloudWatch metrics lag the actual log line by roughly a
    minute)."""
    try:
        desc = sm.describe_training_job(TrainingJobName=training_job_name)
        total_epochs = int(desc["HyperParameters"].get("epochs", 0))
    except Exception as exc:
        print(f"Could not describe training job {training_job_name}: {exc}")
        return None
    if not total_epochs:
        return None
    current_epoch = _get_metric_max(training_job_name, "train:epoch") or 0
    return {"current": current_epoch, "total": total_epochs}


def get_dataset_images_progress(training_job_name: str) -> dict | None:
    """{"current", "total"} of images downloaded so far, or None if the
    total hasn't landed yet."""
    total_images = _get_metric_max(training_job_name, "dataset:total_images")
    if not total_images:
        return None
    current = _get_metric_max(training_job_name, "dataset:images_downloaded") or 0
    return {"current": current, "total": total_images}


def get_training_progress() -> dict | None:
    """None if no execution has ever run. Otherwise the latest execution's
    overall status, each of its 3 steps' status, and (only while TrainYolo
    is Executing, or right after it succeeds) the live epoch/download
    sub-progress."""
    try:
        executions = sm.list_pipeline_executions(
            PipelineName=PIPELINE_NAME, SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
        )["PipelineExecutionSummaries"]
    except Exception as exc:
        print(f"Could not list pipeline executions: {exc}")
        return None
    if not executions:
        return None
    execution = executions[0]
    execution_arn = execution["PipelineExecutionArn"]

    step_status = {"TrainYolo": "NotStarted", "EvaluateYolo": "NotStarted", "RegisterYoloModel": "NotStarted"}
    training_job_name = None
    try:
        steps = sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)["PipelineExecutionSteps"]
        for step in steps:
            name = step["StepName"]
            # ModelStep (RegisterYoloModel) expands into a real step named
            # "RegisterYoloModel-RegisterModel", not the plain name --
            # confirmed live, prefix match tolerates this.
            for key in step_status:
                if name == key or name.startswith(key + "-"):
                    step_status[key] = step.get("StepStatus", "Unknown")
                    break
            if name == "TrainYolo":
                training_job_arn = step.get("Metadata", {}).get("TrainingJob", {}).get("Arn")
                if training_job_arn:
                    training_job_name = training_job_arn.rsplit("/", 1)[-1]
    except Exception as exc:
        print(f"Could not list pipeline execution steps: {exc}")

    epoch_progress = None
    download_progress = None
    # Keep showing the epoch bar through Evaluate/Register too, not just
    # while TrainYolo itself is Executing -- both CloudWatch metrics stay
    # queryable long after the training job finishes.
    if training_job_name and step_status["TrainYolo"] in ("Executing", "Succeeded"):
        epoch_progress = get_epoch_progress(training_job_name)
        if step_status["TrainYolo"] == "Executing" and not epoch_progress:
            download_progress = get_dataset_images_progress(training_job_name)

    start_time = execution.get("StartTime")
    return {
        "execution_arn": execution_arn,
        "status": execution["PipelineExecutionStatus"],
        "start_time": start_time.isoformat() if start_time else None,
        "steps": step_status,
        "epoch_progress": epoch_progress,
        "download_progress": download_progress,
    }


def get_gpu_resource_usage() -> list[dict]:
    """Every InProgress training job in this AWS account (not just this
    app's own pipeline) running on GPU_INSTANCE_TYPE -- the account has
    exactly 1 such instance available in total, shared across every
    consumer in the account. Each returned dict: {"name", "start_time",
    "own_pipeline"} -- own_pipeline is True if the job name matches this
    app's own "pipelines-<execution-id>-TrainYolo-..." naming, so the UI
    can tell "busy with your own run" apart from "busy with something
    else"."""
    try:
        summaries = sm.list_training_jobs(
            StatusEquals="InProgress", SortBy="CreationTime", SortOrder="Descending", MaxResults=20,
        )["TrainingJobSummaries"]
    except Exception as exc:
        print(f"Could not list training jobs: {exc}")
        return []
    jobs = []
    for s in summaries:
        name = s["TrainingJobName"]
        try:
            instance_type = sm.describe_training_job(TrainingJobName=name)["ResourceConfig"]["InstanceType"]
        except Exception as exc:
            print(f"Could not describe training job {name}: {exc}")
            continue
        if instance_type == GPU_INSTANCE_TYPE:
            start_time = s.get("TrainingStartTime")
            jobs.append({
                "name": name,
                "start_time": start_time.isoformat() if start_time else None,
                "own_pipeline": name.startswith("pipelines-") and "-TrainYolo-" in name,
            })
    return jobs


def training_already_running_reason(progress: dict | None) -> str | None:
    """None if it's safe to start a new run, otherwise a human-readable
    reason not to. The account has exactly 1 ml.g4dn.xlarge training
    instance available -- starting a second pipeline execution while
    TrainYolo is still Executing on another one doesn't queue, it fails
    outright with SAGEMAKER_RESOURCE_LIMIT and no retry."""
    if progress and progress["status"] == "Executing":
        run_id = progress["execution_arn"].rsplit("/", 1)[-1]
        return f"A training run ({run_id}) is already in progress. Wait for it to finish before starting another."
    with _pending_submission_lock:
        if _pending_submission:
            return "A training run is still being submitted. Wait for it to start before starting another."
    gpu_jobs = [j for j in get_gpu_resource_usage() if not j["own_pipeline"]]
    if gpu_jobs:
        return (
            f"The account's only {GPU_INSTANCE_TYPE} GPU slot is in use by a job outside this app "
            f"({gpu_jobs[0]['name']}) -- starting a new run now would fail. Wait for it to finish."
        )
    return None


def _s3_object_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        return True
    except botocore.exceptions.ClientError:
        return False


def validate_dataset_folder(folder: str, *, used_for_evaluation: bool) -> None:
    """Checks a chosen folder carries what the pipeline will look for.

    Training and evaluation want different things, and the difference is
    not obvious from the folder: train.py reads data.yaml and the
    images/labels beside it, while evaluate.py additionally reads
    dataset-card.json, which only the replay-derived datasets carry.

    Checked here because the alternative is finding out ~15 minutes in,
    after a GPU training job has already succeeded, when the evaluation
    step dies on a missing file and takes the whole run's model with it.
    """
    prefix = folder.strip("/")
    if not _s3_object_exists(f"{prefix}/data.yaml"):
        raise ValueError(
            f"'{folder}' has no data.yaml, so it is not a YOLO dataset folder. "
            "Pick the folder that contains data.yaml, images/ and labels/."
        )
    if used_for_evaluation and not _s3_object_exists(f"{prefix}/dataset-card.json"):
        raise ValueError(
            f"'{folder}' has no dataset-card.json, which the evaluation step reads. "
            "Training would succeed and evaluation would then fail, losing the run. "
            "Choose a prepared dataset (the replay datasets under "
            "replay/fullframe/datasets/ carry one) as the evaluation folder."
        )


def validate_training_params(epochs, dataset_folder, img_size=640, batch=16, seed=42, eval_dataset_folder="") -> tuple[int, int, int, int]:
    """Shared by the immediate Start Training path and Add to Queue -- both
    need the exact same checks so a bad entry doesn't sit in the queue for
    potentially hours before failing when it's finally its turn to run."""
    if not dataset_folder:
        raise ValueError("No dataset folder selected -- use the browser to pick one first.")
    # Whichever folder evaluation will read is the one that needs the card:
    # with no separate evaluation folder chosen, that is the training one.
    validate_dataset_folder(dataset_folder, used_for_evaluation=not eval_dataset_folder)
    if eval_dataset_folder:
        validate_dataset_folder(eval_dataset_folder, used_for_evaluation=True)
    epochs_int = int(epochs)
    if not (1 <= epochs_int <= 1000):
        raise ValueError(f"Epochs must be between 1 and 1000, got {epochs_int}.")
    img_size_int = int(img_size)
    if not (32 <= img_size_int <= 4096):
        raise ValueError(f"Image size must be between 32 and 4096, got {img_size_int}.")
    batch_int = int(batch)
    if not (1 <= batch_int <= 512):
        raise ValueError(f"Batch size must be between 1 and 512, got {batch_int}.")
    seed_int = int(seed)
    return epochs_int, img_size_int, batch_int, seed_int


def start_training_run(generation, size, epochs, dataset_folder, base_model_folder, notes, img_size=640, batch=16, seed=42, eval_dataset_folder="", source="manual") -> str:
    """Validates and submits one training run via CodeBuild -- the single
    place that builds environmentVariablesOverride and calls
    codebuild.start_build, so the immediate "Start Training" action and
    the training queue's poll loop both go through here. Raises on
    validation failure or a CodeBuild error; returns the new build's id on
    success. Caller is responsible for the already-running (1 GPU slot)
    guard -- see training_already_running_reason."""
    epochs_int, img_size_int, batch_int, seed_int = validate_training_params(epochs, dataset_folder, img_size, batch, seed, eval_dataset_folder)
    env_overrides = [
        {"name": "TRAIN_GENERATION", "value": generation, "type": "PLAINTEXT"},
        {"name": "TRAIN_SIZE", "value": size, "type": "PLAINTEXT"},
        {"name": "TRAIN_EPOCHS", "value": str(epochs_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_IMG_SIZE", "value": str(img_size_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_BATCH", "value": str(batch_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_SEED", "value": str(seed_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_DATA_S3_URI", "value": f"s3://{BUCKET}/{dataset_folder}/", "type": "PLAINTEXT"},
    ]
    # Only sent when a separate evaluation set was chosen. Unset, the
    # pipeline evaluates on the training data, which is its existing
    # behaviour -- sending the training URI here would look like a
    # deliberate choice in the build's environment when it isn't one.
    if eval_dataset_folder:
        env_overrides.append(
            {"name": "EVAL_DATA_S3_URI", "value": f"s3://{BUCKET}/{eval_dataset_folder}/", "type": "PLAINTEXT"}
        )
    if base_model_folder:
        base_model_s3_uri = resolve_base_model_s3_uri(base_model_folder)
        env_overrides.append({"name": "BASE_MODEL_S3_URI", "value": base_model_s3_uri, "type": "PLAINTEXT"})
    if notes:
        env_overrides.append({"name": "TRAIN_NOTES", "value": notes, "type": "PLAINTEXT"})
    response = codebuild.start_build(
        projectName=RUN_PIPELINE_CODEBUILD_PROJECT,
        environmentVariablesOverride=env_overrides,
    )
    build_id = response["build"]["id"]
    global _pending_submission
    with _pending_submission_lock:
        _pending_submission = {"build_id": build_id, "triggered_at": datetime.now(timezone.utc)}
    # After the build is actually submitted, so a run that never started
    # leaves no audit row claiming it did. Listing the dataset can take a
    # moment on a large folder; the build is already on its way by then.
    write_training_audit(
        build_id=build_id, generation=generation, size=size, epochs=epochs_int,
        img_size=img_size_int, batch=batch_int, seed=seed_int,
        dataset_folder=dataset_folder, eval_dataset_folder=eval_dataset_folder,
        base_model_folder=base_model_folder, notes=notes, source=source,
    )
    return build_id


CUSTOM_PIPELINE_NAME = os.environ.get("CUSTOM_TRAINING_PIPELINE_NAME", "track2-yolo-custom-training-pipeline")
CUSTOM_REGISTRY_GROUP = "track2-yolo-custom-detector"


def start_custom_training_run(
    *, dataset_folder, eval_dataset_folder, base_model_uri, epochs, img_size, batch, seed, notes,
) -> dict:
    """Trains on a dataset's own classes, via the separate custom pipeline.

    Started directly rather than through CodeBuild and the replay
    coordinator: none of that path applies here. There is no snapshot to
    pin, no six-to-two class remap to perform, and no production baseline
    to supersede -- the dataset is simply the one that was chosen. A
    useful side effect is that the execution ARN is known at submit time,
    so the audit row links to it exactly instead of matching on time.

    Deliberately a different pipeline and registry group from production:
    a model whose classes are not person/helmet must not be able to reach
    the replay lineage, whose consumers read index 0 as person.
    """
    epochs_int, img_size_int, batch_int, seed_int = validate_training_params(
        epochs, dataset_folder, img_size, batch, seed, eval_dataset_folder,
    )
    if not base_model_uri:
        raise ValueError(
            "A base model is required -- the trainer fine-tunes from a checkpoint rather than "
            "training from scratch."
        )
    run_id = "custom-" + uuid.uuid4().hex[:24]
    run_root = f"s3://{BUCKET}/custom/runs/{run_id}"
    train_uri = f"s3://{BUCKET}/{dataset_folder.strip('/')}/"
    eval_uri = f"s3://{BUCKET}/{eval_dataset_folder.strip('/')}/" if eval_dataset_folder else train_uri
    parameters = {
        "TrainingDatasetUri": train_uri,
        "EvaluationDatasetUri": eval_uri,
        "EvaluationOutputUri": f"{run_root}/evaluation",
        "RunMetadataUri": f"{run_root}/request.json",
        "RunId": run_id,
        "DatasetVersion": run_id,
        "BaseModelUri": base_model_uri,
        "Epochs": str(epochs_int),
        "ImgSize": str(img_size_int),
        "Batch": str(batch_int),
        "Seed": str(seed_int),
    }
    response = sm.start_pipeline_execution(
        PipelineName=CUSTOM_PIPELINE_NAME,
        ClientRequestToken=run_id,
        PipelineExecutionDisplayName=run_id,
        PipelineParameters=[{"Name": k, "Value": v} for k, v in parameters.items()],
    )
    execution_arn = response["PipelineExecutionArn"]
    audit = write_training_audit(
        build_id="", generation="custom", size="any", epochs=epochs_int,
        img_size=img_size_int, batch=batch_int, seed=seed_int,
        dataset_folder=dataset_folder, eval_dataset_folder=eval_dataset_folder,
        base_model_folder="", notes=notes, source="custom",
    )
    # Linked immediately, which the replay path cannot do.
    try:
        training_audit_table().update_item(
            Key={"id": audit["id"]},
            UpdateExpression="SET #execution_arn = :arn, #status = :status",
            ExpressionAttributeNames={"#execution_arn": "execution_arn", "#status": "status"},
            ExpressionAttributeValues={":arn": execution_arn, ":status": "Executing"},
        )
    except Exception as exc:
        print(f"Could not attach execution {execution_arn} to audit {audit['id']}: {exc}")
    return {"run_id": run_id, "execution_arn": execution_arn, "audit_id": audit["id"]}


def _get_build_status(build_id: str) -> str | None:
    try:
        builds = codebuild.batch_get_builds(ids=[build_id])["builds"]
        return builds[0]["buildStatus"] if builds else None
    except Exception as exc:
        print(f"Could not get CodeBuild build status for {build_id}: {exc}")
        return None


def pending_submission_overlay(progress: dict | None) -> str | None:
    """None normally. Otherwise a message the Models page should show
    INSTEAD OF the normal training-progress panel, bridging the CodeBuild-
    to-pipeline-execution gap described where _pending_submission is
    declared above."""
    global _pending_submission
    with _pending_submission_lock:
        pending = dict(_pending_submission) if _pending_submission else None
    if not pending:
        return None
    if progress and progress["start_time"] and progress["start_time"] >= pending["triggered_at"].isoformat():
        with _pending_submission_lock:
            _pending_submission = None
        return None
    status = _get_build_status(pending["build_id"])
    if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
        with _pending_submission_lock:
            _pending_submission = None
        return f"Training submission failed -- CodeBuild build {pending['build_id']} ended {status}. Check that build's own logs for why."
    if status == "SUCCEEDED":
        return "Finalizing new run..."
    return f"Submitting new run... (CodeBuild build {status or 'IN_PROGRESS'})"


# --- ad-hoc training queue -----------------------------------------------------

def queue_snapshot() -> dict:
    """Point-in-time copy of the queue state for a response -- copied out
    under the lock so a concurrent poll-loop mutation is never read half-
    applied."""
    with _queue_lock:
        return {
            "pending": [dict(e) for e in _training_queue],
            "running": dict(_queue_running) if _queue_running else None,
            "started": _queue_started,
            "completed_count": _queue_completed_count,
            "last_error": _queue_last_error,
            "max_size": MAX_QUEUE_SIZE,
        }


def queue_add(entry: dict) -> None:
    with _queue_lock:
        if len(_training_queue) >= MAX_QUEUE_SIZE:
            raise ValueError(f"Queue is full (max {MAX_QUEUE_SIZE} runs) -- remove an entry before adding another.")
        _training_queue.append(entry)


def queue_remove(entry_id: str) -> bool:
    with _queue_lock:
        before = len(_training_queue)
        _training_queue[:] = [e for e in _training_queue if e["id"] != entry_id]
        return len(_training_queue) < before


def queue_start() -> bool:
    """Returns False if there's nothing to start (empty queue, nothing
    already running)."""
    global _queue_started
    with _queue_lock:
        if not _training_queue and _queue_running is None:
            return False
        _queue_started = True
    try_advance_queue()  # don't wait up to ~25s for the next poll tick
    return True


def queue_stop() -> None:
    global _queue_started
    with _queue_lock:
        _queue_started = False


def try_advance_queue() -> None:
    """Called both by the background poll loop (every ~25s) and
    immediately after "Start Queue", so the first run doesn't have to wait
    for the next poll tick. If the queue has been started, nothing else is
    occupying the account's one GPU slot, and there's at least one pending
    entry, pops it and starts it."""
    global _queue_running, _queue_started, _queue_completed_count, _queue_last_error
    with _queue_lock:
        if not _queue_started:
            return
        reason = training_already_running_reason(get_training_progress())
        if reason:
            # Something is Executing -- either the queue's own current
            # entry (leave _queue_running as-is) or an unrelated run
            # started elsewhere. Either way, not safe to start the next.
            return
        if _queue_running is not None:
            _queue_completed_count += 1
            _queue_running = None
        if not _training_queue:
            _queue_started = False
            return
        entry = _training_queue.pop(0)
        try:
            build_id = start_training_run(
                entry["generation"], entry["size"], entry["epochs"],
                entry["dataset_folder"], entry["base_model_folder"], entry["notes"],
                entry["img_size"], entry["batch"], entry["seed"],
                entry.get("eval_dataset_folder", ""), source="queue",
            )
            entry["build_id"] = build_id
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            _queue_running = entry
            _queue_last_error = ""
        except Exception as exc:
            # Don't put it back -- retrying the same bad entry forever
            # would just spin every poll.
            _queue_last_error = f"Queue entry (YOLO{entry['generation']}-{entry['size']}) failed to start, skipped: {exc}"
            print(_queue_last_error)


async def queue_poll_loop() -> None:
    """Runs for the life of the process as an asyncio background task
    (started from main.py's lifespan). Uses asyncio.sleep (not time.sleep)
    since it shares the event loop with every request handler."""
    import asyncio
    while True:
        await asyncio.sleep(25)
        try:
            try_advance_queue()
        except Exception as exc:
            print(f"Queue poll failed: {exc}")
        # Same tick rather than its own loop: both want to know when a run
        # changed state, and one poll is one set of SageMaker calls.
        try:
            reconcile_training_audits()
        except Exception as exc:
            print(f"Training audit reconcile failed: {exc}")
        # The safety net for a teardown that never ran because the process
        # died between the last job and it. Cheap when there is no endpoint:
        # one describe_endpoint that returns not-found.
        #
        # Off the event loop: when it does delete, it waits for the
        # inference component to go first, which took 150s in practice --
        # running that inline would stall every request in this
        # single-worker process for the duration.
        try:
            await asyncio.get_running_loop().run_in_executor(None, sweep_idle_endpoint)
        except Exception as exc:
            print(f"Idle endpoint sweep failed: {exc}")


# --- training audit ------------------------------------------------------
#
# The pipeline already writes a lineage row of its own (LINEAGE_TABLE_NAME,
# record_type "training_run"), but it is written from inside the pipeline
# and only carries what the pipeline itself knows: dataset URI, epochs,
# img_size, base model. It has no evaluation dataset, no file names, no
# batch/seed, no end time, and no metrics -- the metrics live on the
# registered model package, reachable only backwards via its lineage_id.
#
# This table is the app's own record, written at submit time (when the
# operator's exact choices are still known) and completed afterwards by
# reconcile_training_audits when the run finishes. Separate table rather
# than extra fields on the pipeline's row: that row is owned and shaped by
# a different deploy unit, and writing into it from here would race with
# it and blur who owns what.

TRAINING_AUDIT_TABLE_NAME = os.environ.get("TRAINING_AUDIT_TABLE", "track2-yolo-training-audit")

# A YOLO dataset folder can hold tens of thousands of images and a single
# DynamoDB item is capped at 400KB, so the audit keeps a bounded sample of
# names plus the true total. Recording "1,000 of 24,317" is honest; a
# silently truncated list that looks complete is not, so `truncated` is
# stored alongside and the UI says so.
AUDIT_MAX_FILE_NAMES = 1000
# Bounds the submit-time listing itself. 1000 keys per page, so this is up
# to 200k objects -- past that the count is reported as a floor.
AUDIT_MAX_LIST_PAGES = 200


def training_audit_table():
    return _dynamodb().Table(TRAINING_AUDIT_TABLE_NAME)


def list_dataset_image_names(folder: str) -> tuple[list, int, bool]:
    """(sample of image names, total found, whether either bound was hit).

    Recursive, unlike list_s3_images: a YOLO dataset is images/train/... and
    images/val/..., so a Delimiter-scoped listing of the dataset root finds
    no images at all. Names are returned relative to `folder` so the split
    each file came from stays visible.
    """
    if not folder:
        return [], 0, False
    prefix = folder.rstrip("/") + "/"
    names: list = []
    total = 0
    truncated = False
    try:
        paginator = s3.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=BUCKET, Prefix=prefix)
        for page_number, page in enumerate(pages, start=1):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/") or not key.lower().endswith(IMAGE_EXTENSIONS):
                    continue
                total += 1
                if len(names) < AUDIT_MAX_FILE_NAMES:
                    names.append(key[len(prefix):])
            if page_number >= AUDIT_MAX_LIST_PAGES:
                truncated = True
                break
    except Exception as exc:
        print(f"Could not list images under {folder}: {exc}")
        return names, total, True
    if total > len(names):
        truncated = True
    return names, total, truncated


def _dataset_audit_block(folder: str) -> dict:
    names, total, truncated = list_dataset_image_names(folder)
    return {
        "folder": folder,
        "s3_uri": f"s3://{BUCKET}/{folder}/" if folder else "",
        "file_count": total,
        "file_names": names,
        "truncated": truncated,
    }


def write_training_audit(
    *, build_id, generation, size, epochs, img_size, batch, seed,
    dataset_folder, eval_dataset_folder, base_model_folder, notes, source="manual",
) -> dict:
    """One row per submitted run, written before the run is under way so
    the operator's exact inputs are captured even if the run later fails.
    Never raises: an audit write failing must not take down the training
    submission it is describing."""
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": uuid.uuid4().hex[:16],
        "record_type": "training_audit",
        "build_id": build_id or "",
        "source": source,
        "status": "Submitted",
        "submitted_at": now,
        "started_at": None,
        "ended_at": None,
        "generation": str(generation),
        "size": str(size),
        "parameters": {
            "epochs": int(epochs),
            "img_size": int(img_size),
            "batch": int(batch),
            "seed": int(seed),
        },
        "notes": notes or "",
        "train_dataset": _dataset_audit_block(dataset_folder),
        # Falls back to the training dataset, which is what the pipeline
        # evaluates on when no separate evaluation set is chosen -- worth
        # recording explicitly, since "evaluated on its own training data"
        # is exactly the thing an audit should make visible rather than
        # leave as a blank to be misread as "unknown".
        "eval_dataset": _dataset_audit_block(eval_dataset_folder or dataset_folder),
        "eval_is_train_dataset": not eval_dataset_folder,
        "base_model": {
            "folder": base_model_folder or "",
            "source": "folder" if base_model_folder else "registry-latest-approved",
        },
        "model": None,
        "metrics": None,
        "execution_arn": "",
    }
    try:
        training_audit_table().put_item(Item=record)
    except Exception as exc:
        print(f"Could not write training audit for build {build_id}: {exc}")
    return record


def scan_training_audits(limit: int = 100) -> list:
    """Every audit row, newest first by submitted_at."""
    items: list = []
    kwargs: dict = {}
    try:
        while True:
            response = training_audit_table().scan(**kwargs)
            items.extend(response.get("Items", []))
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
    except Exception as exc:
        print(f"Could not scan training audits: {exc}")
        return []
    items.sort(key=lambda item: str(item.get("submitted_at") or ""), reverse=True)
    return items[:limit]


def delete_training_audit(audit_id: str) -> dict:
    """Removes one run from the history, and with it the model version that
    run produced.

    Deleting the record but leaving the model would make the history lie:
    the registry would still hold a version with nothing saying where it
    came from. The model package is deleted first -- if that fails the row
    stays, so the pair never drifts apart silently.
    """
    try:
        audit = training_audit_table().get_item(Key={"id": audit_id}).get("Item")
    except Exception as exc:
        raise ValueError(f"Could not read run {audit_id}: {exc}")
    if not audit:
        raise KeyError(audit_id)

    model = audit.get("model") or {}
    model_package_arn = model.get("model_package_arn")
    deleted_model = None
    if model_package_arn:
        try:
            sm.delete_model_package(ModelPackageName=model_package_arn)
            deleted_model = f"{model.get('group')} v{model.get('version')}"
        except sm.exceptions.ResourceNotFound:
            # Already gone -- deleting the row is still the right outcome.
            deleted_model = None
        except Exception as exc:
            raise ValueError(
                f"Could not delete model package {model_package_arn}: {exc}. "
                "The run was kept, so the history still matches the registry."
            )

    training_audit_table().delete_item(Key={"id": audit_id})
    return {"deleted": True, "deleted_model": deleted_model}


def _decimalize(value):
    """DynamoDB rejects float. Metrics arrive as floats from evaluation.json,
    so they are stored as Decimal via str() -- Decimal(float) would preserve
    the binary representation error (0.1 -> 0.1000000000000000055...)."""
    from decimal import Decimal

    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _decimalize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimalize(v) for v in value]
    return value


def _execution_for_audit(submitted_at: str) -> dict | None:
    """The pipeline execution a submitted audit turned into.

    There is no id shared between the CodeBuild build that submits a run
    and the pipeline execution it creates -- run_pipeline.py starts the
    execution from inside the build -- so they are matched on time: the
    OLDEST execution that started at or after this audit was written. Two
    runs cannot overlap (the account has one GPU slot, enforced by
    training_already_running_reason), so this is unambiguous.
    """
    try:
        summaries = sm.list_pipeline_executions(
            PipelineName=PIPELINE_NAME, SortBy="CreationTime", SortOrder="Descending", MaxResults=20,
        )["PipelineExecutionSummaries"]
    except Exception as exc:
        print(f"Could not list pipeline executions for audit: {exc}")
        return None
    candidates = [
        s for s in summaries
        if s.get("StartTime") and s["StartTime"].isoformat() >= submitted_at
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda s: s["StartTime"])


def _execution_steps(execution_arn: str) -> list:
    try:
        return sm.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)["PipelineExecutionSteps"]
    except Exception as exc:
        print(f"Could not list steps for {execution_arn}: {exc}")
        return []


def _actual_run_parameters(steps: list) -> dict | None:
    """What the training job was ACTUALLY given, read off the job itself.

    Not the same thing as what this app submitted. The deployed
    run_pipeline.py substitutes its own dataset, base model and
    hyperparameters for the ones sent in the build's environment, so an
    audit that only recorded the request would describe a run that never
    happened. Both are kept, and the UI shows the difference.
    """
    job_name = None
    for step in steps:
        arn = (step.get("Metadata") or {}).get("TrainingJob", {}).get("Arn")
        if arn:
            job_name = arn.rsplit("/", 1)[-1]
            break
    if not job_name:
        return None
    try:
        job = sm.describe_training_job(TrainingJobName=job_name)
    except Exception as exc:
        print(f"Could not describe training job {job_name}: {exc}")
        return None
    hyper = job.get("HyperParameters", {}) or {}

    def _int(key):
        try:
            return int(hyper[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {
        "training_job_name": job_name,
        "instance_type": job.get("ResourceConfig", {}).get("InstanceType", ""),
        "epochs": _int("epochs"),
        "img_size": _int("img-size"),
        "batch": _int("batch"),
        "seed": _int("seed"),
        "train_data_s3_uri": hyper.get("train-data-s3-uri", ""),
        "base_model_s3_uri": hyper.get("base-model-s3-uri", ""),
        "run_id": hyper.get("run-id", ""),
    }


def _registered_model_for_execution(steps: list) -> dict | None:
    """The model package this execution registered, read from the
    execution's own RegisterModel step. Returns None until that step has
    run (or if the run never registers anything).

    Asked of the execution rather than searched for by group and time. An
    earlier version guessed the group from the UI's generation and size --
    "yolo11s-detector" for a YOLO11-S run -- and searched it for a package
    created after the run started. That found nothing whenever the pipeline
    registered somewhere else, which is exactly what the deployed
    run_pipeline.py does: it ignores the submitted size and registers into
    track2-ppe-fullframe-replay. The step metadata carries the real ARN, so
    there is nothing left to infer.
    """
    arn = None
    for step in steps:
        metadata = step.get("Metadata") or {}
        register = metadata.get("RegisterModel") or {}
        if register.get("Arn"):
            arn = register["Arn"]
            break
    if not arn:
        return None
    # "arn:aws:sagemaker:<region>:<account>:model-package/<group>/<version>"
    # -- the separator before "model-package" is a COLON, not a slash.
    tail = arn.rsplit(":model-package/", 1)[-1]
    group, _, version = tail.partition("/")
    approval_status = ""
    try:
        approval_status = sm.describe_model_package(ModelPackageName=arn).get("ModelApprovalStatus", "")
    except Exception as exc:
        print(f"Could not describe {arn}: {exc}")
    return {
        "group": group,
        "version": version,
        "model_package_arn": arn,
        "approval_status": approval_status,
    }


TERMINAL_EXECUTION_STATUSES = ("Succeeded", "Failed", "Stopped")


def reconcile_training_audits() -> None:
    """Fills in what is only knowable after submission -- start and end
    time, final status, the registered model version, and its metrics.
    Called from the queue poll loop; never raises."""
    try:
        audits = scan_training_audits(limit=200)
    except Exception as exc:
        print(f"Could not read training audits to reconcile: {exc}")
        return
    for audit in audits:
        # Done means: it finished, we know which model it registered (or
        # established it registered none), and we have that model's metrics.
        # An earlier version stopped retrying as soon as a run reached a
        # terminal status without a model attached, which permanently froze
        # every row whose model lookup had failed even once -- the failure
        # became indistinguishable from "this run registered nothing".
        if audit.get("status") in TERMINAL_EXECUTION_STATUSES and audit.get("actual"):
            has_model = bool(audit.get("model"))
            if has_model and audit.get("metrics") is not None:
                continue
            # Only give up on a run that finished without registering
            # anything. A row that has a model but no metrics is still
            # incomplete, and stopping here is what left the first run
            # showing a model with nothing measured against it.
            if not has_model and audit.get("model_lookup_done"):
                continue
        try:
            _reconcile_one_audit(audit)
        except Exception as exc:
            print(f"Could not reconcile training audit {audit.get('id')}: {exc}")


def _reconcile_one_audit(audit: dict) -> None:
    updates: dict = {}
    execution_arn = audit.get("execution_arn") or ""
    if not execution_arn:
        summary = _execution_for_audit(str(audit.get("submitted_at") or ""))
        if not summary:
            return
        execution_arn = summary["PipelineExecutionArn"]
        updates["execution_arn"] = execution_arn
        updates["started_at"] = summary["StartTime"].isoformat()

    try:
        description = sm.describe_pipeline_execution(PipelineExecutionArn=execution_arn)
    except Exception as exc:
        print(f"Could not describe execution {execution_arn}: {exc}")
        return
    status = description.get("PipelineExecutionStatus", "")
    if status and status != audit.get("status"):
        updates["status"] = status
    start_time = description.get("CreationTime")
    if start_time and not audit.get("started_at"):
        updates["started_at"] = start_time.isoformat()
    if status in TERMINAL_EXECUTION_STATUSES:
        end_time = description.get("LastModifiedTime")
        if end_time and not audit.get("ended_at"):
            updates["ended_at"] = end_time.isoformat()

    # Re-looked-up whenever the metrics are still missing, not only when
    # the model is: that also repairs a row recorded by an earlier, buggier
    # version of this code without anyone having to delete it.
    needs_model = audit.get("model") is None or audit.get("metrics") is None
    steps = _execution_steps(execution_arn) if (needs_model or not audit.get("actual")) else []

    if not audit.get("actual"):
        actual = _actual_run_parameters(steps)
        if actual:
            updates["actual"] = actual

    if needs_model:
        model = _registered_model_for_execution(steps)
        if model:
            updates["model"] = model
            updates["model_lookup_done"] = True
            metrics, _description, _lineage = fetch_metrics_and_lineage(model["model_package_arn"])
            if metrics:
                updates["metrics"] = _decimalize(metrics)
        elif status in TERMINAL_EXECUTION_STATUSES:
            # The run is over and its steps carry no RegisterModel ARN, so
            # there is genuinely no model to find. Recorded so the next
            # pass stops looking, without that being confused with a
            # lookup that merely failed.
            updates["model_lookup_done"] = True

    if not updates:
        return
    expression = ", ".join(f"#{k} = :{k}" for k in updates)
    training_audit_table().update_item(
        Key={"id": audit["id"]},
        UpdateExpression=f"SET {expression}",
        ExpressionAttributeNames={f"#{k}": k for k in updates},
        ExpressionAttributeValues={f":{k}": v for k, v in updates.items()},
    )


# --- evaluation queue (bring-your-own-weights or registry, against a chosen
# dataset, with real detection metrics) ---------------------------------------
#
# Ground truth format matches the training dataset convention documented on
# the Help page: <data_folder>/images/*.jpg + <data_folder>/labels/*.txt
# (YOLO: "class_id cx cy w h", normalized 0-1). An image with no matching
# label file is treated as having zero ground-truth objects, not an error.
#
# Deploying the requested weights before evaluating (rather than running
# inference in-process) means an eval job temporarily occupies that size's
# Inference Component the same way a real deploy would -- by design, per
# the explicit choice to reuse the existing deploy path rather than add a
# second, separate in-process inference mechanism.

EVAL_RUNS_TABLE_NAME = "track2-yolo-eval-runs"
# Must exceed SageMaker's own ModelDataDownloadTimeoutInSeconds (600) +
# ContainerStartupHealthCheckTimeoutInSeconds (600) in deploy_inference_
# component, since those are sequential phases (download, then health-
# check) SageMaker itself is willing to wait through -- an app-level
# timeout shorter than that gives up and reports "failed" on a deploy
# that was still legitimately in progress and might have gone on to
# actually succeed. Confirmed live: a real deploy timed out and was
# marked failed at the old 300s value while SageMaker was still
# reporting a normal in-progress "Updating" (no FailureReason) well
# past that point.
INFERENCE_COMPONENT_READY_TIMEOUT_SECONDS = 1260
INFERENCE_COMPONENT_READY_POLL_SECONDS = 5
# COCO-style mAP50-95 averages AP over these ten IoU thresholds.
MAP_5095_IOU_THRESHOLDS = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))


def eval_runs_table():
    return _dynamodb().Table(EVAL_RUNS_TABLE_NAME)


def _yolo_line_to_xyxy(cx: float, cy: float, w: float, h: float) -> tuple:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _box_iou(a: tuple, b: tuple) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def _average_precision(confidence_and_hit: list, num_ground_truth: int) -> float:
    """Standard (VOC2012-style, continuous) average precision: sort
    predictions by confidence, accumulate TP/FP, take the precision
    envelope (monotonically non-increasing as recall grows, reading from
    the high-recall end backwards), then integrate precision over recall.
    confidence_and_hit must already be sorted by confidence descending."""
    if num_ground_truth == 0 or not confidence_and_hit:
        return 0.0
    precisions, recalls = [], []
    tp = fp = 0
    for _, is_hit in confidence_and_hit:
        if is_hit:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / num_ground_truth)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * (r - prev_recall)
        prev_recall = r
    return ap


def _match_class_predictions(per_image: dict, class_id: int, iou_threshold: float) -> tuple:
    """Greedy IoU matching for one class at one IoU threshold: each
    prediction (highest confidence first) claims its best-IoU unmatched
    ground-truth box in the SAME image; a claim below iou_threshold, or
    with no ground truth left to claim, is a false positive. Returns
    (confidence_and_hit sorted by confidence desc, total ground-truth
    count for this class)."""
    gt_by_image: dict = {}
    matched_by_image: dict = {}
    num_gt = 0
    predictions = []
    for image_key, data in per_image.items():
        boxes = [box for (c, box) in data["gt"] if c == class_id]
        gt_by_image[image_key] = boxes
        matched_by_image[image_key] = [False] * len(boxes)
        num_gt += len(boxes)
        for confidence, c, box in data["pred"]:
            if c == class_id:
                predictions.append((confidence, image_key, box))
    predictions.sort(key=lambda p: -p[0])

    results = []
    for confidence, image_key, box in predictions:
        gts = gt_by_image[image_key]
        matched = matched_by_image[image_key]
        best_iou, best_idx = 0.0, -1
        for i, gt_box in enumerate(gts):
            if matched[i]:
                continue
            iou = _box_iou(box, gt_box)
            if iou > best_iou:
                best_iou, best_idx = iou, i
        is_hit = best_idx != -1 and best_iou >= iou_threshold
        if is_hit:
            matched[best_idx] = True
        results.append((confidence, is_hit))
    return results, num_gt


def evaluate_model_against_dataset(data_folder: str) -> dict:
    """Runs inference (via whatever's currently deployed to the active
    Inference Component -- the caller is expected to have just deployed
    the size/version being evaluated) against every image under
    <data_folder>/images/, compares to YOLO-format ground truth under
    <data_folder>/labels/, and computes mAP50, mAP50-95, precision,
    recall, and accuracy (TP / (TP+FP+FN) at IoU 0.5 -- object detection
    has no single standard "accuracy", this is the closest single-number
    proxy and is reported alongside precision/recall rather than instead
    of them, same shape as the training pipeline's own metrics)."""
    prefix = data_folder.strip("/")
    if not prefix:
        raise ValueError("No dataset folder selected.")
    images_prefix = f"{prefix}/images"
    labels_prefix = f"{prefix}/labels"
    image_names = list_s3_images(images_prefix)
    if not image_names:
        raise ValueError(f"No images found under s3://{BUCKET}/{images_prefix}/")

    per_image: dict = {}
    all_classes: set = set()
    for name in image_names:
        image_key = f"{images_prefix}/{name}"
        stem = name.rsplit(".", 1)[0] if "." in name else name
        label_key = f"{labels_prefix}/{stem}.txt"
        gt_boxes = []
        try:
            body = s3.get_object(Bucket=BUCKET, Key=label_key)["Body"].read().decode("utf-8")
            for line in body.splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                cx, cy, w, h = (float(x) for x in parts[1:5])
                gt_boxes.append((class_id, _yolo_line_to_xyxy(cx, cy, w, h)))
                all_classes.add(class_id)
        except botocore.exceptions.ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
                raise
            # No label file for this image -- zero ground-truth objects,
            # not an error (a real, if unusual, negative example).

        image_bytes = s3.get_object(Bucket=BUCKET, Key=image_key)["Body"].read()
        result = invoke_inference(image_bytes)
        predictions = []
        for d in result.get("detections", []):
            box = tuple(d["box"])
            predictions.append((d["confidence"], d["class"], box))
            all_classes.add(d["class"])

        per_image[image_key] = {"gt": gt_boxes, "pred": predictions}

    if not all_classes:
        raise ValueError("No ground-truth or predicted objects found across the whole dataset -- nothing to score.")

    ap50_per_class = []
    ap_5095_per_class: dict = {t: [] for t in MAP_5095_IOU_THRESHOLDS}
    tp_total = fp_total = fn_total = 0
    for class_id in sorted(all_classes):
        hits_50, num_gt = _match_class_predictions(per_image, class_id, 0.5)
        ap50_per_class.append(_average_precision(hits_50, num_gt))
        tp = sum(1 for _, is_hit in hits_50 if is_hit)
        fp = len(hits_50) - tp
        fn = num_gt - tp
        tp_total += tp
        fp_total += fp
        fn_total += fn
        for threshold in MAP_5095_IOU_THRESHOLDS:
            hits_t, num_gt_t = _match_class_predictions(per_image, class_id, threshold)
            ap_5095_per_class[threshold].append(_average_precision(hits_t, num_gt_t))

    map50 = sum(ap50_per_class) / len(ap50_per_class)
    map_5095 = sum(sum(v) / len(v) for v in ap_5095_per_class.values()) / len(MAP_5095_IOU_THRESHOLDS)
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    accuracy = tp_total / (tp_total + fp_total + fn_total) if (tp_total + fp_total + fn_total) else 0.0

    return {
        "mAP50": map50,
        "mAP50-95": map_5095,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "num_images": len(image_names),
        "num_classes": len(all_classes),
    }


# In-memory control switch only -- DynamoDB itself is the actual queue
# (scanned for status="queued" items), same reasoning as the rest of this
# module's in-memory state: lost on restart, but the persisted job records
# aren't, so nothing queued is silently forgotten, only "started" resets to
# "stopped" and needs clicking again.
_eval_queue_started = False
_eval_worker_thread = None
_eval_worker_lock = threading.Lock()


def queue_eval_run(
    *, size: str, data_folder: str, weights_kind: str,
    model_package_arn: str = None, s3_folder: str = None,
) -> dict:
    if weights_kind not in ("registry", "s3"):
        raise ValueError(f"Unknown weights_kind: {weights_kind!r}")
    if weights_kind == "registry" and not model_package_arn:
        raise ValueError("model_package_arn is required for weights_kind='registry'.")
    if weights_kind == "s3" and not (s3_folder or "").strip("/"):
        raise ValueError("s3_folder is required for weights_kind='s3'.")
    if not data_folder.strip("/"):
        raise ValueError("No dataset folder selected.")

    s3_folder = (s3_folder or "").strip("/") or None
    weights_label = (
        f"YOLO11-{size.upper()} v{model_package_arn.rsplit('/', 1)[-1]}"
        if weights_kind == "registry"
        else f"S3: {s3_folder}"
    )
    now = datetime.now(timezone.utc).isoformat()
    item = {
        "id": uuid.uuid4().hex[:12],
        "status": "queued",
        "size": size,
        "data_folder": data_folder.strip("/"),
        "weights_kind": weights_kind,
        "weights_label": weights_label,
        "model_package_arn": model_package_arn,
        "s3_folder": s3_folder,
        "created_at": now,
        "started_at": None,
        "completed_at": None,
        "metrics": None,
        "error": None,
    }
    eval_runs_table().put_item(Item=item)
    return item


def list_eval_runs() -> list:
    items = []
    kwargs: dict = {}
    while True:
        response = eval_runs_table().scan(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items


def remove_eval_run(run_id: str) -> bool:
    """Only removes a run that hasn't started yet -- once it's running or
    finished, the record stays as history."""
    item = eval_runs_table().get_item(Key={"id": run_id}).get("Item")
    if not item or item.get("status") != "queued":
        return False
    eval_runs_table().delete_item(Key={"id": run_id})
    return True


def eval_queue_snapshot() -> dict:
    return {
        "started": _eval_queue_started,
        "worker_alive": _eval_worker_thread is not None and _eval_worker_thread.is_alive(),
    }


def _next_queued_eval_run():
    queued = [r for r in list_eval_runs() if r.get("status") == "queued"]
    if not queued:
        return None
    queued.sort(key=lambda r: str(r.get("created_at") or ""))
    return queued[0]


def wait_for_inference_component_ready(
    *,
    timeout_seconds: int = INFERENCE_COMPONENT_READY_TIMEOUT_SECONDS,
    poll_seconds: int = INFERENCE_COMPONENT_READY_POLL_SECONDS,
) -> None:
    """Blocks (real time.sleep, meant for a background thread, not the
    asyncio event loop) until the active Inference Component reports
    InService with at least one running copy, or raises after
    timeout_seconds. Used by both the eval queue and the inference queue,
    each of which deploys something and then needs to know when it's
    actually safe to run inference against it -- SageMaker only exposes
    that via polling describe."""
    deadline = time.monotonic() + timeout_seconds
    last_status, last_copies = None, 0
    while time.monotonic() < deadline:
        last_status, last_copies = _inference_component_readiness()
        if last_status == "InService" and last_copies > 0:
            return
        time.sleep(poll_seconds)
    raise RuntimeError(
        f"Timed out after {timeout_seconds}s waiting for {ACTIVE_INFERENCE_COMPONENT_NAME} to become "
        f"InService with a running copy. Last seen: status={last_status!r}, running copies={last_copies}."
    )


def _inference_component_readiness() -> tuple[str | None, int]:
    """(status, running copy count) from ONE describe call.

    Deliberately not describe_deployed_inference_component: that also
    resolves the model's tags for the UI, which is two further SageMaker
    calls. Fine once per page render, wasteful three times every five
    seconds -- and both queues wait while a background thread waits on the
    same component, so the redundant calls arrive doubled and invite
    throttling on exactly the path that must not stall."""
    try:
        desc = sm.describe_inference_component(InferenceComponentName=ACTIVE_INFERENCE_COMPONENT_NAME)
    except sm.exceptions.ResourceNotFound:
        return None, 0
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find" in str(exc):
            return None, 0
        raise
    runtime = desc.get("RuntimeConfig") or {}
    return desc.get("InferenceComponentStatus"), runtime.get("CurrentCopyCount") or 0


def _run_eval_job(run: dict) -> None:
    run_id = run["id"]
    size = run["size"]
    try:
        eval_runs_table().update_item(
            Key={"id": run_id},
            UpdateExpression="SET #s = :status, started_at = :ts",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":status": "deploying", ":ts": datetime.now(timezone.utc).isoformat()},
        )
        if _deployed_matches(run):
            print(f"Eval run {run_id}: {run['weights_label']} already live -- skipping redeploy.")
        else:
            if run["weights_kind"] == "registry":
                deploy_inference_component(size, model_package_arn=run["model_package_arn"])
            else:
                deploy_inference_component(size, weights_s3_prefix=run["s3_folder"])

            wait_for_inference_component_ready()

        eval_runs_table().update_item(
            Key={"id": run_id},
            UpdateExpression="SET #s = :status",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":status": "running"},
        )
        metrics = evaluate_model_against_dataset(run["data_folder"])
        eval_runs_table().update_item(
            Key={"id": run_id},
            UpdateExpression="SET #s = :status, completed_at = :ts, metrics = :metrics",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":status": "completed",
                ":ts": datetime.now(timezone.utc).isoformat(),
                ":metrics": metrics,
            },
        )
    except Exception as exc:
        eval_runs_table().update_item(
            Key={"id": run_id},
            UpdateExpression="SET #s = :status, completed_at = :ts, #e = :error",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={
                ":status": "failed",
                ":ts": datetime.now(timezone.utc).isoformat(),
                ":error": str(exc),
            },
        )
        print(f"Eval run {run_id} failed: {exc}")


def _eval_worker_loop() -> None:
    """Runs on its own daemon thread, never the asyncio event loop --
    each job makes many sequential network calls (one invoke_inference
    per image, plus waiting for a deploy), which would otherwise stall
    every other request in this single-worker process for as long as the
    job takes."""
    global _eval_queue_started
    while _eval_queue_started:
        run = _next_queued_eval_run()
        if run is None:
            break
        _run_eval_job(run)
    # Draining naturally (nothing left queued) is not the same as Stop --
    # reset the flag either way so a later Add doesn't leave "Start" stuck
    # disabled and "Stop" stuck enabled for a queue that's actually idle.
    _eval_queue_started = False
    # Flag cleared first: the teardown check reads it to decide whether
    # anything still needs the endpoint.
    delete_endpoint_if_idle()


def start_eval_queue() -> bool:
    """Returns False if there's nothing queued to start."""
    global _eval_queue_started, _eval_worker_thread
    if _next_queued_eval_run() is None:
        return False
    with _eval_worker_lock:
        _eval_queue_started = True
        if _eval_worker_thread is None or not _eval_worker_thread.is_alive():
            _eval_worker_thread = threading.Thread(target=_eval_worker_loop, daemon=True)
            _eval_worker_thread.start()
    return True


def stop_eval_queue() -> None:
    """Stops picking up further queued runs once the current one (if any)
    finishes -- there's no clean way to abort a run already mid-evaluation."""
    global _eval_queue_started
    _eval_queue_started = False


# --- inference queue (deploy a model, batch-infer a folder, save results,
# then swap to the next queued model) ------------------------------------
#
# Same shape as the eval queue above -- deploy, wait for InService, do the
# real work, record the outcome -- but without ground truth or metrics:
# each item just runs run_batch_inference over its own images_folder/
# output_folder pair. DynamoDB-backed (like the eval queue's own table,
# not the old in-memory list this replaced) -- a queued-but-not-started
# job, and every job's full history (status, errors, result), all
# survive an app restart; only the in-memory started/worker-alive flags
# below reset, same as the eval queue.

INFERENCE_QUEUE_TABLE_NAME = "track2-yolo-inference-queue"

_inference_queue_started = False
_inference_worker_thread = None
_inference_worker_lock = threading.Lock()


def inference_queue_table():
    return _dynamodb().Table(INFERENCE_QUEUE_TABLE_NAME)


def queue_inference_run(
    *, size: str, weights_kind: str, images_folder: str, output_folder: str,
    model_package_arn: str = None, weights_folder: str = None, weights_key: str = None,
    notes: str = None, comment: str = None,
) -> dict:
    if size not in REGISTRY_SIZES:
        raise ValueError(f"Unknown size: {size!r}")
    if weights_kind not in ("registry", "s3"):
        raise ValueError(f"Unknown weights_kind: {weights_kind!r}")
    if weights_kind == "registry":
        if not model_package_arn:
            raise ValueError("model_package_arn is required for weights_kind='registry'.")
    else:
        weights_folder = (weights_folder or "").strip("/")
        weights_key = (weights_key or "").strip("/")
        if bool(weights_folder) == bool(weights_key):
            raise ValueError("Give exactly one of weights_folder or weights_key for weights_kind='s3'.")
    if not images_folder.strip("/"):
        raise ValueError("No images folder selected.")
    if not output_folder.strip("/"):
        raise ValueError("No output folder selected.")

    pending = sum(1 for i in list_inference_queue() if i["status"] == "queued")
    if pending >= MAX_QUEUE_SIZE:
        raise ValueError(f"Queue is full (max {MAX_QUEUE_SIZE} runs) -- remove an entry before adding another.")

    weights_label = (
        f"YOLO11-{size.upper()} v{model_package_arn.rsplit('/', 1)[-1]}"
        if weights_kind == "registry"
        else f"S3: {weights_key or weights_folder}"
    )
    item = {
        "id": uuid.uuid4().hex[:12],
        "status": "queued",
        "size": size,
        "weights_kind": weights_kind,
        "weights_label": weights_label,
        "model_package_arn": model_package_arn,
        "weights_folder": weights_folder or None,
        "weights_key": weights_key or None,
        # Only ever set by the caller for weights_kind="s3" -- a registry
        # version already has its own recorded description via the model
        # registry, but a checkpoint picked from a bare S3 folder has no
        # metadata at all beyond its file path, so this is the only place
        # to record what it actually is.
        "notes": (notes or "").strip() or None,
        # Unlike notes, applies regardless of weights_kind -- a free-text
        # note about this particular run (why it's being done, what the
        # images folder is), not about the checkpoint itself.
        "comment": (comment or "").strip() or None,
        "images_folder": images_folder.strip("/"),
        "output_folder": output_folder.strip("/"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    inference_queue_table().put_item(Item=item)
    return item


def list_inference_queue() -> list:
    items = []
    kwargs: dict = {}
    while True:
        response = inference_queue_table().scan(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items


def delete_inference_queue_item(item_id: str) -> bool:
    """Deletes a queue/history record outright -- queued, done, or
    failed. Refuses only an item actively deploying/running, since the
    worker thread is still writing to that record mid-job. Supersedes
    the old "remove" (which only ever allowed a still-queued item) now
    that history is durable rather than session-only: a done/failed
    record that's no longer wanted needs an explicit way to go too."""
    item = inference_queue_table().get_item(Key={"id": item_id}).get("Item")
    if not item or item.get("status") in ("deploying", "running"):
        return False
    inference_queue_table().delete_item(Key={"id": item_id})
    return True


def inference_queue_snapshot() -> dict:
    return {
        "started": _inference_queue_started,
        "worker_alive": _inference_worker_thread is not None and _inference_worker_thread.is_alive(),
    }


def _next_queued_inference_item():
    queued = [i for i in list_inference_queue() if i.get("status") == "queued"]
    if not queued:
        return None
    queued.sort(key=lambda i: str(i.get("created_at") or ""))
    return queued[0]


def _update_inference_queue_item(item_id: str, **fields) -> None:
    inference_queue_table().update_item(
        Key={"id": item_id},
        UpdateExpression="SET " + ", ".join(f"#{k} = :{k}" for k in fields),
        ExpressionAttributeNames={f"#{k}": k for k in fields},
        ExpressionAttributeValues={f":{k}": v for k, v in fields.items()},
    )


def _deployed_matches(item: dict) -> bool:
    """True if the active Inference Component is already InService with
    at least one running copy of exactly this model. deploy_inference_component
    always creates a brand-new Model resource and swaps the Inference
    Component onto it -- even re-deploying the identical source forces
    SageMaker to reload the model container from scratch. Comparing
    against the live component's own source/version tags (as read back
    by describe_deployed_inference_component) lets a queued job skip
    that reload entirely when nothing would actually change."""
    deployed = describe_deployed_inference_component()
    return bool(
        deployed
        and deployed["status"] == "InService"
        and (deployed.get("current_copy_count") or 0) > 0
        and deployed["active_size"] == item["size"]
        and deployed["source"] == item["weights_label"]
    )


def _run_inference_queue_job(item: dict) -> None:
    item_id = item["id"]
    size = item["size"]
    _update_inference_queue_item(item_id, status="deploying", started_at=datetime.now(timezone.utc).isoformat())
    try:
        if _deployed_matches(item):
            print(f"Inference queue item {item_id}: {item['weights_label']} already live -- skipping redeploy.")
        else:
            if item["weights_kind"] == "registry":
                deploy_inference_component(size, model_package_arn=item["model_package_arn"])
            elif item["weights_key"]:
                deploy_inference_component(size, weights_s3_key=item["weights_key"])
            else:
                deploy_inference_component(size, weights_s3_prefix=item["weights_folder"])

            wait_for_inference_component_ready()

        _update_inference_queue_item(item_id, status="running")
        result = run_batch_inference(item["images_folder"], item["output_folder"], job=item)
        _update_inference_queue_item(
            item_id, status="done", completed_at=datetime.now(timezone.utc).isoformat(), result=result,
        )
    except Exception as exc:
        _update_inference_queue_item(
            item_id, status="failed", completed_at=datetime.now(timezone.utc).isoformat(), error=str(exc),
        )
        print(f"Inference queue item {item_id} failed: {exc}")


def _inference_worker_loop() -> None:
    """Runs on its own daemon thread, same reasoning as _eval_worker_loop
    -- each job blocks on a real deploy wait (up to ~10 minutes) plus one
    invoke_inference call per image."""
    global _inference_queue_started
    while _inference_queue_started:
        item = _next_queued_inference_item()
        if item is None:
            break
        _run_inference_queue_job(item)
    # Draining naturally (nothing left queued) is not the same as Stop --
    # reset the flag either way so a later Add doesn't leave "Start" stuck
    # disabled and "Stop" stuck enabled for a queue that's actually idle.
    _inference_queue_started = False
    # Flag cleared first: the teardown check reads it to decide whether
    # anything still needs the endpoint.
    delete_endpoint_if_idle()


def start_inference_queue() -> bool:
    """Returns False if there's nothing queued to start."""
    global _inference_queue_started, _inference_worker_thread
    if _next_queued_inference_item() is None:
        return False
    with _inference_worker_lock:
        _inference_queue_started = True
        if _inference_worker_thread is None or not _inference_worker_thread.is_alive():
            _inference_worker_thread = threading.Thread(target=_inference_worker_loop, daemon=True)
            _inference_worker_thread.start()
    return True


def stop_inference_queue() -> None:
    """Stops picking up further queued items once the current one (if
    any) finishes -- there's no clean way to abort a job already mid-run."""
    global _inference_queue_started
    _inference_queue_started = False
