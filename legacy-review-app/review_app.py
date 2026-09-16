"""Human-review + training-control web app — replaces Amazon A2I's
reviewer portal, which this account can't use (see infra/sagemaker.tf's
note). Same deploy pattern as the Grafana demo: a plain Python
http.server + boto3 container on a dedicated EC2 instance, no framework
needed for this size.

Two tabs:
  - Review Queue: flagged predictions awaiting approve/reject, plus real
    registry metrics (no fabricated accuracy numbers — this app only
    shows what's actually computable from live AWS state).
  - Scheduling & Automation: manual "train now" trigger, and a
    daily/weekly/monthly schedule editor that edits the retrain
    EventBridge rule live (Terraform's copy of that rule has
    lifecycle.ignore_changes on schedule_expression/state specifically
    so it never fights this UI over the live value — see
    infra/pipeline_schedule.tf).

Reads/writes DynamoDB + SageMaker + EventBridge directly via the
instance's IAM role. Flagged images are served through presigned S3 URLs
rather than proxied through this process.
"""

import html
import json
import os
import re
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, quote, unquote, urlparse

import boto3
import botocore.config
import botocore.exceptions

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
TABLE_NAME = os.environ["REVIEW_QUEUE_TABLE"]
BUCKET = os.environ["PIPELINE_BUCKET"]
RULE_NAME = os.environ.get("RETRAIN_RULE_NAME", "track2-yolo-weekly-retrain-schedule")
PIPELINE_NAME = os.environ.get("TRAINING_PIPELINE_NAME", "track2-yolo-training-pipeline")

# Where real training datasets actually live (confirmed via `aws s3 ls`)
# -- NOT a bare "datasets/" prefix, which nothing has ever been uploaded
# to. pipeline_definition.py's own TRAIN_DATA_S3_URI default points here
# too (at the escalation_dataset/ subfolder specifically).
DATASETS_S3_PREFIX = "data/track2-datasets/sagemaker_training/track2"
# Same folder pipeline_definition.py itself falls back to when
# TRAIN_DATA_S3_URI isn't overridden -- used to pre-select a real, valid
# dataset on first page load (see do_GET) instead of leaving the "Start
# Training" button disabled until the user manually browses for one.
DEFAULT_TRAIN_DATASET = f"{DATASETS_S3_PREFIX}/escalation_dataset"

# Kept in exact sync with model_training/training-container/train.py's
# own IMAGE_EXTENSIONS (this app doesn't import that file -- its own
# deployed source is just this one file, see scripts/deploy_review_app.sh)
# -- shown on the Help tab, not used for any download/counting logic
# here (that all happens inside the training container itself).
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# YOLO11 only, by explicit user decision -- YOLOv8/9/10/12 and YOLO26
# are all deliberately excluded, not just "not yet added" (YOLO12 has no
# registered baseline and is not wanted here regardless). Nano also
# dropped from every generation's size options -- by policy, not because
# those combinations are invalid Ultralytics releases. Must match
# model_training/src/config.py's GENERATION_SIZES exactly (this app
# doesn't import that module -- its own deployed source is just this one
# file, see scripts/deploy_review_app.sh -- so the two copies are kept in
# sync by convention).
GENERATION_SIZES = {
    "11": ("s", "m", "l", "x"),
}
GENERATION_LABELS = {"11": "YOLO11"}
SIZE_LABELS = {"s": "Small", "m": "Medium", "l": "Large", "x": "Extra-Large"}
YOLO_GENERATIONS = tuple(GENERATION_SIZES)

# The one shared, always-on real-time endpoint hosting all four sizes as
# separate Inference Components -- see
# model_training/src/deploy_inference_component.py, which this UI's
# "Deploy" button mirrors (same boto3 calls, triggered from a click
# instead of the CLI).
MULTI_SIZE_ENDPOINT_NAME = os.environ.get("MULTI_SIZE_ENDPOINT_NAME", "track2-yolo-multi-size-endpoint")
MULTI_SIZE_VARIANT_NAME = os.environ.get("MULTI_SIZE_VARIANT_NAME", "AllSizes")
SAGEMAKER_EXECUTION_ROLE_ARN = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN", "")
SERVING_IMAGE_URI = os.environ.get("SERVING_IMAGE_URI", "")
RUN_PIPELINE_CODEBUILD_PROJECT = os.environ.get("RUN_PIPELINE_CODEBUILD_PROJECT", "track2-yolo-run-pipeline")

# --- Training queue: in-memory only (module-level state, no DynamoDB or
# other persistence) -- losing the queue on a container restart is an
# accepted, known limitation. A daemon thread (_queue_poll_loop, started
# at the bottom of this file) advances it in the background since this
# server is a plain single-threaded HTTPServer with no cron/scheduler of
# its own. All access to the state below goes through _queue_lock, since
# that poll thread and incoming HTTP requests (Add to Queue / Start Queue /
# Remove / the immediate Start Training button, whose runs also occupy the
# same single GPU slot) touch it concurrently.
MAX_QUEUE_SIZE = 10
_queue_lock = threading.Lock()
_training_queue: list = []  # pending entries (dicts), FIFO -- run order
_queue_running: dict | None = None  # the entry the queue itself is currently executing, or None
_queue_started = False  # True once "Start Queue" is clicked; auto-clears once the queue drains
_queue_completed_count = 0  # how many entries this queue has finished starting (for the "X of Y" display)
_queue_last_error = ""  # human-readable reason the most recent auto-start attempt failed, if any

# Bridges the real gap between "a codebuild.start_build call succeeded"
# and "a new SageMaker Pipeline Execution actually exists for
# _get_training_progress to report on" -- run_pipeline.py has to check
# out source and run inside CodeBuild first, which takes real time,
# during which the Models tab would otherwise keep showing the PREVIOUS
# run's already-terminal status, making a fresh submission look like
# nothing happened. Also surfaces a genuine submission failure (the
# CodeBuild build itself failing, e.g. a bad source zip) as a real error
# instead of silence -- this happened for real once, see summary.md.
_pending_submission_lock = threading.Lock()
_pending_submission: dict | None = None  # {"build_id": str, "triggered_at": datetime} or None

COMPUTE_REQUIREMENTS = {
    "s": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 2048},
    "m": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 3072},
    "l": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 4096},
    "x": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 6144},
}
# The four real serving sizes (same keys as COMPUTE_REQUIREMENTS) -- the
# Model Registry panel shows one section per size, not just the "s" group
# alone, so m/l/x's own registered candidates aren't invisible.
REGISTRY_SIZES = tuple(COMPUTE_REQUIREMENTS)

# boto3 clients (unlike resources, see below) are documented thread-safe
# -- fine to create once and share across every request-handling thread.
s3 = boto3.client("s3", region_name=REGION)
codebuild = boto3.client("codebuild", region_name=REGION)
events = boto3.client("events", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)
cloudwatch = boto3.client("cloudwatch", region_name=REGION)

# boto3 *resources* (what .Table() returns) are explicitly NOT
# documented as thread-safe, unlike clients above -- now that the
# server is a ThreadingHTTPServer (see __main__), request handlers run
# concurrently on real separate threads, so a single shared resource/
# Table instance is exactly the kind of thing that can misbehave under
# concurrency (credential-refresh races in particular). One
# boto3.resource per thread instead, via threading.local, so each
# thread gets its own -- table()/lineage_table()/automated_queues_table()
# below are the only way any code in this file touches DynamoDB.
_thread_local = threading.local()


def _dynamodb():
    if not hasattr(_thread_local, "dynamodb"):
        _thread_local.dynamodb = boto3.resource("dynamodb", region_name=REGION)
    return _thread_local.dynamodb


def table():
    return _dynamodb().Table(TABLE_NAME)


def lineage_table():
    # Pre-existing, reused table (hash key "id") -- see
    # pipeline_definition.py's DYNAMODB_LINEAGE_TABLE / _write_run_metadata
    # and register_baseline_model.py's _write_baseline_lineage, which
    # write here.
    return _dynamodb().Table("track2-dynamoDB")

# --- Automated Training Queues: persistent, individually-scheduled
# training recipes -- see terraform/infra/automated_queues.tf. Distinct
# from the ad-hoc, in-memory Training Queue above: these survive a
# container restart (stored in DynamoDB) and fire via a real
# per-queue EventBridge rule (so they still run even while the
# review-host EC2 instance is stopped overnight/on weekends -- see
# ec2_schedule.tf -- unlike anything driven by this process's own
# background thread).
AUTOMATED_QUEUES_TABLE_NAME = "track2-yolo-automated-queues"
AUTOMATED_QUEUE_LAMBDA_NAME = "track2-yolo-automated-queue-trigger"
AUTOMATED_QUEUE_RULE_PREFIX = "track2-yolo-auto-queue-"


def automated_queues_table():
    return _dynamodb().Table(AUTOMATED_QUEUES_TABLE_NAME)


# ARNs are fully predictable from region + account + resource name --
# built directly rather than via a Get/Describe call on each one, which
# would need its own IAM grant just to derive a string this process can
# already assemble from what it knows. The account id itself comes from
# one real STS call (not hardcoded anywhere in this file), made once at
# import time and reused for every ARN below.
ACCOUNT_ID = boto3.client("sts", region_name=REGION).get_caller_identity()["Account"]
AUTOMATED_QUEUE_LAMBDA_ARN = f"arn:aws:lambda:{REGION}:{ACCOUNT_ID}:function:{AUTOMATED_QUEUE_LAMBDA_NAME}"
# The one SNS topic every human-facing notification already goes
# through (training pipeline status, drift-triggered review needed,
# weights approved) -- see terraform/infra/monitoring.tf's
# aws_sns_topic.reviewer_notifications and every lambda_src/*.py that
# publishes to it. The Notifications tab manages who's subscribed here.
REVIEWER_NOTIFICATIONS_TOPIC_ARN = f"arn:aws:sns:{REGION}:{ACCOUNT_ID}:track2-yolo-reviewer-notifications"
sns = boto3.client("sns", region_name=REGION)
# sagemaker-runtime is a separate action namespace/client from
# sagemaker itself -- invoking a deployed model isn't a sagemaker:*
# action. Used by the Inference tab only. Extended read_timeout (default
# is 60s) and retries disabled: deploy_inference_component.py's own
# ContainerStartupHealthCheckTimeoutInSeconds=600 means a cold
# Inference Component (nothing invoked it in a while) can legitimately
# take minutes to actually start responding -- confirmed live, a real
# invocation took several minutes on a component DescribeInferenceComponent
# already reported as "InService". Without this, boto3's own default
# would raise a ReadTimeoutError and show a false "Inference failed"
# for a model that was only ever slow to wake up, not broken. Retries
# off (max_attempts=1) so a slow-but-eventually-successful cold start
# isn't given up on partway through and retried from zero.
sm_runtime = boto3.client(
    "sagemaker-runtime", region_name=REGION,
    config=botocore.config.Config(read_timeout=600, connect_timeout=10, retries={"max_attempts": 1}),
)

DAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"]
DAY_LABELS = {"SUN": "Sunday", "MON": "Monday", "TUE": "Tuesday", "WED": "Wednesday",
              "THU": "Thursday", "FRI": "Friday", "SAT": "Saturday"}


# --- helpers --------------------------------------------------------------

def _cron_and_summary_from_schedule_form(mode: str, form: dict) -> tuple[str, str]:
    """Shared by /train/schedule (the single global schedule) and every
    Automated Training Queue -- same Daily/Weekly/Monthly form fields,
    same SGT->UTC conversion (_sgt_to_utc_hour), turned into both a real
    EventBridge cron(...) expression and a human-readable summary to
    store/display without re-deriving it from the cron string every
    render. Raises ValueError for "manual" (no schedule to build) or an
    unrecognized mode -- callers that accept "manual" as meaningful
    (only /train/schedule does) check for it before calling this."""
    if mode == "daily":
        hour = int(form["daily_hour"][0])
        utc_hour, _ = _sgt_to_utc_hour(hour)
        return f"cron(0 {utc_hour} * * ? *)", f"Daily at {hour:02d}:00 SGT"
    if mode == "weekly":
        day = form["weekly_day"][0]
        hour = int(form["weekly_hour"][0])
        utc_hour, shift = _sgt_to_utc_hour(hour)
        cron_day = DAYS[(DAYS.index(day) - 1) % 7] if shift else day
        return f"cron(0 {utc_hour} ? * {cron_day} *)", f"Weekly on {DAY_LABELS[day]} at {hour:02d}:00 SGT"
    if mode == "monthly":
        dom = int(form["monthly_day"][0])
        hour = int(form["monthly_hour"][0])
        utc_hour, shift = _sgt_to_utc_hour(hour)
        cron_dom = (dom - 1 if dom > 1 else 28) if shift else dom
        return f"cron(0 {utc_hour} {cron_dom} * ? *)", f"Monthly on day {dom} at {hour:02d}:00 SGT"
    raise ValueError(f"Unknown schedule mode: {mode!r}")


def _sgt_to_utc_hour(local_hour: int) -> tuple:
    """Returns (utc_hour, day_shift) — day_shift is -1 if the SGT->UTC
    conversion (SGT is UTC+8, no DST) rolls back to the previous day."""
    utc_hour = local_hour - 8
    if utc_hour < 0:
        return utc_hour + 24, -1
    return utc_hour, 0


def _review_queue_counts():
    items = table().scan()["Items"]
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for item in items:
        status = item.get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
    return counts, items


def _fetch_lineage(lineage_id: str) -> dict | None:
    """DYNAMODB_LINEAGE_TABLE get_item -- see pipeline_definition.py's
    _write_run_metadata and register_baseline_model.py's
    _write_baseline_lineage, which write here at registration time under
    this exact id."""
    try:
        item = lineage_table().get_item(Key={"id": lineage_id}).get("Item")
        return dict(item) if item else None
    except Exception as exc:
        print(f"Could not get lineage {lineage_id}: {exc}")
        return None


def _scan_all_lineage() -> list:
    """Every record in DYNAMODB_LINEAGE_TABLE (training_run + baseline),
    newest first by timestamp_utc -- for the read-only Lineage tab. Unlike
    _fetch_lineage (a single get_item by id), this paginates scan() since
    the table can hold more than one page's worth of items."""
    items = []
    kwargs = {}
    while True:
        response = lineage_table().scan(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key
    items.sort(key=lambda item: str(item.get("timestamp_utc") or ""), reverse=True)
    return items


def _model_registry_summary(group_name: str):
    try:
        packages = sm.list_model_packages(ModelPackageGroupName=group_name,
                                           SortBy="CreationTime", SortOrder="Descending")["ModelPackageSummaryList"]
    except botocore.exceptions.ClientError as exc:
        # A size whose group hasn't been created yet (never trained/
        # registered for it) -- treat as "nothing registered" rather than
        # erroring the whole page out.
        if "does not exist" not in str(exc):
            raise
        packages = []
    approved = sum(1 for p in packages if p["ModelApprovalStatus"] == "Approved")
    # list_model_packages' summary has no description/metadata fields at
    # all -- DescribeModelPackage is the only way to get them
    # (pipeline_definition.py's RegisterModel step saves the training
    # form's "Notes" field as the description, and a "lineage_id" in
    # CustomerMetadataProperties pointing at the full record in
    # DYNAMODB_LINEAGE_TABLE). Only fetched for the packages
    # _render_review_tab actually displays (packages[:8]), not the whole
    # group's history -- both are per-package API calls.
    for pkg in packages[:8]:
        try:
            desc = sm.describe_model_package(ModelPackageName=pkg["ModelPackageArn"])
            pkg["Description"] = desc.get("ModelPackageDescription") or ""
            lineage_id = desc.get("CustomerMetadataProperties", {}).get("lineage_id")
            pkg["Lineage"] = _fetch_lineage(lineage_id) if lineage_id else None
        except Exception as exc:
            print(f"Could not describe {pkg['ModelPackageArn']}: {exc}")
            pkg["Description"] = ""
            pkg["Lineage"] = None
    return {"total": len(packages), "approved": approved, "packages": packages}


def _all_model_registries() -> dict:
    """One summary per real serving size (REGISTRY_SIZES), keyed by the
    group name so _render_review_tab can label each section separately."""
    return {f"yolo11{size}-detector": _model_registry_summary(f"yolo11{size}-detector") for size in REGISTRY_SIZES}


def _latest_approved_lookup(registries: dict) -> dict:
    """{generation: {size: {"group": str, "version": str} | None}} -- for
    every generation+size the Start Training form offers, whichever
    Model Package the "No .pt file available -- use the registry's
    latest Approved package instead" checkbox would actually resolve to
    (mirrors pipeline_definition.py's own
    _resolve_latest_approved_base_model exactly: latest-by-CreationTime
    Approved package in that generation+size's group). None where the
    group has no Approved package at all -- training would fail with a
    clear error in that case, so the form says so up front instead of
    only after clicking Start Training. Built from `registries` (already
    fetched unconditionally on every page load, see _render_page) rather
    than issuing its own SageMaker calls."""
    result = {}
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


def _fetch_metrics_and_lineage(model_package_arn: str) -> tuple:
    """One describe_model_package call, returning everything the Metrics
    tab needs per version: (metrics dict-or-None, description str,
    lineage dict-or-None). Combines what used to be two separate fetches
    (evaluation metrics here, description/lineage only ever fetched by
    _model_registry_summary for the Review tab's first 8 packages per
    group) into one, since the Metrics tab shows every version, not just
    8 -- reusing _model_registry_summary's own describe calls isn't an
    option here, so this avoids doubling describe_model_package per
    package instead.

    Metrics: the 5 flat numbers (mAP50, mAP50-95, precision, recall,
    accuracy) from the evaluation.json recorded on ModelMetrics at
    registration time -- see pipeline_definition.py's
    ModelMetrics(model_statistics=...) and evaluate.py's json.dump of
    exactly that shape. Confirmed live: the S3 URI sits at
    ModelMetrics.ModelQuality.Statistics.S3Uri. Baseline registrations
    (register_baseline_model.py) attach no ModelMetrics at all -- None
    for those.

    Description: whatever was typed into the Models tab's "Notes" field
    at training time (pipeline_definition.py saves it as this package's
    own ModelPackageDescription).

    Lineage: DYNAMODB_LINEAGE_TABLE record via the "lineage_id" cross-
    linked onto CustomerMetadataProperties at registration -- the same
    dataset/base-model/hyperparameters shown on the Lineage tab, for
    this exact version.

    Any failure (missing metrics, no lineage_id, a fetch error) leaves
    that one piece None/"" rather than raising, so a row with partial
    data still renders instead of crashing the whole tab."""
    try:
        desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    except Exception as exc:
        print(f"Could not describe model package {model_package_arn}: {exc}")
        return None, "", None

    description = desc.get("ModelPackageDescription") or ""
    lineage_id = desc.get("CustomerMetadataProperties", {}).get("lineage_id")
    lineage = _fetch_lineage(lineage_id) if lineage_id else None

    metrics = None
    s3_uri = desc.get("ModelMetrics", {}).get("ModelQuality", {}).get("Statistics", {}).get("S3Uri")
    if s3_uri and s3_uri.startswith("s3://"):
        try:
            bucket, _, key = s3_uri[len("s3://"):].partition("/")
            # pipeline_definition.py builds this URI as
            # Join(on="/", values=[evaluation_s3_uri, "evaluation.json"])
            # where evaluation_s3_uri already ends in "/" -- confirmed
            # live, the S3Uri that comes back is literally
            # ".../evaluation-output//evaluation.json" (double slash)
            # while the real object sits at the single-slash key. S3
            # keys aren't URL-normalized, so the double slash 404s
            # verbatim -- collapse repeated slashes before requesting it.
            key = re.sub(r"/+", "/", key)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            data = json.loads(body)
            metrics = {
                "mAP50": data.get("mAP50"),
                "mAP50-95": data.get("mAP50-95"),
                "precision": data.get("precision"),
                "recall": data.get("recall"),
                "accuracy": data.get("accuracy"),
            }
        except Exception as exc:
            print(f"Could not fetch evaluation metrics for {model_package_arn}: {exc}")

    return metrics, description, lineage


def _scan_all_model_metrics() -> list:
    """One entry per registered model package version across every YOLO11
    size group (REGISTRY_SIZES) -- for the read-only Metrics tab. Reuses
    _model_registry_summary's own package listing (already lists every
    version in the group via list_model_packages, not just the 8 the
    Review tab renders) rather than re-implementing that call here.
    Each row also carries the training notes (saved as this package's
    own description) and its full lineage record (dataset/base-model/
    hyperparameters at training time), so the Metrics tab can show what
    actually produced these numbers without a separate lookup."""
    rows = []
    for size in REGISTRY_SIZES:
        group_name = f"yolo11{size}-detector"
        summary = _model_registry_summary(group_name)
        for pkg in summary["packages"]:
            metrics, description, lineage = _fetch_metrics_and_lineage(pkg["ModelPackageArn"])
            rows.append({
                "group_name": group_name,
                "size": size,
                "version": pkg["ModelPackageArn"].rsplit("/", 1)[-1],
                "status": pkg["ModelApprovalStatus"],
                "metrics": metrics,
                "description": description,
                "lineage": lineage,
            })
    # Group order matches REGISTRY_SIZES (s, m, l, x -- the same order the
    # Review tab's own registry sections use); within a group, best
    # mAP50-95 first so comparing candidates for one size is a top-to-bottom
    # read. Rows with no metrics (baselines, fetch failures) sort last.
    size_order = {size: i for i, size in enumerate(REGISTRY_SIZES)}
    rows.sort(key=lambda r: (
        size_order[r["size"]],
        -(r["metrics"]["mAP50-95"] if r["metrics"] and isinstance(r["metrics"].get("mAP50-95"), (int, float)) else -1),
    ))
    return rows


def _list_s3_folders(prefix: str) -> list:
    """"Folders" directly under prefix (S3 has none really -- this is
    list_objects_v2's own CommonPrefixes emulation via Delimiter), each
    returned without the parent prefix, trailing slash stripped. Used for
    both the dataset picker and the weights picker."""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    return [p["Prefix"][len(prefix):].rstrip("/") for p in response.get("CommonPrefixes", [])]


def _list_s3_images(prefix: str) -> list:
    """Image files (matching IMAGE_EXTENSIONS) directly under prefix --
    sibling to _list_s3_folders, for the Inference tab's image picker
    (that one lists folders only; this tab needs to pick an actual file
    to run inference against, not a folder)."""
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


def _resolve_base_model_s3_uri(folder: str) -> str:
    """folder is whatever the weights S3 browser's "Use this folder"
    link selected -- resolves it to the exact object key train.py's
    --base-model-s3-uri expects (it accepts either suffix directly, see
    train.py's resolve_base_model_checkpoint_from_s3)."""
    for candidate in ("model.tar.gz", "model.pt"):
        key = f"{folder}/{candidate}"
        try:
            s3.head_object(Bucket=BUCKET, Key=key)
            return f"s3://{BUCKET}/{key}"
        except botocore.exceptions.ClientError:
            continue
    raise FileNotFoundError(f"Neither model.tar.gz nor model.pt found under s3://{BUCKET}/{folder}/")


def _repackage_raw_checkpoint_if_needed(weights_s3_prefix: str) -> str:
    """weights_s3_prefix (e.g. "weights/m/2026-09-07") points at a folder
    under BUCKET holding either an already-packaged model.tar.gz (used
    as-is) or a bare model.pt (tarred up under the exact filename
    inference.py's model_fn expects). Returns the resulting model.tar.gz's
    S3 key. Mirrors model_training/src/deploy_inference_component.py's
    same-named function -- duplicated rather than imported, since this
    app's deployed source is just this one file (see
    scripts/deploy_review_app.sh), not the model_training/ tree."""
    prefix = weights_s3_prefix.rstrip("/") + "/"
    listing = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    keys = {obj["Key"].rsplit("/", 1)[-1] for obj in listing.get("Contents", [])}

    if "model.tar.gz" in keys:
        return f"{prefix}model.tar.gz"
    if "model.pt" not in keys:
        raise FileNotFoundError(f"s3://{BUCKET}/{prefix} has neither model.tar.gz nor model.pt.")

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = os.path.join(tmp, "model.pt")
        s3.download_file(BUCKET, f"{prefix}model.pt", checkpoint_path)
        tarball_path = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(checkpoint_path, arcname="model.pt")
        dest_key = f"{prefix}model.tar.gz"
        s3.upload_file(tarball_path, BUCKET, dest_key)
    return dest_key


def _export_weights_to_folder(model_package_arn: str, destination_folder: str) -> tuple:
    """Downloads and extracts model.pt from a registered package's own
    model.tar.gz, and uploads it to <destination_folder>/selected.pt --
    every exported .pt is always named "selected.pt" (a fixed convention
    for this tool, not the model package's own version number), so
    whatever's currently at that name in a folder is always "the
    selected weights" for whoever reads from it next. Same download-
    extract-upload pattern as terraform/infra/lambda_src/
    dump_approved_weights.py's own _extract_and_upload_new_weights,
    generalized to any destination the user picks instead of one fixed
    per-size path.

    If selected.pt already exists there, it's archived first (a real S3
    move: server-side copy to <destination_folder>/archived/
    selected_<timestamp>.pt, then overwritten at the original key by the
    new upload below -- same "copy old aside, then overwrite" pattern
    that Lambda already uses) rather than silently clobbered. Returns
    (new_key, archived_key_or_None)."""
    destination_folder = destination_folder.strip("/")
    if not destination_folder:
        raise ValueError("No destination folder selected.")
    desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    if desc["ModelApprovalStatus"] != "Approved":
        raise RuntimeError(f"{model_package_arn} is {desc['ModelApprovalStatus']!r}, not Approved.")
    model_data_url = desc["InferenceSpecification"]["Containers"][0]["ModelDataUrl"]
    bucket, _, key = model_data_url[len("s3://"):].partition("/")

    new_key = f"{destination_folder}/selected.pt"
    archived_key = None
    try:
        s3.head_object(Bucket=BUCKET, Key=new_key)
        exists = True
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
        exists = False
    if exists:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
        archived_key = f"{destination_folder}/archived/selected_{timestamp}.pt"
        # Source is BUCKET (the pipeline bucket where selected.pt itself
        # lives), NOT `bucket` (parsed from the model package's own
        # ModelDataUrl a few lines up) -- they're the same bucket in
        # this project today, but conceptually distinct: this copy
        # archives the EXISTING file already sitting at new_key, nothing
        # to do with where the new model's tarball is coming from.
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


def _deploy_inference_component(size: str, *, model_package_arn: str = None, weights_s3_prefix: str = None) -> str:
    """Creates a Model from whichever source was given and creates/updates
    that size's Inference Component to point at it. Returns a short
    description of what was deployed, for the flash message."""
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
        # enabled." Resolving the package's own Image/ModelDataUrl and
        # passing those directly sidesteps the restriction.
        container = desc["InferenceSpecification"]["Containers"][0]
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": container["Image"], "ModelDataUrl": container["ModelDataUrl"]},
            ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
        )
        source_desc = model_package_arn.rsplit("/", 1)[-1]
    else:
        artifact_key = _repackage_raw_checkpoint_if_needed(weights_s3_prefix)
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": SERVING_IMAGE_URI, "ModelDataUrl": f"s3://{BUCKET}/{artifact_key}"},
            ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
        )
        source_desc = weights_s3_prefix

    inference_component_name = f"track2-yolo-{size}"
    specification = {
        "ModelName": model_name,
        "ComputeResourceRequirements": COMPUTE_REQUIREMENTS[size],
        "StartupParameters": {
            "ModelDataDownloadTimeoutInSeconds": 600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        },
    }

    try:
        sm.describe_inference_component(InferenceComponentName=inference_component_name)
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
        sm.update_inference_component(InferenceComponentName=inference_component_name, Specification=specification)
    else:
        sm.create_inference_component(
            EndpointName=MULTI_SIZE_ENDPOINT_NAME,
            VariantName=MULTI_SIZE_VARIANT_NAME,
            InferenceComponentName=inference_component_name,
            Specification=specification,
            RuntimeConfig={"CopyCount": 1},
        )

    return f"{size} <- {source_desc}"


def _describe_deployed_inference_component(size: str) -> dict | None:
    """Current live Inference Component for one size, or None if it's
    never been deployed -- same not-found detection
    _deploy_inference_component itself already has to work around
    (describe_inference_component raises ValidationException for a
    not-yet-created component, not the more obvious ResourceNotFound,
    confirmed live)."""
    name = f"track2-yolo-{size}"
    try:
        desc = sm.describe_inference_component(InferenceComponentName=name)
    except sm.exceptions.ResourceNotFound:
        return None
    except botocore.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find" in str(exc):
            return None
        raise
    spec = desc.get("Specification", {})
    return {
        "status": desc.get("InferenceComponentStatus"),
        "model_name": spec.get("ModelName"),
        "last_modified": desc.get("LastModifiedTime"),
    }


def _invoke_inference(size: str, image_bytes: bytes) -> dict:
    """Real-time prediction from whatever's currently deployed to
    track2-yolo-<size> on the shared endpoint -- see inference.py's own
    model_fn/input_fn/predict_fn/output_fn contract: fixed
    ContentType="image/jpeg" (input_fn rejects anything else by name,
    though PIL's own format-sniffing decodes the actual bytes regardless
    of what content type they're declared as, so a .png/.webp source
    file still works despite this literal string), returns
    {"detections": [{"confidence": float, "class": int, "box": [x1, y1,
    x2, y2]}]} with box coordinates already normalized 0.0-1.0 --
    convenient for an SVG overlay with viewBox="0 0 1 1", no pixel math
    needed. Raises on any failure (component not deployed, endpoint
    cold-starting, malformed image) -- caller renders the error inline
    rather than crashing the tab."""
    response = sm_runtime.invoke_endpoint(
        EndpointName=MULTI_SIZE_ENDPOINT_NAME,
        InferenceComponentName=f"track2-yolo-{size}",
        ContentType="image/jpeg",
        Body=image_bytes,
    )
    return json.loads(response["Body"].read())


def _render_deploy_tab(registries: dict) -> str:
    """One panel per real serving size -- shows what's currently live on
    the shared Inference Component endpoint, and two ways to deploy a
    replacement: pick an Approved registry version (the common path), or
    (collapsed by default, since it bypasses the registry entirely) a
    raw S3 weights folder -- both routes already existed
    (/models/<size>/deploy, /models/<size>/deploy-s3) and both call
    _deploy_inference_component; this tab was the missing piece that
    actually let a person reach them, no new backend needed."""
    sections = ""
    for size in REGISTRY_SIZES:
        group_name = f"yolo11{size}-detector"
        packages = registries.get(group_name, {}).get("packages", [])
        approved = [p for p in packages if p["ModelApprovalStatus"] == "Approved"]
        current = _describe_deployed_inference_component(size)
        current_html = (
            f'''<div class="card-meta" style="margin-bottom:10px">Currently deployed: '''
            f'''<span class="badge badge-ok">{html.escape(current["status"] or "?")}</span> '''
            f'''&middot; model <code>{html.escape(current["model_name"] or "?")}</code> '''
            f'''&middot; updated {current["last_modified"]}</div>'''
            if current else
            '<div class="card-meta" style="margin-bottom:10px">Not currently deployed.</div>'
        )
        if approved:
            options = "".join(
                f'<option value="{html.escape(p["ModelPackageArn"], quote=True)}">'
                f'v{p["ModelPackageArn"].rsplit("/", 1)[-1]} -- {html.escape((p.get("Description") or "no description")[:60])}'
                f'</option>'
                for p in approved
            )
            registry_form = f"""
            <form method="post" action="/models/{size}/deploy">
              <div class="field">
                <label>Approved version to deploy</label>
                <select name="model_package_arn" style="width:100%">{options}</select>
              </div>
              <button class="btn btn-approve" type="submit" style="margin-top:8px">Deploy Selected Model</button>
            </form>
            """
        else:
            registry_form = f'<p class="card-meta">No Approved versions in {html.escape(group_name)} yet -- approve one on the Review Queue tab first.</p>'
        sections += f"""
        <div class="panel">
          <div class="section-title">{html.escape(group_name)} ({SIZE_LABELS[size]})</div>
          {current_html}
          {registry_form}
          <details style="margin-top:12px">
            <summary class="card-meta" style="cursor:pointer">Deploy from a raw S3 folder instead (bypasses the registry)</summary>
            <form method="post" action="/models/{size}/deploy-s3" style="margin-top:8px">
              <div class="field">
                <label>Folder under s3://{BUCKET}/weights/{size}/</label>
                <input type="text" name="s3_folder" placeholder="e.g. 2026-09-08" style="width:100%">
              </div>
              <button class="btn" type="submit" style="margin-top:8px">Deploy From S3</button>
            </form>
          </details>
        </div>
        """
    return f"""
    <h1>Deploy</h1>
    <p class="card-meta" style="margin-bottom:16px">Deploys the selected model to that size's live Inference Component on the shared real-time endpoint (<code>{html.escape(MULTI_SIZE_ENDPOINT_NAME)}</code>) -- takes effect within a few minutes, no downtime for the other sizes.</p>
    {sections}
    """


def _render_image_picker(browse_prefix: str, selected_image: str, selected_size: str) -> str:
    """Folder navigation (mirrors _render_s3_browser's breadcrumb/
    subfolder pattern) plus the actual image files in the current
    folder -- _render_s3_browser itself can't be reused as-is here: it's
    hardcoded to tab=models' nav links and only ever selects a whole
    FOLDER, never one specific file. Every link carries selected_size
    forward too (not just browse_prefix/selected_image), so picking a
    different folder while browsing doesn't silently reset the size
    dropdown back to its default."""
    browse_prefix = browse_prefix.strip("/")
    folders = _list_s3_folders(browse_prefix)
    images = _list_s3_images(browse_prefix)
    size_qs = f"&infer_size={quote(selected_size)}" if selected_size else ""

    def _nav_link(prefix: str) -> str:
        qs = f"tab=inference&infer_browse_prefix={quote(prefix)}{size_qs}"
        if selected_image:
            qs += f"&infer_image_key={quote(selected_image)}"
        return f"/?{qs}"

    segments = [s for s in browse_prefix.split("/") if s]
    crumbs = [f'<a href="{_nav_link("")}">bucket root</a>']
    built = ""
    for seg in segments:
        built = f"{built}{seg}/"
        crumbs.append(f'<a href="{_nav_link(built.rstrip("/"))}">{html.escape(seg)}</a>')
    breadcrumb_html = " / ".join(crumbs)

    rows = ""
    for name in folders:
        sub_prefix = f"{browse_prefix}/{name}".strip("/")
        rows += f'<div class="registry-row"><a href="{_nav_link(sub_prefix)}">&#128193; {html.escape(name)}/</a></div>'
    for name in images:
        full_key = f"{browse_prefix}/{name}".strip("/")
        is_selected = full_key == selected_image
        select_href = f"/?tab=inference&infer_browse_prefix={quote(browse_prefix)}&infer_image_key={quote(full_key)}{size_qs}"
        rows += f"""
        <div class="registry-row">
          <span class="card-meta" style="color:var(--text)">&#128444; {html.escape(name)}</span>
          <a href="{select_href}" class="btn {'btn-approve' if is_selected else ''}" style="margin-left:auto;padding:4px 10px;font-size:11px;text-decoration:none;">
            {"Selected" if is_selected else "Use this image"}
          </a>
        </div>
        """
    if not rows:
        rows = '<div class="empty">No subfolders or images here.</div>'

    selected_html = (
        f'<span class="badge badge-ok">SELECTED</span> <code>s3://{BUCKET}/{html.escape(selected_image)}</code>'
        if selected_image else
        '<span class="badge badge-pending">NONE SELECTED YET</span>'
    )

    return f"""
    <div class="card-meta" style="margin-bottom:8px">{selected_html}</div>
    <div class="card-meta" style="margin-bottom:8px">{breadcrumb_html}</div>
    <div class="panel" style="padding:8px 12px">{rows}</div>
    """


def _render_inference_tab(browse_prefix: str, selected_image: str, selected_size: str, run: bool) -> str:
    """Pick a size with something currently deployed, pick a real image
    from S3, and actually invoke the live endpoint against it -- the
    Deploy tab lets you put a model into service; this is the tab that
    lets you try it. Runs the invocation synchronously on GET (same
    "real computation triggered by a plain link/GET form" pattern the
    Metrics tab's Compare Selected already uses), not a POST, since
    nothing gets stored -- every result is a fresh invoke_endpoint call,
    nothing cached."""
    deployed_sizes = {size: _describe_deployed_inference_component(size) for size in REGISTRY_SIZES}
    if not selected_size:
        selected_size = next((s for s, d in deployed_sizes.items() if d), REGISTRY_SIZES[0])

    size_options = ""
    for size in REGISTRY_SIZES:
        deployed = deployed_sizes[size]
        label = f"{SIZE_LABELS[size]} ({size}) -- {'deployed' if deployed else 'not deployed'}"
        size_options += (
            f'<option value="{size}" {"selected" if size == selected_size else ""} '
            f'{"disabled" if not deployed else ""}>{html.escape(label)}</option>'
        )

    picker_html = _render_image_picker(browse_prefix, selected_image, selected_size)
    can_run = bool(selected_image) and bool(deployed_sizes.get(selected_size))

    result_html = ""
    if run:
        if not selected_image:
            result_html = '<div class="banner banner-error">Pick an image first.</div>'
        elif not deployed_sizes.get(selected_size):
            result_html = f'<div class="banner banner-error">YOLO11-{html.escape(selected_size)} has nothing deployed -- deploy a model on the Deploy tab first.</div>'
        else:
            try:
                image_bytes = s3.get_object(Bucket=BUCKET, Key=selected_image)["Body"].read()
                prediction = _invoke_inference(selected_size, image_bytes)
                detections = prediction.get("detections", [])
                image_url = s3.generate_presigned_url(
                    "get_object", Params={"Bucket": BUCKET, "Key": selected_image}, ExpiresIn=300
                )
                boxes_svg = "".join(
                    f'<rect x="{d["box"][0]:.4f}" y="{d["box"][1]:.4f}" '
                    f'width="{d["box"][2] - d["box"][0]:.4f}" height="{d["box"][3] - d["box"][1]:.4f}" '
                    f'fill="none" stroke="#14e0b4" stroke-width="0.004" vector-effect="non-scaling-stroke">'
                    f'<title>class {d["class"]} -- {d["confidence"]:.2f}</title></rect>'
                    for d in detections
                )
                detection_lines = "".join(
                    f'<div class="card-meta" style="margin-top:2px">class {d["class"]} -- confidence {d["confidence"]:.3f} -- box [{", ".join(f"{v:.3f}" for v in d["box"])}]</div>'
                    for d in detections
                ) or '<div class="card-meta">No detections above the model\'s own confidence threshold.</div>'
                result_html = f"""
                <div class="panel">
                  <div class="section-title">Result ({len(detections)} detection{"s" if len(detections) != 1 else ""})</div>
                  <div style="position:relative;display:inline-block;max-width:100%">
                    <img src="{image_url}" style="max-width:100%;display:block">
                    <svg viewBox="0 0 1 1" preserveAspectRatio="none" style="position:absolute;top:0;left:0;width:100%;height:100%">{boxes_svg}</svg>
                  </div>
                  <div style="margin-top:10px">{detection_lines}</div>
                </div>
                """
            except Exception as exc:
                result_html = f'<div class="banner banner-error">Inference failed: {html.escape(str(exc))}</div>'

    return f"""
    <h1>Inference</h1>
    <p class="card-meta" style="margin-bottom:16px">Runs a real prediction against whatever's currently deployed to a size's Inference Component -- only sizes with something actually deployed (see the Deploy tab) can be tested.</p>

    <div class="panel">
      <div class="section-title">Pick a Size and Image</div>
      <form method="get" action="/">
        <input type="hidden" name="tab" value="inference">
        <input type="hidden" name="infer_browse_prefix" value="{html.escape(browse_prefix)}">
        <input type="hidden" name="infer_image_key" value="{html.escape(selected_image)}">
        <div class="field">
          <label>Size</label>
          <select name="infer_size" style="width:100%">{size_options}</select>
        </div>
        <button class="btn btn-approve" type="submit" name="infer_run" value="1" style="margin-top:10px" {"disabled" if not can_run else ""}>Run Inference</button>
      </form>
    </div>

    <div class="panel">
      <div class="section-title">Browse for an Image</div>
      {picker_html}
    </div>

    {result_html}
    """


def _list_notification_subscriptions() -> list:
    """Every subscription (any protocol, though this tab only ever
    creates "email" ones) on the one SNS topic every human-facing
    notification already goes through -- paginated the same way
    _scan_all_lineage is, since list_subscriptions_by_topic can return
    more than one page. A subscription's own SubscriptionArn is either a
    real ARN (confirmed -- the recipient clicked the link SNS emailed
    them) or the literal string "PendingConfirmation" (not yet
    confirmed) -- SNS itself uses that placeholder, not a real ARN, and
    it can't be used to unsubscribe (there's no API to cancel a pending
    request; per AWS's own docs it just expires after 3 days unconfirmed)."""
    subs = []
    kwargs = {"TopicArn": REVIEWER_NOTIFICATIONS_TOPIC_ARN}
    while True:
        response = sns.list_subscriptions_by_topic(**kwargs)
        subs.extend(response.get("Subscriptions", []))
        token = response.get("NextToken")
        if not token:
            break
        kwargs["NextToken"] = token
    return subs


def _render_notifications_tab(subscriptions: list) -> str:
    """Add/remove email recipients for the one SNS topic behind every
    status notification this stack sends (training pipeline start/
    success/failure, drift-triggered review needed, weights approved --
    see REVIEWER_NOTIFICATIONS_TOPIC_ARN's own comment). All of those
    currently share this single topic, so there's no per-category
    routing here -- subscribing means receiving all of them; splitting
    that apart would be a separate, larger change to the Lambdas that
    publish to it, not something this tab can do on its own."""
    email_subs = [s for s in subscriptions if s.get("Protocol") == "email"]
    rows = ""
    for s in email_subs:
        endpoint = html.escape(s.get("Endpoint", "?"))
        arn = s.get("SubscriptionArn", "")
        confirmed = arn and arn != "PendingConfirmation" and arn != "Deleted"
        if confirmed:
            status_html = '<span class="badge badge-ok">CONFIRMED</span>'
            remove_html = f"""
            <form method="post" action="/notifications/remove" style="display:inline"
                  onsubmit="return confirm('Remove {endpoint} from training/review notifications?');">
              <input type="hidden" name="subscription_arn" value="{html.escape(arn, quote=True)}">
              <button class="btn btn-reject" type="submit">Remove</button>
            </form>
            """
        else:
            status_html = '<span class="badge badge-pending">PENDING CONFIRMATION</span>'
            remove_html = '<span class="card-meta">Awaiting their click on the email SNS sent -- expires in 3 days if never confirmed; can\'t be cancelled early via API.</span>'
        rows += f"""
        <div class="registry-row" style="display:block">
          <div style="display:flex;align-items:center;gap:14px;">
            <span class="card-meta" style="font-size:13px;color:var(--text)">{endpoint}</span>
            {status_html}
            <div style="margin-left:auto">{remove_html}</div>
          </div>
        </div>
        """
    if not rows:
        rows = '<div class="empty">No one is subscribed to notifications yet.</div>'

    return f"""
    <h1>Notifications</h1>
    <p class="card-meta" style="margin-bottom:16px">Everyone below receives an email whenever this stack's training pipeline starts/succeeds/fails, a data-drift review is needed, or a model's weights are approved -- all of these currently share one SNS topic, so adding someone here subscribes them to all of it.</p>

    <div class="panel">
      <div class="section-title">Add a Recipient</div>
      <form method="post" action="/notifications/add">
        <div class="field">
          <label>Email address</label>
          <input type="email" name="email" placeholder="name@example.com" required style="width:100%">
        </div>
        <button class="btn btn-approve" type="submit" style="margin-top:8px">Subscribe</button>
        <p class="card-meta" style="margin-top:6px">They'll get one email from AWS asking them to confirm before any notifications actually reach them.</p>
      </form>
    </div>

    <div class="section-title">Subscribed ({len(email_subs)})</div>
    <div class="panel">{rows}</div>
    """


def _render_help_tab() -> str:
    """Purely static/informational -- no AWS calls, unlike every other
    tab -- describing exactly what train.py (baked into the training
    container) and pipeline_definition.py actually require of a dataset
    folder, taken directly from their real code rather than a generic
    "how YOLO datasets work" writeup: download_dataset_with_progress's
    own hard requirement for data.yaml at the root, IMAGE_EXTENSIONS for
    what counts as an image, and seed_training_data.py's own real,
    working example of the exact shape it writes (the same shape every
    real dataset this pipeline has ever trained from uses)."""
    image_extensions = ", ".join(sorted(e.lstrip(".") for e in IMAGE_EXTENSIONS))
    return f"""
    <h1>Help -- Dataset Format</h1>
    <p class="card-meta" style="margin-bottom:16px">Exactly what the "Dataset folder" picker on the Models tab needs to point at for training to actually work -- taken directly from this pipeline's own code, not a generic YOLO writeup.</p>

    <div class="panel">
      <div class="section-title">Required folder layout</div>
      <p class="card-meta">The dataset folder you pick (or type, for an Automated Queue) must contain a <code>data.yaml</code> file at its root -- training fails immediately with a clear error if it's missing. Everything else in the folder gets downloaded too, but only <code>data.yaml</code>'s own <code>train</code>/<code>val</code> paths matter for where images/labels actually need to live.</p>
      <pre style="background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.6;margin-top:8px">&lt;your dataset folder&gt;/
  data.yaml
  images/
    train/
      photo001.jpg
      photo002.png
      ...
    val/
      photo101.jpg
      ...
  labels/
    train/
      photo001.txt
      photo002.txt
      ...
    val/
      photo101.txt
      ...</pre>
      <p class="card-meta" style="margin-top:8px">The <code>images/train</code> + <code>labels/train</code> (and <code>val</code>) split above is convention, not a hard requirement -- <code>data.yaml</code>'s own <code>train:</code>/<code>val:</code> lines are what actually get read; name the folders whatever those two lines say.</p>
    </div>

    <div class="panel">
      <div class="section-title">data.yaml</div>
      <p class="card-meta">Paths are relative to the dataset folder itself. <code>names</code> lists every class, in the exact index order your label files use.</p>
      <pre style="background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.6;margin-top:8px">train: images/train
val: images/val
names:
  0: helmet
  1: gloves
  2: vehicle
  3: fall_zone</pre>
      <p class="card-meta" style="margin-top:8px">The class list above is this project's own hazard taxonomy (see <code>model_training/src/config.py</code>'s <code>CLASS_NAMES</code>) -- a different dataset can define its own <code>names</code> instead, as long as every label file's class index matches it.</p>
    </div>

    <div class="panel">
      <div class="section-title">Label files (.txt)</div>
      <p class="card-meta">One <code>.txt</code> file per image, same filename stem (e.g. <code>photo001.jpg</code> &rarr; <code>photo001.txt</code>), one line per object in that image:</p>
      <pre style="background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.6;margin-top:8px">class_id x_center y_center width height</pre>
      <p class="card-meta" style="margin-top:4px">All four numbers after the class id are fractions of the image's own width/height (0.0-1.0), not pixels. Example -- one helmet (class 0) covering the center half of the image:</p>
      <pre style="background:var(--panel-2);border:1px solid var(--border);border-radius:8px;padding:12px;overflow-x:auto;font-size:12.5px;line-height:1.6;margin-top:8px">0 0.5 0.5 0.5 0.5</pre>
      <p class="card-meta" style="margin-top:8px">An image with no objects at all just has an empty (or missing) <code>.txt</code> file -- not an error.</p>
    </div>

    <div class="panel">
      <div class="section-title">Supported image file types</div>
      <p class="card-meta">Case-insensitive, matched by file extension (see <code>train.py</code>'s <code>IMAGE_EXTENSIONS</code>) -- anything else in the folder (label <code>.txt</code> files, <code>data.yaml</code> itself, stray files) still downloads but isn't counted toward the "Loading images: N / M" progress bar on the Models tab:</p>
      <p class="card-meta" style="margin-top:6px"><code>{html.escape(image_extensions)}</code></p>
    </div>

    <div class="panel">
      <div class="section-title">Everything else this pipeline reads from your selections</div>
      <ul class="card-meta" style="margin:0;padding-left:18px;line-height:1.8">
        <li><b>Base model</b> -- either an Approved registry package (resolved automatically to that size's latest one if you leave the .pt picker blank), or a raw <code>.pt</code> checkpoint file in a folder you browse to.</li>
        <li><b>Epochs / Image Size / Batch Size / Seed</b> -- passed straight through to Ultralytics' own <code>model.train()</code>, no validation beyond basic range/type checks.</li>
        <li><b>Notes</b> -- free text, saved as the registered model package's own description; not read by training itself.</li>
      </ul>
    </div>
    """


def _render_export_weights_tab(registries: dict, browse_prefix: str, selected_folder: str) -> str:
    """Export an Approved model's raw .pt weights to any S3 folder --
    see _export_weights_to_folder for the actual archive-then-upload
    logic. One dropdown across every Approved package in every size
    group (not per-size panels like Deploy/Inference), since exporting
    is a single action against whichever one model you pick, not a
    per-size concept the way live serving is."""
    approved_options = ""
    for size in REGISTRY_SIZES:
        group_name = f"yolo11{size}-detector"
        for pkg in registries.get(group_name, {}).get("packages", []):
            if pkg["ModelApprovalStatus"] != "Approved":
                continue
            version = pkg["ModelPackageArn"].rsplit("/", 1)[-1]
            approved_options += (
                f'<option value="{html.escape(pkg["ModelPackageArn"], quote=True)}">'
                f'{html.escape(group_name)} v{version}</option>'
            )

    folder_browser_html = _render_s3_browser(
        browse_prefix, selected_folder, "export_browse_prefix", "export_selected_folder", "export-folder", tab="export"
    )

    if not approved_options:
        form_html = '<p class="card-meta">No Approved model packages exist yet -- approve one on the Review Queue tab first.</p>'
    else:
        button_label = (
            f"Export to s3://{BUCKET}/{html.escape(selected_folder)}/selected.pt"
            if selected_folder else "Pick a destination folder above first"
        )
        form_html = f"""
        <form method="post" action="/export-weights">
          <div class="field">
            <label>Model to export</label>
            <select name="model_package_arn" style="width:100%">{approved_options}</select>
          </div>
          <input type="hidden" name="destination_folder" value="{html.escape(selected_folder)}">
          <button class="btn btn-approve" type="submit" style="margin-top:10px" {"disabled" if not selected_folder else ""}>{button_label}</button>
        </form>
        """

    return f"""
    <h1>Export Weights</h1>
    <p class="card-meta" style="margin-bottom:16px">Exports an Approved model's raw <code>.pt</code> weights to any S3 folder you choose, always saved as <code>selected.pt</code>. If that folder already has one, it's archived first -- moved to <code>archived/selected_&lt;date&gt;.pt</code> in the same folder, labelled with when it was moved -- rather than silently overwritten.</p>

    <div class="panel">
      <div class="section-title">Destination Folder</div>
      {folder_browser_html}
    </div>

    <div class="panel">
      <div class="section-title">Export</div>
      {form_html}
    </div>
    """


# --- Training progress: 3-stage pipeline bar (Train -> Evaluate ->
# Register) plus a live epoch sub-bar while TrainYolo is running --------

def _get_epoch_progress(training_job_name: str):
    """{"current": int, "total": int} or None if the job's own "epochs"
    hyperparameter isn't available yet (job just started) or no
    train:epoch datapoint has landed yet (CloudWatch metrics lag the
    actual log line by roughly a minute)."""
    try:
        desc = sm.describe_training_job(TrainingJobName=training_job_name)
        total_epochs = int(desc["HyperParameters"].get("epochs", 0))
    except Exception as exc:
        print(f"Could not describe training job {training_job_name}: {exc}")
        return None
    if not total_epochs:
        return None

    current_epoch = 0
    try:
        now = datetime.now(timezone.utc)
        response = cloudwatch.get_metric_statistics(
            Namespace="/aws/sagemaker/TrainingJobs",
            MetricName="train:epoch",
            Dimensions=[{"Name": "TrainingJobName", "Value": training_job_name}],
            StartTime=now - timedelta(hours=6),
            EndTime=now,
            Period=60,
            Statistics=["Maximum"],
        )
        datapoints = response.get("Datapoints", [])
        if datapoints:
            current_epoch = int(max(dp["Maximum"] for dp in datapoints))
    except Exception as exc:
        print(f"Could not get train:epoch metric for {training_job_name}: {exc}")

    return {"current": current_epoch, "total": total_epochs}


def _get_metric_max(training_job_name: str, metric_name: str):
    """Latest Maximum datapoint for one CloudWatch metric under the
    standard training-job namespace, or None if nothing has landed yet.
    Shared by _get_epoch_progress-style helpers below (extracted since
    _get_dataset_images_progress needs two of these -- current count and
    its fixed total -- not just one)."""
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


def _get_dataset_images_progress(training_job_name: str):
    """{"current": int, "total": int} of images downloaded so far, or None
    if the total hasn't landed yet (dataset:total_images is printed once,
    right after train.py's 20s CloudWatch-metric-filter-attach delay --
    see download_dataset_with_progress). Mirrors _get_epoch_progress's
    shape/pattern (own current+total dict, None until real data exists)
    so _render_training_progress can treat both bars the same way."""
    total_images = _get_metric_max(training_job_name, "dataset:total_images")
    if not total_images:
        return None
    current = _get_metric_max(training_job_name, "dataset:images_downloaded") or 0
    return {"current": current, "total": total_images}


def _get_training_progress():
    """None if no execution has ever run. Otherwise the latest execution's
    overall status, each of its 3 steps' status, and (only while TrainYolo
    is Executing) the live epoch sub-progress."""
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
            # ModelStep (RegisterYoloModel in pipeline_definition.py)
            # expands into a real step named "RegisterYoloModel-RegisterModel",
            # not the plain name -- confirmed live, not assumed (a plain
            # exact-match here silently showed NotStarted forever, even
            # on a real Succeeded run). Prefix match tolerates this.
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
    # while TrainYolo itself is Executing -- both the CloudWatch
    # train:epoch metric and the job's own "epochs" hyperparameter
    # (_get_epoch_progress's two data sources) stay queryable long after
    # the training job finishes, so there's no reason the bar should
    # disappear the moment TrainYolo succeeds. Previously it vanished
    # entirely as soon as the pipeline moved on to EvaluateYolo.
    if training_job_name and step_status["TrainYolo"] in ("Executing", "Succeeded"):
        epoch_progress = _get_epoch_progress(training_job_name)
        # Dataset download happens before any epoch runs (see train.py's
        # main()) -- only worth showing while TrainYolo is still actually
        # Executing and no epoch has started yet; irrelevant once
        # training has already succeeded.
        if step_status["TrainYolo"] == "Executing" and not epoch_progress:
            download_progress = _get_dataset_images_progress(training_job_name)

    return {
        "execution_arn": execution_arn,
        "status": execution["PipelineExecutionStatus"],
        "start_time": execution.get("StartTime"),
        "steps": step_status,
        "epoch_progress": epoch_progress,
        "download_progress": download_progress,
    }


GPU_INSTANCE_TYPE = "ml.g4dn.xlarge"


def _get_gpu_resource_usage() -> list[dict]:
    """Every InProgress training job in this AWS account (not just this
    app's own pipeline) running on GPU_INSTANCE_TYPE -- the account has
    exactly 1 such instance available in total, shared across EVERY
    consumer in the account, not reserved for this app. A completely
    separate system (confirmed live: an unrelated "aidil-synthetic-data-
    yolo" experiment harness, see summary.md) can and does occupy that
    one slot, which _get_training_progress() alone can't see since it
    only looks at this app's own pipeline's executions -- this is what
    caused a real submission to fail with SAGEMAKER_RESOURCE_LIMIT even
    though this app's own "Latest Training Run" panel showed nothing
    running. Each returned dict: {"name", "start_time", "own_pipeline"}
    -- own_pipeline is True if the job name matches this app's own
    "pipelines-<execution-id>-TrainYolo-..." naming, so the UI can tell
    "busy with your own run" apart from "busy with something else"."""
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
            jobs.append({
                "name": name,
                "start_time": s.get("TrainingStartTime"),
                "own_pipeline": name.startswith("pipelines-") and "-TrainYolo-" in name,
            })
    return jobs


def _training_already_running_reason(progress) -> str | None:
    """None if it's safe to start a new run, otherwise a human-readable
    reason not to. The account has exactly 1 ml.g4dn.xlarge training
    instance available (see summary.md) -- starting a second pipeline
    execution while TrainYolo is still Executing on another one doesn't
    queue, it fails outright with SAGEMAKER_RESOURCE_LIMIT and no retry.
    Reuses _get_training_progress()'s own status field rather than a
    second query for the "this app's own pipeline" case, but ALSO checks
    account-wide usage (_get_gpu_resource_usage) -- a job from a
    completely different system occupying the account's one GPU slot
    fails a new submission exactly the same way, and progress alone
    can't see that."""
    if progress and progress["status"] == "Executing":
        run_id = progress["execution_arn"].rsplit("/", 1)[-1]
        return f"A training run ({run_id}) is already in progress. Wait for it to finish before starting another."
    with _pending_submission_lock:
        if _pending_submission:
            return "A training run is still being submitted. Wait for it to start before starting another."
    gpu_jobs = [j for j in _get_gpu_resource_usage() if not j["own_pipeline"]]
    if gpu_jobs:
        return (
            f"The account's only {GPU_INSTANCE_TYPE} GPU slot is in use by a job outside this app "
            f"({gpu_jobs[0]['name']}) -- starting a new run now would fail. Wait for it to finish."
        )
    return None


def _validate_training_params(epochs, dataset_folder, img_size=640, batch=16, seed=42) -> tuple[int, int, int, int]:
    """Shared by the immediate Start Training path (_start_training_run)
    and Add to Queue -- both need the exact same checks so a bad entry
    doesn't sit in the queue for potentially hours before failing when
    it's finally its turn to run. Returns the parsed (epochs, img_size,
    batch, seed) ints; raises ValueError with a human-readable message
    otherwise."""
    if not dataset_folder:
        raise ValueError("No dataset folder selected -- use the browser to pick one first.")
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


def _start_training_run(generation, size, epochs, dataset_folder, base_model_folder, notes, img_size=640, batch=16, seed=42) -> str:
    """Validates and submits one training run via CodeBuild -- the single
    place that builds environmentVariablesOverride and calls
    codebuild.start_build, so /train/dataset's immediate "Start Training"
    button and the training queue's background thread (_try_advance_queue)
    both go through here instead of duplicating this logic. Raises on
    validation failure or a CodeBuild error; returns the new build's id on
    success. Caller is responsible for the already-running (1 GPU slot)
    guard -- see _training_already_running_reason -- since the immediate
    button and the queue check it at different points (immediately vs. at
    each poll)."""
    epochs_int, img_size_int, batch_int, seed_int = _validate_training_params(epochs, dataset_folder, img_size, batch, seed)
    env_overrides = [
        {"name": "TRAIN_GENERATION", "value": generation, "type": "PLAINTEXT"},
        {"name": "TRAIN_SIZE", "value": size, "type": "PLAINTEXT"},
        {"name": "TRAIN_EPOCHS", "value": str(epochs_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_IMG_SIZE", "value": str(img_size_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_BATCH", "value": str(batch_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_SEED", "value": str(seed_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_DATA_S3_URI", "value": f"s3://{BUCKET}/{dataset_folder}/", "type": "PLAINTEXT"},
    ]
    if base_model_folder:
        base_model_s3_uri = _resolve_base_model_s3_uri(base_model_folder)
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
    return build_id


def _get_build_status(build_id: str) -> str | None:
    """CodeBuild's own buildStatus for one build (IN_PROGRESS/SUCCEEDED/
    FAILED/FAULT/STOPPED/TIMED_OUT), or None on any lookup failure."""
    try:
        builds = codebuild.batch_get_builds(ids=[build_id])["builds"]
        return builds[0]["buildStatus"] if builds else None
    except Exception as exc:
        print(f"Could not get CodeBuild build status for {build_id}: {exc}")
        return None


def _pending_submission_overlay(progress) -> str | None:
    """None normally. Otherwise a message _render_training_progress shows
    INSTEAD OF the normal panel, bridging the gap described where
    _pending_submission is declared above. Clears the tracked submission
    once either a newer execution shows up (the gap is over -- normal
    rendering resumes) or the CodeBuild build itself reaches a terminal
    failure state (retrying the check forever would just show the same
    stale error; the caller already saw it once that happens)."""
    global _pending_submission
    with _pending_submission_lock:
        pending = dict(_pending_submission) if _pending_submission else None
    if not pending:
        return None
    if progress and progress["start_time"] and progress["start_time"] >= pending["triggered_at"]:
        with _pending_submission_lock:
            _pending_submission = None
        return None
    status = _get_build_status(pending["build_id"])
    if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
        with _pending_submission_lock:
            _pending_submission = None
        return f"Training submission failed -- CodeBuild build {pending['build_id']} ended {status}. Check that build's own logs for why."
    if status == "SUCCEEDED":
        # CodeBuild itself finished (meaning run_pipeline.py returned 0 --
        # it did call start_pipeline_execution) but SageMaker's own
        # list_pipeline_executions hasn't reflected it on this poll yet.
        return "Finalizing new run..."
    return f"Submitting new run... (CodeBuild build {status or 'IN_PROGRESS'})"


def _queue_snapshot() -> dict:
    """Point-in-time copy of the queue state for rendering -- copied out
    under the lock so the HTML-building code below (which does not hold
    the lock) never reads a half-mutated list while the poll thread is
    also touching it."""
    with _queue_lock:
        return {
            "pending": [dict(e) for e in _training_queue],
            "running": dict(_queue_running) if _queue_running else None,
            "started": _queue_started,
            "completed_count": _queue_completed_count,
            "last_error": _queue_last_error,
        }


def _try_advance_queue():
    """Called both by the background poll thread (every ~25s) and
    immediately after "Start Queue" is clicked, so the first run doesn't
    have to wait for the next poll tick. If the queue has been started,
    nothing else is occupying the account's one ml.g4dn.xlarge slot (same
    _training_already_running_reason guard the immediate Start Training
    button and /train/now use), and there's at least one pending entry,
    pops it and starts it via _start_training_run."""
    global _queue_running, _queue_started, _queue_completed_count, _queue_last_error
    with _queue_lock:
        if not _queue_started:
            return
        reason = _training_already_running_reason(_get_training_progress())
        if reason:
            # Something is Executing -- either the queue's own current
            # entry (leave _queue_running as-is so the UI keeps showing
            # it) or an unrelated run started elsewhere (immediate Start
            # Training, or the Schedule tab's Train Now). Either way, not
            # safe to start the next one yet.
            return
        if _queue_running is not None:
            # The entry the queue itself started is no longer Executing
            # -- count it as finished (succeeded or failed; this queue
            # doesn't distinguish, see the module's Training Queue notes)
            # and free the slot for the next one.
            _queue_completed_count += 1
            _queue_running = None
        if not _training_queue:
            _queue_started = False
            return
        entry = _training_queue.pop(0)
        try:
            build_id = _start_training_run(
                entry["generation"], entry["size"], entry["epochs"],
                entry["dataset_folder"], entry["base_model_folder"], entry["notes"],
                entry["img_size"], entry["batch"], entry["seed"],
            )
            entry["build_id"] = build_id
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            _queue_running = entry
            _queue_last_error = ""
        except Exception as exc:
            # Don't put it back -- retrying the same bad entry forever
            # would just spin every poll. Drop it and let the next poll
            # try the next queued entry, if any.
            _queue_last_error = f"Queue entry (YOLO{entry['generation']}-{entry['size']}) failed to start, skipped: {exc}"
            print(_queue_last_error)


def _queue_poll_loop():
    """Runs for the life of the process on its own daemon thread (started
    before serve_forever() in __main__) -- the main thread is tied up
    inside HTTPServer.serve_forever()'s blocking request loop (this is a
    plain HTTPServer, not a ThreadingHTTPServer), so nothing else would
    ever advance the queue without a separate thread doing it."""
    while True:
        time.sleep(25)
        try:
            _try_advance_queue()
        except Exception as exc:
            print(f"Queue poll failed: {exc}")


def _render_gpu_resource_status(jobs: list) -> str:
    """Live, account-wide view of the shared ml.g4dn.xlarge GPU slot --
    distinct from "Latest Training Run" below, which only shows THIS
    app's own pipeline. This is the piece that would have shown the
    account's GPU busy with an unrelated system's job even while this
    app's own panel said "No training in progress" (see summary.md).
    Refreshed on every page load; this tab also auto-refreshes every 15s
    while a run/queue is active (see should_refresh in _render_page)."""
    if not jobs:
        return f"""
        <div class="panel">
          <div class="section-title">SageMaker GPU Resource</div>
          <div style="color:var(--info);font-weight:700;">Available</div>
          <div class="card-meta" style="margin-top:4px">No {html.escape(GPU_INSTANCE_TYPE)} training job is running anywhere in this AWS account right now.</div>
        </div>
        """
    lines = ""
    for j in jobs:
        origin = "this app's own pipeline" if j["own_pipeline"] else "a different system, not this app"
        lines += (
            f'<div class="card-meta" style="margin-top:4px">'
            f'<code>{html.escape(j["name"])}</code> &middot; {origin} &middot; started {j["start_time"]}'
            f'</div>'
        )
    return f"""
    <div class="panel">
      <div class="section-title">SageMaker GPU Resource</div>
      <div style="color:var(--reject);font-weight:700;">In Use</div>
      {lines}
    </div>
    """


def _render_training_progress(progress, pending_message: str | None = None) -> str:
    badge_class = {
        "Executing": "badge-pending", "Succeeded": "badge-ok",
        "Failed": "badge-reject", "Stopped": "badge-reject", "NotStarted": "",
    }

    if pending_message:
        return f"""
        <div class="panel">
          <div class="section-title">Latest Training Run</div>
          <div class="card-meta" style="color:var(--info);font-weight:600;">{html.escape(pending_message)}</div>
        </div>
        """

    # Nothing Executing right now -- either no execution has ever run, or
    # the last one already reached a terminal state. Either way, don't
    # keep showing that terminal run's own "Succeeded"/"Failed" badges as
    # if they were still the current, relevant thing on screen (this
    # looked exactly like "starting a new run did nothing" right after
    # clicking Start Training on top of an old finished run).
    if progress is None or progress["status"] != "Executing":
        last_run_html = ""
        if progress:
            last_run_html = f"""
            <div class="card-meta" style="margin-top:6px">
              Last run: <span class="badge {badge_class.get(progress["status"], '')}">{progress["status"]}</span>
              &middot; started {progress["start_time"]}
            </div>
            """
        return f"""
        <div class="panel">
          <div class="section-title">Latest Training Run</div>
          <div style="color:var(--info);font-weight:700;">No training in progress</div>
          {last_run_html}
        </div>
        """

    step_order = [("TrainYolo", "Train"), ("EvaluateYolo", "Evaluate"), ("RegisterYoloModel", "Register")]
    segments = ""
    for step_key, label in step_order:
        status = progress["steps"][step_key]
        segments += f"""
        <div class="progress-segment">
          <div class="progress-segment-label">{label}</div>
          <span class="badge {badge_class.get(status, '')}">{status}</span>
        </div>
        """

    epoch_html = ""
    ep = progress["epoch_progress"]
    if ep and ep["total"]:
        pct = min(100, round(100 * ep["current"] / ep["total"]))
        epoch_html = f"""
        <div class="card-meta" style="margin-top:10px">Epoch {ep["current"]} / {ep["total"]}</div>
        <div class="epoch-bar"><div class="epoch-bar-fill" style="width:{pct}%"></div></div>
        """
    elif progress["download_progress"] and progress["download_progress"]["total"]:
        dp = progress["download_progress"]
        pct = min(100, round(100 * dp["current"] / dp["total"]))
        epoch_html = f"""
        <div class="card-meta" style="margin-top:10px">Loading images: {dp["current"]} / {dp["total"]}</div>
        <div class="epoch-bar"><div class="epoch-bar-fill" style="width:{pct}%"></div></div>
        """

    return f"""
    <div class="panel">
      <div class="section-title">Latest Training Run</div>
      <div class="card-meta" style="margin-bottom:8px">
        Overall: <span class="badge {badge_class.get(progress["status"], '')}">{progress["status"]}</span>
        &middot; started {progress["start_time"]}
      </div>
      <div class="progress-stages">{segments}</div>
      {epoch_html}
    </div>
    """


def _current_schedule():
    try:
        rule = events.describe_rule(Name=RULE_NAME)
        return {"enabled": rule["State"] == "ENABLED", "expression": rule.get("ScheduleExpression", "")}
    except events.exceptions.ResourceNotFoundException:
        return {"enabled": False, "expression": ""}


def _rule_name_for_queue(queue_id: str) -> str:
    return f"{AUTOMATED_QUEUE_RULE_PREFIX}{queue_id}"


def _automated_queue_payload(queue: dict) -> str:
    """The exact JSON this queue's EventBridge rule hands the shared
    Lambda (automated_queue_trigger.py) as its fixed Input on every
    firing -- everything that Lambda needs to call codebuild.start_build
    with this queue's own training parameters, with base_model_folder
    already resolved to a concrete S3 URI (see _put_automated_queue)
    rather than re-resolved by the Lambda on each firing."""
    return json.dumps({
        "queue_id": queue["id"],
        "generation": queue["generation"],
        "size": queue["size"],
        "epochs": queue["epochs"],
        "img_size": queue["img_size"],
        "batch": queue["batch"],
        "seed": queue["seed"],
        "train_data_s3_uri": f"s3://{BUCKET}/{queue['dataset_folder']}/",
        "base_model_s3_uri": queue.get("base_model_s3_uri") or None,
        "notes": queue.get("notes") or None,
    })


def _put_automated_queue_rule(queue: dict) -> None:
    """Creates (or, called again on an edit, replaces) the one
    EventBridge rule backing this queue -- put_rule/put_targets are both
    idempotent upserts by name/id, so this is also how an edited queue's
    schedule and parameters get pushed to its existing rule, no separate
    "update" AWS call needed."""
    rule_name = _rule_name_for_queue(queue["id"])
    events.put_rule(Name=rule_name, ScheduleExpression=queue["schedule_expression"], State="ENABLED")
    events.put_targets(
        Rule=rule_name,
        Targets=[{"Id": "trigger", "Arn": AUTOMATED_QUEUE_LAMBDA_ARN, "Input": _automated_queue_payload(queue)}],
    )


def _delete_automated_queue_rule(queue_id: str) -> None:
    rule_name = _rule_name_for_queue(queue_id)
    try:
        events.remove_targets(Rule=rule_name, Ids=["trigger"])
    except events.exceptions.ResourceNotFoundException:
        pass
    try:
        events.delete_rule(Name=rule_name)
    except events.exceptions.ResourceNotFoundException:
        pass


def _scan_automated_queues() -> list:
    """Every Automated Training Queue, newest first -- same paginated-
    scan-then-sort pattern as _scan_all_lineage (this table is small,
    a handful of persistent recipes at most, so a full scan per page
    load is cheap)."""
    items = []
    kwargs = {}
    while True:
        response = automated_queues_table().scan(**kwargs)
        items.extend(response.get("Items", []))
        last_key = response.get("LastEvaluatedKey")
        if not last_key:
            break
        kwargs["ExclusiveStartKey"] = last_key
    items.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return items


# --- rendering --------------------------------------------------------------

def _render_page(
    active_tab: str,
    message: str = "",
    msg_is_error: bool = False,
    browse_prefix: str = "",
    selected_dataset: str = "",
    weights_browse_prefix: str = "",
    selected_weights: str = "",
    compare_keys: list = None,
    edit_auto_queue: str = "",
    infer_browse_prefix: str = "",
    infer_image_key: str = "",
    infer_size: str = "",
    infer_run: bool = False,
    export_browse_prefix: str = "",
    export_selected_folder: str = "",
) -> str:
    counts, _review_items = _review_queue_counts()
    registries = _all_model_registries()
    schedule = _current_schedule()

    message_html = (
        f'<div class="banner{" banner-error" if msg_is_error else ""}">{html.escape(message)}</div>' if message else ""
    )

    # Unlike the other two tabs' data, gathered only when active -- this
    # tab's data costs 4x list_model_packages + 4x
    # describe_inference_component + 2x list_objects_v2, no reason to pay
    # that on every load of the other two tabs.
    models_html = ""
    training_progress = None
    queue_snapshot = None
    pending_message = None
    if active_tab == "models":
        training_progress = _get_training_progress()
        pending_message = _pending_submission_overlay(training_progress)
        queue_snapshot = _queue_snapshot()
        models_html = _render_models_tab(
            browse_prefix,
            selected_dataset,
            weights_browse_prefix,
            selected_weights,
            training_progress,
            queue_snapshot,
            pending_message,
            registries,
            automated_queues=_scan_automated_queues(),
            editing_id=edit_auto_queue,
        )

    # Schedule tab's own "Run Now" button needs the same already-running
    # check as the Models tab's "Start Training" -- fetched separately
    # (not reusing the `training_progress` above) since it's only ever
    # non-None while active_tab == "models", same lazy-fetch reasoning.
    schedule_training_progress = None
    schedule_automated_queues = None
    if active_tab == "schedule":
        schedule_training_progress = _get_training_progress()
        schedule_automated_queues = _scan_automated_queues()

    # Also gathered only when active, same reasoning as the Models tab's
    # data above -- a full table scan on every load of the other tabs
    # would be wasted work.
    lineage_html = ""
    if active_tab == "lineage":
        lineage_html = _render_lineage_tab(_scan_all_lineage())

    # Also gathered only when active -- describes every registered package
    # version across all 4 size groups (1 describe_model_package + up to 1
    # s3.get_object each), no reason to pay that on every load of the
    # other tabs.
    metrics_html = ""
    if active_tab == "metrics":
        metrics_html = _render_metrics_tab(_scan_all_model_metrics(), compare_keys or [])

    # Also gathered only when active -- 1 describe_inference_component per
    # size (4 total), same lazy-fetch reasoning as the tabs above.
    # `registries` itself is already fetched unconditionally above (see
    # top of this function), reused here rather than re-fetched.
    deploy_html = ""
    if active_tab == "deploy":
        deploy_html = _render_deploy_tab(registries)

    # Also gathered only when active -- one SNS ListSubscriptionsByTopic
    # call, no reason to pay that on every load of the other tabs.
    notifications_html = ""
    if active_tab == "notifications":
        notifications_html = _render_notifications_tab(_list_notification_subscriptions())

    # Purely static content, no AWS calls -- still lazy-gated for
    # consistency with every other tab above, not because it's expensive.
    help_html = ""
    if active_tab == "help":
        help_html = _render_help_tab()

    # Also gathered only when active -- up to 4 describe_inference_component
    # calls plus (only when infer_run) a real invoke_endpoint + S3 fetch.
    inference_html = ""
    if active_tab == "inference":
        inference_html = _render_inference_tab(infer_browse_prefix, infer_image_key, infer_size, infer_run)

    # `registries` is already fetched unconditionally above -- reused
    # here rather than re-fetched, same as the Deploy tab.
    export_html = ""
    if active_tab == "export":
        export_html = _render_export_weights_tab(registries, export_browse_prefix, export_selected_folder)

    # Auto-refresh only while a run is actually in flight (or the queue is
    # armed and waiting for its turn), and only on the tab that shows it --
    # otherwise this would yank focus away from someone mid-edit on the
    # Review Queue or Schedule tabs for no reason.
    refresh_html = ""
    should_refresh = bool(pending_message) or (training_progress and training_progress["status"] == "Executing") or (
        queue_snapshot and (queue_snapshot["started"] or queue_snapshot["running"])
    )
    if should_refresh:
        # Plain "content=15" would reload "/" bare, losing ?tab=models
        # and whatever folder the S3 browsers were pointed at -- rebuild
        # the exact current URL instead.
        refresh_qs = f"tab=models&browse_prefix={quote(browse_prefix)}&weights_browse_prefix={quote(weights_browse_prefix)}"
        if selected_dataset:
            refresh_qs += f"&selected_dataset={quote(selected_dataset)}"
        if selected_weights:
            refresh_qs += f"&selected_weights={quote(selected_weights)}"
        refresh_html = f'<meta http-equiv="refresh" content="15;url=/?{refresh_qs}">'

    # Sidebar rail rather than the old horizontal tab strip, matching the
    # 8501 app's shell: ten destinations in a row wrapped badly and left no
    # room for names to grow. Preconnect + the display face are the same
    # ones that frontend loads; if the request is blocked the stack falls
    # back to a real monospace and nothing else changes.
    nav_links = "".join(
        f'<a href="/?tab={slug}" class="tab {"active" if active_tab == slug else ""}">{label}</a>'
        for slug, label in (
            # Slug stays "models" though the label reads Training: existing
            # links and bookmarks are all ?tab=models, and renaming the
            # label is not worth breaking them. Matches what the 8501 app
            # calls the same screen ("Track2 Training").
            ("models", "Training"),
            ("review", "Review Queue"),
            ("schedule", "Scheduling &amp; Automation"),
            ("lineage", "Lineage"),
            ("metrics", "Metrics"),
            ("deploy", "Deploy"),
            ("inference", "Inference"),
            ("notifications", "Notifications"),
            ("export", "Export Weights"),
            ("help", "Help"),
        )
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8">{refresh_html}<title>MLOps Control</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Special+Elite&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
<div class="layout">
<aside class="sidebar">
  <div class="brand">
    <span class="brand-name">MLOps Control</span>
    <span class="brand-sub">Jurong Port &middot; Track2</span>
  </div>
  <nav class="tabs">{nav_links}</nav>
</aside>
<main>
{message_html}
{_render_review_tab(counts, registries) if active_tab == 'review' else ''}
{models_html}
{_render_schedule_tab(schedule, schedule_training_progress, schedule_automated_queues, edit_auto_queue) if active_tab == 'schedule' else ''}
{lineage_html}
{metrics_html}
{deploy_html}
{notifications_html}
{help_html}
{inference_html}
{export_html}
</main>
</div>
</body></html>"""


def _format_lineage(lineage: dict | None) -> str:
    """Short, human-readable summary of a DYNAMODB_LINEAGE_TABLE record --
    different shape for a real training run vs. a baseline registration
    (see pipeline_definition.py's _write_run_metadata vs.
    register_baseline_model.py's _write_baseline_lineage)."""
    if not lineage:
        return ""
    if lineage.get("record_type") == "baseline":
        parts = [f"baseline checkpoint: {lineage.get('checkpoint', '?')}"]
        if lineage.get("approved_immediately"):
            parts.append("auto-approved on registration")
    else:
        dataset_uri = (lineage.get("dataset") or {}).get("s3_uri", "?")
        base_model = lineage.get("base_model") or {}
        base_desc = base_model.get("model_package_arn") or base_model.get("s3_uri") or "?"
        hp = lineage.get("hyperparameters") or {}
        parts = [
            f"dataset: {dataset_uri}",
            f"base model ({base_model.get('source', '?')}): {base_desc}",
            f"epochs={hp.get('epochs', '?')}, img_size={hp.get('img_size', '?')}, batch={hp.get('batch', '?')}, seed={hp.get('seed', '?')}",
        ]
    return html.escape(" | ".join(str(p) for p in parts))


def _render_registry_rows(packages: list) -> str:
    rows = ""
    for pkg in packages[:8]:
        status = pkg["ModelApprovalStatus"]
        badge_class = {"Approved": "badge-ok", "Rejected": "badge-reject", "PendingManualApproval": "badge-pending"}.get(status, "badge-pending")
        version = pkg["ModelPackageArn"].rsplit("/", 1)[-1]
        arn_enc = quote(pkg["ModelPackageArn"], safe="")
        description = html.escape(pkg.get("Description") or "")
        lineage_line = _format_lineage(pkg.get("Lineage"))
        actions = ""
        if status == "PendingManualApproval":
            actions = f"""
            <form method="post" action="/model/{arn_enc}/approve" style="display:inline">
              <button class="btn btn-approve" type="submit">Approve</button>
            </form>
            <form method="post" action="/model/{arn_enc}/reject" style="display:inline">
              <button class="btn btn-reject" type="submit">Reject</button>
            </form>
            """
        # Delete is offered regardless of status (including Approved) --
        # DeleteModelPackage itself doesn't care about approval status, and
        # cleaning up a bad/duplicate/test registration is a real need
        # whatever state it's in. Irreversible (no undo, unlike Approve/
        # Reject which just flip a status) -- the only destructive action
        # here, so it gets its own confirm() and reject-styled button,
        # unlike Approve/Reject above.
        actions += f"""
        <form method="post" action="/model/{arn_enc}/delete" style="display:inline"
              onsubmit="return confirm('Permanently delete v{version}? This cannot be undone.');">
          <button class="btn btn-reject" type="submit">Delete</button>
        </form>
        """
        rows += f"""
        <div class="registry-row" style="display:block">
          <div style="display:flex;align-items:center;gap:14px;">
            <span class="registry-version">v{version}</span>
            <span class="badge {badge_class}">{status.upper()}</span>
            <span class="registry-date" style="margin-left:auto">{pkg['CreationTime']}</span>
            {actions}
          </div>
          {f'<div class="card-meta" style="margin-top:4px">{description}</div>' if description else ''}
          {f'<div class="card-meta" style="margin-top:4px;opacity:0.8">{lineage_line}</div>' if lineage_line else ''}
        </div>
        """
    return rows or '<div class="empty">No models registered yet.</div>'


def _render_review_tab(counts, registries: dict) -> str:
    total_registered = sum(r["total"] for r in registries.values())
    total_approved = sum(r["approved"] for r in registries.values())
    stat_cards = f"""
    <div class="stat-grid">
      <div class="stat-card">
        <div class="stat-label">Pending Reviews</div>
        <div class="stat-value accent">{counts.get('pending', 0)}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Registered Model Versions (all sizes)</div>
        <div class="stat-value">{total_registered}</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Approved (all sizes)</div>
        <div class="stat-value ok">{total_approved}</div>
      </div>
    </div>
    """

    registry_sections = ""
    for group_name, summary in registries.items():
        registry_sections += f"""
        <div class="section-title">Model Registry — {group_name} ({summary['approved']} approved / {summary['total']} total)</div>
        <div class="panel">{_render_registry_rows(summary["packages"])}</div>
        """

    return f"""
    <h1>Review Queue</h1>
    {stat_cards}
    {registry_sections}
    """


def _render_schedule_picker_fields(mode: str, hour: int, day: str, dom: int) -> str:
    """Daily/Weekly/Monthly radio+select fields for one Automated
    Training Queue's schedule -- no "manual" option (an automated
    queue's whole purpose is having a real schedule; use the ad-hoc
    Training Queue or Start Training directly for a one-off run
    instead). Every option is pre-selected from the given values, not
    just the radios, so an existing queue's Edit form shows its actual
    saved schedule rather than always defaulting to the first option in
    each dropdown."""
    day_options = "".join(f'<option value="{d}" {"selected" if d == day else ""}>{DAY_LABELS[d]}</option>' for d in DAYS)
    hour_options = "".join(f'<option value="{h}" {"selected" if h == hour else ""}>{h:02d}:00 SGT</option>' for h in range(24))
    dom_options = "".join(f'<option value="{d}" {"selected" if d == dom else ""}>{d}</option>' for d in range(1, 29))
    return f"""
    <div class="freq-row">
      <label><input type="radio" name="mode" value="daily" {'checked' if mode == 'daily' else ''}> Daily</label>
      at <select name="daily_hour">{hour_options}</select>
    </div>
    <div class="freq-row">
      <label><input type="radio" name="mode" value="weekly" {'checked' if mode == 'weekly' else ''}> Weekly</label>
      on <select name="weekly_day">{day_options}</select>
      at <select name="weekly_hour">{hour_options}</select>
    </div>
    <div class="freq-row">
      <label><input type="radio" name="mode" value="monthly" {'checked' if mode == 'monthly' else ''}> Monthly</label>
      on day <select name="monthly_day">{dom_options}</select>
      at <select name="monthly_hour">{hour_options}</select>
    </div>
    """


def _render_automated_queues_panel(queues: list, editing_id: str | None, tab: str = "models") -> str:
    """The "Automated Training Queues" panel -- shared verbatim between
    the Models tab (where new queues are added, reusing the Start
    Training form's own generation/size/dataset/etc. fields via a third
    submit button) and the Scheduling & Automation tab (view/edit/delete
    only there -- adding a new one needs the dataset/weights S3 browsers,
    which only the Models tab's form already has; duplicating those here
    too was more than "configurable + deletable" asked for). Each
    queue's own EventBridge rule (see _put_automated_queue_rule) is what
    actually fires it -- these fire independent of whether the review
    app's own EC2 host happens to be running, unlike the ad-hoc Training
    Queue above."""
    if not queues:
        rows = '<div class="empty">No automated queues yet -- fill in the form above and click "Add to Automated Pipeline".</div>'
    else:
        rows = ""
        for q in queues:
            if q["id"] == editing_id:
                rows += f"""
                <div class="registry-row" style="display:block">
                  <form method="post" action="/automated-queue/update/{html.escape(q['id'], quote=True)}">
                    <input type="hidden" name="return_tab" value="{tab}">
                    <div class="field-grid">
                      <div class="field"><label>Generation</label>
                        <select name="generation">{"".join(f'<option value="{g}" {"selected" if g == q["generation"] else ""}>{GENERATION_LABELS[g]}</option>' for g in YOLO_GENERATIONS)}</select>
                      </div>
                      <div class="field"><label>Size</label>
                        <select name="size">{"".join(f'<option value="{s}" {"selected" if s == q["size"] else ""}>{SIZE_LABELS[s]} ({s})</option>' for s in GENERATION_SIZES.get(q["generation"], ()))}</select>
                      </div>
                      <div class="field"><label>Epochs</label><input type="number" name="epochs" value="{html.escape(str(q['epochs']))}" min="1" max="1000" required></div>
                      <div class="field"><label>Image Size</label><input type="number" name="img_size" value="{html.escape(str(q['img_size']))}" min="32" max="4096" step="32" required></div>
                      <div class="field"><label>Batch Size</label><input type="number" name="batch" value="{html.escape(str(q['batch']))}" min="1" max="512" required></div>
                      <div class="field"><label>Seed</label><input type="number" name="seed" value="{html.escape(str(q['seed']))}" required></div>
                    </div>
                    <div class="field">
                      <label>Dataset folder (bucket-relative path)</label>
                      <input type="text" name="dataset_folder" value="{html.escape(q['dataset_folder'])}" style="width:100%" required>
                    </div>
                    <div class="field">
                      <label>Base model folder (blank = registry's latest Approved package)</label>
                      <input type="text" name="base_model_folder" value="{html.escape(q.get('base_model_folder') or '')}" style="width:100%">
                    </div>
                    <div class="field">
                      <label>Notes</label>
                      <textarea name="notes" rows="2" style="width:100%;resize:vertical;font-family:inherit;">{html.escape(q.get('notes') or '')}</textarea>
                    </div>
                    <div class="section-title" style="margin-top:16px">Schedule</div>
                    {_render_schedule_picker_fields(q["schedule_mode"], q["schedule_hour"], q.get("schedule_day", "SUN"), q.get("schedule_dom", 1))}
                    <button class="btn btn-approve" type="submit" style="margin-top:10px">Save</button>
                    <a href="/?tab={tab}" class="btn" style="margin-top:10px;margin-left:8px;text-decoration:none;display:inline-block;">Cancel</a>
                  </form>
                </div>
                """
                continue
            gen = html.escape(q["generation"])
            size = html.escape(q["size"])
            queue_id = html.escape(q["id"], quote=True)
            base = html.escape(q["base_model_folder"]) if q.get("base_model_folder") else "registry's latest Approved package"
            notes_line = f" &middot; notes: {html.escape(q['notes'])}" if q.get("notes") else ""
            rows += f"""
            <div class="registry-row" style="display:block">
              <div style="display:flex;align-items:center;gap:14px;">
                <span class="badge badge-ok">{html.escape(q.get('schedule_summary', '?'))}</span>
                <span class="card-meta">YOLO{gen}-{size} &middot; {html.escape(str(q['epochs']))} epochs &middot; img {html.escape(str(q['img_size']))} &middot; batch {html.escape(str(q['batch']))} &middot; seed {html.escape(str(q['seed']))}</span>
                <a href="/?tab={tab}&edit_auto_queue={queue_id}" class="btn" style="margin-left:auto;text-decoration:none;">Configure</a>
                <form method="post" action="/automated-queue/delete/{queue_id}" onsubmit="return confirm('Delete this automated queue? Its schedule will stop firing immediately.');">
                  <input type="hidden" name="return_tab" value="{tab}">
                  <button class="btn btn-reject" type="submit">Delete</button>
                </form>
              </div>
              <div class="card-meta" style="margin-top:4px"><code>s3://{BUCKET}/{html.escape(q['dataset_folder'])}/</code></div>
              <div class="card-meta" style="margin-top:2px;opacity:0.8">base model: {base}{notes_line}</div>
            </div>
            """

    return f"""
    <div class="section-title">Automated Training Queues ({len(queues)})</div>
    <p class="card-meta" style="margin-bottom:12px">Each queue below fires on its own schedule via a dedicated EventBridge rule -- unlike the ad-hoc Training Queue above, these keep firing even while this review app's own EC2 host is stopped (it auto-stops nightly and over weekends).</p>
    <div class="panel">{rows}</div>
    """


def _render_queue_panel(queue_snapshot: dict) -> str:
    """The "Training Queue" panel -- lists up to MAX_QUEUE_SIZE pending
    entries in run order, shows which one the queue is currently running
    (if any), and offers Start Queue / Stop Queue. All entry fields are
    user-controlled (form input, ultimately from S3 folder names an
    attacker could also POST directly since this app has no auth) so
    every one of them is html.escape()'d before going into the page."""
    pending = queue_snapshot["pending"]
    running = queue_snapshot["running"]
    started = queue_snapshot["started"]
    completed_count = queue_snapshot["completed_count"]
    last_error = queue_snapshot["last_error"]
    total_in_batch = completed_count + (1 if running else 0) + len(pending)

    def _describe(entry: dict) -> tuple:
        gen = html.escape(str(entry["generation"]))
        size = html.escape(str(entry["size"]))
        epochs = html.escape(str(entry["epochs"]))
        img_size = html.escape(str(entry.get("img_size", 640)))
        batch = html.escape(str(entry.get("batch", 16)))
        seed = html.escape(str(entry.get("seed", 42)))
        dataset = html.escape(str(entry["dataset_folder"]))
        base = html.escape(str(entry["base_model_folder"])) if entry["base_model_folder"] else "registry's latest Approved package"
        notes = html.escape(str(entry["notes"])) if entry["notes"] else ""
        return gen, size, epochs, img_size, batch, seed, dataset, base, notes

    running_html = ""
    if running:
        gen, size, epochs, img_size, batch, seed, dataset, base, notes = _describe(running)
        position = completed_count + 1
        running_html = f"""
        <div class="card-meta" style="margin-bottom:10px">
          <span class="badge badge-pending">RUNNING {position} OF {total_in_batch}</span>
          &nbsp; YOLO{gen}-{size}, {epochs} epochs, img {img_size}, batch {batch}, seed {seed}, <code>s3://{BUCKET}/{dataset}/</code>
        </div>
        """
    elif started and pending:
        running_html = """
        <div class="card-meta" style="margin-bottom:10px">
          <span class="badge badge-pending">QUEUE STARTED</span>
          &nbsp; waiting for a free GPU slot to start the next run.
        </div>
        """

    rows = ""
    start_num = completed_count + (1 if running else 0) + 1
    for i, entry in enumerate(pending):
        gen, size, epochs, img_size, batch, seed, dataset, base, notes = _describe(entry)
        entry_id = html.escape(entry["id"], quote=True)
        notes_line = f" &middot; notes: {notes}" if notes else ""
        rows += f"""
        <div class="registry-row" style="display:block">
          <div style="display:flex;align-items:center;gap:14px;">
            <span class="registry-version">#{start_num + i}</span>
            <span class="card-meta">YOLO{gen}-{size} &middot; {epochs} epochs &middot; img {img_size} &middot; batch {batch} &middot; seed {seed}</span>
            <form method="post" action="/queue/remove/{entry_id}" style="margin-left:auto">
              <button class="btn btn-reject" type="submit">Remove</button>
            </form>
          </div>
          <div class="card-meta" style="margin-top:4px"><code>s3://{BUCKET}/{dataset}/</code></div>
          <div class="card-meta" style="margin-top:2px;opacity:0.8">base model: {base}{notes_line}</div>
        </div>
        """
    if not rows:
        rows = '<div class="empty">No runs queued.</div>'

    start_queue_disabled = started or (not pending and not running)
    error_html = (
        f'<p class="card-meta" style="margin-top:10px;color:var(--reject)">{html.escape(last_error)}</p>'
        if last_error else ""
    )
    stop_button_html = f"""
    <form method="post" action="/queue/stop" style="display:inline;margin-left:8px">
      <button class="btn btn-reject" type="submit">Stop Queue</button>
    </form>
    """ if started else ""

    return f"""
    <div class="panel">
      <div class="section-title">Training Queue ({len(pending)}/{MAX_QUEUE_SIZE} queued{' + 1 running' if running else ''})</div>
      <p class="card-meta" style="margin-bottom:12px">Runs execute one at a time (this account has a single GPU training slot) in the order queued below. "Start Queue" begins auto-advancing through them as each prior run finishes.</p>
      {running_html}
      <div style="margin-bottom:4px">{rows}</div>
      <div style="margin-top:12px">
        <form method="post" action="/queue/start" style="display:inline">
          <button class="btn btn-approve" type="submit" {"disabled" if start_queue_disabled else ""}>Start Queue</button>
        </form>
        {stop_button_html}
      </div>
      {error_html}
    </div>
    """


def _render_models_tab(browse_prefix, selected_dataset, weights_browse_prefix, selected_weights, training_progress, queue_snapshot=None, pending_message=None, registries=None, automated_queues=None, editing_id="") -> str:
    # Size options for whichever generation happens to be first in the
    # dropdown -- just the initial render; updateSizeOptions() below
    # rebuilds this from GENERATION_SIZES the moment the page loads and
    # on every Generation change, since the valid sizes genuinely differ
    # per generation (YOLOv9 has no l/x at all, YOLOv10 adds a "b").
    default_generation = YOLO_GENERATIONS[0]
    size_options = "".join(
        f'<option value="{s}">{SIZE_LABELS[s]} ({s})</option>' for s in GENERATION_SIZES[default_generation]
    )
    generation_sizes_json = json.dumps(GENERATION_SIZES)
    size_labels_json = json.dumps(SIZE_LABELS)
    latest_approved_json = json.dumps(_latest_approved_lookup(registries or {}))
    dataset_browser_html = _render_s3_browser(
        browse_prefix, selected_dataset, "browse_prefix", "selected_dataset", "dataset"
    )
    weights_browser_html = _render_s3_browser(
        weights_browse_prefix, selected_weights, "weights_browse_prefix", "selected_weights", "weights"
    )
    training_progress_html = _render_training_progress(training_progress, pending_message)
    gpu_status_html = _render_gpu_resource_status(_get_gpu_resource_usage())
    # Mirrors the server-side check in do_POST's /train/dataset and
    # /train/now handlers (_training_already_running_reason) -- this is
    # just the UX half (a clearer disabled button beats submitting and
    # getting bounced back with an error), the server-side check is what
    # actually prevents the SAGEMAKER_RESOURCE_LIMIT collision, since a
    # disabled attribute alone is trivially bypassable.
    already_running = _training_already_running_reason(training_progress)
    start_disabled = not selected_dataset or bool(already_running)

    # Add to Queue doesn't need a free GPU slot right now (that's the
    # whole point of queueing) -- only disabled when no dataset is picked
    # yet or the queue is already at its cap.
    queue_snapshot = queue_snapshot or _queue_snapshot()
    queue_full = len(queue_snapshot["pending"]) >= MAX_QUEUE_SIZE
    add_to_queue_disabled = not selected_dataset or queue_full

    return f"""
    <h1>Track2 Training</h1>
    <p class="page-sub">Dataset/weights browsing, GPU status, training progress, and the training queue for the Track2 YOLO pipeline.</p>

    <div class="grid-2">
      {gpu_status_html}
      {training_progress_html}
    </div>

    <div class="panel">
      <div class="section-title">Start training</div>
      <p class="card-meta" style="margin-bottom:14px">Registers the result into that generation/size's Model Package Group (created automatically if it doesn't exist yet).</p>
      <form method="post" action="/train/dataset">
       <div class="form-2col">
        <div>
          <!-- Generation is a hidden field, not a dropdown: GENERATION_SIZES
               has exactly one entry, so the select offered a single option
               and cost a whole row to say nothing. The size label carries
               the generation instead. Kept as an input (not removed) so the
               form still posts it and the size-options script, which reads
               .value off this id, keeps working untouched. -->
          <input type="hidden" name="generation" id="train-generation" value="{default_generation}">
          <div class="field">
            <label>Size ({GENERATION_LABELS[default_generation]})</label>
            <select name="size" id="train-size" required>{size_options}</select>
          </div>
          <div class="field-row-2">
            <div class="field">
              <label>Epochs</label>
              <input type="number" name="epochs" value="50" min="1" max="1000" required>
            </div>
            <div class="field">
              <label>Image size</label>
              <input type="number" name="img_size" value="640" min="32" max="4096" step="32" required>
            </div>
          </div>
          <div class="field-row-2">
            <div class="field">
              <label>Batch</label>
              <input type="number" name="batch" value="16" min="1" max="512" required>
            </div>
            <div class="field">
              <label>Seed</label>
              <input type="number" name="seed" value="42" required>
            </div>
          </div>
          <div class="field">
            <label>Notes</label>
            <textarea name="notes" rows="12" placeholder="e.g. re-running with the corrected data.yaml after last week's mislabeled class fix" style="width:100%;resize:vertical;font-family:inherit;"></textarea>
          </div>
        </div>

        <div>
        <div class="field">
          <label>Dataset folder (browse the whole bucket)</label>
          <input type="hidden" name="dataset_folder" value="{selected_dataset}">
          {dataset_browser_html}
        </div>
        <div class="field" style="margin-top:14px">
          <label>Base model .pt file (browse the whole bucket)</label>
          <input type="hidden" name="base_model_folder" id="base-model-hidden" value="{selected_weights}">
          {weights_browser_html}
          <label style="display:flex;flex-direction:row;align-items:center;gap:6px;text-transform:none;font-weight:400;font-size:12.5px;margin-top:6px;">
            <input type="checkbox" id="no-base-model-checkbox" style="width:auto;" {"checked" if not selected_weights else ""}
              onchange="document.getElementById('base-model-hidden').value = this.checked ? '' : {selected_weights!r}; updateLatestApprovedInfo();">
            No .pt file available -- use the registry's latest Approved package instead
          </label>
          <div id="latest-approved-info" class="card-meta" style="margin-top:6px"></div>
        </div>
        </div>
       </div>

        <div class="form-actions">
          <button class="btn btn-approve" type="submit" {"disabled" if start_disabled else ""}>Start Training</button>
          <button class="btn btn-secondary" type="submit" formaction="/queue/add" formmethod="post" {"disabled" if add_to_queue_disabled else ""}>Add to Queue</button>
        </div>
        <!-- No Stop Training button here, unlike the 8501 screen this is
             modelled on: that app has an endpoint for it and this one does
             not (only /train/dataset and /train/now exist). A control that
             posts to a route the server has never heard of is worse than
             an absent one. -->
        {f'<p class="card-meta" style="margin-top:8px;color:var(--text-muted)">{html.escape(already_running)}</p>' if already_running else ""}
        {f'<p class="card-meta" style="margin-top:8px;color:var(--text-muted)">Queue is full ({MAX_QUEUE_SIZE} max) -- remove an entry below before adding another.</p>' if queue_full and not already_running else ""}

        <div class="section-title" style="margin-top:24px">Automated Schedule (for "Add to Automated Pipeline" below)</div>
        <p class="card-meta" style="margin-bottom:8px">Runs the parameters above on a repeating schedule of its own -- keeps firing even while this review app's own server is stopped overnight/on weekends, unlike the ad-hoc queue below.</p>
        {_render_schedule_picker_fields("weekly", 9, "SUN", 1)}
        <button class="btn" type="submit" formaction="/automated-queue/add" formmethod="post" style="margin-top:10px">Add to Automated Pipeline</button>
      </form>
    </div>

    {_render_queue_panel(queue_snapshot)}

    {_render_automated_queues_panel(automated_queues or [], editing_id, tab="models")}

    <script>
    var GENERATION_SIZES = {generation_sizes_json};
    var SIZE_LABELS = {size_labels_json};
    var LATEST_APPROVED = {latest_approved_json};
    function updateSizeOptions() {{
      var gen = document.getElementById('train-generation').value;
      var sizeSelect = document.getElementById('train-size');
      var previous = sizeSelect.value;
      var sizes = GENERATION_SIZES[gen] || [];
      sizeSelect.innerHTML = '';
      sizes.forEach(function(s) {{
        var opt = document.createElement('option');
        opt.value = s;
        opt.textContent = SIZE_LABELS[s] + ' (' + s + ')';
        sizeSelect.appendChild(opt);
      }});
      if (sizes.indexOf(previous) !== -1) {{
        sizeSelect.value = previous;
      }}
      updateLatestApprovedInfo();
    }}
    // Shows which exact registered model package the "No .pt file
    // available" checkbox would resolve to for whichever
    // Generation+Size is currently selected -- mirrors
    // pipeline_definition.py's _resolve_latest_approved_base_model
    // (latest-by-CreationTime Approved package in that group) so the
    // form can say up front which weights training will actually start
    // from, or that none exist yet, instead of only failing later.
    function updateLatestApprovedInfo() {{
      var infoDiv = document.getElementById('latest-approved-info');
      var checked = document.getElementById('no-base-model-checkbox').checked;
      if (!checked) {{
        infoDiv.textContent = '';
        return;
      }}
      var gen = document.getElementById('train-generation').value;
      var size = document.getElementById('train-size').value;
      var entry = (LATEST_APPROVED[gen] || {{}})[size];
      if (entry) {{
        infoDiv.textContent = 'Latest Approved: ' + entry.group + ' v' + entry.version;
        infoDiv.style.color = 'var(--text-muted)';
      }} else {{
        infoDiv.textContent = 'No Approved package in yolo' + gen + size + '-detector yet -- training will fail until one is approved.';
        infoDiv.style.color = 'var(--reject)';
      }}
    }}
    document.getElementById('train-generation').addEventListener('change', updateSizeOptions);
    document.getElementById('train-size').addEventListener('change', updateLatestApprovedInfo);
    updateSizeOptions();
    </script>
    """


def _render_s3_browser(browse_prefix: str, selected: str, browse_param: str, select_param: str, elem_id: str, tab: str = "models") -> str:
    """Breadcrumb + folder-list browser for picking any folder anywhere in
    the bucket -- state carried entirely in the URL query string (its own
    pair of browse_param/select_param, so two independent instances of
    this browser -- dataset folder, base-model weights -- can coexist on
    one page without stepping on each other's state), since this app
    keeps no server-side session. "Open" drills into a subfolder (GET,
    just changes what's displayed); "Use this folder" sets the actual
    selection, written into a hidden input (id=elem_id) the enclosing
    form submits. `tab` defaults to "models" (its original only caller)
    -- parametrized rather than hardcoded once a third tab (Export
    Weights) needed this exact same folder-picking behavior."""
    browse_prefix = browse_prefix.strip("/")
    folders = _list_s3_folders(browse_prefix)

    def _nav_link(prefix: str, keep_selection: bool = True) -> str:
        qs = f"tab={tab}&{browse_param}={quote(prefix)}"
        if keep_selection and selected:
            qs += f"&{select_param}={quote(selected)}"
        return f"/?{qs}"

    segments = [s for s in browse_prefix.split("/") if s]
    crumbs = [f'<a href="{_nav_link("")}">bucket root</a>']
    built = ""
    for seg in segments:
        built = f"{built}{seg}/"
        crumbs.append(f'<a href="{_nav_link(built.rstrip("/"))}">{seg}</a>')
    breadcrumb_html = " / ".join(crumbs)

    rows = ""
    for name in folders:
        sub_prefix = f"{browse_prefix}/{name}".strip("/")
        select_href = f"/?tab={tab}&{browse_param}={quote(browse_prefix)}&{select_param}={quote(sub_prefix)}"
        is_selected = sub_prefix == selected
        rows += f"""
        <div class="registry-row">
          <a href="{_nav_link(sub_prefix)}">&#128193; {name}/</a>
          <a href="{select_href}" class="btn {'btn-approve' if is_selected else ''}" style="margin-left:auto;padding:4px 10px;font-size:11px;text-decoration:none;">
            {"Selected" if is_selected else "Use this folder"}
          </a>
        </div>
        """
    if not rows:
        rows = '<div class="empty">No subfolders here.</div>'

    selected_html = (
        f'<span class="badge badge-ok">SELECTED</span> <code>s3://{BUCKET}/{selected}/</code>'
        if selected else
        '<span class="badge badge-pending">NONE SELECTED YET</span>'
    )

    return f"""
    <div class="card-meta" style="margin-bottom:8px" id="{elem_id}-status">{selected_html}</div>
    <div class="card-meta" style="margin-bottom:8px">{breadcrumb_html}</div>
    <div class="panel" style="padding:8px 12px">{rows}</div>
    """


def _render_lineage_rows(records: list) -> str:
    """One row per DYNAMODB_LINEAGE_TABLE record -- read-only, no
    forms/actions (unlike _render_registry_rows). Same fields
    _format_lineage summarizes for a registry row, but shown in full here
    (own dataset/base-model/hyperparameters/notes lines rather than one
    joined string) since this tab's whole purpose is browsing lineage."""
    rows = ""
    for record in records:
        record_type = record.get("record_type", "unknown")
        badge_class = {"training_run": "badge-ok", "baseline": "badge-pending"}.get(record_type, "")
        record_id = html.escape(str(record.get("id", "?")))
        timestamp = html.escape(str(record.get("timestamp_utc", "?")))
        yolo = record.get("yolo") or {}
        yolo_desc = html.escape(
            f"YOLO{yolo.get('generation', '?')}-{yolo.get('size', '?')} "
            f"({yolo.get('model_package_group_name', '?')})"
        )

        detail_lines = []
        if record_type == "baseline":
            checkpoint = html.escape(str(record.get("checkpoint", "?")))
            detail_lines.append(f"checkpoint: {checkpoint}")
            model_data_url = html.escape(str(record.get("model_data_url", "?")))
            detail_lines.append(f"weights: {model_data_url}")
            detail_lines.append(
                "approval: auto-approved on registration" if record.get("approved_immediately")
                else "approval: pending manual approval on registration"
            )
        else:
            dataset_uri = html.escape(str((record.get("dataset") or {}).get("s3_uri", "?")))
            detail_lines.append(f"dataset: {dataset_uri}")
            base_model = record.get("base_model") or {}
            base_desc = html.escape(str(base_model.get("model_package_arn") or base_model.get("s3_uri") or "?"))
            detail_lines.append(f"base model ({html.escape(str(base_model.get('source', '?')))}): {base_desc}")
            hp = record.get("hyperparameters") or {}
            detail_lines.append(
                f"epochs={html.escape(str(hp.get('epochs', '?')))}, img_size={html.escape(str(hp.get('img_size', '?')))}, "
                f"batch={html.escape(str(hp.get('batch', '?')))}, seed={html.escape(str(hp.get('seed', '?')))}"
            )
            notes = record.get("notes")
            if notes:
                detail_lines.append(f"notes: {html.escape(str(notes))}")

        detail_html = "".join(f'<div class="card-meta" style="margin-top:4px">{line}</div>' for line in detail_lines)

        rows += f"""
        <div class="registry-row" style="display:block">
          <div style="display:flex;align-items:center;gap:14px;">
            <span class="registry-version">{record_id}</span>
            <span class="badge {badge_class}">{record_type.upper()}</span>
            <span class="registry-date" style="margin-left:auto">{timestamp}</span>
          </div>
          <div class="card-meta" style="margin-top:4px">{yolo_desc}</div>
          {detail_html}
        </div>
        """
    return rows or '<div class="empty">No lineage records yet.</div>'


def _render_lineage_tab(records: list) -> str:
    return f"""
    <h1>Lineage</h1>
    <p class="card-meta" style="margin-bottom:16px">Every training-run and baseline registration recorded in DynamoDB, newest first. Read-only.</p>
    <div class="section-title">Records ({len(records)})</div>
    <div class="panel">{_render_lineage_rows(records)}</div>
    """


def _format_metric(value) -> str:
    return f"{value:.3f}" if isinstance(value, (int, float)) else "?"


def _format_metrics_line(metrics: dict | None) -> str:
    """Short, human-readable summary of the 5 flat evaluation metrics --
    same joined-with-" | "-style convention as _format_lineage. Numbers
    are f-string formatted (safe unescaped); this function returns no
    other string fields, so no html.escape needed here."""
    if not metrics:
        return "no metrics recorded (baseline registration, or evaluation.json unavailable)"
    return " | ".join([
        f"mAP50={_format_metric(metrics.get('mAP50'))}",
        f"mAP50-95={_format_metric(metrics.get('mAP50-95'))}",
        f"precision={_format_metric(metrics.get('precision'))}",
        f"recall={_format_metric(metrics.get('recall'))}",
        f"accuracy={_format_metric(metrics.get('accuracy'))}",
    ])


def _compare_key(row: dict) -> str:
    """Stable identifier for one model package version, used as the
    checkbox value / query-param round-trip in the Compare form below --
    "version" alone isn't unique across the 4 separate yolo11{size}-detector
    groups (each group's own version numbering starts back at 1), so this
    pairs it with group_name."""
    return f"{row['group_name']}/{row['version']}"


def _render_metrics_rows(rows: list, selected_keys: set) -> str:
    """One row per registered model package version, grouped under a
    header per YOLO size (rows already arrive sorted by group -- see
    _scan_all_model_metrics -- so a header is emitted whenever
    group_name changes) so it's obvious at a glance which size's
    versions you're looking at, not just a small label on each row's
    right edge. Read-only except for one addition: a checkbox (only on
    rows that actually have metrics -- a None-metrics row has nothing to
    plot, so it gets a disabled checkbox instead) feeding the "Compare
    Selected" form in _render_metrics_tab."""
    out = ""
    last_group = None
    for row in rows:
        group_name = html.escape(row["group_name"])
        if row["group_name"] != last_group:
            last_group = row["group_name"]
            out += f'<div class="section-title" style="margin:20px 0 8px">{group_name}</div>'
        version = html.escape(str(row["version"]))
        status = row["status"]
        badge_class = {"Approved": "badge-ok", "Rejected": "badge-reject", "PendingManualApproval": "badge-pending"}.get(status, "badge-pending")
        status_esc = html.escape(status)
        metrics_line = _format_metrics_line(row["metrics"])
        if row["metrics"]:
            key = html.escape(_compare_key(row))
            checked = "checked" if _compare_key(row) in selected_keys else ""
            checkbox_html = f'<input type="checkbox" name="compare" value="{key}" {checked} title="Select for comparison" style="width:auto;margin:0;">'
        else:
            checkbox_html = '<input type="checkbox" disabled title="No metrics recorded -- can\'t be compared" style="width:auto;margin:0;opacity:0.35;">'
        # Notes: whatever was typed into the Models tab's "Notes" field
        # at training time (saved as this package's own description).
        # Lineage: the dataset/base-model/hyperparameters that actually
        # produced this version, from DYNAMODB_LINEAGE_TABLE -- same
        # _format_lineage summary line the Review/Lineage tabs already
        # use, so the phrasing is consistent everywhere it appears.
        notes_html = ""
        if row.get("description"):
            notes_html = f'<div class="card-meta" style="margin-top:2px;margin-left:30px;opacity:0.85">notes: {html.escape(row["description"])}</div>'
        lineage_line = _format_lineage(row.get("lineage"))
        lineage_html = (
            f'<div class="card-meta" style="margin-top:2px;margin-left:30px;opacity:0.85">lineage: {lineage_line}</div>'
            if lineage_line else ""
        )
        out += f"""
        <div class="registry-row" style="display:block">
          <div style="display:flex;align-items:center;gap:14px;">
            {checkbox_html}
            <span class="registry-version">v{version}</span>
            <span class="badge {badge_class}">{status_esc.upper()}</span>
          </div>
          <div class="card-meta" style="margin-top:4px;margin-left:30px">{metrics_line}</div>
          {notes_html}
          {lineage_html}
        </div>
        """
    return out or '<div class="empty">No model packages registered yet.</div>'


# Dark-mode categorical palette, adjacent-pair CVD-validated (see the
# dataviz skill's references/palette.md) -- fixed hue order, never cycled
# or generated past this list, which is exactly why MAX_COMPARE below is
# capped at its length rather than some arbitrary round number.
_COMPARE_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
MAX_COMPARE = len(_COMPARE_COLORS)

_COMPARE_METRIC_KEYS = ["mAP50", "mAP50-95", "precision", "recall", "accuracy"]
_COMPARE_METRIC_LABELS = {"mAP50": "mAP50", "mAP50-95": "mAP50-95", "precision": "Precision", "recall": "Recall", "accuracy": "Accuracy"}


def _bar_path(x: float, y: float, width: float, height: float, radius: float) -> str:
    """SVG path for one bar: 4px-rounded top corners, square baseline
    (matches the dataviz skill's bar mark spec) -- a plain <rect> can only
    round all four corners uniformly, so this draws the outline by hand."""
    if height <= 0:
        return ""
    r = min(radius, height, width / 2)
    baseline = y + height
    return (
        f"M{x:.1f},{baseline:.1f} "
        f"L{x:.1f},{y + r:.1f} "
        f"Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} "
        f"L{x + width - r:.1f},{y:.1f} "
        f"Q{x + width:.1f},{y:.1f} {x + width:.1f},{y + r:.1f} "
        f"L{x + width:.1f},{baseline:.1f} Z"
    )


def _render_metrics_bar_chart(selected_rows: list) -> str:
    """Grouped bar chart (one cluster per metric, one bar per selected
    version within it) as plain server-rendered inline SVG -- no charting
    library or CDN script, consistent with this app's whole approach
    elsewhere (e.g. _render_training_progress's epoch bar is just a
    width-percentage <div>). Bar height is the metric value (already a
    0.0-1.0 float) against a shared 0-100% y-axis."""
    bar_width, bar_gap, cluster_gap = 20, 3, 34
    left_margin, right_margin, top_margin, bottom_margin = 40, 12, 26, 34
    plot_height = 160

    n = len(selected_rows)
    cluster_width = n * bar_width + (n - 1) * bar_gap
    plot_width = len(_COMPARE_METRIC_KEYS) * cluster_width + (len(_COMPARE_METRIC_KEYS) - 1) * cluster_gap
    svg_width = left_margin + plot_width + right_margin
    svg_height = top_margin + plot_height + bottom_margin
    direct_label = n <= 4  # dataviz skill: legend always shown, direct labels only up to 4 series to avoid clutter

    def y_for(fraction: float) -> float:
        return top_margin + plot_height * (1 - fraction)

    gridlines = ""
    for pct in (0, 25, 50, 75, 100):
        y = y_for(pct / 100)
        gridlines += f'<line x1="{left_margin}" y1="{y:.1f}" x2="{left_margin + plot_width}" y2="{y:.1f}" stroke="#223047" stroke-width="1"/>'
        gridlines += f'<text x="{left_margin - 8}" y="{y + 3:.1f}" font-size="10" text-anchor="end" fill="#7c8ba1">{pct}%</text>'

    bars, labels = "", ""
    for gi, key in enumerate(_COMPARE_METRIC_KEYS):
        group_x = left_margin + gi * (cluster_width + cluster_gap)
        for vi, row in enumerate(selected_rows):
            value = (row["metrics"] or {}).get(key)
            bar_x = group_x + vi * (bar_width + bar_gap)
            color = _COMPARE_COLORS[vi % len(_COMPARE_COLORS)]
            if isinstance(value, (int, float)):
                clamped = max(0.0, min(1.0, value))
                bar_h = plot_height * clamped
                bar_y = top_margin + plot_height - bar_h
                title = f"{html.escape(row['group_name'])} v{html.escape(str(row['version']))} -- {key}: {value:.3f}"
                bars += f'<path d="{_bar_path(bar_x, bar_y, bar_width, bar_h, 4)}" fill="{color}"><title>{title}</title></path>'
                if direct_label:
                    bars += f'<text x="{bar_x + bar_width / 2:.1f}" y="{bar_y - 4:.1f}" font-size="9" text-anchor="middle" fill="#7c8ba1">{value:.2f}</text>'
            else:
                # No value for this particular metric on this version --
                # a short dash at the baseline instead of a misleading
                # zero-height (indistinguishable-from-0.0) bar.
                dash_y = top_margin + plot_height
                bars += f'<line x1="{bar_x:.1f}" y1="{dash_y:.1f}" x2="{bar_x + bar_width:.1f}" y2="{dash_y:.1f}" stroke="#7c8ba1" stroke-width="2"><title>{html.escape(row["group_name"])} v{html.escape(str(row["version"]))} -- {key}: no data</title></line>'
        labels += f'<text x="{group_x + cluster_width / 2:.1f}" y="{top_margin + plot_height + 18:.1f}" font-size="11" text-anchor="middle" fill="#e8edf4">{html.escape(_COMPARE_METRIC_LABELS[key])}</text>'

    svg = f"""
    <svg width="{svg_width}" height="{svg_height}" viewBox="0 0 {svg_width} {svg_height}" role="img" aria-label="Bar chart comparing {n} model versions across 5 metrics">
      {gridlines}
      {bars}
      {labels}
    </svg>
    """

    legend_html = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:6px;margin-right:16px;margin-bottom:6px;">'
        f'<span style="width:11px;height:11px;border-radius:3px;background:{_COMPARE_COLORS[vi % len(_COMPARE_COLORS)]};display:inline-block;"></span>'
        f'<span class="card-meta" style="margin-top:0">{html.escape(row["group_name"])} v{html.escape(str(row["version"]))}</span>'
        f'</span>'
        for vi, row in enumerate(selected_rows)
    )

    return f"""
    <div style="overflow-x:auto">{svg}</div>
    <div style="margin-top:6px;display:flex;flex-wrap:wrap;">{legend_html}</div>
    <p class="card-meta" style="margin-top:8px">Hover a bar for its exact value; the checked rows above list all 5 metrics as plain numbers too.</p>
    """


def _render_metrics_tab(rows: list, compare_keys: list) -> str:
    selected_keys = set(compare_keys)
    rows_by_key = {_compare_key(row): row for row in rows if row["metrics"]}
    selected_rows = [rows_by_key[key] for key in compare_keys if key in rows_by_key]

    capped_notice = ""
    if len(selected_rows) > MAX_COMPARE:
        capped_notice = f'<p class="card-meta" style="margin-bottom:10px">Comparing the first {MAX_COMPARE} of {len(selected_rows)} selected versions (capped for chart readability/color-distinctness).</p>'
        selected_rows = selected_rows[:MAX_COMPARE]

    if len(selected_rows) >= 2:
        compare_html = f"""
        <div class="section-title">Comparison ({len(selected_rows)} versions selected)</div>
        <div class="panel">
          {capped_notice}
          {_render_metrics_bar_chart(selected_rows)}
        </div>
        """
    elif len(compare_keys) == 1:
        compare_html = '<p class="card-meta" style="margin:8px 0 0">Check at least one more version to compare.</p>'
    else:
        compare_html = ""

    return f"""
    <h1>Metrics</h1>
    <p class="card-meta" style="margin-bottom:16px">Evaluation metrics recorded on each registered model package's ModelMetrics field at registration time (pipeline-trained candidates only -- baseline registrations carry none). Read-only; check 2 or more versions with recorded metrics and click Compare to chart them side by side.</p>
    <form method="get" action="/">
      <input type="hidden" name="tab" value="metrics">
      <div class="section-title">Model Package Versions ({len(rows)})</div>
      <div class="panel">{_render_metrics_rows(rows, selected_keys)}</div>
      <button class="btn btn-approve" type="submit">Compare Selected</button>
    </form>
    {compare_html}
    """


def _render_schedule_tab(schedule, training_progress=None, automated_queues=None, editing_id="") -> str:
    enabled = schedule["enabled"]
    mode_label = "Automated Pipeline" if enabled else "Manual Trigger"
    expr = schedule["expression"] or "not set"
    # Same already-running guard as the Models tab's Start Training button
    # -- see _training_already_running_reason. UX only; do_POST's
    # /train/now handler is what actually enforces this.
    already_running = _training_already_running_reason(training_progress)

    day_options = "".join(f'<option value="{d}">{DAY_LABELS[d]}</option>' for d in DAYS)
    hour_options = "".join(f'<option value="{h}">{h:02d}:00 SGT</option>' for h in range(24))
    day_of_month_options = "".join(f'<option value="{d}">{d}</option>' for d in range(1, 29))
    automated_queues_html = _render_automated_queues_panel(automated_queues or [], editing_id, tab="schedule")

    return f"""
    <h1>Scheduling &amp; Automation</h1>
    <p class="card-meta" style="margin-bottom:16px">Training runs on AWS (SageMaker/CodeBuild), not in your browser -- once a run or an automated queue has started, it keeps going even if you close this page, and Automated Queues below keep firing on schedule even while this review app's own server is stopped overnight/on weekends.</p>

    <div class="panel">
      <div class="section-title">Training Strategy</div>
      <div class="strategy-row">
        <div class="strategy-chip {'active' if not enabled else ''}">Manual Trigger</div>
        <div class="strategy-chip {'active' if enabled else ''}">Automated Pipeline</div>
      </div>
      <div class="card-meta" style="margin-top:8px">Current: <b class="{'ok' if enabled else 'accent'}">{mode_label}</b> &middot; <code>{expr}</code></div>
    </div>

    <div class="panel">
      <div class="section-title">Run Now</div>
      <p class="card-meta">Kicks off a real training run immediately (~$0.70/hr GPU instance), independent of the schedule below.</p>
      <form method="post" action="/train/now">
        <button class="btn btn-approve" type="submit" {"disabled" if already_running else ""}>Train Now</button>
        {f'<p class="card-meta" style="margin-top:6px;color:var(--text-muted)">{html.escape(already_running)}</p>' if already_running else ""}
      </form>
    </div>

    <div class="panel">
      <div class="section-title">Schedule Frequency</div>
      <form method="post" action="/train/schedule">
        <div class="freq-row">
          <label><input type="radio" name="mode" value="manual" {'checked' if not enabled else ''}> Manual only (disable schedule)</label>
        </div>
        <div class="freq-row">
          <label><input type="radio" name="mode" value="daily"> Daily</label>
          at <select name="daily_hour">{hour_options}</select>
        </div>
        <div class="freq-row">
          <label><input type="radio" name="mode" value="weekly" {'checked' if enabled else ''}> Weekly</label>
          on <select name="weekly_day">{day_options}</select>
          at <select name="weekly_hour">{hour_options}</select>
        </div>
        <div class="freq-row">
          <label><input type="radio" name="mode" value="monthly"> Monthly</label>
          on day <select name="monthly_day">{day_of_month_options}</select>
          at <select name="monthly_hour">{hour_options}</select>
        </div>
        <button class="btn btn-approve" style="margin-top:12px" type="submit">Save Configuration</button>
      </form>
    </div>

    {automated_queues_html}
    """


_CSS = """
/* Ported from the JP_UI frontend's design tokens (web/src/index.css) so
   this app and the one on 8501 read as the same product rather than two
   tools that happen to share a host. Plain CSS, no Tailwind: the class
   names below are the ones review_app.py already emits, so this is a
   restyle, not a rewrite of the markup.

   DIRECTION: customs case-file & manifest stamps. Manila/kraft neutrals
   are the ground for every surface, hairline and body of text. One real
   saturated hue exists -- alarm red -- spent only on destructive actions
   and things that need a human's eyes. Verdict inks (red/green/blue) are
   the one exception, and they are three dark pigments of one family, not
   a status-light triad. */
:root {
  color-scheme: light;

  /* Manila/kraft ground -- warm off-white paper, not cool SaaS gray. */
  --bg: #efe7d4;
  --panel: #f8f2e4;
  --panel-2: #e8dfc8;
  --border: #cdbfa0;
  --baseline: #b0a07c;
  --text: #241d16;
  --text-secondary: #5b5142;
  --text-muted: #736a58;

  /* "Primary action ink" -- a deep umber, not a second hue. */
  --accent: #2e2417;
  --accent-hover: #1a130a;
  --accent-dim: #e4d6b3;

  /* Verdict inks. Each is a dark, desaturated pigment chosen to sit on
     the cream paper at the same weight, so they read as one physical set
     of dies. Colour is never the only carrier -- every badge says its
     state in words. */
  --ok: #1f6b3a;
  --ok-dim: #dde7dc;
  --warn: #7a5a1e;
  --warn-dim: #ece0c4;
  --reject: #9c1c17;
  --reject-dim: #f1ddd4;
  --info: #1c4f9c;

  --font-sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-mono: ui-monospace, "IBM Plex Mono", "SFMono-Regular", Menlo, Consolas, monospace;
  /* Header-only display face: a real manual-typewriter face, reserved for
     page titles and top-level section headers. Degrades to a monospace
     stack with zero network if the font request is blocked. */
  --font-display: "Special Elite", ui-monospace, "Courier New", monospace;

  /* Paper grain -- inline SVG fractal noise, no image bytes. The surface
     grain is roughly half strength: a case-file sheet reads as slightly
     less textured than the desk it sits on, never more. */
  --texture-page: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='matrix' values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.035 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
  --texture-surface: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='240' height='240'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix type='matrix' values='0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0.018 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
* { box-sizing: border-box; }
body {
  background-color: var(--bg);
  background-image: var(--texture-page);
  color: var(--text);
  font-family: var(--font-sans);
  margin: 0;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--accent); text-decoration: none; }
a:visited { color: var(--accent); }
a:hover { text-decoration: underline; }
a.btn { display: inline-block; background: var(--panel); border: 1px solid var(--border); color: var(--text); text-decoration: none; }
a.btn:hover { text-decoration: none; background: var(--panel-2); }
a.btn.btn-approve { background: var(--accent); color: var(--panel); border: 1px solid var(--accent); }

/* Synthetic bold on a single-weight display face renders poorly, so
   headers carry no extra weight -- the face's own ink jitter is enough. */
h1, h2 { font-family: var(--font-display); font-weight: 400; letter-spacing: 0.02em; }
h1 { font-size: 22px; margin: 0 0 20px; }
h2 { font-size: 16px; }

/* Left sidebar, matching 8501's shell. The old horizontal tab strip put
   ten destinations in a row that wrapped awkwardly and gave no room for
   names to grow; a rail reads as a filing cabinet, which is the point. */
.layout { display: flex; min-height: 100vh; }
.sidebar {
  width: 248px; flex-shrink: 0;
  background-color: var(--panel); background-image: var(--texture-surface);
  border-right: 1px solid var(--border);
  padding: 20px 12px; position: sticky; top: 0; height: 100vh; overflow-y: auto;
}
.brand { display: flex; flex-direction: column; gap: 2px; padding: 0 10px 18px; }
.brand-name { font-family: var(--font-display); font-size: 17px; letter-spacing: 0.02em; }
.brand-sub { font-family: var(--font-mono); font-size: 11px; color: var(--text-muted); }
.tabs { display: flex; flex-direction: column; gap: 2px; }
.tab {
  display: block; color: var(--text-secondary); text-decoration: none;
  padding: 8px 10px; border-radius: 3px; font-size: 13px; font-weight: 500;
  border-left: 2px solid transparent;
}
.tab:hover { background: rgba(0,0,0,0.04); text-decoration: none; }
.tab.active { background: var(--accent-dim); color: var(--text); font-weight: 600; border-left-color: var(--accent); }

main { flex: 1; min-width: 0; max-width: 1240px; padding: 28px 32px; }

/* Ruled hairlines and square-ish corners throughout -- ledger stock, not
   rounded SaaS cards. */
.banner { background: var(--accent-dim); border: 1px solid var(--border); color: var(--text); padding: 10px 16px; border-radius: 3px; margin-bottom: 20px; font-size: 13px; }
.banner-error { background: var(--reject-dim); border-color: var(--reject); color: var(--reject); }
.section-title { text-transform: uppercase; letter-spacing: 0.06em; font-size: 11px; color: var(--text-muted); margin: 28px 0 12px; font-weight: 700; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 14px; }
.stat-card { background-color: var(--panel); background-image: var(--texture-surface); border: 1px solid var(--border); border-radius: 3px; padding: 16px; }
.stat-label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }
/* Counts and measurements are data: typewriter face, tabular figures. */
.stat-value { font-family: var(--font-mono); font-variant-numeric: tabular-nums; font-size: 25px; font-weight: 700; margin-top: 6px; }
.stat-value.accent { color: var(--accent); }
.stat-value.ok { color: var(--ok); }
.review-list { display: flex; flex-direction: column; gap: 14px; }
.card { background-color: var(--panel); background-image: var(--texture-surface); border: 1px solid var(--border); border-radius: 3px; padding: 16px; display: flex; gap: 16px; align-items: flex-start; }
.thumb { width: 140px; height: 100px; object-fit: cover; border-radius: 2px; background: var(--panel-2); border: 1px solid var(--border); }
.card-body { flex: 1; min-width: 0; }
.card-title-row { display: flex; align-items: center; gap: 10px; }
.card-title { font-weight: 700; font-size: 14px; }
.card-meta { color: var(--text-muted); font-size: 12px; margin-top: 4px; font-family: var(--font-mono); }
.boxes { background: var(--panel-2); border: 1px solid var(--border); border-radius: 2px; padding: 8px; font-size: 11px; font-family: var(--font-mono); max-height: 90px; overflow: auto; margin-top: 8px; }
.card-actions { display: flex; flex-direction: column; gap: 8px; }
.btn { border: 1px solid var(--accent); background: var(--accent); color: var(--panel); border-radius: 3px; padding: 8px 16px; font-weight: 600; font-size: 12.5px; cursor: pointer; font-family: inherit; }
.btn:hover { background: var(--accent-hover); }
.btn-approve { background: var(--accent); color: var(--panel); }
/* Secondary action: outlined, so the primary one stays the only filled
   thing in the row and reads as the default. */
.btn-secondary { background: transparent; border: 1px solid var(--border); color: var(--text); }
.btn-secondary:hover { background: var(--panel-2); }
.btn:disabled, .btn[disabled] { opacity: 0.4; cursor: not-allowed; }
.btn:disabled:hover, .btn[disabled]:hover { background: var(--accent); }
.btn-secondary:disabled:hover { background: transparent; }
/* Destructive actions get the one alarm red, outlined rather than filled
   -- it should read as a warning, not as the primary thing to click. */
.btn-reject { background: transparent; border: 1px solid var(--reject); color: var(--reject); }
.btn-reject:hover { background: var(--reject-dim); }
.badge { font-size: 10px; font-weight: 700; padding: 3px 9px; border-radius: 2px; letter-spacing: 0.05em; text-transform: uppercase; border: 1px solid transparent; }
.badge-pending { background: var(--warn-dim); color: var(--warn); border-color: var(--warn); }
.badge-ok { background: var(--ok-dim); color: var(--ok); border-color: var(--ok); }
.badge-reject { background: var(--reject-dim); color: var(--reject); border-color: var(--reject); }
.empty { color: var(--text-muted); font-size: 13px; padding: 24px; text-align: center; background: var(--panel); border-radius: 3px; border: 1px dashed var(--border); }
.panel { background-color: var(--panel); background-image: var(--texture-surface); border: 1px solid var(--border); border-radius: 3px; padding: 18px 20px; margin-bottom: 16px; }
.registry-row { display: flex; align-items: center; gap: 14px; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.registry-row:last-child { border-bottom: none; }
.registry-version { font-weight: 700; width: 60px; font-family: var(--font-mono); }
.registry-date { color: var(--text-muted); margin-left: auto; font-size: 11px; font-family: var(--font-mono); }
.progress-stages { display: flex; gap: 10px; }
.progress-segment { flex: 1; background: var(--panel-2); border: 1px solid var(--border); border-radius: 3px; padding: 10px 12px; text-align: center; }
.progress-segment-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); margin-bottom: 6px; }
.epoch-bar { background: var(--panel-2); border: 1px solid var(--border); border-radius: 999px; height: 9px; overflow: hidden; margin-top: 6px; }
.epoch-bar-fill { background: var(--accent); height: 100%; transition: width 0.4s ease; }
.strategy-row { display: flex; gap: 8px; margin-top: 8px; }
.strategy-chip { flex: 1; text-align: center; padding: 10px; border-radius: 3px; border: 1px solid var(--border); color: var(--text-muted); font-size: 12.5px; font-weight: 600; }
.strategy-chip.active { border-color: var(--accent); color: var(--text); background: var(--accent-dim); }
.freq-row { display: flex; align-items: center; gap: 8px; padding: 8px 0; font-size: 13px; }
select, input[type="text"], input[type="number"], input[type="file"], textarea {
  background: var(--panel); color: var(--text); border: 1px solid var(--border);
  border-radius: 3px; padding: 6px 8px; font-size: 13px; font-family: inherit;
}
input[type="checkbox"], input[type="radio"] { accent-color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; border-radius: 2px; }
code { background: var(--panel-2); border: 1px solid var(--border); padding: 1px 5px; border-radius: 2px; font-size: 11px; font-family: var(--font-mono); }
pre { font-family: var(--font-mono); }
table { border-collapse: collapse; }
th { text-transform: uppercase; letter-spacing: 0.05em; font-size: 11px; color: var(--text-muted); font-weight: 700; }
td { font-variant-numeric: tabular-nums; }
/* Standfirst under a page title -- says what the screen is for, in the
   body face, so the typewriter h1 stays a title rather than an
   explanation. */
.page-sub { color: var(--text-secondary); font-size: 13px; line-height: 1.5; margin: -10px 0 22px; max-width: 76ch; }

/* A panel's own leading section-title becomes a card header with a rule
   under it, matching the 8501 Card component. Done here rather than in
   each renderer so every panel in the app picks it up from one rule --
   the section-titles that appear mid-panel keep the small uppercase
   label treatment they already had. */
.panel > .section-title:first-child {
  font-size: 14px; font-weight: 600; text-transform: none; letter-spacing: 0;
  color: var(--text); margin: -18px -20px 16px; padding: 14px 20px 12px;
  border-bottom: 1px solid var(--border);
}
/* Two cards abreast (GPU resource / latest run), collapsing to one
   column before they get too narrow to read. */
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; align-items: start; }
.grid-2 > .panel { margin-bottom: 0; }
/* Parameters on the left, the two S3 browsers on the right: the browsers
   are tall, and stacking them under the fields pushed the submit buttons
   far below the fold. */
.form-2col { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
.form-2col .field-grid { margin-bottom: 14px; }
/* Standalone fields need their own rhythm here -- inside .field-grid the
   grid gap supplied it. */
.form-2col .field { margin-bottom: 14px; }
.form-2col .field:last-child { margin-bottom: 0; }
/* Exactly two per row (Epochs/Image size, Batch/Seed) rather than
   auto-fit, which reflows to three-up on a wide screen and breaks the
   pairing that makes them readable. */
.field-row-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.field-row-2 .field { margin-bottom: 14px; }
.form-actions { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 18px; }
.form-actions .btn { margin: 0; }
@media (max-width: 1100px) {
  .grid-2, .form-2col { grid-template-columns: 1fr; }
}

.field-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin-bottom: 14px; }
.field { display: flex; flex-direction: column; gap: 6px; }
.field label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }
.field select, .field input { width: 100%; }

@media (max-width: 900px) {
  .layout { flex-direction: column; }
  .sidebar { width: 100%; height: auto; position: static; border-right: none; border-bottom: 1px solid var(--border); }
  .tabs { flex-direction: row; flex-wrap: wrap; }
  .tab { border-left: none; border-bottom: 2px solid transparent; }
  .tab.active { border-left: none; border-bottom-color: var(--accent); }
  main { padding: 20px 16px; }
}
"""


# --- HTTP handler -----------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    def _redirect(self, tab: str, message: str = "", is_error: bool = False):
        location = f"/?tab={tab}"
        if message:
            location += f"&msg={quote(message)}"
            if is_error:
                location += "&msg_level=error"
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            qs = parse_qs(parsed.query)
            tab = qs.get("tab", ["review"])[0]
            msg = qs.get("msg", [""])[0]
            msg_is_error = qs.get("msg_level", [""])[0] == "error"
            browse_prefix = qs.get("browse_prefix", [DATASETS_S3_PREFIX])[0]
            # Default to the same dataset pipeline_definition.py itself
            # defaults to (see DEFAULT_TRAIN_DATASET below) so the "Start
            # Training" button isn't silently disabled on first load --
            # previously it required browsing to and clicking a dataset
            # folder before Start Training would do anything at all, which
            # from a user's perspective just looked like the button (and
            # the whole form) not working.
            selected_dataset = qs.get("selected_dataset", [DEFAULT_TRAIN_DATASET])[0]
            weights_browse_prefix = qs.get("weights_browse_prefix", [""])[0]
            selected_weights = qs.get("selected_weights", [""])[0]
            compare_keys = qs.get("compare", [])
            edit_auto_queue = qs.get("edit_auto_queue", [""])[0]
            infer_browse_prefix = qs.get("infer_browse_prefix", [""])[0]
            infer_image_key = qs.get("infer_image_key", [""])[0]
            infer_size = qs.get("infer_size", [""])[0]
            infer_run = qs.get("infer_run", [""])[0] == "1"
            export_browse_prefix = qs.get("export_browse_prefix", [""])[0]
            export_selected_folder = qs.get("export_selected_folder", [""])[0]
            body = _render_page(
                tab if tab in ("review", "models", "schedule", "lineage", "metrics", "deploy", "notifications", "help", "inference", "export") else "review",
                msg,
                msg_is_error=msg_is_error,
                browse_prefix=browse_prefix,
                selected_dataset=selected_dataset,
                weights_browse_prefix=weights_browse_prefix,
                selected_weights=selected_weights,
                edit_auto_queue=edit_auto_queue,
                infer_browse_prefix=infer_browse_prefix,
                infer_image_key=infer_image_key,
                infer_size=infer_size,
                infer_run=infer_run,
                export_browse_prefix=export_browse_prefix,
                export_selected_folder=export_selected_folder,
                compare_keys=compare_keys,
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global _queue_started
        parsed = urlparse(self.path)
        parts = parsed.path.strip("/").split("/")

        if len(parts) == 3 and parts[0] == "review" and parts[2] in ("approve", "reject"):
            review_id, decision = parts[1], parts[2]
            table().update_item(
                Key={"review_id": review_id},
                UpdateExpression="SET #s = :status, reviewed_at = :ts",
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":status": "approved" if decision == "approve" else "rejected",
                    ":ts": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._redirect("review", f"Review {review_id} marked {decision}d")
            return

        if len(parts) == 3 and parts[0] == "model" and parts[2] in ("approve", "reject"):
            model_package_arn, decision = unquote(parts[1]), parts[2]
            sm.update_model_package(
                ModelPackageArn=model_package_arn,
                ModelApprovalStatus="Approved" if decision == "approve" else "Rejected",
            )
            version = model_package_arn.rsplit("/", 1)[-1]
            self._redirect("review", f"Model v{version} marked {decision}d")
            return

        if len(parts) == 3 and parts[0] == "model" and parts[2] == "delete":
            model_package_arn = unquote(parts[1])
            version = model_package_arn.rsplit("/", 1)[-1]
            try:
                sm.delete_model_package(ModelPackageName=model_package_arn)
                self._redirect("review", f"Deleted model package v{version}")
            except botocore.exceptions.ClientError as exc:
                self._redirect("review", f"Delete failed for v{version}: {exc}", is_error=True)
            return

        if len(parts) == 3 and parts[0] == "models" and parts[2] == "deploy":
            size = parts[1]
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            model_package_arn = form["model_package_arn"][0]
            try:
                result = _deploy_inference_component(size, model_package_arn=model_package_arn)
                self._redirect("models", f"Deployed {result}")
            except Exception as exc:
                self._redirect("models", f"Deploy failed: {exc}", is_error=True)
            return

        if len(parts) == 3 and parts[0] == "models" and parts[2] == "deploy-s3":
            size = parts[1]
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            s3_folder = form["s3_folder"][0]
            try:
                result = _deploy_inference_component(size, weights_s3_prefix=f"weights/{size}/{s3_folder}")
                self._redirect("models", f"Deployed {result}")
            except Exception as exc:
                self._redirect("models", f"Deploy failed: {exc}", is_error=True)
            return

        if parsed.path == "/train/dataset":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            generation = form["generation"][0]
            size = form["size"][0]
            epochs = form.get("epochs", ["50"])[0]
            img_size = form.get("img_size", ["640"])[0]
            batch = form.get("batch", ["16"])[0]
            seed = form.get("seed", ["42"])[0]
            # Full bucket-relative key path now (the dataset browser lets
            # you pick anywhere in the bucket, not just one fixed
            # prefix) -- not a bare folder name to prefix with
            # DATASETS_S3_PREFIX like the old single-level dropdown was.
            dataset_folder = form["dataset_folder"][0]
            base_model_folder = form.get("base_model_folder", [""])[0]
            # Optional -- saved as the registered model package's own
            # description (see pipeline_definition.py's TRAIN_NOTES),
            # not just logged here, so it's still readable months later
            # from the registry itself. CodeBuild env var values cap out
            # at 2048 chars; trimmed defensively since this is meant to
            # be a short note, not a full writeup.
            notes = form.get("notes", [""])[0][:2048]
            try:
                already_running = _training_already_running_reason(_get_training_progress())
                if already_running:
                    raise ValueError(already_running)
                build_id = _start_training_run(generation, size, epochs, dataset_folder, base_model_folder, notes, img_size, batch, seed)
                self._redirect("models", f"Training started (YOLO{generation}-{size}, {epochs} epochs, {dataset_folder}): {build_id}")
            except Exception as exc:
                self._redirect("models", f"Training trigger failed: {exc}", is_error=True)
            return

        if parsed.path == "/queue/add":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            generation = form["generation"][0]
            size = form["size"][0]
            epochs = form.get("epochs", ["50"])[0]
            img_size = form.get("img_size", ["640"])[0]
            batch = form.get("batch", ["16"])[0]
            seed = form.get("seed", ["42"])[0]
            dataset_folder = form.get("dataset_folder", [""])[0]
            base_model_folder = form.get("base_model_folder", [""])[0]
            notes = form.get("notes", [""])[0][:2048]
            try:
                # Same validation _start_training_run itself does -- catch
                # a bad entry now rather than letting it sit in the queue
                # only to fail once it's finally its turn to run.
                epochs_int, img_size_int, batch_int, seed_int = _validate_training_params(epochs, dataset_folder, img_size, batch, seed)
                with _queue_lock:
                    if len(_training_queue) >= MAX_QUEUE_SIZE:
                        raise ValueError(f"Queue is full (max {MAX_QUEUE_SIZE} runs) -- remove an entry before adding another.")
                    _training_queue.append({
                        "id": uuid.uuid4().hex[:12],
                        "generation": generation,
                        "size": size,
                        "epochs": epochs_int,
                        "img_size": img_size_int,
                        "batch": batch_int,
                        "seed": seed_int,
                        "dataset_folder": dataset_folder,
                        "base_model_folder": base_model_folder,
                        "notes": notes,
                        "added_at": datetime.now(timezone.utc).isoformat(),
                    })
                self._redirect("models", f"Added to queue (YOLO{generation}-{size}, {epochs_int} epochs, {dataset_folder})")
            except Exception as exc:
                self._redirect("models", f"Add to queue failed: {exc}", is_error=True)
            return

        if len(parts) == 3 and parts[0] == "queue" and parts[1] == "remove":
            entry_id = unquote(parts[2])
            with _queue_lock:
                before = len(_training_queue)
                _training_queue[:] = [e for e in _training_queue if e["id"] != entry_id]
                removed = len(_training_queue) < before
            if removed:
                self._redirect("models", "Removed from queue")
            else:
                self._redirect("models", "Queue entry not found (already started or removed)", is_error=True)
            return

        if parsed.path == "/queue/start":
            with _queue_lock:
                if not _training_queue and _queue_running is None:
                    self._redirect("models", "Queue is empty -- add at least one run first.", is_error=True)
                    return
                _queue_started = True
            # Try to start the first run immediately rather than waiting
            # up to ~25s for the next background poll tick.
            _try_advance_queue()
            self._redirect("models", "Queue started")
            return

        if parsed.path == "/queue/stop":
            with _queue_lock:
                _queue_started = False
            # Only stops auto-advancing to further queued entries -- a run
            # the queue already handed to CodeBuild keeps running (there's
            # no "cancel a submitted training run" feature here, same as
            # the immediate Start Training button).
            self._redirect("models", "Queue stopped -- any run already in progress will finish; remaining entries stay queued.")
            return

        if parsed.path == "/automated-queue/add":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            generation = form["generation"][0]
            size = form["size"][0]
            epochs = form.get("epochs", ["50"])[0]
            img_size = form.get("img_size", ["640"])[0]
            batch = form.get("batch", ["16"])[0]
            seed = form.get("seed", ["42"])[0]
            dataset_folder = form.get("dataset_folder", [""])[0]
            base_model_folder = form.get("base_model_folder", [""])[0]
            notes = form.get("notes", [""])[0][:2048]
            mode = form.get("mode", ["weekly"])[0]
            try:
                epochs_int, img_size_int, batch_int, seed_int = _validate_training_params(
                    epochs, dataset_folder, img_size, batch, seed
                )
                schedule_expression, schedule_summary = _cron_and_summary_from_schedule_form(mode, form)
                # Resolved once now, not re-resolved by the Lambda on
                # every firing -- see _automated_queue_payload's own note.
                base_model_s3_uri = _resolve_base_model_s3_uri(base_model_folder) if base_model_folder else None
                now = datetime.now(timezone.utc).isoformat()
                queue = {
                    "id": uuid.uuid4().hex[:12],
                    "generation": generation, "size": size,
                    "epochs": epochs_int, "img_size": img_size_int, "batch": batch_int, "seed": seed_int,
                    "dataset_folder": dataset_folder, "base_model_folder": base_model_folder,
                    "base_model_s3_uri": base_model_s3_uri, "notes": notes,
                    "schedule_mode": mode,
                    "schedule_hour": int(form.get(f"{mode}_hour", ["9"])[0]),
                    "schedule_day": form.get("weekly_day", ["SUN"])[0] if mode == "weekly" else None,
                    "schedule_dom": int(form["monthly_day"][0]) if mode == "monthly" else None,
                    "schedule_expression": schedule_expression,
                    "schedule_summary": schedule_summary,
                    "created_at": now, "updated_at": now,
                }
                automated_queues_table().put_item(Item=queue)
                _put_automated_queue_rule(queue)
                self._redirect("models", f"Automated queue added (YOLO{generation}-{size}, {schedule_summary})")
            except Exception as exc:
                self._redirect("models", f"Add automated queue failed: {exc}", is_error=True)
            return

        if len(parts) == 3 and parts[0] == "automated-queue" and parts[1] == "update":
            queue_id = unquote(parts[2])
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            return_tab = form.get("return_tab", ["models"])[0]
            generation = form["generation"][0]
            size = form["size"][0]
            epochs = form.get("epochs", ["50"])[0]
            img_size = form.get("img_size", ["640"])[0]
            batch = form.get("batch", ["16"])[0]
            seed = form.get("seed", ["42"])[0]
            dataset_folder = form.get("dataset_folder", [""])[0]
            base_model_folder = form.get("base_model_folder", [""])[0]
            notes = form.get("notes", [""])[0][:2048]
            mode = form.get("mode", ["weekly"])[0]
            try:
                existing = automated_queues_table().get_item(Key={"id": queue_id}).get("Item")
                if not existing:
                    raise ValueError(f"Automated queue {queue_id} not found (already deleted?).")
                epochs_int, img_size_int, batch_int, seed_int = _validate_training_params(
                    epochs, dataset_folder, img_size, batch, seed
                )
                schedule_expression, schedule_summary = _cron_and_summary_from_schedule_form(mode, form)
                base_model_s3_uri = _resolve_base_model_s3_uri(base_model_folder) if base_model_folder else None
                queue = {
                    **existing,
                    "generation": generation, "size": size,
                    "epochs": epochs_int, "img_size": img_size_int, "batch": batch_int, "seed": seed_int,
                    "dataset_folder": dataset_folder, "base_model_folder": base_model_folder,
                    "base_model_s3_uri": base_model_s3_uri, "notes": notes,
                    "schedule_mode": mode,
                    "schedule_hour": int(form.get(f"{mode}_hour", ["9"])[0]),
                    "schedule_day": form.get("weekly_day", ["SUN"])[0] if mode == "weekly" else None,
                    "schedule_dom": int(form["monthly_day"][0]) if mode == "monthly" else None,
                    "schedule_expression": schedule_expression,
                    "schedule_summary": schedule_summary,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                automated_queues_table().put_item(Item=queue)
                # put_rule/put_targets are upserts by name -- this replaces
                # the existing rule's schedule/payload in place, no separate
                # "delete old rule first" step needed.
                _put_automated_queue_rule(queue)
                self._redirect(return_tab, f"Automated queue updated (YOLO{generation}-{size}, {schedule_summary})")
            except Exception as exc:
                self._redirect(return_tab, f"Update automated queue failed: {exc}", is_error=True)
            return

        if len(parts) == 3 and parts[0] == "automated-queue" and parts[1] == "delete":
            queue_id = unquote(parts[2])
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            return_tab = form.get("return_tab", ["models"])[0]
            _delete_automated_queue_rule(queue_id)
            automated_queues_table().delete_item(Key={"id": queue_id})
            self._redirect(return_tab, "Automated queue deleted")
            return

        if parsed.path == "/train/now":
            already_running = _training_already_running_reason(_get_training_progress())
            if already_running:
                self._redirect("schedule", already_running, is_error=True)
                return
            response = sm.start_pipeline_execution(PipelineName=PIPELINE_NAME)
            arn = response["PipelineExecutionArn"]
            self._redirect("schedule", f"Started: {arn.rsplit('/', 1)[-1]}")
            return

        if parsed.path == "/train/schedule":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            mode = form.get("mode", ["manual"])[0]

            if mode == "manual":
                events.disable_rule(Name=RULE_NAME)
                self._redirect("schedule", "Schedule disabled - manual trigger only")
                return

            try:
                expr, _summary = _cron_and_summary_from_schedule_form(mode, form)
            except ValueError as exc:
                self._redirect("schedule", str(exc), is_error=True)
                return

            events.put_rule(Name=RULE_NAME, ScheduleExpression=expr, State="ENABLED")
            self._redirect("schedule", f"Schedule updated: {expr}")
            return

        if parsed.path == "/notifications/add":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            email = form.get("email", [""])[0].strip()
            try:
                if not email:
                    raise ValueError("No email address given.")
                sns.subscribe(TopicArn=REVIEWER_NOTIFICATIONS_TOPIC_ARN, Protocol="email", Endpoint=email)
                self._redirect("notifications", f"Confirmation email sent to {email} -- they must click the link before notifications start.")
            except Exception as exc:
                self._redirect("notifications", f"Subscribe failed: {exc}", is_error=True)
            return

        if parsed.path == "/notifications/remove":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            subscription_arn = form.get("subscription_arn", [""])[0]
            try:
                if not subscription_arn:
                    raise ValueError("No subscription specified.")
                sns.unsubscribe(SubscriptionArn=subscription_arn)
                self._redirect("notifications", "Removed from notifications.")
            except Exception as exc:
                self._redirect("notifications", f"Remove failed: {exc}", is_error=True)
            return

        if parsed.path == "/export-weights":
            length = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(length).decode())
            model_package_arn = form.get("model_package_arn", [""])[0]
            destination_folder = form.get("destination_folder", [""])[0]
            try:
                if not model_package_arn:
                    raise ValueError("No model selected.")
                new_key, archived_key = _export_weights_to_folder(model_package_arn, destination_folder)
                msg = f"Exported to s3://{BUCKET}/{new_key}"
                if archived_key:
                    msg += f" (previous weights archived to s3://{BUCKET}/{archived_key})"
                self._redirect("export", msg)
            except Exception as exc:
                self._redirect("export", f"Export failed: {exc}", is_error=True)
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


if __name__ == "__main__":
    # Daemon (dies with the process, no explicit shutdown needed) --
    # advances the training queue in the background independent of
    # request handling either way.
    threading.Thread(target=_queue_poll_loop, daemon=True).start()
    # ThreadingHTTPServer (not plain HTTPServer): this app is public
    # (0.0.0.0/0, no auth of its own -- see terraform/infra/ec2.tf), and
    # a plain HTTPServer handles exactly one request at a time -- a
    # single stalled/slow connection (an internet scanner that opens a
    # socket and never finishes reading/writing, for instance) blocks
    # every other visitor indefinitely. Confirmed live: this exact thing
    # wedged the whole app on 2026-09-08, fixed only by restarting the
    # container. Every module-level mutable state shared across request
    # threads is deliberately either lock-protected (_queue_lock,
    # _pending_submission_lock) or made thread-local (the DynamoDB
    # resource accessors, table()/lineage_table()/automated_queues_table()
    # above -- boto3 resources, unlike clients, aren't documented
    # thread-safe) specifically because of this switch.
    class Server(ThreadingHTTPServer):
        # socketserver.TCPServer's default request_queue_size is 5 -- the
        # OS-level backlog of not-yet-accepted connections, set via
        # listen() during __init__/server_activate() (too late to change
        # after construction, hence subclassing rather than setting the
        # attribute on an instance). With multiple real users plus the
        # steady trickle of internet scanners this public, unauthenticated
        # app gets hit by, a burst of more than 5 simultaneous new
        # connection attempts could get refused/reset before the accept
        # loop -- fast as it is with ThreadingHTTPServer -- gets to them.
        # Not the same bottleneck the hang above was (that was about
        # requests already being handled), but a related one worth
        # closing at the same time.
        request_queue_size = 128

    Server(("0.0.0.0", 8090), Handler).serve_forever()
