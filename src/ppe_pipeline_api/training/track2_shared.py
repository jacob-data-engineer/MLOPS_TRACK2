"""Shared config, boto3 clients, and business logic for every `track2_*`
router — the YOLO/Track2 training-pipeline control surface (review queue,
model registry, training runs, scheduling, deploy, inference, lineage,
notifications) that used to live in the standalone stdlib app at
`terraform/review-app/review_app.py`. That app had no authentication and
ran on its own public EC2 instance; this migration ports its business
logic into this FastAPI app (gated by the existing session auth) and
its React shell (matching the rest of the UI's look), dropping only the
old app's hand-rolled HTML rendering and manual query-string routing.

Config is read directly from `os.environ`, matching this API's existing
no-central-Settings convention (see `auth.py`).
"""

from __future__ import annotations

import concurrent.futures
import csv
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone

import decimal

import boto3
import botocore.config
import botocore.exceptions
from boto3.dynamodb.conditions import Attr, Key

from .hyperparams import normalize_hyperparams

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")

def _yaml():
    """PyYAML, imported on use rather than at module load.

    This module is zipped whole as the queue-runner Lambda, which advances
    the training queue when the app container is not up to do it. Lambda's
    Python runtime ships boto3 and botocore and nothing else, so a
    top-level `import yaml` would mean vendoring PyYAML into the zip or
    building it in CodeBuild -- a second packaging path to keep in step
    with the first, for a library five functions in six thousand lines
    actually touch.

    Lazy instead. The queue-advance path never reads a data.yaml, so the
    Lambda never imports it; everything in the container still does, on
    first use, and the container has it.
    """
    import yaml

    return yaml


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
GENERATION_SIZES = {"11": ("s", "m", "l", "x"), "26": ("s", "m", "l", "x")}
GENERATION_LABELS = {"11": "YOLO11", "26": "YOLO26"}
#: The generation a detector gets if it does not say -- what "yolo11s" in
#: an old record means, and what a new run defaults to.
DEFAULT_GENERATION = "11"
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
from ..storage import s3  # noqa: E402 -- one client, shared with the data side
codebuild = boto3.client("codebuild", region_name=REGION)
events = boto3.client("events", region_name=REGION)
# Adaptive retries, not the default.
#
# boto3's default retry mode is `legacy`: 4 attempts, a fixed backoff, no
# idea that it is being rate-limited. The registry now holds 53 model
# package groups, and the Review & Metrics scan lists every one of them.
# Fifty-three ListModelPackages calls in a row is enough to exhaust those
# four attempts, and when they run out botocore RAISES -- so the page
# 500s. Measured on the live account, serially, with no concurrency
# involved at all.
#
# `adaptive` adds a client-side rate limiter that slows down when it sees
# throttling instead of hammering until it gives up, and 10 attempts
# gives it room to ride out a burst. Slower under load, which is the
# correct trade against a page that fails.
sm = boto3.client(
    "sagemaker", region_name=REGION,
    config=botocore.config.Config(retries={"max_attempts": 10, "mode": "adaptive"}),
)
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
    # weights approved) -- see terraform/platform/monitoring.tf.
    return f"arn:aws:sns:{REGION}:{account_id()}:track2-yolo-reviewer-notifications"


# --- the data side's, since the split (2026-10-28) ---------------------------
#
# These lived here. They are data preparation, so they moved to
# `data/assemble.py` and the shared S3 helpers to `storage.py`. The names
# stay, as the data side's FRONT DOOR, so every router and test that says
# `t2.assemble_dataset` still works -- and a patch on one of these names
# still intercepts the calls this module makes to it.
from ..storage import _list_keys, _s3_object_exists, list_s3_folders  # noqa: E402
from .. import data as _data  # noqa: E402

FILTERED_DATASET_MARKER = _data.FILTERED_DATASET_MARKER
ASSEMBLED_DATASET_PREFIX = _data.ASSEMBLED_DATASET_PREFIX
SOURCES_FILE = _data.SOURCES_FILE
_YAML_IS_THE_ONLY_SOURCE_OF_NAMES = _data.YAML_IS_THE_ONLY_SOURCE_OF_NAMES
_basenames = _data.image_names_by_stem
discover_dataset_sources = _data.discover_dataset_sources
inspect_dataset_layout = _data.inspect_dataset_layout
assemble_dataset = _data.assemble_dataset
dataset_source_map = _data.dataset_source_map
class_filter_for = _data.class_filter_for
derive_filtered_dataset = _data.derive_filtered_dataset
_evaluates_on_its_training_images = _data.evaluates_on_its_training_images
dataset_classes = _data.dataset_classes


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
#: True from the moment an entry is popped until its start call returns.
#: The lock is deliberately NOT held across that call -- it is a SageMaker
#: round trip, and a mutex held across it blocks every other queue reader,
#: including the card polling every 15 seconds. This flag keeps the mutual
#: exclusion the lock was providing without keeping the lock.
_queue_starting = False

# Bridges the real gap between "a codebuild.start_build call succeeded"
# and "a new SageMaker Pipeline Execution actually exists for
# get_training_progress to report on" -- run_pipeline.py has to check out
# source and run inside CodeBuild first, which takes real time, during
# which the UI would otherwise keep showing the PREVIOUS run's already-
# terminal status.
_pending_submission_lock = threading.Lock()
# The custom pipeline's equivalent: when a run was submitted, so the
# single-GPU guard holds the slot until the execution shows up as Executing.
_custom_submission_lock = threading.Lock()
_custom_submission_at: datetime | None = None
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


#: Group listings, by group name: {group_name: (fetched_at, summary)}.
#:
#: The registry holds 53 groups between them holding 20 versions, so one
#: load of Review & Metrics is 53 ListModelPackages calls to find 20
#: rows -- measured at 10 of the page's 19 seconds, and enough on its own
#: to trip SageMaker's rate limiter.
#:
#: Five minutes, not thirty seconds. The first version of this used a
#: short TTL on the theory that approval status lives in this answer --
#: but EVERY action that changes it already clears the cache outright
#: (see `forget_registry_listing`), so the TTL was never what kept the
#: page honest. All thirty seconds achieved was that the start-up warm-up
#: had gone stale before the first person arrived, and they paid the full
#: 7-12 seconds anyway -- measured, not assumed.
#:
#: What the TTL actually covers is a change made SOMEWHERE ELSE: a
#: pipeline registering a version, or the AWS console. Those are worth
#: picking up, but not worth re-listing 53 groups every half minute to
#: catch, and the background refresh below keeps the window to about
#: four minutes regardless.
_listing_cache: dict[str, tuple[float, dict]] = {}
_LISTING_TTL = 300.0
#: How often the background refresher re-warms. Under the TTL, so the
#: cache is replaced while it is still valid and no request ever waits
#: for a cold listing.
_LISTING_REFRESH_SECONDS = 240.0


def forget_registry_listing(group_name: str = "") -> None:
    """Drop a cached listing -- one group's, or all of them.

    Called by everything that approves, rejects, registers or deletes,
    so the page that triggered the change never shows the state before
    it. A cache nobody invalidates is just a bug with a timer on it.
    """
    if group_name:
        _listing_cache.pop(group_name, None)
    else:
        _listing_cache.clear()


def model_registry_summary(group_name: str, *, max_age: float = _LISTING_TTL) -> dict:
    """Every version in a group, newest first, plus how many are Approved.

    Cached for `max_age` seconds; pass `max_age=0` to force a fresh call.

    PAGINATED. It used to pass no MaxResults and take whatever one call
    returned, which happens to be enough today -- the 14-version helmet
    group comes back whole -- and silently stops being enough at the
    service cap. This feeds `scan_all_model_metrics`, so the failure
    would be versions quietly missing from the comparison chart and from
    every count on the page, with nothing erroring to say so. That is the
    same shape as `list_s3_images` scoring a 4,000-image dataset on the
    first 1,000 keys.
    """
    hit = _listing_cache.get(group_name)
    if hit and max_age > 0 and (time.time() - hit[0]) < max_age:
        return hit[1]

    try:
        packages = []
        token = ""
        while True:
            kwargs = {"ModelPackageGroupName": group_name, "SortBy": "CreationTime",
                      "SortOrder": "Descending", "MaxResults": 100}
            if token:
                kwargs["NextToken"] = token
            page = sm.list_model_packages(**kwargs)
            packages.extend(page["ModelPackageSummaryList"])
            token = page.get("NextToken") or ""
            if not token:
                break
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
        # Through `fetch_metrics_and_lineage` rather than describing here,
        # so this shares that function's per-ARN cache. The metrics scan
        # describes the very same versions moments later; done separately
        # that was every version fetched twice per page load.
        _metrics, description, lineage = fetch_metrics_and_lineage(pkg["ModelPackageArn"])
        pkg["Description"] = description
        pkg["Lineage"] = lineage

    summary = {"total": len(packages), "approved": approved, "packages": packages}
    _listing_cache[group_name] = (time.time(), summary)
    return summary


def all_model_registries() -> dict:
    """One summary per real serving size, keyed by group name."""
    return {f"yolo11{size}-detector": model_registry_summary(f"yolo11{size}-detector") for size in REGISTRY_SIZES}


# The group open-taxonomy runs used before per-size groups existed. Listed
# so models registered then stay visible rather than becoming unreachable.
# Spelled out rather than aliasing CUSTOM_REGISTRY_GROUP, which is defined
# much further down this module and is not bound yet at import time here.
LEGACY_OPEN_REGISTRY_GROUP = "track2-yolo-custom-detector"


def builtin_open_groups() -> list:
    """Every (generation, size, group) an open-taxonomy run on the BUILTIN
    detector can register into.

    This list was built in two places -- the metrics scan and the
    base-model picker -- and both spelled the generation literally as
    `yolo11{size}`. Once YOLO26 shipped, an open run registered into
    `yolo26s-open-detector` and then appeared on neither: trained,
    Approved, and invisible, with no error anywhere to say so. Named
    detectors already enumerate generation-aware through
    `detector_groups`; this is the builtin catching up to them.

    A group that has never been created is not a problem to enumerate --
    `model_registry_summary` reports a missing group as empty rather than
    raising -- so listing generation 26 before anything registers there
    costs an empty section, not an error.
    """
    return [
        (generation, size, open_registry_group(f"yolo{generation}{size}"))
        for generation, sizes in GENERATION_SIZES.items()
        for size in sizes
        if f"yolo{generation}{size}" in KNOWN_ARCHITECTURES
    ]


def open_model_registries() -> dict:
    """One summary per open-taxonomy group: the four per-size groups plus
    the legacy single group.

    Deliberately NOT merged into all_model_registries. That feeds
    serialize_registries_brief, which the Deploy, Eval and Export pages
    read -- and the inference path maps class indices to the production
    person/helmet labels, so offering a six-class model there would
    mislabel every prediction without failing. These are listed where
    models are reviewed, not where they are deployed.
    """
    groups = [open_registry_group(a) for a in KNOWN_ARCHITECTURES] + [LEGACY_OPEN_REGISTRY_GROUP]
    # Every non-builtin detector's per-size groups. The builtin's groups are
    # the production ones, already covered by all_model_registries.
    for detector in list_detectors():
        if detector["builtin"]:
            continue
        # HISTORY groups: after a rename the versions trained under the
        # old identifier are still in the old groups, and a summary that
        # showed only the current name would report a detector with a
        # long history as having none.
        groups += [group for _gen, _size, group in detector_history_groups(detector)]
    return {group: model_registry_summary(group) for group in dict.fromkeys(groups)}


def serialize_registries_brief(include_open: bool = False) -> dict:
    """`all_model_registries()`, JSON-serializable and trimmed to what the
    Deploy/Export pages need (no lineage/description) -- shared so both
    routers list registry versions identically.

    include_open adds the open-taxonomy groups. Off by default, and
    deliberately so for Deploy: serving a six-class model through an
    inference path that reads index 0 as `person` mislabels every
    prediction without failing. Export has no such consumer -- it copies a
    .pt to a folder -- so it passes True.
    """
    registries = all_model_registries()
    if include_open:
        registries = {**registries, **open_model_registries()}
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


def _read_evaluation_dataset(data: dict) -> dict:
    """Where a registered version's training images came from.

    The two evaluators name the source differently: the production one
    records `snapshot_uri` (the pinned pull it was derived from), the
    open-taxonomy one records `dataset_uri` (the folder chosen directly).
    Both record `dataset_version`, which is the thing that actually pins
    the data -- a content hash for a cardless folder, the card's version
    otherwise.

    Empty strings rather than None so the UI can render the block without
    a null check per field; an absent source is shown as unknown, which is
    the honest answer for the legacy flat reports that recorded none.
    """
    return {
        "source_uri": data.get("snapshot_uri") or data.get("dataset_uri") or "",
        "dataset_version": data.get("dataset_version") or "",
        # "dataset-card" or "file-contents" -- whether the version came from
        # a prepared card or was hashed from the files themselves.
        "provenance": data.get("dataset_provenance") or "",
        "validation_count": data.get("validation_count"),
        "baseline_uri": data.get("baseline_uri") or "",
    }


def _read_evaluation_classes(data: dict) -> list:
    """The class names a registered model actually detects.

    Two report shapes carry it. The open-taxonomy evaluator writes a
    top-level "classes" map ({"0": "helmet", ...}); the production one
    writes no such map but does write "per_class", keyed by class name.
    Reading both means a version's taxonomy is visible whichever pipeline
    produced it -- which for a six-class model is the difference between
    reading its numbers correctly and misreading them as person/helmet.
    """
    classes = data.get("classes")
    if isinstance(classes, dict) and classes:
        # Keys are the class indices; order by them, not by insertion.
        return [name for _index, name in sorted(classes.items(), key=lambda kv: int(kv[0]))]
    if isinstance(classes, list) and classes:
        return [str(name) for name in classes]
    per_class = data.get("per_class")
    if isinstance(per_class, dict) and per_class:
        return list(per_class)
    return []


#: What has already been described, by ARN.
#:
#: A registered version's metrics, description and lineage are written
#: when it is registered and never change afterwards -- the only mutable
#: thing about a package is its approval status, and that comes off the
#: LIST call, not this one. So this is a cache with no staleness to it.
#:
#: It matters because the registry now holds 53 groups, and every load of
#: Review & Metrics described every version in all of them. That is the
#: traffic that was tripping SageMaker's rate limiter.
_metrics_cache: dict[str, tuple] = {}
#: Nothing here is ever invalidated, so it needs a ceiling. Well above
#: any plausible version count; it only exists so a long-lived container
#: cannot grow without bound.
_METRICS_CACHE_MAX = 4000


def fetch_metrics_and_lineage(model_package_arn: str) -> tuple[dict | None, str, dict | None]:
    """One describe_model_package call -> (metrics-or-None, description,
    lineage-or-None). Metrics: the 5 flat numbers (mAP50, mAP50-95,
    precision, recall, accuracy) from evaluation.json recorded on
    ModelMetrics at registration time. Baseline registrations attach no
    ModelMetrics at all -- None for those.

    Answered from `_metrics_cache` when this ARN has been described
    before, which is almost always: the set of versions grows slowly and
    the page is refreshed constantly."""
    cached = _metrics_cache.get(model_package_arn)
    if cached is not None:
        return cached

    try:
        desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    except Exception as exc:
        # NOT cached. A throttle or a blip is a reason to try again next
        # time, and caching it would turn one bad moment into a row that
        # stays blank until the container restarts.
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
            report = json.loads(body)
            metrics = _read_evaluation_metrics(report)
            # Carried on the metrics dict rather than as a separate return
            # value: every caller already threads `metrics` through, and a
            # fourth tuple member would have to be handled at each one.
            metrics["classes"] = _read_evaluation_classes(report)
            metrics["dataset"] = _read_evaluation_dataset(report)
        except Exception as exc:
            print(f"Could not fetch evaluation metrics for {model_package_arn}: {exc}")

    answer = (metrics, description, lineage)
    if len(_metrics_cache) < _METRICS_CACHE_MAX:
        _metrics_cache[model_package_arn] = answer
    return answer


def exportable_groups() -> list:
    """Every registry group a model COULD be registered into, whether or
    not anything is in one yet.

    The distinction matters. Reading the groups off the versions that
    exist -- which is what the export card did -- makes a Class that has
    only ever trained at YOLO11-S offer YOLO11 and S and nothing else, so
    there is no way to set up an export for the yolo26x group before its
    first run lands. An empty group is not a missing one, and
    `model_registry_summary` reports one that has never been created as
    empty rather than raising, so enumerating ahead costs nothing.

    Each entry carries the generation, which the metrics scan discarded.
    Naming is NEVER rebuilt here: production is the four fixed names,
    the open groups come from `builtin_open_groups` and a Class's own
    come from `detector_groups`. Rebuilding "yolo{gen}{size}-{slug}" by
    hand is how the Models card once told somebody a YOLO26 run would
    register into `yolo11s-helmet-detector` while the run used the right
    one -- a card and a run disagreeing about where a model was going.
    """
    out = [{"registry": "production", "generation": DEFAULT_GENERATION,
            "size": size, "group_name": f"yolo11{size}-detector"}
           for size in REGISTRY_SIZES]
    # Filled in at the end, for every entry at once -- see below.
    out += [{"registry": "custom", "generation": generation, "size": size,
             "group_name": group}
            for generation, size, group in builtin_open_groups()]
    # No generation and no size: its name does not state either, and
    # inventing one would file it under a generation it never had.
    out += [{"registry": "custom", "generation": "", "size": "",
             "group_name": LEGACY_OPEN_REGISTRY_GROUP}]
    for detector in list_detectors():
        if detector["builtin"]:
            continue
        # History, not just the current name -- see all_open_registries.
        out += [{"registry": detector["slug"], "generation": generation,
                 "size": size, "group_name": group}
                for generation, size, group in detector_history_groups(detector)]
    seen = set()
    unique = []
    for entry in out:
        if entry["group_name"] in seen:
            continue
        seen.add(entry["group_name"])
        # The stem an export of this group will actually carry. Handed
        # over rather than left to the caller: the card was composing it
        # as `group_name.replace(/-detector$/, "")`, which gives
        # `yolo11s-helmet` where the file lands as `yolo11-s-helmet` --
        # so the line promising where a file would be written named a key
        # that never appears in the bucket.
        entry["export_stem"] = export_stem(entry["group_name"])
        unique.append(entry)
    return unique


def warm_registry_cache() -> dict:
    """Fill the registry caches so the first page load does not pay for them.

    Review & Metrics was measured at 9-12 seconds on a cold container and
    effectively instant once warm. Every deploy ends every session, so
    without this the first person back in after a deploy always pays the
    cold price -- and a page that takes twelve seconds the first time
    reads as broken, not as busy.

    Deliberately swallows everything. This is a warm-up: if SageMaker is
    throttling or a group will not list, the routes still work, they are
    just cold. A start-up that fell over because a cache could not be
    pre-filled would be a worse trade than a slow first page.
    """
    started = time.time()
    try:
        rows = scan_all_model_metrics()
    except Exception as exc:                                       # noqa: BLE001
        print(f"Registry warm-up skipped: {exc}")
        return {"warmed": False, "rows": 0, "seconds": 0.0}
    took = time.time() - started
    print(f"Registry warm-up: {len(rows)} versions in {took:.1f}s "
          f"({len(_listing_cache)} groups, {len(_metrics_cache)} versions cached).")
    return {"warmed": True, "rows": len(rows), "seconds": round(took, 1)}


async def registry_warm_loop() -> None:
    """Keep the registry cache warm for as long as the app is up.

    The start-up warm-up alone was not enough. It filled the cache, the
    entries aged out before anybody arrived, and the first real request
    paid the full price regardless -- 7.6 seconds, measured over HTTP
    against the warmed process.

    Refreshes just inside the TTL, so an entry is replaced while it is
    still valid and no request ever waits for a cold listing. In a
    thread each time, because the work is blocking boto3 and this is
    running on the event loop that is also serving requests.
    """
    import asyncio

    while True:
        try:
            await asyncio.sleep(_LISTING_REFRESH_SECONDS)
            await asyncio.to_thread(warm_registry_cache)
        except asyncio.CancelledError:
            raise
        except Exception as exc:                                   # noqa: BLE001
            # Never let the refresher die: a cache that stops being
            # refreshed degrades silently into the slow page this exists
            # to prevent.
            print(f"Registry refresh failed, will try again: {exc}")


def scan_all_model_metrics() -> list:
    """One entry per registered model package version across every size --
    each row also carries training notes and full lineage."""
    rows = []
    # Production groups, open-taxonomy groups and every Class's own, each
    # row saying which registry it came from. Previously only the
    # production groups were scanned, so a custom-trained model's metrics
    # existed in the registry and were invisible on this page.
    #
    # ONE enumeration, shared with the export card. Two lists of "every
    # group" is two chances for a group to be exportable and unchartable,
    # or the reverse.
    scanned = [(g["group_name"], g["size"], g["registry"]) for g in exportable_groups()]
    seen_groups = set()
    groups = []
    for group_name, size, registry in scanned:
        if group_name in seen_groups:
            continue
        seen_groups.add(group_name)
        groups.append((group_name, size, registry))

    # The page was taking 12-15 seconds to show a single row, which reads
    # as broken rather than busy. Nearly all of it is network wait, not
    # work: one ListModelPackages per group, then one
    # DescribeModelPackage per version plus whatever lineage and
    # evaluation.json each pulls behind it.
    #
    # The LISTING stays serial. Running it eight-wide was tried and
    # SageMaker threw `ListModelPackages ... (reached max retries: 4):
    # Rate exceeded`, which botocore raises and which 500s this whole
    # page. It turned out to throttle SERIALLY too: the registry holds 53
    # groups, which is 53 list calls in a row, and that alone was enough
    # to exhaust the default retry policy. The client now uses adaptive
    # retries (see `sm` at the top of this module) and each group is
    # allowed to fail on its own rather than taking the page with it.
    ordered = []
    for group_name, size, registry in groups:
        try:
            summary = model_registry_summary(group_name)
        except Exception as exc:
            # One group that will not list is a group missing from the
            # table, not a reason for the page to show nothing. Before
            # this, the first throttled group raised straight out of the
            # route and Review & Metrics returned a 500 with every other
            # group's rows already in hand.
            print(f"Could not list versions of {group_name}: {exc}")
            continue
        for pkg in summary["packages"]:
            ordered.append((group_name, size, registry, pkg))

    # The DESCRIBES are where the time is -- one per version, and there
    # are far more versions than groups. Four at a time, well under what
    # tripped the limiter above.
    #
    # Threads, not async: botocore clients are safe to call from several
    # threads, these calls are pure I/O, and the rest of this module stays
    # synchronous. `map` keeps input order, so each result still lines up
    # with the package it came from and the rows are assembled in exactly
    # the order the serial loop produced them.
    def describe_all(parallel: bool) -> list:
        if not parallel:
            return [fetch_metrics_and_lineage(item[3]["ModelPackageArn"])
                    for item in ordered]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(
                lambda item: fetch_metrics_and_lineage(item[3]["ModelPackageArn"]),
                ordered,
            ))

    try:
        details = describe_all(parallel=True)
    except botocore.exceptions.ClientError as exc:
        # Throttling is the expected failure if the account is already
        # busy, and a speed-up that can take the page down is not a
        # speed-up. Fall back to the slow path that always worked rather
        # than hand the browser a 500.
        print(f"Parallel metrics scan failed ({exc}); falling back to serial.")
        details = describe_all(parallel=False)

    for (group_name, size, registry, pkg), (metrics, description, lineage) in zip(
            ordered, details):
        rows.append({
            "group_name": group_name,
            "registry": registry,
            # Stated, not inferred by the page. Every row's generation
            # is right there in its group name, and the metrics UI was
            # rendering "YOLO11" over all of them because the row never
            # carried it -- a YOLO26 model charted as a YOLO11 one.
            # "" for the legacy group, whose name does not say.
            "generation": parse_group_name(group_name)[0] or "",
            "size": size,
            "version": pkg["ModelPackageArn"].rsplit("/", 1)[-1],
            "model_package_arn": pkg["ModelPackageArn"],
            "status": pkg["ModelApprovalStatus"],
            "metrics": metrics,
            "description": description,
            "lineage": lineage,
        })
    # Group order matches REGISTRY_SIZES; within a group, VERSION ASCENDING.
    # It used to be best mAP50-95 first, which reads as shuffled the moment
    # several versions score the same -- 3, 5, 4 for three runs that all
    # landed on 0.727, because nothing broke the tie. Version order is the
    # one ordering that is stable, predictable, and matches how the numbers
    # were handed out.
    size_order = {size: i for i, size in enumerate(REGISTRY_SIZES)}

    def version_number(row: dict) -> int:
        """Numeric, not the string off the ARN: as text, v10 sorts before
        v9 and the list silently goes wrong at the tenth version."""
        try:
            return int(str(row.get("version") or 0))
        except ValueError:
            return 0

    # .get, not [] -- the legacy open group has no size, and indexing it
    # would raise KeyError and 500 the whole page. Production rows first,
    # then custom, so the two registries read as distinct blocks.
    rows.sort(key=lambda r: (
        0 if r.get("registry") == "production" else 1,
        size_order.get(r["size"], len(size_order)),
        version_number(r),
    ))
    return rows


# --- S3 browsing / weights ----------------------------------------------------


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


#: What the data.yaml browser offers as selectable files. Both spellings,
#: because Ultralytics writes `.yaml` and plenty of exports write `.yml`.
YAML_FILE_EXTENSIONS = (".yaml", ".yml")


def list_s3_folders_and_yaml_files(prefix: str) -> tuple[list, list]:
    """Folders and the .yaml files beside them, from one round trip.

    The dataset browser used to return folders only, so a Class's class
    names could be read from `<folder>/data.yaml` and from nothing else.
    A dataset whose file is `dataset.yaml`, or which carries several, was
    unreachable -- and the folder tick box said nothing about which file
    it was about to read.
    """
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix, Delimiter="/")
    folders = [p["Prefix"][len(prefix):].rstrip("/") for p in response.get("CommonPrefixes", [])]
    yaml_files = []
    for obj in response.get("Contents", []):
        key = obj["Key"]
        if key == prefix:
            continue
        name = key[len(prefix):]
        if name.lower().endswith(YAML_FILE_EXTENSIONS):
            yaml_files.append(name)
    return folders, yaml_files


def classes_from_yaml_text(text) -> dict:
    """The taxonomy a data.yaml declares, as {index: name}.

    Takes the TEXT, so the same parser serves a file in the bucket and
    one off somebody's laptop. The alternative was a second
    implementation in TypeScript for the browser half, and this
    codebase already carries two hand-rolled readers of this exact key
    (`pull_control._classes_from_yaml` cannot import PyYAML because it
    is the Lambda handler). A third would be the one that drifts.
    """
    manifest = _yaml().safe_load(text) or {}
    names = manifest.get("names") or {}
    # `names: [person, helmet]` is as standard as the mapping form -- it
    # is what a Roboflow export writes.
    if isinstance(names, (list, tuple)):
        names = {index: value for index, value in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def classes_from_yaml_key(key: str) -> dict:
    """The taxonomy a SPECIFIC yaml in the bucket declares.

    Keyed on the exact object rather than a folder, so picking
    `dataset.yaml` reads that file and not the `data.yaml` that may or
    may not sit beside it. `dataset_classes` stays for every caller that
    genuinely has a folder and wants the conventional name.
    """
    key = (key or "").strip().lstrip("/")
    if not key.lower().endswith(YAML_FILE_EXTENSIONS):
        raise ValueError(f"'{key}' is not a .yaml file.")
    return classes_from_yaml_text(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())


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


# Where "upload your own checkpoint" lands. Separate from the folders the
# weights browser already shows, so an uploaded file is never mistaken for
# something the pipeline itself produced.
BASE_MODEL_UPLOAD_PREFIX = "external_models/uploaded"

#: Official Ultralytics COCO releases, mirrored into this bucket once so
#: a run never depends on reaching GitHub.
#:
#: A FOLDER in the pipeline bucket, not a bucket of its own: the instance
#: role already reads and writes here, the folder browser already walks
#: it, and a new bucket would need an IAM grant plus a change to the
#: platform stack -- which has drift, so it would be a targeted AWS
#: change rather than an apply. Nothing about these files needs their own
#: bucket.
YOLO_WEIGHTS_LIBRARY_PREFIX = "track2-yolo-weights"


def list_yolo_weights_library() -> list:
    """The mirrored releases available to import, newest generation first.

    Reads what is ACTUALLY there rather than listing what ought to be:
    the folder is seeded by a script, and a card promising `yolo26l.pt`
    when the upload failed would send somebody looking for a file that
    does not exist.
    """
    out = []
    token = ""
    while True:
        kwargs = {"Bucket": BUCKET, "Prefix": f"{YOLO_WEIGHTS_LIBRARY_PREFIX}/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        for obj in page.get("Contents", []):
            name = obj["Key"].rsplit("/", 1)[-1]
            if not name.endswith(".pt"):
                continue
            architecture = name[: -len(".pt")].lower()
            if architecture not in KNOWN_ARCHITECTURES:
                continue
            out.append({
                "architecture": architecture,
                "key": obj["Key"],
                "size_bytes": int(obj.get("Size") or 0),
            })
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken") or ""
        if not token:
            break
    out.sort(key=lambda row: row["architecture"])
    return out


# Randomly-initialised seeds, one per architecture, built by
# model_training/custom_training/make_scratch_checkpoint.py. "Train from
# scratch" is expressed as "fine-tune from a checkpoint whose weights are
# random", because no training path in this estate can express "no base
# model" -- see that script's docstring.
SCRATCH_SEED_PREFIX = "external_models/scratch"


def resolve_scratch_seed_uri(architecture: str) -> str:
    """The random-init seed for an architecture, or a named error."""
    architecture = (architecture or "").strip().lower()
    if not architecture:
        raise ValueError("Pick an architecture to train from scratch.")
    key = f"{SCRATCH_SEED_PREFIX}/{architecture}/model.pt"
    if not _s3_object_exists(key):
        raise ValueError(
            f"No scratch seed for '{architecture}' yet (looked for s3://{BUCKET}/{key}). "
            "Generate one with model_training/custom_training/make_scratch_checkpoint.py -- "
            "it must run inside the training image, not on a workstation."
        )
    return f"s3://{BUCKET}/{key}"


def resolve_base_model_key(key: str) -> str:
    """An exact object key -> its s3:// URI.

    This is the "pick any .pt in the bucket" path. resolve_base_model_s3_uri
    below only ever looked for `<folder>/model.tar.gz` or `<folder>/model.pt`,
    a convention that holds for uploads and for nothing else -- so a real
    weights key like `ming_yang_yolos/<run>/weights/best.pt` was unreachable
    even though the trainer accepts any key.
    """
    key = (key or "").strip().lstrip("/")
    if not key:
        raise ValueError("No base model file selected.")
    if not key.lower().endswith(WEIGHT_FILE_EXTENSIONS):
        raise ValueError(
            f"'{key}' is not a checkpoint -- pick a file ending in "
            + " or ".join(WEIGHT_FILE_EXTENSIONS) + "."
        )
    if not _s3_object_exists(key):
        raise ValueError(f"s3://{BUCKET}/{key} does not exist.")
    return f"s3://{BUCKET}/{key}"


def scratch_refusal(detector_slug: str, architecture: str) -> str:
    """Why this Class may not be trained from scratch, or "".

    A Class with registered versions has a LINEAGE: v3 continues v2
    continues v1, and that is the whole meaning of the version number.
    Training it from random weights produces a v4 that continues
    nothing -- same group, same name, unrelated model -- and every
    comparison made afterwards between v3 and v4 is a comparison of two
    different things. There is no way to see that from the registry,
    because the only thing that would say so is the lineage the run
    just broke.

    Scratch is for a Class's FIRST model. After that, start from a
    version; to restart deliberately, make a new Class, which costs
    nothing and keeps both histories readable.

    Returns a reason rather than a bool so the caller can say it.
    Empty when the run is allowed -- including when the registry cannot
    be read, because refusing a run over a listing failure would make an
    AWS hiccup look like a policy.
    """
    detector_slug = (detector_slug or "").strip()
    if not detector_slug:
        return ""
    try:
        detector = get_detector(detector_slug)
        group = detector_group_name(detector, architecture or "")
        packages = model_registry_summary(group).get("packages") or []
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not check {detector_slug} for existing versions: {exc}")
        return ""
    if not packages:
        return ""
    newest = packages[0].get("ModelPackageArn", "").rsplit("/", 1)[-1]
    count = len(packages)
    return (
        f"Training from scratch is not permitted -- {group} already has "
        f"{count} registered version{'' if count == 1 else 's'} "
        f"(newest v{newest}). Random weights would register a version that "
        f"continues nothing, under a number that says it continues the "
        f"last. Delete the existing version{'' if count == 1 else 's'}, or "
        f"choose another Class, model or size."
    )


def resolve_base_model_source(*, scratch: bool = False, scratch_architecture: str = "",
                              base_model_key: str = "",
                              base_model_folder: str = "", base_model_package_arn: str = "") -> tuple:
    """(s3_uri, source_label) for exactly one of the four ways to choose a
    base model.

    One dispatcher rather than a pairwise check repeated in the router, the
    two start_* functions and the queue: there are four inputs now, and a
    four-way check duplicated in four places is how they drift apart.
    """
    chosen = [
        # Keyed on the flag, not on the architecture: choosing "from
        # scratch" without yet picking an architecture is an incomplete
        # scratch selection, and must say so rather than report that no
        # source was chosen at all.
        ("scratch", bool(scratch)),
        ("file", bool(base_model_key)),
        ("folder", bool(base_model_folder)),
        ("package", bool(base_model_package_arn)),
    ]
    picked = [name for name, given in chosen if given]
    if len(picked) != 1:
        raise ValueError(
            "Pick exactly one base model source: train from scratch, a .pt in the bucket, "
            "an uploaded checkpoint, or a registered package"
            + (f" (got {len(picked)}: {', '.join(picked)})." if picked else " (got none).")
        )
    source = picked[0]
    if source == "scratch":
        return resolve_scratch_seed_uri(scratch_architecture), source
    if source == "file":
        return resolve_base_model_key(base_model_key), source
    if source == "folder":
        return resolve_base_model_s3_uri(base_model_folder), source
    return resolve_package_model_artifact(base_model_package_arn), source

# What an uploaded checkpoint is allowed to declare itself as. Ultralytics
# records the config a model was built from inside the checkpoint, so the
# declaration can be checked rather than believed -- see inspect_checkpoint.
#
# DERIVED from GENERATION_SIZES rather than listed separately: this estate
# trains yolo11 s/m/l/x and nothing else, and a hand-maintained second list
# drifts from the size dropdown the moment either changes.
KNOWN_ARCHITECTURES = tuple(f"yolo{generation}{size}"
                            for generation, sizes in GENERATION_SIZES.items()
                            for size in sizes)


def _base_model_upload_key(filename: str, architecture: str) -> tuple:
    """(key, folder, normalised architecture) for an uploaded checkpoint."""
    architecture = (architecture or "").strip().lower()
    if architecture not in KNOWN_ARCHITECTURES:
        raise ValueError(
            f"'{architecture}' is not a known architecture. Pick one of: "
            + ", ".join(KNOWN_ARCHITECTURES)
        )
    name = os.path.basename((filename or "").strip().replace("\\", "/"))
    if not name.lower().endswith(".pt"):
        raise ValueError("Upload a YOLO .pt checkpoint.")
    folder = f"{BASE_MODEL_UPLOAD_PREFIX}/{architecture}/{uuid.uuid4().hex[:12]}"
    return f"{folder}/model.pt", folder, architecture


def upload_base_model_fileobj(fileobj, filename: str, architecture: str) -> dict:
    """Streams an uploaded checkpoint from the request straight into S3.

    Same-origin, unlike the presigned PUT below: that goes from the browser
    directly to S3, so the BUCKET's CORS policy decides whether it is
    allowed, and track2-s3 permits only *.sagemaker.aws origins. Adding this
    app's origin there is not durable either -- the review host has no
    Elastic IP, so its address changes on most scheduled restarts.

    upload_fileobj streams in parts rather than reading the file into
    memory, which matters because these checkpoints run to hundreds of
    megabytes and the API shares one c5d.large with everything else.
    """
    key, folder, architecture = _base_model_upload_key(filename, architecture)
    s3.upload_fileobj(fileobj, BUCKET, key)
    return {"key": key, "folder": folder, "s3_uri": f"s3://{BUCKET}/{key}",
            "declared_architecture": architecture}


def presign_base_model_upload(filename: str, architecture: str) -> dict:
    """A presigned PUT the browser uploads straight to S3 with.

    Deliberately not a multipart POST through this app: these checkpoints
    run from ~6MB (yolo11n) to well over 100MB (yolov8x), and the API
    container shares a c5d.large with everything else on the host. Sending
    the bytes through it buys nothing and risks the whole app for the sake
    of one file.
    """
    architecture = (architecture or "").strip().lower()
    if architecture not in KNOWN_ARCHITECTURES:
        raise ValueError(
            f"'{architecture}' is not a known architecture. Pick one of: "
            + ", ".join(KNOWN_ARCHITECTURES)
        )
    name = os.path.basename((filename or "").strip().replace("\\", "/"))
    if not name.endswith(".pt"):
        raise ValueError("Upload a YOLO .pt checkpoint.")
    # Slugged, not trusted: the name reaches an S3 key, and the browser is
    # free to send whatever it likes here.
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", name[: -len(".pt")])[:64] or "checkpoint"
    folder = f"{BASE_MODEL_UPLOAD_PREFIX}/{architecture}/{uuid.uuid4().hex[:12]}"
    key = f"{folder}/model.pt"
    url = s3.generate_presigned_url(
        "put_object",
        Params={"Bucket": BUCKET, "Key": key, "ContentType": "application/octet-stream"},
        ExpiresIn=3600,
    )
    return {
        "upload_url": url,
        "key": key,
        "folder": folder,
        "s3_uri": f"s3://{BUCKET}/{key}",
        "declared_architecture": architecture,
        "original_filename": name,
        "slug": stem,
    }


def _architecture_from_checkpoint(path: str) -> dict:
    """Reads the architecture and class names back out of a YOLO .pt.

    Ultralytics stores both the config the model was built from and the
    args it was trained with, but not consistently in one place across
    versions, so both are tried before giving up.
    """
    import torch  # heavy: imported here so it never costs app startup

    # weights_only=True cannot load an Ultralytics checkpoint (it pickles a
    # live nn.Module), so this is a full unpickle -- see the note in
    # inspect_checkpoint about who is allowed to reach it.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        ckpt = {"model": ckpt}

    model = ckpt.get("model")
    names = getattr(model, "names", None) or ckpt.get("names") or {}
    if isinstance(names, (list, tuple)):
        names = {i: n for i, n in enumerate(names)}
    names = {int(k): str(v) for k, v in dict(names).items()}

    yaml_cfg = getattr(model, "yaml", None)
    yaml_cfg = yaml_cfg if isinstance(yaml_cfg, dict) else {}
    scale = str(yaml_cfg.get("scale") or "").strip().lower()

    candidates = [yaml_cfg.get("yaml_file")]
    train_args = ckpt.get("train_args")
    if isinstance(train_args, dict):
        candidates.append(train_args.get("model"))

    detected = ""
    for candidate in candidates:
        if not candidate:
            continue
        stem = os.path.basename(str(candidate)).split(".")[0].strip().lower()
        if stem in KNOWN_ARCHITECTURES:
            detected = stem
            break
        # "yolov8.yaml" + scale "n" is how Ultralytics describes a model
        # built from the family config rather than a per-size one.
        if scale and f"{stem}{scale}" in KNOWN_ARCHITECTURES:
            detected = f"{stem}{scale}"
            break

    return {"architecture": detected, "classes": names}


def assert_inspectable_checkpoint(key: str) -> str:
    """Guards what may be read back as a checkpoint.

    Reading one means unpickling it, which can execute code inside the API
    container -- the one holding the estate's credentials. So the file has
    to be a real checkpoint that already exists in THIS bucket: either one
    this app just took as an upload, or one already sitting in the bucket,
    which is the same set the training form will hand to the trainer as
    starting weights. What it excludes is an arbitrary path, another
    bucket, or a name for an object that is not there.
    """
    key = (key or "").strip().lstrip("/")
    if key.startswith(BASE_MODEL_UPLOAD_PREFIX + "/"):
        return key
    # Reuses the bucket-scoped checks: right extension, actually exists.
    resolve_base_model_key(key)
    return key


def inspect_checkpoint(key: str, declared_architecture: str) -> dict:
    """Downloads an uploaded checkpoint and checks it is what it claims.

    The declaration is the operator's; this is the checkpoint's own answer.
    Reported side by side rather than one silently overriding the other --
    the same thing the training audit does with submitted-vs-actual
    hyperparameters, and for the same reason: a mislabelled base model
    follows a trained model through the registry forever.

    Trust boundary worth stating plainly: reading the architecture means
    unpickling the file, which can run code. Only an authenticated
    operator can reach this, and the training container already loads the
    same checkpoint the same way -- but this runs it inside the API
    container, which holds the estate's credentials.
    """
    declared = (declared_architecture or "").strip().lower()
    with tempfile.TemporaryDirectory() as work:
        local = os.path.join(work, "model.pt")
        s3.download_file(BUCKET, key, local)
        size_bytes = os.path.getsize(local)
        try:
            found = _architecture_from_checkpoint(local)
        except Exception as exc:
            raise ValueError(f"Could not read this file as a YOLO checkpoint: {exc}")

    detected = found["architecture"]
    classes = found["classes"]
    if not classes:
        raise ValueError("This checkpoint declares no class names -- it is not a usable detector.")
    return {
        "key": key,
        "s3_uri": f"s3://{BUCKET}/{key}",
        "folder": key.rsplit("/", 1)[0],
        "declared_architecture": declared,
        "detected_architecture": detected,
        # Unknown is not a mismatch: a checkpoint from a fork or an older
        # Ultralytics may simply not say, and failing the upload over that
        # would be worse than recording what we do know.
        "architecture_matches": (not detected) or (detected == declared),
        "architecture_known": bool(detected),
        "classes": classes,
        "size_bytes": size_bytes,
    }


# The serve container recorded on a registered version. Copied from the
# open pipeline's RegisterModel step so an imported version and a trained
# one are described identically -- anything that reads one must be able to
# read the other without knowing which it is.
SERVE_IMAGE = "125192671932.dkr.ecr.ap-southeast-1.amazonaws.com/track2-yolo-train:serve-latest"

IMPORTED_WEIGHTS_PREFIX = "external_models/registered"


#: Prefixed to the description of a version registered as a SEED -- one
#: whose weights detect something other than the Class it is filed under.
#: On the description rather than only in metadata because that is the
#: field the metrics page, the base-model picker and the registry console
#: all already show.
#: The short token everything tests for. One string, because the browser
#: reads it off the description too (`isSeedVersion` in web/src/lib/slug.ts)
#: and a longer sentence drifting on one side would silently stop
#: matching on the other.
SEED_MARKER = "SEED --"
SEED_DESCRIPTION_PREFIX = f"{SEED_MARKER} NOT A DETECTOR FOR THIS CLASS YET."


def is_seed_description(description: str) -> bool:
    """Was this version registered as a seed rather than trained?

    NOT "is it version 1". A v1 produced by a training run is a real
    model, and calling it a seed would be a lie in the one field whose
    job is saying what a version is.
    """
    return str(description or "").lstrip().startswith(SEED_MARKER)


def _slice_detect_head(net, keep: list) -> dict:
    """Cut a YOLO detect head down to `keep`, renumbered 0..n-1.

    The class scores live in the head's `cv3` branch -- three levels, each
    ending in a 1x1 `Conv2d` with one output channel per class. Keeping a
    subset is taking those rows out of the weight and the bias; everything
    else in the network is untouched, which is why the kept classes come
    out bit-for-bit identical rather than merely similar. Measured, not
    assumed: max absolute difference 0.0 on both the class scores and the
    box predictions.

    The box branch (`cv2`) is not touched at all. Boxes are predicted
    class-agnostically and then scored, so dropping a class removes
    scores, never geometry.
    """
    import torch

    head = net.model[-1]
    if getattr(head, "end2end", False):
        # v10-style heads carry a second one2one branch that would have to
        # be cut in step. Refused rather than half-done: a head sliced on
        # one branch and not the other predicts two different taxonomies.
        raise ValueError(
            "This checkpoint has an end-to-end head, which carries a second "
            "set of class outputs. Slicing it is not supported -- import the "
            "whole file instead."
        )
    if not hasattr(head, "cv3") or not hasattr(head, "nc"):
        raise ValueError(
            "This checkpoint's head is not the detect head slicing knows how "
            "to cut. Import the whole file instead."
        )

    # The CALLER'S order, not numeric order. When somebody types their own
    # class names, their first line is class 0 -- so keeping [7, 2] has to
    # mean {0: whatever 7 was, 1: whatever 2 was}. Sorting here would
    # silently file every label under the wrong name, and nothing would
    # error: the model would simply be wrong about what it detects.
    ordered: list = []
    for raw in keep:
        value = int(raw)
        if value not in ordered:
            ordered.append(value)
    index = torch.as_tensor(ordered, dtype=torch.long)
    if len(index) == 0:
        raise ValueError("No classes chosen, so there would be nothing to detect.")
    if int(index.max()) >= head.nc:
        raise ValueError(
            f"Class {int(index.max())} is not in this checkpoint, which has "
            f"{head.nc}."
        )

    for level in range(len(head.cv3)):
        conv = head.cv3[level][-1]
        if getattr(conv, "out_channels", None) != head.nc:
            raise ValueError(
                "The last layer of the class branch is not the class "
                "convolution; this head is shaped differently than expected."
            )
        new = torch.nn.Conv2d(
            conv.in_channels, len(index), conv.kernel_size,
            conv.stride, conv.padding, bias=conv.bias is not None,
        )
        with torch.no_grad():
            new.weight.copy_(conv.weight[index])
            if conv.bias is not None:
                new.bias.copy_(conv.bias[index])
        new.to(conv.weight.device, conv.weight.dtype)
        head.cv3[level][-1] = new

    old_names = dict(net.names)
    head.nc = int(len(index))
    # Outputs per anchor: the box distribution plus one score per class.
    head.no = head.nc + head.reg_max * 4
    names = {i: old_names[int(k)] for i, k in enumerate(index)}
    net.nc = head.nc
    net.names = names
    if isinstance(getattr(net, "yaml", None), dict):
        net.yaml["nc"] = head.nc
    return names


def slice_checkpoint_classes(key: str, keep: list, *, architecture: str = "") -> dict:
    """A new .pt that detects only `keep`, written beside the original.

    WHY THIS EXISTS. Picking a subset of a checkpoint's classes used to
    produce a SEED: the registry said three classes, the weights still
    predicted eighty, and only a training run reconciled them. That is the
    right answer when the subset is a starting point for new data. It is
    the wrong answer when the weights are already good and the extra
    classes are simply unwanted -- then there is nothing to learn, and a
    training run is an hour of GPU spent reproducing what the file already
    does.

    Cutting the head gives the same model minus some outputs. The kept
    classes predict exactly what they predicted before, so the result is
    deployable immediately and is not a seed.

    The original is never modified: the slice is a new key, so the
    eighty-class file is still there to cut differently later.
    """
    from ultralytics import YOLO

    # Order-preserving dedupe, for the reason in `_slice_detect_head`:
    # the caller's order is the new numbering.
    seen: set = set()
    keep = [int(k) for k in keep if not (int(k) in seen or seen.add(int(k)))]
    if not keep:
        raise ValueError("No classes chosen, so there would be nothing to detect.")

    with tempfile.TemporaryDirectory() as work:
        local = os.path.join(work, "model.pt")
        s3.download_file(BUCKET, key, local)
        model = YOLO(local)
        before = dict(model.model.names)
        names = _slice_detect_head(model.model, keep)

        out = os.path.join(work, "sliced.pt")
        # Through Ultralytics' own save, so the file it writes is the file
        # it knows how to load -- a hand-rolled torch.save of the state
        # dict loses the metadata that tells it what this is.
        model.save(out)

        stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        folder = key.rsplit("/", 1)[0] if "/" in key else ""
        new_key = (f"{folder}/" if folder else "") + f"{stem}-{len(keep)}cls-{stamp}.pt"
        s3.upload_file(out, BUCKET, new_key)

    return {
        "key": new_key,
        "classes": names,
        "kept": keep,
        "was": before,
        "architecture": architecture,
    }


def register_uploaded_base_model(
    *, key: str, detector_slug: str, architecture: str, notes: str = "",
    allow_class_mismatch: bool = False,
) -> dict:
    """Registers an already-uploaded .pt as a version in a detector's registry.

    Training is not the only way a model arrives. A pretrained checkpoint
    from elsewhere was previously usable as a starting point but could not
    be *recorded* -- it had no version number, no place in the registry and
    no lineage, so "which weights did that run start from" was answerable
    only by an S3 key in the audit row.

    The checkpoint is re-read here rather than trusted: by default its own
    class names must match the detector's exactly, for the same reason a
    dataset's must. Registering a ship checkpoint into the helmet registry
    is a mislabelling that follows every model fine-tuned from it.

    `allow_class_mismatch` is the deliberate exception, and it exists
    because the refusal blocked the import card's whole purpose: a COCO
    `yolo11s.pt` detects eighty things, so under the strict rule it could
    only ever create an eighty-class Class, and could never be filed as
    the first version of a three-class one. Training rebuilds the head for
    whatever the DATASET declares, so the checkpoint's own names genuinely
    do not survive the first run.

    What does not follow is that the record may be silent about it. Until
    that first run, the weights detect one thing and the registry says
    another -- and `pick_best(criterion="newest")` in Export Control will
    happily hand that .pt to somebody. So a mismatched registration is
    marked, in the description every page already shows and in the
    package's own metadata, and it says what it is: a seed, not a
    detector for this Class yet.
    """
    detector = get_detector(detector_slug)
    if detector["builtin"]:
        raise ValueError(
            "The built-in PPE detector owns the production groups, which are "
            "fed by the replay pipeline. Import into one of your own detectors."
        )
    group = detector_group_name(detector, architecture)

    found = inspect_checkpoint(key, architecture)
    expected = detector["classes"]
    actual = {int(k): str(v) for k, v in (found["classes"] or {}).items()}
    mismatched = actual != expected
    if mismatched and not allow_class_mismatch:
        raise ValueError(
            f"This checkpoint detects {', '.join(actual.values()) or 'nothing'}, but the "
            f"'{detector_slug}' detector is {', '.join(expected.values())}. "
            "Import it into a detector whose classes match, or fix the detector."
        )

    # Wrapped as model.tar.gz, the shape every other version in the registry
    # uses. Registering the bare .pt would work for provenance but would
    # break publish_versioned_weights and anything that later deploys it.
    # A FIRST VERSION IS A SEED, whatever its classes say.
    #
    # Decided before the group is touched, because the question is "did
    # this Class have any version before this import" -- and the restart
    # below may delete and recreate an empty group, after which the
    # answer is no longer readable. Computed from the summary rather than
    # from `restart_group_numbering_if_empty`'s return: that returns False
    # for a group that does not exist yet, which is the commonest first
    # import of all.
    #
    # Unknown is not first. If the registry cannot be read, the version
    # is described as an import rather than claimed as a seed -- a seed
    # marker on a Class that already had trained versions would be a lie
    # in the one field whose job is saying what a version is.
    try:
        first_version = not model_registry_summary(group)["total"]
    except Exception:                                          # noqa: BLE001
        first_version = False
    # Before the group is ensured, not after: an empty registry should show
    # v1 rather than carrying the numbering of versions since deleted.
    restart_group_numbering_if_empty(group)
    # Checked, not assumed. This used to be fire-and-forget, so a group
    # that failed to come back left the registration to fail later -- with
    # the artifact already uploaded and the group gone.
    if not ensure_model_package_group(group, f"{detector['label']} -- {architecture}."):
        raise ValueError(
            f"The registry group {group} could not be created, so there is "
            "nowhere to register this checkpoint. Try again in a moment."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    artifact_key = f"{IMPORTED_WEIGHTS_PREFIX}/{group}/{stamp}/model.tar.gz"
    with tempfile.TemporaryDirectory() as work:
        local_pt = os.path.join(work, "model.pt")
        s3.download_file(BUCKET, key, local_pt)
        tarball = os.path.join(work, "model.tar.gz")
        with tarfile.open(tarball, "w:gz") as archive:
            # arcname, not the temp path: the archive must contain exactly
            # "model.pt" at its root, which is what the extractor looks for.
            archive.add(local_pt, arcname="model.pt")
        s3.upload_file(tarball, BUCKET, artifact_key)

    response = sm.create_model_package(
        ModelPackageGroupName=group,
        ModelPackageDescription=(
            (f"{SEED_DESCRIPTION_PREFIX} Its weights detect "
             f"{', '.join(actual.values()) or 'nothing recorded'}; this Class is "
             f"{', '.join(expected.values())}. The head is rebuilt on the first "
             f"training run. {notes}"
             if mismatched else
             # Same marker, so `is_seed_description` and everything that
             # reads it agree: the first version of a Class is where its
             # lineage starts, not a model that has learned this Class.
             f"{SEED_MARKER} First version of this Class -- an imported "
             f"checkpoint, not trained here. Training continues from it. {notes}"
             if first_version else
             f"Imported checkpoint, not trained here. {notes}").strip()
        )[:1024],
        InferenceSpecification={
            "Containers": [{
                "Image": SERVE_IMAGE,
                "Environment": {},
                "ModelDataUrl": f"s3://{BUCKET}/{artifact_key}",
            }],
            "SupportedContentTypes": ["image/jpeg"],
            "SupportedResponseMIMETypes": ["application/json"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.g4dn.xlarge"],
            "SupportedTransformInstanceTypes": ["ml.g4dn.xlarge"],
        },
        # Every value is clamped: this map's values have a service minimum
        # of 1 and maximum of 256 characters, and an empty one fails the
        # whole call.
        # Every value goes through metadata_value: the service rejects the
        # whole call over one bad character, and it does so only after the
        # artifact upload above has already happened.
        CustomerMetadataProperties={
            "model_role": "open-detector",
            "deployment_status": "candidate",
            "architecture": metadata_value(architecture),
            "detector": metadata_value(detector_slug),
            # The distinguishing fact. Everything else in this group was
            # produced by a run; this was not, and its metrics are unknown
            # rather than zero.
            "base_model_source": "import",
            "imported_from": metadata_value(f"s3://{BUCKET}/{key}"),
            "imported_at": metadata_value(datetime.now(timezone.utc).isoformat()),
            "detected_architecture": metadata_value(found["detected_architecture"], "unknown"),
            # Separated with " / " rather than ", ": a comma is not in the
            # service's allowed character set.
            "classes": metadata_value(" / ".join(expected.values()), "unspecified"),
            # What the WEIGHTS say, beside what the Class says. Equal on
            # an ordinary import; on a seed they differ, and that
            # difference is the whole fact somebody needs.
            "checkpoint_classes": metadata_value(
                " / ".join(actual.values()), "unspecified"),
            "class_mismatch": "true" if mismatched else "false",
        },
        # A seed is not an approved detector for this Class -- it is
        # waiting for the run that makes it one. Pending keeps it out of
        # everything that filters on Approved (`export_weights_to_folder`
        # refuses outright) while leaving it usable as a starting point,
        # which the base-model picker allows precisely because a new
        # taxonomy's first run has nothing approved to start from.
        ModelApprovalStatus="PendingManualApproval" if mismatched else "Approved",
    )
    arn = response["ModelPackageArn"]
    print(f"Registered imported checkpoint {key} as {arn}.")
    # A new version in a cached listing is a version the page will not
    # show for another half minute -- and the import form's whole point
    # is seeing it arrive.
    forget_registry_listing(group)
    return {
        "model_package_arn": arn,
        "group_name": group,
        "version": arn.rsplit("/", 1)[-1],
        "artifact_uri": f"s3://{BUCKET}/{artifact_key}",
        "detected_architecture": found["detected_architecture"],
        "classes": expected,
    }


def list_base_model_packages(limit: int = 200) -> list[dict]:
    """Every registered model package across every group, newest first.

    Approval status is reported, not filtered on: a custom-taxonomy run is
    explicitly allowed to start from an unapproved checkpoint (see
    train.py's --allow-unapproved-base), so hiding them here would hide
    the option the custom path exists to offer.
    """
    packages: list[dict] = []
    # The groups this estate owns, named rather than discovered. Discovery
    # needs sagemaker:ListModelPackageGroups, which the review-host role has
    # never been granted -- the call failed with AccessDenied, the exception
    # was swallowed, and the picker rendered "No registered packages found"
    # as though the registry were empty. ListModelPackages IS granted on *,
    # so naming the groups works with no policy change and cannot regress
    # into a silent empty list.
    groups = [(f"yolo11{size}-detector", "production") for size in REGISTRY_SIZES]
    groups += [(group, "custom") for _gen, _size, group in builtin_open_groups()]
    groups += [(LEGACY_OPEN_REGISTRY_GROUP, "custom")]
    # Each detector's own per-size groups. Without these a detector could
    # only ever be trained from a production person/helmet checkpoint --
    # there would be no way to carry a vehicle model from v1 to v2, which
    # is the whole point of giving it its own versioned registry.
    for detector in list_detectors():
        if detector["builtin"]:
            continue
        # History groups too, so a run can still continue from a version
        # trained before the Model Name was renamed. Those weights are
        # exactly as good as they were; only the name moved.
        groups += [(group, detector["slug"])
                   for _gen, _size, group in detector_history_groups(detector)]

    # Deduped on the group NAME, like scan_all_model_metrics: listing one
    # group twice would offer the same package under two registries.
    seen_groups: set[str] = set()
    for group, registry in groups:
        if group in seen_groups:
            continue
        seen_groups.add(group)
        try:
            # Through `model_registry_summary` rather than listing here,
            # so this shares that function's cache. It was its own copy of
            # the same 53-group sweep, which is why this route stayed at
            # 14 seconds after every other page dropped under one.
            page_items = model_registry_summary(group)["packages"]
        except Exception as exc:
            print(f"Could not list packages in {group}: {exc}")
            continue
        for item in page_items:
            packages.append({
                "arn": item["ModelPackageArn"],
                "group": group,
                # Which registry, so the picker can separate production
                # baselines from open-taxonomy versions instead of listing
                # every group flat.
                "registry": registry,
                "version": item.get("ModelPackageVersion"),
                "status": item.get("ModelPackageStatus", ""),
                "approval_status": item.get("ModelApprovalStatus", ""),
                "created_at": item["CreationTime"].isoformat() if item.get("CreationTime") else None,
                "description": item.get("ModelPackageDescription", ""),
            })
    packages.sort(key=lambda p: p.get("created_at") or "", reverse=True)
    return packages[:limit]


def resolve_package_model_artifact(package_arn: str) -> str:
    """A registered package -> the S3 model.tar.gz behind it.

    Resolved here rather than by passing the ARN down to the trainer, for
    two reasons. The training image is pinned by digest and its train.py
    refuses any package that is not Approved, which is exactly the gate a
    custom-taxonomy run needs to be free of -- a new taxonomy's first run
    has nothing approved to start from. And the same train.py applies no
    such gate to an S3 URI (see its resolve_base_model_checkpoint_from_s3),
    so handing it the artifact directly is the supported path, not a
    workaround.

    The ARN is still recorded on the run and on the registered model, so
    resolving it here costs no provenance.
    """
    desc = sm.describe_model_package(ModelPackageName=package_arn)
    containers = (desc.get("InferenceSpecification") or {}).get("Containers") or []
    if not containers or not containers[0].get("ModelDataUrl"):
        raise ValueError(
            f"Package '{package_arn}' has no model artifact to fine-tune from."
        )
    return containers[0]["ModelDataUrl"]


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


def parse_group_name(group_name: str) -> tuple:
    """(generation, size, rest) out of `yolo{generation}{size}[-rest]-detector`.

    (None, None, None) for a name not in that shape -- the legacy custom
    group, or anything hand-made -- so a caller can fall back instead of
    being handed a wrong generation that looks right.

    Exists because the generation was being asserted in two places that
    both had the group name in hand and both wrote `11` into it: the
    export filename and the Model tag the Deploy page reads back.
    """
    stem = group_name.removesuffix("-detector")
    match = re.match(r"^yolo(\d+)([smlx])(-.*)?$", stem)
    if not match:
        return (None, None, None)
    generation, size, rest = match.groups()
    return (generation, size, rest or "")


def registry_source_desc(model_package_arn: str, size: str) -> str:
    """The Model tag the Deploy page reads back as "what is serving".

    The generation comes from the group the package is IN, not from a
    literal -- this was `f"YOLO11-{size.upper()} v{version}"`, so a
    generation-26 package deployed as `YOLO26-S v3` would have been tagged
    `YOLO11-S v3` and become indistinguishable, on the one page whose job
    is saying which model is live, from the generation-11 package at the
    same size and version.

    A group whose name does not parse is reported as itself rather than
    given a generation it never stated.
    """
    version = model_package_arn.rsplit("/", 1)[-1]
    group_name = model_package_arn.rsplit("/", 2)[-2]
    generation, _size, _rest = parse_group_name(group_name)
    if not generation:
        return f"{group_name} v{version}"
    return f"YOLO{generation}-{size.upper()} v{version}"


def _export_stem(group_name: str) -> str:
    """`yolo11s-detector` -> `yolo11-s`; `yolo26s-open-detector` -> `yolo26-s-open`.

    Byte-identical to the old `removeprefix("yolo11")` for every
    generation-11 group, so nothing already exported changes name or stops
    matching the archive-on-collision check above. A group this does not
    recognise keeps its own stem rather than being given a yolo11 prefix
    it never earned.
    """
    generation, size, rest = parse_group_name(group_name)
    if generation is None:
        return group_name.removesuffix("-detector")
    return f"yolo{generation}-{size}{rest}"


#: The same function under a public name. `metrics_export` builds its own
#: export filenames and has to produce EXACTLY these stems, or the two
#: export paths would name the same checkpoint two different ways in the
#: same folder. Aliased rather than renamed so the tests that pin the
#: private name -- including one that reads this module's source -- keep
#: describing the thing they were written about.
export_stem = _export_stem


def export_weights_to_folder(model_package_arn: str, destination_folder: str) -> tuple[str, str | None]:
    """Downloads/extracts model.pt from a registered package's own
    model.tar.gz and uploads it to
    <destination_folder>/yolo<generation>-<size>-v<version>-<YYYYMMDD>.pt --
    self-describing so a folder holding several exports stays legible without
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

    # yolo{generation}{size}[-{slug}]-detector/{version}. The generation is
    # READ from the group, not assumed: `removeprefix("yolo11")` is a no-op
    # on a yolo26 group, so a YOLO26 export landed as
    # `yolo11-yolo26s-open-v3-<date>.pt` -- a generation-26 checkpoint
    # carrying 11 in its name, in a folder whose entire purpose is being
    # legible without opening the files.
    group_name, version = model_package_arn.rsplit("/", 2)[-2:]
    export_date = datetime.now(timezone.utc).strftime("%Y%m%d")
    file_name = f"{_export_stem(group_name)}-v{version}-{export_date}.pt"

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


#: On-demand SageMaker rates, fetched once per process from the AWS
#: Pricing API. Cached because prices move a few times a year and the
#: call is free but slow.
_RATE_CACHE: dict = {}
_RATE_CACHE_LOCK = threading.Lock()


def sagemaker_hourly_rate(instance_type: str, component: str) -> float:
    """$/hour for one SageMaker instance, or 0.0 when it cannot be read.

    From the Pricing API rather than a table in this file: a hard-coded
    price is wrong the first time AWS changes one and nothing says so.
    `component` is "Training" or "Processing" -- the same instance costs
    different amounts in each, and the usage types Cost Explorer reports
    ("APS1-Train:ml.g4dn.xlarge") are built from exactly that pair, so
    the two sources agree by construction.

    Pricing lives only in us-east-1 regardless of where the work runs.
    """
    key = (instance_type, component)
    with _RATE_CACHE_LOCK:
        if key in _RATE_CACHE:
            return _RATE_CACHE[key]
    rate = 0.0
    try:
        client = boto3.client("pricing", region_name="us-east-1")
        got = client.get_products(
            ServiceCode="AmazonSageMaker",
            Filters=[
                {"Type": "TERM_MATCH", "Field": "instanceName", "Value": instance_type},
                {"Type": "TERM_MATCH", "Field": "regionCode", "Value": REGION},
                {"Type": "TERM_MATCH", "Field": "component", "Value": component},
            ],
            MaxResults=5,
        )
        for entry in got.get("PriceList") or []:
            document = json.loads(entry)
            for term in (document.get("terms", {}).get("OnDemand") or {}).values():
                for dimension in (term.get("priceDimensions") or {}).values():
                    price = float(dimension.get("pricePerUnit", {}).get("USD") or 0)
                    if price > 0:
                        rate = price
                        break
                if rate:
                    break
            if rate:
                break
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not price {component} {instance_type}: {exc}")
    with _RATE_CACHE_LOCK:
        _RATE_CACHE[key] = rate
    return rate


def training_cost_today() -> dict:
    """What today's runs have cost, measured from the JOBS.

    Cost Explorer is the authority and lags roughly a day, so on the
    afternoon of a day spent training it reports $0.00 -- which reads as
    "this is free" rather than "ask again tomorrow". That is the one
    reading a cost figure must never produce.

    So this counts what was actually consumed: SageMaker reports
    `BillableTimeInSeconds` per training job, which is the quantity it
    bills on, and processing jobs are billed on wall-clock between start
    and end. Multiplied by the published rate, it agrees with the bill
    to the cent for the jobs it can see -- and it can see them the
    moment they finish rather than a day later.

    An ESTIMATE, and labelled one: it covers training and processing
    only. The Studio notebooks that dominated this account's bill on
    4 October are not here, and neither is storage.
    """
    start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    jobs: list = []
    total = 0.0
    priced = True

    def add(name: str, kind: str, instance: str, seconds: float, status: str,
            *, own: bool, began=None, ended=None) -> None:
        nonlocal total, priced
        component = "Training" if kind == "training" else "Processing"
        rate = sagemaker_hourly_rate(instance, component) if instance else 0.0
        if not rate:
            priced = False
        hours = max(seconds, 0) / 3600.0
        cost = hours * rate
        total += cost
        jobs.append({"name": name, "kind": kind, "instance_type": instance,
                     "seconds": int(max(seconds, 0)), "hours": round(hours, 4),
                     "rate": rate, "cost": round(cost, 4), "status": status,
                     # Whose job this is. The panel lists everything on
                     # the account, because what the bill charges for is
                     # everything -- but "we spent this" and "something
                     # else spent this" are different facts and the row
                     # has to say which.
                     "own_pipeline": own,
                     # WHEN it ran. The panel showed a duration and a
                     # cost and no clock time at all, so "what was the
                     # GPU doing at three o'clock" could not be answered
                     # from the one card that lists every job.
                     #
                     # ISO with the offset, not a local string: the
                     # browser renders it in the reader's own zone, and a
                     # server that formatted it would be guessing which.
                     "started_at": began.isoformat() if began else None,
                     "ended_at": ended.isoformat() if ended else None})

    try:
        summaries = sm.list_training_jobs(
            CreationTimeAfter=start, SortBy="CreationTime",
            SortOrder="Descending", MaxResults=100,
        )["TrainingJobSummaries"]
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not list today's training jobs: {exc}")
        summaries = []
    for summary in summaries:
        name = summary["TrainingJobName"]
        try:
            described = sm.describe_training_job(TrainingJobName=name)
        except Exception as exc:                               # noqa: BLE001
            print(f"Could not describe {name}: {exc}")
            priced = False
            continue
        seconds = described.get("BillableTimeInSeconds")
        if seconds is None:
            # Still running: billed so far is start -> now.
            began = described.get("TrainingStartTime")
            seconds = (datetime.now(timezone.utc) - began).total_seconds() if began else 0
        add(name, "training",
            described.get("ResourceConfig", {}).get("InstanceType", ""),
            float(seconds or 0), described.get("TrainingJobStatus", ""),
            own=is_own_pipeline_job(name),
            began=described.get("TrainingStartTime"),
            ended=described.get("TrainingEndTime"))

    try:
        processing = sm.list_processing_jobs(
            CreationTimeAfter=start, SortBy="CreationTime",
            SortOrder="Descending", MaxResults=100,
        )["ProcessingJobSummaries"]
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not list today's processing jobs: {exc}")
        processing = []
    for summary in processing:
        name = summary["ProcessingJobName"]
        try:
            described = sm.describe_processing_job(ProcessingJobName=name)
        except Exception as exc:                               # noqa: BLE001
            print(f"Could not describe {name}: {exc}")
            priced = False
            continue
        began = described.get("ProcessingStartTime")
        ended = described.get("ProcessingEndTime") or datetime.now(timezone.utc)
        seconds = (ended - began).total_seconds() if began else 0
        add(name, "processing",
            described.get("ProcessingResources", {}).get("ClusterConfig", {}).get("InstanceType", ""),
            float(seconds), described.get("ProcessingJobStatus", ""),
            own=is_own_processing_job(name),
            began=began, ended=described.get("ProcessingEndTime"))

    jobs.sort(key=lambda row: row["cost"], reverse=True)
    return {
        "date": start.date().isoformat(),
        "currency": "USD",
        "estimated": round(total, 4),
        # False when a rate or a job could not be read, so the UI can say
        # "at least this" rather than presenting a short total as final.
        "complete": priced,
        "jobs": jobs,
        "training_seconds": sum(j["seconds"] for j in jobs if j["kind"] == "training"),
        "processing_seconds": sum(j["seconds"] for j in jobs if j["kind"] == "processing"),
        # The same total, split by whose job it was. A day's spend that
        # is mostly somebody else's is a different thing to know than a
        # day's spend that is all ours, and one number cannot say it.
        "estimated_ours": round(sum(j["cost"] for j in jobs if j["own_pipeline"]), 4),
        "estimated_other": round(sum(j["cost"] for j in jobs if not j["own_pipeline"]), 4),
    }


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
        source_desc = registry_source_desc(model_package_arn, size)
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


def safe_key_part(value: str, *, field: str) -> str:
    """A caller-supplied folder, checked before it is built into an S3 key.

    S3 answers a malformed key with a 400, and a 400 is not one of the
    codes a `NoSuchKey` check looks for -- so the ClientError escaped
    `read_run_lineage` and the browser got a 500 with a traceback behind
    it. `?output_folder=../../etc/passwd` was enough to do it.

    Nothing was reachable through it: `..` has no meaning to S3, so the
    key simply did not exist. But a 500 says "this service is broken"
    when the honest answer is "that is not a folder", and a stack trace
    is a poor way to say either.
    """
    cleaned = str(value or "").strip().strip("/")
    if not cleaned:
        raise ValueError(f"{field} is empty.")
    if any(part in ("..", ".") for part in cleaned.split("/")):
        raise ValueError(
            f"{field} {value!r} is not a folder -- '.' and '..' mean nothing in S3."
        )
    if any(ord(ch) < 32 for ch in cleaned):
        raise ValueError(f"{field} contains a character a key cannot hold.")
    return cleaned


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

    key = f"{safe_key_part(output_folder, field='output_folder')}/lineage.json"
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


def get_training_progress(pipeline_name: str = "") -> dict | None:
    """None if no execution has ever run. Otherwise the latest execution's
    overall status, each of its 3 steps' status, and (only while TrainYolo
    is Executing, or right after it succeeds) the live epoch/download
    sub-progress.

    Takes the pipeline name because this was hardcoded to the PRODUCTION
    pipeline: while a custom run executed, the queue showed the last
    production execution's progress instead. A run configured for 1 epoch
    displayed "Epoch 0 / 50" -- numbers from a different run, on a
    different pipeline, that had already finished.
    """
    pipeline_name = pipeline_name or PIPELINE_NAME
    try:
        executions = sm.list_pipeline_executions(
            PipelineName=pipeline_name, SortBy="CreationTime", SortOrder="Descending", MaxResults=1,
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
        # WHICH pipeline this describes. Carried because the panel shows
        # one of two and they are not interchangeable: a custom run and a
        # production run differ in taxonomy, registry group and epoch
        # count, so "Succeeded" without a name is not an answer.
        "pipeline": pipeline_name,
        "custom": pipeline_name == OPEN_PIPELINE_NAME,
    }


def get_cpu_resource_usage() -> list[dict]:
    """Every InProgress PROCESSING job -- the CPU half of a run.

    A pipeline execution is not one machine. `TrainYolo` holds the single
    GPU; `EvaluateYolo` runs afterwards on an ml.m5.xlarge, and
    `RegisterYoloModel` is an API call. So between the end of training
    and the end of the run there is a stretch -- seven or eight minutes
    of it -- where the queue correctly says a run is going and the GPU
    card correctly says the GPU is free, and the two read as a
    contradiction to anybody who has not internalised the step layout.

    Reporting the processing job directly is the answer: "CPU in use"
    beside "GPU free" describes the machine that is actually working.

    Each dict: {"name", "step", "instance_type", "own_pipeline"}.
    """
    try:
        summaries = sm.list_processing_jobs(
            StatusEquals="InProgress", SortBy="CreationTime",
            SortOrder="Descending", MaxResults=20,
        )["ProcessingJobSummaries"]
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not list processing jobs: {exc}")
        return []
    jobs = []
    for summary in summaries:
        name = summary["ProcessingJobName"]
        instance_type = ""
        try:
            instance_type = sm.describe_processing_job(
                ProcessingJobName=name,
            )["ProcessingResources"]["ClusterConfig"]["InstanceType"]
        except Exception as exc:                               # noqa: BLE001
            # Named but not described: still worth reporting as busy.
            print(f"Could not describe processing job {name}: {exc}")
        # "pipelines-<execution>-<StepName>-<suffix>" is how SageMaker
        # names a step's job, so the step is readable straight off it.
        step = ""
        parts = name.split("-")
        if name.startswith("pipelines-") and len(parts) >= 3:
            step = parts[2]
        jobs.append({
            "name": name,
            "step": step,
            "instance_type": instance_type,
            "own_pipeline": name.startswith("pipelines-"),
        })
    return jobs


def progress_pipeline_name() -> str:
    """Which pipeline the training-progress panel should describe.

    NOT "the custom one while it is Executing". That test is true only
    DURING a custom run, so the instant one ended the panel fell back to
    the production pipeline and reported its last execution instead. On
    2026-10-06 that meant a custom run which had just FAILED at
    EvaluateYolo was displayed as "Succeeded -- Epoch 0 / 50": the status
    of a different pipeline, on a different taxonomy, that had finished
    four days earlier. A panel that certifies a failure as a success is
    worse than one that shows nothing at all, because nobody goes
    looking.

    So the rule is simply WHICHEVER RAN LAST, finished or not. A
    production run after a custom one takes the panel back, which is
    right -- that is the newer answer to "what happened".
    """
    newest_name, newest_at = PIPELINE_NAME, None
    for name in (PIPELINE_NAME, OPEN_PIPELINE_NAME):
        try:
            executions = sm.list_pipeline_executions(
                PipelineName=name, SortBy="CreationTime", SortOrder="Descending",
                MaxResults=1,
            )["PipelineExecutionSummaries"]
        except Exception as exc:
            print(f"Could not list executions of {name}: {exc}")
            continue
        if not executions:
            continue
        started = executions[0].get("StartTime")
        if started is None:
            continue
        if newest_at is None or started > newest_at:
            newest_name, newest_at = name, started
    return newest_name


#: Managed-spot training and on-demand training draw on SEPARATE quotas,
#: so a spot job cannot exhaust the one our runs need. Verified, not
#: assumed:
#:
#:     aws service-quotas list-service-quotas --service-code sagemaker #:       --region ap-southeast-1 #:       --query "Quotas[?contains(QuotaName,'g4dn.xlarge')].[QuotaName,Value]"
#:
#:     ml.g4dn.xlarge for training job usage        1.0
#:     ml.g4dn.xlarge for spot training job usage   1.0
#:
#: And our own TrainYolo step is on-demand -- `EnableManagedSpotTraining`
#: is not set on it, which `describe-pipeline` shows.
SPOT_USES_A_DIFFERENT_QUOTA = True


def is_own_pipeline_job(name: str) -> bool:
    """Was this job created by one of THIS app's pipeline steps?

    SageMaker names a job a pipeline created
    `pipelines-<execution-id>-<StepName>-<suffix>`. Both of this app's
    pipelines call their training step TrainYolo, so that pair is the
    signature -- and it is the same rule the single-GPU guard uses to
    decide "busy with your own run" versus "busy with something else".

    Defined once because two copies of it would eventually disagree, and
    the disagreement would be a job the guard treats as foreign and the
    cost panel labels as ours, or the reverse.
    """
    return name.startswith("pipelines-") and "-TrainYolo-" in name


def is_own_processing_job(name: str) -> bool:
    """The same question for a processing job.

    The evaluation step runs under the same execution and is named the
    same way, but its step is not TrainYolo -- so the training rule
    would call this app's own evaluation foreign.
    """
    return name.startswith("pipelines-")


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
            details = sm.describe_training_job(TrainingJobName=name)
            instance_type = details["ResourceConfig"]["InstanceType"]
        except Exception as exc:
            print(f"Could not describe training job {name}: {exc}")
            continue
        if instance_type == GPU_INSTANCE_TYPE:
            # From the DESCRIBE, not the summary. `TrainingJobSummary`
            # carries CreationTime, TrainingEndTime, LastModifiedTime and
            # TrainingJobStatus -- and no TrainingStartTime at all, so
            # this read None for every job, always. Nothing noticed while
            # the field was only used for display; the moment the status
            # bar timed the GPU from it, a run that had been training for
            # seventeen minutes reported "waiting for capacity".
            #
            # The describe is already being made on the line above for
            # the instance type, so this costs nothing.
            start_time = details.get("TrainingStartTime")
            jobs.append({
                "name": name,
                "start_time": start_time.isoformat() if start_time else None,
                "own_pipeline": is_own_pipeline_job(name),
                # SPOT OR ON-DEMAND, because they are different quotas and
                # only one of them is ours to run out of. See
                # SPOT_USES_A_DIFFERENT_QUOTA above.
                "spot": bool(details.get("EnableManagedSpotTraining")),
                # Null until an instance is actually allocated. A job
                # sitting in a capacity retry loop has none, is billed
                # nothing, and is holding no hardware -- worth showing,
                # because "InProgress" on its own reads as "running".
                "started": bool(start_time),
            })
    return jobs


def stop_foreign_gpu_jobs() -> dict:
    """Stop every GPU training job in the account this app did not start.

    The single-GPU guard refuses to start a run while ANY on-demand
    g4dn job holds the slot, including one started elsewhere -- which is
    correct, and left nothing that could clear it. The slot could be
    held for hours by something this app could see and not touch.

    `own_pipeline` jobs are excluded on purpose: those belong to a
    pipeline execution, and stopping the JOB under a running execution
    leaves the execution to fail its step rather than stop cleanly.
    `stop_pipeline_execution` is the right call for those, and
    `/train/stop` makes it first.

    Spot jobs are included. They do not BLOCK a run -- different quota --
    but "stop everything on the GPU" meaning "everything except that
    one" would be a surprise, and the operator can see it in the list.

    Returns what happened, per job. Never raises: one job refusing must
    not hide the others that stopped.
    """
    out: dict = {"stopped": [], "problems": []}
    for job in get_gpu_resource_usage():
        if job.get("own_pipeline"):
            continue
        name = job["name"]
        try:
            sm.stop_training_job(TrainingJobName=name)
            out["stopped"].append(name)
        except Exception as exc:                               # noqa: BLE001
            text = str(exc)
            if "AccessDenied" in text or "not authorized" in text:
                # Named for what it is. This grant lives in
                # terraform/app/audit.tf and is NOT applied by deploy.sh,
                # so the code can ship before the permission exists.
                out["problems"].append(
                    f"{name}: this host is not allowed to stop training jobs. "
                    "It needs sagemaker:StopTrainingJob, which is a policy "
                    "change (terraform/app/audit.tf), not a code change."
                )
            else:
                out["problems"].append(f"{name}: {text}")
    return out


def custom_run_in_progress() -> str | None:
    """The id of an Executing custom-pipeline run, or None.

    Needed because neither of the other two checks sees one.
    get_training_progress lists executions of the PRODUCTION pipeline only,
    and get_gpu_resource_usage marks the job `own_pipeline` -- the custom
    pipeline names its training step `TrainYolo` too, so the job is called
    `pipelines-<id>-TrainYolo-...` exactly like a production one -- after
    which training_already_running_reason filters it out.

    Without this, the single-GPU guard silently passed while a custom run
    held the slot, and the second run died with SAGEMAKER_RESOURCE_LIMIT.
    """
    try:
        executions = sm.list_pipeline_executions(
            PipelineName=OPEN_PIPELINE_NAME, SortBy="CreationTime", SortOrder="Descending", MaxResults=10,
        )["PipelineExecutionSummaries"]
    except Exception as exc:
        print(f"Could not list custom pipeline executions: {exc}")
        return None
    # Any Executing run, not just the newest: start A, then start and stop
    # B, and a newest-only check reports nothing while A still holds the GPU.
    for execution in executions:
        if execution.get("PipelineExecutionStatus") == "Executing":
            # Visible now, so the submission marker has done its job and
            # would otherwise keep reporting a second, phantom reason.
            global _custom_submission_at
            with _custom_submission_lock:
                _custom_submission_at = None
            return execution["PipelineExecutionArn"].rsplit("/", 1)[-1]
    return None


def custom_run_execution_arn() -> str | None:
    """The full ARN of an Executing open-pipeline run, for stopping it.

    Separate from custom_run_in_progress, which returns the short id for a
    human-readable message. Without this there was no way to stop a custom
    run at all: the stop endpoint only knew the production pipeline, while
    the single-GPU guard correctly refused every new run until the custom
    one finished or hit its 24h MaxRuntimeInSeconds.
    """
    try:
        executions = sm.list_pipeline_executions(
            PipelineName=OPEN_PIPELINE_NAME, SortBy="CreationTime", SortOrder="Descending", MaxResults=10,
        )["PipelineExecutionSummaries"]
    except Exception as exc:
        print(f"Could not list custom pipeline executions: {exc}")
        return None
    for execution in executions:
        if execution.get("PipelineExecutionStatus") == "Executing":
            return execution["PipelineExecutionArn"]
    return None


#: How long a submission blocks the next one, in seconds.
#:
#: Covers the gap between CodeBuild accepting a build and SageMaker
#: reporting the pipeline as Executing. The in-memory version of this used
#: 120s for the custom path; CodeBuild has to build before the pipeline
#: starts, so the window here is wider.
SUBMISSION_GUARD_SECONDS = 900


def mark_training_submitted(reference: str) -> None:
    """Record that a run was just submitted, where OTHER processes can see it.

    The guards this backs up (`_pending_submission`, `_custom_submission_at`)
    are module globals, which made them invisible to anything but the
    process that set them. That was survivable while one container was the
    only thing starting runs. It stopped being survivable twice over: the
    dev and stable containers share these tables and both run the poll
    loop, and the queue-runner Lambda is a fresh process on every tick.

    The failure it prevents is specific. CodeBuild accepts the build and
    returns; SageMaker does not report Executing for minutes. In that
    window a second process sees an idle GPU, pops the next entry and
    submits it -- and the account has exactly one training instance, so
    that run fails outright with SAGEMAKER_RESOURCE_LIMIT and no retry.

    Best effort: a run that is already submitted must not be undone
    because a marker could not be written.
    """
    try:
        training_audit_table().put_item(Item={
            "id": TRAINING_SUBMISSION_MARKER_ID,
            "record_type": "training_submission",
            "reference": str(reference or ""),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": int(time.time()) + SUBMISSION_GUARD_SECONDS,
        })
    except Exception as exc:                                       # noqa: BLE001
        print(f"Could not record the training submission marker: {exc}")


def clear_training_submitted() -> None:
    """Drop the marker once the run is visibly executing, or visibly dead.

    Without this the guard would hold for its full window even after the
    pipeline is plainly running -- which is harmless for correctness and
    wrong for the operator, who would be told to wait for something that
    has already happened.
    """
    try:
        training_audit_table().delete_item(Key={"id": TRAINING_SUBMISSION_MARKER_ID})
    except Exception as exc:                                       # noqa: BLE001
        print(f"Could not clear the training submission marker: {exc}")


def recent_training_submission() -> dict | None:
    """The marker, if one is live. None once it has expired."""
    try:
        item = training_audit_table().get_item(
            Key={"id": TRAINING_SUBMISSION_MARKER_ID}).get("Item")
    except Exception as exc:                                       # noqa: BLE001
        # Fail CLOSED is wrong here and fail open is wrong too. An
        # unreadable table means we cannot tell -- and the caller's other
        # checks (SageMaker executions, the in-memory markers) still
        # apply, so reporting "no marker" leaves those in force rather
        # than inventing a block nobody can clear.
        print(f"Could not read the training submission marker: {exc}")
        return None
    if not item:
        return None
    try:
        age = (datetime.now(timezone.utc)
               - datetime.fromisoformat(str(item["submitted_at"]))).total_seconds()
    except (KeyError, ValueError):
        return None
    if age > SUBMISSION_GUARD_SECONDS:
        return None
    return {"reference": str(item.get("reference") or ""), "age": int(age)}


def stopping_training_jobs() -> list:
    """Training jobs that have been told to stop but have not let go yet.

    A stopped job sits in `Stopping` for seconds to a minute while it
    still holds its instance. `get_gpu_resource_usage` lists InProgress
    only, so a run queued straight after Stop Training saw a free GPU,
    called CreateTrainingJob, and failed at once on the 1-instance quota
    -- twice on 2026-10-20, 8 seconds before the old job released it.
    Unknown is not busy: a listing that fails returns [] rather than
    holding the queue forever."""
    try:
        summaries = sm.list_training_jobs(StatusEquals="Stopping", MaxResults=10)["TrainingJobSummaries"]
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not list stopping training jobs: {exc}")
        return []
    return [s["TrainingJobName"] for s in summaries]


def training_already_running_reason(progress: dict | None) -> str | None:
    """None if it's safe to start a new run, otherwise a human-readable
    reason not to. The account has exactly 1 ml.g4dn.xlarge training
    instance available -- starting a second pipeline execution while
    TrainYolo is still Executing on another one doesn't queue, it fails
    outright with SAGEMAKER_RESOURCE_LIMIT and no retry."""
    if progress and progress["status"] == "Executing":
        run_id = progress["execution_arn"].rsplit("/", 1)[-1]
        return f"A training run ({run_id}) is already in progress. Wait for it to finish before starting another."
    custom_run = custom_run_in_progress()
    if custom_run:
        return (
            f"A custom-class training run ({custom_run}) is already in progress. "
            "Wait for it to finish before starting another."
        )
    # The custom equivalent of _pending_submission, which cannot be reused:
    # that one identifies a CodeBuild build and is cleared by polling its
    # status, and a custom run has no build. Without this there is a window
    # between start_pipeline_execution returning and the execution being
    # listed as Executing, during which this function reports the GPU free
    # and the queue pops a second entry onto the one slot.
    with _custom_submission_lock:
        submitted_at = _custom_submission_at
    if submitted_at and (datetime.now(timezone.utc) - submitted_at).total_seconds() < 120:
        return (
            "A custom-class training run was just submitted and has not been "
            "reported as running yet. Wait a moment before starting another."
        )
    # The DURABLE marker first, because it is the only one a different
    # process can see -- the queue-runner Lambda, or the other container.
    # The two in-memory checks below stay as well: they are free, and they
    # catch the case where the marker could not be written.
    submitted = recent_training_submission()
    if submitted:
        return (f"A training run was submitted {submitted['age']}s ago and has not "
                "been reported as running yet. Wait for it to start before "
                "starting another.")
    with _pending_submission_lock:
        if _pending_submission:
            return "A training run is still being submitted. Wait for it to start before starting another."
    # ON-DEMAND jobs only. A managed-spot job draws on a DIFFERENT quota
    # and cannot exhaust ours, so blocking on one refuses a run that
    # would have started perfectly well -- which is what happened: a spot
    # smoke test stuck in a capacity retry loop, holding no instance and
    # billing nothing, made the card read "GPU in use -- another job" and
    # the queue refuse to pop for the whole of its 45-minute wait.
    stopping = stopping_training_jobs()
    if stopping:
        return (f"A training job ({stopping[0]}) is still stopping and holds the GPU. "
                "The next run starts as soon as it lets go.")
    gpu_jobs = [j for j in get_gpu_resource_usage()
                if not j["own_pipeline"] and not j.get("spot")]
    if gpu_jobs:
        return (
            f"The account's only {GPU_INSTANCE_TYPE} GPU slot is in use by a job outside this app "
            f"({gpu_jobs[0]['name']}) -- starting a new run now would fail. Wait for it to finish."
        )
    return None


PRODUCTION_CLASSES = {0: "person", 1: "helmet"}

DETECTOR_RECORD_TYPE = "detector"
# Detectors share the audit table, which is keyed on a bare `id`. Namespacing
# the key means a slug can never land on a run's id, whatever someone types.
DETECTOR_KEY_PREFIX = "detector#"

# The person/helmet detector, built in rather than stored. It owns the
# legacy group names `yolo11{size}-detector`, which are the four the
# EventBridge approval rule watches and which every existing version is
# already registered under -- so it cannot be renamed or deleted without
# orphaning them. Every other detector is a row in the table.
BUILTIN_DETECTOR = {
    "slug": "ppe",
    "label": "PPE (person/helmet)",
    "classes": {int(k): v for k, v in PRODUCTION_CLASSES.items()},
    "group_pattern": "yolo11{size}-detector",
    # YOLO11 ONLY, and not an oversight. These four names are the
    # exact-match list the approval rule watches; a `yolo26s-detector`
    # would be inert there, so a YOLO26 model registered under the
    # built-in would look production but never deploy. User-defined
    # detectors get every generation -- see `detector_generations`.
    "generations": ("11",),
    "builtin": True,
}


def detector_group_name(detector: dict, architecture: str) -> str:
    """The registry group a detector's models of one size register into.

    `yolo11s-vehicle-detector`, and so on. Three constraints shape this:

    - It must never equal one of `yolo11{s,m,l,x}-detector`. Those four are
      an exact-match list on the rule that fires dump_approved_weights,
      which swaps the live inference component and overwrites production's
      weights. Any other name is inert there.
    - Starting `yolo` and ending `-detector` keeps it inside the app's
      existing `model-package/yolo*-detector/*` IAM grant, so adding a
      detector needs no Terraform.
    - SageMaker allows alphanumerics and hyphens only, 63 characters max.
    """
    generation, size = split_architecture(architecture)
    return detector_group(detector, size, generation)


def split_architecture(architecture: str) -> tuple:
    """"yolo26m" -> ("26", "m").

    Tried longest-prefix-first rather than assuming one generation. The
    previous version stripped `yolo{YOLO_GENERATIONS[0]}` only, so once a
    second generation existed "yolo26m" kept its prefix and became a SIZE
    called "yolo26m" -- which would have produced the group
    `yoloyolo26m-...-detector` instead of raising.
    """
    arch = (architecture or "").strip().lower()
    for generation in sorted(GENERATION_SIZES, key=len, reverse=True):
        size = arch.removeprefix(f"yolo{generation}")
        if size != arch and size in GENERATION_SIZES[generation]:
            return generation, size
    raise ValueError(f"'{architecture}' is not one of {', '.join(KNOWN_ARCHITECTURES)}.")


def ensure_model_package_group(group_name: str, description: str = "") -> bool:
    """Creates the registry group if it is missing. True if it now exists.

    Registering does NOT create the group implicitly -- CreateModelPackage
    fails outright when it is absent. That failure lands at the very last
    step of the pipeline, *after* the GPU hours are spent, which is the
    worst possible place to discover it. Verified on yolo11s-open-detector:
    the group was created 4h20m before its first version, so nothing about
    registering had brought it into being.

    Never raises. Until the training-control policy grants the create, this
    is expected to fail and an admin pre-creates the groups instead; a run
    into a group that does exist must not be blocked by that.
    """
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
        return True
    except Exception:
        pass
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=(
                description or f"Versions registered into {group_name}."
            )[:1024],
        )
        print(f"Created model package group {group_name}.")
        return True
    except Exception as exc:
        print(f"Could not ensure model package group {group_name}: {exc}")
        return False


# Characters a CustomerMetadataProperties value may contain beyond letters,
# digits and spaces. The service pattern is
# ([\p{L}\p{Z}\p{N}_.:\/=+\-@]*){1,256} -- note what is NOT in it: a COMMA.
_METADATA_PUNCTUATION = frozenset("_.:/=+-@")


def metadata_value(text, fallback: str = "unspecified") -> str:
    """Coerces a value into what CustomerMetadataProperties will accept.

    The map's values must be 1-256 characters drawn from a restricted set.
    A comma is not in it, so joining class names with ", " failed the
    whole CreateModelPackage call -- and failed it at the very end, after
    the artifact had already been uploaded. Sanitising every value here
    rather than hand-checking each one means the next field added cannot
    reintroduce the same failure.
    """
    # Collapse any whitespace first: \p{Z} covers spaces, not tabs or
    # newlines, and a description pasted from elsewhere may hold either.
    collapsed = " ".join(str(text if text is not None else "").split())
    cleaned = "".join(
        ch for ch in collapsed
        if ch.isalnum() or ch == " " or ch in _METADATA_PUNCTUATION
    ).strip()
    return (cleaned or fallback)[:256]


def restart_group_numbering_if_empty(group_name: str) -> bool:
    """Puts an empty registry back to v1 by recreating its group.

    SageMaker version numbers are monotonic per group and are never
    reused: register four versions, delete all four, and the next one is
    v5 in a registry that looks empty. The only way back to v1 is to
    delete the GROUP and let it be recreated -- which is safe precisely
    when it holds nothing.

    Measured against the service rather than assumed: emptying a group and
    registering again gave v3; deleting the group, recreating it and
    registering gave v1.

    Never raises. A higher version number is cosmetic; a failed import is
    not, so every failure here leaves the numbering alone and carries on.
    """
    try:
        if model_registry_summary(group_name)["total"]:
            return False
    except Exception:
        return False
    try:
        sm.delete_model_package_group(ModelPackageGroupName=group_name)
    except Exception as exc:
        # Usually "does not exist", which needs nothing doing -- the caller
        # creates it next and it starts at v1 anyway.
        print(f"Leaving {group_name} numbering as it is: {exc}")
        return False
    # The delete is ASYNCHRONOUS. Recreating immediately fails while it is
    # still in flight, which left the group deleted and the registration
    # with nowhere to go -- the group simply vanished and the import
    # silently produced nothing.
    if not _wait_for_group_gone(group_name):
        print(f"{group_name} is still deleting; leaving it to settle.")
        return False
    print(f"Recreating empty group {group_name} so numbering restarts at v1.")
    return True


def _wait_for_group_gone(group_name: str, timeout_seconds: float = 30.0) -> bool:
    """Blocks until DescribeModelPackageGroup stops finding the group.

    DeleteModelPackageGroup returns before the group is actually gone, so
    a create issued straight after it races the delete.
    """
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            sm.describe_model_package_group(ModelPackageGroupName=group_name)
        except Exception:
            return True
        time.sleep(1.0)
    return False


def _detector_from_item(item: dict) -> dict:
    classes = item.get("classes") or {}
    slug = item.get("slug") or str(item["id"]).removeprefix(DETECTOR_KEY_PREFIX)
    return {
        "slug": slug,
        "label": item.get("label") or slug,
        "classes": {int(k): str(v) for k, v in classes.items()},
        "group_pattern": _generic_pattern(
            item.get("group_pattern") or f"yolo11{{size}}-{slug}-detector", slug),
        "generations": tuple(GENERATION_SIZES),
        "builtin": False,
        # Identifiers this detector used to have. A rename cannot move
        # the versions already registered -- SageMaker has no rename for
        # a model package group, only create and delete -- so the old
        # groups stay where they are and this is how they are still
        # found. Registration never looks at it; listing does.
        "previous_slugs": [str(x) for x in (item.get("previous_slugs") or [])],
    }


def _generic_pattern(pattern: str, slug: str) -> str:
    """A stored `yolo11{size}-...` pattern, widened to carry the generation.

    Records written before YOLO26 existed name the generation literally.
    Widening them is SAFE rather than a migration, because the two produce
    the same string for generation 11:

        "yolo11{size}-x-detector".format(size="s")              -> yolo11s-x-detector
        "yolo{generation}{size}-x-detector".format(generation="11", size="s")
                                                                -> yolo11s-x-detector

    So every version already registered keeps the group it is in, and the
    detector simply gains the generation-26 groups it did not have.
    """
    if "{generation}" in pattern:
        return pattern
    if pattern.startswith(f"yolo{DEFAULT_GENERATION}{{size}}-"):
        return "yolo{generation}{size}-" + pattern.split("{size}-", 1)[1]
    # Something hand-written we do not recognise: leave it exactly alone
    # rather than guess, and let it stay a YOLO11-only detector.
    return pattern


def detector_generations(detector: dict) -> tuple:
    """Which YOLO generations this detector has registries for."""
    stated = detector.get("generations")
    if stated:
        return tuple(stated)
    if "{generation}" not in detector.get("group_pattern", ""):
        return (DEFAULT_GENERATION,)
    return tuple(GENERATION_SIZES)


def detector_group(detector: dict, size: str, generation: str = DEFAULT_GENERATION) -> str:
    """One registry group name for this detector at this generation+size."""
    return detector["group_pattern"].format(generation=generation, size=size)


def detector_groups(detector: dict) -> list:
    """Every (generation, size, group name) this detector registers into.

    One place, because the group list is built in five -- the registry
    summary, the metrics scan, the base-model picker, creation and
    deletion. A detector whose groups are enumerated differently in two
    of them is a model that exists on one page and not another.
    """
    out = []
    for generation in detector_generations(detector):
        for size in GENERATION_SIZES.get(generation, REGISTRY_SIZES):
            out.append((generation, size, detector_group(detector, size, generation)))
    return out


def detector_history_groups(detector: dict) -> list:
    """Every group this detector's versions could be in, old names too.

    `detector_groups` is what a NEW run registers into and must stay the
    current name only. This is for LISTING: after a rename the versions
    trained under the old identifier are still in the old groups, and a
    detector that stopped showing its own history would look like one
    that had never been trained.
    """
    seen = {group for _, _, group in detector_groups(detector)}
    out = list(detector_groups(detector))
    for old_slug in detector.get("previous_slugs") or []:
        was = dict(detector)
        was["group_pattern"] = "yolo{generation}{size}-" + f"{old_slug}-detector"
        for generation, size, group in detector_groups(was):
            if group not in seen:
                seen.add(group)
                out.append((generation, size, group))
    return out


# Reading detectors means scanning the audit table, which grows with every
# run, and three separate paths need the list to build one page (the
# registry summary, the metrics scan and the base-model picker). Cached for
# a few seconds so that is one scan rather than three, and invalidated on
# write so an edit is never served stale.
_detector_cache: tuple[float, list] | None = None
_detector_cache_lock = threading.Lock()
DETECTOR_CACHE_SECONDS = 30


def list_detectors(*, fresh: bool = False) -> list:
    """Every detector: the built-in PPE one first, then the stored ones.

    The built-in is merged in rather than seeded into the table so it
    cannot be edited into something that no longer matches the versions
    already registered under its legacy group names.
    """
    global _detector_cache
    if not fresh:
        cached = _detector_cache
        if cached and time.monotonic() - cached[0] < DETECTOR_CACHE_SECONDS:
            return [dict(d) for d in cached[1]]
    stored = []
    try:
        # Paginated, and filtered service-side. A bare scan() returns only
        # the first 1MB, so once the audit table passed that a detector
        # would simply stop appearing -- silently, and only in production.
        # ConsistentRead, because this list is read immediately after it is
        # written. A plain Scan is eventually consistent, so deleting a
        # class list and refetching could still return it -- and the stale
        # answer would then be cached for 30 seconds, so the row appeared
        # to come back from the dead.
        kwargs: dict = {
            "FilterExpression": Attr("record_type").eq(DETECTOR_RECORD_TYPE),
            "ConsistentRead": True,
        }
        while True:
            response = training_audit_table().scan(**kwargs)
            # Re-checked client-side as well. The table holds audit rows and
            # queue entries too, and reading one of those as a detector
            # would invent a registry group out of a run id.
            stored.extend(
                _detector_from_item(item)
                for item in response.get("Items", [])
                if item.get("record_type") == DETECTOR_RECORD_TYPE
            )
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
    except Exception as exc:
        # A failure here must not blank the registries: fall back to the
        # last good list if there is one, and to the built-in otherwise.
        print(f"Could not list detectors: {exc}")
        if _detector_cache:
            return [dict(d) for d in _detector_cache[1]]
        return [dict(BUILTIN_DETECTOR)]
    stored.sort(key=lambda d: d["label"].lower())
    detectors = [dict(BUILTIN_DETECTOR)] + stored
    with _detector_cache_lock:
        _detector_cache = (time.monotonic(), detectors)
    return [dict(d) for d in detectors]


def _invalidate_detector_cache() -> None:
    global _detector_cache
    with _detector_cache_lock:
        _detector_cache = None


def get_detector(slug: str) -> dict:
    slug = (slug or "").strip().lower()
    for detector in list_detectors():
        if detector["slug"] == slug:
            return detector
    raise ValueError(f"No detector '{slug}'. Known: {', '.join(d['slug'] for d in list_detectors())}.")


def upsert_detector(slug: str, label: str, classes: dict) -> dict:
    """Creates or updates a detector.

    The class list is the assertion that stops a vehicle model being read
    as person/helmet, so it is validated here rather than trusted: indices
    must be a contiguous 0..n-1 range, which is what Ultralytics requires
    of data.yaml and what every consumer assumes when it reads index 0.
    """
    slug = (slug or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,30})", slug or ""):
        raise ValueError(
            "Slug must be lowercase letters, digits and hyphens, starting with a letter "
            "or digit, at most 31 characters -- it becomes part of the registry group name."
        )
    if slug == BUILTIN_DETECTOR["slug"]:
        raise ValueError(
            f"'{slug}' already exists -- unable to comply. It is the built-in "
            "detector and owns the production registries."
        )
    parsed = {int(k): str(v).strip() for k, v in (classes or {}).items()}
    if not parsed:
        raise ValueError("A detector needs at least one class.")
    if sorted(parsed) != list(range(len(parsed))):
        raise ValueError(
            f"Class indices must run 0..{len(parsed) - 1} with no gaps; got {sorted(parsed)}."
        )
    if any(not name for name in parsed.values()):
        raise ValueError("Every class needs a name.")
    item = {
        "id": f"{DETECTOR_KEY_PREFIX}{slug}",
        "slug": slug,
        "record_type": DETECTOR_RECORD_TYPE,
        "label": (label or slug).strip(),
        "classes": {str(k): v for k, v in parsed.items()},
        # Carries the generation, so one detector spans YOLO11 and YOLO26
        # rather than needing to be registered twice under two names.
        "group_pattern": "yolo{generation}{size}-" + f"{slug}-detector",
    }
    training_audit_table().put_item(Item=item)
    _invalidate_detector_cache()
    detector = _detector_from_item(item)
    # Up front, not at registration time: see ensure_model_package_group.
    # Best-effort -- a detector whose groups could not be created is still
    # worth saving, and the run path ensures them again anyway.
    for generation, size, group in detector_groups(detector):
        ensure_model_package_group(
            group,
            f"{detector['label']} -- yolo{generation}{size}. Classes: "
            f"{', '.join(detector['classes'].values())}.",
        )
    return detector


def rename_detector(old_slug: str, new_slug: str, label: str = "") -> dict:
    """Give a Model Name a new identifier, keeping its history readable.

    WHAT CAN AND CANNOT MOVE, because the difference decides the design:

      * The LABEL is ours and changes freely.
      * The IDENTIFIER derives the registry group name, and a group
        cannot be renamed -- SageMaker exposes create, delete, describe
        and list for model package groups, and no update. That is an AWS
        limit, not ours.

    So a rename points the detector at new groups for everything it
    trains from now on, and records the old identifier so the versions
    already registered stay findable. Nothing is deleted in SageMaker and
    nothing is re-registered: the old groups keep their versions under
    their own name, and `detector_history_groups` reads both.
    """
    old_slug = (old_slug or "").strip().lower()
    new_slug = (new_slug or "").strip().lower()
    if not old_slug:
        raise ValueError("Which Model Name is being renamed?")
    if old_slug == BUILTIN_DETECTOR["slug"]:
        raise ValueError(
            "The built-in detector owns the production registries and cannot "
            "be renamed."
        )
    if not re.fullmatch(r"[a-z0-9]([a-z0-9-]{0,30})", new_slug or ""):
        raise ValueError(
            "The new identifier must be lowercase letters, digits and hyphens, "
            "starting with a letter or digit, at most 31 characters -- it "
            "becomes part of the registry group name."
        )
    if new_slug == BUILTIN_DETECTOR["slug"]:
        raise ValueError(f"'{new_slug}' is the built-in detector's name.")

    current = get_detector(old_slug)
    if not current:
        raise ValueError(f"No Model Name called {old_slug}.")
    if new_slug != old_slug and get_detector(new_slug):
        raise ValueError(
            f"'{new_slug}' is already a Model Name. Pick another identifier, or "
            f"delete that one first."
        )

    previous = [x for x in (current.get("previous_slugs") or []) if x != new_slug]
    if new_slug != old_slug and old_slug not in previous:
        previous.append(old_slug)

    item = {
        "id": f"{DETECTOR_KEY_PREFIX}{new_slug}",
        "slug": new_slug,
        "record_type": DETECTOR_RECORD_TYPE,
        "label": (label or current.get("label") or new_slug).strip(),
        "classes": {str(k): v for k, v in (current.get("classes") or {}).items()},
        "group_pattern": "yolo{generation}{size}-" + f"{new_slug}-detector",
        "previous_slugs": previous,
    }
    training_audit_table().put_item(Item=item)
    if new_slug != old_slug:
        # After the new row is written, never before: a crash between the
        # two leaves BOTH, which is visible and fixable. The other order
        # leaves neither.
        training_audit_table().delete_item(Key={"id": f"{DETECTOR_KEY_PREFIX}{old_slug}"})
    _invalidate_detector_cache()

    detector = _detector_from_item(item)
    for generation, size, group in detector_groups(detector):
        ensure_model_package_group(
            group,
            f"{detector['label']} -- yolo{generation}{size}. Classes: "
            f"{', '.join(detector['classes'].values())}.",
        )
    return detector


def retarget_detector_references(old_slug: str, new_slug: str) -> list:
    """Point everything that names a Model Name at its new identifier.

    A rename that changed only the registry would leave the whole page
    pointing at an identifier that no longer exists: the training form
    would have nothing selected, the unattended retrain would refuse
    with "no Model Name", and the queued entries would submit against a
    detector the server could not find. That is not a rename, it is a
    rename plus five silent breakages.

    WHAT IS REWRITTEN is live configuration -- what the system will do
    next. WHAT IS NOT is history: audit rows and dataset snapshots record
    what actually happened under the name it happened under, and editing
    them would make the record disagree with the registry groups the
    versions are really in.

    Returns a line per thing changed, for the card to show. Each step is
    independent and best effort: a rename must not half-apply because one
    of five keys could not be written.
    """
    from .. import pull_control as _pc

    changed: list = []
    if not old_slug or old_slug == new_slug:
        return changed

    try:
        config = _pc.read_config()
    except Exception as exc:                                   # noqa: BLE001
        return [f"could not read the settings: {exc}"]

    patch: dict = {}

    chosen = dict(config.get("default_detector") or {})
    if str(chosen.get("slug") or "") == old_slug:
        chosen["slug"] = new_slug
        patch["default_detector"] = chosen
        changed.append("the Model Name everything trains as")

    for key, label in (("dataset_assemble", "the Assemble card"),
                       ("import_checkpoint", "the Import Checkpoint card"),
                       ("retrain_schedule", "the unattended retrain")):
        section = dict(config.get(key) or {})
        if str(section.get("detector_slug") or "") == old_slug:
            section["detector_slug"] = new_slug
            patch[key] = section
            changed.append(label)

    if patch:
        try:
            _pc.write_config(patch)
        except Exception as exc:                               # noqa: BLE001
            changed.append(f"could not save the settings: {exc}")

    # Queued runs have not started, so they should train as the renamed
    # Model Name -- an entry submitted against a slug that no longer
    # resolves fails at the preflight, hours after anybody was watching.
    queued = 0
    try:
        with _queue_lock:
            for entry in _training_queue:
                if str(entry.get("detector_slug") or "") == old_slug:
                    entry["detector_slug"] = new_slug
                    queued += 1
                    _persist_queue_entry(entry)
    except Exception as exc:                                   # noqa: BLE001
        changed.append(f"could not update the queue: {exc}")
    if queued:
        changed.append(f"{queued} queued run{'' if queued == 1 else 's'}")

    return changed


def delete_detector(slug: str, force: bool = False) -> dict:
    """Removes a detector definition.

    Refuses by default while its registries hold versions: deleting the
    definition must not silently orphan models, so it reports how many
    exist and stops. `force` removes those versions and their groups too,
    which is what an operator clearing out a detector actually wants --
    but it is destructive, so it is never the default.
    """
    detector = get_detector(slug)
    if detector["builtin"]:
        raise ValueError("The built-in PPE detector cannot be deleted.")
    # Every group it owns, under every name it has had. Deleting only
    # the current ones would leave the old groups behind with their
    # versions in them and nothing pointing at them.
    groups = [group for _gen, _size, group in detector_history_groups(detector)]

    existing = []
    for group in groups:
        summary = model_registry_summary(group)
        if summary["total"]:
            existing.append((group, summary))
    if existing and not force:
        raise ValueError(
            "Delete or move its registered versions first: "
            + ", ".join(f"{g} ({s['total']})" for g, s in existing)
        )

    removed = 0
    if force:
        for group, summary in existing:
            for package in summary["packages"]:
                # Raw SageMaker summaries, so the key is ModelPackageArn.
                arn = package.get("ModelPackageArn")
                if not arn:
                    continue
                try:
                    delete_model_package_and_detach_audit(arn)
                    removed += 1
                except Exception as exc:
                    # Reported rather than swallowed: a half-deleted
                    # detector whose row is gone but whose versions remain
                    # is worse than a refusal.
                    raise ValueError(f"Could not delete {arn}: {exc}")

    # The groups go too, so the names are free and nothing is left behind
    # holding a version counter. Empty by this point either way.
    for group in groups:
        try:
            sm.delete_model_package_group(ModelPackageGroupName=group)
        except Exception:
            # Usually "does not exist" -- a group is only created when it
            # is first needed, so most of a detector's four never are.
            pass

    training_audit_table().delete_item(Key={"id": f"{DETECTOR_KEY_PREFIX}{detector['slug']}"})
    _invalidate_detector_cache()
    return {"slug": detector["slug"], "deleted": True, "versions_removed": removed}


def validate_dataset_folder(folder: str, *, used_for_evaluation: bool, custom_classes: bool = False, detector_slug: str = "") -> None:
    """Checks a chosen folder carries what the pipeline will look for.

    Both pipelines now take an ordinary Ultralytics dataset -- data.yaml
    plus images/ and labels/ -- and no longer require a dataset-card.json.
    The card is still read when present, but a folder without one is a
    perfectly good YOLO dataset and is treated as one.

    What replaced the card as the evaluation gate is the taxonomy, read
    from data.yaml directly. The production pipeline's registry group has
    consumers that read index 0 as `person`, so a model trained on some
    other taxonomy is mislabelled at every step downstream without failing
    anywhere. That check is the one that mattered; the card was only ever
    a second copy of the same assertion.

    Checked here because the alternative is finding out ~15 minutes in,
    after a GPU training job has already succeeded, when the evaluation
    step dies and takes the whole run's model with it.
    """
    prefix = folder.strip("/")
    if not _s3_object_exists(f"{prefix}/data.yaml"):
        raise ValueError(
            f"'{folder}' has no data.yaml, so it is not a YOLO dataset folder. "
            "Pick the folder that contains data.yaml, images/ and labels/."
        )
    if detector_slug:
        # A detector declares its taxonomy, so the dataset must match THAT,
        # not person/helmet. This is what stops a vehicle dataset being
        # trained into the helmet detector's registry group and read with
        # the wrong labels for the rest of its life.
        expected = get_detector(detector_slug)["classes"]
        try:
            names = dataset_classes(folder)
        except Exception as exc:
            raise ValueError(f"Could not read '{folder}/data.yaml': {exc}")
        # A SUPERSET is fine and is the normal case: a site dataset is
        # labelled once with everything on it, and the run derives a copy
        # holding only this detector's classes. What cannot be worked
        # around is a class with no labels anywhere in the dataset.
        absent = [name for name in expected.values() if name not in names.values()]
        if absent:
            # Same correction as in `derive_dataset_for_detector`: this
            # compares the detector's class list against data.yaml, and
            # reads no label file at all.
            raise ValueError(
                f"'{folder}' does not declare {', '.join(absent)} in its "
                f"data.yaml, so a '{detector_slug}' model cannot be trained "
                f"from it. The yaml declares "
                f"{', '.join(names.values()) or 'nothing'}. "
                + _YAML_IS_THE_ONLY_SOURCE_OF_NAMES
            )
        return
    if custom_classes:
        # Any taxonomy is legitimate here -- that is what this path is for.
        return
    try:
        names = dataset_classes(folder)
    except Exception as exc:
        raise ValueError(f"Could not read '{folder}/data.yaml': {exc}")
    if not names:
        raise ValueError(f"'{folder}/data.yaml' declares no class names.")
    if used_for_evaluation and names != PRODUCTION_CLASSES:
        raise ValueError(
            f"'{folder}' declares {names}, but this pipeline evaluates "
            f"{PRODUCTION_CLASSES} only. Its registry group's consumers read index 0 as "
            "'person', so a model trained on these classes would be mislabelled everywhere "
            "downstream without any step failing. Tick \"Train on this dataset's own "
            "classes\" to run it through the custom pipeline instead, which has no such "
            "consumer."
        )


def validate_training_params(epochs, dataset_folder, img_size=640, batch=16, seed=42, eval_dataset_folder="", custom_classes=False, detector_slug="") -> tuple[int, int, int, int]:
    """Shared by the immediate Start Training path and Add to Queue -- both
    need the exact same checks so a bad entry doesn't sit in the queue for
    potentially hours before failing when it's finally its turn to run."""
    if not dataset_folder:
        raise ValueError("No dataset folder selected -- use the browser to pick one first.")
    # Whichever folder evaluation will read is the one that needs the card:
    # with no separate evaluation folder chosen, that is the training one.
    validate_dataset_folder(dataset_folder, used_for_evaluation=True,
                            custom_classes=custom_classes, detector_slug=detector_slug)
    if eval_dataset_folder:
        validate_dataset_folder(eval_dataset_folder, used_for_evaluation=True,
                                custom_classes=custom_classes, detector_slug=detector_slug)
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


def encode_hyperparams(hyperparams: dict | None) -> str:
    """The extra Ultralytics arguments, base64 of JSON. "" for none.

    Empty values are dropped rather than sent as blanks: a form that
    leaves a field alone means "whatever Ultralytics does by default",
    and sending `lr0: ""` would be a different statement entirely.
    """
    # Typed and checked against the one schema (hyperparams.py) -- every
    # submit path, manual, queued or scheduled, encodes through here, so a
    # legacy all-string value ("0.01", "false", "10") becomes the number
    # or boolean Ultralytics requires, and a bad one raises before a job.
    clean = normalize_hyperparams(hyperparams)
    if not clean:
        return ""
    import base64

    return base64.b64encode(json.dumps(clean, sort_keys=True).encode()).decode()


def start_training_run(generation, size, epochs, dataset_folder, base_model_folder, notes, img_size=640, batch=16, seed=42, eval_dataset_folder="", source="manual", base_model_key="", base_model_scratch=False, base_model_package_arn="", architecture="", hyperparams: dict | None = None) -> str:
    """Validates and submits one training run via CodeBuild -- the single
    place that builds environmentVariablesOverride and calls
    codebuild.start_build, so the immediate "Start Training" action and
    the training queue's poll loop both go through here. Raises on
    validation failure or a CodeBuild error; returns the new build's id on
    success. Caller is responsible for the already-running (1 GPU slot)
    guard -- see training_already_running_reason."""
    # FIRST, before any AWS call: a bad value is refused before a filtered
    # copy, a CodeBuild start or a pipeline execution exists. Queue
    # entries and the scheduled chain arrive here with whatever they
    # stored -- legacy strings included -- and are typed the same way.
    hyperparams = normalize_hyperparams(hyperparams)
    epochs_int, img_size_int, batch_int, seed_int = validate_training_params(epochs, dataset_folder, img_size, batch, seed, eval_dataset_folder)
    env_overrides = [
        {"name": "TRAIN_GENERATION", "value": generation, "type": "PLAINTEXT"},
        {"name": "TRAIN_SIZE", "value": size, "type": "PLAINTEXT"},
        {"name": "TRAIN_EPOCHS", "value": str(epochs_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_IMG_SIZE", "value": str(img_size_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_BATCH", "value": str(batch_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_SEED", "value": str(seed_int), "type": "PLAINTEXT"},
        {"name": "TRAIN_DATA_S3_URI", "value": f"s3://{BUCKET}/{dataset_folder}/", "type": "PLAINTEXT"},
        # Everything else Ultralytics takes, as one base64 blob. Encoded
        # because it travels on to a SageMaker hyperparameter and then to
        # a command line, and a JSON object through those layers is a
        # quoting problem that looks like the parameter being ignored.
        # train.py validates it against the installed Ultralytics and
        # REFUSES an unknown key, so a typo fails the job in seconds
        # rather than training an hour on silent defaults.
        {"name": "TRAIN_HYPERPARAMS", "value": encode_hyperparams(hyperparams),
         "type": "PLAINTEXT"},
    ]
    # Only sent when a separate evaluation set was chosen. Unset, the
    # pipeline evaluates on the training data, which is its existing
    # behaviour -- sending the training URI here would look like a
    # deliberate choice in the build's environment when it isn't one.
    if eval_dataset_folder:
        env_overrides.append(
            {"name": "EVAL_DATA_S3_URI", "value": f"s3://{BUCKET}/{eval_dataset_folder}/", "type": "PLAINTEXT"}
        )
    # All four sources, same dispatcher as the open pipeline. Nothing is
    # sent when none is chosen, which is how "use the registry's latest
    # Approved package" still reaches production_baseline() untouched.
    base_model_s3_uri = ""
    base_model_source = ""
    if base_model_scratch or base_model_key or base_model_folder or base_model_package_arn:
        base_model_s3_uri, base_model_source = resolve_base_model_source(
            scratch=base_model_scratch,
            scratch_architecture=architecture or f"yolo{generation}{size}",
            base_model_key=base_model_key,
            base_model_folder=base_model_folder,
            base_model_package_arn=base_model_package_arn,
        )
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
    # And durably, so the other container and the queue-runner Lambda see
    # it too. In-memory alone meant each process had its own idea of
    # whether the single GPU slot was spoken for.
    mark_training_submitted(build_id)
    # After the build is actually submitted, so a run that never started
    # leaves no audit row claiming it did. Listing the dataset can take a
    # moment on a large folder; the build is already on its way by then.
    write_training_audit(
        build_id=build_id, generation=generation, size=size, epochs=epochs_int,
        img_size=img_size_int, batch=batch_int, seed=seed_int,
        dataset_folder=dataset_folder, eval_dataset_folder=eval_dataset_folder,
        base_model_folder=base_model_folder or base_model_s3_uri, notes=notes, source=source,
        architecture=architecture, base_model_package_arn=base_model_package_arn,
        base_model_source=base_model_source, hyperparams=hyperparams,
    )
    return build_id


CUSTOM_PIPELINE_NAME = os.environ.get("CUSTOM_TRAINING_PIPELINE_NAME", "track2-yolo-custom-training-pipeline")

# Where custom-class runs actually go. The older CUSTOM_PIPELINE_NAME above
# runs the digest-pinned "recovery" training image, whose wrapper asserts
# `model.names == {0: "person", 1: "helmet"}` on the BASE checkpoint -- so
# it can only ever fine-tune from a production checkpoint, which defeats
# the purpose of an open-taxonomy pipeline. This one runs track2-yolo-train
# (the repo's own train.py, no such assertion), pinned by digest, and is
# what the three base-model modes need. The old pipeline is left defined
# but unused.
OPEN_PIPELINE_NAME = os.environ.get("OPEN_TRAINING_PIPELINE_NAME", "track2-yolo-open-training-pipeline")

#: Pipeline parameters that have no DefaultValue on
#: `track2-yolo-open-training-pipeline`, so a blank one cannot simply be
#: left out. Read off the live pipeline, not guessed:
#:     aws sagemaker describe-pipeline #:       --pipeline-name track2-yolo-open-training-pipeline #:       --query PipelineDefinition
#: Every other parameter declares a default, which is what makes omitting
#: a blank one safe.
PIPELINE_PARAMETERS_WITHOUT_DEFAULTS = (
    "TrainingDatasetUri",
    "EvaluationDatasetUri",
    "EvaluationOutputUri",
    "RunMetadataUri",
    "RunId",
    "DatasetVersion",
    "BaseModelUri",
)
CUSTOM_REGISTRY_GROUP = "track2-yolo-custom-detector"


VERSIONED_WEIGHTS_PREFIX = "models"


def publish_versioned_weights(model_package_arn: str) -> str:
    """Lands a registered version's checkpoint at a stable, versioned key.

        s3://<bucket>/models/<group>/v<version>/model.pt

    SageMaker stores the artifact as a tarball under an execution-scoped
    path (training-output/pipelines-<exec>-TrainYolo-<suffix>/output/
    model.tar.gz), which is neither a .pt nor guessable from the version
    number. This unpacks the checkpoint to a key you can point anything at.

    Immutable by construction: a versioned key is written once and never
    replaced, so unlike latest_approved_model_weights/<size>/model.pt there
    is nothing to archive -- an existing object means the work is already
    done. That also makes this safe to call repeatedly, which matters
    because the audit poller retries until it succeeds.

    Only model.pt is extracted, not the whole archive: the tarball also
    carries the bundled inference code, and extractall would write every
    member for a file we already know the name of.
    """
    # arn:...:model-package/<group>/<version>
    group = model_package_arn.split("model-package/")[-1].split("/")[0]
    version = model_package_arn.rsplit("/", 1)[-1]
    key = f"{VERSIONED_WEIGHTS_PREFIX}/{group}/v{version}/model.pt"
    uri = f"s3://{BUCKET}/{key}"
    if _s3_object_exists(key):
        return uri

    described = sm.describe_model_package(ModelPackageName=model_package_arn)
    containers = (described.get("InferenceSpecification") or {}).get("Containers") or []
    model_data_url = containers[0].get("ModelDataUrl") if containers else ""
    if not model_data_url:
        raise ValueError(f"{model_package_arn} has no ModelDataUrl to publish.")

    source_bucket, source_key = model_data_url.removeprefix("s3://").split("/", 1)
    with tempfile.TemporaryDirectory() as work:
        local = os.path.join(work, "artifact")
        s3.download_file(source_bucket, source_key, local)
        if source_key.endswith(".tar.gz"):
            with tarfile.open(local, "r:gz") as archive:
                members = [m for m in archive.getmembers() if m.name in {"model.pt", "./model.pt"}]
                if len(members) != 1 or not members[0].isfile():
                    raise ValueError(f"{model_data_url} does not contain exactly one model.pt")
                extracted = os.path.join(work, "model.pt")
                with archive.extractfile(members[0]) as source, open(extracted, "wb") as output:
                    shutil.copyfileobj(source, output)
            s3.upload_file(extracted, BUCKET, key)
        else:
            s3.upload_file(local, BUCKET, key)
    return uri


def open_registry_group(architecture: str) -> str:
    """The per-size registry group an open-taxonomy run registers into.

    One group per size so version numbers mean something per size:
    "yolo11m v3" is the third medium model, not the third model overall.

    The name deliberately differs from the production group for that size
    (`yolo11m-detector`). Those four names are an exact-match list in the
    EventBridge rule that fires dump_approved_weights, which does not just
    dump weights -- it also swaps the live inference component on the
    shared endpoint and overwrites latest_approved_model_weights/<size>/
    model.pt. Since these runs register as Approved with no human in the
    loop, landing in a watched group would auto-deploy every experiment
    over production. "-open-" keeps them inert there while still matching
    the app's existing `yolo*-detector/*` IAM grant, so no policy change.
    """
    architecture = (architecture or "").strip().lower()
    if architecture not in KNOWN_ARCHITECTURES:
        raise ValueError(
            f"'{architecture}' is not a known architecture. Pick one of: "
            + ", ".join(KNOWN_ARCHITECTURES)
        )
    return f"{architecture}-open-detector"


_PIPELINE_PARAMETER_NAMES: dict = {}


def pipeline_parameter_names(name: str = "", *, expect: str = "") -> set:
    """What the live pipeline actually declares.

    Cached per pipeline name for the life of the process: it is a
    describe call, the answer only changes when somebody runs
    `update-pipeline`, and a container restart is a cheap way to
    re-read it.

    This exists so that sending a parameter the pipeline has never
    heard of fails HERE, with a sentence naming the fix, rather than
    as a ValidationException from `start_pipeline_execution` that
    reads like the run itself was malformed.
    """
    name = name or OPEN_PIPELINE_NAME
    # `expect` RE-READS when the cached answer lacks that name.
    #
    # The cache lives for the life of the process, and the thing it
    # caches changes when somebody runs `update-pipeline`. Without this,
    # adding a parameter would be refused by every warm container until
    # each was restarted -- and the refusal would name the very command
    # that had just been run. One extra describe, only on the path that
    # is about to refuse anyway.
    if expect and expect not in _PIPELINE_PARAMETER_NAMES.get(name, {expect}):
        _PIPELINE_PARAMETER_NAMES.pop(name, None)
    if name not in _PIPELINE_PARAMETER_NAMES:
        try:
            body = json.loads(sm.describe_pipeline(
                PipelineName=name)["PipelineDefinition"])
            _PIPELINE_PARAMETER_NAMES[name] = {
                str(p.get("Name") or "") for p in body.get("Parameters") or []}
        except Exception:                                      # noqa: BLE001
            # Unknown, not empty. Returning an empty set would make every
            # parameter look undeclared and refuse every run.
            return set()
    return _PIPELINE_PARAMETER_NAMES[name]


def start_custom_training_run(
    *, dataset_folder, eval_dataset_folder, epochs, img_size, batch, seed, notes,
    base_model_package_arn="", architecture="", base_model_folder="", base_model_key="",
    base_model_scratch=False, source="custom", detector_slug="", patience=0,
    snapshot=None,
    # A dataset COMPOSED from folders that share no parent. `dataset_folder`
    # is then the `_generated/<hash>` folder holding the yaml and nothing
    # else, so the prefixes to fetch have to be named separately -- one
    # comma-separated string, in the order train.py takes them -- and so
    # does the yaml, which no longer sits at the top of any of them.
    #
    # Blank on every ordinary run, which keeps deriving both from
    # `dataset_folder` exactly as before.
    train_data_uris="", data_yaml_uri="",
    # The training container rewrites its DOWNLOADED copy of each label
    # file to drop boxes whose class the yaml does not name, then trains.
    # The files in S3 are never touched. Off for every run that existed
    # before this did.
    drop_undeclared_classes=False,
    # Val folders the evaluator fetches itself, beside its one prefix.
    eval_extra_uris="",
    # Everything else Ultralytics takes, as one validated blob -- see
    # `encode_hyperparams` and train.py's `_validated_hyperparams`.
    hyperparams: dict | None = None,
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
    # FIRST, before any AWS call: a bad value is refused before a filtered
    # copy, a CodeBuild start or a pipeline execution exists. Queue
    # entries and the scheduled chain arrive here with whatever they
    # stored -- legacy strings included -- and are typed the same way.
    hyperparams = normalize_hyperparams(hyperparams)
    # A dataset labelled with everything on site is the normal case --
    # nobody labels a separate corpus per model type -- so training a
    # single-purpose detector from it means isolating its classes
    # somehow. `class_filter_for` decides which way, and the choice is a
    # fact about the two taxonomies rather than a preference. See there.
    # The lineage rule, checked where every custom run passes: the card,
    # the queue replaying an entry queued before a version existed, and
    # the unattended chain all arrive here.
    if base_model_scratch:
        refusal = scratch_refusal(detector_slug, architecture)
        if refusal:
            raise ValueError(refusal)

    class_filter: dict = {}
    if detector_slug:
        detector_for_run = get_detector(detector_slug)
        class_filter = class_filter_for(dataset_folder, detector_for_run)
        if class_filter.get("derive") and str(train_data_uris or "").strip():
            # `derive_filtered_dataset` copies `images/` and `labels/` from
            # UNDER the dataset folder. A composed dataset has nothing
            # under its folder but the yaml -- the images are at their own
            # keys, in several places -- so the copy would succeed and
            # produce an empty dataset, and the run would train on nothing
            # without any step failing.
            raise ValueError(
                f"This dataset is composed from several folders, and a "
                f"'{detector_slug}' run would need a filtered copy of it. "
                "Copying is what cannot be done here: the images live at "
                "their own keys rather than under one folder. Either pick "
                "a detector whose classes match the composed yaml, or "
                "assemble the sources into one folder first."
            )
        if class_filter.get("derive"):
            # A copy is the only way for this pairing. Before validation,
            # because what validation checks is that the dataset's classes
            # equal the detector's -- and after filtering they do.
            class_filter = {}
            dataset_folder = derive_filtered_dataset(dataset_folder, detector_for_run)
            if eval_dataset_folder:
                # The evaluation set too, or the metrics would be computed
                # against a taxonomy the model was never trained on.
                eval_dataset_folder = derive_filtered_dataset(
                    eval_dataset_folder, detector_for_run)

    epochs_int, img_size_int, batch_int, seed_int = validate_training_params(
        epochs, dataset_folder, img_size, batch, seed, eval_dataset_folder,
        custom_classes=True,
        # Skipped when the filtering happens at load time: the dataset
        # keeps its own six classes on disk and Ultralytics narrows them,
        # so demanding that the yaml already equal the detector's list
        # would reject exactly the case this exists to serve.
        detector_slug="" if class_filter.get("classes") else detector_slug,
    )
    base_model_package_arn = (base_model_package_arn or "").strip()
    # The one dispatcher decides the source and resolves it to a URI. There
    # is deliberately no pre-resolved base_model_uri parameter any more: it
    # silently outranked every other source, and a leftover line normalising
    # it here referenced a name that no longer existed, so every queued
    # custom run died with UnboundLocalError.
    # Required, not optional: it decides which per-size registry group the
    # model is versioned in, so "unspecified" would put every size in one
    # bucket and defeat per-size version numbers.
    # A detector names its own per-size group; without one the run falls
    # back to the shared open groups, which is what every custom run used
    # before detectors existed.
    registry_group = (
        detector_group_name(get_detector(detector_slug), architecture)
        if detector_slug else open_registry_group(architecture)
    )
    # The last line of defence against the pipeline failing at RegisterModel
    # after the GPU hours are already spent. Only yolo11s-open-detector was
    # ever created, so a custom run at m, l or x would have died there too.
    ensure_model_package_group(registry_group)
    base_model_uri, base_model_source = resolve_base_model_source(
        scratch=base_model_scratch,
        scratch_architecture=architecture,
        base_model_key=base_model_key,
        base_model_folder=base_model_folder,
        base_model_package_arn=base_model_package_arn,
    )
    # Full hex, not [:24]. This value is also the ClientRequestToken below,
    # whose documented minimum is 32 characters -- "custom-" + 24 is 31, so
    # botocore rejected EVERY custom run with a 400 before it reached AWS.
    # That is why this pipeline had never had a single execution.
    run_id = "custom-" + uuid.uuid4().hex
    run_root = f"s3://{BUCKET}/custom/runs/{run_id}"
    # SEVERAL prefixes, or the one derived from the folder. train.py splits
    # this on commas and keeps each key's full path when there is more than
    # one; with a single prefix it strips it, exactly as every run so far.
    train_uri = (str(train_data_uris or "").strip()
                 or f"s3://{BUCKET}/{dataset_folder.strip('/')}/")
    # The evaluation step is a ProcessingInput and takes ONE prefix. With
    # no evaluation folder it falls back to the training URI below -- fine
    # for a single prefix, fatal for a list: SageMaker looks for objects
    # under the literal comma-joined string, finds none, and the run that
    # has just trained for an hour fails in two seconds and registers
    # nothing. Refused here instead, before the GPU starts.
    if "," in train_uri and not eval_dataset_folder:
        raise ValueError(
            "This dataset is composed from several prefixes, and the "
            "evaluation step can read only one. The build did not produce "
            "an evaluation folder of its own -- the validation folders must "
            "share a single root (they do whenever the golden set is the "
            "held-out set). Pick an evaluation folder, or tick the golden set."
        )
    eval_uri = (eval_dataset_folder.rstrip("/") + "/" if str(eval_dataset_folder).startswith("s3://")
                else f"s3://{BUCKET}/{eval_dataset_folder.strip('/')}/" if eval_dataset_folder
                else train_uri)
    parameters = {
        "TrainingDatasetUri": train_uri,
        "EvaluationDatasetUri": eval_uri,
        # Whether the numbers will come from a HELD-OUT set or from the
        # training corpus's own val split. The second is the default
        # here, and it is also the split Ultralytics picks `best.pt` on,
        # so the score is optimistic by an unmeasured amount. The
        # evaluation report says which, rather than leaving a reader to
        # assume the generous interpretation.
        "EvaluationDatasetIsSeparate": "true" if eval_dataset_folder else "false",
        "EvaluationOutputUri": f"{run_root}/evaluation",
        "RunMetadataUri": f"{run_root}/request.json",
        "RunId": run_id,
        "DatasetVersion": run_id,
        # Always an S3 artifact by this point, whichever source was picked.
        "BaseModelUri": base_model_uri,
        # Provenance only: recorded on the registered model, never passed
        # to the training job (the pinned image's train.py does not know
        # this argument, and would reject the run outright).
        #
        # Never empty: these land in CustomerMetadataProperties, whose map
        # values have a service minimum of 1 character. An empty string
        # fails CreateModelPackage -- after the training job has already
        # run for hours, at the very last step.
        "BaseModelPackageArn": base_model_package_arn or "none",
        "Architecture": architecture,
        "RegistryGroup": registry_group,
        # How the operator chose the starting weights: scratch / file /
        # folder / package. Recorded on the version so lineage is readable
        # from the registry alone, without pattern-matching a URI prefix.
        "BaseModelSource": base_model_source or "unspecified",
        # Which detector this version belongs to. The group name encodes it
        # too, but recording it explicitly means a version can say what it
        # detects without anyone parsing an ARN.
        "Detector": detector_slug or "open",
        "Epochs": str(epochs_int),
        "ImgSize": str(img_size_int),
        "Batch": str(batch_int),
        "Seed": str(seed_int),
        # Early stopping. 0 means "leave it to Ultralytics", whose own
        # default is 100 -- on a 50-epoch run that never fires, which is
        # why 0 is not "stop immediately" but "no limit set here".
        "Patience": str(max(int(patience or 0), 0)),
        # Which of the dataset's own class indices to train on, and
        # whether to collapse them into a single output. Blank and
        # "false" mean "train on everything the yaml declares", which is
        # what every run did before this existed.
        "Classes": ",".join(str(i) for i in class_filter.get("classes") or []),
        "SingleCls": "true" if class_filter.get("single_cls") else "false",
        # Everything else Ultralytics takes -- optimiser, learning rate,
        # schedule, augmentation -- as one base64 blob rather than a
        # pipeline parameter per knob. Adding another then changes
        # nothing here, nothing in the pipeline, and nothing in the
        # training image: train.py validates whatever arrives against the
        # Ultralytics it actually has installed.
        #
        # Blank means "whatever Ultralytics does by default", which is how
        # every run behaved before this existed.
        "Hyperparams": encode_hyperparams(hyperparams),
        # Where the data.yaml is, when it is not at the top of the one
        # prefix being downloaded. Blank -- and so omitted below -- for
        # every dataset that lives in a single folder.
        "DataYamlUri": str(data_yaml_uri or "").strip(),
        # Sent only when on. Blank is omitted below and the pipeline's own
        # default ("false") applies -- so a pipeline that has not learned
        # this parameter yet still runs every ordinary job.
        "DropUndeclaredClasses": "true" if drop_undeclared_classes else "",
        "EvalExtraUris": str(eval_extra_uris or "").strip(),
    }
    # A BLANK parameter is omitted, not sent blank.
    #
    # `PipelineParameters[].Value` has a service minimum of ONE character,
    # and that is checked before anything else -- so passing "" fails the
    # whole call with
    #   [string "" is too short (length: 0, required minimum: 1)]
    # even for a parameter whose own DefaultValue in the pipeline is "".
    #
    # Omitting it lets that default apply, which is the same value and is
    # exactly what "blank means leave it to Ultralytics" was meant to do.
    # `Classes` and `Hyperparams` are blank on every run that does not
    # filter classes or pin hyperparameters -- which is most of them -- so
    # from the moment those two were added, every custom run died here.
    # Before any GPU time was spent, at least: the queue reported "failed
    # to start, skipped" and moved on.
    sent = {k: v for k, v in parameters.items() if str(v) != ""}
    # A composed dataset cannot run on a pipeline that does not know
    # where to find its yaml. Checked before the call so the message
    # names the one command that fixes it.
    # Each optional parameter that IS being sent must be one the live
    # pipeline declares; `expect` re-reads the cache for exactly that
    # name when it is missing, so an update-pipeline is seen without a
    # restart.
    declared = pipeline_parameter_names()
    for optional in ("DataYamlUri", "DropUndeclaredClasses", "EvalExtraUris"):
        if sent.get(optional):
            declared = pipeline_parameter_names(expect=optional)
    undeclared = [k for k in sent if declared and k not in declared]
    if undeclared:
        raise ValueError(
            "The training pipeline does not declare "
            + ", ".join(sorted(undeclared))
            + ". Update it with `aws sagemaker update-pipeline "
            + f"--pipeline-name {OPEN_PIPELINE_NAME}` before a run that "
            + "needs it."
        )
    # The ones the pipeline declares with NO default cannot be omitted --
    # dropping one would trade this error for a stranger one further in.
    blank_required = [k for k in PIPELINE_PARAMETERS_WITHOUT_DEFAULTS
                      if k not in sent]
    if blank_required:
        raise ValueError(
            "Cannot start the run: "
            + ", ".join(sorted(blank_required))
            + " came through empty, and the pipeline has no default for "
            + ("them." if len(blank_required) > 1 else "it.")
        )
    response = sm.start_pipeline_execution(
        PipelineName=OPEN_PIPELINE_NAME,
        ClientRequestToken=run_id,
        PipelineExecutionDisplayName=run_id,
        PipelineParameters=[{"Name": k, "Value": v} for k, v in sent.items()],
    )
    execution_arn = response["PipelineExecutionArn"]
    # Holds the single GPU slot until the execution is listed as Executing.
    global _custom_submission_at
    with _custom_submission_lock:
        _custom_submission_at = datetime.now(timezone.utc)
    audit = write_training_audit(
        train_data_uris=train_uri if "," in train_uri else "",
        data_yaml_uri=str(data_yaml_uri or "").strip(),
        eval_extra_uris=str(eval_extra_uris or "").strip(),
        eval_uri=eval_uri,
        build_id="", generation="custom", size=architecture or "any", epochs=epochs_int,
        img_size=img_size_int, batch=batch_int, seed=seed_int,
        dataset_folder=dataset_folder, eval_dataset_folder=eval_dataset_folder,
        base_model_folder=base_model_uri, notes=notes, source=source,
        architecture=architecture, base_model_package_arn=base_model_package_arn,
        base_model_source=base_model_source,
        detector_slug=detector_slug, snapshot=snapshot,
        patience=patience, hyperparams=hyperparams,
        training_options={
            "classes": list(class_filter.get("classes") or []),
            "single_cls": bool(class_filter.get("single_cls")),
            "drop_undeclared_classes": bool(drop_undeclared_classes),
        },
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
        clear_training_submitted()
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
        pending = [dict(e) for e in _training_queue]
        running = dict(_queue_running) if _queue_running else None
    # Entries on a dataset version queued before the version's folders
    # were carried on the entry: filled from the version, so the card
    # names where the images come from rather than `_datasets/<id>`.
    for e in pending + ([running] if running else []):
        # ...or before the per-folder counts were (2026-10-26).
        if e.get("dataset_version_id") and (not e.get("dataset_version_train_images")
                                            or not e.get("dataset_source_counts")):
            try:
                from .. import data as _data
                e.update(_data.version_card_fields(e["dataset_version_id"]))
            except Exception as exc:                               # noqa: BLE001
                print(f"Could not read dataset version {e.get('dataset_version_id')}: {exc}")
    with _queue_lock:
        return {
            "pending": pending,
            "running": running,
            "started": _queue_started,
            "completed_count": _queue_completed_count,
            "last_error": _queue_last_error,
            "max_size": MAX_QUEUE_SIZE,
        }


#: The one row that says "a run was just submitted". A fixed id, not
#: one per run: there is one GPU slot, so there is one marker.
TRAINING_SUBMISSION_MARKER_ID = "_training_submission"

QUEUE_RECORD_TYPE = "queue_entry"


def _persist_queue_entry(entry: dict) -> None:
    """Writes a pending entry so it survives a restart.

    The queue was a plain in-memory list, so every deploy, crash and the
    nightly 22:00 instance stop silently emptied it -- entries queued in
    the evening were simply gone by morning, with nothing recorded to say
    they had ever existed.
    """
    try:
        training_audit_table().put_item(Item={**entry, "record_type": QUEUE_RECORD_TYPE})
    except Exception as exc:
        # Never fatal: a queue that runs but is not durable is still better
        # than refusing to queue at all.
        print(f"Could not persist queue entry {entry.get('id')}: {exc}")


def _forget_queue_entry(entry_id: str) -> None:
    try:
        training_audit_table().delete_item(Key={"id": entry_id})
    except Exception as exc:
        print(f"Could not remove persisted queue entry {entry_id}: {exc}")


def restore_queue(*, refresh: bool = False) -> int:
    """Reloads pending entries after a restart. Returns how many came back.

    Ordering is by added_at: the queue is FIFO, and a DynamoDB scan has no
    inherent order, so without this a restored queue would run in an
    arbitrary sequence.

    `refresh=True` replaces what is in memory instead of leaving it alone.
    **The queue-runner Lambda must pass it.** A Lambda container is reused
    between the two-minute ticks, so its copy of the queue outlives the
    invocation that built it -- and the app, ticking the same queue, pops
    entries and deletes their rows without the Lambda ever hearing. The
    warm container then holds an entry that has already run, and once that
    run finishes and the submission marker ages out, it starts it again.
    A second unattended GPU run of a job that is already done.

    The default stays False because it is right for the APP: there the
    in-memory list is the live one, holding entries this process queued
    but has not yet written, and replacing it from a scan would lose them.
    DynamoDB is the truth for a Lambda; memory is the truth for the app.
    """
    global _training_queue
    try:
        items = _scan_queue_rows()
    except Exception as exc:
        print(f"Could not restore the training queue: {exc}")
        return 0
    items.sort(key=lambda item: item.get("added_at") or "")
    global _queue_started
    with _queue_lock:
        # Only when empty, unless a refresh was asked for: a restore must
        # never duplicate entries that the running process already holds.
        if _training_queue and not refresh:
            return 0
        _training_queue[:] = [_undecimalize(item) for item in items]
    # And the SWITCH. Restoring the entries without it is what left a
    # queue full and stopped after every deploy: the run was there, the
    # GPU was free, and nothing moved because the flag the poll loop
    # checks had reset to False.
    #
    # Read on a refresh too, and that matters: the early return above used
    # to skip it, so a warm Lambda never learned that Stop Queue had been
    # pressed and went on starting runs somebody had stopped.
    was_started = _read_queue_started()
    with _queue_lock:
        _queue_started = was_started
    if items:
        print(f"Restored {len(items)} queued training run(s) after restart "
              f"({'running' if was_started else 'stopped -- press Start Queue'}).")
    return len(items)


def _scan_queue_rows() -> list:
    """Every pending queue row, across every page.

    A DynamoDB scan returns at most 1MB and sets `LastEvaluatedKey` when
    there is more. This used to read the first page only. The audit table
    shares this table and grows with every run, so the day it crossed 1MB
    the queue would have started losing entries on restart -- silently,
    and exactly the loss the durable queue exists to prevent.
    """
    table = training_audit_table()
    rows: list = []
    start_key = None
    while True:
        page = table.scan(**({"ExclusiveStartKey": start_key} if start_key else {}))
        rows += [item for item in page.get("Items", [])
                 if item.get("record_type") == QUEUE_RECORD_TYPE]
        start_key = page.get("LastEvaluatedKey")
        if not start_key:
            return rows


def _undecimalize(item: dict) -> dict:
    """DynamoDB returns numbers as Decimal; the queue entry's ints are fed
    straight back into the training APIs, which expect ints."""
    out = {}
    for key, value in item.items():
        if isinstance(value, decimal.Decimal):
            out[key] = int(value) if value == value.to_integral_value() else float(value)
        else:
            out[key] = value
    out.pop("record_type", None)
    return out


def dataset_source_folders(folder: str, split: str = "") -> list:
    """The distinct folders an assembled dataset's images came FROM.

    `split` narrows to one of them. Without it both come back and the
    caller ends up printing "Training images: .../train, .../val", which
    says the training set includes the held-out images -- the opposite of
    what the split means.

    Empty for a folder that was not assembled, in which case the folder
    itself is already the origin and there is nothing to resolve.
    """
    sources = dataset_source_map(folder)
    if not sources:
        return []
    prefix = f"images/{split}/" if split else "images/"
    return sorted({
        origin.rsplit("/", 1)[0]
        for name, origin in sources.items()
        if name.startswith(prefix) and "/" in origin
    })


def queue_add(entry: dict) -> None:
    # Resolved once, here, rather than on every poll: the queue card is
    # re-read every 15s and this is an S3 GET. `data/assembled/<date>` is
    # a name nobody chose; what belongs on the card is the folder the
    # operator actually picked.
    train_folder = str(entry.get("dataset_folder") or "")
    eval_folder = str(entry.get("eval_dataset_folder") or "") or train_folder
    # Kept when the caller already knows them -- a composed run's real
    # folders, which the yaml's own `_generated/` home cannot reveal.
    if train_folder and not entry.get("dataset_source_folders"):
        entry["dataset_source_folders"] = dataset_source_folders(train_folder, "train")
    if eval_folder and not entry.get("eval_source_folders"):
        # The evaluator runs model.val(split="val"), so the evaluation
        # images are the val split's origins -- not both splits'. A
        # dataset with no val split has data.yaml point val at
        # images/train, and then the train origins are the honest answer.
        entry["eval_source_folders"] = (
            dataset_source_folders(eval_folder, "val")
            or dataset_source_folders(eval_folder, "train")
        )
    with _queue_lock:
        if len(_training_queue) >= MAX_QUEUE_SIZE:
            raise ValueError(f"Queue is full (max {MAX_QUEUE_SIZE} runs) -- remove an entry before adding another.")
        _training_queue.append(entry)
    _persist_queue_entry(entry)


def queue_remove(entry_id: str) -> bool:
    with _queue_lock:
        before = len(_training_queue)
        _training_queue[:] = [e for e in _training_queue if e["id"] != entry_id]
        removed = len(_training_queue) < before
    if removed:
        _forget_queue_entry(entry_id)
    return removed


#: The id of the row recording whether the queue is running. One row,
#: overwritten -- it is a switch, not a history.
QUEUE_STATE_ID = "queue-state"
QUEUE_STATE_RECORD_TYPE = "queue_state"


def _persist_queue_started(started: bool) -> None:
    """Record whether the queue is running, so a restart can resume it.

    The entries already survived a restart; the SWITCH did not. So after
    every deploy the queue came back full and stopped, and an entry sat
    there indefinitely with nothing on screen explaining why -- "Restored
    1 queued training run(s)" in the log, and then silence.

    Persisted rather than defaulted either way: defaulting to started
    would spend GPU nobody asked for after a crash, and defaulting to
    stopped is what produced this. The honest answer is the state it was
    actually in.
    """
    try:
        training_audit_table().put_item(Item={
            "id": QUEUE_STATE_ID,
            "record_type": QUEUE_STATE_RECORD_TYPE,
            "started": bool(started),
            "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not persist the queue's started flag: {exc}")


def _read_queue_started() -> bool:
    try:
        got = training_audit_table().get_item(Key={"id": QUEUE_STATE_ID}).get("Item")
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not read the queue's started flag: {exc}")
        return False
    return bool((got or {}).get("started"))


def queue_start() -> bool:
    """Returns False if there's nothing to start (empty queue, nothing
    already running)."""
    global _queue_started
    with _queue_lock:
        if not _training_queue and _queue_running is None:
            return False
        _queue_started = True
    _persist_queue_started(True)
    try_advance_queue()  # don't wait up to ~25s for the next poll tick
    return True


def queue_stop() -> None:
    global _queue_started
    with _queue_lock:
        _queue_started = False
    _persist_queue_started(False)


def try_advance_queue() -> None:
    """Called both by the background poll loop (every ~25s) and
    immediately after "Start Queue", so the first run doesn't have to wait
    for the next poll tick. If the queue has been started, nothing else is
    occupying the account's one GPU slot, and there's at least one pending
    entry, pops it and starts it."""
    global _queue_running, _queue_started, _queue_completed_count
    global _queue_last_error, _queue_starting
    with _queue_lock:
        if _queue_starting:
            # Another caller is mid-submission. It has already taken its
            # entry, and the GPU check below cannot yet see the run it is
            # starting -- so without this, two tickers would each pop one.
            return
        reason = training_already_running_reason(get_training_progress())
        if reason:
            # Something is Executing -- either the queue's own current
            # entry (leave _queue_running as-is) or an unrelated run
            # started elsewhere. Either way, not safe to start the next.
            return
        # Clearing the finished entry happens BEFORE the started check, not
        # after. Pressing Stop Queue sets _queue_started False, and this
        # function used to return on that immediately -- so the entry that
        # had just been stopped stayed in _queue_running forever, the card
        # went on saying "Running" with nothing executing, and every further
        # Stop Training answered 409 because there was genuinely nothing
        # left to stop. Stopping the queue means "start no more", not
        # "forget how to notice the current one ended".
        if _queue_running is not None:
            _queue_completed_count += 1
            _queue_running = None
        if not _queue_started:
            return
        if not _training_queue:
            _queue_started = False
            return
        entry = _training_queue.pop(0)
        _queue_starting = True

    # --- out of the lock from here ----------------------------------------
    #
    # What follows is two network round trips. Holding `_queue_lock` across
    # them is what made adding a run, removing one, or merely looking at
    # the card wait on a SageMaker call. `_queue_starting`, set above,
    # keeps the mutual exclusion that the lock was providing.
    try:
        # Dropped from the durable copy as soon as it leaves the queue,
        # whether it goes on to start or to be skipped -- a restart must
        # not resurrect a run that has already had its turn.
        _forget_queue_entry(entry["id"])
        try:
            if entry.get("custom_classes"):
                # Started directly as a pipeline execution rather than via
                # CodeBuild, so there is no build_id -- the execution arn
                # is what identifies it. Recorded under its own key so the
                # UI can tell the two kinds of queued run apart.
                result = start_custom_training_run(
                    dataset_folder=entry["dataset_folder"],
                    eval_dataset_folder=entry.get("eval_dataset_folder", ""),
                    base_model_folder=entry.get("base_model_folder", ""),
                    base_model_key=entry.get("base_model_key", ""),
                    base_model_scratch=entry.get("base_model_scratch", False),
                    base_model_package_arn=entry.get("base_model_package_arn", ""),
                    architecture=entry.get("architecture", ""),
                    detector_slug=entry.get("detector_slug", ""),
                    # Carried from the entry, where it was captured when
                    # the run was QUEUED -- the live slot may have been
                    # restored to something else in the meantime.
                    snapshot={
                        "snapshot_id": entry.get("snapshot_id", ""),
                        "created_at": entry.get("snapshot_created_at", ""),
                        "files": entry.get("snapshot_files"),
                        "pinned": bool(entry.get("snapshot_pinned")),
                        "note": entry.get("snapshot_note", ""),
                    } if entry.get("snapshot_id") else None,
                    patience=entry.get("patience", 0),
                    # Captured when the run was QUEUED, like the snapshot
                    # above: the card may have been changed since, and the
                    # entry should run what it was queued with.
                    hyperparams=entry.get("hyperparams") or {},
                    # A composed dataset's prefixes and yaml, captured
                    # when the run was queued. Blank for the ordinary
                    # single-folder case.
                    train_data_uris=entry.get("train_data_uris", ""),
                    data_yaml_uri=entry.get("data_yaml_uri", ""),
                    drop_undeclared_classes=bool(entry.get("drop_undeclared_classes")),
                    eval_extra_uris=entry.get("eval_extra_uris", ""),
                    source="queue",
                    epochs=entry["epochs"], img_size=entry["img_size"],
                    batch=entry["batch"], seed=entry["seed"],
                    notes=entry["notes"],
                )
                entry["build_id"] = ""
                entry["execution_arn"] = result["execution_arn"]
                entry["run_id"] = result["run_id"]
                if entry.get("dataset_version_id"):
                    from .. import data as _data
                    _data.mark_version(entry["dataset_version_id"], state="started",
                                           run_id=result["run_id"])
            else:
                entry["build_id"] = start_training_run(
                    entry["generation"], entry["size"], entry["epochs"],
                    entry["dataset_folder"], entry["base_model_folder"], entry["notes"],
                    entry["img_size"], entry["batch"], entry["seed"],
                    entry.get("eval_dataset_folder", ""), source="queue",
                )
            entry["started_at"] = datetime.now(timezone.utc).isoformat()
            with _queue_lock:
                _queue_running = entry
                _queue_last_error = ""
        except Exception as exc:
            # Don't put it back -- retrying the same bad entry forever
            # would just spin every poll.
            with _queue_lock:
                _queue_last_error = (
                    f"Queue entry (YOLO{entry.get('generation', '?')}-"
                    f"{entry.get('size', '?')}) failed to start, skipped: {exc}")
            print(_queue_last_error)
    finally:
        # Cleared whatever happened, or one failed submission would stop
        # the queue advancing for the life of the process.
        with _queue_lock:
            _queue_starting = False


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


def list_dataset_label_names(folder: str) -> tuple[list, int, bool]:
    """The label files, listed exactly as the images are.

    A label file is not an accessory to the image -- it IS the supervision
    the run learned from. Two datasets can hold identical images and train
    to completely different models, and recording only the images cannot
    tell those two runs apart.
    """
    return _list_dataset_names(folder, (".txt",))


def list_dataset_image_names(folder: str) -> tuple[list, int, bool]:
    """(sample of image names, total found, whether either bound was hit).

    Recursive, unlike list_s3_images: a YOLO dataset is images/train/... and
    images/val/..., so a Delimiter-scoped listing of the dataset root finds
    no images at all. Names are returned relative to `folder` so the split
    each file came from stays visible.
    """
    return _list_dataset_names(folder, IMAGE_EXTENSIONS)


def _list_dataset_names(folder: str, extensions: tuple) -> tuple[list, int, bool]:
    """Shared by the image and label listings above -- same recursion,
    same bounds, same relative-name shape, only the extensions differ."""
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
                if key.endswith("/") or not key.lower().endswith(extensions):
                    continue
                total += 1
                # Every name, uncapped: the caller writes the complete
                # list to S3 and only trims what goes into DynamoDB.
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


TRAINING_FILES_TABLE_NAME = os.environ.get(
    "TRAINING_FILES_TABLE", "track2-yolo-training-files",
)
# DynamoDB accepts at most 25 items per BatchWriteItem call.
_BATCH_WRITE_MAX = 25


def training_files_table():
    return _dynamodb().Table(TRAINING_FILES_TABLE_NAME)


def write_file_rows(audit_id: str, role: str, names: list, folder: str = "",
                    sources: dict | None = None) -> int:
    """One row per file a run used. Returns how many were written.

    Not on the run's own row: a DynamoDB item is capped at 400KB, and at
    ~101 bytes per filename a 24,000-image dataset needs 2.4MB. Going over
    does not truncate the list -- it rejects the write, so the whole audit
    row would be lost because its file list was too long.

    Labels get rows too, not just images: a .txt file IS the supervision
    the run learned from, so a record that omits it cannot distinguish two
    runs over the same images with different labels. `kind` separates them.

    `name` stays exactly what it has always been -- the path relative to
    the dataset folder -- because the by-file-name GSI is keyed on it and
    runs_using_file() looks files up by that value. The folder it was
    relative TO is what was missing: without it a row says
    "images/train/x.jpg" and cannot say where that lived, so the same
    dataset copied to a second location is indistinguishable from the
    first. `folder`, `s3_uri`, `split` and `file_name` are added
    alongside, never in place of, `name`.

    Never raises. Losing the file rows must not lose the run.
    """
    if not audit_id or not names:
        return 0
    written = 0
    table = training_files_table()
    root = (folder or "").strip("/")
    sources = sources or {}
    try:
        # The index is zero-padded so the sort order is the listing order,
        # and prefixed with the role AND the kind so one Query can fetch
        # just the training files, or just the evaluation ones, or just
        # the labels -- begins_with("train#") still spans both kinds.
        counters: dict = {}
        with table.batch_writer() as batch:
            for name in names:
                kind = "label" if name.lower().endswith(".txt") else "image"
                index = counters.get(kind, 0)
                counters[kind] = index + 1
                item = {
                    "audit_id": audit_id,
                    "file_key": f"{role}#{kind}#{index:06d}",
                    "role": role,
                    "kind": kind,
                    "name": name,
                }
                if root:
                    item["folder"] = root
                    item["s3_uri"] = f"s3://{BUCKET}/{root}/{name}"
                # Where the file came from BEFORE it was copied into the
                # assembled folder. The assembled path is an artefact of
                # the copy; this is the folder a person actually chose.
                origin = sources.get(name)
                if origin:
                    item["source_key"] = origin
                # The split the file actually sat in, which `role` does
                # NOT give: role is "the folder used for training" vs
                # "the folder used for evaluation", and each lists the
                # whole dataset, both splits included. Without this a row
                # cannot answer whether a file was trained on or held out.
                parts = name.split("/")
                if len(parts) >= 2 and parts[0] in ("images", "labels"):
                    item["split"] = parts[1]
                item["file_name"] = parts[-1]
                batch.put_item(Item=item)
                written += 1
    except Exception as exc:
        print(f"Could not write file rows for {audit_id}/{role}: {exc}")
    return written


def read_file_rows(audit_id: str, role: str = "", kind: str = "") -> list:
    """Every file name a run used, in listing order.

    `kind` narrows to "image" or "label"; omitted, both come back. Rows
    written before labels were recorded have no kind segment in their
    key, so a kind filter simply finds none of them rather than
    mis-reporting them as images.
    """
    if not audit_id:
        return []
    names: list = []
    key = Key("audit_id").eq(audit_id)
    if role:
        prefix = f"{role}#{kind}#" if kind else f"{role}#"
        key = key & Key("file_key").begins_with(prefix)
    kwargs: dict = {"KeyConditionExpression": key}
    try:
        while True:
            page = training_files_table().query(**kwargs)
            names.extend(str(item.get("name", "")) for item in page.get("Items", []))
            last = page.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
    except Exception as exc:
        print(f"Could not read file rows for {audit_id}/{role}: {exc}")
    return names


def read_file_row_paths(audit_id: str, role: str = "") -> list:
    """The fully-qualified path of every file a run used, in listing order.

    The origin where the row recorded one, the assembled path otherwise.
    `read_file_rows` still returns the bare relative names, because that
    is what the by-file-name index is keyed on and what
    runs_using_file() takes.
    """
    if not audit_id:
        return []
    out: list = []
    key = Key("audit_id").eq(audit_id)
    if role:
        key = key & Key("file_key").begins_with(f"{role}#")
    kwargs: dict = {"KeyConditionExpression": key}
    try:
        while True:
            page = training_files_table().query(**kwargs)
            for item in page.get("Items", []):
                name = str(item.get("name", ""))
                folder = str(item.get("folder", "")).strip("/")
                out.append(
                    str(item.get("source_key"))
                    if item.get("source_key")
                    else (f"{folder}/{name}" if folder else name)
                )
            last = page.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
    except Exception as exc:
        print(f"Could not read file paths for {audit_id}/{role}: {exc}")
    return out


def runs_using_file(name: str, limit: int = 50) -> list:
    """Which training runs used a given image -- reverse lineage.

    The reason these are per-file rows rather than a blob of names: a list
    can be read back, but it cannot be asked this question.
    """
    if not name:
        return []
    try:
        page = training_files_table().query(
            IndexName="by-file-name",
            KeyConditionExpression=Key("name").eq(name),
            Limit=limit,
        )
    except Exception as exc:
        print(f"Could not look up runs using {name}: {exc}")
        return []
    # KEYS_ONLY, so the item carries the audit id and nothing else.
    seen, out = set(), []
    for item in page.get("Items", []):
        audit_id = str(item.get("audit_id", ""))
        if audit_id and audit_id not in seen:
            seen.add(audit_id)
            out.append(audit_id)
    return out


def delete_file_rows(audit_id: str) -> int:
    """Removes a run's file rows. Called when the run itself is deleted,
    so thousands of rows are not left behind pointing at nothing."""
    removed = 0
    try:
        table = training_files_table()
        kwargs: dict = {"KeyConditionExpression": Key("audit_id").eq(audit_id),
                        "ProjectionExpression": "audit_id, file_key"}
        keys = []
        while True:
            page = table.query(**kwargs)
            keys.extend(
                {"audit_id": i["audit_id"], "file_key": i["file_key"]}
                for i in page.get("Items", [])
            )
            last = page.get("LastEvaluatedKey")
            if not last:
                break
            kwargs["ExclusiveStartKey"] = last
        with table.batch_writer() as batch:
            for key in keys:
                batch.delete_item(Key=key)
                removed += 1
    except Exception as exc:
        print(f"Could not delete file rows for {audit_id}: {exc}")
    return removed


def _run_artifacts(audit_id: str, folder: str) -> dict:
    """This run's yaml and file lists, in a folder of its own.

    Never raises: losing the copy must not lose the submission it
    describes.
    """
    from . import run_artifacts

    try:
        return run_artifacts.capture(audit_id, folder)
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not capture run artifacts for {audit_id}: {exc}")
        return {}


def _dataset_audit_block(folder: str, audit_id: str = "", role: str = "") -> dict:
    """What was used, recorded twice over.

    Every file name goes to the per-file table; the DynamoDB row keeps
    only a bounded sample for inline display, because a single item is
    capped at 400KB and a dataset of 24,000 images would blow it, taking
    the whole audit write with it. So `file_names` is a preview and the
    per-file rows are the truth.

    WHERE those names come from depends on the dataset. A recursive
    listing of the folder is right when the folder IS the selection --
    which it was while `assemble_dataset` copied the chosen images into
    one. It is wrong now that a yaml names week folders in place: the
    folder is the whole root, and a run that trained on 587 images would
    be recorded as having used all 748.

    So when the yaml points at FILE LISTS, those lists are the selection
    and they are read instead. An audit that overstates what a run saw is
    worse than none, because it reads as precise.
    """
    from . import run_artifacts

    selection = run_artifacts.read_selection(folder)
    if selection["images"]:
        names = selection["images"]
        label_names = selection["labels"]
        total, label_total = len(names), len(label_names)
        truncated = label_truncated = False
    else:
        names, total, truncated = list_dataset_image_names(folder)
        label_names, label_total, label_truncated = list_dataset_label_names(folder)
    sources = dataset_source_map(folder)
    rows_written = (
        write_file_rows(audit_id, role, names + label_names, folder, sources)
        if (audit_id and role) else 0
    )
    root = (folder or "").strip("/")

    def displayed(name: str) -> str:
        """The origin where one is known, the assembled path otherwise.

        Fully qualified either way: a bare "images/train/x.jpg" names a
        file in no particular dataset, and those names recur in every
        pull.
        """
        return sources.get(name) or (f"{root}/{name}" if root else name)
    return {
        "folder": folder,
        "s3_uri": f"s3://{BUCKET}/{folder}/" if folder else "",
        # Still the IMAGE count, deliberately: the UI reads it as "N
        # images" and a dataset does not gain images by having its labels
        # recorded. The labels are counted separately.
        "file_count": total,
        "label_count": label_total,
        # Bounded for the 400KB item limit; the per-file rows are complete.
        "file_names": [displayed(n) for n in names[:AUDIT_MAX_FILE_NAMES]],
        # The distinct folders the images actually came from. Usually one
        # per split; shown instead of the assembled folder, which nobody
        # chose and nobody recognises.
        # Narrowed to the split this role actually reads, for the same
        # reason: listing both under "Training images" claims the run
        # trained on its own held-out set.
        "source_folders": sorted({
            sources[n].rsplit("/", 1)[0]
            for n in names
            if n in sources and n.startswith(f"images/{'val' if role == 'eval' else 'train'}/")
        })[:8] or sorted({
            sources[n].rsplit("/", 1)[0] for n in names if n in sources
        })[:8],
        "truncated": truncated or label_truncated or len(names) > AUDIT_MAX_FILE_NAMES,
        # How many rows the file table holds for this role, so a reader
        # can tell "none recorded" from "recorded, go and Query them".
        "file_rows": rows_written,
        # "list" -- the names are what the yaml's file lists selected.
        # "folder" -- they are everything under the dataset folder. The
        # two mean different things and a reader cannot tell them apart
        # from the names alone.
        "names_from": "list" if selection["images"] else "folder",
        "yaml_key": selection["yaml_key"],
    }


def presign_weights_download(uri: str, expires_seconds: int = 900) -> str:
    """A time-limited download link for a .pt already in this bucket.

    The browser fetches straight from S3. Streaming a 19MB checkpoint
    through the API container would put it on the one small box that also
    serves every page.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"'{uri}' is not an s3:// uri.")
    bucket, key = uri.removeprefix("s3://").split("/", 1)
    filename = key.rsplit("/", 1)[-1] or "model.pt"
    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            # Named after the group and version rather than arriving as
            # yet another "model.pt" in the downloads folder.
            "ResponseContentDisposition":
                f'attachment; filename="{_weights_download_name(key, filename)}"',
        },
        ExpiresIn=expires_seconds,
    )


def _weights_download_name(key: str, fallback: str) -> str:
    """models/<group>/v<n>/model.pt -> <group>-v<n>.pt"""
    parts = key.strip("/").split("/")
    if len(parts) >= 3 and parts[-1].endswith(".pt"):
        return f"{parts[-3]}-{parts[-2]}.pt"
    return fallback


def run_settings_for(audit: dict) -> dict:
    """Patience, hyperparameters and class options for a run's record.

    From the record itself for runs submitted since these were stored
    (2026-10-26). For older runs, read back from the SageMaker execution's
    own parameters -- the values the job was really sent -- so their
    export is complete too. `recorded_in` says which.
    """
    if "hyperparams" in audit:
        return {
            "patience": (audit.get("parameters") or {}).get("patience"),
            "hyperparams": dict(audit.get("hyperparams") or {}),
            "training_options": dict(audit.get("training_options") or {}),
            "recorded_in": "the training record",
        }
    arn = str(audit.get("execution_arn") or "")
    if not arn:
        return {"recorded_in": ""}
    try:
        params = {p["Name"]: p.get("Value", "") for p in
                  sm.list_pipeline_parameters_for_execution(
                      PipelineExecutionArn=arn).get("PipelineParameters", [])}
    except Exception as exc:                                       # noqa: BLE001
        print(f"Could not read the parameters of {arn}: {exc}")
        return {"recorded_in": ""}
    hyper: dict = {}
    if params.get("Hyperparams"):
        import base64
        try:
            hyper = json.loads(base64.b64decode(params["Hyperparams"]).decode())
        except Exception:                                          # noqa: BLE001
            hyper = {"(could not decode)": params["Hyperparams"]}
    classes = [c for c in str(params.get("Classes") or "").split(",") if c.strip()]
    return {
        "patience": int(params["Patience"]) if str(params.get("Patience") or "").isdigit() else None,
        "hyperparams": hyper,
        "training_options": {
            "classes": classes,
            "single_cls": str(params.get("SingleCls") or "").lower() == "true",
            "drop_undeclared_classes": str(params.get("DropUndeclaredClasses") or "").lower() == "true",
        },
        "recorded_in": "the SageMaker execution's parameters",
    }


def audit_as_csv(audit: dict, settings: dict | None = None) -> str:
    """One run as a CSV a person can read -- in SECTIONS, two columns each.

    Run, Model, Training settings, Hyperparameters, Class options, Data
    version and snapshot, Training data, Evaluation data, Metrics, then
    anything else the record holds, and last every file name used. A
    section is a title row, a header row, its rows and a blank line, so it
    reads top to bottom in Excel and still parses as CSV.

    It was `section,key,value` with dotted keys (`model.version`) and no
    hyperparameters at all, which parsed fine and read badly.

    `settings` is `run_settings_for(audit)`; left out, only what the record
    itself holds is used, so this never reaches AWS on its own.
    """
    from .hyperparams import GROUPS

    settings = settings if settings is not None else (
        run_settings_for(audit) if "hyperparams" in audit else {"recorded_in": ""})
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    shown: set = set()

    def text(value) -> str:
        if value is None or value == "":
            return ""
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, (list, tuple)):
            return "; ".join(text(v) for v in value)
        if isinstance(value, dict):
            return "; ".join(f"{k}={text(v)}" for k, v in value.items())
        return str(value)

    def section(title: str, rows: list, header=("Field", "Value")) -> None:
        # The name in capitals, a trailing "(N files)" left as it is.
        head, _, tail = title.partition(" (")
        writer.writerow([head.upper() + (f" ({tail}" if tail else "")])
        writer.writerow(list(header))
        if rows:
            writer.writerows(rows)
        else:
            writer.writerow(["(nothing recorded)"])
        writer.writerow([])

    def flat(value, path: str = "") -> list:
        """Nested dicts as `a > b` rows, file name samples left out."""
        rows: list = []
        if isinstance(value, dict):
            for key, inner in value.items():
                if key == "file_names":
                    continue
                rows += flat(inner, f"{path} > {key}" if path else str(key))
        else:
            rows.append([path, text(value)])
        return rows

    def take(*keys):
        shown.update(keys)
        return [audit.get(k) for k in keys]

    writer.writerow(["TRAINING RUN RECORD"])
    writer.writerow(["Run ID", text(audit.get("id"))])
    writer.writerow([])

    status, submitted, started, ended, source, notes, arn, build = take(
        "status", "submitted_at", "started_at", "ended_at", "source", "notes",
        "execution_arn", "build_id")
    shown.update({"id", "record_type"})
    section("Run", [
        ["Status", text(status)],
        ["Submitted", text(submitted)],
        ["Started", text(started)],
        ["Ended", text(ended)],
        ["Started by", text(source)],
        ["Notes", text(notes)],
        ["SageMaker execution", text(arn)],
        *([["Build", text(build)]] if build else []),
    ])

    slug, arch, gen, size, base, model = take(
        "detector_slug", "architecture", "generation", "size", "base_model", "model")
    base = base or {}
    section("Model", [
        ["Model Name", text(slug)],
        ["Architecture", text(arch or size)],
        ["Pipeline", text(gen)],
        ["Started from", text(base.get("source"))],
        ["Base model package", text(base.get("package_arn"))],
        ["Base model folder", text(base.get("folder"))],
        *[[f"Registered model > {k}", text(v)] for k, v in (model or {}).items()],
    ])

    params = (take("parameters")[0] or {})
    patience = settings.get("patience", params.get("patience"))
    section("Training settings", [
        ["Epochs", text(params.get("epochs"))],
        ["Image size", text(params.get("img_size"))],
        ["Batch", text(params.get("batch"))],
        ["Seed", text(params.get("seed"))],
        ["Patience (early stopping)",
         "not recorded" if patience is None
         else "not set (Ultralytics default)" if int(patience) == 0 else text(patience)],
        *[[k, text(v)] for k, v in params.items()
          if k not in {"epochs", "img_size", "batch", "seed", "patience"}],
    ])

    shown.update({"hyperparams", "training_options"})
    hyper = settings.get("hyperparams")
    if hyper is None:
        hyper_rows = [["", "Not recorded for this run", ""]]
    else:
        hyper_rows = []
        for group, keys in GROUPS.items():
            for key in keys:
                hyper_rows.append([group, key, text(hyper[key]) if key in hyper
                                   else "default"])
        grouped = {k for keys in GROUPS.values() for k in keys}
        hyper_rows += [["Other", k, text(v)] for k, v in hyper.items() if k not in grouped]
        set_count = len(hyper)
        hyper_rows.insert(0, ["", f"{set_count} changed from Ultralytics' default"
                              + (f" -- from {settings['recorded_in']}" if settings.get("recorded_in") else ""),
                              ""])
    section("Hyperparameters", hyper_rows, header=("Group", "Setting", "Value"))

    options = settings.get("training_options")
    section("Class options", [] if options is None else [
        ["Classes trained (indices)", text(options.get("classes")) or "all the yaml declares"],
        ["Single class", text(bool(options.get("single_cls")))],
        ["Undeclared boxes dropped", text(bool(options.get("drop_undeclared_classes")))],
    ])

    snapshot, artifacts = take("snapshot", "artifacts")
    section("Dataset version and snapshot", flat(snapshot or {}, "Snapshot")
            + flat(artifacts or {}, "Copied artifacts"))

    for role, title in (("train_dataset", "Training data"), ("eval_dataset", "Evaluation data")):
        shown.add(role)
        section(title, flat(audit.get(role) or {}))

    metrics = take("metrics")[0]
    section("Metrics", flat(metrics or {}))

    rest = {k: v for k, v in audit.items() if k not in shown}
    if rest:
        section("Other recorded fields", flat(rest))

    # Every name, from the per-file rows rather than the capped sample on
    # the audit row -- the point of the export is completeness.
    audit_id = str(audit.get("id") or "")
    for role, prefix, title in (("train_dataset", "train", "Training files"),
                                ("eval_dataset", "eval", "Evaluation files")):
        names = read_file_row_paths(audit_id, prefix)
        block = audit.get(role) or {}
        if not names:
            # Runs audited before the file table existed still export what
            # they have, rather than exporting nothing.
            names = list(block.get("file_names") or [])
        # Qualify only what is not already qualified: a leading "images/"
        # or "labels/" is what a relative name always starts with.
        folder = str(block.get("folder") or "").strip("/")
        paths = [f"{folder}/{n}" if (n.startswith(("images/", "labels/")) and folder) else n
                 for n in names
                 # A dataset version's own list files (train.txt, val.txt)
                 # were recorded beside the images; they are not training
                 # files, and listed here they made 7,996 images read 7,999.
                 if not (n.startswith("_datasets/") and n.endswith(".txt"))]
        # No thousands separator: a comma in the title would quote the cell.
        section(f"{title} ({len(paths)} files)", [[p] for p in paths], header=("Path",))

    return out.getvalue()


_AUDIT_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def _composed_entries(data_yaml_uri: str) -> tuple:
    """(train entries, val entries) from a composed run's own data.yaml."""
    bucket, _, key = data_yaml_uri.removeprefix("s3://").partition("/")
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    doc = _yaml().safe_load(body) or {}
    def as_list(v):
        return [str(x) for x in (v if isinstance(v, list) else [v] if v else [])]
    return as_list(doc.get("train")), as_list(doc.get("val"))


def _composed_audit_block(entries: list, *, uri: str, multi_bucket: bool,
                          audit_id: str = "", role: str = "") -> dict:
    """An audit block for a run that read SEVERAL folders.

    `_dataset_audit_block` lists one dataset folder, and a composed run's
    folder is `_generated/<hash>` -- the yaml's home, holding no image. So
    the History card said "0 images" for a run that trained on 6,807, and
    "did not run what was submitted" because that folder differed from the
    prefixes the job really fetched. This lists the folders the yaml names
    instead, and records `uri` exactly as the job received it.
    """
    names, label_names, sources_seen = [], [], []
    for entry in entries:
        bucket, key = BUCKET, entry.strip("/")
        if multi_bucket and "/" in key:
            bucket, key = key.split("/", 1)
        try:
            for obj in _list_all(bucket, key + "/"):
                k = obj["Key"]
                if k.lower().endswith(_AUDIT_IMAGE_EXTS):
                    names.append(k if bucket == BUCKET else f"{bucket}/{k}")
            parts = key.split("/")
            if "images" in parts:
                i = len(parts) - 1 - parts[::-1].index("images")
                lab = "/".join(parts[:i] + ["labels"] + parts[i + 1:])
                for obj in _list_all(bucket, lab + "/"):
                    if obj["Key"].endswith(".txt"):
                        label_names.append(obj["Key"] if bucket == BUCKET else f"{bucket}/{obj['Key']}")
        except Exception as exc:                                   # noqa: BLE001
            print(f"Audit could not list {bucket}/{key}: {exc}")
        sources_seen.append(entry)
    rows = (write_file_rows(audit_id, role, names + label_names, "", {})
            if (audit_id and role) else 0)
    return {
        "folder": ", ".join(entries),
        "s3_uri": uri,
        "file_count": len(names),
        "label_count": len(label_names),
        "file_names": names[:AUDIT_MAX_FILE_NAMES],
        "source_folders": sources_seen[:8],
        "truncated": len(names) > AUDIT_MAX_FILE_NAMES,
        "file_rows": rows,
        "names_from": "folder",
        "yaml_key": "",
        "composed": True,
    }


def _list_all(bucket: str, prefix: str):
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        yield from page.get("Contents") or []


def composed_audit_fields(*, train_data_uris: str, data_yaml_uri: str,
                          eval_uri: str = "", eval_extra_uris: str = "",
                          audit_id: str = "") -> dict:
    """The three audit fields a composed run needs instead of the folder-
    based ones: train_dataset, eval_dataset, eval_is_train_dataset.

    Held out when no folder is on both sides. The evaluation-only yaml
    repeats `val:` under `train:` because Ultralytics requires the key;
    judging "held out" from that file reported every composed run as
    evaluated on its training set."""
    if "/_datasets/dv-" in data_yaml_uri:
        return _version_audit_fields(data_yaml_uri, train_data_uris=train_data_uris,
                                     eval_uri=eval_uri, eval_extra_uris=eval_extra_uris,
                                     audit_id=audit_id)
    train, val = _composed_entries(data_yaml_uri)
    multi = len({u.removeprefix("s3://").split("/", 1)[0]
                 for u in train_data_uris.split(",") if u.strip()}) > 1
    eval_uri_shown = ",".join(u for u in [eval_uri] + eval_extra_uris.split(",") if u.strip())
    return {
        "train_dataset": _composed_audit_block(train, uri=train_data_uris, multi_bucket=multi,
                                               audit_id=audit_id, role="train"),
        "eval_dataset": _composed_audit_block(val, uri=eval_uri_shown, multi_bucket=multi,
                                              audit_id=audit_id, role="eval"),
        "eval_is_train_dataset": bool(set(t.strip("/") for t in train) & set(v.strip("/") for v in val)),
    }


def _version_audit_fields(data_yaml_uri: str, *, train_data_uris: str, eval_uri: str,
                          eval_extra_uris: str, audit_id: str = "") -> dict:
    """Audit fields for a run on a dataset version: its manifest IS the
    list of images, so nothing is re-listed and nothing can have moved."""
    bucket, _, key = data_yaml_uri.removeprefix("s3://").partition("/")
    folder = key.rsplit("/", 1)[0]
    manifest = json.loads(s3.get_object(Bucket=bucket, Key=f"{folder}/manifest.json")["Body"].read())

    def block(names: list, uri: str, role: str) -> dict:
        shown = [n.split("/", 1)[1] if n.split("/", 1)[0] == BUCKET else n for n in names]
        rows = write_file_rows(audit_id, role, shown, "", {}) if (audit_id and role) else 0
        return {"folder": f"{folder} (dataset version)", "s3_uri": uri,
                "file_count": len(shown), "label_count": 0,
                "file_names": shown[:AUDIT_MAX_FILE_NAMES],
                "source_folders": list(manifest.get("sources") or [])[:8],
                "truncated": len(shown) > AUDIT_MAX_FILE_NAMES, "file_rows": rows,
                "names_from": "list", "yaml_key": f"{folder}/data.yaml", "composed": True,
                "dataset_version_id": manifest.get("version_id", "")}
    eval_shown = ",".join(u for u in [eval_uri] + eval_extra_uris.split(",") if u.strip())
    train = list(manifest.get("train") or [])
    val = list(manifest.get("val") or [])
    return {"train_dataset": block(train, train_data_uris, "train"),
            "eval_dataset": block(val, eval_shown, "eval"),
            "eval_is_train_dataset": bool(set(train) & set(val))}


def refresh_composed_audits(limit: int = 50) -> list:
    """Rewrite the dataset fields of composed runs already in the history,
    so cards written before this fix read correctly. Only rows whose job
    received a comma-separated training URI; nothing else is touched."""
    fixed = []
    table = training_audit_table()
    rows, kwargs = [], {"FilterExpression": "record_type = :t",
                        "ExpressionAttributeValues": {":t": "training_audit"}}
    while True:                          # every page, not just the first 1 MB
        resp = table.scan(**kwargs)
        rows += resp.get("Items") or []
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    for row in rows:
        actual = row.get("actual") or {}
        uri = str(actual.get("train_data_s3_uri") or "")
        if "," not in uri:
            continue
        yaml_key = ((row.get("artifacts") or {}).get("yaml_key") or "")
        yaml_uri = f"s3://{BUCKET}/{yaml_key}" if yaml_key else ""
        if not yaml_uri and row.get("execution_arn"):
            # No copy in the run's own folder: the yaml the execution was
            # actually given, from its own parameters.
            try:
                params = sm.list_pipeline_parameters_for_execution(
                    PipelineExecutionArn=row["execution_arn"])["PipelineParameters"]
                yaml_uri = next((x["Value"] for x in params if x["Name"] == "DataYamlUri"), "")
            except Exception as exc:                               # noqa: BLE001
                print(f"Could not read parameters for {row.get('id')}: {exc}")
        if not yaml_uri:
            continue
        try:
            fields = composed_audit_fields(train_data_uris=uri, data_yaml_uri=yaml_uri,
                                           eval_uri=str(row.get("eval_dataset", {}).get("s3_uri") or ""))
            table.update_item(
                Key={"id": row["id"]},
                UpdateExpression="SET train_dataset = :t, eval_dataset = :e, eval_is_train_dataset = :h",
                ExpressionAttributeValues={":t": fields["train_dataset"], ":e": fields["eval_dataset"],
                                           ":h": fields["eval_is_train_dataset"]})
            fixed.append(row["id"])
        except Exception as exc:                                   # noqa: BLE001
            print(f"Could not refresh audit {row.get('id')}: {exc}")
        if len(fixed) >= limit:
            break
    return fixed


def write_training_audit(
    *, build_id, generation, size, epochs, img_size, batch, seed,
    dataset_folder, eval_dataset_folder, base_model_folder, notes, source="manual",
    architecture="", base_model_package_arn="", base_model_source="",
    detector_slug="", snapshot=None,
    # A composed run: the prefixes and yaml it really reads.
    train_data_uris="", data_yaml_uri="", eval_extra_uris="", eval_uri="",
    # Everything else the job was sent (2026-10-26). Without these the
    # record named four of the run's settings and the CSV export said
    # nothing about the learning rate, optimiser or augmentation used.
    patience=None, hyperparams=None, training_options=None,
) -> dict:
    """One row per submitted run, written before the run is under way so
    the operator's exact inputs are captured even if the run later fails.
    Never raises: an audit write failing must not take down the training
    submission it is describing."""
    now = datetime.now(timezone.utc).isoformat()
    # Generated up front: the file manifests are keyed on it, and they are
    # written while this record is being built.
    audit_id = uuid.uuid4().hex[:16]
    record = {
        "id": audit_id,
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
            **({"patience": int(patience or 0)} if patience is not None else {}),
        },
        # The extra Ultralytics arguments exactly as sent -- typed by
        # `normalize_hyperparams`, so 0.01 is a number and False a bool.
        # {} means every one was left at Ultralytics' default; a MISSING
        # key means the run predates this field (see run_settings_for).
        "hyperparams": _decimalize(normalize_hyperparams(hyperparams)),
        # Which classes, single-class or not, and whether undeclared boxes
        # were dropped -- the three choices that change what is learned.
        "training_options": _decimalize(dict(training_options or {})),
        "notes": notes or "",
        # The yaml and the file lists, copied into a folder of this run's
        # own. Copied rather than pointed at: the live slots are
        # overwritten by the next Save, so a pointer answers "what does
        # that path hold now", which is a different question from "what
        # did this run train on".
        "artifacts": _run_artifacts(audit_id, dataset_folder),
        "train_dataset": _dataset_audit_block(dataset_folder, audit_id, "train"),
        # Falls back to the training dataset, which is what the pipeline
        # evaluates on when no separate evaluation set is chosen -- worth
        # recording explicitly, since "evaluated on its own training data"
        # is exactly the thing an audit should make visible rather than
        # leave as a blank to be misread as "unknown".
        "eval_dataset": _dataset_audit_block(
            eval_dataset_folder or dataset_folder, audit_id, "eval",
        ),
        "eval_is_train_dataset": _evaluates_on_its_training_images(
            eval_dataset_folder or dataset_folder, bool(eval_dataset_folder),
        ),
        # Which architecture was actually fine-tuned. Production runs carry
        # it as generation+size; a custom run has neither, and without this
        # field "which model is this?" is unanswerable after the fact.
        "architecture": str(architecture or ""),
        # WHICH CLASS this run was for. The registry group implies it --
        # "yolo11s-helmet-detector v9" -- but only to somebody who knows
        # how the group names are built, and a run with no detector
        # chosen has no group at all until it registers. Recorded
        # outright so the history answers it directly.
        "detector_slug": str(detector_slug or ""),
        # The frozen dataset, when there was one. It was reaching the
        # history only inside the free-text notes, which means it was
        # there by convention: nothing guaranteed it, nothing could
        # filter on it, and a run queued without that habit lost the
        # provenance entirely.
        "snapshot": dict(snapshot or {}),
        "base_model": {
            "folder": base_model_folder or "",
            "package_arn": base_model_package_arn or "",
            # How it was chosen, as the dispatcher resolved it: scratch,
            # file, folder or package. Falls back to the old inference for
            # production runs, which do not go through the dispatcher.
            "source": base_model_source or (
                "registry-package" if base_model_package_arn
                else "folder" if base_model_folder
                else "registry-latest-approved"
            ),
        },
        "model": None,
        "metrics": None,
        "execution_arn": "",
    }
    # A composed run's dataset fields come from the folders its yaml
    # names, not from `_generated/<hash>` -- see composed_audit_fields.
    if train_data_uris and data_yaml_uri:
        try:
            record.update(composed_audit_fields(
                train_data_uris=train_data_uris, data_yaml_uri=data_yaml_uri,
                eval_uri=eval_uri, eval_extra_uris=eval_extra_uris, audit_id=audit_id))
        except Exception as exc:                                   # noqa: BLE001
            print(f"Composed audit fields unavailable: {exc}")
    try:
        training_audit_table().put_item(Item=record)
    except Exception as exc:
        print(f"Could not write training audit for build {build_id}: {exc}")
    return record


def scan_training_audits(limit: int = 100) -> list:
    """Every audit row, newest first by submitted_at.

    Filtered on record_type: the same table now also holds the pending
    training queue, and without this those entries would appear in the
    history as runs that never started.
    """
    items: list = []
    kwargs: dict = {}
    try:
        while True:
            response = training_audit_table().scan(**kwargs)
            items.extend(
                item for item in response.get("Items", [])
                if item.get("record_type", "training_audit") == "training_audit"
            )
            last_key = response.get("LastEvaluatedKey")
            if not last_key:
                break
            kwargs["ExclusiveStartKey"] = last_key
    except Exception as exc:
        print(f"Could not scan training audits: {exc}")
        return []
    items.sort(key=lambda item: str(item.get("submitted_at") or ""), reverse=True)
    return items[:limit]


def delete_model_package_and_detach_audit(model_package_arn: str) -> dict:
    """Deletes a registered version and unlinks the run that produced it.

    The reverse of delete_training_audit, which deletes the pair together.
    Deleting only the package left the history claiming a run produced a
    version that no longer exists -- the same lie that function's docstring
    exists to prevent, pointing the other way.

    The audit ROW is kept, not deleted: it records the datasets, the
    hyperparameters and what the job was actually given, none of which the
    model package held. Removing a version is not a reason to destroy the
    record of the run. Its `model` is cleared so nothing dangles, and
    `model_lookup_done` stays set so the poller does not go looking for a
    package that was deliberately removed.
    """
    sm.delete_model_package(ModelPackageName=model_package_arn)
    # So the page it was deleted from does not go on listing it.
    forget_registry_listing()
    detached = []
    try:
        for audit in training_audit_table().scan().get("Items", []):
            model = audit.get("model") or {}
            if model.get("model_package_arn") != model_package_arn:
                continue
            training_audit_table().update_item(
                Key={"id": audit["id"]},
                UpdateExpression="SET #model = :none, #done = :true",
                ExpressionAttributeNames={"#model": "model", "#done": "model_lookup_done"},
                ExpressionAttributeValues={":none": None, ":true": True},
            )
            detached.append(audit["id"])
    except Exception as exc:
        # The package is already gone; failing here would report the delete
        # as failed and invite a retry that cannot succeed.
        print(f"Deleted {model_package_arn} but could not detach its audit rows: {exc}")
    return {"model_package_arn": model_package_arn, "deleted": True, "detached_runs": detached}


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
    # The file rows go too. Thousands of rows pointing at a run that no
    # longer exists would be invisible and never collected.
    files_removed = delete_file_rows(audit_id)
    return {"deleted": True, "deleted_model": deleted_model, "files_removed": files_removed}


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


def _other_runs_still_need(folder: str, audit_id: str) -> bool:
    """Is any OTHER run still going to read this folder?

    Assemble once, queue three runs on it: deleting when the first
    finishes would pull the dataset out from under the other two. A run
    that has not reached a terminal state has not read the folder yet.
    """
    folder = (folder or "").strip("/")
    if not folder:
        return True
    for other in scan_training_audits(limit=500):
        if str(other.get("id")) == str(audit_id):
            continue
        if str(other.get("status") or "") in TERMINAL_EXECUTION_STATUSES:
            continue
        for role in ("train_dataset", "eval_dataset"):
            if str((other.get(role) or {}).get("folder") or "").strip("/") == folder:
                return True
    return False


def cleanup_assembled_dataset(audit: dict) -> int:
    """Delete the assembled copy once the run that needed it is over.

    The copy is scaffolding: it exists because the trainer needs one
    prefix in the Ultralytics layout, and once the run has read it the
    only thing worth keeping is the RECORD of what it held -- which is
    already in the file table, one row per image and per label, with each
    row's origin.

    Four guards, because this deletes data:

    - only under ASSEMBLED_DATASET_PREFIX, so a folder someone chose
      themselves (`test_delete/...`, a pull) is never touched;
    - only once the run is terminal;
    - only if the lineage rows were actually written, so a failed record
      never becomes a silent data loss;
    - only if no other live run still points at the same folder.

    Returns how many objects were deleted. Never raises: losing the
    cleanup must not lose the audit update that carries it.
    """
    folder = str((audit.get("train_dataset") or {}).get("folder") or "").strip("/")
    prefix = ASSEMBLED_DATASET_PREFIX.strip("/")
    if not folder or not folder.startswith(f"{prefix}/"):
        return 0
    if str(audit.get("status") or "") not in TERMINAL_EXECUTION_STATUSES:
        return 0
    if audit.get("assembled_cleaned_at"):
        return 0
    rows = int((audit.get("train_dataset") or {}).get("file_rows") or 0)
    if rows <= 0:
        print(f"Not deleting {folder}: no file rows were recorded for {audit.get('id')}")
        return 0
    if _other_runs_still_need(folder, str(audit.get("id"))):
        print(f"Not deleting {folder}: another run still points at it")
        return 0

    deleted = 0
    try:
        keys = _list_keys(f"{folder}/")
        for start in range(0, len(keys), 1000):
            chunk = [{"Key": k} for k in keys[start:start + 1000]]
            if not chunk:
                continue
            s3.delete_objects(Bucket=BUCKET, Delete={"Objects": chunk, "Quiet": True})
            deleted += len(chunk)
    except Exception as exc:
        print(f"Could not delete assembled dataset {folder}: {exc}")
        return 0
    print(f"Deleted assembled dataset {folder}: {deleted} objects")
    return deleted


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
    # `weights_uri is None` alone made this retry on EVERY poll, forever,
    # for any run whose package the app cannot describe -- the replay group
    # is outside its IAM grant, so the publish could never succeed and the
    # poller hammered DescribeModelPackage every ~15s. `weights_publish_failed`
    # records that we tried and could not, which is not the same as "not
    # tried yet"; clearing that field is what asks for a retry.
    needs_model = (audit.get("model") is None or audit.get("metrics") is None
                   or (audit.get("weights_uri") is None
                       and not audit.get("weights_publish_failed")))
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
            # Unpack the checkpoint to a stable versioned key. Done here
            # rather than in a Lambda on the pipeline event because this
            # poller already retries until the model is found, so a failure
            # or an app restart simply retries on the next pass. Never fatal
            # to the audit update that carries it.
            try:
                updates["weights_uri"] = publish_versioned_weights(model["model_package_arn"])
            except Exception as exc:
                # Recorded, not just logged: without this the next poll
                # tries again immediately and forever, which for a package
                # outside this role's IAM grant is a permanent failure
                # retried every 15 seconds.
                updates["weights_publish_failed"] = str(exc)[:500]
                print(f"Could not publish versioned weights for {model['model_package_arn']}: {exc}")
        elif status in TERMINAL_EXECUTION_STATUSES:
            # The run is over and its steps carry no RegisterModel ARN, so
            # there is genuinely no model to find. Recorded so the next
            # pass stops looking, without that being confused with a
            # lookup that merely failed.
            updates["model_lookup_done"] = True

    # The run is over and its lineage is recorded, so the copy it was
    # reading has nothing left to say that the file rows do not.
    merged = {**audit, **updates}
    if str(merged.get("status") or "") in TERMINAL_EXECUTION_STATUSES:
        removed = cleanup_assembled_dataset(merged)
        if removed:
            updates["assembled_cleaned_at"] = datetime.now(timezone.utc).isoformat()
            updates["assembled_objects_deleted"] = removed

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
        registry_source_desc(model_package_arn, size)
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
        registry_source_desc(model_package_arn, size)
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
