"""Full-frame replay retraining coordinator; serialized Lambda invocations.

Daily import is separate. Each request pins a completed immutable snapshot.
Training/evaluation read only its derived person=0, helmet=1 dataset.
No endpoint deployment or model approval is performed here.
"""
import hashlib
import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from datetime import timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from prepare_replay_dataset import prepare

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
BUCKET = "track2-s3"
ROOT = "replay/fullframe/"
PIPELINE = "track2-yolo-training-pipeline"
s3 = boto3.client("s3", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)

# Per-run tunable hyperparameters: the event's key -> the pipeline
# parameter it sets. The pipeline declares a DefaultValue for each, so a
# request that names none of them runs exactly as it did before these
# were parameters at all -- which is what the weekly schedule does.
HYPERPARAMETERS = {"epochs": "Epochs", "img_size": "ImgSize", "batch": "Batch", "seed": "Seed"}
# Bounds mirror the UI's own validation. Enforced here too because this
# Lambda is reachable without going through it.
HYPERPARAMETER_LIMITS = {"epochs": (1, 1000), "img_size": (32, 4096), "batch": (1, 512), "seed": (0, 2**31 - 1)}


def _s3_folder(value, label):
    value = str(value).strip()
    if not value.startswith("s3://") or value.count("/") < 3:
        raise ValueError("{} must be an s3:// folder URI, got {!r}".format(label, value))
    return value if value.endswith("/") else value + "/"


def requested_datasets(event):
    """An explicit choice of what to train and evaluate on.

    Absent, the run derives its dataset from the pinned snapshot exactly as
    it always has -- that path is untouched. Present, the caller has picked
    folders directly (the Training page's browsers), so the snapshot is not
    what gets trained on and, importantly, must not be marked consumed:
    doing so would make the next scheduled run skip a snapshot nothing had
    actually trained on.
    """
    chosen = {}
    if event.get("dataset_uri"):
        chosen["dataset_uri"] = _s3_folder(event["dataset_uri"], "dataset_uri")
    if event.get("eval_dataset_uri"):
        chosen["eval_dataset_uri"] = _s3_folder(event["eval_dataset_uri"], "eval_dataset_uri")
    return chosen


def requested_base_model(event):
    """An explicit choice of starting weights.

    Absent, the run uses the production pointer exactly as it always has --
    which is what the weekly schedule does, since its EventBridge event
    carries no base model. Present, the operator picked a checkpoint (a
    scratch seed, a registered package's artifact, or any .pt in the
    bucket) and it must survive to the pipeline instead of being replaced.
    """
    uri = event.get("base_model_uri")
    if not uri:
        return {}
    if not uri.startswith("s3://" + BUCKET + "/"):
        raise ValueError("base_model_uri must be an object in s3://" + BUCKET + "/")
    if not uri.endswith((".pt", ".tar.gz")):
        raise ValueError("base_model_uri must be a .pt or .tar.gz checkpoint")
    return {"base_model_uri": uri, "base_model_package_arn": "",
            "base_model_selection": "operator_choice",
            "base_model_selected_at": datetime.now(timezone.utc).isoformat()}


def requested_hyperparameters(event):
    """The overrides this request carries, validated. Absent or blank keys
    are omitted entirely rather than defaulted here, so the pipeline's own
    DefaultValue stays the single source of the fallback."""
    chosen = {}
    for key in HYPERPARAMETERS:
        raw = event.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError("{} must be a whole number, got {!r}".format(key, raw))
        low, high = HYPERPARAMETER_LIMITS[key]
        if not low <= value <= high:
            raise ValueError("{} must be between {} and {}, got {}".format(key, low, high, value))
        chosen[key] = str(value)
    return chosen


def read(key, optional=False):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except ClientError as exc:
        if optional and exc.response["Error"]["Code"] in {"NoSuchKey", "404"}:
            return None
        raise


def write(key, value):
    s3.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(value).encode(), ContentType="application/json")


def production_baseline():
    """Resolve the designated production artifact, never the newest candidate."""
    pointer = read(ROOT + "production/current.json")
    arn = (pointer or {}).get("model_package_arn", "")
    prefix = "arn:aws:sagemaker:ap-southeast-1:125192671932:model-package/track2-ppe-fullframe-replay/"
    if not arn.startswith(prefix) or not arn.removeprefix(prefix).isdigit():
        raise ValueError("valid full-frame production pointer required")
    package = sm.describe_model_package(ModelPackageName=arn)
    containers = package.get("InferenceSpecification", {}).get("Containers", [])
    if (package.get("ModelPackageStatus") != "Completed"
            or package.get("ModelApprovalStatus") != "Approved" or len(containers) != 1):
        raise ValueError("production package must be completed, approved and single-model")
    uri = containers[0].get("ModelDataUrl", "")
    if not uri.startswith("s3://track2-s3/") or uri != pointer.get("weights_uri"):
        raise ValueError("production pointer and registry weights disagree")
    if not uri.endswith((".pt", ".tar.gz")):
        raise ValueError("unsupported production artifact format")
    return {"base_model_uri": uri, "base_model_package_arn": arn,
            "base_model_selection": "production_pointer",
            "base_model_selected_at": datetime.now(timezone.utc).isoformat()}


def snapshot():
    latest = read("data/track2-datasets/track2_pull/latest.json")
    uri = latest["snapshot"]
    if not uri.startswith("s3://track2-s3/data/track2-datasets/track2_pull/pull") or not uri.endswith("/"):
        raise ValueError("unexpected snapshot location")
    prefix = uri.removeprefix("s3://track2-s3/")
    marker = read(prefix + "_SUCCESS.json")
    for key in ("snapshot", "source_fingerprint", "status"):
        if marker.get(key) != latest.get(key):
            raise ValueError("latest pointer and success marker disagree")
    if marker["status"] != "complete":
        raise ValueError("snapshot is not complete")
    return marker


def derive(marker, publish=True):
    prefix = marker["snapshot"].removeprefix("s3://track2-s3/")
    full = prefix + "track2/escalation_dataset_full_frame/"
    with tempfile.TemporaryDirectory() as temporary:
        src, dst = Path(temporary) / "snapshot", Path(temporary) / "derived"
        src.mkdir()
        (src / "_SUCCESS.json").write_text(json.dumps(marker))
        objects = [obj for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=full)
                   for obj in page.get("Contents", []) if not obj["Key"].endswith("/")]
        def download(obj):
            relative = Path(obj["Key"].removeprefix(prefix))
            if ".." in relative.parts or relative.is_absolute():
                raise ValueError("unsafe snapshot key")
            path = src / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            body = s3.get_object(Bucket=BUCKET, Key=obj["Key"], IfMatch=obj["ETag"])["Body"].read()
            path.write_bytes(body)
        with ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(download, objects))
        card = prepare(src, dst, "hero")
        # Metadata is descriptive, not a source of nonexistent training files.
        # Record stale entries; never manufacture missing images or move val to train.
        metadata = json.loads((src / "track2/escalation_dataset_full_frame/metadata.json").read_text())
        present = {Path(r["image"]).stem for r in card["samples"]}
        card["metadata_cases_without_image"] = sorted(set(metadata.get("cases", {})) - present)
        frames = {split: {case.get("frame_key") for key, case in metadata.get("cases", {}).items()
                         if key in present and case.get("split") == split and case.get("frame_key")}
                  for split in ("train", "val")}
        if frames["train"] & frames["val"]:
            raise ValueError("source frame identity leaks across full-frame train and val")
        card["validation"]["frame_key_train_val_leakage"] = False
        (dst / "dataset-card.json").write_text(json.dumps(card))
        destination = ROOT + "datasets/" + card["dataset_version"] + "/"
        if publish:
            existing = read(destination + "_READY.json", optional=True)
            if existing and existing.get("dataset_version") != card["dataset_version"]:
                raise ValueError("conflicting derived dataset")
            if not existing:
                files = [p for p in dst.rglob("*") if p.is_file() and p.name != "_READY.json"]
                def upload(path):
                    s3.upload_file(str(path), BUCKET, destination + path.relative_to(dst).as_posix())
                with ThreadPoolExecutor(max_workers=12) as pool:
                    list(pool.map(upload, files))
                write(destination + "_READY.json", {"dataset_version": card["dataset_version"]})
        return card, "s3://" + BUCKET + "/" + destination


def handler(event, context):
    if event.get("production_preflight") is True:
        return {"status": "validated_production_only", **production_baseline()}
    if event.get("hardening_failure_probe") is True:
        # Explicit operator-only fault injection via IAM-authorized invocation.
        # Runs before dataset/model access and cannot submit a training job.
        raise RuntimeError("ISOLATED_HARDENING_FAILURE_PROBE:" + str(event.get("probe_id")))
    monitoring = event.get("reconcile") is True
    if event.get("source") == "aws.events" and event.get("detail-type") == "Scheduled Event":
        write(ROOT + "schedule/last_received.json", {"event_id": event.get("id"),
              "scheduled_at": event["time"], "received_at": datetime.now(timezone.utc).isoformat()})
    # Completion events drain only a previously queued request; never create one.
    completion = monitoring or event.get("detail-type") == "SageMaker Model Building Pipeline Execution Status Change"
    if completion and not monitoring and event.get("detail", {}).get("currentPipelineExecutionStatus") not in {"Succeeded", "Failed", "Stopped"}:
        return {"status": "ignored"}
    active = sm.list_pipeline_executions(PipelineName=PIPELINE, MaxResults=100)["PipelineExecutionSummaries"]
    if monitoring:
        reconcile(active)
        if event.get("monitor_only") is True:
            return {"status": "reconciled_without_start", "executions_checked": len(active)}
    running = [x for x in active if x["PipelineExecutionStatus"] in {"Executing", "Stopping"}]
    if completion:
        request = read(ROOT + "pending.json", optional=True)
        if not request or request.get("status") != "queued" or running:
            return {"status": "no_pending_work"}
    else:
        manual = event.get("manual") is True
        if manual and not all(event.get(k) for k in ("request_id", "initiator", "reason")):
            raise ValueError("manual requests require request_id, initiator and reason")
        marker = snapshot()
        if event.get("dry_run"):
            baseline = production_baseline()
            card, uri = derive(marker, publish=False)
            return {"status": "validated", "dataset_uri": uri, "dataset_card": card, **baseline}
        identity = event.get("request_id") if manual else event.get("id") or marker["source_fingerprint"]
        run_id = "ff-" + hashlib.sha256(str(identity).encode()).hexdigest()[:32]
        previous = read(ROOT + "runs/" + run_id + "/request.json", optional=True)
        if previous:
            if manual and any(previous.get(k) != event.get(k) for k in ("initiator", "reason")):
                raise ValueError("idempotency key reused with different request details")
            request = previous
            if request.get("status") in {"started", "superseded"}:
                return request
        else:
            if not manual and read(ROOT + "consumed/" + marker["source_fingerprint"] + ".json", optional=True):
                return {"status": "already_consumed_snapshot"}
            request = {"run_id": run_id, "snapshot": marker, "manual": manual,
                       "initiator": event.get("initiator", "friday-schedule"), "reason": event.get("reason", "weekly retraining"),
                       "requested_at": datetime.now(timezone.utc).isoformat(), "status": "requested",
                       # Recorded on the request, not read from the event at
                       # submit time: a retry of this run_id must train with
                       # the values it was created with, not whatever a later
                       # caller happens to send.
                       "hyperparameters": requested_hyperparameters(event),
                       "datasets": requested_datasets(event),
                       **requested_base_model(event)}
    key = ROOT + "runs/" + request["run_id"] + "/request.json"
    write(key, request)
    pending = read(ROOT + "pending.json", optional=True)
    if not completion and pending and pending.get("status") == "queued" and pending["run_id"] != request["run_id"]:
        pending.update(status="superseded", superseded_by=request["run_id"])
        write(ROOT + "runs/" + pending["run_id"] + "/request.json", pending)
        write(ROOT + "pending.json", pending)
    if running:
        previous = read(ROOT + "pending.json", optional=True)
        if previous and previous["run_id"] != request["run_id"] and previous.get("status") == "queued":
            previous.update(status="superseded", superseded_by=request["run_id"])
            write(ROOT + "runs/" + previous["run_id"] + "/request.json", previous)
        request["status"] = "queued"
        write(ROOT + "pending.json", request)
        write(key, request)
        return request
    # An operator's explicit choice is as final as the pointer's. Without
    # "operator_choice" here, a chosen checkpoint was overwritten by
    # production_baseline() a few lines before submission, which is why the
    # base-model picker was decorative on this path.
    if request.get("base_model_selection") not in {"production_pointer", "operator_choice"}:
        request.update(production_baseline())
        # Persist before preparing data/submitting; retries retain this selection.
        write(key, request)
    run_root = "s3://" + BUCKET + "/" + ROOT + "runs/" + request["run_id"]
    chosen = request.get("datasets") or {}
    if chosen.get("dataset_uri"):
        # Caller picked the folder, so there is nothing to derive and no
        # dataset card to read classes from -- the training container reads
        # them out of the dataset's own data.yaml. classes is left unset
        # rather than asserting person/helmet about data nobody checked.
        uri = chosen["dataset_uri"]
        dataset_version = "manual-" + hashlib.sha256(uri.encode()).hexdigest()[:16]
        request.update(dataset_uri=uri, dataset_version=dataset_version, dataset_source="chosen",
                       model_role="fullframe-replay",
                       checkpoint_uri=run_root + "/checkpoints/",
                       recovery_policy="one_infrastructure_retry_per_step")
    else:
        card, uri = derive(request["snapshot"])
        dataset_version = card["dataset_version"]
        request.update(dataset_uri=uri, dataset_version=dataset_version, dataset_source="snapshot",
                       model_role="fullframe-replay", classes={"0": "person", "1": "helmet"},
                       checkpoint_uri=run_root + "/checkpoints/",
                       recovery_policy="one_infrastructure_retry_per_step")
    # Unset, evaluation reads the training data -- which is what this
    # pipeline has always done. Always sent, because the parameter has no
    # default and the step's input cannot fall back on its own.
    eval_uri = chosen.get("eval_dataset_uri") or uri
    request["eval_dataset_uri"] = eval_uri
    request["evaluation_is_held_out"] = eval_uri != uri
    write(key, request)
    parameters = {
        "TrainingDatasetUri": uri, "EvaluationDatasetUri": eval_uri,
        "EvaluationOutputUri": run_root + "/evaluation",
        "RunMetadataUri": run_root + "/request.json", "RunId": request["run_id"],
        "DatasetVersion": dataset_version, "BaseModelUri": request["base_model_uri"]}
    # Only the ones this request actually chose; the rest fall through to
    # the pipeline's declared defaults.
    for key, name in HYPERPARAMETERS.items():
        value = (request.get("hyperparameters") or {}).get(key)
        if value:
            parameters[name] = value
    response = sm.start_pipeline_execution(
        PipelineName=PIPELINE, ClientRequestToken=request["run_id"], PipelineExecutionDisplayName=request["run_id"],
        PipelineParameters=[{"Name": k, "Value": v} for k, v in parameters.items()])
    request.update(status="started", pipeline_execution_arn=response["PipelineExecutionArn"])
    write(key, request)
    # Only a run that actually trained on the snapshot consumes it. Marking
    # it consumed after a run on a hand-picked folder would make the next
    # scheduled run skip a snapshot nothing had trained on -- silently
    # losing a week of new data.
    if request.get("dataset_source") != "chosen":
        write(ROOT + "consumed/" + request["snapshot"]["source_fingerprint"] + ".json", request)
    if completion:
        write(ROOT + "pending.json", request)
    return request


def reconcile(executions):
    """Repair submission bookkeeping and detect missed Friday delivery; no new run."""
    for execution in executions:
        name = execution.get("PipelineExecutionDisplayName", "")
        if not name.startswith("ff-") or not name[3:].isalnum():
            continue
        key = ROOT + "runs/" + name + "/request.json"
        request = read(key, optional=True)
        if not request:
            continue
        state = execution["PipelineExecutionStatus"]
        if request.get("pipeline_status") != state or request.get("status") != "started":
            request.update(status="started", pipeline_status=state,
                           pipeline_execution_arn=execution["PipelineExecutionArn"],
                           reconciled_at=datetime.now(timezone.utc).isoformat())
            write(key, request)
            write(ROOT + "consumed/" + request["snapshot"]["source_fingerprint"] + ".json", request)
    start = os.environ.get("HARDENING_MONITOR_START")
    if not start:
        return
    now = datetime.now(ZoneInfo("Asia/Singapore"))
    due = (now - timedelta(days=(now.weekday() - 4) % 7)).replace(hour=23, minute=59, second=0, microsecond=0)
    if due > now:
        due -= timedelta(days=7)
    receipt = read(ROOT + "schedule/last_received.json", optional=True)
    received = datetime.fromisoformat(receipt["scheduled_at"].replace("Z", "+00:00")) if receipt else None
    missing = due >= datetime.fromisoformat(start) and now >= due + timedelta(hours=1) and (received is None or received < due)
    boto3.client("cloudwatch", region_name=REGION).put_metric_data(Namespace="Track2/FullframeRetraining",
        MetricData=[{"MetricName": "MissingScheduledDelivery", "Value": int(missing), "Unit": "Count"}])
