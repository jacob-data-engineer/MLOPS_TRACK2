"""Triggered by SageMaker's own "Model Package State Change" EventBridge
event whenever a model package's ModelApprovalStatus becomes Approved (see
monitoring.tf's event rule, filtered to just that transition, across all
four YOLO11 size groups).

For the size the approval was for (resolved from the event's
ModelPackageGroupName via GROUP_TO_SIZE):

1. Pulls the approved package's weights from its ModelDataUrl (same
   describe_model_package -> S3 fetch -> extract model.pt pattern already
   used in train.py's resolve_base_model_checkpoint()), and lands them at
   a fixed, well-known, per-size S3 location:

     s3://<PIPELINE_BUCKET>/latest_approved_model_weights/<size>/model.pt

   If that size's latest_approved_model_weights/<size>/model.pt already
   exists, it is archived first (not overwritten silently) to
   .../archive/model_<timestamp>.pt, via a server-side S3 copy.

2. Creates a new Model straight from the approved package (no re-download
   needed for this step -- create_model can reference a ModelPackageName
   directly) and creates/updates that size's Inference Component
   (track2-yolo-<size>) on the shared multi-size endpoint to point at it
   -- this is what makes "approve a candidate" and "it's now what the
   review-app UI serves for that size" the same action, no separate manual
   deploy step. Mirrors
   model_training/src/deploy_inference_component.py's registry-driven
   path; duplicated here rather than imported since this Lambda's zip is
   built standalone from this directory, not from model_training/.

3. Publishes an SNS notification to the existing reviewer_notifications
   topic so stakeholders are emailed.
"""

import json
import os
import tarfile
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

PIPELINE_BUCKET = os.environ["PIPELINE_BUCKET"]
NOTIFY_TOPIC_ARN = os.environ["NOTIFY_TOPIC_ARN"]
GROUP_TO_SIZE = json.loads(os.environ["GROUP_TO_SIZE_JSON"])  # {"track2-yolo11": "s", ...}
MULTI_SIZE_ENDPOINT_NAME = os.environ["MULTI_SIZE_ENDPOINT_NAME"]
MULTI_SIZE_VARIANT_NAME = os.environ["MULTI_SIZE_VARIANT_NAME"]
SAGEMAKER_EXECUTION_ROLE_ARN = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
PROJECT = os.environ.get("PROJECT", "track2-yolo")

COMPUTE_REQUIREMENTS = {
    "s": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 2048},
    "m": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 3072},
    "l": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 4096},
    "x": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 6144},
}

sm = boto3.client("sagemaker")
s3 = boto3.client("s3")
sns = boto3.client("sns")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _archive_existing_latest_if_present(latest_key: str, archive_prefix: str) -> str | None:
    """Archives the current latest_approved_model_weights/<size>/model.pt
    (if any) to archive/model_<timestamp>.pt via a server-side copy.
    Returns the archive key, or None if there was nothing to archive."""
    try:
        s3.head_object(Bucket=PIPELINE_BUCKET, Key=latest_key)
    except s3.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey"):
            return None
        raise

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    archive_key = f"{archive_prefix}model_{timestamp}.pt"
    s3.copy_object(Bucket=PIPELINE_BUCKET, Key=archive_key, CopySource={"Bucket": PIPELINE_BUCKET, "Key": latest_key})
    print(f"Archived existing latest weights -> s3://{PIPELINE_BUCKET}/{archive_key}")
    return archive_key


def _extract_and_upload_new_weights(model_data_url: str, latest_key: str, download_dir: str) -> None:
    bucket, key = _parse_s3_uri(model_data_url)
    tarball_path = os.path.join(download_dir, "model.tar.gz")
    s3.download_file(bucket, key, tarball_path)

    extract_dir = os.path.join(download_dir, "extracted")
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tarball_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    checkpoint_path = os.path.join(extract_dir, "model.pt")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Extracted '{model_data_url}' but found no model.pt at its root (searched {extract_dir})."
        )

    s3.upload_file(checkpoint_path, PIPELINE_BUCKET, latest_key)
    print(f"Uploaded new latest weights -> s3://{PIPELINE_BUCKET}/{latest_key}")


def _inference_component_exists(name: str) -> bool:
    try:
        sm.describe_inference_component(InferenceComponentName=name)
        return True
    except sm.exceptions.ResourceNotFound:
        return False
    except ClientError as exc:
        # Confirmed live: a not-found InferenceComponentName raises
        # ValidationException ("Could not find inference component ..."),
        # not ResourceNotFound -- inconsistent with most other Describe*
        # calls in this same API, but that's what the service returns.
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find" in str(exc):
            return False
        raise


def _deploy_to_inference_component(size: str, model_package_arn: str, container: dict) -> str:
    """Creates a Model straight from the approved package's own
    Image/ModelDataUrl and creates/updates that size's Inference
    Component to point at it. Returns the Model name, for the
    notification message.

    PrimaryContainer={"ModelPackageName": ...} (the simpler path
    deploy_endpoint.py uses for the single-model endpoint) is rejected
    on an Inference-Component-enabled endpoint -- confirmed live:
    "Model package cannot mix with SageMaker or bring-your-own images,
    and only one model package should be used for endpoints with
    Inference Components enabled." Resolving the package's own
    Image/ModelDataUrl (passed in by the caller, which already fetched
    it) and passing those directly sidesteps the restriction entirely."""
    suffix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_name = f"{PROJECT}-{size}-model-{suffix}"

    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={"Image": container["Image"], "ModelDataUrl": container["ModelDataUrl"]},
        ExecutionRoleArn=SAGEMAKER_EXECUTION_ROLE_ARN,
    )

    inference_component_name = f"{PROJECT}-{size}"
    specification = {
        "ModelName": model_name,
        "ComputeResourceRequirements": COMPUTE_REQUIREMENTS[size],
        "StartupParameters": {
            "ModelDataDownloadTimeoutInSeconds": 600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        },
    }

    if _inference_component_exists(inference_component_name):
        sm.update_inference_component(InferenceComponentName=inference_component_name, Specification=specification)
        print(f"Updated {inference_component_name!r} -> {model_package_arn} (model: {model_name})")
    else:
        sm.create_inference_component(
            EndpointName=MULTI_SIZE_ENDPOINT_NAME,
            VariantName=MULTI_SIZE_VARIANT_NAME,
            InferenceComponentName=inference_component_name,
            Specification=specification,
            RuntimeConfig={"CopyCount": 1},
        )
        print(f"Created {inference_component_name!r} -> {model_package_arn} (model: {model_name})")

    return model_name


def _notify(
    model_package_arn: str,
    size: str,
    model_data_url: str,
    archive_key: str | None,
    latest_key: str,
    inference_component_model_name: str,
    metrics: dict,
) -> None:
    lines = [
        f"A new YOLO11-{size} model package has been approved and is now serving live.",
        "",
        f"Model package: {model_package_arn}",
        f"Source weights: {model_data_url}",
        f"Latest weights snapshot: s3://{PIPELINE_BUCKET}/{latest_key}",
        f"Now serving on: track2-yolo-{size} (Inference Component, model: {inference_component_model_name})",
    ]
    if archive_key:
        lines.append(f"Previous weights archived to: s3://{PIPELINE_BUCKET}/{archive_key}")
    else:
        lines.append("No previous weights existed to archive (first approval for this size).")
    if metrics:
        lines.append("")
        lines.append(f"Model metrics: {json.dumps(metrics, indent=2)}")

    sns.publish(
        TopicArn=NOTIFY_TOPIC_ARN,
        Subject=f"Track2 YOLO11-{size}: model approved and deployed ({model_package_arn.split('/')[-1]})",
        Message="\n".join(lines),
    )


def handler(event, context):
    detail = event["detail"]
    model_package_arn = detail["ModelPackageArn"]
    approval_status = detail.get("ModelApprovalStatus")
    group_name = detail.get("ModelPackageGroupName")

    if approval_status != "Approved":
        # Defense-in-depth: the EventBridge rule itself is already filtered
        # to Approved, but don't act on anything else if it ever changes.
        print(f"Ignoring event for {model_package_arn}: status is {approval_status!r}, not Approved.")
        return {"skipped": True, "reason": f"status is {approval_status!r}"}

    size = GROUP_TO_SIZE.get(group_name)
    if size is None:
        print(f"Ignoring event for {model_package_arn}: group {group_name!r} isn't one of {list(GROUP_TO_SIZE)}.")
        return {"skipped": True, "reason": f"unrecognized group {group_name!r}"}

    desc = sm.describe_model_package(ModelPackageName=model_package_arn)
    container = desc["InferenceSpecification"]["Containers"][0]
    model_data_url = container["ModelDataUrl"]
    metrics = desc.get("ModelMetrics", {})

    latest_key = f"latest_approved_model_weights/{size}/model.pt"
    archive_prefix = f"latest_approved_model_weights/{size}/archive/"

    archive_key = _archive_existing_latest_if_present(latest_key, archive_prefix)

    with tempfile.TemporaryDirectory() as download_dir:
        _extract_and_upload_new_weights(model_data_url, latest_key, download_dir)

    inference_component_model_name = _deploy_to_inference_component(size, model_package_arn, container)

    _notify(model_package_arn, size, model_data_url, archive_key, latest_key, inference_component_model_name, metrics)

    return {
        "model_package_arn": model_package_arn,
        "size": size,
        "latest_weights": f"s3://{PIPELINE_BUCKET}/{latest_key}",
        "archived": archive_key,
        "inference_component": f"{PROJECT}-{size}",
    }
