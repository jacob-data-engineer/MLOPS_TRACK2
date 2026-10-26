"""Emails stakeholders when a track2-yolo-training-pipeline execution
starts (Executing) and when it reaches a terminal state (Succeeded/
Failed/Stopped). Triggered by SageMaker's own built-in "SageMaker Model
Building Pipeline Execution Status Change" EventBridge event -- no
polling.

Separate from dump_approved_weights.py: that one fires on model-package
*approval* (a human decision, made after review); this one fires on
*training finishing* (an automatic outcome), regardless of whether anyone
has looked at the result yet. Covers Failed/Stopped too, not just
Succeeded, so a training failure doesn't go silently unnoticed -- only
the subject line and body differ by outcome.

On a Succeeded run, best-effort looks up the most recently registered
model package and includes its ARN/status in the email, so the
notification doubles as "here's the new candidate waiting for you to
review" without needing a second EventBridge rule on model-package
creation. Checks across all four YOLO11 size groups (MODEL_PACKAGE_GROUPS)
and reports whichever has the newest package, rather than one fixed
group -- this one shared pipeline definition can register into any of
the four sizes depending on TRAIN_SIZE at build time (see
pipeline_definition.py), so which group actually got the new candidate
isn't knowable ahead of time from this event alone. Wrapped in
try/except: a failure here must never break the notification itself.
"""

import json
import os
from datetime import datetime

import boto3

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
NOTIFY_TOPIC_ARN = os.environ["NOTIFY_TOPIC_ARN"]
MODEL_PACKAGE_GROUPS = json.loads(os.environ.get("MODEL_PACKAGE_GROUPS_JSON", '["track2-yolo11"]'))

sns = boto3.client("sns", region_name=REGION)
sm = boto3.client("sagemaker", region_name=REGION)

# What each status is, for subscribers who chose kinds on the
# Notifications page (an SNS filter policy on their subscription; see
# src/ppe_pipeline_api/notify.py in the app). A failure is both an end
# and an error, so it goes to whoever ticked either.
EVENTS = {
    "Executing": ["training_started"],
    "Succeeded": ["training_ended"],
    "Stopped": ["training_ended"],
    "Failed": ["training_ended", "error"],
}

# The app's own pipeline. Its runs register into per-Model-Name groups,
# not the four YOLO11 size groups, so the "newest candidate" lookup below
# would name an unrelated package -- it is skipped for these.
OPEN_PIPELINE = "track2-yolo-open-training-pipeline"

SUBJECTS = {
    "Executing": "Training Started",
    "Succeeded": "Training Completed",
    "Failed": "Training FAILED",
    "Stopped": "Training Stopped",
}


def _duration(start_iso: str, end_iso: str) -> str:
    try:
        start = datetime.fromisoformat(start_iso)
        end = datetime.fromisoformat(end_iso)
        seconds = int((end - start).total_seconds())
        minutes, secs = divmod(seconds, 60)
        return f"{minutes}m {secs}s"
    except Exception:
        return "unknown"


def _latest_candidate_summary() -> str:
    try:
        newest = None
        for group in MODEL_PACKAGE_GROUPS:
            packages = sm.list_model_packages(
                ModelPackageGroupName=group,
                SortBy="CreationTime",
                SortOrder="Descending",
                MaxResults=1,
            )["ModelPackageSummaryList"]
            if packages and (newest is None or packages[0]["CreationTime"] > newest["CreationTime"]):
                newest = packages[0]

        if newest is None:
            return ""
        return (
            f"\nNewly registered candidate:\n"
            f"  {newest['ModelPackageArn']}\n"
            f"  Status: {newest['ModelApprovalStatus']}\n"
        )
    except Exception as exc:
        print(f"Could not look up latest model package, continuing anyway: {exc}")
        return ""


def handler(event, context):
    detail = event["detail"]
    status = detail["currentPipelineExecutionStatus"]
    execution_arn = detail["pipelineExecutionArn"]
    display_name = detail.get("pipelineExecutionDisplayName", "")
    start_time = detail["executionStartTime"]
    end_time = detail.get("executionEndTime")

    pipeline = detail.get("pipelineArn", "").rsplit("/", 1)[-1] or "training pipeline"
    subject = SUBJECTS.get(status, f"Training pipeline: {status}")
    if display_name:
        subject = f"{subject}: {display_name}"

    message = (
        f"{pipeline} execution {status}\n\n"
        f"Execution: {execution_arn}\n"
        f"Display name: {display_name}\n"
        f"Started: {start_time}\n"
    )

    if status == "Executing":
        message += "Still running -- you'll get a separate email when it finishes.\n"
    else:
        message += f"Ended: {end_time}\nDuration: {_duration(start_time, end_time)}\n"

    if status == "Succeeded" and pipeline != OPEN_PIPELINE:
        message += _latest_candidate_summary()

    events = EVENTS.get(status, [])
    attributes = ({"event": {"DataType": "String.Array",
                             "StringValue": json.dumps(sorted(events))}}
                  if events else {})
    sns.publish(TopicArn=NOTIFY_TOPIC_ARN, Subject=subject[:100], Message=message,
                MessageAttributes=attributes)
    print(f"Published '{subject}' notification for {execution_arn}")

    return {"status": status, "execution": execution_arn}
