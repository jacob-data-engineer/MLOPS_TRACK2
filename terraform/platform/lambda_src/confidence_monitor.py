"""Custom Data Quality monitor for the YOLO endpoint.

AWS's built-in SageMaker Data Quality Model Monitor expects tabular/JSON
feature data, not raw image bytes — it doesn't apply directly to a CV
endpoint's captured requests. This replaces it with a purpose-built
check: read the endpoint's captured input/output records for the last
window, compute average prediction confidence, compare to a stored
baseline, and:
  - publish a CloudWatch metric so the existing alarm (monitoring.tf)
    fires the same way it would for a built-in monitor
  - write a constraint_violations.json in the shape trigger_review.py
    already expects, when confidence has dropped meaningfully

Runs on the EventBridge schedule defined alongside this Lambda.
"""

import json
import os
import statistics
from datetime import datetime, timedelta, timezone

import boto3

s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")

BUCKET = os.environ["PIPELINE_BUCKET"]
DATA_CAPTURE_PREFIX = "data-capture/"
BASELINE_KEY = "monitor/baseline/confidence_baseline.json"
REPORTS_PREFIX = "monitor/reports/"
DRIFT_THRESHOLD = float(os.environ.get("DRIFT_THRESHOLD", "0.15"))
LOOKBACK_MINUTES = int(os.environ.get("LOOKBACK_MINUTES", "60"))


def _load_baseline():
    try:
        obj = s3.get_object(Bucket=BUCKET, Key=BASELINE_KEY)
        return json.loads(obj["Body"].read())["mean_confidence"]
    except s3.exceptions.NoSuchKey:
        return None


def _recent_capture_keys():
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=LOOKBACK_MINUTES)
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=DATA_CAPTURE_PREFIX):
        for obj in page.get("Contents", []):
            if obj["LastModified"] >= cutoff:
                keys.append(obj["Key"])
    return keys


def _extract_confidences_and_examples(keys):
    confidences = []
    examples = []
    for key in keys:
        body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        for line in body.decode().splitlines():
            record = json.loads(line)
            output = json.loads(record["captureData"]["endpointOutput"]["data"])
            for detection in output.get("detections", []):
                confidences.append(detection["confidence"])
                if detection["confidence"] < 0.5:
                    examples.append(
                        {
                            "example_image_s3_uri": record["captureData"]["endpointInput"].get("uri", ""),
                            "predicted_boxes": [detection],
                        }
                    )
    return confidences, examples


def handler(event, context):
    baseline_mean = _load_baseline()
    keys = _recent_capture_keys()
    confidences, flagged_examples = _extract_confidences_and_examples(keys)

    if not confidences:
        return {"status": "no captured inferences in window"}

    current_mean = statistics.mean(confidences)

    cloudwatch.put_metric_data(
        Namespace="Track2Yolo/ModelMonitor",
        MetricData=[
            {"MetricName": "mean_confidence", "Value": current_mean, "Unit": "None"},
        ],
    )

    if baseline_mean is None:
        # first run: this window's mean becomes the baseline going forward
        s3.put_object(
            Bucket=BUCKET,
            Key=BASELINE_KEY,
            Body=json.dumps({"mean_confidence": current_mean}).encode(),
        )
        cloudwatch.put_metric_data(
            Namespace="Track2Yolo/ModelMonitor",
            MetricData=[{"MetricName": "feature_baseline_drift_confidence", "Value": 0, "Unit": "None"}],
        )
        return {"status": "baseline established", "mean_confidence": current_mean}

    drift = baseline_mean - current_mean  # positive = confidence dropped
    cloudwatch.put_metric_data(
        Namespace="Track2Yolo/ModelMonitor",
        MetricData=[
            {"MetricName": "feature_baseline_drift_confidence", "Value": max(drift, 0), "Unit": "None"},
        ],
    )

    if drift > DRIFT_THRESHOLD and flagged_examples:
        report_key = f"{REPORTS_PREFIX}{datetime.now(timezone.utc).isoformat()}_constraint_violations.json"
        s3.put_object(
            Bucket=BUCKET,
            Key=report_key,
            Body=json.dumps({"violations": flagged_examples, "drift": drift}).encode(),
        )
        return {"status": "drift detected", "drift": drift, "report_key": report_key}

    return {"status": "within baseline", "drift": drift}
