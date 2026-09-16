"""Model Monitor -> human review glue.

Triggered by the CloudWatch alarm on the Data Quality monitor's drift
metric (see monitoring.tf). Reads that execution's constraint_violations
report from S3, and for each violated example writes one row to the
review-queue DynamoDB table + sends an SNS notification.

Originally called Amazon A2I's StartHumanLoop here. Both A2I and
SageMaker private-workforce management are closed to new AWS customers
account-wide (confirmed via a real CreateWorkforce/CreateHumanTaskUi 400
on this account, not fixable by retrying) — see sagemaker.tf's note.
review_queue.tf's DynamoDB table + review_app.py (deployed to the EC2
instance) are the replacement review surface.
"""

import json
import os
import uuid
from datetime import datetime, timezone

import boto3

s3 = boto3.client("s3")
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")

BUCKET = os.environ["PIPELINE_BUCKET"]
REVIEW_QUEUE_TABLE = os.environ["REVIEW_QUEUE_TABLE"]
NOTIFY_TOPIC_ARN = os.environ["NOTIFY_TOPIC_ARN"]
REVIEW_APP_URL = os.environ["REVIEW_APP_URL"]
REPORTS_PREFIX = "monitor/reports/"

table = dynamodb.Table(REVIEW_QUEUE_TABLE)


def _latest_violations_key():
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=REPORTS_PREFIX)
    candidates = [
        obj["Key"]
        for obj in resp.get("Contents", [])
        if obj["Key"].endswith("constraint_violations.json")
    ]
    if not candidates:
        return None
    return sorted(candidates)[-1]


def handler(event, context):
    key = _latest_violations_key()
    if key is None:
        return {"queued": 0, "reason": "no violations report found"}

    body = json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    violations = body.get("violations", [])
    if not violations:
        return {"queued": 0, "reason": "report present, no violations"}

    queued = []
    now = datetime.now(timezone.utc).isoformat()
    for violation in violations:
        review_id = uuid.uuid4().hex[:12]
        table.put_item(
            Item={
                "review_id": review_id,
                "status": "pending",
                "image_s3_uri": violation.get("example_image_s3_uri", ""),
                "predicted_boxes": json.dumps(violation.get("predicted_boxes", [])),
                "drift": str(body.get("drift", "")),
                "created_at": now,
            }
        )
        queued.append(review_id)

    sns.publish(
        TopicArn=NOTIFY_TOPIC_ARN,
        Subject="Track2 YOLO: model results flagged for review",
        Message=(
            f"{len(queued)} flagged prediction(s) need review "
            f"(drift={body.get('drift', 'n/a')}).\n\n"
            f"Review them at: {REVIEW_APP_URL}"
        ),
    )

    return {"queued": len(queued), "review_ids": queued}
