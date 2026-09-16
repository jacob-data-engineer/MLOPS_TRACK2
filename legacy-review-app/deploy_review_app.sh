#!/usr/bin/env bash
# Builds and runs the review-app container on track2-yolo-review-host.
#
# Run from anywhere: ./MLOPS_TRACK2/deploy_review_app.sh
# Reads its settings from the terraform outputs in INFRA_DIR (defaults to
# jurong-port/terraform/infra), so that stack must be applied first.
#
# Uploads Dockerfile + review_app.py + a generated run_container.sh to S3
# first, then has the instance pull all three from there via SSM (its
# IAM role already has read access to this bucket) — more reliable than
# embedding file contents directly in the SSM command text (quoting/
# escaping issues bit the Grafana demo's equivalent script more than
# once). The docker run command specifically now lives in
# run_container.sh, not inlined into the SSM command JSON: it grew a
# MODEL_PACKAGE_GROUPS_JSON value (a JSON object) once per-size Model
# Package Groups were added, and a JSON blob nested inside SSM's own
# manually-escaped JSON command list was too fragile to keep inlining.
#
# Assumes the instance is running — start it first if needed:
#   aws ec2 start-instances --instance-ids <id> --region ap-southeast-1

set -euo pipefail

REGION="ap-southeast-1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# The infra this reads its settings from stays where it is; only the app
# and this script moved here. Override if the terraform lives elsewhere.
INFRA_DIR="${INFRA_DIR:-$SCRIPT_DIR/../../jurong-port/terraform/infra}"

if [ ! -d "$INFRA_DIR" ]; then
  echo "No terraform infra at $INFRA_DIR — set INFRA_DIR to point at it." >&2
  exit 1
fi

INSTANCE_ID=$(cd "$INFRA_DIR" && terraform output -raw review_host_instance_id)
BUCKET=$(cd "$INFRA_DIR" && terraform output -raw pipeline_bucket)
TABLE=$(cd "$INFRA_DIR" && terraform output -raw review_queue_table)
MODEL_GROUPS_JSON=$(cd "$INFRA_DIR" && terraform output -json model_package_group_names | tr -d '\n ')
SAGEMAKER_ROLE_ARN=$(cd "$INFRA_DIR" && terraform output -raw sagemaker_execution_role_arn)
ECR_URI=$(cd "$INFRA_DIR" && terraform output -raw training_ecr_repository_uri)

echo "Instance: $INSTANCE_ID"
echo "Bucket:   $BUCKET"
echo "Table:    $TABLE"

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --query "Reservations[0].Instances[0].State.Name" --output text)
if [ "$STATE" != "running" ]; then
  echo "Instance is '$STATE', not 'running' — start it first:"
  echo "  aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION"
  exit 1
fi

RUN_SCRIPT="$(mktemp)"
trap 'rm -f "$RUN_SCRIPT"' EXIT

cat > "$RUN_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
cd /opt/review-app && docker build -t review-app .
docker rm -f review-app-container 2>/dev/null || true
docker run -d --name review-app-container --restart unless-stopped -p 8090:8090 \\
  -e REVIEW_QUEUE_TABLE='$TABLE' \\
  -e PIPELINE_BUCKET='$BUCKET' \\
  -e AWS_REGION='$REGION' \\
  -e RETRAIN_RULE_NAME='track2-yolo-weekly-retrain-schedule' \\
  -e TRAINING_PIPELINE_NAME='track2-yolo-training-pipeline' \\
  -e MODEL_PACKAGE_GROUPS_JSON='$MODEL_GROUPS_JSON' \\
  -e MULTI_SIZE_ENDPOINT_NAME='track2-yolo-multi-size-endpoint' \\
  -e MULTI_SIZE_VARIANT_NAME='AllSizes' \\
  -e SAGEMAKER_EXECUTION_ROLE_ARN='$SAGEMAKER_ROLE_ARN' \\
  -e SERVING_IMAGE_URI='$ECR_URI:serve-latest' \\
  -e RUN_PIPELINE_CODEBUILD_PROJECT='track2-yolo-run-pipeline' \\
  review-app
sleep 3
docker ps --filter name=review-app-container --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
EOF

echo "Uploading source to s3://$BUCKET/review-app-source/ ..."
aws s3 cp "$SCRIPT_DIR/Dockerfile" "s3://$BUCKET/review-app-source/Dockerfile" --region "$REGION"
aws s3 cp "$SCRIPT_DIR/review_app.py" "s3://$BUCKET/review-app-source/review_app.py" --region "$REGION"
aws s3 cp "$RUN_SCRIPT" "s3://$BUCKET/review-app-source/run_container.sh" --region "$REGION"

echo "Deploying via SSM ..."
COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --parameters "{\"commands\":[\
\"mkdir -p /opt/review-app\",\
\"aws s3 cp s3://$BUCKET/review-app-source/Dockerfile /opt/review-app/Dockerfile --region $REGION\",\
\"aws s3 cp s3://$BUCKET/review-app-source/review_app.py /opt/review-app/review_app.py --region $REGION\",\
\"aws s3 cp s3://$BUCKET/review-app-source/run_container.sh /opt/review-app/run_container.sh --region $REGION\",\
\"chmod +x /opt/review-app/run_container.sh\",\
\"/opt/review-app/run_container.sh\"\
]}" \
  --region "$REGION" \
  --query "Command.CommandId" --output text)

echo "SSM command: $COMMAND_ID — waiting for result ..."
sleep 15

aws ssm get-command-invocation \
  --command-id "$COMMAND_ID" \
  --instance-id "$INSTANCE_ID" \
  --region "$REGION" \
  --query "[Status,StandardOutputContent,StandardErrorContent]" \
  --output text
