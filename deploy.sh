#!/usr/bin/env bash
# Ships this folder to the review host. Run from your machine:
#
#   ./MLOPS_TRACK2/deploy.sh
#
# Your machine only zips the source and waits. The image is built in AWS
# by CodeBuild (this folder's own terraform/) and
# pushed to ECR; the instance is then told over SSM to pull and restart.
# Nothing is built locally -- no Docker, no node, no multi-GB layers on
# your disk -- and there is deliberately no GitHub Actions deploy.
#
# One-time setup, because the build installs aiops-pipeline from the
# private SNAIC-JP/Track2 repo:
#
#   aws ssm put-parameter --name /track2-yolo/github-token \
#     --type SecureString --value ghp_xxx --region ap-southeast-1

set -euo pipefail

REGION="ap-southeast-1"
PORT="${PORT:-8090}"
CONTAINER="${CONTAINER:-track2-app}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# This folder's own stack -- nothing here reads the jurong-port state.
INFRA_DIR="${INFRA_DIR:-$SCRIPT_DIR/terraform}"

if [ ! -d "$INFRA_DIR" ]; then
  echo "No terraform at $INFRA_DIR." >&2
  exit 1
fi
if [ ! -f "$INFRA_DIR/terraform.tfstate" ] && [ ! -d "$INFRA_DIR/.terraform" ]; then
  echo "Build infrastructure not created yet. Run it once:" >&2
  echo "  cd $INFRA_DIR && terraform init && terraform apply" >&2
  exit 1
fi

echo "Reading settings from terraform ..."
INSTANCE_ID=$(cd "$INFRA_DIR" && terraform output -raw review_host_instance_id)
BUCKET=$(cd "$INFRA_DIR" && terraform output -raw source_bucket)
SOURCE_PREFIX=$(cd "$INFRA_DIR" && terraform output -raw source_prefix)
APP_REPO=$(cd "$INFRA_DIR" && terraform output -raw ecr_repository_uri)
BUILD_PROJECT=$(cd "$INFRA_DIR" && terraform output -raw codebuild_project)

if [ -z "$INSTANCE_ID" ]; then
  echo "No instance tagged Name=track2-yolo-review-host found." >&2
  exit 1
fi

# Runtime settings the container needs. Derived here rather than read from
# the other stack's outputs, so this deploy does not require that stack's
# state to be present on the machine running it.
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
SAGEMAKER_ROLE_ARN="arn:aws:iam::${ACCOUNT_ID}:role/track2-yolo-sagemaker-execution"
SERVING_IMAGE="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/track2-yolo-train"

TAG="$(date -u +%Y%m%d-%H%M%S)"
IMAGE="${APP_REPO}:${TAG}"
REGISTRY="${APP_REPO%%/*}"

echo "Instance: $INSTANCE_ID"
echo "Image:    $IMAGE"

STATE=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" --region "$REGION" \
  --query "Reservations[0].Instances[0].State.Name" --output text)
if [ "$STATE" != "running" ]; then
  echo "Instance is '$STATE', not 'running'. Start it first:" >&2
  echo "  aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION" >&2
  exit 1
fi

# --- 1. ship the source to S3 ------------------------------------------
# Excludes are not just size: .venv and node_modules are host-specific
# build output that would shadow what the image builds for itself, .git
# would put the whole history in a build context for no reason, and
# terraform/.terraform is a ~700MB provider cache with nothing to do with
# the image -- it alone is a hundred times the size of the real source.
ZIP="$(mktemp -d)/track2-app.zip"
echo "Packaging source ..."
if command -v zip >/dev/null 2>&1; then
  ( cd "$SCRIPT_DIR" && zip -q -r "$ZIP" . \
      -x '*.venv/*' 'node_modules/*' 'web/node_modules/*' 'web/dist/*' \
         '*__pycache__/*' '*.pytest_cache/*' 'legacy-review-app/*' '*.pyc' \
         '.git/*' 'terraform/.terraform/*' 'terraform/*.tfstate*' )
else
  # Git Bash on Windows ships no `zip`, so this would otherwise only run
  # from WSL/mac. The system bsdtar writes real zip archives. It is a
  # Windows binary, so the archive path has to be Windows-shaped.
  #
  # The members are globbed (with dotglob, so .dockerignore is not missed)
  # rather than given as ".": CodeBuild looks for the Dockerfile at the
  # archive root, and this build of bsdtar has no -s to strip the "./"
  # that archiving "." would prefix every entry with.
  BSDTAR="${BSDTAR:-/c/Windows/System32/tar.exe}"
  if [ ! -x "$BSDTAR" ]; then
    echo "Need either 'zip' or bsdtar (tar.exe) to package the source." >&2
    exit 1
  fi
  ( cd "$SCRIPT_DIR" && shopt -s dotglob && "$BSDTAR" -c -a -f "$(cygpath -w "$ZIP")" \
      --exclude '*/.venv/*' --exclude '*/node_modules/*' \
      --exclude 'web/dist/*' --exclude '*/__pycache__/*' \
      --exclude '*/.pytest_cache/*' --exclude 'legacy-review-app/*' \
      --exclude '*.pyc' --exclude '.git/*' \
      --exclude '*/.terraform/*' --exclude '*.tfstate' \
      --exclude '*.tfstate.backup' \
      -- * )
fi
echo "  $(du -h "$ZIP" | cut -f1)"
aws s3 cp "$ZIP" "s3://$BUCKET/$SOURCE_PREFIX/app.zip" --region "$REGION"
rm -rf "$(dirname "$ZIP")"

# --- 2. build it in AWS -------------------------------------------------
echo "Starting CodeBuild ($BUILD_PROJECT) ..."
BUILD_ID=$(aws codebuild start-build \
  --project-name "$BUILD_PROJECT" \
  --environment-variables-override "name=IMAGE_TAG,value=$TAG,type=PLAINTEXT" \
  --region "$REGION" --query "build.id" --output text)
echo "  $BUILD_ID"

while :; do
  PHASE=$(aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query "builds[0].buildStatus" --output text)
  [ "$PHASE" != "IN_PROGRESS" ] && break
  sleep 15
done

if [ "$PHASE" != "SUCCEEDED" ]; then
  echo "Build $PHASE. Logs:" >&2
  aws codebuild batch-get-builds --ids "$BUILD_ID" --region "$REGION" \
    --query "builds[0].logs.deepLink" --output text >&2
  exit 1
fi
echo "Build succeeded."

# --- 3. run it on the instance -----------------------------------------
RUN_SCRIPT="$(mktemp)"
trap 'rm -f "$RUN_SCRIPT"' EXIT
cat > "$RUN_SCRIPT" <<EOF
#!/bin/bash
set -euo pipefail
# Prune before pulling: this image is multi-GB and the host has filled its
# disk on unpinned layers before. Runs before the stop/rm below, so the
# image of the still-running container is in use and protected.
docker image prune -a -f
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $REGISTRY
docker pull $IMAGE

# Reuse the password the container already running here has, so a deploy
# doesn't silently change how people log in. Read on the instance, not by
# the deploy script: the secret then never passes through the operator's
# terminal, the SSM payload, or this script's copy in S3. Runs before the
# rm below, while the old container is still there to read it from.
APP_PASSWORD="${PPE_APP_PASSWORD:-}"
if [ -z "\$APP_PASSWORD" ]; then
  # Both "|| true"s matter under set -euo pipefail: a container in the
  # list that doesn't exist makes docker inspect fail, and pipefail would
  # otherwise abort the whole deploy on the first name that misses rather
  # than moving on to the next.
  for c in $CONTAINER track2-app ppe-pipeline-track2-test; do
    APP_PASSWORD=\$(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "\$c" 2>/dev/null \\
      | sed -n 's/^PPE_APP_PASSWORD=//p' | head -1 || true)
    if [ -n "\$APP_PASSWORD" ]; then break; fi
  done
fi
if [ -z "\$APP_PASSWORD" ]; then
  echo "No container on this host carries PPE_APP_PASSWORD and none was" >&2
  echo "exported -- refusing to start the app with no password." >&2
  exit 1
fi

docker rm -f $CONTAINER 2>/dev/null || true
docker run -d --name $CONTAINER --restart unless-stopped -p $PORT:8501 \\
  -e PPE_APP_PASSWORD="\$APP_PASSWORD" \\
  -e AWS_REGION='$REGION' \\
  -e REVIEW_QUEUE_TABLE='track2-yolo-review-queue' \\
  -e PIPELINE_BUCKET='$BUCKET' \\
  -e RETRAIN_RULE_NAME='track2-yolo-weekly-retrain-schedule' \\
  -e TRAINING_PIPELINE_NAME='track2-yolo-training-pipeline' \\
  -e MULTI_SIZE_ENDPOINT_NAME='track2-yolo-multi-size-endpoint' \\
  -e MULTI_SIZE_VARIANT_NAME='AllSizes' \\
  -e SAGEMAKER_EXECUTION_ROLE_ARN='$SAGEMAKER_ROLE_ARN' \\
  -e SERVING_IMAGE_URI='$SERVING_IMAGE:serve-latest' \\
  -e RUN_PIPELINE_CODEBUILD_PROJECT='track2-yolo-run-pipeline' \\
  $IMAGE
# Poll rather than one fixed sleep: the app takes longer than any single
# wait to start serving, and a single early check fails a deploy that
# actually worked.
for i in \$(seq 1 40); do
  curl -sf http://localhost:$PORT/health >/dev/null && break
  sleep 3
done
curl -sf http://localhost:$PORT/health
echo
docker ps --filter name=$CONTAINER --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
EOF

aws s3 cp "$RUN_SCRIPT" "s3://$BUCKET/$SOURCE_PREFIX/run_container.sh" --region "$REGION" >/dev/null

echo "Deploying to the instance ..."
COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment "Deploy track2 app $TAG" \
  --parameters "{\"commands\":[\
\"mkdir -p /opt/track2-app\",\
\"aws s3 cp s3://$BUCKET/$SOURCE_PREFIX/run_container.sh /opt/track2-app/run_container.sh --region $REGION\",\
\"chmod +x /opt/track2-app/run_container.sh\",\
\"/opt/track2-app/run_container.sh\"\
]}" \
  --region "$REGION" --query "Command.CommandId" --output text)

# send-command returns when the command is ACCEPTED, not when it finishes.
# Reporting success here would hide a deploy that failed on the instance.
while :; do
  STATUS=$(aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" \
    --region "$REGION" --query Status --output text 2>/dev/null || echo Pending)
  case "$STATUS" in Success|Failed|Cancelled|TimedOut) break ;; esac
  sleep 10
done

aws ssm get-command-invocation --command-id "$COMMAND_ID" --instance-id "$INSTANCE_ID" \
  --region "$REGION" --query "{Status:Status,Out:StandardOutputContent,Err:StandardErrorContent}" --output text

if [ "$STATUS" != "Success" ]; then
  echo "Deploy FAILED on the instance (SSM status: $STATUS)." >&2
  echo "The image is in ECR; the running container was not replaced." >&2
  exit 1
fi

echo
echo "Deployed $TAG -> port $PORT"
