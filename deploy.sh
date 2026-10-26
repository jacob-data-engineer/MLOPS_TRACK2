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
# Shown in the UI beside the build tag, so which container you are looking
# at is never a guess. Two may now run side by side -- see the sweep below.
APP_ROLE="${APP_ROLE:-stable}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# This folder's own stack -- nothing here reads the jurong-port state.
# terraform/app is this app's own stack (ECR, CodeBuild, the audit table).
# terraform/platform beside it owns the shared estate -- the instance, the
# roles, the bucket -- and is never what this script reads.
INFRA_DIR="${INFRA_DIR:-$SCRIPT_DIR/terraform/app}"

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

# Setting IMAGE_TAG redeploys an image that is already in ECR instead of
# building a new one. When only the instance-side step failed -- a bad
# pull, a container that wouldn't start -- the image from that run is
# perfectly good, and rebuilding it to retry a `docker pull` costs five
# minutes for nothing.
TAG="${IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
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

if [ -n "${IMAGE_TAG:-}" ]; then
  echo "IMAGE_TAG set -- redeploying an existing image, skipping build."
  if ! aws ecr describe-images --repository-name "${APP_REPO##*/}" \
      --image-ids "imageTag=$TAG" --region "$REGION" >/dev/null 2>&1; then
    echo "No image tagged '$TAG' in ${APP_REPO##*/}. Drop IMAGE_TAG to build one." >&2
    exit 1
  fi
else

# --- 1. ship the source to S3 ------------------------------------------
# Excludes are not just size: .venv and node_modules are host-specific
# build output that would shadow what the image builds for itself, .git
# would put the whole history in a build context for no reason, and
# terraform/.terraform is a ~700MB provider cache with nothing to do with
# the image -- it alone is a hundred times the size of the real source.
#
# model_training/ is excluded for the same reason: it holds the SageMaker
# pipelines, evaluators and baseline checkpoints, which run in AWS, not in
# this container. Left in, it took the build source from 824KB to 13MB and
# put the weights inside the app image too.
ZIP="$(mktemp -d)/track2-app.zip"
echo "Packaging source ..."
if command -v zip >/dev/null 2>&1; then
  ( cd "$SCRIPT_DIR" && zip -q -r "$ZIP" . \
      -x '*.venv/*' 'node_modules/*' 'web/node_modules/*' 'web/dist/*' \
         '*__pycache__/*' '*.pytest_cache/*' 'legacy-review-app/*' '*.pyc' \
         '.git/*' 'terraform/.terraform/*' 'terraform/*.tfstate*' 'model_training/*' 'platform/*' )
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
  # Both spellings of every directory exclude: bsdtar matches these
  # patterns against the whole member path, so '*/.venv/*' requires a
  # leading directory and silently misses a top-level '.venv/...'. A venv
  # at the repo root is the normal case (uv puts it there), and without
  # the bare pattern it went into the build source -- 351MB instead of
  # 824KB, on every deploy, for a directory the image never uses.
  ( cd "$SCRIPT_DIR" && shopt -s dotglob && "$BSDTAR" -c -a -f "$(cygpath -w "$ZIP")" \
      --exclude '.venv/*' --exclude 'node_modules/*' \
      --exclude '__pycache__/*' --exclude '.pytest_cache/*' \
      --exclude '*/.venv/*' --exclude '*/node_modules/*' \
      --exclude 'web/dist/*' --exclude '*/__pycache__/*' \
      --exclude '*/.pytest_cache/*' --exclude 'legacy-review-app/*' \
      --exclude '*.pyc' --exclude '.git/*' \
      --exclude '*/.terraform/*' --exclude '*.tfstate' \
      --exclude '*.tfstate.backup' --exclude 'model_training/*' \
      --exclude 'platform/*' \
      -- * )
fi
echo "  $(du -h "$ZIP" | cut -f1)"
# Keyed by container, not a single app.zip. Two deploys running at once
# -- development and production, now that both exist -- otherwise share
# one source object: the second upload replaces the first, and whichever
# build fetches later compiles the other one's code and ships it under
# its own tag. Nothing about that looks wrong afterwards.
SOURCE_KEY="$SOURCE_PREFIX/app-$CONTAINER.zip"
aws s3 cp "$ZIP" "s3://$BUCKET/$SOURCE_KEY" --region "$REGION"
rm -rf "$(dirname "$ZIP")"

# --- 2. build it in AWS -------------------------------------------------
echo "Starting CodeBuild ($BUILD_PROJECT) ..."
# The project's own source is still .../app.zip; this points the build at
# this deploy's copy instead, so the project does not have to be edited
# and two builds cannot read the same object.
BUILD_ID=$(aws codebuild start-build \
  --project-name "$BUILD_PROJECT" \
  --source-type-override S3 \
  --source-location-override "$BUCKET/$SOURCE_KEY" \
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
fi

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
  # The "|| true" is load-bearing under set -euo pipefail: a name in the
  # list that isn't running makes docker inspect fail, pipefail fails the
  # whole pipeline for it, and a failed command substitution in an
  # assignment aborts the script -- so without it the deploy dies on the
  # first name that misses instead of trying the next.
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

# REPLACE whatever occupies THIS PORT, and leave anything on another port
# alone. Two containers may now run side by side -- a stable one and a
# development one -- which is why this is no longer a sweep of every
# container from this repo.
#
# The port is the right thing to key on. The failure that made this a
# sweep in the first place was a deploy landing on a new container while
# the old one kept serving the URL people actually had: a stale build
# nobody could see was stale. Two containers cannot share a port, so
# clearing this port guarantees the URL being deployed to is the build
# just deployed, while a container on a different port is left running
# on purpose.
#
# Scoped to this repo, so nothing else on the box is touched, and the
# running image tag is now shown in the UI -- which is the part that
# makes two containers safe rather than confusing.
for cid in \$(docker ps -a --format '{{.ID}} {{.Image}} {{.Ports}} {{.Names}}' \
              | grep -F '$APP_REPO' \
              | grep -E ":$PORT->|[[:space:]]$CONTAINER\$" \
              | awk '{print \$1}' | sort -u); do
  echo "removing app container \$cid (holds port $PORT or the name $CONTAINER)"
  docker rm -f "\$cid" >/dev/null 2>&1 || true
done
docker rm -f $CONTAINER 2>/dev/null || true
# Settings live on the HOST, not in the container, so they survive a
# deploy replacing it. Per container name, so development cannot overwrite
# production's saved settings by saving under the same name.
mkdir -p /opt/track2-app/settings/$CONTAINER
docker run -d --name $CONTAINER --restart unless-stopped -p $PORT:8501 \\
  -v /opt/track2-app/settings/$CONTAINER:/settings \\
  -e PPE_APP_PASSWORD="\$APP_PASSWORD" \\
  -e APP_BUILD_TAG='$TAG' \\
  -e APP_ROLE='$APP_ROLE' \\
  -e APP_PORT='$PORT' \\
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

# Also per container, and for a worse reason than the zip: this script
# carries the container NAME, the PORT and the role. Overwritten by a
# concurrent deploy between upload and fetch, a production deploy would
# bring the container up on development's port under development's name.
# It is uploaded on the promotion path too (IMAGE_TAG skips the build but
# not this), so the production route is exactly the one at risk.
RUN_KEY="$SOURCE_PREFIX/run_container-$CONTAINER.sh"
aws s3 cp "$RUN_SCRIPT" "s3://$BUCKET/$RUN_KEY" --region "$REGION" >/dev/null

echo "Deploying to the instance ..."
COMMAND_ID=$(aws ssm send-command \
  --instance-ids "$INSTANCE_ID" \
  --document-name AWS-RunShellScript \
  --comment "Deploy track2 app $TAG" \
  --parameters "{\"commands\":[\
\"mkdir -p /opt/track2-app\",\
\"aws s3 cp s3://$BUCKET/$RUN_KEY /opt/track2-app/run_container-$CONTAINER.sh --region $REGION\",\
\"chmod +x /opt/track2-app/run_container-$CONTAINER.sh\",\
\"/opt/track2-app/run_container-$CONTAINER.sh\"\
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

# --- 4. the Lambda runs the SAME source file -----------------------------
#
# terraform/app zips src/ppe_pipeline_api/pull_control.py straight into the
# pull-control router, so the container and the Lambda share one file and
# one routing rule. But a container deploy does not touch Lambda, and the
# Lambda only changes on a targeted apply -- so they drift in silence.
#
# That happened: the router spent a day filing into the old flat layout
# while the app had moved to images/train, and nothing reported a problem
# because each half was self-consistent.
#
# `source_code_hash` makes this a no-op when the file has not changed, so
# it costs nothing on a frontend-only deploy.
#
# NOTE: there is ONE router, shared by both containers. A development
# deploy therefore updates the function that sweeps unattended against
# real data -- the same trade already accepted for the shared tables.
if [ "${SKIP_LAMBDA:-}" = "1" ]; then
  echo "SKIP_LAMBDA=1 -- leaving the router on its current code."
else
  echo
  # The queue runner's zip is assembled here, not by Terraform: an
  # archive_file can take one source_dir, and this needs two -- the
  # package, and PyYAML, which the advance needs because validating a
  # dataset reads its data.yaml.
  #
  # Only the `yaml/` package is copied, never the `_yaml` C extension
  # beside it: that is built for THIS machine, and PyYAML falls back to
  # its pure-Python parser when the extension will not import. Copying a
  # Windows .pyd into a Linux Lambda is how that fallback stops being
  # theoretical.
  QR_BUILD="$INFRA_DIR/.build/queue_runner"
  rm -rf "$QR_BUILD"
  mkdir -p "$QR_BUILD"
  cp -r "$SCRIPT_DIR/src/ppe_pipeline_api" "$QR_BUILD/ppe_pipeline_api"
  find "$QR_BUILD" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  if [ -d "$SCRIPT_DIR/.venv/Lib/site-packages/yaml" ]; then
    cp -r "$SCRIPT_DIR/.venv/Lib/site-packages/yaml" "$QR_BUILD/yaml"
  elif [ -d "$SCRIPT_DIR/.venv/lib/python3.12/site-packages/yaml" ]; then
    cp -r "$SCRIPT_DIR/.venv/lib/python3.12/site-packages/yaml" "$QR_BUILD/yaml"
  else
    echo >&2 "WARNING: PyYAML not found in .venv -- the queue runner would fail"
    echo >&2 "  at dataset validation. Install it and redeploy."
  fi

  echo "Updating the pull-control router (shares pull_control.py) ..."
  if (cd "$INFRA_DIR" && terraform apply -auto-approve -input=false         -target=aws_lambda_function.pull_control_router         -target=aws_lambda_function.queue_runner >/dev/null 2>&1); then
    LAMBDA_AT=$(aws lambda get-function-configuration       --function-name track2-app-pull-control-router       --region "$REGION" --query LastModified --output text 2>/dev/null || echo unknown)
    echo "  router code as of $LAMBDA_AT"
  else
    # Loud, and after the container is already up, so the state is
    # unambiguous rather than half-reported.
    echo >&2
    echo "WARNING: the container deployed but the router was NOT updated." >&2
    echo "  Unattended sweeps are still running the previous code, which may" >&2
    echo "  file into a different layout than this build expects." >&2
    echo "  Most likely a Terraform state lock from a concurrent deploy." >&2
    echo "  Re-run:  cd $INFRA_DIR && terraform apply \\" >&2
    echo "             -target=aws_lambda_function.pull_control_router" >&2
    exit 1
  fi
fi

echo
echo "Deployed $TAG -> port $PORT"
