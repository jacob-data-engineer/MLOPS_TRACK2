#!/usr/bin/env bash
# Resolves the correct SageMaker PyTorch training base image for this
# region (via the SageMaker SDK, not a hand-guessed DLC registry account
# ID — those differ per region), builds this Dockerfile on top of it, and
# pushes to the ECR repo Terraform created.

set -euo pipefail

REGION="ap-southeast-1"
ACCOUNT_ID="125192671932"
REPO_NAME="track2-yolo-train"
TAG="latest"

cd "$(dirname "$0")"

# Uses this repo's own venv (../.venv), not a bare `python`/`python3` on
# PATH — the system Python install on this machine is broken.
PY="$(dirname "$0")/../.venv/Scripts/python.exe"

# `| tail -n 1` matters: the SageMaker SDK prints "sagemaker.config INFO
# - ..." lines to stdout alongside the actual return value, which
# corrupts the captured variable (and the docker login that reads it)
# unless only the final line is kept.
TRAIN_BASE=$("$PY" -c "
from sagemaker import image_uris
print(image_uris.retrieve(
    framework='pytorch', region='$REGION', version='2.3',
    py_version='py311', instance_type='ml.g4dn.xlarge', image_scope='training'
))
" | tail -n 1 | tr -d '\r')
SERVE_BASE=$("$PY" -c "
from sagemaker import image_uris
print(image_uris.retrieve(
    framework='pytorch', region='$REGION', version='2.3',
    py_version='py311', instance_type='ml.g4dn.xlarge', image_scope='inference'
))
" | tail -n 1 | tr -d '\r')
echo "Training base: $TRAIN_BASE"
echo "Serving base:  $SERVE_BASE"

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"

# Base images also need a login — their registry account differs from ours
for BASE in "$TRAIN_BASE" "$SERVE_BASE"; do
  BASE_REGISTRY=$(echo "$BASE" | cut -d/ -f1)
  aws ecr get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$BASE_REGISTRY"
done

docker build --build-arg BASE_IMAGE="$TRAIN_BASE" -f Dockerfile -t "$REPO_NAME:train-$TAG" .
docker tag "$REPO_NAME:train-$TAG" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:train-$TAG"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:train-$TAG"

docker build --build-arg BASE_IMAGE="$SERVE_BASE" -f Dockerfile.serve -t "$REPO_NAME:serve-$TAG" .
docker tag "$REPO_NAME:serve-$TAG" "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:serve-$TAG"
docker push "$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:serve-$TAG"

echo "Pushed: $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/$REPO_NAME:{train,serve}-$TAG"
