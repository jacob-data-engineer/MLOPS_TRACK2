#!/usr/bin/env bash
# Triggers the training pipeline with deliberately different
# hyperparameters than the fixed defaults (brainstorming.md Gap #4) —
# each call parks a genuinely different candidate to compare, not just
# a re-run of identical config.
#
# Usage:
#   bash trigger_training_variant.sh [epochs] [size] [img_size]
#   bash trigger_training_variant.sh 40 m 640
#
# size selects which YOLO11 size's Model Package Group this run trains
# into (s/m/l/x -- see config.py's model_package_group_name); the base
# checkpoint is always resolved from that group's latest Approved
# package, never a bare Ultralytics name (pipeline_definition.py raises
# if the group has no Approved package yet -- register/approve a
# starting point for a size before triggering training for it).
#
# Any arg can be omitted to use pipeline_definition.py's own default for
# that value (20 / s / 640).

set -euo pipefail

EPOCHS="${1:-20}"
SIZE="${2:-s}"
IMG_SIZE="${3:-640}"
REGION="ap-southeast-1"

echo "Triggering with epochs=$EPOCHS size=$SIZE img_size=$IMG_SIZE"

aws codebuild start-build \
  --project-name track2-yolo-run-pipeline \
  --environment-variables-override \
    "name=TRAIN_EPOCHS,value=$EPOCHS,type=PLAINTEXT" \
    "name=TRAIN_SIZE,value=$SIZE,type=PLAINTEXT" \
    "name=TRAIN_IMG_SIZE,value=$IMG_SIZE,type=PLAINTEXT" \
  --region "$REGION" \
  --query "build.{Id:id,Status:buildStatus}" --output json
