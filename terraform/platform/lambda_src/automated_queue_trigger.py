"""Fires one Automated Training Queue's configured run when its
EventBridge rule schedule triggers -- review_app.py creates one such
rule per queue (dynamically, via boto3, not Terraform, since queues are
runtime app state) with this Lambda as its target and that queue's own
training parameters as the rule's fixed Input JSON payload.

Deliberately triggers a CodeBuild build (same mechanism as the review
app's own immediate "Start Training" button and ad-hoc Training Queue --
see review_app.py's _start_training_run), not a direct
sagemaker:StartPipelineExecution -- the pipeline definition only picks
up a given generation/size/dataset/hyperparameters at *build* time
(pipeline_definition.py reads TRAIN_* env vars when run_pipeline.py
calls build_pipeline()), so each firing has to rebuild+resubmit the
pipeline with this queue's own parameters, exactly like every other
training trigger path in this app.

Does NOT check the account's one-GPU-slot guard (unlike the review
app's own routes) -- a collision here surfaces the same
SAGEMAKER_RESOURCE_LIMIT failure the review app already knows how to
diagnose (see summary.md), and building real cross-process coordination
between this Lambda and the review app's in-memory state was out of
scope for what was asked.
"""

import json
import os

import boto3

codebuild = boto3.client("codebuild")

RUN_PIPELINE_CODEBUILD_PROJECT = os.environ["RUN_PIPELINE_CODEBUILD_PROJECT"]


def handler(event, context):
    queue_id = event.get("queue_id", "?")
    env_overrides = [
        {"name": "TRAIN_GENERATION", "value": str(event["generation"]), "type": "PLAINTEXT"},
        {"name": "TRAIN_SIZE", "value": str(event["size"]), "type": "PLAINTEXT"},
        {"name": "TRAIN_EPOCHS", "value": str(event["epochs"]), "type": "PLAINTEXT"},
        {"name": "TRAIN_IMG_SIZE", "value": str(event["img_size"]), "type": "PLAINTEXT"},
        {"name": "TRAIN_BATCH", "value": str(event["batch"]), "type": "PLAINTEXT"},
        {"name": "TRAIN_SEED", "value": str(event["seed"]), "type": "PLAINTEXT"},
        {"name": "TRAIN_DATA_S3_URI", "value": event["train_data_s3_uri"], "type": "PLAINTEXT"},
    ]
    # Resolved to an exact S3 object once, when the queue was created/
    # last edited (see review_app.py's _resolve_base_model_s3_uri call
    # site) -- not re-resolved here on every firing, so a queue's base
    # model stays exactly what it was configured with even if the
    # source folder's contents change later.
    if event.get("base_model_s3_uri"):
        env_overrides.append({"name": "BASE_MODEL_S3_URI", "value": event["base_model_s3_uri"], "type": "PLAINTEXT"})
    if event.get("notes"):
        env_overrides.append({"name": "TRAIN_NOTES", "value": event["notes"], "type": "PLAINTEXT"})

    response = codebuild.start_build(
        projectName=RUN_PIPELINE_CODEBUILD_PROJECT,
        environmentVariablesOverride=env_overrides,
    )
    build_id = response["build"]["id"]
    print(f"Automated queue {queue_id} fired -- started CodeBuild build {build_id}")
    return {"build_id": build_id}
