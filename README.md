# MLOPS_TRACK2

The Track2 YOLO MLOps app: a FastAPI backend and a React SPA served
together on one port, plus the training-pipeline assets behind them.

Everything needed to build, deploy and change this app lives in this
folder. It used to sit inside the JP_UI repo and no longer refers to it.

```
src/ppe_pipeline_api/   FastAPI app -- routers/ per page, track2_shared.py for the logic
web/                    React + Vite SPA, built into web/dist and served by the same process
terraform/              this app's own stack: ECR repo, CodeBuild project, audit table, IAM
model_training/         the SageMaker training pipelines, evaluators and dataset preparers
deploy.sh               laptop -> S3 -> CodeBuild -> ECR -> SSM -> EC2
tests/                  pytest
```

## Deploying

Nothing is built locally -- no Docker, no node, no multi-GB layers on your
disk. The script zips the source, CodeBuild builds the image in AWS, and
the instance is told over SSM to pull and restart.

```bash
PORT=8501 CONTAINER=ppe-pipeline-track2-test bash deploy.sh
```

`IMAGE_TAG=<existing tag>` redeploys an image already in ECR and skips the
build -- what to use when only the instance-side step failed.

There is deliberately no GitHub Actions deploy.

Live at `http://<instance public IP>:8501/`. The password is read off the
running container at deploy time, so a deploy never silently changes how
people log in, and the secret never leaves the instance.

## The two training pipelines

| | production replay | custom |
|---|---|---|
| Pipeline | `track2-yolo-training-pipeline` | `track2-yolo-custom-training-pipeline` |
| Registry group | `track2-ppe-fullframe-replay` | `track2-yolo-custom-detector` |
| Classes | fixed: `person`, `helmet` | whatever the dataset's `data.yaml` declares |
| Dataset | pinned pull snapshot, six classes remapped to two | a folder you pick |
| Started by | CodeBuild -> the replay coordinator Lambda | the app, directly |

They are separate on purpose. The production path **discards** every box
outside person/helmet, its evaluator refuses any checkpoint whose class
names differ, and its registry group's consumers read index 0 as `person`.
A model with a different taxonomy in that chain would be mislabelled at
every step without failing anywhere. See
`model_training/custom_training/README.md`.

Epochs, image size, batch and seed are tunable on both. Every pipeline
parameter defaults to the value that used to be hardcoded, so a run that
specifies none of them behaves exactly as it always did -- which is what
keeps the weekly schedule unaffected.

## What is NOT in this folder

The shared platform still lives in the JP_UI repo under
`jurong-port/terraform/infra`: the EC2 instance this app runs on, the
`track2-yolo-review-host` IAM role, the S3 bucket, the retraining
coordinator Lambda and its schedule, the SNS topics.

This app's stack reads those through data sources rather than owning them,
so `terraform apply` here can never disturb them. Moving that stack as
well is a separate operation: it manages resources well beyond this app,
and its state would have to move with it.

Two things there still need an apply from that repo when changed -- the
retraining coordinator (mirrored at `model_training/replay_lambda/`) and
the training pipeline's CodeBuild source zip.

## Conventions worth knowing

- **Line endings are LF**, enforced by `.gitattributes`. A CRLF checkout
  breaks `deploy.sh` under bash and makes Terraform report a permanent
  phantom diff on the CodeBuild buildspec.
- **Terraform state is local** (`terraform/terraform.tfstate`) and not in
  git. It is currently the only copy; an S3 backend is the right fix and
  has not been done.
- The app runs **single-worker with in-memory queues**. Anything that must
  survive a restart is in DynamoDB, and jobs left mid-flight are recovered
  at startup -- see `recover_interrupted_jobs`. Skipping that is what once
  left an endpoint billing for a day.
