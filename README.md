# MLOPS_TRACK2

The Track2 YOLO MLOps app: a FastAPI backend and a React SPA served
together on one port, plus the training-pipeline assets behind them.

Everything needed to build, deploy and change this app lives in this
folder. It used to sit inside the JP_UI repo and no longer refers to it.

```
src/ppe_pipeline_api/   FastAPI app -- routers/ per page, track2_shared.py for the logic
web/                    React + Vite SPA, built into web/dist and served by the same process
terraform/app/          this app's stack: ECR repo, CodeBuild project, audit table, IAM
terraform/platform/     the shared estate: EC2, roles, bucket, lambdas, schedules, SNS
model_training/         the SageMaker training pipelines, evaluators and dataset preparers
platform/               the estate's own scripts, docs and architecture diagrams
deploy.sh               laptop -> S3 -> CodeBuild -> ECR -> SSM -> EC2
tests/                  pytest
```

**Two Terraform stacks, separate on purpose.** `terraform/app` owns what
belongs to this app and reads the rest through data sources, so applying
it can never disturb the instance or the roles. `terraform/platform` owns
those. `deploy.sh` reads `terraform/app` and must never read the other.

`terraform/platform` refers to its build sources as
`${path.module}/../../model_training/...`, which resolves to this repo's
own `model_training/`. That is why the stack sits two levels down.

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

## Applying the platform stack

`terraform/platform` carries pre-existing drift: a plan there reports
around 15 in-place changes that nobody made today. Most are Lambda code
being re-uploaded from local zips, because that state was rebuilt by
import and never recorded a source hash.

**Prefer a targeted apply there.** A blanket one pushes every local zip
over whatever is deployed:

```powershell
terraform apply -target=aws_lambda_function.weekly_retrain
```

The one case where that is exactly what you want is after editing
`model_training/replay_lambda/` -- its `archive_file` packages that
directory, and applying is how the coordinator gets deployed.

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
