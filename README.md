# MLOPS_TRACK2 — the review app on port 8090

The application served at `http://<review-host>:8090`, e.g.

```
http://47.129.112.197:8090/?tab=models&browse_prefix=...&selected_dataset=...
```

It is **not** the JP_UI app. Both run on the same EC2 instance and talk to
the same AWS resources, which is what makes them easy to confuse:

| | This app (8090) | JP_UI (8501) |
|---|---|---|
| Code | `review_app.py`, one file | `src/` + `web/`, FastAPI + React |
| Server | Python stdlib `http.server` | FastAPI / uvicorn |
| Dependencies | `boto3`, nothing else | full `pyproject.toml` |
| Image | `review-app`, built on the instance | `ppe-pipeline`, built in CI, pushed to ECR |
| Container | `review-app-container` | `ppe-pipeline-track2-test` |
| Deployed by | `deploy_review_app.sh`, by hand | GitHub Actions on push to `track2_ui` |
| State in the URL | query string (`?tab=models&...`) | client-side routes (`/track2/models`) |

Query-string navigation is the quickest way to tell them apart from a link
alone: a `?tab=` URL is this app.

## What's here

- `review_app.py` — the whole application. Renders HTML server-side and
  calls AWS directly; no build step, no framework.
- `Dockerfile` — `python:3.12-slim` plus `boto3`.
- `deploy_review_app.sh` — the deployment.

## Where it deploys

| | |
|---|---|
| Instance | `i-0eee645bac49b482a` (`track2-yolo-review-host`, ap-southeast-1) |
| Path on host | `/opt/review-app/` |
| Container | `review-app-container`, port 8090, `--restart unless-stopped` |
| Image | `review-app`, built **on the instance** — never pushed to ECR |
| Source staging | `s3://track2-s3/review-app-source/` |

`deploy_review_app.sh` uploads `Dockerfile`, `review_app.py` and a generated
`run_container.sh` to S3, then drives the instance over SSM to pull all
three, build the image locally and restart the container. It goes through
S3 rather than embedding the files in the SSM command because quoting a
JSON blob inside SSM's own JSON proved too fragile.

There is **no CI for this app**. It ships only when someone runs that
script, which is worth knowing before assuming a push deployed it.

## Deploying

The instance must be running (it is stopped nightly at 22:00 SGT and over
weekends), and the terraform stack must be applied — the script reads the
instance id, bucket, table and role ARN from its outputs:

```bash
aws ec2 start-instances --instance-ids i-0eee645bac49b482a --region ap-southeast-1
./MLOPS_TRACK2/deploy_review_app.sh
```

`INFRA_DIR` overrides where terraform is read from; it defaults to
`jurong-port/terraform/infra`.

## Configuration

All environment variables, set by `deploy_review_app.sh` from terraform
outputs — not read from a file on the host:

`REVIEW_QUEUE_TABLE`, `PIPELINE_BUCKET`, `AWS_REGION`, `RETRAIN_RULE_NAME`,
`TRAINING_PIPELINE_NAME`, `MODEL_PACKAGE_GROUPS_JSON`,
`MULTI_SIZE_ENDPOINT_NAME`, `MULTI_SIZE_VARIANT_NAME`,
`SAGEMAKER_EXECUTION_ROLE_ARN`, `SERVING_IMAGE_URI`,
`RUN_PIPELINE_CODEBUILD_PROJECT`.

It authenticates as the instance's own IAM role (`track2-yolo-review-host`);
there are no credentials in the image.

## Shared state, and why that matters

This app and JP_UI read and write the **same** DynamoDB tables, S3 bucket,
SageMaker endpoint and Model Registry. A deploy from one is visible to the
other, and either can start training jobs or change what is served.

Most relevant: JP_UI now creates the inference endpoint on demand and
deletes it once its queue drains. If this app assumes a standing endpoint,
it will find one missing that used to always be there.
