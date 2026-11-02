# Track2 Retraining Platform — Technical Documentation

*Preliminary — 2 November 2026. What each part of the repository does, and
the AWS infrastructure it runs on. For how to USE the platform, see
[documentation.md](documentation.md). For the running handover and open
threads, see [CONTINUE.md](CONTINUE.md).*

---

## 1. The system in one picture

```
                         Browser (operator)
                               │  HTTPS-less HTTP, bearer token
                               ▼
     ┌────────── EC2 review host  i-0eee645bac49b482a (18.140.196.76) ──────────┐
     │  Docker:  :8501 production (ppe-pipeline-track2-test, role stable)       │
     │           :8090 development (track2-app-dev, role dev)                   │
     │  Each = one FastAPI process serving the API + the built React SPA,       │
     │  plus a 25-second "tick" (chain/ticker.py) doing background work.        │
     │  Settings files: /opt/track2-app/settings/<container>/  (host-mounted)   │
     └───────┬──────────────┬─────────────┬───────────────┬────────────────────┘
             │              │             │               │
             ▼              ▼             ▼               ▼
        S3 buckets     DynamoDB      SageMaker        RDS Postgres
        track2-s3      config,       pipelines,       track2-catalog (ours, records)
        safety-model-  queues,       registry,        violation-review-postgres  ┐ review app's,
        dataset(-dev)  audit,        training jobs    jpui-preview-crop-authority┘ read via Lambda
                       snapshots
             ▲              ▲
             │              │
     Lambdas on EventBridge rules: pull router (arrival/schedule),
     queue runner (every 2 min), notifications, approved-weights dump
```

**Two halves of the app** — mirrored in the code (`data/`, `training/`,
`chain/`) and in the screens (Data Preparation, Multi-Auto Training):

- **Data side** turns files in S3 into a *dataset version* a run can train on.
- **Training side** runs, queues and records training, and what happens to a
  model afterwards.
- **The chain** is the arrow between them: pull → build a version → queue → train.

---

## 2. Infrastructure

Region **ap-southeast-1**, account **125192671932**. Two Terraform stacks:

| Stack | Folder | Owns | Note |
|---|---|---|---|
| App | `terraform/app/` | ECR + CodeBuild for the app, audit tables, pull-control Lambdas and rules, the catalog database | What `deploy.sh` reads. Has a few unrelated pending changes — apply new resources with `-target` |
| Platform | `terraform/platform/` | The EC2 host, IAM roles, the bucket, SageMaker groups, monitoring, schedules | **Has drift** (~18 resources). Never apply whole; make the AWS change and edit the `.tf` to match |

### 2.1 Compute and build

| Resource | What it is |
|---|---|
| EC2 `i-0eee645bac49b482a` (`track2-yolo-review-host`) | Runs the app containers. Stable public IP (no nightly stop); 09:00 weekday start rule kept as a way back |
| Security group `sg-0e1e86337e2a3c54a` | Opens 8501 and 8090 (8092 closed 2026-11-02) |
| ECR `track2-app` | App images, tagged `YYYYMMDD-HHMMSS` |
| ECR `track2-app-base` | Mirrored base images (Node, uv/Python) — keeps builds ~4 min instead of up to 18 |
| ECR `track2-yolo-train` | Training + serving images for SageMaker |
| CodeBuild `track2-app` | Builds the app image (`tsc -b && vite build` is the TypeScript gate) |
| CodeBuild `track2-yolo-train-images` | Builds the training container from `model_training/training-container-src.zip` |
| CodeBuild `track2-yolo-run-pipeline` | Starts SageMaker pipeline executions off any laptop |

### 2.2 Storage

| Resource | Holds |
|---|---|
| S3 `track2-s3` | The app's bucket: week folders (`retraining/`), golden set, `_datasets/<dv-id>/` frozen versions, `_dataset_snapshots/`, generated yamls, weights, exports (in the central store folder), code inputs under `custom/` |
| S3 `safety-model-dataset-dev`, `safety-model-dataset` | Source and review-app data — e.g. `eval/alarm_images/` frames the SQL tab finds |
| DynamoDB `track2-pull-control` | **Live state**: settings (config), pull run state, claims ("only one container does this"), pull queue, chain progress |
| DynamoDB `track2-yolo-dataset-snapshots` | Snapshot and dataset-version records (being copied to the catalog) |
| DynamoDB `track2-yolo-training-audit`, `track2-yolo-training-files` | Every run's record and every file it used (copied to the catalog) |
| DynamoDB `track2-yolo-eval-runs`, `-inference-lineage`, `-inference-queue`, `-review-queue` | Evaluation, inference and review records/queues |
| RDS `track2-catalog` (Postgres 16, db.t4g.micro) | **The catalog** — records in SQL: dataset versions + members, training runs (numbered) + files, exports, sync log. Private, encrypted, 7-day backups, deletion-protected. Password in Secrets Manager (RDS-managed) |
| RDS `violation-review-postgres`, `jpui-preview-crop-authority` | **The review app's** databases. Read-only from here, through the query Lambdas, as user `track2_readonly` |
| SSM `/track2/pg-query/<db>/readonly-password` | The read-only users' passwords (encrypted) |

### 2.3 Lambdas and schedules (this app)

| Lambda | Triggered by | Does |
|---|---|---|
| `track2-app-pull-control-router` | Rules `track2-app-pull-control-arrival` / `-schedule` | Runs a pull: routes arriving files into week folders. Built from `pull_control.py` alone |
| `track2-app-queue-runner` | Rule `track2-app-queue-tick` (every 2 min) | Advances the training queue when no app container is up. Read-only on S3 |
| `track2-pg-query-violation-review`, `-jpui-preview` | Invoked by the app | Read-only SQL inside each review database's VPC (`terraform/app/lambda_src/pg_query.py`) |
| `track2-yolo-dump-approved-weights` | Rule `track2-yolo-model-package-approved` | When a registry version is Approved, writes its weights out |
| `track2-yolo-training-completed-notify` | Rule `track2-yolo-training-pipeline-completed` | Emails when a pipeline run starts/ends (SNS `track2-yolo-reviewer-notifications`) |
| `track2-yolo-review-host-started-notify` | Rule `track2-yolo-review-host-running` | Emails the app URL when the host starts |
| `track2-yolo-confidence-monitor`, `-trigger-review` | Every 15 min / monitor | Model-quality monitoring into the review queue |
| `track2-yolo-weekly-retrain` | Rule `track2-yolo-weekly-retrain-schedule` | The older full-frame replay retrain (Fridays) |

*Not this app's, though named `track2-` or nearby: the `track2-fullframe-*`
rules, CodeBuild and ECR repos, `track2-jlma-dev-dataset-snapshot`, and the
`ppe-*` ECR repositories.*

### 2.4 SageMaker

| Resource | Use |
|---|---|
| Pipeline `track2-yolo-open-training-pipeline` | What custom/scheduled runs start: train → evaluate → register. Definition = `model_training/custom_training/pipeline.json` (pinned image digest) |
| Pipelines `track2-yolo-custom-training-pipeline`, `track2-yolo-training-pipeline` | Older definitions |
| Model package groups `yolo<gen><size>-<model>-detector` | One per Model Name and size, e.g. `yolo11s-helmet-detector`. **Never** name a group exactly `yolo11<size>-detector` — those four drive the live-weights swap |

### 2.5 Costs that run continuously

- EC2 review host (always on).
- RDS `track2-catalog` (~US$15–25/month, must stay running — it records versions).
- RDS review databases belong to the review app; `jpui-preview` is stopped when idle.
- Training is per run (GPU instances only while a pipeline runs).

---

## 3. Repository map

### 3.1 Top level

| File | Does |
|---|---|
| `README.md` | Layout overview |
| `CLAUDE.md` | Working rules for this repo (deploying, testing, conventions, known gaps) |
| `CONTINUE.md` | Running handover: state, changes, open threads |
| `documentation.md` | User guide |
| `technical_documentation.md` | This file |
| `changes.md`, `uat.md` | Requested changes; UAT notes (13 Oct) |
| `Dockerfile` | Two stages: build the React SPA (Node), then the Python app (uv) serving it |
| `deploy.sh` | Packages the tree, builds in CodeBuild, swaps the container on a port over SSM. `IMAGE_TAG=` redeploys an existing image |
| `mirror.yml` | CodeBuild buildspec that refreshes the mirrored base images |
| `pyproject.toml`, `uv.lock`, `.python-version` | Python dependencies (FastAPI, boto3, ultralytics, pg8000 …) |
| `.gitignore`, `.gitattributes`, `.dockerignore` | Keep state/build output out; force LF line endings |

### 3.2 `src/ppe_pipeline_api/` — the backend

**Top level (shared by both sides)**

| File | Does |
|---|---|
| `main.py` | The FastAPI app: registers every router, serves `web/dist`, starts the tick (unless `APP_BACKGROUND=off`) |
| `auth.py`, `session_store.py` | Password login (`PPE_APP_PASSWORD`), bearer tokens, sessions |
| `pull_control.py` | Pull control and **the settings store**: week folders, routing, triggers, settings files (`_DP`/`_AT` scopes), claims. Also the pull Lambda's whole source |
| `storage.py` | The bucket name and S3 helpers both sides use |
| `notify.py` | Who gets which emails |
| `_door.py` | How each side's front door (`data/`, `training/`) exposes functions |
| `CONTRACT.md` | The API contract for the SPA |
| `dataset_*.py`, `golden_set.py`, `retention.py`, `track2_shared.py` … at top level | **Forwarders only** — old import paths that now point at the moved modules |

**`data/` — the data side** (imports neither `training` nor `chain`)

| File | Does |
|---|---|
| `__init__.py` | Front door: the functions the data side offers |
| `dataset_yaml.py` | Builds a `data.yaml` from week folders / picked folders, without moving files |
| `retention.py`, `retained_lists.py` | Recency weighting (decay) and the image lists that enforce it |
| `golden_set.py` | Splits a curated folder into train/val |
| `assemble.py`, `selection.py` | Datasets from scattered folders; what a dataset selects |
| `dataset_snapshots.py`, `dataset_versions.py` | Frozen snapshots; per-run frozen dataset versions under `_datasets/<id>/` |
| `sql_explorer.py` | The SQL tab: query Lambdas, image detection, human-verified presets, labels from database boxes, pulls, previews, start/stop |
| `catalog.py` | The catalog database: schema + migrations, connection (verified TLS), status |
| `catalog_sync.py` | Copies records from DynamoDB into the catalog every 5 minutes; never deletes |
| `catalog_view.py` | The Data Catalog page's reads, CSV lists, zip exports streamed to S3 |
| `rds-ca-bundle.pem` | Amazon's RDS root certificates, for verifying the catalog connection |

**`training/` — the training side**

| File | Does |
|---|---|
| `__init__.py` | Front door |
| `track2_shared.py` | The bulk of training logic: registry, runs, queue, audit, evaluation, inference, deploy, cost |
| `hyperparams.py` | The schema for extra Ultralytics arguments |
| `run_artifacts.py` | Keeps what a run consumed where it cannot move |
| `metrics_export.py` | Exports a Class's best checkpoint, by hand or on a watch |

**`chain/` — between the sides**

| File | Does |
|---|---|
| `ticker.py` | The 25-second clock: queue, after-pull chain, training schedule, scheduled assemble, catalog sync, pull queue, exports |
| `retrain_schedule.py` | The unattended run: rebuild → freeze → queue → train; the training trigger (pull / clock / off); `submit="build"` stops before training |
| `queue_after_pull.py` | Chains a finished pull into training when the training schedule follows the pull |
| `assemble_after_pull.py` | Assemble's *Scheduled*: rebuild the dataset after each scheduled pull, no training |
| `queue_runner.py` | The queue-runner Lambda's handler |

**`routers/` — HTTP routes, one per area**

| File | Prefix | Area |
|---|---|---|
| `track2_pull_control.py` | `/track2/pull-control` | Pull control, settings files, week folders |
| `track2_retention.py` | `/track2/retention` | Retention, snapshots, the training schedule |
| `track2_models.py` | `/track2/models` | Training: datasets, queue, runs, audit, GPU status |
| `track2_sql.py` | `/track2/sql` | SQL tab |
| `track2_catalog.py` | `/track2/catalog` | Catalog status, sync, runs, versions, files, exports |
| `track2_metrics.py`, `track2_lineage.py` | `/track2/metrics`, `/track2/lineage` | Registry metrics; lineage (read only) |
| `track2_eval.py`, `track2_deploy.py`, `track2_export.py` | `/track2/eval`, `/deploy`, `/export` | Evaluation queue; deploy & inference; weight export |
| `track2_review.py`, `track2_schedule.py`, `track2_cost.py`, `track2_notifications.py` | … | Review queue & approvals; legacy schedule; SageMaker cost; email subscribers |

### 3.3 `web/` — the React SPA

| Path | Does |
|---|---|
| `src/App.tsx` | Routes |
| `src/components/Layout.tsx` | Sidebar (tabs + status colours), build badge |
| `src/lib/api.ts` | Every call to the backend, with its types |
| `src/lib/bus.ts`, `cardState.ts`, `pendingChanges.ts`, `useGoldenSync.ts` | Cards telling each other things changed; saved open/closed state; unapplied-change tracking |
| `src/lib/format.ts`, `slug.ts` | Time formatting (SGT); name → registry identifier |
| `src/routes/Track2DataPrepPage.tsx` | Data Preparation |
| `src/routes/Track2ModelsPage.tsx` | Multi-Auto Training |
| `src/routes/Track2SqlPage.tsx` | SQL Database Query |
| `src/routes/Track2CatalogPage.tsx` | Data Catalog |
| `src/routes/Track2Scheduled/Metrics/Notifications/Help…Page.tsx` | The other tabs; Deploy, Eval, Export, Lineage, Schedule, Cost, Synthetic are routed but off the sidebar |
| `src/components/Track2PinnedBar.tsx` | The per-tab toolbar: settings file + schedule |
| `src/components/Track2*Panel.tsx` | The cards: Assemble, Golden Set, Pull Control, Auto Training, Retention, Pull Queue, Central Store, Detectors, Import, Training Audit, Dataset Versions … |
| `src/components/Card.tsx`, `ConfirmDialog.tsx`, `S3FolderField.tsx`, `Track2S3FolderBrowser.tsx` | Shared building blocks |

### 3.4 `terraform/`

| Path | Does |
|---|---|
| `app/main.tf`, `variables.tf`, `outputs.tf` | App build infrastructure; outputs `deploy.sh` reads |
| `app/audit.tf` | Training audit tables and access |
| `app/pull_control.tf` | Drop folders, the pull router and queue runner Lambdas, their rules |
| `app/base_images.tf` | The mirrored base-image repository |
| `app/catalog.tf` | The catalog database, its network rule, its access |
| `app/lambda_src/pg_query.py` | The SQL tab's read-only query Lambda |
| `platform/*.tf` | EC2, IAM, S3, ECR, CodeBuild, SageMaker groups, monitoring, schedules, review queue, Jurong Port serving |
| `platform/lambda_src/*.py` | The platform Lambdas listed in 2.3 |

### 3.5 `model_training/` — what runs in SageMaker

| Path | Does |
|---|---|
| `training-container/train.py` | The training entry point (ships by its own CodeBuild route, then re-pin the digest in `pipeline.json`) |
| `training-container/evaluate.py`, `inference.py`, `Dockerfile*` | Evaluation, serving, images |
| `custom_training/pipeline.json` | **The live open-pipeline definition** (`pipeline_open.json` is stale) |
| `custom_training/evaluate_any.py` | Evaluator for any class set — uploaded to `s3://track2-s3/custom/code/`, no build |
| `custom_training/prepare_any_dataset.py`, `make_scratch_checkpoint.py` | Dataset preparation; from-scratch seeds |
| `src/*.py` | Pipeline definition, deploy/approve/reject, model cards, baselines |
| `live_pipeline/`, `replay_lambda/` | The older full-frame pipeline and weekly replay coordinator |

### 3.6 `scripts/`, `platform/`, `tests/`

| Path | Does |
|---|---|
| `scripts/setup_sql_tab.sh`, `sql_tab_policy.json` | One-time SQL-tab setup; the app policy (Start/Stop pending approval) |
| `scripts/deploy_pipelines.py`, `make_scratch_seeds.py`, `mirror_yolo_weights.py`, `test_week_partitioning.py` | Pipeline deploy, seeds, COCO weight mirror, a week-partitioning check |
| `platform/` | The Jurong Port SageMaker BYOC serving scaffold and its notes |
| `tests/` | ~140 test files, ~2,490 tests: `.venv/Scripts/python.exe -m pytest tests/ -q` |

---

## 4. How code reaches each environment

| What | Route |
|---|---|
| App (API + SPA) | `PORT=8090 CONTAINER=track2-app-dev APP_ROLE=dev bash deploy.sh` → CodeBuild → ECR → SSM swap. Production is **promoted** with `IMAGE_TAG=<dev's tag>` on 8501, never rebuilt |
| Pull router / queue runner Lambdas | Updated by a dev or stable deploy (skipped when `APP_BACKGROUND=off`) |
| `train.py` | Repack the container zip → upload → CodeBuild `track2-yolo-train-images` → re-pin digest in `pipeline.json` → `aws sagemaker update-pipeline` |
| `evaluate_any.py` | Upload to `s3://track2-s3/custom/code/` (back up the old one first) |
| Infrastructure | Terraform: app stack with `-target`; platform stack by targeted AWS change + matching `.tf` edit |

## 5. Background work, and where it runs

| Job | Runs in | How often |
|---|---|---|
| Pull (arrival / schedule) | Pull router Lambda | Per EventBridge rule |
| Advance training queue | App tick **and** queue-runner Lambda | 25 s / 2 min |
| Chain after a pull; training clock; scheduled assemble | App tick | 25 s |
| Catalog sync from DynamoDB | App tick (one container per slot) | 5 min |
| Exports | Background thread in the app | On request |

The production and development containers both tick against the same data;
`claim_once` (a conditional write in `track2-pull-control`) makes sure each
scheduled job happens once.
