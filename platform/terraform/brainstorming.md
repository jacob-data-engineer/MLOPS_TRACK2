# Brainstorm: park → train → evaluate → keep, AWS-native

## The goal, restated

Train a YOLO candidate, "park" it somewhere cheap and harmless, look at how
well it actually performs, and only promote ("keep") the ones you like —
without every experiment costing money to sit around, and without hand-rolling
a tracking system.

## The AWS-native answer: this is what SageMaker Model Registry *is*

You don't need to invent "parking" — **Amazon SageMaker Model Registry**
(a `Model Package Group`, already created in this stack as
`track2-yolo-models`) is exactly this primitive:

- Every trained candidate registers as a new **version** in the group.
- A parked version costs **nothing** — it's metadata + a pointer to the S3
  artifact, no compute, no storage cost beyond the artifact itself (which
  you're paying for anyway once training produces it).
- Each version carries an approval status: `PendingManualApproval` (parked,
  default), `Approved` (kept), `Rejected` (discarded, but still visible in
  history — nothing is silently deleted).
- You can park as many candidates as you want and compare them side by side
  before deciding.

This is not a new thing to build — **it's already wired up** in this repo.
The gap is the *evaluation* step in between (see below).

## The full loop, mapped to what already exists in `terraformer_track_2`

```
 ┌─────────────┐   ┌──────────────┐   ┌───────────────┐   ┌────────────┐
 │ Train        │──▶│ Park          │──▶│ Evaluate       │──▶│ Keep?      │
 │ (TrainingStep│   │ (RegisterModel│   │ (MISSING —     │   │ approve /  │
 │  on ECR      │   │  → Model      │   │  see gaps      │   │ reject     │
 │  train-latest│   │  Package      │   │  below)        │   │            │
 │  image)      │   │  Group)       │   │                │   │            │
 └─────────────┘   └──────────────┘   └───────────────┘   └─────┬──────┘
                                                                  │
                                        ┌─────────────────────────┘
                                        ▼
                              ┌───────────────────┐
                              │ Deploy (only the   │
                              │ kept one) —        │
                              │ serverless endpoint │
                              └───────────────────┘
```

| Stage | AWS service | Already built? | Where |
|---|---|---|---|
| Train | SageMaker Training Job (via Pipeline) | ✅ yes | `src/pipeline_definition.py`'s `TrainingStep`, `training-container/train.py` |
| Park | SageMaker Model Registry | ✅ yes | `RegisterModel` step, registers to `track2-yolo-models`, status `PendingManualApproval` |
| Evaluate | *(nothing yet)* | ❌ **gap** | see below |
| Keep / discard | SageMaker Model Registry approval status | ✅ yes (manual gate) | `src/approve_model.py` — no "reject" companion yet, easy to add |
| Deploy the kept one | SageMaker Serverless Endpoint | ✅ yes, gated on `Approved` | `src/deploy_endpoint.py` |
| Run it all on AWS, not locally | CodeBuild | ✅ yes | `infra/codebuild.tf` (containers), `infra/codebuild_pipeline.tf` (pipeline trigger) |

## Walkthrough with the actual commands, today

```sh
# 1. Train a candidate — parks itself automatically on success
aws codebuild start-build --project-name track2-yolo-run-pipeline --region ap-southeast-1

# 2. See what's parked, across all past runs
uv run python src/approve_model.py --list
#   arn:...:model-package/track2-yolo-models/3  status=PendingManualApproval
#   arn:...:model-package/track2-yolo-models/2  status=PendingManualApproval
#   arn:...:model-package/track2-yolo-models/1  status=Rejected

# 3. Look at how it actually performed
#    <-- this is the missing piece, see "Gaps" below -->

# 4. Liked it? Keep it.
uv run python src/approve_model.py <arn>

# 5. Put the kept one live
uv run python src/deploy_endpoint.py <arn>

# 6. Didn't like it? Leave it PendingManualApproval (still parked, zero
#    cost, nothing to clean up) or explicitly mark it Rejected once a
#    reject script exists (see gaps).
```

Every step already runs via CodeBuild (step 1) or a local/AWS-agnostic
Python script (steps 2-5) — nothing here requires a persistent server.

## Gaps — status

### 1. A real evaluation step — BUILT, pending live validation
`training-container/evaluate.py`, run as a new `ProcessingStep`
(`EvaluateYolo`) between `TrainingStep` and `RegisterModel` in
`pipeline_definition.py`. Reuses the training image (already has
Ultralytics) rather than a separate evaluation container — no new image
to maintain. Calls Ultralytics' own `model.val()` against the held-out
validation split (`images/dataset/images/val/`, 55 images) — real
mAP50/mAP50-95/precision/recall, not hand-rolled box-matching.

Attached to the registered model via `RegisterModel`'s `model_metrics`
parameter (`ModelMetrics(model_quality=MetricsSource(...))`) — every
parked candidate now carries real numbers in the SageMaker console's
Model Registry view, not a blind approve/reject decision.

**Real bug caught while wiring this up**: `seed_training_data.py`'s
`data.yaml` had a hardcoded absolute `path:` specific to the *training*
job's channel mount (`/opt/ml/input/data/train`) — would have silently
broken inside the *Processing* job's different mount path
(`/opt/ml/processing/data`) the moment `EvaluateYolo` actually ran.
Fixed by dropping the hardcoded `path:` entirely — Ultralytics defaults
it to the YAML's own directory, which resolves correctly under either
mount point.

### 2. `reject_model.py` — BUILT
`src/reject_model.py`, same shape as `approve_model.py`, sets
`ModelApprovalStatus=Rejected`. Trivial, as expected.

### 3. Real bounding-box labels — still not done
Unchanged from before: every candidate still trains against **synthetic
placeholder boxes**. Not an AWS-architecture question — a data problem.
Nothing else in this loop needs to change once real Label Studio exports
exist.

### 4. Parking *variations* — BUILT (the simple path; HPO still deferred)
`pipeline_definition.py` now reads `epochs`/`base_model`/`img_size` from
environment variables (`TRAIN_EPOCHS`/`TRAIN_BASE_MODEL`/
`TRAIN_IMG_SIZE`), falling back to the original fixed defaults.
`scripts/trigger_training_variant.sh` triggers a run with explicit
overrides via CodeBuild's `--environment-variables-override`. A full
SageMaker Hyperparameter Tuning Job (many parallel runs across a search
space, N × GPU cost) is still deferred — per the original reasoning,
worth doing once real labels exist and repeating a training run many
times over is actually worth the spend.

### 5. SageMaker Experiments for side-by-side comparison — BUILT
Every pipeline execution now gets its own trial
(`pipeline_experiment_config=PipelineExperimentConfig(experiment_name=
"track2-yolo-training", trial_name=ExecutionVariables.
PIPELINE_EXECUTION_ID)` on the `Pipeline` itself), and `evaluate.py`
logs its computed metrics into that trial.

**Deliberately not using bare `load_run()` auto-detection** — verified
via a real GitHub issue
([aws/sagemaker-python-sdk#4114](https://github.com/aws/sagemaker-python-sdk/issues/4114))
that this specific pattern (`load_run()` inside a `ProcessingStep`
within a pipeline, no explicit experiment config) has a documented SDK
bug. Instead, `EXPERIMENT_NAME`/`RUN_NAME` are passed into the
Processing job's container environment explicitly (same mechanism as
`PIPELINE_BUCKET` for `train.py`), and `evaluate.py` constructs the
`Run` directly rather than relying on auto-detection. Wrapped in
try/except — Experiments logging failing doesn't fail the pipeline step;
the `evaluation.json` output and model registration are what actually
matter, comparison in Studio is a bonus.

**Not yet validated against a real execution** — this is new pipeline
code (the `ProcessingStep`, the `model_metrics` wiring, the experiment
config) that has never actually run on AWS. A real run is in progress
to confirm the `sagemaker_execution` role's `AmazonSageMakerFullAccess`
managed policy actually covers `CreateProcessingJob`/`CreateExperiment`/
`CreateTrial`/`CreateTrialComponent` as expected (reasoned through, not
confirmed) — update this section once that run completes.

## Plan: register a pretrained baseline in the Model Registry (do tomorrow)

Goal: get *something* real and working into `track2-yolo-models` immediately,
without waiting on real labels or a full training run — a generic
COCO-pretrained checkpoint, registered as-is. Gives a concrete baseline to
compare every future fine-tuned candidate against, and something
approvable/deployable today if wanted.

**Not executed yet — this is the plan, ready to run.**

1. Write `src/register_baseline_model.py`:
   - `YOLO("yolov8n.pt")` — same auto-download Ultralytics already does in
     `train.py`, just without any training step.
   - Package the weights as `model.tar.gz` (a `model.pt` file at the root —
     same shape training output already produces), upload to
     `s3://track2-yolo-pipeline-125192671932/model-registry/yolov8n-baseline/model.tar.gz`.
   - Call `boto3`'s `sagemaker.create_model_package(...)` directly (not
     through the Pipeline's `RegisterModel` step — that's for
     pipeline-produced models specifically):
     - `ModelPackageGroupName`: `track2-yolo-models`
     - Container: the already-built `serve-latest` inference image
     - `ModelApprovalStatus`: `PendingManualApproval`
     - `CustomerMetadataProperties`: `{"source": "pretrained-coco-baseline"}` —
       so it's clearly distinguishable from fine-tuned candidates later
2. Run it locally or via a one-off CodeBuild trigger (same S3-source
   pattern as `codebuild_pipeline.tf` — could even reuse that project
   with a different buildspec command, rather than standing up a third
   CodeBuild project for a one-time action).
3. Verify: `uv run python src/approve_model.py --list` shows it alongside
   any real training runs.
4. Decide then whether it's worth approving/deploying as a placeholder
   while real-labeled fine-tuning is still in progress, or just leaving
   it parked as a reference point.

## Can the training pipeline be replaced by a native AWS service instead of custom code?

Short answer: **partially** — re-triggering, yes; defining the pipeline's
steps, not cleanly.

**Verified via Terraform's own provider docs** (not guessed — see sources
below, after getting console navigation wrong earlier in this build meant
being more careful here):

- `aws_sagemaker_pipeline` **does exist** as a Terraform resource — so in
  principle, the pipeline could be Terraform-managed instead of created via
  `run_pipeline.py`'s `pipeline.upsert()` call.
- But its `pipeline_definition` argument takes a **raw JSON string** (or an
  S3 pointer to one) — and that JSON is exactly what the SageMaker Python
  SDK compiles *for you* from the readable `TrainingStep`/`ModelStep`
  objects in `pipeline_definition.py`. Hand-authoring or regenerating that
  JSON directly wouldn't eliminate custom logic — it would just move it
  from a maintainable ~60-line Python file to a much less maintainable raw
  JSON blob. **Not a real win**, so: keep `pipeline_definition.py` as-is.

**What genuinely *is* replaceable — and worth doing**: once a pipeline is
already defined, **re-running it needs no custom code or CodeBuild at
all**:
```sh
aws sagemaker start-pipeline-execution --pipeline-name track2-yolo-training-pipeline --region ap-southeast-1
```
This is a single native AWS CLI/API call. Even *scheduled* re-runs could
skip CodeBuild entirely, using the exact same EventBridge Scheduler
"universal target" pattern already used for the EC2 instance's
stop/start schedule (`ec2_schedule.tf`) — pointed at
`arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution` instead of
`ec2:startInstances`.

**Net conclusion**: `track2-yolo-run-pipeline` (the CodeBuild project) is
only strictly necessary for the rare act of *defining or updating* the
pipeline's steps. For routine re-runs of an already-defined pipeline, a
plain `aws sagemaker start-pipeline-execution` call — or a native
EventBridge Scheduler rule, zero custom code — is the right tool, not
CodeBuild.

Sources:
- [aws_sagemaker_pipeline | Resources | hashicorp/aws | Terraform Registry](https://registry.terraform.io/providers/hashicorp/aws/latest/docs/resources/sagemaker_pipeline)
- [AWS::SageMaker::Pipeline CloudFormation reference (PipelineDefinition property)](https://docs.aws.amazon.com/zh_cn/AWSCloudFormation/latest/TemplateReference/aws-resource-sagemaker-pipeline.html)

## Plan: scheduled training trigger, native SageMaker + EventBridge (do tomorrow)

**Yes, achievable — and further along than it looks.** Three things asked
for: (1) push a model into the registry, (2) have "a pipe", (3) a
scheduled trigger for that pipe, via SageMaker natively. Checking each
against what's actually live right now:

| Ask | Status as of today |
|---|---|
| "A pipe" | **Already exists.** `track2-yolo-training-pipeline` is a real, `Active` SageMaker Pipeline resource (confirmed via `describe-pipeline`), created during today's run. We stopped the *execution*, not the pipeline definition — it's still there, ready to be started again with no redefinition needed. |
| Push a model into the registry | **Mechanism confirmed, two paths.** (a) The pipe's own `RegisterModel` step already does this automatically on every successful training run. (b) The separate baseline-registration plan (above) covers pushing in a model that *wasn't* produced by the pipeline. |
| Scheduled trigger, via SageMaker | **Not built yet, but a clean native fit — better than expected.** See below. |

### The scheduled trigger — better option than the EC2 pattern

For the EC2 stop/start schedule (`ec2_schedule.tf`), EventBridge Scheduler
only offered a generic **universal target**
(`arn:aws:scheduler:::aws-sdk:ec2:startInstances`) — a customizable
catch-all for any AWS API action.

For SageMaker Pipelines specifically, EventBridge Scheduler has something
better: a **dedicated templated target** built for exactly this
(`SageMakerPipelineParameters`) — not the generic workaround, a first-class
supported integration. Practically: point a schedule at the pipeline's own
ARN, and Scheduler knows to call `StartPipelineExecution` automatically,
optionally passing pipeline parameters.

**What this means for tomorrow**: no Lambda, no CodeBuild, no polling
script — just:
- One `aws_scheduler_schedule` Terraform resource (new file, e.g.
  `pipeline_schedule.tf`), same shape as `ec2_schedule.tf`'s pattern
  — pick a cadence (daily/weekly retrain, or whatever cadence makes sense
    once real labels exist and new data is worth retraining on)
- One IAM role for the scheduler to assume, scoped to
  `sagemaker:StartPipelineExecution` on `track2-yolo-training-pipeline`'s
  ARN specifically (tighter than the EC2 schedule's role, since the
  pipeline ARN is real and known — no "doesn't exist yet" problem the EC2
  schedule didn't have either, since that instance's ARN was also
  interpolated from the resource being created in the same apply)

**Net conclusion**: everything asked for is achievable with what exists
today, and the scheduled-trigger piece is a small, well-precedented
addition — genuinely one new Terraform file, not a new architecture.

**Correction #1, found before implementing**: the templated
`sagemaker_pipeline_parameters` target has a documented Terraform
provider bug — "unable to create pipeline target with no parameters" —
and `pipeline_definition.py` defines zero `Parameter*` objects, which
matches that broken scenario exactly. Tried the **universal target**
instead (`arn:aws:scheduler:::aws-sdk:sagemaker:startPipelineExecution`,
same pattern already proven for the EC2 schedule).

**Correction #2, hit at apply time**: the universal target's raw
`StartPipelineExecution` API call requires a `ClientRequestToken`
(32-128 chars) for idempotency — `CreateSchedule` failed with
`ValidationException: ... missing ... ClientRequestToken`. A **static**
token from Terraform config breaks the whole point of a *recurring*
schedule: every week's invocation would carry the identical token,
which SageMaker's idempotency check would treat as a duplicate of the
first call — meaning it would silently stop actually retraining after
week one, not error, just quietly do nothing on weeks 2+.

**What actually shipped**: a tiny Lambda (`trigger_weekly_retrain.py`,
`infra/pipeline_schedule.tf`) that calls `boto3`'s
`start_pipeline_execution(PipelineName=...)` directly — boto3
auto-generates a fresh idempotency token per call when one isn't
supplied, so this sidesteps the problem entirely — triggered by a
**classic EventBridge rule** (`aws_cloudwatch_event_rule`, cron
`0 15 ? * SUN *` = Sunday 23:00 Asia/Singapore, converted by hand since
classic EventBridge cron is UTC-only, unlike EventBridge Scheduler's
`schedule_expression_timezone`). This is the exact same mechanism
`confidence_monitor.py` already uses successfully elsewhere in this
stack — not a new pattern, a proven one, applied a second time.

**Applied and live**: `track2-yolo-weekly-retrain` Lambda +
`track2-yolo-weekly-retrain-schedule` EventBridge rule, both confirmed
created.

**Lesson for next time**: prefer the pattern already proven working in
this codebase over a theoretically-more-native alternative, especially
once the "more native" option has already failed once. Two failed
attempts before landing on the boring, already-proven choice — worth
trying that first next time a similar "AWS should have a first-class way
to do this" instinct comes up.

Sources:
- [SageMakerPipelineParameters — EventBridge Scheduler API Reference](https://docs.aws.amazon.com/fr_fr/scheduler/latest/APIReference/API_SageMakerPipelineParameters.html)
- [Using templated targets in EventBridge Scheduler](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-templated.html)
- [Using universal targets in EventBridge Scheduler](https://docs.aws.amazon.com/scheduler/latest/UserGuide/managing-targets-universal.html)

## What NOT to build

- **No standing/always-on endpoint per parked candidate.** Only the
  *approved* one should ever get a real endpoint — everything else stays
  as a zero-cost registry entry. This is already how `deploy_endpoint.py`
  is written (refuses anything not `Approved`).
- **No custom "model store" service.** SageMaker Model Registry already
  does versioning, approval state, and artifact pointers — reinventing
  this with a DynamoDB table or S3 prefix convention would just be a
  worse copy of a service AWS already provides for free (metadata-only
  cost).
