# Summary: Deploying a YOLO11s model (with weights) to SageMaker (2026-08-31)

> Note: the actual code for this work lives in `c:\SNAIC\terraformer_track_2`
> (the Track2 YOLO demo pipeline), not in this `JURONG PORT` project — this
> summary is filed here at explicit request, as a cross-project record of
> how a model-with-weights gets pushed into SageMaker.

## Starting point
A YOLO11s checkpoint (`experimental_full_frame_yolo11s_e5e4bea8728b7c60.pt`,
18.3MB) had already been uploaded to S3 (bucket since deleted, see the
2026-09-06 update below) — not yet packaged for SageMaker, not registered
anywhere, not deployed.

## What "pushing a model into SageMaker" actually means
There's no literal file upload to SageMaker — it always works off S3 as the
artifact store, and pulls what it needs itself:

1. **Downloaded** the raw `.pt` checkpoint from S3 to local disk.
2. **Repackaged** it into `model.tar.gz`, renaming the file to exactly
   `model.pt` inside the tarball — the existing serving container's
   `inference.py`'s `model_fn` hardcodes that filename (`YOLO(f"{model_dir}/model.pt")`),
   so this rename is required regardless of the original filename.
3. **Uploaded** the tarball back to S3, into the same bucket as the
   original weights (since deleted), rather than moving it into
   `terraformer_track_2`'s own pipeline bucket.
4. **Registered** it into the SageMaker Model Registry via `create_model_package`
   — a metadata record pointing at that S3 URI plus the existing serving
   container image (`track2-yolo-train:serve-latest`, already built via
   CodeBuild in an earlier session). No file transfer happens at this step,
   it's a pointer. Landed as `PendingManualApproval`.
5. **Approved** it (`ModelApprovalStatus=Approved`).
6. **Deployed**: created a `Model` (referencing the approved model package),
   an `EndpointConfig`, and an `Endpoint`. At endpoint-creation time,
   SageMaker itself — via the `track2-yolo-sagemaker-execution` IAM role —
   reaches into S3 and pulls `model.tar.gz` down into the running container.

## Decisions made along the way
- **Model Package Group renamed**: the group was recreated (empty, no data
  lost) from the generic `track2-yolo-models` to `track2-yolo11s-helmet`,
  since this line is specific to a yolo11s/helmet-detection model, not the
  original pipeline's generic candidates. Required updating the hardcoded
  name in `infra/sagemaker.tf`, `infra/iam.tf` (the review-app's IAM ARNs),
  `src/config.py`, `review-app/review_app.py`, and
  `scripts/deploy_review_app.sh` — all six places the old name appeared in
  live code (docs/READMEs left as historical narrative, not updated).

## Real bugs found along the way (not hypothetical — hit live, on the actual first real run of `deploy_endpoint.py`)
`deploy_endpoint.py` had never actually been executed before this session
(per the project's own status notes) — so this was the first real test of
it, and it surfaced three genuine platform/code issues in sequence:

1. **Serverless Inference rejects `DataCaptureConfig` entirely.** The
   original script combined `ServerlessConfig` with `DataCaptureConfig`
   (needed by `confidence_monitor.py`'s drift detection) — AWS rejects that
   combination outright. Fixed by dropping data capture; flagged that this
   means `confidence_monitor.py` has nothing to read for a
   serverless-deployed endpoint, a genuine capability tradeoff, not a
   config bug.
2. **Serverless Inference caps container images at 10GB.** This stack's
   serving image (`track2-yolo-train:serve-latest`) is ~14.57GB — a hard
   platform limit, not fixable by config. This forced abandoning serverless
   entirely.
3. **Switched to Asynchronous Inference (instance-based, `ml.g4dn.xlarge`)**
   instead — the same architecture already used for InstructIR/SwinIR/
   Real-ESRGAN in the `JURONG PORT` and `real-esrgan-sagemaker` projects.
   Async Inference has no image-size cap, and is the one endpoint
   architecture that supports scaling to 0 instances when idle (confirmed
   multiple times this session: classic real-time endpoints are hard-floored
   at 1 instance, Application Auto Scaling rejects `min_capacity=0` for
   them). Idle window set to **15 minutes** (user's explicit choice — a
   shorter window than the 30 minutes used elsewhere, accepting some risk of
   more frequent cold starts against a ~14.57GB image in exchange for
   tighter cost control).
4. **Fixed-name resource collisions, twice**: the script originally used
   fixed (non-timestamped) names for both the `Model` and `EndpointConfig`.
   Since both are immutable in SageMaker, the very first retry (after the
   serverless failure) collided with the earlier attempt's leftover
   `EndpointConfig`. Fixed by timestamp-suffixing both names on every run
   (same pattern already used in `real-esrgan-sagemaker/deploy.py`).
5. **`UpdateEndpoint` cannot recover a `Failed` endpoint** — confirmed via a
   live `ValidationException`: "Cannot update failed endpoint ... Please
   delete the endpoint and create a new endpoint." The script's original
   "update if exists, else create" logic assumed `UpdateEndpoint` always
   works on an existing endpoint; a `Failed` one specifically requires
   `DeleteEndpoint` + `CreateEndpoint` instead. Handled manually this run
   (delete stale failed endpoint, then `CreateEndpoint` reusing the
   `Model`/`EndpointConfig` that had already been created successfully,
   rather than creating yet another duplicate pair).

## Cleanup performed alongside this work
At the user's explicit, twice-confirmed instruction, also removed from
SageMaker: the separate real `track2-plane-models` (3 registered packages)
and `track2-ppe-models` (empty) Model Package Groups belonging to the
unrelated, already-applied `SNAIC-JP/Track2` infra; two stale
pipeline-created `Model` resources
(`pipelines-lezk75ridl6f-CreateModelStep-...`,
`pipelines-6pi33y45c8jt-CreateModelStep-...`); and — knowingly, since this
breaks their next invocation until a `Model` is recreated for each — the
live `jurong-port-swinir-jl` and `jurong-port-instructir-jl` `Model`
resources backing this session's SwinIR/InstructIR endpoints.

## Update: endpoint deployed, then renamed again (still 2026-08-31)

`track2-yolo-endpoint` reached `InService` (Async Inference, `ml.g4dn.xlarge`,
1 instance) — but hit two more real errors getting there, both fixed:

- **`UpdateEndpoint` cannot recover a `Failed` endpoint at all.** Confirmed
  via a live `ValidationException`: "Cannot update failed endpoint ...
  Please delete the endpoint and create a new endpoint." The very first
  retry attempt (after fixing the immutable-name collision below) tried
  `UpdateEndpoint` against the endpoint's prior `Failed` state (left over
  from the original serverless/image-size failure) and was rejected
  outright. Fixed by deleting the stale `Failed` endpoint first, then a
  plain `CreateEndpoint` — reusing the `Model`/`EndpointConfig` that had
  already been created successfully in the failed attempt, rather than
  creating yet another duplicate pair.
- **Fixed-name resource collision, again.** `deploy_endpoint.py` originally
  used fixed (non-timestamped) `MODEL_NAME`/`ENDPOINT_CONFIG_NAME`
  constants. Since both are immutable in SageMaker, the very first retry
  after the serverless failure collided with the earlier attempt's leftover
  `EndpointConfig`. Fixed properly this time: both names are now
  timestamp-suffixed on every run (matching `real-esrgan-sagemaker/deploy.py`'s
  established pattern), so re-running this script is now always safe.

### Further renames requested and completed
After the endpoint went live, two more names were changed:

1. **Model Package Group**: `track2-yolo11s-helmet` → **`track2-yolo11`**.
   Since SageMaker group names are immutable and the group wasn't empty
   (held the one approved model package), this required: capture the
   existing package's `ModelDataUrl`/image/metadata → delete that package →
   `terraform apply` to replace the group under the new name (Terraform
   showed `-/+ replace`, 0 data loss since it was emptied first) → 
   re-register a fresh model package (`track2-yolo11/1`) with identical
   data, pre-`Approved`.
2. **Model resource**: `track2-yolo-model-<timestamp>` → **`track2-yolo11-endpoint`**
   (SageMaker `Model`s are also immutable — no in-place rename exists).
   Standard blue/green swap: created the new `Model` pointing at
   `track2-yolo11/1`, a new `EndpointConfig` referencing it, then
   `UpdateEndpoint` against the *live* (`InService`, not `Failed`) endpoint
   — this time `UpdateEndpoint` worked correctly, confirming the earlier
   failure really was specific to updating a `Failed` endpoint, not a
   general limitation. Verified the swap via `describe-endpoint-config`
   before deleting the old `Model`/`EndpointConfig`.

Endpoint name itself (`track2-yolo-endpoint`) was deliberately left
unchanged — only the Model Package Group and Model resource were renamed,
per what was actually asked.

### Terraform drift discovered while checking "has Terraform been updated for all this?"
Checking live state against `terraform plan` surfaced two real gaps that
had nothing to do with the endpoint/Model renames above (those were never
Terraform's concern — they're `deploy_endpoint.py`/CLI-managed by design):

- **`review_app_access` IAM policy was stale** — its `ModelRegistryDescribe`
  statement still referenced the *original* `track2-yolo-models` name, never
  updated through either rename. Fixed in `infra/iam.tf`, pending apply.
- **Scale-to-zero autoscaling had never actually been applied at all** —
  despite `infra/autoscaling.tf` being written and `terraform validate`d
  earlier, none of its 5 resources (target, 2 policies, 2 alarms) existed
  in AWS yet. `track2-yolo-endpoint` has been running with **zero**
  scale-to-zero behavior this whole time.

Both fixes are one `terraform apply -target=...` away (command given to the
user) but require interactive `yes` confirmation the automation couldn't
supply — **not yet applied as of this write-up.**

## Status as of end of this write-up
- `track2-yolo11` Model Package Group: **live**, one `Approved` model
  package (`/1`) pointing at the packaged weights.
- `track2-yolo-endpoint`: **`InService`**, 1 instance, serving via
  `track2-yolo11-endpoint-config` → Model `track2-yolo11-endpoint`. All
  stale pre-rename Model/EndpointConfig resources cleaned up.
- Scale-to-zero autoscaling and the `review_app_access` IAM fix: **written,
  not yet applied** — needs `terraform apply` run interactively (see command
  above).
- `jurong-port-swinir-endpoint-jl` / `jurong-port-instructir-endpoint-jl`:
  still show `InService` but their backing `Model` resources were deleted —
  **their next invocation will fail** until a `Model` is recreated for each.

## Update: the training pipeline actually ran end-to-end, for the first time ever (2026-09-01)

`track2-yolo-training-pipeline` (TrainYolo -> EvaluateYolo -> RegisterYoloModel)
had never once completed successfully before today, per this project's own
prior status notes. Getting it to actually run surfaced a chain of real,
previously-undiscovered bugs, fixed one at a time as each was hit live:

1. **`f"{evaluation_s3_uri}/evaluation.json"` on a `PipelineVariable`** —
   SageMaker SDK deliberately blocks `__str__()` on these (resolved only at
   execution time) to catch exactly this mistake. Fixed with the SDK's
   `Join(on="/", values=[...])`.
2. **`ModelMetrics(model_quality=...)`** — that parameter doesn't exist in
   installed `sagemaker==2.257.6` (confirmed via `inspect.signature`); the
   real one is `model_statistics` (maps to the API's `ModelQuality.Statistics`
   field underneath).
3. **`ProcessingStep(code="evaluate.py")`** — a relative path resolved
   against the *current working directory at invocation time*, not
   `pipeline_definition.py`'s location; broke because `evaluate.py` lives in
   the sibling `training-container/` dir, not `src/` (where `run_pipeline.py`
   is meant to be run from). Fixed with an absolute path.
4. **That absolute path itself then broke on Windows** — a plain
   `C:\...` string got misread by the SDK's own `urlparse()`-based
   scheme-sniffing as if `C` were a URL scheme ("url scheme c is not
   recognized"). Fixed using `Path(...).as_uri()` (`file:///C:/...`), one of
   the two schemes (`file`, `s3`) the SDK's own code explicitly handles.
5. **Stale IAM policies, discovered by breaking a real (billed) training
   run**: `track2-yolo-sagemaker-execution`'s S3 policy still granted access
   to the deleted `track2-yolo-pipeline-125192671932` bucket, not `track2-s3`
   — despite `iam.tf`'s *source* already using a dynamic
   `aws_s3_bucket.pipeline.arn` reference, that specific policy resource had
   never actually been re-applied since the bucket rename (the Terraform
   *data source* recomputes correctly on every plan; the *resource* pushed
   to AWS does not, until something actually applies it). First training
   attempt burned real GPU minutes failing on `AccessDenied` uploading
   profiler output before this was caught and fixed.
6. **The same staleness pattern, again, blocking the container rebuild**:
   `track2-yolo-codebuild`'s S3 read policy was also still pointed at the
   deleted bucket, and the CodeBuild project's own `source.location` (an
   `aws_s3_object` uploaded from a local `training-container-src.zip`) was
   both stale-pathed *and* stale-content (predated all of today's `train.py`
   changes) — because **`train.py` is baked into the training container
   image at build time**, editing the local file alone did nothing; the
   image had to be rebuilt and repushed via the `track2-yolo-train-images`
   CodeBuild project before a training job would ever see the new code.
   Regenerated the zip, fixed the CodeBuild project's source, rebuilt.
7. **Swept the rest of the project's IAM policies for the same class of
   drift** rather than fix these one at a time forever — found and fixed 4
   more (`codebuild`, `codebuild_run_pipeline`, `confidence_monitor_lambda`,
   `review_trigger_lambda`). **Caught a near-miss doing this**: including
   `review_host_scheduler` in the same targeted apply would have silently
   destroyed and recreated the live review-host EC2 instance as an
   unrelated side effect (its policy depends on the instance's ARN, and an
   unrelated AMI drift had made Terraform want to replace it) — excluded it;
   it doesn't reference S3 at all, so it never needed touching for this fix.
8. **`EvaluateYolo` failed on a real, separate AWS account quota**:
   SageMaker *Processing Jobs* have their own independent instance quota
   from Training Jobs and Endpoints — `ml.g4dn.xlarge for processing job
   usage` was 0 in this account. Rather than request a quota increase,
   switched the evaluator to `ml.m5.xlarge` (CPU) — `evaluate.py` only runs
   Ultralytics' `model.val()` (inference, no backprop) on 17 validation
   images, entirely adequate on CPU, and that instance type already had
   real headroom (16).

**Result**: full pipeline succeeded, `TrainYolo` (`ml.g4dn.xlarge`, 280
billable seconds) -> `EvaluateYolo` (`ml.m5.xlarge`) -> `RegisterYoloModel`,
producing genuinely new registry entry `track2-yolo11/2`
(`PendingManualApproval`) with real attached metrics:
`mAP50=0.744, mAP50-95=0.419, precision=0.570, recall=0.753` — a fine-tune of
the `yolo11s-demo` base model on the real 137/17 train/val hazard data, 20
epochs. Not yet approved as of this write-up (pending a decision on whether
these numbers are good enough).

### Note on compute: training/evaluation instances are fully separate from the live endpoint
Worth documenting since it came up and isn't obvious: `TrainYolo` and
`EvaluateYolo` each provision their own brand-new, fully ephemeral SageMaker
instance (a Training Job instance, a Processing Job instance respectively) —
auto-terminated by SageMaker itself the moment the job completes, billing
stops immediately and permanently at that point
(`BillableTimeInSeconds` is a final, fixed number once `Completed`). This is
**completely separate compute** from `track2-yolo-endpoint`'s own persistent
`ml.g4dn.xlarge` instance, which stays up (or idles to 0 per the 15-minute
scale-to-zero setup) independent of anything happening in the pipeline. The
two only connect at the data layer, never the compute layer: training reads
its base weights from the same registry the endpoint happens to be serving
from, and a newly trained+approved candidate lands in that same registry —
but the live endpoint never picks up a new model automatically; that still
requires an explicit `deploy_endpoint.py` re-run.

## Update: fine-tuning from the Model Registry, a serious incident, and an approval-notification Lambda (still 2026-08-31)

### Training pipeline changed to require a registered base model, no fallback
At explicit request, `terraformer_track_2/training-container/train.py` and
`src/pipeline_definition.py` were changed so the training pipeline **only**
fine-tunes from a model already in the SageMaker Model Registry — never
from a bare Ultralytics checkpoint name (`yolov8n.pt` etc.), with no
fallback path at all:

- `pipeline_definition.py` now calls `list_model_packages(..., ModelApprovalStatus="Approved", SortBy=CreationTime, Descending, MaxResults=1)` on `CONFIG.model_package_group_name` at
  pipeline-build time, and **raises before any Estimator/TrainingStep is
  built** if the group has no `Approved` package — so a misconfigured
  registry never wastes GPU billing on a job that would fail anyway.
- `train.py`'s `--base-model` arg was removed entirely (not deprioritized —
  removed) and replaced with a required `--base-model-package-arn`. A new
  `resolve_base_model_checkpoint()` does `describe_model_package` → checks
  `ModelApprovalStatus == "Approved"` (fails loud if not, repeating the
  pipeline's own check as defense-in-depth in case the script is ever run
  directly) → downloads the package's `ModelDataUrl` tarball → extracts
  `model.pt` → that becomes `YOLO(...)`'s starting checkpoint.
- Ran `seed_training_data.py` again at request (idempotent — same seeded
  80/20 split, 275 real images / synthetic boxes) before this was tested
  further; training itself was not run this session.

### Serious incident: permanently deleted 17 files belonging to an unrelated project
While renaming `track2-yolo-pipeline-125192671932` → `track2-s3` (S3 has no
rename API — this was always going to be a delete-and-recreate under a new
name, per explicit instruction after being warned; `track2-S3` was rejected
first as an invalid bucket name, uppercase isn't allowed), the bucket was
emptied (1690 object versions deleted, since versioning was enabled and a
non-empty bucket can't be deleted) **before checking what was actually in
it broken down by prefix.**

**The mistake**: 17 of those 1690 objects belonged to
`track1_vlm_verdicts/` — real, timestamped VLM-evaluation run outputs
(e.g. `20260827-043308-...-few-shot-(filter)--cropped.jsonl`, dated Aug 7
through Aug 27) from a **completely separate, unrelated Track1 PPE/VLM
pipeline** that was apparently sharing this bucket (it even had its own
lifecycle rule, `expire-noncurrent-ppe-evaluation-exports`, managing that
exact prefix, set up entirely outside this project's Terraform — that's
how it surfaced, in a `terraform plan` diff, only after the deletion had
already happened).

**This was not recoverable.** The delete specified explicit `VersionId`s,
which is a permanent S3 operation, not a soft-delete — confirmed the
bucket was left with 0 remaining versions/delete markers. No AWS Backup or
cross-region replication was configured on this bucket. Searched the
entire `c:\SNAIC` workspace for local copies of the deleted filenames and
for any equivalent output/results directories in the Track1-related
projects — found nothing. **The 17 files are permanently gone.** Disclosed
in full to the user immediately on discovery, before any further
destructive action; the user was asked whether Track1's pipeline might
still be actively using this bucket (unanswered) and explicitly confirmed
to proceed with the delete-and-recreate regardless.

**Lesson, stated plainly for next time**: before emptying or deleting any
bucket, inventory its contents by top-level prefix first and check whether
any of them look like they belong to a different project — do not assume
a bucket belongs solely to the project whose Terraform manages it.

### Bucket successfully replaced, but this broke the live endpoint (found and fixed)
`infra/s3.tf`'s `aws_s3_bucket.pipeline` (and its versioning/public-access-block/
encryption sub-resources) were hardcoded to `"track2-s3"` (decoupled from
the `${var.project}-pipeline-${var.account_id}` convention, same technique
as the earlier Model/Model Package Group renames) and applied via
`terraform apply -target=...` (run by the user — same non-interactive
`terraform apply` limitation as every other apply this session). Confirmed
live: `track2-s3` exists, `track2-yolo-pipeline-125192671932` returns 404.

This silently broke the **live** `track2-yolo-endpoint`: its
`AsyncInferenceConfig` (`S3OutputPath`/`S3FailurePath`) was baked in at
endpoint-config-creation time, still pointing at the now-deleted bucket —
any invocation from that point on would have failed to write its result.
Found by checking `describe-endpoint-config` directly rather than assuming
the config.py fix alone was sufficient. Fixed with the same blue/green
pattern as the earlier renames: new EndpointConfig
(`track2-yolo11-endpoint-config-v2`) with corrected `s3://track2-s3/async-output/`
and `.../async-failure/` paths, `UpdateEndpoint`, confirmed back to
`InService`. `src/config.py`'s `pipeline_bucket` was also updated to
`"track2-s3"` (the only other hardcoded reference to the old name in live
code — `brainstorming.md`'s mention left as historical narrative).

### New feature: auto-dump approved weights to S3 + email stakeholders
Built at explicit request, mirroring the existing `confidence_monitor.py`/
`trigger_review.py` Lambda+EventBridge pattern already in this project:

- **New Lambda**: `infra/lambda_src/dump_approved_weights.py`. Triggered by
  SageMaker's own built-in EventBridge event (`source: aws.sagemaker`,
  `detail-type: "SageMaker Model Package State Change"`) — no polling,
  fires natively whenever `approve_model.py`'s `UpdateModelPackage` call
  changes a package's status.
- **EventBridge rule** (`aws_cloudwatch_event_rule.model_package_approved`)
  filtered to `detail.ModelApprovalStatus: ["Approved"]` and
  `detail.ModelPackageGroupName: ["track2-yolo11"]` — scoped to just this
  project's group so approvals elsewhere in the account don't trigger it.
  Note: the `ModelPackageGroupName` filter field assumes AWS's documented
  event schema; not yet confirmed against a real fired event in this
  account.
- **On every approval**: downloads the approved package's `model.tar.gz`
  (via `describe_model_package` → `ModelDataUrl`, same pattern as
  `train.py`'s `resolve_base_model_checkpoint()`), extracts `model.pt`,
  uploads to the fixed location `s3://track2-s3/latest_approved_model_weights/model.pt`.
- **Archiving, not overwriting**: if weights already exist at that fixed
  location, they're first copied (server-side `copy_object`, no
  re-download needed) to
  `s3://track2-s3/latest_approved_model_weights/archive/model_<dd-Mon-yy>.pt`
  (e.g. `model_31-Aug-26.pt`) — today's date, marking when they were
  superseded — before the new weights are uploaded.
- **Email notification**: reuses the existing `reviewer_notifications` SNS
  topic (already has a confirmed email subscription from earlier in this
  project) rather than standing up a second topic — publishes the model
  package ARN, source/destination S3 locations, the archive path (if any),
  and any attached evaluation metrics.
- New dedicated IAM role (`track2-yolo-dump-approved-weights-lambda`),
  scoped to: S3 read/write on `track2-s3`, `sagemaker:DescribeModelPackage`
  scoped to `track2-yolo11/*`, and `sns:Publish` scoped to
  `reviewer_notifications`.
- `terraform validate` clean, targeted `terraform plan` clean (7 to add, 0
  change, 0 destroy) — **apply command handed to the user, not yet
  confirmed applied as of this write-up.**

## Where to resume from next

1. **Apply the pending Terraform** — the `dump_approved_weights` Lambda
   +EventBridge rule + IAM (7 resources) is written and validated but not
   yet applied:
   ```
   cd terraformer_track_2\infra
   terraform apply -var="operator_ip=103.11.48.56" -target=aws_iam_role.dump_approved_weights_lambda -target=aws_iam_policy.dump_approved_weights_lambda -target=aws_iam_role_policy_attachment.dump_approved_weights_lambda -target=aws_lambda_function.dump_approved_weights -target=aws_cloudwatch_event_rule.model_package_approved -target=aws_cloudwatch_event_target.dump_approved_weights -target=aws_lambda_permission.allow_eventbridge_model_approval
   ```
   Once applied, verify end-to-end by approving a model package and
   checking CloudWatch Logs for `track2-yolo-dump-approved-weights` +
   whether the email actually arrives (the `ModelPackageGroupName` event
   filter is unverified against a real fired event).
2. **Fine-tune-from-registry code is written but never actually run** —
   `train.py`/`pipeline_definition.py`'s "always pull base model from
   `track2-yolo11`'s latest Approved package, error if none" logic hasn't
   been exercised against a real `run_pipeline.py` execution yet.
3. **Known, separate outstanding issue from earlier this session**:
   `jurong-port-swinir-endpoint-jl` and `jurong-port-instructir-endpoint-jl`
   still show `InService` but have no backing `Model` resource (deleted
   earlier per explicit instruction) — their next invocation will fail
   until a `Model` is recreated for each. Not touched during this update.
4. Track2's autoscaling (15-min idle scale-to-zero) and the
   `review_app_access` IAM fix from the earlier rename round — confirm
   these were actually applied (they were handed to the user as a separate
   `terraform apply -target=...` command earlier and not explicitly
   re-verified since).

## Update: SwinIR/InstructIR removed, Real-ESRGAN decommissioned, DynamoDB experiment rolled back, Track2 pipeline merged into this repo and pushed to GitHub (2026-09-06)

### `gp-jurong-port-s3-jl` bucket is now permanently gone
Confirmed deleted (via `aws s3api head-bucket`) at the user's own action
("i delete it"), and left alone per explicit instruction ("leave that
bucket alone") — no recreation has been attempted. **This makes several
S3 paths referenced earlier in this file stale**: the `yolo11s-weights/`
checkpoint, the packaged `model.tar.gz`, and the 154-label/zero-image
6-class dataset under `Track_2/Training/Testing_Dataset/...` all lived in
this bucket and are gone with it. Any future work resuming items 2–3 in
the "Where to resume" list above will need to re-source that data, not
just re-run the existing scripts against it.

### SwinIR / InstructIR fully removed from `JURONG PORT`
At explicit request ("remove the swineir and the instructir", "anything
related to instrustir and swineir need to be removed"): deleted
`autoscaling.tf`, `ecr.tf`, `sagemaker.tf`, `outputs.tf`, `codebuild.tf` +
`buildspec.yml` + `codebuild_trigger.ps1` + `build_and_push.sh`; trimmed
`iam.tf` (removed the AsyncInference + ECR-pull statements); deleted
`models/swinir/`, `models/instructir/`, both `invoke_*.py` scripts,
`webui/`, `tests/test_server.py`; rewrote `variables.tf`/`.tfvars`/
`.example`, `requirements.txt`, `README.md` to be model-agnostic; made
`common/server.py` require a `MODEL_TYPE` env var and dynamically
`importlib.import_module` the matching `models.<type>.inference` module
instead of hardcoding SwinIR. This — combined with the bucket deletion
above — supersedes the older "pending" note about
`jurong-port-swinir-endpoint-jl`/`jurong-port-instructir-endpoint-jl`
having no backing `Model`: both endpoints' entire dependency chain (code,
IAM, and now the S3 bucket their weights lived in) is gone, not just the
`Model` resource.

### Real-ESRGAN project decommissioned (separate project, `real-esrgan-sagemaker`)
At explicit request, deleted the live `real-esrgan-endpoint-jl` SageMaker
endpoint, the `real-esrgan-sagemaker` ECR repository (force-deleted, had
images), and the `real-esrgan-image-builder` CodeBuild project. Removed
the matching Terraform (`autoscaling.tf`, `codebuild.tf`, `ecr.tf`
deleted outright). This project's model artifacts also lived in
`gp-jurong-port-s3-jl`, so it's unrecoverable the same way
SwinIR/InstructIR are. (The remaining execution-role IAM was later fully
torn down too — see the 2026-09-06 cleanup entry below.)

### DynamoDB experiment: tried, then fully rolled back
Briefly stood up DynamoDB table(s) + IAM to hold some JSON files
("put tall the json file into a data base for me... spin up the
necessary weight infrature"), then the user reversed course
("roll back all these data") — everything from that experiment was torn
down again; nothing from it persists in AWS or in Terraform state.

### Terraformer Track2 YOLO pipeline migrated into `JURONG PORT` and pushed to GitHub
At explicit request ("can u migrate the terraformer track 2 yolo pipeline
to the jurong port folder aftr hwich do a push to git hub"), merged
`terraformer_track_2`'s entire `terraform/` tree into this repo:

- Verified live AWS state for both projects before changing anything.
- Found and resolved the one real Terraform address collision:
  `aws_iam_role.sagemaker_execution` existed in **both** projects'
  state, pointing at two different real IAM roles
  (`jurong-port-sagemaker-execution-role-jl` vs
  `track2-yolo-sagemaker-execution`). Resolved via `terraform state mv`
  (renaming Track2's copy to `jurong_port_sagemaker_execution` and its
  two role-policy attachments to match) — no resource was destroyed or
  recreated.
- New file `terraform/infra/jurong_port.tf` holds Track2's renamed IAM
  role/policies plus its own S3 bucket resource and variables
  (`jurong_port_*` prefixed).
- Physically moved all of Track2's `terraform/` content —
  `infra/`, `src/`, `training-container/`, `scripts/`, `review-app/`,
  `docs/`, both source zips, root docs (`README.md`,
  `TERRAFORM_STATE.md`, `brainstorming.md`) — into this repo's
  `terraform/` folder. Per explicit instruction ("leave the git hirtory
  behind"), Track2's `.git` folder was left in place at its original
  location; `terraformer_track_2/terraform/` on disk now contains
  nothing but that orphaned `.git`.
- `terraform plan` before and after the physical move showed **identical
  output** (only pre-existing, unrelated drift — the still-not-recreated
  `gp-jurong-port-s3-jl` bucket, plus 2 unrelated changes) — confirms the
  merge introduced zero new drift.
- Caught a real gap before committing: 4 numbered
  `terraform.tfstate.<timestamp>.backup` files leaked into `git add -A`
  staging because the moved-in nested `.gitignore` only had an
  exact-match pattern, not a wildcard, for backup files — fixed the
  pattern to a bare `terraform.tfstate.*` and deleted the redundant
  backups before staging again.
- Committed (58 files changed) and pushed —
  `https://github.com/jacob-data-engineer/jurong_port`, commit `017eaba`.

### Where to resume from next (supersedes the list above — re-check anything from before 2026-09-06)
1. **`terraform/infra/` under `JURONG PORT` is now the single source of
   truth** for both the Track2 YOLO pipeline and Jurong Port's own
   scaffolding — any future `terraform apply` for either runs from here,
   not from `terraformer_track_2` (which no longer has a working tree).
2. The pending `dump_approved_weights` Lambda + EventBridge Terraform (7
   resources, validated but never applied) and the autoscaling/
   `review_app_access` IAM fixes from the earlier rename round have not
   been re-verified since the migration — confirm their current state
   before assuming they're still simply "pending" in the merged repo.
3. `gp-jurong-port-s3-jl` bucket recreation remains explicitly un-applied
   ("leave that bucket alone").
4. `terraformer_track_2/terraform/.git` is an intentionally orphaned repo
   with no working-tree content — no action requested on it.
5. `real-esrgan-sagemaker` is not a git repo and has not been asked to be
   pushed anywhere.

## Update: dead bucket references fully removed (2026-09-06, later same day)

At explicit request, removed every remaining reference to the deleted
`gp-jurong-port-s3-jl` bucket that was still "not used anymore" — both in
live Terraform config and in this file's own narrative:

- **`JURONG PORT/terraform/infra/iam.tf`**: removed the
  `Yolo11sDemoWeightsRead` statement from `sagemaker_s3_scoped` (the
  Track2 SageMaker execution role no longer grants any access to the
  dead bucket).
- **`JURONG PORT/terraform/infra/monitoring.tf`**: removed the
  `ExternalModelDataRead` statement from `dump_approved_weights_lambda`'s
  policy (same dead-bucket grant, different role).
- **`real-esrgan-sagemaker`**: its remaining orphaned
  `aws_iam_role.sagemaker_execution` + `aws_iam_role_policy.s3_model_read`
  + `aws_iam_role_policy.cloudwatch_logging` (all that was left after the
  endpoint/ECR/CodeBuild were already deleted, kept alive only to grant
  access to the dead bucket) were destroyed via `terraform apply`
  (confirmed: 0 added, 0 changed, 3 destroyed). `iam.tf` was then deleted
  outright and `variables.tf` trimmed back to just `aws_region`,
  `project_name`, `tags` — otherwise a future `terraform apply` would
  have silently recreated the same dead-end role. `terraform plan`
  confirms 0 resources now tracked besides the two data sources; matches
  configuration exactly.
- **This file**: trimmed the stale `gp-jurong-port-s3-jl` path references
  out of the "Starting point", "Decisions made along the way", the
  dataset-ingestion narrative, and the `dump_approved_weights` IAM
  description above, and renumbered the two "Where to resume" lists
  accordingly.

## Update: Model Registry reorganized, review-app fixes, and live dataset-download progress (2026-09-07)

### Model Registry reorganized to a uniform naming scheme
Replaced the old irregular per-size group names with uniform
`yolo11<size>-detector` naming, and cleared out stale/experimental
history:
- Deleted all packages + groups: `track2-yolo11` (12 versions, 4
  Approved — the original, pre-per-size-groups history), `track2-yolo11m`,
  `-l`, `-x` (1 Approved each), the never-populated `track2-yolo8n`, and
  the separate `experimental-full-frame-yolo11s` (19 test packages,
  unrelated to the generation/size flow).
- Created 5 fresh groups (`yolo11n/s/m/l/x-detector`) via
  `terraform/infra/sagemaker.tf`'s `aws_sagemaker_model_package_group.yolo`
  (`for_each` over `local.yolo_size_group_names`), each seeded with one
  freshly-downloaded pretrained COCO baseline via
  `model_training/src/register_baseline_model.py` (now accepts
  `--generation` and `--approve`, and a `REGISTERABLE_SIZES` including
  Nano that's intentionally separate from `config.py`'s `SIZES`/
  `GENERATION_SIZES`). Registered **Approved** (not the usual
  `PendingManualApproval`) so training works immediately without a manual
  approval step.
- `yolo11n-detector` was later deleted (down to 4 groups: s/m/l/x) — via
  the **AWS Console**, not by any script here (confirmed via CloudTrail:
  a browser session assuming the SageMaker Studio execution role, from
  the same operator IP, at 2026-09-07T16:53).
- Code updated to match: `model_training/src/config.py` gained a single
  uniform `model_package_group_name(generation, size)` function,
  replacing the old `model_package_group_names` dict and its irregular
  generation-11-special-case in `pipeline_definition.py`'s
  `_resolve_model_package_group_name`. `approve_model.py`/
  `reject_model.py`/`register_baseline_model.py` all updated to call it.
  `terraform/review-app/review_app.py`'s `GENERATION_SIZES` narrowed to
  YOLO11 only (12 and 26 were briefly offered, then removed — 26 isn't a
  real Ultralytics release yet, confirmed against `GITHUB_ASSETS_NAMES`;
  12 was pulled too since nothing's registered for it, same "guaranteed
  failure on Start Training" reasoning).
- Real production risk caught before it happened: an untargeted
  `terraform plan` during this work tried to overwrite the **live**
  `track2-yolo-review-host` security group's actual current operator IPs
  (`103.4.232.199`, `103.30.215.143` — discovered via that same plan, not
  guessed) with a stale placeholder value, and separately surfaced an
  unrelated pending `gp-jurong-port-s3-jl` bucket recreation that isn't
  wanted. Every apply this session used precise `-target`ing of just the
  resources actually intended, verified via plan output each time.
- Every "rebuild `model_training/run-pipeline-src.zip`, apply via
  targeted `terraform apply -target=aws_s3_object.run_pipeline_source`"
  round from earlier sessions repeated correctly multiple times this
  session — that staleness gotcha (editing the `.py` source alone does
  nothing until the zip is rebuilt) is now a well-worn, reliable
  procedure, not just a one-off lesson.

### Review app (`terraform/review-app/review_app.py`) fixes and additions
- **Fixed the actual "nothing happens" bug**: the "Start Training" button
  was `disabled` by default on every page load — it silently required a
  separate S3-folder-browser selection before it would do anything at
  all. Now defaults `selected_dataset` to `DEFAULT_TRAIN_DATASET`
  (`data/track2-datasets/sagemaker_training/track2/escalation_dataset`,
  matching `pipeline_definition.py`'s own default), so the button works
  out of the box.
- Model Registry panel (Review Queue tab) now shows **all four** size
  groups with their own section/counts, not just one hardcoded group —
  `_model_registry_summary()` takes a `group_name` param,
  `_all_model_registries()` builds all four. Removed the now-dead
  `MODEL_GROUP`/`MODEL_PACKAGE_GROUP` env var.
- Registry rows now show each version's **Description** (the training
  form's "Notes" field, already saved as the real
  `ModelPackageDescription` via `pipeline_definition.py`'s
  `model.register(description=train_notes, ...)` — previously captured
  but never displayed anywhere in the app).
- Added a **Delete** button on every registry row (any status), gated
  behind a JS `confirm()` — new `sagemaker:DeleteModelPackage` IAM
  permission added to the existing scoped `ModelRegistryDescribeAndApprove`
  statement (renamed `ModelRegistryDescribeApproveAndDelete`), still
  scoped to just the 5 `yolo11*-detector` groups, not `*`.
- Removed the "Pending Approval" flagged-predictions section and the
  "Reviewed (approved/rejected)" stat card (both part of a separate,
  unused confidence-monitor review-queue feature) at explicit request.
  Reworded "Registered Models" to "Registered Model Versions" after
  confirming the count (then 5) was correct but ambiguous — it sums
  versions across all groups, not group count.

### Live dataset-download progress bar
SageMaker's own automatic input-channel download gives **no percentage
at all** — confirmed via a real `DescribeTrainingJob`'s
`SecondaryStatusTransitions`: image pull and dataset download are
bundled into one opaque "Downloading" phase with a status message and
timestamps, nothing more. Getting a real percentage required moving
dataset delivery out of SageMaker's managed channel entirely:
- **`model_training/training-container/train.py`**: new
  `download_dataset_with_progress()` — lists every object under the
  dataset's S3 prefix, sums total bytes, then downloads each file via
  `s3.download_file(..., Callback=...)`, printing `DATASET_DOWNLOAD: NN%`
  as it goes; rewrites the downloaded `data.yaml`'s `path:` key to the
  local destination dir. Runs before `resolve_base_model_checkpoint*` and
  `model.train()`, both now inside one widened `tempfile.TemporaryDirectory`
  scope (the dataset must survive past the checkpoint-download's original
  narrower scope). `--train-dir`/`SM_CHANNEL_TRAIN` replaced with
  `--train-data-s3-uri`/`TRAIN_DATA_S3_URI`. Verified locally against the
  real 527-file dataset before any container rebuild: correct 0->100%
  progression, correct file counts (449 train + 85 val image/label
  pairs), correct `data.yaml` rewrite.
- **`model_training/src/pipeline_definition.py`**: `TrainingStep` no
  longer has an `inputs=` channel; `train-data-s3-uri` passed as a
  hyperparameter instead. Added a second `metric_definitions` regex,
  `dataset:download_pct` matching `DATASET_DOWNLOAD:\s+(\d+)%`, same
  log-scraping-into-CloudWatch-metric technique already used for
  `train:epoch`.
- **`review_app.py`**: new `_get_dataset_download_progress()` polls the
  `dataset:download_pct` metric the same way `_get_epoch_progress()`
  already polls `train:epoch`; `_render_training_progress()` shows the
  download bar only while epoch progress hasn't started yet (download
  happens first, chronologically, in `train.py`'s own `main()`).
- Both CodeBuild source zips rebuilt and applied:
  `run-pipeline-src.zip` (for `pipeline_definition.py`) and
  `training-container-src.zip` (for `train.py`, the actual Docker image
  source — a **different**, heavier CodeBuild project,
  `track2-yolo-train-images`, ~14 minutes wall clock, real base-image
  pulls + two `docker build`s + two `docker push`es). Triggered and
  confirmed **SUCCEEDED** (2026-09-07T17:44:07+08:00) — `train-latest`
  and `serve-latest` in ECR now both reflect the new `train.py`.
- Review app redeployed with the new progress-bar rendering.
- **Not yet done**: an actual end-to-end training run through the new
  container + pipeline code together has not been triggered yet — only
  the download function's logic was verified standalone (locally,
  outside the container). First real run through the full chain should
  be watched closely for anything the standalone test couldn't catch
  (container-specific paths, the widened-tempfile-scope refactor,
  hyperparameter plumbing).

### Discussed, not built: database for run-lineage tracking
Walked through (design only, nothing executed) whether `run_metadata.json`
(one file per run in `s3://track2-s3/training-runs/<run_id>/`, written
once at pipeline-build time — not live-updated during training) should
move to a real database. Recommendation if/when pursued: DynamoDB,
hybrid (keep the S3 JSON as the durable full record, add a lean indexed
table alongside — partition key `run_id`, a GSI on `model_package_arn`
for O(1) reverse lookup from a registered version back to its lineage,
matching the existing `track2-yolo-review-queue` DynamoDB precedent
already in this codebase) over Athena-over-S3 (zero write-path change,
but batch/analytical rather than live-UI-friendly) or a full swap to
DynamoDB-only (loses the plain-file human-readable audit trail the
current code explicitly values). Not started.

## Update: run-lineage now in DynamoDB, and the review app is publicly exposed (2026-09-07, later same day)

### Run lineage: DynamoDB built (the design from above, now implemented)
Reused the existing, previously-empty `track2-dynamoDB` table (hash key
`id`, created 2026-09-01, "purpose not yet documented by whoever created
it" — confirmed explicitly with the user that this, not a new table, is
what "the DynamoDB" meant) rather than creating a new one:
- **`pipeline_definition.py`**: `_write_run_metadata()` now writes the
  same record to both S3 (unchanged, still the durable full copy) and
  `track2-dynamoDB` (`Item={**record, "id": run_id, "record_type":
  "training_run"}`). Returns `(s3_uri, run_id)` now, not just the URI.
  `RegisterModel`'s `customer_metadata_properties` gained a `lineage_id`
  key (`= run_id`) alongside the pre-existing `run_metadata_uri`.
- **`register_baseline_model.py`**: matching `_write_baseline_lineage()`
  — same table, `record_type: "baseline"`, `id` prefixed
  `baseline-yolo<gen>-<size>-<timestamp>`. `CustomerMetadataProperties`
  gained the same `lineage_id` key, so review_app.py's lookup is uniform
  across both registration paths ("store all the information" meaning
  every path, not just pipeline-triggered runs).
- **IAM**: `codebuild_run_pipeline` role gained scoped `dynamodb:PutItem`
  on `track2-dynamoDB` (the role `_write_run_metadata()` actually runs
  under, at `build_pipeline()` time — not `sagemaker_execution`, same
  reasoning as its existing S3 `WriteRunMetadata` grant).
  `review_app_access` gained scoped `dynamodb:GetItem` on the same table.
- **`review_app.py`**: `_model_registry_summary()` now also pulls each
  package's `CustomerMetadataProperties.lineage_id` (from the same
  `describe_model_package` call already fetching the Description) and
  resolves it to a `get_item` — no extra API call shape needed, and no
  S3 round-trip for the common case. New `_format_lineage()` renders a
  short summary line (dataset + base model + hyperparameters for a
  training run; checkpoint + approval mode for a baseline), shown under
  the existing Description line on each registry row.
- **Verified end-to-end with real writes/reads, not just unit-level**:
  a throwaway `_write_run_metadata()` call confirmed the DynamoDB item's
  exact shape (nested maps, `N`/`S`/`NULL` types all correct) before
  wiring the review app up to it; a real `register_baseline_model.py
  --size m --approve` run then showed up correctly in the live UI
  ("baseline checkpoint: yolo11m.pt | auto-approved on registration"),
  which was then deleted via the review app's own Delete button (a real
  HTTP POST, not simulated) — first live use of that feature too. Test
  artifacts (the DynamoDB item, the S3 baseline weights, the model
  package version) all cleaned up afterward.

### Review app opened to the public internet — explicit, informed request
At explicit user request, after being clearly warned first: the review
app (port 8090) no longer restricts access to `operator_ips` at all.
**The app has no authentication of its own** — the IP allow-list was the
only access control. With this change, anyone with the URL
(`http://13.214.132.219:8090/`) can trigger real paid SageMaker training
jobs, approve/reject/delete any registered model version, and edit the
retrain schedule, with no login step.
- `terraform/infra/variables.tf`: new `review_app_public` bool, default
  `false`, documented in-line with exactly what turning it on exposes.
  Must be passed explicitly (`-var='review_app_public=true'`) — a future
  apply that doesn't pass it reverts to IP-restricted.
- `terraform/infra/ec2.tf`: the security group's ingress `cidr_blocks`
  is `["0.0.0.0/0"]` when `review_app_public` is true, the old
  per-operator-IP list otherwise. Confirmed live: `curl` from an
  unrelated IP now gets HTTP 200.
- Also re-confirmed (again — third time this project) that the
  operator's own IP had drifted since the last fix: was
  `103.11.48.56`/`119.56.17.252` two sessions ago, `103.4.232.199`/
  `103.30.215.143` this session (added `119.56.17.32` before the
  public-access request superseded the need for it). If IP-restricted
  access is ever restored, check the current real IP first
  (`curl https://checkip.amazonaws.com` from the operator's own machine)
  rather than assuming any previously-recorded value is still current.

### Where to resume from next (supersedes the list above)
1. **Trigger a real training run** through the new container +
   pipeline-definition code together, and watch the review app's new
   "Downloading dataset: NN%" bar live — confirms the whole chain (image
   rebuild, hyperparameter plumbing, metric regex, review-app polling)
   actually works end to end, not just each piece in isolation. This is
   also the first real opportunity to confirm a pipeline-triggered run's
   lineage record (not just a baseline's) shows up correctly in the UI.
2. `yolo11n-detector` is gone (deleted via the Console, not by any script
   here) — re-register it (`register_baseline_model.py --size n
   --approve`) if Nano's baseline should still exist for reference.
3. YOLO12 has no registered baseline (only YOLO11 does) — register one
   the same way as the others if/when YOLO12 should become selectable in
   the web page again.
4. **Security**: the review app is currently fully public (see above) —
   revisit whether/when to lock it back down.
5. None of this session's code changes have been committed to git yet.

## Update: real end-to-end training run + progress-bar root cause found (2026-09-07, later still)

### Making the app public had an immediate, real consequence
Within minutes of opening the review app publicly, three near-simultaneous
`StartBuild` calls (19:03, 19:04, 19:05 SGT) each tried to start a real
SageMaker pipeline execution. This account's **`ml.g4dn.xlarge` training
job quota is 1 instance** — the first execution
(`ip83wq0s0221`, 19:04:13) got the slot; the other two
(`e2vk4f8ygu75` at 19:05:35, and an earlier `bbrq63u1n223` at 16:59:38
that hit the same collision against a still-running 16:58:24 job) failed
immediately with `ResourceLimitExceeded` on `CreateTrainingJob` — **not a
code bug**, confirmed via `list_pipeline_execution_steps`' `FailureReason`
on both. Worth requesting a quota increase, or adding a "training already
running" guard to the Start Training button, before this app stays public
long-term.

### Dataset-download progress bar: real bug found via the live run
`ip83wq0s0221` (the execution that got the GPU slot) gave a chance to
verify the whole new-container feature end to end for the first time.
Good news: `train.py`'s own container logs confirmed the dataset really
did download with `DATASET_DOWNLOAD: 42%` through `100%` printed
correctly, then trained normally through several real epochs (mAP50
reaching 0.887 by epoch 1, 0.938 by epoch 3) — the download-and-train
logic itself works.

**But the CloudWatch metric had zero datapoints** despite those exact log
lines existing in raw CloudWatch Logs. Root cause, confirmed by comparing
against `train:epoch` (which DID have datapoints on the same job):
SageMaker's `metric_definitions` CloudWatch Logs metric filter takes a
few seconds to attach after a training job starts. `train:epoch` prints
continuously for minutes, so even if its first few lines are missed, later
ones get through. This dataset is small enough that the *entire*
`DATASET_DOWNLOAD` burst (0% to 100%) finished and was logged within
that initial attach-delay window — invisible to the metric even though
visible in the raw logs.

**Fix**: `train.py`'s `download_dataset_with_progress()` now
`time.sleep(20)` right after printing the file-count/size line, before
any per-file downloads (and therefore before any `DATASET_DOWNLOAD: NN%`
lines) start — buys the metric filter time to attach first. Rebuilt
`training-container-src.zip`, applied via targeted `terraform apply`,
triggered a fresh `track2-yolo-train-images` CodeBuild rebuild. Not yet
re-verified with another live run (queued as a follow-up).

### Three background agents dispatched for review_app.py UI work
At the user's explicit request, delegated further review-app UI work to
background agents rather than doing it all directly — **sequenced, not
parallel**, since `review_app.py` is a single file deployed as one
wholesale Docker image replacement to one live container: two agents
editing/deploying it concurrently would silently clobber each other.
1. **Lineage viewer tab** (dispatched first): scans `track2-dynamoDB` and
   lists every `training_run`/`baseline` record — a new
   `dynamodb:Scan` IAM grant added to the existing `ReadRunLineage`
   statement (same table, same resource, just the added action).
2. **Metrics viewer tab** (queued, not yet dispatched): waits for #1 to
   fully finish (edit + deploy) before starting, so it begins from the
   merged file rather than racing it.
Briefed both agents explicitly on this repo's terraform gotchas (always
`-target`, always pass both `operator_ips` and `review_app_public`
explicitly — neither has a default) and the two-zip-rebuild pattern.

**Update: Lineage tab completed and independently re-verified.** Agent
added `_scan_all_lineage()` (paginated `scan()` over `track2-dynamoDB`,
sorted newest-first) and a read-only `_render_lineage_rows`/
`_render_lineage_tab` pair, wired into the nav and `do_GET`'s tab
whitelist. Added `dynamodb:Scan` to `ReadRunLineage` in `iam.tf`, applied
with `-target=aws_iam_policy.review_app_access` (1 changed, nothing
else touched), deployed. The task carried an automated "blocked by
classifier" flag — reviewed the actual diff line-by-line before trusting
it: no new POST routes, every field goes through `html.escape()`, IAM
change is exactly the one added action on the one existing table
resource. Independently re-confirmed live via SSM (curl from inside the
instance, not just trusting the agent's own report): all four tabs
(`review`, `models`, `schedule`, `lineage`) return HTTP 200, lineage tab
shows 2 real `training_run` records correctly. Container logs show only
harmless `ConnectionResetError` noise from internet port scanners
hitting the now-public 8090 port — not an application error.

**Update: Metrics tab completed and independently re-verified.** Agent
added `_fetch_evaluation_metrics`, `_scan_all_model_metrics`, and a
read-only `_render_metrics_rows`/`_render_metrics_tab` pair, wired into
the nav and `do_GET`'s whitelist -- same pattern as the lineage tab.
Confirmed live: the S3 URI sits at
`ModelMetrics.ModelQuality.Statistics.S3Uri` on each package, matching
AWS docs, and baseline registrations genuinely carry no `ModelMetrics`
key at all (not hypothetical -- confirmed against the real v1 baseline
packages). **Found and fixed a real latent bug along the way**:
`pipeline_definition.py`'s `Join(on="/", values=[evaluation_s3_uri,
"evaluation.json"])` produces a double slash
(`evaluation-output//evaluation.json`) since `evaluation_s3_uri` already
ends in `/` -- S3 keys aren't URL-normalized, so that would have 404'd
against the real single-slash object and silently shown "no metrics"
for every pipeline-trained package. Worked around at read time with a
`re.sub(r"/+", "/", key)` in the new fetch function (did not touch
`pipeline_definition.py` itself -- reads are enough, and safer than
touching the write path). No IAM changes needed. Reviewed the diff
myself before trusting it (no new POST routes, every string field
`html.escape()`d) and independently re-verified live via SSM curl from
inside the instance (not just the agent's own report): all 5 tabs
return HTTP 200, and the metrics tab shows real rows -- `yolo11s-detector`
v3 (PendingManualApproval, real mAP/precision/recall numbers) plus the
four v1 baselines correctly showing "no metrics recorded". Container
logs clean, no new tracebacks.

All three background-agent deliverables the user asked for (progress
bar, lineage tab, metrics tab) are now complete and independently
verified live.

### Recurring "training run failed" investigated again, root cause identical, guard added
User reported another failed run from the Models tab screenshot
(execution `e2vk4f8ygu75`, started 19:05:35 SGT). Investigated (no
changes made until explicitly authorized): confirmed via
`list_pipeline_execution_steps`'s `FailureReason` that this was, again,
the account's `ml.g4dn.xlarge` quota (1 instance) -- the previous run's
training job (`ip83wq0s0221`, started 19:04:15) was still `InProgress`
(didn't finish until 19:12:41) when this one tried to start 80 seconds
later. Same root cause as the earlier quota-collision update, not a
regression from any of the tab work.

**Live progress-bar fix confirmation** (user approved a read-only
CloudWatch check against the separate, currently-running execution
`70tmdf5i1rnm`): confirmed via real log lines
(`DATASET_DOWNLOAD: 0%` through `48%` in CloudWatch Logs) AND a real
CloudWatch metric datapoint (`dataset:download_pct = 55.0` at
19:39:00 SGT) -- the `time.sleep(20)` timing fix works. Previously this
metric had zero datapoints on a real run despite matching log lines
existing; now it populates correctly.

**Added the already-running guard** (user approved): new
`_training_already_running_reason(progress)` in `review_app.py`, reusing
`_get_training_progress()`'s own status field (no new IAM, no new AWS
calls). Wired into:
- `do_POST`'s `/train/dataset` handler (Start Training) -- raises and
  redirects with the reason instead of calling `codebuild.start_build`.
- `do_POST`'s `/train/now` handler (Schedule tab's Train Now) -- same,
  before `sm.start_pipeline_execution`.
- Both buttons' rendering (`_render_models_tab`, `_render_schedule_tab`)
  -- disabled with an explanatory message when a run is `Executing`, as
  a UX nicety; the POST-side check is what actually enforces it, since a
  disabled attribute alone is trivially bypassable.
Deployed and independently verified live via SSM: both buttons render
`disabled` with "A training run (70tmdf5i1rnm) is already in progress"
while the live run was still executing, AND a direct POST straight to
`/train/now` (bypassing the disabled button entirely) was rejected
server-side with a 303 redirect carrying the same message -- confirming
the real protection, not just the cosmetic one. All 5 tabs still return
HTTP 200, logs clean.

**Training-container rebuild (the `time.sleep(20)` CloudWatch-timing
fix for the dataset-download progress bar) succeeded and was
subsequently confirmed live** on a real training run: `DATASET_DOWNLOAD`
log lines 0%-48% plus a real CloudWatch datapoint
(`dataset:download_pct = 55.0`) on job
`pipelines-70tmdf5i1rnm-TrainYolo-OQAs2yH5X1` -- previously this metric
had zero datapoints on a real run despite matching log lines existing.

### Epoch progress bar disappearing once Train succeeded (user-reported, screenshot)
User showed a screenshot with Train=Succeeded, Evaluate=Executing, and
no epoch bar visible -- it used to show "Epoch X/Y" during training but
vanished the instant TrainYolo finished. Root cause:
`_get_training_progress()` only computed `epoch_progress` while
`step_status["TrainYolo"] == "Executing"` -- the moment Train flips to
Succeeded, the bar's data source goes `None`, even though both of
`_get_epoch_progress`'s sources (the CloudWatch `train:epoch` metric and
the job's own hyperparameters) stay queryable long after the job
finishes. Fixed by widening the condition to
`in ("Executing", "Succeeded")`, keeping the download-progress fallback
gated to `Executing` only (irrelevant once training's done). Deployed
and verified live in the exact same state as the screenshot
(Train=Succeeded, Evaluate=Executing): bar now correctly shows
"Epoch 20 / 20" at 100% instead of disappearing. All 5 tabs still 200.

### Where to resume from next (supersedes the list above)
1. Consider adding a progress indicator for the Evaluate step too (user
   asked if this is possible) -- currently that stage just shows an
   "Executing" badge with no sub-progress, unlike Train's epoch/download
   bars. Would need evaluate.py to emit a scrapeable log line (e.g.
   validation batch N/M) plus a matching metric_definitions entry on
   the EvaluateYolo ProcessingStep's ScriptProcessor, mirroring exactly
   how train:epoch/dataset:download_pct already work for TrainYolo.
   Not yet actioned -- discuss approach with user first.
2. Consider also fixing the double-slash bug at its source in
   `pipeline_definition.py`'s `Join(...)` call (currently only patched
   at read time in the review app) so any other future consumer of that
   S3 URI doesn't hit the same 404.
3. Consider a training-job-quota increase request, given the public app
   already caused wasted/failed executions before the already-running
   guard was added (now fixed, see above).
4. `yolo11n-detector` still gone; YOLO12 still has no baseline (both
   from the previous update, still unresolved).
5. Review app is still fully public (deliberate, user-confirmed); still
   nothing committed to git.

### Dataset-download bar changed from a byte-% to a real "N / M images loaded" count (user request)
User asked specifically to see "the number of images that is being
loaded", not an abstract percentage. Reworked the whole mechanism rather
than relabeling the existing one:
- **`train.py`**: `download_dataset_with_progress` no longer tracks raw
  bytes via a boto3 download `Callback`. It now counts only files
  matching real image extensions (`IMAGE_EXTENSIONS` -- jpg/jpeg/png/
  bmp/tif/tiff/webp), printing `DATASET_TOTAL_IMAGES: N` once (after the
  same 20s CloudWatch-filter-attach delay as before) and
  `DATASET_IMAGES: n/N` after each image file finishes downloading.
  Label files/data.yaml are still downloaded, just not counted --
  they're not what "loading images" means to someone watching progress.
- **`pipeline_definition.py`**: `metric_definitions` swapped
  `dataset:download_pct` for two metrics --
  `dataset:total_images`/`dataset:images_downloaded` -- since a
  CloudWatch metric_definitions regex captures exactly one numeric
  group, the running count and its denominator need separate regexes.
- **`review_app.py`**: `_get_dataset_download_progress` replaced with
  `_get_dataset_images_progress` (returns `{"current", "total"}`, same
  shape as `_get_epoch_progress` so `_render_training_progress` treats
  both bars identically); extracted a shared `_get_metric_max` helper.
  Bar now reads "Loading images: n / N" instead of "Downloading
  dataset: NN%".
- Both CodeBuild source zips rebuilt and applied via targeted
  `terraform apply` (`-target=aws_s3_object.codebuild_source
  -target=aws_s3_object.run_pipeline_source`) -- **note**: the first
  `run-pipeline-src.zip` attempt accidentally swept up ~439MB of stray
  local baseline checkpoint files (`yolo11x.pt`, `*-baseline.tar.gz`,
  etc.) sitting in `model_training/src/` from earlier baseline
  registration work; rebuilt filtered to only `.py`/`.txt` before
  uploading (24.7KB, correct). `track2-yolo-train-images` CodeBuild
  rebuild triggered for the new `train.py`; review app redeployed
  immediately (independent of that image rebuild) and confirmed all 5
  tabs still return HTTP 200.
- **Not yet verified live**: the actual `DATASET_IMAGES`/
  `DATASET_TOTAL_IMAGES` metrics haven't been checked against a real
  training run yet -- needs the container rebuild to finish, then a
  fresh run.

### Two more feature requests queued (user), to be built as sequenced background agents
Same reasoning as the Lineage/Metrics tabs earlier: both touch the same
single-file `review_app.py` deployment, so they'll run one after the
other, not concurrently.
1. **Training run queue**: select dataset/model/settings and add up to
   10 runs to a queue (new section below Start Training), with its own
   "Start Queue" action, and visibility into which queued run is
   currently executing. Needs to integrate with the already-running
   guard (`_training_already_running_reason`) to auto-advance to the
   next queued run only once the current one finishes -- tricky part is
   this app has no persistent background scheduler (plain
   `http.server`), so advancing likely has to be driven by page
   load/poll rather than a true background worker.
2. **Metrics tab comparison**: select 2+ registered model versions and
   compare their evaluation metrics as a bar graph, instead of just the
   current flat list of rows.

**Training queue agent dispatched first** (metrics-comparison queued
behind it, same single-file/single-container reasoning as before).
Briefed to: add a "Training Queue" panel below the existing Start
Training form (cap 10 entries, remove-before-start, Start Queue button,
shows which entry is currently running); refactor `/train/dataset`'s
pipeline-submission body into a shared `_start_training_run(...)`
function so there's exactly one place that calls
`codebuild.start_build`; use a daemon background `threading.Thread`
(the server itself is a plain single-threaded `HTTPServer`, but the
process can run extra threads fine) polling every ~20-30s, gated by the
same `_training_already_running_reason` check the immediate Start
Training button already uses, so the queue can't blow past the
account's 1-GPU-slot quota either; in-memory only (losing the queue on
a container restart is an accepted limitation, not a bug).

**Completed and independently re-verified.** Reviewed the actual diff
(not just the agent's report): correct `threading.Lock` usage around
all shared state, correct `global` scoping (Python subtlety -- in-place
list mutations like `_training_queue.append(...)`/`_training_queue[:] =
[...]` don't need `global`, but reassigning `_queue_started` inside
`do_POST` does, and it's declared once at the top of that method,
covering both the `/queue/start` and `/queue/stop` blocks further down
in the same function), and every user-controlled field routed through
`html.escape()`. Independently re-verified live via SSM: all 5 tabs
still 200, Training Queue panel renders "0/10 queued" with the empty
state, no errors in `docker logs`. Did not force a real end-to-end
Start-Queue-to-actual-training-job test (would burn real GPU
cost/quota) -- everything downstream of "pop entry, call
`_start_training_run`" is the same code path the pre-existing
Start Training button already proven live.

Committed and pushed separately once verified: `2cd145e` (`3ad8b0f..
2cd145e`), since it landed after the earlier bulk commit (`3ad8b0f`,
`017eaba..3ad8b0f`) that intentionally excluded `review_app.py` while
this agent was still mid-edit. Also pushed the ~430MB of stray local
checkpoint downloads and the terraform plan-file clutter as `.gitignore`
exclusions rather than committing them.

**Metrics-comparison feature completed and independently re-verified.**
Checkboxes on the Metrics tab (disabled for baseline rows with no
recorded metrics) feed a plain browser-native GET form; 2+ selections
render a grouped bar chart (one cluster per metric, one bar per
version) as hand-built inline SVG -- no charting library, no CDN.
Capped at 8 versions, tied to a fixed 8-color CVD-validated palette
rather than an arbitrary round number. Reviewed the diff myself
(correct `_compare_key` group+version disambiguation across the 4
separate size groups, every string field `html.escape()`'d including
inside SVG `<title>` tooltips) and independently re-verified live:
comparing `yolo11s-detector` v3 vs v4 renders exactly 10 `<path>`
bars (5 metrics x 2 versions) with correct real values (mAP50=0.989,
mAP50-95=0.870, precision=0.973, recall=0.970, accuracy=0.945 for
both -- they're near-identical registrations), and selecting only 1
version correctly shows "Check at least one more version to compare."
instead of a broken chart. All 5 tabs still 200, logs clean. Committed
and pushed as `661173c` (`2cd145e..661173c`).

All three user-requested review-app features from this stretch of the
session (training queue, image-count download progress, metrics
comparison) are now built, independently verified live, and pushed.

### YOLO12 confirmed already fully excluded; leftover comments cleaned up
User asked to "remove YOLO12". Checked live: it was never actually
selectable anywhere -- `GENERATION_SIZES` in both `config.py` and
`review_app.py` only ever had `"11"` as a key, `sagemaker.tf` has no
yolo12 references, and `list-model-package-groups` confirmed only the
four `yolo11{s,m,l,x}-detector` groups exist live. What DID exist were
several comments describing YOLO12 as "not yet added, re-add later" --
rewritten to state the exclusion as deliberate/permanent. No behavioral
change, no redeploy needed (`9f1bb9a`).

### Two real, unrelated bugs found and fixed while investigating "no loading indicator" (2026-09-07, evening)
User reported starting a training run (YOLO11-m, 5 epochs) showed no
sign of anything happening -- the Models tab kept showing the PREVIOUS
run's stale "Succeeded" panel. Investigating turned up two genuine,
separate production bugs, not just a UI polish issue:

**Bug 1 -- `run-pipeline-src.zip` was missing `evaluate.py` entirely.**
When the zip was rebuilt earlier today to filter out ~430MB of stray
checkpoint binaries (see the "images loaded" progress-bar update
above), it was built from `model_training/src/*.py` only --
`pipeline_definition.py` documents (and requires) `evaluate.py` to sit
flat right next to it in that same zip (see its own
`_EVALUATE_CANDIDATES` comment), but `evaluate.py` actually lives in
the sibling `model_training/training-container/` directory. Every
`run_pipeline.py` invocation after that zip rebuild crashed instantly
with `FileNotFoundError: evaluate.py not found` -- confirmed via the
real CodeBuild logs for the user's own failed build
(`track2-yolo-run-pipeline:4a3de23b-...`). This is why NOTHING appeared
to happen: the submission never even got as far as creating a new
Pipeline Execution for the UI to show. Fixed by rebuilding the zip to
copy `evaluate.py` in alongside the `src/` files, reapplied via
`terraform apply -target=aws_s3_object.run_pipeline_source`. Confirmed
fixed by re-submitting the exact same request via raw `codebuild
start-build` -- a new execution appeared and reached `TrainYolo
Executing`.

**Bug 2 -- `yolo11m-detector`'s registered baseline pointed at an S3
object that was never actually uploaded.** The re-triggered run then
failed differently: `TrainYolo` crashed with a real `404` on
`HeadObject` for `s3://track2-s3/model-registry/yolo11m-baseline-11m/
model.tar.gz`. Checked all four sizes' registered `ModelDataUrl`s
against what actually exists in S3: s/l/x/n all correctly point at
their `-baseline-11{size}/` (new naming) objects, which exist -- only
`m` was left pointing at a `-baseline-11m/` object that was never
created; only the OLD-naming `-baseline-m/model.tar.gz` exists (a
leftover from before the earlier registry-naming reorg documented
elsewhere in this file). Fixed by copying the existing working tarball
to the path the registration already expects
(`aws s3 cp .../yolo11m-baseline-m/model.tar.gz .../yolo11m-baseline-11m/model.tar.gz`)
rather than re-running registration (which would've created an
unwanted new package version). Confirmed fixed by re-submitting the
user's exact original request (YOLO11-m, 5 epochs) again -- the
training job reached `InProgress` and is running for real as of this
note.

### UI improvements built alongside the bug investigation
Two things the user explicitly asked for, on top of the bug fixes:
1. **A real "Submitting new run..." / "Finalizing..." indicator** for
   the CodeBuild-submission gap (the window between "Start Training was
   clicked" and "a new Pipeline Execution actually exists" -- previously
   invisible, which is exactly what made Bug 1 look like nothing
   happened instead of showing a real error). New module-level
   `_pending_submission` state (set by `_start_training_run`, the one
   shared function both the immediate button and the training queue
   already went through), a `_get_build_status`/`_pending_submission_overlay`
   pair polling the actual CodeBuild build's status, and a genuine error
   message if that build itself fails (surfacing exactly the kind of
   failure Bug 1 was, instead of silence). Also folded into
   `_training_already_running_reason` so a second click during this gap
   is blocked the same way an Executing run already blocks it. Needed
   one new IAM action, `codebuild:BatchGetBuilds`, scoped to
   `build/track2-yolo-run-pipeline:*` (confirmed this scoping works live
   before committing to it, rather than assuming `"*"` was required).
2. **A calm idle state once a run finishes.** Previously the panel kept
   showing the last run's "Succeeded"/"Failed" badge indefinitely, which
   looked like something was still active or relevant. Now, whenever
   nothing is Executing and no submission is pending, the panel shows
   "No training in progress" in a new blue (`--info: #3987e5`) accent
   color, with the last run's outcome underneath as a small muted note
   rather than the main focus.
Both confirmed live against the real run above: idle blue state
rendered correctly while nothing was active, and the full detailed
Executing panel (steps + epoch/image bars) rendered correctly once the
re-triggered run picked back up. The "Submitting..." message itself
was exercised via a direct CLI-triggered build (which intentionally
doesn't set `_pending_submission`, since that's only set by the app's
own `_start_training_run`) -- still queued to verify through the app's
own `/train/dataset` route once the current GPU slot frees up, to avoid
a double-submission collision while a real run is active.

**Update: the re-triggered YOLO11-m/5-epoch run (started 20:50:43 SGT)
completed TrainYolo successfully** (`Completed`, pipeline moved on into
Evaluate/Register) -- confirms both bugs above are genuinely fixed end
to end, not just "the crash went away." Committed and pushed as
`09ad629` (`9f1bb9a..09ad629`).

### Two more small UI requests, done same evening
1. **Error/failure messages now render in red.** Every
   `self._redirect(...)` call site was audited and classified: failures
   and blocked actions (delete failed, deploy failed, training trigger
   failed, already-running blocks, queue-full, unknown schedule mode,
   etc.) now pass `is_error=True`, rendering with a new `.banner-error`
   style (`--reject` red) instead of the same green/teal accent color
   every message used before. Successes are unchanged. Also fixed the
   redirect message not being `html.escape()`d or URL-`quote()`d before
   this pass touched that exact line (pre-existing gap, cheap to close
   alongside).
2. **Metrics tab rows now grouped under a header per YOLO size** (e.g.
   "yolo11s-detector") instead of only a small label on each row's
   right edge -- makes it obvious at a glance which size's versions
   you're looking at. Confirmed live: headers render correctly for all
   four groups (s/m/l/x), in the same sorted order rows already used.
Both deployed and verified live via SSM (banner color-coding tested
with real success/error query strings; header rendering confirmed
against the real page HTML). Pushed together as `09ad629` above.

### Review app was unreachable -- found and fixed, but flags a real recurring gap
User reported the URL wouldn't load. Investigated: the EC2 instance
itself was **stopped** (not a security-group or app problem) --
confirmed via CloudTrail that an existing automated schedule
(`terraform/infra/ec2_schedule.tf`, present from earlier work, not
something built this session) stops it nightly at 22:00 SGT and only
restarts it weekdays at 09:00 SGT (off entirely over weekends). This
stop happened exactly on schedule, not a crash. Started it back up
manually; the app container came back up on its own (has a restart
policy) and is healthy.

**Real gap surfaced by this**: the instance has no Elastic IP, so its
public IP changes on every stop/start cycle -- the previously-bookmarked
URL (`13.214.132.219`) is now dead; the current one is `13.214.164.164`.
This will keep happening every single day the auto-stop/start schedule
fires. Flagged to the user with a recommendation to attach an Elastic
IP (no real cost difference under AWS's current public-IPv4 billing,
which charges hourly for a dynamic public IP either way) -- **not yet
approved or done**, waiting on the user's answer.

### Next-day update (2026-09-08): batch/seed/image-size made real, epochs default raised, Training Strategy shown
User asked (after having me analyze an unrelated "aidil" 2x3-factorial
experiment script for its training parameters, static analysis only --
that script is a separate system, not touched) to bring some of the
same parameters into this app's own Start Training form for real:
- **Epochs default** on the form changed from 20 to 50.
- **Image Size, Batch Size, Seed** added as genuine, fully-wired
  parameters -- not just UI display. `train.py` gained real `--batch`/
  `--seed` args passed straight into Ultralytics' `model.train()`;
  `pipeline_definition.py` reads `TRAIN_BATCH`/`TRAIN_SEED` env vars the
  same way it already did for `TRAIN_EPOCHS`/`TRAIN_IMG_SIZE` (image
  size turned out to already be plumbed through pipeline_definition.py,
  just never exposed on the form). All three flow through
  `_start_training_run`/`_validate_training_params` and the ad-hoc
  Training Queue's entry shape identically to epochs, including the
  Lineage tab's hyperparameter display.
- **Training Strategy** shown as a fixed, disabled "Full" field --
  explicitly not a real selector, since this pipeline has no concept of
  alternate strategies (transfer/full-retrain, etc.) the way the aidil
  script does.
Rebuilt and redeployed both CodeBuild source zips + the training
container image. Verified with hard evidence, not just code review: a
real training job's actual `HyperParameters` showed `batch: "8"`,
`seed: "99"` -- the exact distinctive test values sent through the live
`/train/dataset` route, proving the whole chain works. Pushed as
`6f2313d`.

### Automated Training Queues (2026-09-08, later) -- persistent, individually-scheduled training recipes
User asked for something bigger and genuinely different from the
ad-hoc Training Queue: pick all training parameters, click a separate
"Add to Automated Pipeline" button, and have it land in its own queue
window that also appears on the Scheduling & Automation tab -- multiple
of these can exist at once, each independently configurable, deletable,
and carrying its own schedule (not one shared global schedule).

**Key architecture decision, asked of the user before building**: since
this review-host EC2 instance auto-stops nightly (22:00 SGT) and all
weekend (see the EC2-unreachable incident above), a purely in-process
scheduler would silently skip most "daily" or weekend-touching
schedules. User chose the reliable option: a **real EventBridge rule
per queue**, not an in-app polling thread.

**What got built**:
- New DynamoDB table (`track2-yolo-automated-queues`) storing each
  queue's full parameter set + schedule (mode/hour/day, plus the
  derived cron expression and a human summary).
- New shared Lambda (`track2-yolo-automated-queue-trigger`,
  `terraform/infra/lambda_src/automated_queue_trigger.py`) -- every
  queue's EventBridge rule targets this ONE Lambda, passing that
  queue's own parameters as the rule's fixed Input JSON. The Lambda
  just calls `codebuild.start_build` with those params, the exact same
  mechanism the immediate Start Training button and ad-hoc queue
  already use (no pipeline-parameterization refactor needed -- each
  firing rebuilds+resubmits the pipeline via CodeBuild, same as every
  other trigger path in this app).
- `review_app.py` creates/updates/deletes each queue's own EventBridge
  rule directly via boto3 (`events.put_rule`/`put_targets`/
  `delete_rule`) -- not Terraform-managed, since queues are dynamic
  runtime state. IAM scoped so the review-app role can only touch rules
  under the `track2-yolo-auto-queue-` prefix (can't touch the
  pre-existing single global retrain rule or anything else in
  EventBridge); the Lambda's own invoke permission uses a matching
  wildcarded `SourceArn` so no new Terraform is needed per queue.
- Shared `_render_automated_queues_panel()` renders identically on the
  Models tab (where "Add to Automated Pipeline" lives, reusing the Start
  Training form's own dataset/base-model fields) and the Scheduling tab
  (view/edit/delete only there -- adding a new one needs the S3
  browsers, which only the Models tab's form has).
- Extracted `_cron_and_summary_from_schedule_form` as a shared helper
  (the existing `/train/schedule` route had this cron-building logic
  duplicated inline; refactored to reduce, not just added onto).
- Added the user's other same-session ask: a note on the Scheduling tab
  clarifying training runs on AWS, not the browser -- it keeps going
  after the page closes, and Automated Queues keep firing even while
  this server itself is stopped.

**Verified against real infrastructure at every step, not just code
review**: created a real queue through the actual `/automated-queue/add`
route and confirmed its DynamoDB item, EventBridge rule, and target
payload all matched exactly; invoked the Lambda directly with a
distinctive test payload (batch=8, seed=77, epochs=1) and confirmed the
resulting real SageMaker training job's `HyperParameters` matched
exactly, proving the full EventBridge-rule-payload -> Lambda ->
CodeBuild -> SageMaker chain works without needing to wait for an
actual scheduled fire; edited the queue and confirmed the rule's
schedule expression and payload both updated in place (put_rule/
put_targets are upserts, no separate delete-then-recreate needed);
deleted it and confirmed both the DynamoDB item and the EventBridge
rule were fully removed. All 5 tabs stayed at HTTP 200 throughout,
container logs clean. Pushed as `bdafe0e`.

### Stopping point -- where to resume next
1. **Decide on the Elastic IP** for the review-host instance (still
   unresolved from the outage investigation) -- if yes, add an
   `aws_eip`/`aws_eip_association` to `terraform/infra/ec2.tf`, targeted
   apply, update `review_app_url` output and anywhere else the IP is
   referenced.
2. Current live URL (until it next restarts): `http://13.214.164.164:8090/`.
3. `yolo11n-detector` still gone (deleted via Console, never
   re-registered) -- unresolved from earlier in the session.
4. Review app is still fully public (deliberate, user-confirmed).
5. All bugs found this session (evaluate.py zip regression, yolo11m
   missing-checkpoint) remain fixed and confirmed via real successful
   end-to-end runs; nothing else outstanding from that list.
6. Automated Training Queues feature is fully built and verified but
   has never had a real *scheduled* firing observed yet (only a direct
   Lambda invocation, deliberately, to avoid waiting on a real cron
   time) -- if one gets added with a near-term schedule, worth
   double-checking it fires for real on time.

### Real production hang diagnosed and fixed, plus a related backlog fix and a new Deploy tab (2026-09-08, later still)
User asked "what is the url" and it turned out the app had genuinely
**hung** -- container alive, port 8090 listening, but every request
timed out with no response at all (confirmed via `curl -m 8`, exit code
28). A container restart fixed it immediately. Root cause: `review_app.py`
ran on a plain single-threaded `HTTPServer`, which handles exactly one
request at a time -- since this app is deliberately public (0.0.0.0/0,
no auth) and visibly gets hit by internet scanners all session (harmless
`BrokenPipeError`s in the logs throughout), any one stalled/slow
connection can wedge the entire server for every real user
indefinitely.

**Fixed properly, not just patched**: switched to `ThreadingHTTPServer`,
then audited every piece of shared module-level state for the
concurrency this introduces before deploying it:
- The ad-hoc Training Queue (`_queue_lock`) and `_pending_submission`
  (`_pending_submission_lock`) were already lock-protected -- built that
  way originally because a background poll thread already touched them
  concurrently with request handlers, so no changes needed.
- The three shared DynamoDB `boto3.resource().Table()` objects (review
  queue, lineage, automated queues) were **not** safe -- boto3
  *resources*, unlike *clients*, aren't documented thread-safe, and
  three request threads sharing one now genuinely matters. Converted to
  thread-local via `threading.local()` (`table()`/`lineage_table()`/
  `automated_queues_table()` as functions, not shared globals) across
  all 9 call sites.
- **Verified the actual fix, not just "nothing broke"**: held a stalled
  raw TCP connection open against port 8090 (`nc`, no data sent) and
  confirmed a normal concurrent request still completed in ~1 second
  instead of hanging. Pushed as `7f680e1`.

**Related, same session**: user separately reported "allow more than 5
users to log in at one go" -- there's no login system in this app at
all (by design, still), but the number 5 is a precise match for
`socketserver`'s default `request_queue_size` (TCP accept backlog).
Raised to 128 via a small `ThreadingHTTPServer` subclass (has to be a
class attribute -- `listen()` already happens by the time an instance
exists, too late to set it afterward).

**New Deploy tab**: user asked to "add a tab to deploy the selected
model endpoint". Investigating turned up that the backend for this
(`_deploy_inference_component`, `/models/<size>/deploy` and
`/deploy-s3` routes) already existed and fully worked -- there was just
no UI left anywhere that actually reached it (likely lost during one of
the many `_render_models_tab` rewrites this session, since routes live
in `do_POST` separately from tab-rendering code). Built the tab: one
panel per size showing the real live Inference Component status (
verified: all four sizes genuinely `InService` with real model
names/timestamps from earlier session work) plus a dropdown of that
group's Approved versions to redeploy, and a collapsed raw-S3-folder
option for bypassing the registry. No new IAM needed
(`DescribeInferenceComponent` was already granted). Verified via a real
POST through the new route -- `yolo11s-detector` v1 redeployed
successfully. Pushed as `3db4e30`.

### Where to resume next
1. Elastic IP decision for the review-host instance -- still open.
2. `yolo11n-detector` still gone; nothing else outstanding from the bug
   list.
3. Automated Training Queues: still no observed real scheduled firing
   (only direct invocation tested).
4. Consider whether the Deploy tab should also expose the m/l/x-specific
   compute requirements or CopyCount if that ever needs tuning -- not
   asked for, not built.

### Another training failure diagnosed, and it revealed who else is using this account's shared GPU quota
User asked "why did the last training run fail?" -- same
`SAGEMAKER_RESOURCE_LIMIT` failure as before, but this time the account's
one `ml.g4dn.xlarge` slot was occupied by a job from a **completely
different system**: `ffy11s-ctxscaled-transfer-184-...`, part of a
scaling-study batch (train-size 46/92/184, tagged
`experiment: context-scaled-yolo11s-2x3`) run by a real separate IAM
user, `sitaidil` -- confirmed via CloudTrail (MFA-authenticated, from a
WSL2 Linux box, same "aidil-synthetic-data-yolo" S3 prefix/experiment
family analyzed earlier in the session). Not a bug anywhere in this
app; just two independent systems sharing one account-wide quota.

**Live "SageMaker GPU Resource" indicator added** (user asked for
this directly, "live"): a new panel on the Models tab showing
account-wide (not just this app's own pipeline) `ml.g4dn.xlarge` usage
via `sagemaker:ListTrainingJobs` + `DescribeTrainingJob`, distinguishing
"this app's own run" from "a different system." **Also closed a real
gap this surfaced**: `_training_already_running_reason` (the guard
behind the disabled Start Training button and the server-side check)
previously only looked at this app's own pipeline executions --
extended it to check account-wide GPU usage too, so a future collision
with an outside job like this one gets blocked before submission
instead of failing after. Needed one new IAM action,
`sagemaker:ListTrainingJobs`. Verified live: the panel correctly showed
the real `sitaidil` job as "a different system, not this app" while it
was still running.

### Notifications tab and Help tab added (same session, continuing)
Two more user requests, both built and verified live:
- **Notifications tab**: add/remove email recipients for the single SNS
  topic every human-facing notification already goes through (training
  pipeline status, drift-triggered review needed, weights approved --
  see `monitoring.tf`'s `reviewer_notifications` topic) -- previously
  only editable via Terraform's `reviewer_emails` variable. Shows real
  confirmation status per subscriber (Confirmed vs PendingConfirmation
  -- SNS has no API to cancel a pending one early, it just expires after
  3 days). All notification types currently share this one topic, so
  there's no per-category subscription control yet -- flagged clearly
  in the tab's own copy rather than silently pretending otherwise;
  splitting that apart would mean changing the Lambdas that publish to
  it. Needed `sns:Subscribe`/`Unsubscribe`/`ListSubscriptionsByTopic`.
  Verified live read-only (real existing subscriber shows Confirmed) --
  deliberately did NOT exercise the Subscribe action with a real email
  during testing, since that sends an actual confirmation email to
  whatever address is used.
- **Help tab**: purely static, no AWS calls -- documents the exact
  dataset folder format `train.py`/`pipeline_definition.py` actually
  require (root-level `data.yaml`, images/labels layout, YOLO label
  line format, supported image extensions), pulled directly from that
  real code and `seed_training_data.py`'s own working example rather
  than a generic YOLO writeup.

The review app now has 8 tabs total: Review Queue, Models, Scheduling &
Automation, Lineage, Metrics, Deploy, Notifications, Help.

### Where to resume next (supersedes the list above)
1. Elastic IP decision for the review-host instance -- still open.
2. `yolo11n-detector` still gone; nothing else outstanding from the bug
   list.
3. Automated Training Queues: still no observed real scheduled firing.
4. Consider requesting an AWS Service Quota increase for
   `ml.g4dn.xlarge` training usage, now that it's confirmed two
   independent systems (this app + `sitaidil`'s experiment harness)
   actively compete for the same single-instance quota.
5. If per-notification-category subscriptions ever get asked for, that
   needs the publishing Lambdas changed too, not just the new tab.

### Metrics tab now shows training notes and full lineage per row
User asked for the notes typed into the Models tab and the full
training lineage to also appear next to each version's metrics.
Replaced `_fetch_evaluation_metrics` with `_fetch_metrics_and_lineage`
(one `describe_model_package` call per version instead of two -- this
tab shows every version, not just the first 8 per group the Review tab
already covers, so it needed its own fetch rather than reusing that
one). Verified live: a real version's row now shows both its actual
training notes and its full lineage line (dataset URI, base model ARN,
hyperparameters) via the same `_format_lineage` formatting the Lineage
tab already uses. Pushed as `28fac1d`.

### Inference and Export Weights tabs added -- two real IAM bugs and one real production issue found via live testing
User asked for two more tabs: one to actually run inference against a
deployed model, one to export a model's raw `.pt` weights to any S3
folder with automatic archiving.

**Inference tab**: pick a size with something deployed, browse S3 for a
real image, invoke it live, see the result as bounding boxes drawn
directly over the image (inline SVG, `viewBox="0 0 1 1"` matching
`inference.py`'s own normalized 0.0-1.0 box coordinates exactly -- no
pixel math needed) plus the raw detections. Getting this actually
working surfaced two real IAM mistakes, both corrected only after
seeing the real `AccessDeniedException` name the actual requirement
(not guessed up front):
1. The IAM action is `sagemaker:InvokeEndpoint`, not
   `sagemaker-runtime:InvokeEndpoint` -- the boto3 client name doesn't
   match the IAM action's own prefix.
2. When `InferenceComponentName` is set on the call (every call here
   does), the resource-level check is against the **inference
   component's own ARN**, not the shared endpoint's -- a second
   attempt scoped to only the endpoint ARN still got denied, naming the
   component ARN instead.
Also found and fixed a real, separate bug: boto3's default retries were
compounding a genuinely slow cold start into what looked like an
indefinite hang (240s+, zero bytes back). Disabled retries and extended
`read_timeout` to 600s on the `sagemaker-runtime` client specifically --
after that, a real invocation completed cleanly in 61s.

**While testing this (not a bug in the new code), found a real,
currently-active production issue**: `yolo11s`'s deployed Inference
Component has been failing its own internal health check (`/ping`)
continuously for minutes, confirmed in its own CloudWatch logs
(`/aws/sagemaker/InferenceComponents/track2-yolo-s`) -- every real
invocation attempt timed out against SageMaker's own ~60s real-time
hard limit because the worker itself is unresponsive, not because of
anything in the new Inference tab. No root cause (OOM, crash, deadlock)
was visible in the log text itself. Flagged separately -- redeploying
that size via the Deploy tab (forces a fresh Model + IC update) is the
most likely quick recovery, not yet tried.

**Export Weights tab**: export any Approved model's raw `.pt` to any S3
folder, always saved as `selected.pt` (a fixed convention for this
tool). If that folder already has one, it's archived first -- a real
server-side S3 copy to `archived/selected_<date>.pt` -- rather than
silently overwritten. Mirrors the exact archive-then-overwrite pattern
`terraform/infra/lambda_src/dump_approved_weights.py` already uses for
its own fixed-path automatic flow, generalized to any destination.
Refactored `_render_s3_browser` to accept a `tab` parameter (previously
hardcoded to `"models"`) since this became its third caller (Models,
now Export Weights).

Both verified against real infrastructure, not just code review: a real
Approved package's weights were exported to a scratch S3 folder
(confirmed the actual `.pt` landed at the right size), exported again
to the same folder (confirmed the first file was correctly archived
with a real timestamp and the new one took its place), then the test
folder was cleaned up. All 10 tabs confirmed healthy throughout. Pushed
as `09e17df`.

The review app now has 10 tabs: Review Queue, Models, Scheduling &
Automation, Lineage, Metrics, Deploy, Inference, Notifications, Help,
Export Weights.

### Where to resume next (supersedes the list above)
1. **Investigate/fix the yolo11s Inference Component health-check
   failure** -- try redeploying it via the Deploy tab first as the
   likely quick fix; if it recurs, consider whether COMPUTE_REQUIREMENTS
   for size "s" (1 CPU core / 2048MB) is simply too tight.
2. Elastic IP decision for the review-host instance -- still open.
3. `yolo11n-detector` still gone; nothing else outstanding from the bug
   list.
4. Automated Training Queues: still no observed real scheduled firing.
5. Consider an `ml.g4dn.xlarge` training-usage Service Quota increase,
   given the confirmed collision with `sitaidil`'s own experiment
   harness.
6. An untracked `JP_UI/` directory appeared in the working tree,
   unrelated to anything in this session -- left alone, not committed,
   since its origin/purpose isn't known.
