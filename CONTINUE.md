# Continue here

State of this repo as of 2026-09-16, what was changed, and what is still
outstanding. `README.md` describes the layout; this file is the handover.

---

## Right now

The app is **live and healthy** at `http://47.129.112.197:8501/`, image
`20260916-110046`, deployed from this repo.

```bash
PORT=8501 CONTAINER=ppe-pipeline-track2-test bash deploy.sh
```

The inference endpoint is **deleted** (its normal resting state — it is
recreated on demand and torn down when idle). Nothing is billing for it.

---

## Outstanding, in order

### 1. This repo has no remote

`git remote -v` is empty. Every commit, including the entire history
carried over from JP_UI, exists **only on this machine**. So does
`terraform/app/terraform.tfstate` and `terraform/platform/terraform.tfstate`,
which are gitignored and have no other copy.

Push it somewhere before doing anything else.

### 2. One pending Terraform change

```powershell
cd terraform\app
terraform apply
```

`Plan: 0 to add, 1 to change, 0 to destroy` — the app's grant to start the
custom training pipeline and delete custom model packages. Until this is
applied, custom training returns AccessDenied.

### 3. Custom training is built but not switched on

Neither AWS resource exists yet:

```powershell
cd model_training\custom_training
aws s3 cp evaluate_any.py s3://track2-s3/custom/code/evaluate_any.py --region ap-southeast-1
aws sagemaker create-model-package-group --model-package-group-name track2-yolo-custom-detector --model-package-group-description "Detectors trained on a chosen dataset's own classes." --region ap-southeast-1
aws sagemaker create-pipeline --pipeline-name track2-yolo-custom-training-pipeline --pipeline-definition file://pipeline.json --role-arn arn:aws:iam::125192671932:role/track2-yolo-sagemaker-execution --region ap-southeast-1
```

Then prepare a dataset once (it writes `dataset-card.json` in place, which
the evaluation step requires):

```powershell
python prepare_any_dataset.py s3://track2-s3/images/dataset/
```

**None of this path has run end to end yet.** The code is written and
reviewed but unproven. Expect the first run to surface something.

### 4. Not started

- **Base-model picker is decorative on the production path.** The replay
  coordinator selects the production baseline itself via
  `production_pointer`. Changing that is a decision about how replay runs
  choose starting weights.
- **Custom models are not deployable through the inference tab**, which
  maps class indices to the production person/helmet labels. Needs that
  mapping made dynamic.
- **Terraform state should move to an S3 backend.** Local-only state on one
  laptop is the single largest operational risk here.

---

## What was done

### The app moved out of JP_UI and now owns the whole estate

Extracted with `git subtree split`, so all 332 commits came with it. Both
Terraform stacks live here (`terraform/app`, `terraform/platform`), plus
`model_training/` and the estate docs. **Nothing references JP_UI.**

Verified before the originals were deleted: identical plans from both
locations, byte-identical state files, and a full deploy.

Two traps a repo split sets, both hit and fixed: the parent's `.gitignore`
and `.gitattributes` do not travel. Without the first, `git add -A`
committed a 600MB provider binary and `terraform.tfstate` (rewritten out;
`.git` went 453MB → 1.2MB). Without the second, the clone checked out CRLF,
which breaks `deploy.sh` under bash and makes Terraform report a permanent
phantom diff on the CodeBuild buildspec.

### Training inputs actually reach AWS now

They never did before. Epochs, image size, batch and seed were hardcoded in
the pipeline definition, and the folder pickers were ignored — the UI
looked like it controlled a run it had no influence over.

They are now pipeline parameters, each defaulting to the value it replaced,
carried from the page through CodeBuild and the coordinator. A run that
sets none of them behaves exactly as before, which is what keeps the weekly
schedule unaffected. Proven live: a run submitted with `Epochs=1, Batch=1`
reached the training job with exactly that.

`EvaluateYolo` also read `TrainingDatasetUri`, so **every metric this
pipeline has ever reported was measured on the model's own training data.**
It now reads a separate `EvaluationDatasetUri`.

### Training audit

One DynamoDB row per run: both dataset folders and their file names, every
hyperparameter, notes, the registered model version, its metrics, and
start/end times. It records what was **submitted** and what the job
**actually got**, and flags the difference in red — which is how the
ignored-parameters problem became visible at all.

Sits under the training queue as `HISTORY`, each row deleting the model
version it registered along with it.

### Fixes worth knowing about

- **Stop Training** called an API nothing allowed. Granted.
- **The build reported FAILED on successful runs** — `run_pipeline.py` asked
  boto3 for a `pipeline_execution_complete` waiter, which does not exist. It
  threw one second after starting the execution. Replaced with polling.
- **Cost tiles always read $0.00.** The filter matched `":Host:"`, but usage
  types are `APS1-Host:...` — a hyphen. Affected the Deploy tile and the
  Cost page headline.
- **Metrics never appeared** for full-frame runs: the reader only understood
  the legacy flat `evaluation.json`, and the current evaluator nests them
  under `metrics` and reports `f1` where the old one reported `accuracy`.
- **An endpoint billed for a day.** One inference job was left at
  `deploying` when its process died; the teardown check treats that as
  unfinished work and correctly declined, forever. Jobs are now recovered at
  startup, and an idle sweep removes the endpoint after 30 minutes with
  nothing needing it — the teardown no longer depends on a queue draining.
- Automated training queues were removed entirely, UI and AWS resources.

---

## Traps

**`terraform/platform` carries pre-existing drift** — a plan reports ~15
in-place changes nobody made. Most are Lambda code being re-uploaded from
local zips, because that state was rebuilt by import and never recorded a
source hash. **Use targeted applies there.** A blanket apply pushes every
local zip over whatever is deployed.

```powershell
terraform apply -target=aws_lambda_function.weekly_retrain
```

That stack's `archive_file` packages `model_training/replay_lambda/`, so
that command is also how the coordinator gets deployed after editing it.

**The production pipeline is a two-class system by design.** Its remap
*discards* every box outside person/helmet, its evaluator refuses any
checkpoint with different class names, and its registry group's consumers
read index 0 as `person`. A six-class model in that chain is mislabelled at
every step without failing anywhere. That is why custom training is a
separate pipeline, group and evaluator — see
`model_training/custom_training/README.md`.

**A dataset needs `dataset-card.json` to be evaluated.** `train.py` only
needs `data.yaml`, so a folder without a card trains fine and then loses the
run at the evaluation step. Both folders are checked before a run starts.

**The last production run failed there**, before that check existed —
`escalation_dataset` has no card. Training succeeded; evaluation did not.
