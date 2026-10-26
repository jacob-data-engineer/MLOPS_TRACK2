# track2-yolo-weekly-retrain (full-frame replay coordinator)

This is the **deployed** source of the `track2-yolo-weekly-retrain` Lambda,
downloaded from AWS. It is here because it was not in version control at
all: `terraform/platform/lambda_src/trigger_weekly_retrain.py` still holds a
23-line stub that starts the pipeline with no parameters, while the
function that actually runs is this three-module replay coordinator.

## Deploying

This directory **is** the deploy source. Terraform packages it:

```powershell
cd C:\Projects\MLOPS_TRACK2\terraform\platform
terraform apply -target=aws_lambda_function.weekly_retrain
```

It was not always safe to do that. `pipeline_schedule.tf` used to point
`archive_file` at `source_file = lambda_src/trigger_weekly_retrain.py` --
a stale 23-line stub, and one file where the function needs three
(`trigger_weekly_retrain` imports `prepare_replay_dataset`, which imports
`replay_policy`). An apply would have published a single-module zip that
failed on import. It also declared `timeout = 30` and the default 128MB
against a function deployed at 900s/2048MB, and no
`reserved_concurrent_executions` against a deployed reservation of 1 --
which this coordinator depends on, since it reads and rewrites
`pending.json` with no locking of its own. All four now match what runs.

The three inline policies on the role (`fullframe-snapshot-routing`,
`fullframe-production-registry-read`, `fullframe-retraining-hardening`)
are **not** Terraform-managed and are left alone by an apply, because no
`inline_policy` block is declared on the role. Do not add one without
moving all three in first.

## What it does

Each request pins a completed, immutable dataset snapshot, derives a
person=0 / helmet=1 dataset from it, selects the current production
baseline as the starting weights, and starts one execution of
`track2-yolo-training-pipeline`. Invocations are serialized: a request
that arrives while another is executing is queued as `pending.json` and
supersedes any earlier queued one. `run_id` is a hash of the caller's
`request_id`, which is what makes a retry idempotent.

It does not deploy an endpoint or approve a model.

## Tunable hyperparameters

`requested_hyperparameters` reads `epochs`, `img_size`, `batch` and `seed`
from the invoking event, validates them, and records them on the request
before the run starts -- so a retry of the same `run_id` trains with the
values it was created with rather than whatever a later caller sends. They
are passed to `start_pipeline_execution` as the `Epochs`, `ImgSize`,
`Batch` and `Seed` pipeline parameters.

Anything not supplied is left out of the call entirely, so the pipeline's
own `DefaultValue` applies. That is what keeps the weekly schedule, which
sends none of them, running exactly as it did before these existed.

The pipeline must declare those four parameters for this to work -- see
`../live_pipeline/`.

## Chosen datasets

`requested_datasets` reads `dataset_uri` and `eval_dataset_uri`. Either
absent, the run behaves exactly as before: the dataset is derived from the
pinned snapshot, and evaluation reads the training data.

Two things change when a dataset IS chosen:

- No dataset card exists to read classes from, so `classes` is left unset
  rather than asserting person/helmet about data nobody validated. The
  training container reads them from the dataset's own `data.yaml`.
- The snapshot is **not** marked consumed. A run on a hand-picked folder
  did not train on the snapshot, and writing `consumed/<fingerprint>.json`
  for it would make the next scheduled run skip a snapshot nothing had
  trained on -- silently dropping a week of new data.

`evaluation_is_held_out` is recorded on the request so a run scored on its
own training data is identifiable afterwards rather than having to be
inferred by comparing URIs.
