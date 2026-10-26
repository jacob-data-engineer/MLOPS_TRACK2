# Custom-taxonomy training

Trains a detector on whatever classes a dataset declares, instead of the
two the production replay pipeline is built around.

It is a **separate pipeline, separate registry group and separate
evaluator**. Nothing here touches `track2-yolo-training-pipeline`,
`track2-ppe-fullframe-replay` or the replay coordinator, and a model
trained here can never enter the production replay lineage by accident --
which is the whole reason it is separate rather than a mode of the
existing one.

| | production replay | this |
|---|---|---|
| Pipeline | `track2-yolo-training-pipeline` | `track2-yolo-custom-training-pipeline` |
| Registry group | `track2-ppe-fullframe-replay` | `track2-yolo-custom-detector` |
| Classes | fixed: `person`, `helmet` | whatever `data.yaml` declares |
| Evaluator | `evaluate.py` (asserts the two classes) | `evaluate_any.py` |
| Dataset source | pinned pull snapshot, remapped | a folder you choose |
| Baseline | required, compared against | optional |

## Why the production pipeline cannot just be pointed at a 6-class folder

Three separate locks, all deliberate:

- `replay_policy.REMAP` **discards** every box outside the target
  taxonomy. Training the production pipeline on the raw six-class source
  silently drops gloves, working_boots, man_cage and safe_area.
- `evaluate.py` raises unless the checkpoint's names are exactly
  `{0: person, 1: helmet}` and the card says `model: hero`.
- The registry group is hardcoded, and its consumers (the production
  pointer, the inference path) read index 0 as `person`.

A model with a different taxonomy in that chain is mislabelled at every
step without failing anywhere, which is far worse than not running.

## Which pipeline actually runs

`track2-yolo-custom-training-pipeline` (below) is **superseded and unused.**
It runs the digest-pinned "recovery" training image, whose wrapper asserts
`model.names == {0: "person", 1: "helmet"}` on the BASE checkpoint -- so it
can only ever fine-tune from a production checkpoint, which defeats the
point of an open-taxonomy pipeline.

Custom-class runs go to **`track2-yolo-open-training-pipeline`**, defined by
`pipeline_open.json`. It runs `track2-yolo-train` (this repo's own
`train.py`, no class assertion) pinned by digest, sends no `run-id`, and has
no `CheckpointConfig`.

```powershell
cd C:\Projects\MLOPS_TRACK2\model_training\custom_training
aws sagemaker create-pipeline `
  --pipeline-name track2-yolo-open-training-pipeline `
  --pipeline-definition file://pipeline_open.json `
  --role-arn arn:aws:iam::125192671932:role/track2-yolo-sagemaker-execution `
  --region ap-southeast-1
```

Seeds for "train from scratch" must exist before that mode works, one per
architecture. Build them **inside the training image** -- a `.pt` is a
pickled live `nn.Module`, so the writer's library versions must match the
reader's:

```powershell
aws s3 cp make_scratch_checkpoint.py s3://track2-s3/custom/code/make_scratch_checkpoint.py --region ap-southeast-1
# then a SageMaker Processing job on track2-yolo-train running
#   python3 /opt/ml/processing/input/code/make_scratch_checkpoint.py --arch yolo11s
# -> s3://track2-s3/external_models/scratch/yolo11s/{model.pt,manifest.json}
```

## One-time setup

```powershell
cd C:\Projects\MLOPS_TRACK2\model_training\custom_training

# 1. the evaluator the pipeline runs
aws s3 cp evaluate_any.py s3://track2-s3/custom/code/evaluate_any.py --region ap-southeast-1

# 2. the registry group custom models land in
aws sagemaker create-model-package-group `
  --model-package-group-name track2-yolo-custom-detector `
  --model-package-group-description "Detectors trained on a chosen dataset's own classes." `
  --region ap-southeast-1

# 3. the pipeline itself
aws sagemaker create-pipeline `
  --pipeline-name track2-yolo-custom-training-pipeline `
  --pipeline-definition file://pipeline.json `
  --role-arn arn:aws:iam::125192671932:role/track2-yolo-sagemaker-execution `
  --region ap-southeast-1
```

Re-running step 3 for an updated definition uses `update-pipeline` with
the same arguments minus `--role-arn`.

## Preparing a dataset

The evaluation step reads `dataset-card.json`, which an ordinary YOLO
folder does not have. `prepare_any_dataset.py` writes one **in place**,
next to the folder's `data.yaml`:

```powershell
python prepare_any_dataset.py s3://track2-s3/images/dataset/
```

It validates before it writes: every label line must parse, every class
must appear in `data.yaml`, every box must lie inside the image. A missing
label file is treated as a negative example, not an error.

What it will not do is invent provenance. A replay dataset's card names
the snapshot it came from and the leakage checks that ran; a folder has
neither, so the card records `snapshot_uri: null` and
`scene_leakage_checked: false`. A card that claimed otherwise would defeat
its own purpose, which is letting a metric be traced back to data someone
can trust.

## Running one

The Training page starts these directly through
`StartPipelineExecution` -- no CodeBuild, no replay coordinator, because
none of the snapshot machinery applies. The execution ARN is therefore
known at submit time, so the training audit links to it exactly rather
than matching on time.

## Known limits

- The base model is still required by the trainer, and will usually be a
  two-class checkpoint. Ultralytics reinitialises the detection head for
  the new class count, so this is transfer learning from those weights,
  not a from-scratch run. The first model for a new taxonomy therefore has
  no meaningful baseline, and `evaluate_any.py` reports the candidate
  alone rather than refusing to run.
- Registration is `PendingManualApproval`, same as production. Nothing
  here approves a model.
- These models are not deployable through the inference tab as-is: that
  path maps class indices to the production labels. Deploying a custom
  model needs that mapping made dynamic, which is not part of this.
