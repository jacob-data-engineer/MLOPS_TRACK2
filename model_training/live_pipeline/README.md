# The pipeline definition that actually runs

`track2-yolo-training-pipeline.json` is the definition of the live
SageMaker pipeline, with four hyperparameters made tunable.
`...before.json` is exactly what was deployed beforehand, kept so the
change is reviewable as a diff and revertable in one command.

`../src/pipeline_definition.py` does **not** build this pipeline any more.
It still describes the older crop pipeline, and editing it changes
nothing about what runs -- the live definition is upserted out of band and
is only recoverable from SageMaker itself, which is why a copy lives here.

## What changed

Four parameters were added, each with a `DefaultValue` equal to the value
that was previously hardcoded:

| Parameter  | Default | Was            |
|------------|---------|----------------|
| `Epochs`   | `50`    | `"epochs": "50"`    |
| `ImgSize`  | `640`   | `"img-size": "640"` |
| `Batch`    | `16`    | `"batch": "16"`     |
| `Seed`     | `42`    | `"seed": "42"`      |

and `TrainYolo`'s `HyperParameters` now reference them via
`{"Get": "Parameters.Epochs"}` and so on.

Because every default matches the previous literal, an execution that
passes none of them behaves identically to before the change. Nothing else
in the definition is touched -- same training image, same instance type,
same steps, same retry policy.

## Applying it

```powershell
aws sagemaker update-pipeline `
  --pipeline-name track2-yolo-training-pipeline `
  --pipeline-definition file://track2-yolo-training-pipeline.json `
  --region ap-southeast-1
```

To revert, run the same command against
`track2-yolo-training-pipeline.before.json`. Updating a definition does
not disturb executions already running; they keep the definition they
started with.

A fifth parameter, `EvaluationDatasetUri`, was added at the same time, and
`EvaluateYolo` now reads its data from that instead of from
`TrainingDatasetUri` -- which is what let every run until now be scored on
the data it had just trained on.

It deliberately has **no** `DefaultValue`, matching the other URI
parameters: the coordinator supplies it on every start (falling back to
the training URI when no separate evaluation set was chosen), so there is
no run without a value, and a start that forgets it fails immediately
rather than quietly evaluating against the wrong data.

## Still fixed elsewhere

The base-model picker on the Training page is still not wired through: the
coordinator selects the current production baseline itself via
`production_pointer`. Changing that means changing how a replay run
chooses its starting weights, which is a different decision from choosing
data or hyperparameters.
