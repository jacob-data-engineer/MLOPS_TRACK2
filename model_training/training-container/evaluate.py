"""SageMaker Processing entrypoint: evaluates a trained YOLO checkpoint
against the held-out validation split, using Ultralytics' own built-in
val() (computes mAP50, mAP50-95, precision, recall) rather than
hand-rolling detection-matching logic.

Runs as a ProcessingStep between TrainingStep and RegisterModel (see
pipeline_definition.py) — not baked into the training container image,
passed in via ScriptProcessor's own code-upload mechanism instead, so no
container rebuild is needed just to add this file.

Writes evaluation.json to the Processing job's output path, which
pipeline_definition.py attaches to the registered model via
RegisterModel's model_metrics parameter — every parked candidate in the
Model Registry now carries real numbers next to it, instead of requiring
a blind approve/reject decision (brainstorming.md's Gap #1).

The SageMaker Studio Model Registry console's "Performance metrics" panel
(Precision/Recall/F1/Accuracy score tiles) doesn't read arbitrary JSON —
it only parses the specific Model Monitor model-quality report schema:
metrics nested under one of regression_metrics/binary_classification_metrics/
multiclass_classification_metrics, each as {"value": ..., "standard_deviation":
...} (see docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-
metrics.html). A flat {"precision": 0.57, ...} dict — which is what this
used to write — is invisible to that panel even though the key names match,
since it isn't nested under the expected wrapper. binary_classification_metrics
is the closest fit available for an object-detection task (there's no
"object_detection_metrics" schema).

Confirmed all four tiles (Precision/Recall/F1/Accuracy) still rendered "-"
even with precision/recall/f1 correctly nested, despite the key names
matching the schema. Every real example AWS documents for
binary_classification_metrics includes confusion_matrix and accuracy
alongside precision/recall/f1 — never a subset — so the working theory is
the console validates against the full shape as one unit and renders
nothing if fields are missing, rather than degrading per-tile. Both are now
included, and both are real, not faked: Ultralytics' own model.val() already
computes a full (nc+1, nc+1) confusion matrix internally (the extra index
is a "background" bucket for false positives/negatives) — accessible as
results.confusion_matrix.matrix, keyed by results.confusion_matrix.names —
previously just never surfaced. accuracy is derived from that same matrix
(correct-class diagonal / total recorded outcomes), not fabricated.

Experiment logging is explicit (EXPERIMENT_NAME/RUN_NAME passed in as
container env vars from pipeline_definition.py), not bare load_run() —
that auto-detection path has a documented SageMaker SDK bug inside
ProcessingSteps specifically (github.com/aws/sagemaker-python-sdk#4114).
Degrades gracefully (prints a warning, doesn't fail the step) if
Experiments logging fails for any reason — the metrics.json output and
the model registration are what matters; comparison in Studio is a
bonus, not a hard dependency (Gap #5).
"""

import json
import os
import tarfile

from ultralytics import YOLO

MODEL_DIR = "/opt/ml/processing/model"
DATA_DIR = "/opt/ml/processing/data"
OUTPUT_DIR = "/opt/ml/processing/output"


def _extract_model() -> str:
    tar_path = os.path.join(MODEL_DIR, "model.tar.gz")
    if os.path.exists(tar_path):
        with tarfile.open(tar_path) as tar:
            tar.extractall(MODEL_DIR)
    model_path = os.path.join(MODEL_DIR, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found after extracting {tar_path}")
    return model_path


def _log_to_experiments(metrics: dict) -> None:
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    run_name = os.environ.get("RUN_NAME")
    if not experiment_name or not run_name:
        print("EXPERIMENT_NAME/RUN_NAME not set - skipping Experiments logging")
        return
    try:
        from sagemaker.experiments.run import Run

        with Run(experiment_name=experiment_name, run_name=run_name) as run:
            for name, value in metrics.items():
                run.log_metric(name=name, value=value)
        print(f"Logged metrics to Experiments run {experiment_name}/{run_name}")
    except Exception as exc:
        # Best-effort: comparison in Studio is a bonus, not a hard
        # dependency for the pipeline itself to succeed.
        print(f"Experiments logging failed, continuing anyway: {exc}")


def main():
    model_path = _extract_model()
    data_yaml = os.path.join(DATA_DIR, "data.yaml")

    model = YOLO(model_path)
    results = model.val(data=data_yaml, split="val")

    precision = float(results.box.mp)
    recall = float(results.box.mr)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Real confusion matrix from Ultralytics' own validator -- (nc+1, nc+1),
    # the extra row/col is a "background" bucket catching false positives
    # (predicted, no matching ground truth) and false negatives (ground
    # truth, no matching prediction). Diagonal entries [0:nc] are correct
    # per-class detections; the [nc, nc] cell is never incremented (there's
    # no "correctly predicted nothing" event in object detection), so
    # summing the whole diagonal only counts genuine true positives.
    cm_matrix = results.confusion_matrix.matrix
    total_outcomes = float(cm_matrix.sum())
    accuracy = float(cm_matrix.trace() / total_outcomes) if total_outcomes > 0 else 0.0
    confusion_matrix_json = {
        str(i): {str(j): int(cm_matrix[i, j]) for j in range(cm_matrix.shape[1])}
        for i in range(cm_matrix.shape[0])
    }

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }
    print(f"Evaluation metrics: {metrics}")

    # binary_classification_metrics wrapper is what makes Studio's
    # Model Registry "Performance metrics" panel actually populate --
    # see the module docstring. Kept alongside the flat metrics above
    # (those remain the source of truth for our own reporting/Experiments).
    # confusion_matrix's keys are class indices as strings (index `nc` is
    # the "background" bucket) -- matching AWS's own documented example's
    # key convention, not results.confusion_matrix.names' class name strings.
    report = {
        **metrics,
        "binary_classification_metrics": {
            "confusion_matrix": confusion_matrix_json,
            "precision": {"value": precision, "standard_deviation": "NaN"},
            "recall": {"value": recall, "standard_deviation": "NaN"},
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
            "f1": {"value": f1, "standard_deviation": "NaN"},
        },
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "evaluation.json"), "w") as f:
        json.dump(report, f, indent=2)

    _log_to_experiments(metrics)


if __name__ == "__main__":
    main()
