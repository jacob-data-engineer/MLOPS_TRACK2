"""Assembles the real sagemaker.workflow.pipeline.Pipeline:
TrainingStep -> EvaluateStep -> RegisterModel.

TrainingStep: YOLO training, via the custom ECR image in
  ../training-container.
EvaluateStep: runs ../training-container/evaluate.py as a Processing job
  against the held-out validation split, computing real mAP/precision/
  recall (brainstorming.md Gap #1) instead of registering candidates
  blind. Reuses the training image (already has ultralytics) rather than
  a separate evaluation container.
RegisterModel: into track2-yolo-models, ModelApprovalStatus=
  PendingManualApproval so nothing reaches production without a person
  explicitly approving it (see ../docs/production_promotion.md) — now
  carrying the EvaluateStep's real metrics via model_metrics, so that
  decision isn't blind either.

Hyperparameters (epochs/image size) are read from environment variables
with the original fixed defaults as fallback — lets
`aws codebuild start-build --environment-variables-override ...` park
deliberately different candidates per run (Gap #4), not just re-runs of
identical config. See scripts/trigger_training_variant.sh.

The base model to fine-tune from comes from one of two sources, resolved
at pipeline-build time, never a bare Ultralytics checkpoint name pulled
fresh from Ultralytics' own hub:
  - By default, the latest Approved package in whichever
    generation+size's Model Package Group TRAIN_GENERATION/TRAIN_SIZE
    selects (config.py's model_package_group_name: uniform
    yolo<generation><size>-detector naming for every generation, created
    on demand here if it doesn't exist yet). If that group has no Approved
    package,
    build_pipeline() raises before any Estimator/TrainingStep is even
    constructed, so a misconfigured registry never wastes GPU billing on
    a job that was going to fail inside the container anyway (train.py
    repeats this same Approved check as defense-in-depth).
  - If BASE_MODEL_S3_URI is set instead, that exact S3 object (a
    model.tar.gz or a bare .pt) is used directly, bypassing the registry
    entirely -- the escape hatch for a generation/size that's never been
    trained/registered yet, e.g. a checkpoint uploaded via the
    review-app's "Upload Another YOLO Model" section. review_app.py's
    /train/dataset route resolves a chosen S3 folder down to this exact
    object key before triggering the build.

Every pipeline execution gets its own SageMaker Experiments trial
(pipeline_experiment_config below), and EvaluateStep logs its metrics
into that trial explicitly (not via load_run() auto-detection — that has
a documented SDK bug inside ProcessingSteps specifically, see
evaluate.py's own docstring) — makes candidates comparable side-by-side
in SageMaker Studio's Experiments view (Gap #5).
"""

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import boto3
from sagemaker.estimator import Estimator
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.functions import Join
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline_experiment_config import PipelineExperimentConfig
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

from config import CONFIG, GENERATION_SIZES, model_package_group_name

EXPERIMENT_NAME = "track2-yolo-training"
# Absolute, not "evaluate.py" -- ProcessingStep resolves a relative `code`
# path against the current working directory at run_pipeline.py invocation
# time, not against this file's location. Confirmed live: running
# `python run_pipeline.py` from src/ (as its own usage docstring says to)
# raised "code evaluate.py wasn't found" since evaluate.py actually lives
# in the sibling training-container/ directory, not src/.
#
# Two different layouts this file can find itself running from:
#   - full local git checkout: this file is at src/pipeline_definition.py,
#     evaluate.py is a sibling-directory file at ../training-container/evaluate.py
#   - track2-yolo-run-pipeline CodeBuild's run-pipeline-src.zip: flat --
#     evaluate.py sits right next to this file, no training-container/
#     subfolder at all (see infra/codebuild_pipeline.tf)
# Try the local-checkout layout first, fall back to the flat one, so
# build_pipeline() works unmodified from either.
_EVALUATE_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "training-container" / "evaluate.py",
    Path(__file__).resolve().parent / "evaluate.py",
]
_evaluate_script = next((p for p in _EVALUATE_CANDIDATES if p.is_file()), None)
if _evaluate_script is None:
    raise FileNotFoundError(
        f"evaluate.py not found in any of: {[str(p) for p in _EVALUATE_CANDIDATES]}"
    )

# .as_uri() (a "file:///C:/..." URI), not a plain Windows path string:
# the SDK's own code-path handler runs the string through urlparse() to
# decide "local file" vs "s3://" -- a plain "C:\..." path confused it into
# reading "C" itself as a URL scheme ("code ... url scheme c is not
# recognized"), since urlparse treats anything before a colon as a
# candidate scheme. "file://" is one of the two schemes urlparse
# correctly resolves back to a real local path (the other being "s3").
EVALUATE_SCRIPT_PATH = _evaluate_script.as_uri()


def _resolve_model_package_group_name(generation: str, size: str) -> str:
    """Delegates to config.py's model_package_group_name -- uniform
    yolo<generation><size>-detector naming for every generation, including
    generation 11 (whose Terraform-declared groups in infra/sagemaker.tf
    use this exact pattern; every other generation's group is created on
    demand below, the first time that generation/size is trained)."""
    return model_package_group_name(generation, size)


def _ensure_model_package_group_exists(region: str, group_name: str) -> None:
    """Idempotent: Model Package Groups aren't pre-declared in Terraform
    for every possible generation/size combination (16+ generations x 4
    sizes would mean dozens of groups, most never used) -- created on
    demand here instead, the first time a generation/size is trained."""
    sm = boto3.client("sagemaker", region_name=region)
    try:
        sm.describe_model_package_group(ModelPackageGroupName=group_name)
    except sm.exceptions.ClientError as exc:
        if "does not exist" not in str(exc):
            raise
        sm.create_model_package_group(
            ModelPackageGroupName=group_name,
            ModelPackageGroupDescription=f"YOLO candidates for {group_name}, created on demand by pipeline_definition.py",
        )
        print(f"Created Model Package Group {group_name!r} (didn't exist yet).")


def _resolve_latest_approved_base_model(region: str, model_package_group_name: str) -> str:
    """Looks up the latest Approved model package in the group -- the only
    supported source for a fine-tuning base model, no fallback. Raises if
    the group has no Approved package at all."""
    sm = boto3.client("sagemaker", region_name=region)
    response = sm.list_model_packages(
        ModelPackageGroupName=model_package_group_name,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
        MaxResults=1,
    )
    packages = response["ModelPackageSummaryList"]
    if not packages:
        raise RuntimeError(
            f"No registered model found in Model Package Group '{model_package_group_name}' -- "
            "register and approve a base model before training. Nothing to fine-tune from."
        )
    arn = packages[0]["ModelPackageArn"]
    print(f"Resolved base model for fine-tuning: {arn} (latest Approved in '{model_package_group_name}')")
    return arn


DYNAMODB_LINEAGE_TABLE = "track2-dynamoDB"  # pre-existing, reused table (hash key "id") -- see summary.md's 2026-09-07 entry


def _readable_hyperparams(blob: str) -> dict:
    """The blob as a plain dict, for the run record. {} if it will not decode."""
    if not blob:
        return {}
    for attempt in (lambda: json.loads(base64.b64decode(blob).decode("utf-8")),
                    lambda: json.loads(blob)):
        try:
            got = attempt()
            if isinstance(got, dict):
                return got
        except Exception:                                          # noqa: BLE001
            continue
    return {}


def _write_run_metadata(
    *,
    region: str,
    bucket: str,
    generation: str,
    size: str,
    model_package_group_name: str,
    train_data_s3_uri: str,
    base_model_package_arn: str | None,
    base_model_s3_uri: str | None,
    epochs: int,
    img_size: int,
    batch: int,
    seed: int,
    notes: str | None,
) -> tuple[str, str]:
    """Writes one lineage record per training run to BOTH
    s3://<bucket>/training-runs/<run_id>/run_metadata.json (the durable,
    human-readable full copy) and DynamoDB (DYNAMODB_LINEAGE_TABLE, for
    fast indexed lookup -- see review_app.py's registry rows, which
    resolve a package's "lineage_id" CustomerMetadataProperty straight to
    a get_item instead of an S3 fetch+parse). Answers "what dataset/YOLO/
    checkpoint/notes went into this candidate", since none of that is
    otherwise recoverable from the trained artifact itself. Returns
    (s3_uri, run_id) -- the caller cross-links run_id onto the registered
    model package as "lineage_id" (kept short there, not the whole
    record, since CustomerMetadataProperties values are size-limited and
    a free-text note could be long)."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"yolo{generation}-{size}-{timestamp}"
    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "yolo": {
            "generation": generation,
            "size": size,
            "model_package_group_name": model_package_group_name,
        },
        "dataset": {"s3_uri": train_data_s3_uri},
        "base_model": {
            "source": "registry" if base_model_package_arn else "s3",
            "model_package_arn": base_model_package_arn,
            "s3_uri": base_model_s3_uri,
        },
        "hyperparameters": {
            "epochs": epochs, "img_size": img_size, "batch": batch, "seed": seed,
            # Readable here even though it travels encoded -- an audit row
            # saying "hyperparams: eyJvcHRpbWl6ZXIi..." answers nothing
            # anybody would ask it.
            **_readable_hyperparams(os.environ.get("TRAIN_HYPERPARAMS", "").strip()),
        },
        "notes": notes,
    }
    key = f"training-runs/{run_id}/run_metadata.json"
    s3_uri = f"s3://{bucket}/{key}"
    boto3.client("s3", region_name=region).put_object(
        Bucket=bucket, Key=key, Body=json.dumps(record, indent=2).encode(), ContentType="application/json"
    )
    print(f"Wrote run metadata -> {s3_uri}")

    dynamodb = boto3.resource("dynamodb", region_name=region)
    # "id" is the table's actual (pre-existing) hash key -- "run_id" is
    # kept too, as the same human-readable field name the S3 copy uses.
    dynamodb.Table(DYNAMODB_LINEAGE_TABLE).put_item(Item={**record, "id": run_id, "record_type": "training_run"})
    print(f"Wrote run lineage -> DynamoDB {DYNAMODB_LINEAGE_TABLE}/{run_id}")

    return s3_uri, run_id


def build_pipeline(pipeline_session: PipelineSession) -> Pipeline:
    # Real hazard-detection data (6-class: helmet/gloves/person/
    # working_boots/man_cage/safe_area), synced from the external
    # escalation-dataset pull pipeline via the track2-sagemaker-pull-training
    # Lambda -- not the old images/dataset/ synthetic-box placeholder
    # seed_training_data.py generates. TrainingStep and EvaluateStep both
    # read data.yaml from here, so both automatically pick up whatever
    # class taxonomy this data currently has -- no code change needed on
    # either side when it changes (confirmed: evaluate.py/train.py never
    # hardcode class names, they read data.yaml directly).
    # Gap #4: deliberate variation per run, via env vars a CodeBuild
    # --environment-variables-override can set — falls back to the
    # original fixed values when not overridden. Base model is NOT one of
    # these -- see _resolve_latest_approved_base_model above.
    #
    # TRAIN_DATA_S3_URI defaults to the real escalation-dataset location,
    # same as always -- overriding it (e.g. to seed_training_data.py's
    # s3://track2-s3/images/dataset/ synthetic-box placeholder) is how a
    # one-off dummy/proof-of-mechanics run is done without touching the
    # production data source for every other run.
    train_data_s3_uri = os.environ.get(
        "TRAIN_DATA_S3_URI",
        f"s3://{CONFIG.pipeline_bucket}/data/track2-datasets/sagemaker_training/track2/escalation_dataset/",
    )
    # EVAL_DATA_S3_URI lets the EvaluateStep measure against a dataset the
    # model was NOT trained on. Unset, it falls back to train_data_s3_uri,
    # which is what this pipeline has always done -- so metrics from a run
    # without it are training-set metrics, not held-out ones. Defaulting
    # rather than requiring it keeps every existing caller working.
    eval_data_s3_uri = os.environ.get("EVAL_DATA_S3_URI") or train_data_s3_uri
    epochs = int(os.environ.get("TRAIN_EPOCHS", "20"))
    img_size = int(os.environ.get("TRAIN_IMG_SIZE", "640"))
    batch = int(os.environ.get("TRAIN_BATCH", "16"))
    seed = int(os.environ.get("TRAIN_SEED", "42"))
    # Opaque on purpose: it travels as a SageMaker hyperparameter, which
    # becomes a command-line argument, and a JSON object through that
    # layer is a quoting problem that presents as the parameter being
    # silently ignored. train.py decodes and validates it.
    hyperparams_blob = os.environ.get("TRAIN_HYPERPARAMS", "").strip()

    # Which YOLO size this run trains/registers -- selects a Model
    # Package Group via generation+size (defaults to "11"/"s", the
    # original/only combination before per-generation/size groups
    # existed). Generation must be one of GENERATION_SIZES' keys (a real,
    # officially-released Ultralytics generation); the group itself is
    # created on demand below if it doesn't exist yet.
    train_generation = os.environ.get("TRAIN_GENERATION", "11")
    train_size = os.environ.get("TRAIN_SIZE", "s")
    # Validated against which sizes are offered for the selected generation
    # (config.py's GENERATION_SIZES: YOLO11 only, s/m/l/x only -- see its
    # own comment for why those two policies exist) -- not the fixed
    # 4-letter SIZES tuple, which is specifically the four already-deployed
    # serving Inference Components (a separate, narrower concept from
    # "valid sizes to train").
    valid_sizes = GENERATION_SIZES.get(train_generation)
    if valid_sizes is None:
        raise ValueError(
            f"TRAIN_GENERATION {train_generation!r} isn't a recognized YOLO generation "
            f"(config.py's GENERATION_SIZES has: {sorted(GENERATION_SIZES)})."
        )
    if train_size not in valid_sizes:
        raise ValueError(f"TRAIN_SIZE must be one of {valid_sizes} for generation {train_generation!r}, got {train_size!r}")
    model_package_group_name = _resolve_model_package_group_name(train_generation, train_size)
    _ensure_model_package_group_exists(CONFIG.region, model_package_group_name)

    # Base model: BASE_MODEL_S3_URI bypasses the registry entirely (for a
    # generation/size with nothing Approved yet -- see module docstring);
    # otherwise the usual latest-Approved-in-group lookup, which still
    # raises before any Estimator/TrainingStep is built if the group (now
    # guaranteed to exist, but not necessarily populated) has no Approved
    # package.
    base_model_s3_uri = os.environ.get("BASE_MODEL_S3_URI")
    base_model_package_arn = None
    if base_model_s3_uri:
        base_model_hyperparameter = {"base-model-s3-uri": base_model_s3_uri}
        print(f"Using base model directly from S3 (bypassing the registry): {base_model_s3_uri}")
    else:
        base_model_package_arn = _resolve_latest_approved_base_model(CONFIG.region, model_package_group_name)
        base_model_hyperparameter = {"base-model-package-arn": base_model_package_arn}

    # Optional free-text note on why this run happened -- the review-app's
    # Train form passes it through as this env var. Saved both on the
    # run-metadata record below and directly as the registered model
    # package's own description field, so it's readable from the
    # registry itself long after the run, not just in this build's logs.
    train_notes = os.environ.get("TRAIN_NOTES", "") or None

    run_metadata_uri, run_id = _write_run_metadata(
        region=CONFIG.region,
        bucket=CONFIG.pipeline_bucket,
        generation=train_generation,
        size=train_size,
        model_package_group_name=model_package_group_name,
        train_data_s3_uri=train_data_s3_uri,
        base_model_package_arn=base_model_package_arn,
        base_model_s3_uri=base_model_s3_uri,
        epochs=epochs,
        img_size=img_size,
        batch=batch,
        seed=seed,
        notes=train_notes,
    )

    estimator = Estimator(
        image_uri=CONFIG.training_image_uri,
        role=CONFIG.sagemaker_execution_role_arn,
        instance_type=CONFIG.training_instance_type,
        instance_count=1,
        output_path=f"s3://{CONFIG.pipeline_bucket}/training-output/",
        sagemaker_session=pipeline_session,
        hyperparameters={
            "epochs": epochs,
            "img-size": img_size,
            "batch": batch,
            "seed": seed,
            **({"hyperparams": hyperparams_blob} if hyperparams_blob else {}),
            # No SageMaker input channel for the dataset (see TrainingStep
            # below) -- train.py downloads it itself via this hyperparameter,
            # specifically so it can print real byte-progress (see
            # download_dataset_with_progress's own docstring for why
            # SageMaker's own automatic channel download can't give that).
            "train-data-s3-uri": train_data_s3_uri,
            **base_model_hyperparameter,
        },
        # Scrapes two of train.py's own print lines into real CloudWatch
        # metrics so the review-app can show live progress during TrainYolo,
        # not just the step's Executing/Succeeded status:
        #  - train:epoch, from Ultralytics' own per-epoch console table
        #    ("      Epoch    GPU_mem   box_loss ..."). Regex verified
        #    against a real local training run's actual captured stdout
        #    (CPU: "1/2  0G  ..."), then checked against the documented
        #    GPU-instance format ("7/20  2.31G  ...") -- (?:\.\d+)? makes
        #    the GPU-memory decimal optional so both match. Captures ONLY
        #    the numerator (current epoch); the denominator (total epochs)
        #    is already known separately, from this same job's own
        #    "epochs" hyperparameter (DescribeTrainingJob), not scraped.
        #  - dataset:total_images / dataset:images_downloaded, from
        #    download_dataset_with_progress's own "DATASET_TOTAL_IMAGES: N"
        #    (printed once) and "DATASET_IMAGES: n/N" (printed after each
        #    image file finishes) print lines -- happens before train:epoch
        #    ever starts, so the review-app shows this first ("n / N images
        #    loaded"), then epoch progress once it appears. Two separate
        #    metrics, not one: a CloudWatch metric_definitions regex
        #    captures exactly one numeric group, so the running count and
        #    its fixed denominator need their own regex/metric each (same
        #    reason train:epoch's total comes from DescribeTrainingJob's
        #    hyperparameters instead of a second regex).
        metric_definitions=[
            {"Name": "train:epoch", "Regex": r"(\d+)/\d+\s+\d+(?:\.\d+)?G"},
            {"Name": "dataset:total_images", "Regex": r"DATASET_TOTAL_IMAGES:\s+(\d+)"},
            {"Name": "dataset:images_downloaded", "Regex": r"DATASET_IMAGES:\s+(\d+)/\d+"},
        ],
    )

    training_step = TrainingStep(
        name="TrainYolo",
        estimator=estimator,
    )

    # --- EvaluateStep: real metrics before registration (Gap #1) ---------

    evaluator = ScriptProcessor(
        image_uri=CONFIG.training_image_uri,  # reuse train-latest, already has ultralytics
        command=["python3"],
        role=CONFIG.sagemaker_execution_role_arn,
        # CPU, not CONFIG.training_instance_type (ml.g4dn.xlarge): this
        # account's SageMaker Processing Job quota for ml.g4dn.xlarge is 0
        # (confirmed live via Service Quotas -- separate and independent
        # from the Training Job quota, which does have headroom). evaluate.py
        # only runs Ultralytics' model.val() (inference, no backprop) on 17
        # validation images -- CPU is slower but entirely adequate at this
        # scale, and ml.m5.xlarge has real quota headroom (16) in this account.
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=pipeline_session,
        env={
            "EXPERIMENT_NAME": EXPERIMENT_NAME,
            "RUN_NAME": ExecutionVariables.PIPELINE_EXECUTION_ID,
        },
    )

    evaluate_step = ProcessingStep(
        name="EvaluateYolo",
        processor=evaluator,
        code=EVALUATE_SCRIPT_PATH,
        inputs=[
            ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                # The evaluation set, which is train_data_s3_uri unless
                # EVAL_DATA_S3_URI was given -- evaluate.py reads data.yaml
                # from whatever lands here, so a separate evaluation set
                # needs its own data.yaml, same layout as a training one.
                source=eval_data_s3_uri,
                destination="/opt/ml/processing/data",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output",
                # Explicit destination -- without this, SageMaker silently
                # falls back to its own default bucket (sagemaker-<region>-
                # <account>) instead of the project's own track2-s3, which
                # is where every other output (training-output/, async-
                # output/, the dataset) already lives.
                destination=f"s3://{CONFIG.pipeline_bucket}/evaluation-output/",
            ),
        ],
    )

    evaluation_s3_uri = (
        evaluate_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri
    )

    model_metrics = ModelMetrics(
        # "model_quality" doesn't exist as a param in installed sagemaker==2.257.6
        # (confirmed via inspect.signature) -- "model_statistics" is the
        # correct one; it populates the API request's ModelQuality.Statistics
        # field, which is exactly what this is (mAP/precision/recall JSON).
        model_statistics=MetricsSource(
            # evaluation_s3_uri is a PipelineVariable (resolved only at
            # execution time) -- an f-string here raises TypeError by
            # design (SageMaker SDK blocks __str__ on PipelineVariables
            # specifically to catch this mistake). Join is the SDK-provided
            # way to concatenate one with a literal at definition time.
            s3_uri=Join(on="/", values=[evaluation_s3_uri, "evaluation.json"]),
            content_type="application/json",
        )
    )

    model = Model(
        image_uri=CONFIG.serving_image_uri,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        role=CONFIG.sagemaker_execution_role_arn,
        sagemaker_session=pipeline_session,
    )

    register_step = ModelStep(
        name="RegisterYoloModel",
        step_args=model.register(
            content_types=["image/jpeg"],
            response_types=["application/json"],
            inference_instances=["ml.g4dn.xlarge"],
            transform_instances=["ml.g4dn.xlarge"],
            model_package_group_name=model_package_group_name,
            approval_status="PendingManualApproval",
            model_metrics=model_metrics,
            description=train_notes,
            # Just pointers, not the whole record -- CustomerMetadataProperties
            # values are size-limited and a free-text note could be long.
            # run_metadata_uri: the full run_metadata.json is one S3
            # GetObject away. lineage_id: review_app.py resolves this
            # straight to a DynamoDB get_item (DYNAMODB_LINEAGE_TABLE) for
            # the same record without the S3 round-trip.
            customer_metadata_properties={"run_metadata_uri": run_metadata_uri, "lineage_id": run_id},
        ),
    )

    return Pipeline(
        name="track2-yolo-training-pipeline",
        steps=[training_step, evaluate_step, register_step],
        sagemaker_session=pipeline_session,
        pipeline_experiment_config=PipelineExperimentConfig(
            experiment_name=EXPERIMENT_NAME,
            trial_name=ExecutionVariables.PIPELINE_EXECUTION_ID,
        ),
    )
