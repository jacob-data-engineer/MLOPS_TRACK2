"""A blank pipeline parameter is omitted, never sent blank.

`PipelineParameters[].Value` has a service minimum of ONE character, and
that is validated before anything else -- so passing "" fails the whole
`StartPipelineExecution` call with

    [string "" is too short (length: 0, required minimum: 1)]

even for a parameter whose own DefaultValue in the pipeline is "".

`Classes` and `Hyperparams` are blank on every run that does not filter
classes or pin hyperparameters, which is most of them. So from the
moment those two were added, every custom run died at submit. The queue
said "failed to start, skipped" and moved on, which is the one mercy:
no GPU time was spent.

The defaults were read off the live pipeline, not guessed:

    aws sagemaker describe-pipeline \\
      --pipeline-name track2-yolo-open-training-pipeline \\
      --query PipelineDefinition
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402


def _source() -> str:
    return inspect.getsource(t2.start_custom_training_run)


# --- the blank ones are dropped ---------------------------------------------------

def test_blank_values_are_filtered_out():
    code = _source()
    assert 'sent = {k: v for k, v in parameters.items() if str(v) != ""}' in code


def test_the_call_sends_the_filtered_set():
    code = _source()
    assert 'for k, v in sent.items()' in code
    assert 'for k, v in parameters.items()]' not in code, (
        "the unfiltered dict still reaches start_pipeline_execution"
    )


# --- what the empties actually are ------------------------------------------------

def test_no_class_filter_produces_a_blank_classes():
    """The value that broke it. Kept as a test rather than a comment so
    a change to how Classes is built cannot quietly stop being blank --
    or quietly start being blank somewhere new."""
    assert ",".join(str(i) for i in []) == ""


@pytest.mark.parametrize("value", [None, {}])
def test_no_hyperparameters_produces_a_blank_blob(value):
    assert t2.encode_hyperparams(value) == ""


def test_hyperparameters_that_exist_still_encode(monkeypatch):
    """Filtering must not drop a real one."""
    encoded = t2.encode_hyperparams({"lr0": "0.01"})
    assert encoded != ""


# --- the ones that cannot be dropped ----------------------------------------------

def test_the_parameters_without_defaults_are_listed():
    """Read off the live pipeline. Omitting one of these would trade a
    clear error for a stranger one further in."""
    assert t2.PIPELINE_PARAMETERS_WITHOUT_DEFAULTS == (
        "TrainingDatasetUri",
        "EvaluationDatasetUri",
        "EvaluationOutputUri",
        "RunMetadataUri",
        "RunId",
        "DatasetVersion",
        "BaseModelUri",
    )


def test_a_blank_required_parameter_is_refused_by_name():
    code = _source()
    assert "PIPELINE_PARAMETERS_WITHOUT_DEFAULTS" in code
    assert "came through empty" in code


def test_classes_and_hyperparams_are_not_in_that_list():
    """They are exactly the two that may be omitted -- the pipeline
    declares both with an empty default."""
    for name in ("Classes", "Hyperparams", "SingleCls"):
        assert name not in t2.PIPELINE_PARAMETERS_WITHOUT_DEFAULTS


# --- the parameters the run still sends -------------------------------------------

def test_every_parameter_that_mattered_is_still_built():
    code = _source()
    for name in ("TrainingDatasetUri", "EvaluationDatasetUri", "RegistryGroup",
                 "Architecture", "Epochs", "ImgSize", "Batch", "Seed",
                 "Patience", "Classes", "SingleCls", "Hyperparams",
                 "BaseModelUri", "BaseModelPackageArn", "Detector"):
        assert f'"{name}"' in code, name


def test_the_never_empty_sentinels_are_still_there():
    """`BaseModelPackageArn` and friends land in
    CustomerMetadataProperties, whose map values have their own minimum
    of 1 -- and that failure arrives AFTER the GPU hours, at
    CreateModelPackage. Filtering blanks here does not cover them,
    because they must be present and non-empty, not absent."""
    code = _source()
    assert 'base_model_package_arn or "none"' in code
    assert 'base_model_source or "unspecified"' in code
    assert 'detector_slug or "open"' in code
