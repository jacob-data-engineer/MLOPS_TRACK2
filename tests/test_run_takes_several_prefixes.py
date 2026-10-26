"""`start_custom_training_run` can be told several prefixes and a yaml.

A composed dataset's folder -- `_generated/<hash>` -- holds the data.yaml
and nothing else, so neither the training data nor the yaml can be
derived from it the way every run so far has derived both. Both are
named explicitly instead, and both are blank for an ordinary run, which
keeps that path byte-for-byte what it was.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402


def _src() -> str:
    return " ".join(inspect.getsource(t2.start_custom_training_run).split())


def test_the_arguments_exist():
    sig = inspect.signature(t2.start_custom_training_run).parameters
    assert sig["train_data_uris"].default == ""
    assert sig["data_yaml_uri"].default == ""


def test_several_prefixes_win_over_the_folder():
    assert 'train_uri = (str(train_data_uris or "").strip() or f"s3://{BUCKET}/' in _src()


def test_an_ordinary_run_still_derives_it_from_the_folder():
    """Blank means "as before" -- not "no training data"."""
    assert "or f\"s3://{BUCKET}/{dataset_folder.strip('/')}/\"" in _src()


def test_the_yaml_uri_is_a_pipeline_parameter():
    assert '"DataYamlUri": str(data_yaml_uri or "").strip(),' in _src()


def test_a_blank_yaml_uri_is_omitted_rather_than_sent_empty():
    """`PipelineParameters[].Value` has a service minimum of ONE
    character, and that is checked before anything else."""
    src = _src()
    assert 'sent = {k: v for k, v in parameters.items() if str(v) != ""}' in src
    assert src.index('"DataYamlUri"') < src.index("sent = {k: v")


def test_an_undeclared_parameter_is_refused_with_the_fix_named():
    """Rather than a ValidationException that reads like the run itself
    was malformed."""
    src = _src()
    assert "declared = pipeline_parameter_names(expect=" in src
    assert "The training pipeline does not declare" in src
    assert "aws sagemaker update-pipeline" in src


def test_a_stale_cache_does_not_outlive_an_update_pipeline():
    """The cache lives for the life of the process and what it caches
    changes when somebody runs `update-pipeline`. Without the re-read,
    a parameter added at 12:00 stays refused by every warm container --
    and the refusal names the very command that was just run."""
    src = inspect.getsource(t2.pipeline_parameter_names)
    assert "expect and expect not in _PIPELINE_PARAMETER_NAMES.get(name, {expect})" in src
    assert "_PIPELINE_PARAMETER_NAMES.pop(name, None)" in src


def test_the_re_read_only_happens_on_the_path_about_to_refuse():
    """One extra describe per optional parameter actually being sent, and
    only when the cached answer lacks that name."""
    src = _src()
    assert 'for optional in ("DataYamlUri", "DropUndeclaredClasses", "EvalExtraUris"):' in src
    assert "if sent.get(optional): declared = pipeline_parameter_names(expect=optional)" in src


def test_a_pipeline_that_cannot_be_read_refuses_nothing():
    src = inspect.getsource(t2.pipeline_parameter_names)
    assert "return set()" in src
    assert "Unknown, not empty" in src


def test_the_answer_is_cached():
    """A describe call per queued run, for something that changes only
    when somebody runs update-pipeline."""
    src = inspect.getsource(t2.pipeline_parameter_names)
    assert "_PIPELINE_PARAMETER_NAMES" in src


def test_a_composed_dataset_cannot_be_class_filtered_by_copying():
    """`derive_filtered_dataset` copies images/ and labels/ from UNDER
    the dataset folder. A composed one has nothing under it but the
    yaml, so the copy would succeed and produce an empty dataset."""
    src = _src()
    assert 'if class_filter.get("derive") and str(train_data_uris or "").strip():' in src
    assert "composed from several folders" in src


def test_the_queue_carries_both_to_the_runner():
    src = " ".join(inspect.getsource(t2.try_advance_queue).split())
    assert 'train_data_uris=entry.get("train_data_uris", "")' in src
    assert 'data_yaml_uri=entry.get("data_yaml_uri", "")' in src
