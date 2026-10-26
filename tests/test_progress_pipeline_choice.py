"""The progress panel must never report one pipeline's result as another's.

Live on 2026-10-06: a custom run failed at EvaluateYolo, and the Models
page displayed "Succeeded -- Epoch 0 / 50". Those numbers belonged to the
PRODUCTION pipeline's last execution, four days old, on a different
taxonomy. The old rule was "report the custom pipeline WHILE a custom run
is Executing", which stops being true the instant the run ends -- so the
panel fell back and certified a failure as a success.
"""
from __future__ import annotations

import datetime as dt
import importlib

t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")


def _at(day, hour=0):
    return dt.datetime(2026, 10, day, hour, tzinfo=dt.timezone.utc)


def _fake_sm(monkeypatch, newest_by_pipeline):
    def list_pipeline_executions(PipelineName, **kw):
        started = newest_by_pipeline.get(PipelineName)
        if started is None:
            return {"PipelineExecutionSummaries": []}
        return {"PipelineExecutionSummaries": [{
            "PipelineExecutionArn": f"arn:...:pipeline/{PipelineName}/execution/x",
            "PipelineExecutionStatus": "Failed",
            "StartTime": started,
        }]}
    monkeypatch.setattr(t2.sm, "list_pipeline_executions", list_pipeline_executions)


def test_the_newer_custom_run_wins(monkeypatch):
    _fake_sm(monkeypatch, {t2.PIPELINE_NAME: _at(2), t2.OPEN_PIPELINE_NAME: _at(6)})
    assert t2.progress_pipeline_name() == t2.OPEN_PIPELINE_NAME


def test_a_finished_custom_run_still_wins(monkeypatch):
    """The regression. The run is over -- Failed, not Executing -- and it
    is still the most recent answer to "what happened"."""
    _fake_sm(monkeypatch, {t2.PIPELINE_NAME: _at(2), t2.OPEN_PIPELINE_NAME: _at(6, 9)})
    assert t2.progress_pipeline_name() == t2.OPEN_PIPELINE_NAME


def test_a_newer_production_run_takes_it_back(monkeypatch):
    """Not a one-way switch: the newer answer is the right one either way."""
    _fake_sm(monkeypatch, {t2.PIPELINE_NAME: _at(7), t2.OPEN_PIPELINE_NAME: _at(6)})
    assert t2.progress_pipeline_name() == t2.PIPELINE_NAME


def test_no_custom_executions_at_all(monkeypatch):
    _fake_sm(monkeypatch, {t2.PIPELINE_NAME: _at(2), t2.OPEN_PIPELINE_NAME: None})
    assert t2.progress_pipeline_name() == t2.PIPELINE_NAME


def test_nothing_anywhere_falls_back_to_production(monkeypatch):
    _fake_sm(monkeypatch, {})
    assert t2.progress_pipeline_name() == t2.PIPELINE_NAME


def test_a_listing_that_raises_does_not_break_the_page(monkeypatch):
    def boom(**kw):
        raise RuntimeError("no")
    monkeypatch.setattr(t2.sm, "list_pipeline_executions", boom)
    assert t2.progress_pipeline_name() == t2.PIPELINE_NAME


def test_the_models_endpoint_uses_the_chooser():
    """Not `OPEN_PIPELINE_NAME if custom_run_id else ""` -- that is the
    expression that caused it."""
    import inspect
    models = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
    source = inspect.getsource(models.get_models)
    assert "progress_pipeline_name()" in source
    assert "OPEN_PIPELINE_NAME if custom_run_id" not in source


def test_the_live_endpoint_uses_it_too():
    import inspect
    models = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
    source = inspect.getsource(models.get_models_live)
    assert "progress_pipeline_name()" in source
    # The CODE, not the docstring -- which names the heavy keys to say
    # why they are deliberately absent.
    body = source.split('"""', 2)[-1]
    for key in ("gpu_jobs", "queue", "training_progress"):
        assert key in body, key
    for heavy in ("dataset_browser", "registries", "list_s3_folders"):
        assert heavy not in body, f"{heavy} makes this too expensive to poll"
