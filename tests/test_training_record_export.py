"""The training record export: sections a person can read, and every
setting the job was sent -- hyperparameters included (2026-10-26)."""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402
from ppe_pipeline_api.hyperparams import GROUPS, SCHEMA  # noqa: E402

AUDIT = {
    "id": "r1", "status": "Succeeded", "notes": "a run", "detector_slug": "helmet",
    "architecture": "yolo11s", "parameters": {"epochs": 5, "img_size": 640, "batch": 16,
                                              "seed": 42, "patience": 20},
    "hyperparams": {"lr0": 0.005, "optimizer": "AdamW", "mosaic": 0.0},
    "training_options": {"classes": [0], "single_cls": False, "drop_undeclared_classes": True},
    "model": {"version": 3}, "metrics": {"map50": 0.71},
    "train_dataset": {"folder": "d"}, "eval_dataset": {"folder": "d"},
}


def _csv(monkeypatch, audit=AUDIT):
    monkeypatch.setattr(t2, "read_file_row_paths", lambda audit_id, prefix: [])
    return t2.audit_as_csv(audit)


def test_it_reads_in_sections(monkeypatch):
    text = _csv(monkeypatch)
    for title in ("RUN", "MODEL", "TRAINING SETTINGS", "HYPERPARAMETERS", "CLASS OPTIONS",
                  "TRAINING DATA", "EVALUATION DATA", "METRICS"):
        assert f"\n{title}\n" in text, title


def test_every_hyperparameter_is_listed_set_or_default(monkeypatch):
    text = _csv(monkeypatch)
    assert "Optimiser,lr0,0.005" in text
    assert "Optimiser,optimizer,AdamW" in text
    assert "Augmentation,mosaic,0.0" in text
    assert "Optimiser,momentum,default" in text
    assert "3 changed from Ultralytics' default" in text


def test_the_groups_cover_the_whole_schema():
    grouped = [k for keys in GROUPS.values() for k in keys]
    assert sorted(grouped) == sorted(SCHEMA)


def test_patience_and_class_options_are_shown(monkeypatch):
    text = _csv(monkeypatch)
    assert "Patience (early stopping),20" in text
    assert "Classes trained (indices),0" in text
    assert "Undeclared boxes dropped,yes" in text


def test_an_old_run_says_not_recorded_rather_than_default(monkeypatch):
    old = {k: v for k, v in AUDIT.items() if k not in ("hyperparams", "training_options")}
    text = _csv(monkeypatch, old)
    assert "Not recorded for this run" in text


def test_older_runs_are_read_back_from_their_execution(monkeypatch):
    import base64
    import json
    blob = base64.b64encode(json.dumps({"lr0": 0.002}).encode()).decode()
    monkeypatch.setattr(t2, "sm", type("Sm", (), {
        "list_pipeline_parameters_for_execution": lambda self, **kw: {"PipelineParameters": [
            {"Name": "Hyperparams", "Value": blob}, {"Name": "Patience", "Value": "7"},
            {"Name": "Classes", "Value": "0"}, {"Name": "SingleCls", "Value": "false"}]}})())
    got = t2.run_settings_for({"execution_arn": "arn:x"})
    assert got["hyperparams"] == {"lr0": 0.002} and got["patience"] == 7
    assert got["training_options"]["classes"] == ["0"]


def test_new_runs_record_what_they_were_sent():
    sig = inspect.signature(t2.write_training_audit).parameters
    assert {"patience", "hyperparams", "training_options"} <= set(sig)
    body = inspect.getsource(t2.write_training_audit)
    assert '"hyperparams": _decimalize(normalize_hyperparams(hyperparams))' in body
    src = inspect.getsource(t2.start_custom_training_run)
    assert "patience=patience, hyperparams=hyperparams," in src
    assert '"drop_undeclared_classes": bool(drop_undeclared_classes),' in src


def test_the_export_route_reads_older_runs_back():
    from ppe_pipeline_api.routers import track2_models as tm
    assert "t2.audit_as_csv(audit, t2.run_settings_for(audit))" in inspect.getsource(tm.export_training_audit)


def test_a_versions_list_files_are_not_listed_as_training_files(monkeypatch):
    monkeypatch.setattr(t2, "read_file_row_paths", lambda audit_id, prefix: [
        "d/images/train/a.jpg", "_datasets/dv-1/train/train.txt", "_datasets/dv-1/eval/val.txt"])
    text = t2.audit_as_csv({"id": "x", "train_dataset": {}, "eval_dataset": {}})
    assert "TRAINING FILES (1 files)" in text and "train.txt" not in text
