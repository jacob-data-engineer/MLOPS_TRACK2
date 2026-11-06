"""Which run produced each registered version -- shown beside the version in
"Train on an existing model", and as Model name / type / size / version in
the Data Catalog's Training runs table, so the right .pt can be found."""

import pathlib

from ppe_pipeline_api import main
from ppe_pipeline_api.data import catalog_view

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PICKER = (WEB / "components" / "Track2BaseModelPicker.tsx").read_text(encoding="utf-8")
CATALOG = (WEB / "routes" / "Track2CatalogPage.tsx").read_text(encoding="utf-8")


def test_the_route_exists():
    assert "/track2/catalog/runs-by-model" in main.app.openapi()["paths"]


def test_no_catalog_means_no_labels_not_an_error(monkeypatch):
    monkeypatch.setattr(catalog_view.catalog, "configured", lambda: False)
    assert catalog_view.runs_by_model() == {}


def test_keyed_on_the_package_arn(monkeypatch):
    monkeypatch.setattr(catalog_view.catalog, "configured", lambda: True)
    monkeypatch.setattr(catalog_view, "_rows", lambda sql, params=(): [
        {"model_package_arn": "arn:x/yolo11s-helmet-detector/32", "run_number": 58,
         "run_id": "r58", "submitted_at": None, "status": "Succeeded"}])
    got = catalog_view.runs_by_model()
    assert got == {"arn:x/yolo11s-helmet-detector/32":
                   {"run_number": 58, "run_id": "r58", "submitted_at": None, "status": "Succeeded"}}


def test_the_picker_does_not_download_the_training_history():
    """The history carries every run's file lists (~10 MB for sixty runs);
    the column sat on dashes until all of it arrived."""
    assert "getTrack2RunsByModel" in PICKER
    assert "getTrack2TrainingAudit" not in PICKER
    assert "Training run" in PICKER


def test_the_catalog_names_the_model_the_way_the_models_page_does():
    for header in ('"Model name"', '"Model type"', '"Size"', '"Model version"'):
        assert header in CATALOG
