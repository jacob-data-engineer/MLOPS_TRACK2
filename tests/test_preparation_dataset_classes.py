"""Preparation preserves the dataset taxonomy; training selects a model later."""
import copy

import pytest
import yaml

from ppe_pipeline_api import retrain_schedule as rs
from ppe_pipeline_api.data import assemble, dataset_yaml as dy


NAMES = {0: "helmet", 1: "gloves", 2: "person", 3: "working_boots",
         4: "man_cage", 5: "safe_area"}


@pytest.mark.parametrize("model", ["helmet", "person", "", "deleted-model"])
@pytest.mark.parametrize("legacy_filter", [True, False])
def test_prepared_yaml_keeps_six_classes_then_training_selects_by_name(
        monkeypatch, model, legacy_filter):
    config = {"dataset_assemble": {
        "detector_slug": model, "drop_undeclared_labels": legacy_filter,
        "label_classes": {str(k): v for k, v in NAMES.items()},
        "train_sources": [{"images": "ds/images/train"}],
        "eval_sources": [{"images": "ds/images/val"}],
    }}
    before = copy.deepcopy(config)
    plan = rs.assemble_plan(config)
    assert not plan["problems"]
    monkeypatch.setattr(dy.pc, "_list_objects", lambda *a, **k: [])
    monkeypatch.setattr(dy, "_existing_yaml", lambda *a: {})
    # Source numbering differs, but every class must survive the remap.
    monkeypatch.setattr(dy, "_source_taxonomy", lambda *a: (
        {i: name for i, name in enumerate(reversed(list(NAMES.values())))}, ""))
    out = dy.build_in_place(splits=plan["splits"], classes=plan["classes"],
                            bucket="test", drop_undeclared=plan["drop_undeclared"])
    document = yaml.safe_load(out["text"])
    assert document["names"] == NAMES
    assert document["nc"] == 6
    assert document["x_label_remap"]
    for mapping in document["x_label_remap"].values():
        assert set(mapping.values()) == set(NAMES)
        assert -1 not in mapping.values()

    monkeypatch.setattr(assemble, "dataset_classes", lambda _: document["names"])
    assert assemble.class_filter_for("ds", {"classes": {0: "helmet"}}) == {
        "classes": [0], "single_cls": True}
    assert assemble.class_filter_for("ds", {"classes": {0: "person"}}) == {
        "classes": [2], "single_cls": True}
    assert assemble.class_filter_for("ds", {"classes": NAMES}) == {}
    assert document["names"] == NAMES
    assert config == before


def test_preparation_route_ignores_legacy_drop_flag(monkeypatch):
    from ppe_pipeline_api.routers import track2_models as routes
    calls = []
    monkeypatch.setattr(routes.dy, "build_in_place", lambda **kw: calls.append(kw) or {})
    monkeypatch.setattr(routes.dy, "write_in_place", lambda **kw: calls.append(kw) or {})
    payload = routes.DatasetYamlInPlaceRequest(
        classes={str(k): v for k, v in NAMES.items()}, drop_undeclared=True)
    routes.dataset_yaml_preview(payload, session={})
    routes.dataset_yaml_write(payload, session={})
    assert len(calls) == 2
    assert all(c["classes"] == payload.classes and c["drop_undeclared"] is False
               for c in calls)
