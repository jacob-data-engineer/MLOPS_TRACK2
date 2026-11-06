import io
from types import SimpleNamespace

from botocore.exceptions import ClientError

from ppe_pipeline_api.data import retention as rt, retention_quality as rq


def test_contribution_reports_volume_instead_of_equating_decay_with_balance():
    p = rt.plan([{"name": "week_1", "stems": [str(i) for i in range(10000)]},
                 {"name": "week_2", "stems": [str(i) for i in range(100)]}],
                {"enabled": True, "weekly_rate": .5, "use_floor": False})
    c = rt.composition(p["weeks"], {}, golden=100)
    assert c["historical"] == 5000 and c["recent"] == 100
    assert c["golden"] == 100 and c["total"] == 5200
    assert c["historical_share"] > .96


def test_excluded_week_still_ages_history_but_does_not_contribute():
    p = rt.plan([{"name": "week_1", "stems": [str(i) for i in range(100)]},
                 {"name": "week_2", "stems": ["a"]}],
                {"enabled": True, "weekly_rate": .5, "use_floor": False,
                 "excluded_weeks": ["week_2"]})
    rows = {r["week"]: r for r in p["weeks"]}
    assert rows["week_1"]["keep"] == 50
    assert rows["week_2"]["keep"] == 0 and rows["week_2"]["excluded"]


def test_lost_classes_and_unknown_labels_are_reported(monkeypatch):
    ids = [f"frame{i}" for i in range(20)]
    config = {"enabled": True, "weekly_rate": .5, "use_floor": False, "min_class_images": 3}
    kept = set(rt.select_stems(ids, .5, config))
    lost = next(i for i in ids if i not in kept)
    missing = next(i for i in kept)
    inv = [{"name": "week_1", "train": {i: f"root/week_1/images/train/{i}.jpg" for i in ids}, "val": {}},
           {"name": "week_2", "train": {"new": "root/week_2/images/train/new.jpg"}, "val": {}}]
    monkeypatch.setattr(rt, "read_inventory", lambda *a, **kw: inv)
    def get_object(**kw):
        name = kw["Key"].rsplit("/", 1)[-1][:-4]
        if name == missing:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(f"{7 if name == lost else 0} 0.5 0.5 0.2 0.2".encode())}
    monkeypatch.setattr(rq.pc, "s3", SimpleNamespace(get_object=get_object))
    monkeypatch.setattr(rq.storage, "BUCKET", "test")
    out = rq.inspect(root="root", config=config)
    classes = {c["class_id"]: c for c in out["sources"][0]["classes"]}
    assert classes[7]["before"] == 1 and classes[7]["lost"]
    assert not out["complete"]
    assert out["sources"][0]["missing_labels"] == 1
    assert any("no selected" in s for s in out["issues"])


def test_validation_only_golden_set_is_not_counted_as_training(monkeypatch):
    monkeypatch.setattr(rq.pc, "_list_objects", lambda p, b:
                        [{"Key": "gold/images/val/a.jpg"}] if "train" not in p else [])
    assert rt.golden_summary({"use_golden": True, "golden_root": "gold"})["available"] == 0


def test_threshold_validation_and_saved_settings(monkeypatch):
    import pytest
    from pydantic import ValidationError
    from ppe_pipeline_api.routers import track2_retention as routes
    with pytest.raises(ValidationError):
        routes.RetentionConfig(min_class_images=0)
    saved = {}
    monkeypatch.setattr(routes, "_settings", lambda: dict(rt.DEFAULTS))
    monkeypatch.setattr(routes.pc, "write_config", lambda patch: saved.update(patch))
    routes.save_settings(routes.RetentionConfig(min_class_images=12), session=None)
    assert saved["retention"]["min_class_images"] == 12


def test_validation_selection_id_changes_with_membership_and_taxonomy():
    from ppe_pipeline_api.data.dataset_versions import validation_selection_id as identity
    a = [("bucket", "a.jpg"), ("bucket", "b.jpg")]
    names = {0: "helmet"}
    baseline = identity(a, names, {})
    assert identity(list(reversed(a)), names, {}) == baseline
    assert identity(a + [("bucket", "c.jpg")], names, {}) != baseline
    assert identity(a, {0: "person"}, {}) != baseline
    assert identity(a, names, {"labels": {1: 0}}) != baseline
