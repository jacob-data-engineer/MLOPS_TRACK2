"""Tests for detectors: the per-model-type registries.

Each YOLO model type (PPE, vehicles, ...) owns its own per-size registry
groups, so a version number means "the 3rd vehicle model of this size"
rather than the 3rd of anything. The group name is the load-bearing part:
get it wrong and either the approval rule fires on a model that must never
reach production, or IAM denies the registration after hours of training.
"""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient

from ppe_pipeline_api import track2_shared as t2
from ppe_pipeline_api.main import app

_PASSWORD = "test-suite-password"


class FakeTable:
    """Keyed on `id`, like the real audit table. Its scan ignores the
    FilterExpression on purpose -- the code must not depend on the service
    filtering for it, since the table holds three record types."""

    def __init__(self):
        self.items: dict[str, dict] = {}

    def scan(self, **kwargs):
        return {"Items": list(self.items.values())}

    def put_item(self, Item):
        self.items[Item["id"]] = Item

    def delete_item(self, Key):
        self.items.pop(Key["id"], None)


@pytest.fixture
def client(monkeypatch) -> TestClient:
    monkeypatch.setenv("PPE_APP_PASSWORD", _PASSWORD)
    c = TestClient(app)
    token = c.post("/auth/login", json={"password": _PASSWORD}).json()["token"]
    c.headers.update({"Authorization": f"Bearer {token}"})
    return c


@pytest.fixture
def detector_table(monkeypatch) -> FakeTable:
    fake = FakeTable()
    monkeypatch.setattr(t2, "training_audit_table", lambda: fake)
    # Saving a detector now creates its registry groups. Stubbed out here so
    # these tests never reach SageMaker; the real one is tested below.
    fake.groups = []
    monkeypatch.setattr(t2, "ensure_model_package_group",
                        lambda group, description="": fake.groups.append(group) or True)
    # The list is memoised; without clearing it a test would see whatever
    # the previous one wrote.
    t2._invalidate_detector_cache()
    yield fake
    t2._invalidate_detector_cache()


def test_detector_group_names_never_collide_with_the_production_groups():
    """The four `yolo11{s,m,l,x}-detector` names are an exact-match list on
    the EventBridge rule that swaps the live inference weights. A detector
    whose group landed on one of those would push an unrelated taxonomy
    into production the moment someone approved it."""
    production = {f"yolo11{size}-detector" for size in t2.REGISTRY_SIZES}
    detector = {"group_pattern": "yolo11{size}-vehicle-detector"}
    for architecture in t2.KNOWN_ARCHITECTURES:
        group = t2.detector_group_name(detector, architecture)
        assert group not in production
        # Still inside the `model-package/yolo*-detector/*` IAM grant, which
        # is what lets a detector be added without a Terraform apply.
        assert group.startswith("yolo") and group.endswith("-detector")
        assert len(group) <= 63


def test_detector_group_name_is_per_size():
    detector = {"group_pattern": "yolo11{size}-vehicle-detector"}
    assert t2.detector_group_name(detector, "yolo11s") == "yolo11s-vehicle-detector"
    assert t2.detector_group_name(detector, "yolo11x") == "yolo11x-vehicle-detector"


def test_detector_group_name_rejects_an_unknown_architecture():
    with pytest.raises(ValueError):
        t2.detector_group_name({"group_pattern": "yolo11{size}-v-detector"}, "yolo11")


def test_upsert_and_list_detectors(detector_table):
    t2.upsert_detector("vehicle", "Vehicle detection", {"0": "vehicle", "1": "forklift"})
    # The built-in is always first and is never stored.
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe", "vehicle"]
    assert len(detector_table.items) == 1

    stored = t2.get_detector("vehicle")
    assert stored["classes"] == {0: "vehicle", 1: "forklift"}
    assert stored["builtin"] is False
    assert t2.detector_group(stored, "m") == "yolo11m-vehicle-detector"
    assert t2.detector_group(stored, "m", "26") == "yolo26m-vehicle-detector"


def test_the_builtin_is_merged_in_not_stored(detector_table):
    """It owns the legacy production group names, so it must not be
    editable into something inconsistent with the versions already there."""
    builtin = t2.get_detector("ppe")
    assert builtin["builtin"] is True
    assert builtin["group_pattern"].format(size="s") == "yolo11s-detector"
    assert detector_table.items == {}


def test_detector_rows_are_namespaced_away_from_run_ids(detector_table):
    """Detectors share the audit table, which is keyed on a bare `id`. An
    un-namespaced key would let a slug collide with a training-audit row."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    assert list(detector_table.items) == ["detector#vehicle"]


def test_list_detectors_ignores_the_other_record_types(detector_table):
    """The table also holds audit rows and queue entries. Reading one as a
    detector would invent a registry group out of a run id."""
    detector_table.put_item(Item={"id": "run-1", "record_type": "training_audit"})
    detector_table.put_item(Item={"id": "q-1", "record_type": "queue_entry"})
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe", "vehicle"]


@pytest.mark.parametrize("bad", [
    "my vehicle",  # a space -- SageMaker allows alphanumerics and hyphens only
    "-vehicle",    # must start with an alphanumeric
    "veh_icle",    # underscores are not allowed in a group name
    "",
    "v" * 32,      # long enough to risk the 63-character group-name limit
])
def test_upsert_detector_rejects_a_slug_that_would_break_the_group_name(detector_table, bad):
    with pytest.raises(ValueError):
        t2.upsert_detector(bad, "x", {"0": "a"})
    assert detector_table.items == {}


def test_a_slug_is_normalised_rather_than_rejected(detector_table):
    """Case and surrounding space are the operator's, not the registry's.
    Normalising keeps the group name deterministic without making them
    retype it."""
    t2.upsert_detector("  Vehicle  ", "Vehicle", {"0": "a"})
    assert list(detector_table.items) == ["detector#vehicle"]
    assert t2.detector_group(t2.get_detector("vehicle"), "s") == "yolo11s-vehicle-detector"


def test_upsert_detector_refuses_to_redefine_the_builtin(detector_table):
    with pytest.raises(ValueError):
        t2.upsert_detector("ppe", "Something else", {"0": "cat"})


@pytest.mark.parametrize("classes", [
    {"0": "a", "2": "b"},   # a gap
    {"1": "a", "2": "b"},   # does not start at 0
    {},                     # nothing to detect
    {"0": "a", "1": "  "},  # an unnamed class
])
def test_upsert_detector_requires_a_usable_class_list(detector_table, classes):
    """Ultralytics requires 0..n-1, and the inference path reads index 0 by
    position -- a gap silently shifts every label rather than failing."""
    with pytest.raises(ValueError):
        t2.upsert_detector("vehicle", "Vehicle", classes)


def test_upsert_detector_updates_in_place(detector_table):
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    t2.upsert_detector("vehicle", "Vehicles and plant", {"0": "vehicle", "1": "forklift"})
    assert len(detector_table.items) == 1
    assert t2.get_detector("vehicle")["label"] == "Vehicles and plant"
    assert t2.get_detector("vehicle")["classes"] == {0: "vehicle", 1: "forklift"}


def test_delete_detector_refuses_while_versions_exist(detector_table, monkeypatch):
    """Deleting the definition must not silently orphan registered models:
    the group would still hold versions that nothing could name."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    monkeypatch.setattr(
        t2, "model_registry_summary",
        lambda group: {"total": 2 if group == "yolo11s-vehicle-detector" else 0, "packages": []},
    )
    with pytest.raises(ValueError, match="yolo11s-vehicle-detector"):
        t2.delete_detector("vehicle")
    assert "vehicle" in [d["slug"] for d in t2.list_detectors()]


def test_delete_detector_removes_an_empty_one(detector_table, monkeypatch):
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    monkeypatch.setattr(t2, "model_registry_summary", lambda group: {"total": 0, "packages": []})
    monkeypatch.setattr(t2, "sm", _SmDeletingGroups())
    assert t2.delete_detector("vehicle") == {
        "slug": "vehicle", "deleted": True, "versions_removed": 0,
    }
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe"]


class _SmDeletingGroups:
    def __init__(self):
        self.deleted_groups = []

    def delete_model_package_group(self, ModelPackageGroupName):
        self.deleted_groups.append(ModelPackageGroupName)


def test_deleting_a_detector_takes_its_empty_groups_with_it(detector_table, monkeypatch):
    """Leaving the groups behind would hold the name and its version
    counter, so recreating the detector would resume at the old number."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    fake = _SmDeletingGroups()
    monkeypatch.setattr(t2, "sm", fake)
    monkeypatch.setattr(t2, "model_registry_summary", lambda group: {"total": 0, "packages": []})
    t2.delete_detector("vehicle")
    assert fake.deleted_groups == [
        f"yolo{generation}{size}-vehicle-detector"
        for generation in t2.GENERATION_SIZES
        for size in t2.GENERATION_SIZES[generation]
    ]


def test_force_deletes_the_versions_too(detector_table, monkeypatch):
    """Refusing is right by default, but an operator clearing out a
    detector should not have to delete each version by hand first."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    removed = []
    fake = _SmDeletingGroups()
    monkeypatch.setattr(t2, "sm", fake)
    monkeypatch.setattr(t2, "delete_model_package_and_detach_audit",
                        lambda arn: removed.append(arn))
    monkeypatch.setattr(
        t2, "model_registry_summary",
        lambda group: {"total": 2, "packages": [
            {"ModelPackageArn": f"arn:{group}/1"}, {"ModelPackageArn": f"arn:{group}/2"},
        ]} if group == "yolo11s-vehicle-detector" else {"total": 0, "packages": []},
    )
    out = t2.delete_detector("vehicle", force=True)
    assert out == {"slug": "vehicle", "deleted": True, "versions_removed": 2}
    assert removed == ["arn:yolo11s-vehicle-detector/1", "arn:yolo11s-vehicle-detector/2"]
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe"]


def test_force_stops_rather_than_half_deleting(detector_table, monkeypatch):
    """A detector whose row is gone but whose versions remain is worse
    than a refusal -- nothing could name them afterwards."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    monkeypatch.setattr(t2, "sm", _SmDeletingGroups())
    def boom(arn):
        raise RuntimeError("AccessDenied")
    monkeypatch.setattr(t2, "delete_model_package_and_detach_audit", boom)
    monkeypatch.setattr(
        t2, "model_registry_summary",
        lambda group: {"total": 1, "packages": [{"ModelPackageArn": "arn:x/1"}]}
        if group == "yolo11s-vehicle-detector" else {"total": 0, "packages": []},
    )
    with pytest.raises(ValueError, match="Could not delete"):
        t2.delete_detector("vehicle", force=True)
    assert "vehicle" in [d["slug"] for d in t2.list_detectors()]


def test_force_still_refuses_the_builtin(detector_table):
    with pytest.raises(ValueError):
        t2.delete_detector("ppe", force=True)


def test_delete_detector_refuses_the_builtin(detector_table):
    with pytest.raises(ValueError):
        t2.delete_detector("ppe")


def test_get_detector_names_the_known_slugs_when_it_misses(detector_table):
    with pytest.raises(ValueError, match="ppe"):
        t2.get_detector("nope")


def test_a_write_invalidates_the_cached_detector_list(detector_table):
    """The list is memoised so one page load is one scan rather than three
    -- but an edit must never be served stale."""
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe"]
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe", "vehicle"]


def test_the_cached_list_cannot_be_mutated_by_a_caller(detector_table):
    """It is handed out by reference otherwise, and one caller appending to
    it would corrupt every later read for the cache's lifetime."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    first = t2.list_detectors()
    first.append({"slug": "injected"})
    first[0]["label"] = "tampered"
    second = t2.list_detectors()
    assert [d["slug"] for d in second] == ["ppe", "vehicle"]
    assert second[0]["label"] == t2.BUILTIN_DETECTOR["label"]


def test_a_scan_failure_falls_back_to_the_builtin(monkeypatch):
    """A detector read failing must not blank the registries page."""
    class Exploding:
        def scan(self, **kwargs):
            raise RuntimeError("throttled")

    monkeypatch.setattr(t2, "training_audit_table", lambda: Exploding())
    t2._invalidate_detector_cache()
    try:
        assert [d["slug"] for d in t2.list_detectors()] == ["ppe"]
    finally:
        t2._invalidate_detector_cache()


# --- the API surface -----------------------------------------------------------

def test_detectors_endpoint_reports_the_per_size_groups(client, detector_table):
    t2.upsert_detector("vehicle", "Vehicle detection", {"0": "vehicle"})
    body = client.get("/track2/models/detectors").json()
    assert body["sizes"] == list(t2.REGISTRY_SIZES)
    vehicle = next(d for d in body["detectors"] if d["slug"] == "vehicle")
    assert vehicle["groups"] == {
        size: f"yolo11{size}-vehicle-detector" for size in t2.REGISTRY_SIZES
    }
    # Indices cross the wire as strings, which is how data.yaml reads back.
    assert vehicle["classes"] == {"0": "vehicle"}


def test_detectors_endpoint_rejects_a_bad_definition(client, detector_table):
    response = client.post(
        "/track2/models/detectors",
        json={"slug": "my vehicle", "label": "x", "classes": {"0": "a"}},
    )
    assert response.status_code == 400
    assert "lowercase" in response.json()["detail"]


def test_detector_round_trips_through_the_api(client, detector_table, monkeypatch):
    save = client.post(
        "/track2/models/detectors",
        json={"slug": "vehicle", "label": "Vehicle", "classes": {"0": "a", "1": "b"}},
    )
    assert save.status_code == 200, save.json()
    listed = client.get("/track2/models/detectors").json()["detectors"]
    assert [d["slug"] for d in listed] == ["ppe", "vehicle"]

    monkeypatch.setattr(t2, "model_registry_summary", lambda group: {"total": 0, "packages": []})
    assert client.post("/track2/models/detectors/vehicle/delete").json()["deleted"] is True
    listed = client.get("/track2/models/detectors").json()["detectors"]
    assert [d["slug"] for d in listed] == ["ppe"]


# --- registry groups must exist before a run registers -------------------------

def test_saving_a_detector_creates_its_per_size_groups(detector_table):
    """Registering does NOT create the group; CreateModelPackage fails on a
    missing one, at the END of the pipeline, after the GPU hours are gone."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    assert detector_table.groups == [
        f"yolo{generation}{size}-vehicle-detector"
        for generation in t2.GENERATION_SIZES
        for size in t2.GENERATION_SIZES[generation]
    ]


def test_a_detector_saves_even_if_its_groups_cannot_be_created(detector_table, monkeypatch):
    """The create is denied until the training-control policy is applied.
    Losing the definition over that would be worse than a missing group,
    which an admin can create separately."""
    monkeypatch.setattr(t2, "ensure_model_package_group",
                        lambda group, description="": False)
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle"})
    assert [d["slug"] for d in t2.list_detectors()] == ["ppe", "vehicle"]


def test_ensure_group_is_a_no_op_when_it_already_exists(monkeypatch):
    created = []

    class FakeSm:
        def describe_model_package_group(self, ModelPackageGroupName):
            return {"ModelPackageGroupName": ModelPackageGroupName}

        def create_model_package_group(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(t2, "sm", FakeSm())
    assert t2.ensure_model_package_group("yolo11s-vehicle-detector") is True
    assert created == []


def test_ensure_group_creates_a_missing_one(monkeypatch):
    created = []

    class FakeSm:
        def describe_model_package_group(self, ModelPackageGroupName):
            raise RuntimeError("ValidationException: does not exist")

        def create_model_package_group(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(t2, "sm", FakeSm())
    assert t2.ensure_model_package_group("yolo11s-vehicle-detector", "Vehicle, small.") is True
    assert created[0]["ModelPackageGroupName"] == "yolo11s-vehicle-detector"
    assert created[0]["ModelPackageGroupDescription"] == "Vehicle, small."


def test_ensure_group_never_raises_when_the_create_is_denied(monkeypatch):
    """It runs on the path that starts a run. Raising here would turn a
    missing-group warning into a refusal to train at all."""
    class FakeSm:
        def describe_model_package_group(self, ModelPackageGroupName):
            raise RuntimeError("AccessDenied")

        def create_model_package_group(self, **kwargs):
            raise RuntimeError("AccessDenied")

    monkeypatch.setattr(t2, "sm", FakeSm())
    assert t2.ensure_model_package_group("yolo11s-vehicle-detector") is False


def test_a_group_description_is_truncated_to_the_service_limit(monkeypatch):
    created = []

    class FakeSm:
        def describe_model_package_group(self, ModelPackageGroupName):
            raise RuntimeError("missing")

        def create_model_package_group(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(t2, "sm", FakeSm())
    t2.ensure_model_package_group("yolo11s-vehicle-detector", "x" * 5000)
    assert len(created[0]["ModelPackageGroupDescription"]) == 1024


# --- importing a pretrained checkpoint ----------------------------------------

class FakeSmRegistry:
    def __init__(self):
        self.created = []

    def describe_model_package_group(self, ModelPackageGroupName):
        return {"ModelPackageGroupName": ModelPackageGroupName}

    def create_model_package(self, **kwargs):
        self.created.append(kwargs)
        group = kwargs["ModelPackageGroupName"]
        return {"ModelPackageArn": f"arn:aws:sagemaker:ap-southeast-1:1:model-package/{group}/1"}


@pytest.fixture
def importable(detector_table, monkeypatch):
    """A vehicle detector, a checkpoint that matches it, and fake S3/SM."""
    t2.upsert_detector("vehicle", "Vehicle", {"0": "vehicle", "1": "forklift"})
    registry = FakeSmRegistry()
    monkeypatch.setattr(t2, "sm", registry)
    monkeypatch.setattr(t2, "inspect_checkpoint", lambda key, arch: {
        "key": key, "detected_architecture": arch,
        "classes": {0: "vehicle", 1: "forklift"},
    })

    class FakeS3:
        def __init__(self):
            self.uploaded = []

        def download_file(self, bucket, key, local):
            open(local, "wb").write(b"WEIGHTS")

        def upload_file(self, local, bucket, key):
            self.uploaded.append(key)

    s3 = FakeS3()
    monkeypatch.setattr(t2, "s3", s3)
    return registry, s3


def test_importing_registers_into_the_detectors_group(importable):
    registry, _ = importable
    out = t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="vehicle", architecture="yolo11s", notes="from the vendor",
    )
    assert out["group_name"] == "yolo11s-vehicle-detector"
    assert out["version"] == "1"
    created = registry.created[0]
    assert created["ModelPackageGroupName"] == "yolo11s-vehicle-detector"
    assert created["ModelApprovalStatus"] == "Approved"


def test_an_import_is_marked_as_imported_not_trained(importable):
    """Every other version in the group came out of a run. This did not,
    and its metrics are unknown rather than zero -- the metadata has to say
    so or it reads as a trained model whose evaluation went missing."""
    registry, _ = importable
    t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="vehicle", architecture="yolo11s",
    )
    meta = registry.created[0]["CustomerMetadataProperties"]
    assert meta["base_model_source"] == "import"
    assert meta["detector"] == "vehicle"
    assert meta["architecture"] == "yolo11s"
    assert meta["imported_from"].endswith("external_models/uploaded/yolo11s/abc/model.pt")


def test_every_metadata_value_is_within_the_service_limits(importable):
    """CustomerMetadataProperties values must be 1-256 characters drawn
    from ([\\p{L}\\p{Z}\\p{N}_.:/=+\\-@]*). This test previously checked only
    the length, which is why a comma got through: class names joined with
    ", " failed the whole call -- after the artifact had been uploaded."""
    registry, _ = importable
    t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="vehicle", architecture="yolo11s",
    )
    allowed = set("_.:/=+-@ ")
    for name, value in registry.created[0]["CustomerMetadataProperties"].items():
        assert 1 <= len(value) <= 256, f"{name} is {len(value)} characters"
        bad = [ch for ch in value if not (ch.isalnum() or ch in allowed)]
        assert not bad, f"{name} has disallowed characters {bad!r} in {value!r}"


def test_a_multi_class_detector_does_not_produce_a_comma(importable, monkeypatch):
    """The exact failure seen in production: four class names joined with
    ", " put a comma in the map and CreateModelPackage refused the lot."""
    registry, _ = importable
    classes = {0: "vertical_plane", 1: "horizontal_plane", 2: "dangerous_plane", 3: "hatch_wall"}
    t2.upsert_detector("segment", "Segment", {str(k): v for k, v in classes.items()})
    monkeypatch.setattr(t2, "inspect_checkpoint", lambda key, arch: {
        "key": key, "detected_architecture": "", "classes": classes,
    })
    t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="segment", architecture="yolo11s",
    )
    written = registry.created[-1]["CustomerMetadataProperties"]
    assert "," not in written["classes"]
    assert written["classes"] == "vertical_plane / horizontal_plane / dangerous_plane / hatch_wall"
    # An architecture the file did not record must still be a legal value,
    # not an empty string -- the map's minimum length is 1.
    assert written["detected_architecture"] == "unknown"


@pytest.mark.parametrize("raw,expected", [
    ("a, b, c", "a b c"),                        # commas dropped
    ("s3://bucket/some file.pt", "s3://bucket/some file.pt"),  # legal as-is
    ("line\nbreak\ttab", "line break tab"),      # \p{Z} is not \n or \t
    ("", "unspecified"),                          # minimum length is 1
    ("   ", "unspecified"),
    (None, "unspecified"),
    ("!!!", "unspecified"),                       # nothing legal survives
    ("x" * 400, "x" * 256),                       # maximum length is 256
])
def test_metadata_values_are_coerced_into_the_allowed_set(raw, expected):
    assert t2.metadata_value(raw) == expected


def test_the_checkpoint_is_wrapped_as_a_tarball(importable):
    """The registry's other versions are all model.tar.gz. Registering the
    bare .pt would work for provenance but break publish_versioned_weights
    and anything that later deploys it."""
    _, s3 = importable
    out = t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="vehicle", architecture="yolo11s",
    )
    assert s3.uploaded[0].endswith("/model.tar.gz")
    assert out["artifact_uri"].endswith("/model.tar.gz")


def test_the_tarball_holds_model_pt_at_its_root(importable, tmp_path, monkeypatch):
    """publish_versioned_weights looks for exactly "model.pt" or
    "./model.pt" -- a tarball keyed on the temp path would not match."""
    import tarfile as _tarfile
    written = {}

    class CapturingS3:
        def download_file(self, bucket, key, local):
            open(local, "wb").write(b"WEIGHTS")

        def upload_file(self, local, bucket, key):
            written["names"] = _tarfile.open(local, "r:gz").getnames()

    monkeypatch.setattr(t2, "s3", CapturingS3())
    t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="vehicle", architecture="yolo11s",
    )
    assert written["names"] == ["model.pt"]


def test_importing_refuses_a_checkpoint_whose_classes_disagree(importable, monkeypatch):
    """The whole point of the detector's class list. A ship checkpoint in
    the helmet registry is a mislabelling that follows every model later
    fine-tuned from it."""
    registry, _ = importable
    monkeypatch.setattr(t2, "inspect_checkpoint", lambda key, arch: {
        "key": key, "detected_architecture": arch, "classes": {0: "ship"},
    })
    with pytest.raises(ValueError, match="ship"):
        t2.register_uploaded_base_model(
            key="external_models/uploaded/yolo11s/abc/model.pt",
            detector_slug="vehicle", architecture="yolo11s",
        )
    assert registry.created == []


def test_importing_refuses_a_checkpoint_with_the_classes_reordered(importable, monkeypatch):
    """Index order is not cosmetic: inference reads index 0 by position, so
    a swapped pair mislabels every prediction without failing."""
    registry, _ = importable
    monkeypatch.setattr(t2, "inspect_checkpoint", lambda key, arch: {
        "key": key, "detected_architecture": arch,
        "classes": {0: "forklift", 1: "vehicle"},
    })
    with pytest.raises(ValueError):
        t2.register_uploaded_base_model(
            key="external_models/uploaded/yolo11s/abc/model.pt",
            detector_slug="vehicle", architecture="yolo11s",
        )
    assert registry.created == []


def test_importing_into_the_builtin_is_refused(importable):
    """Its four groups are the exact-match list on the rule that swaps the
    live inference weights. An unevaluated import auto-approved into one
    would go straight to production."""
    registry, _ = importable
    with pytest.raises(ValueError, match="production"):
        t2.register_uploaded_base_model(
            key="external_models/uploaded/yolo11s/abc/model.pt",
            detector_slug="ppe", architecture="yolo11s",
        )
    assert registry.created == []


def test_importing_rejects_an_unknown_architecture(importable):
    with pytest.raises(ValueError):
        t2.register_uploaded_base_model(
            key="external_models/uploaded/yolo11s/abc/model.pt",
            detector_slug="vehicle", architecture="yolo11",
        )


# --- what may be read back as a checkpoint ------------------------------------

def test_a_fresh_upload_is_inspectable_without_touching_s3(monkeypatch):
    """The upload prefix short-circuits: this app just wrote the object, so
    re-checking its existence is a round trip for nothing."""
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: pytest.fail("should not check"))
    key = f"{t2.BASE_MODEL_UPLOAD_PREFIX}/yolo11s/abc/model.pt"
    assert t2.assert_inspectable_checkpoint(key) == key


def test_an_existing_bucket_checkpoint_is_inspectable(monkeypatch):
    """Importing a .pt already in the bucket -- the same set the training
    form will hand the trainer as starting weights."""
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    assert t2.assert_inspectable_checkpoint("ming_yang_yolos/run/weights/best.pt") \
        == "ming_yang_yolos/run/weights/best.pt"


def test_a_missing_object_is_refused(monkeypatch):
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: False)
    with pytest.raises(ValueError, match="does not exist"):
        t2.assert_inspectable_checkpoint("nowhere/best.pt")


@pytest.mark.parametrize("key", ["", "   ", "notes/readme.txt", "data/images/a.jpg"])
def test_a_non_checkpoint_is_refused(monkeypatch, key):
    """Reading one means unpickling it inside the container that holds the
    estate's credentials, so the file has to be a checkpoint."""
    monkeypatch.setattr(t2, "_s3_object_exists", lambda k: True)
    with pytest.raises(ValueError):
        t2.assert_inspectable_checkpoint(key)


# --- an empty registry starts again at v1 -------------------------------------

def test_an_empty_group_is_recreated_so_numbering_restarts(monkeypatch):
    """SageMaker version numbers are monotonic per group and never reused:
    register v1..v4, delete all four, and the next is v5 in a registry that
    looks empty. Measured against the service -- emptying gave v3,
    deleting and recreating the group gave v1."""
    deleted = []

    class FakeSm:
        def delete_model_package_group(self, ModelPackageGroupName):
            deleted.append(ModelPackageGroupName)

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"total": 0, "packages": []})
    assert t2.restart_group_numbering_if_empty("yolo11s-helmet-detector") is True
    assert deleted == ["yolo11s-helmet-detector"]


def test_a_group_holding_versions_is_never_deleted(monkeypatch):
    """The guard that makes this safe. Deleting a group with versions in it
    would destroy them."""
    class FakeSm:
        def delete_model_package_group(self, ModelPackageGroupName):
            pytest.fail("must not delete a group that holds versions")

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"total": 1, "packages": []})
    assert t2.restart_group_numbering_if_empty("yolo11s-helmet-detector") is False


def test_a_failed_reset_never_breaks_the_import(monkeypatch):
    """A higher version number is cosmetic; a failed import is not."""
    class FakeSm:
        def delete_model_package_group(self, ModelPackageGroupName):
            raise RuntimeError("AccessDenied")

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"total": 0, "packages": []})
    assert t2.restart_group_numbering_if_empty("yolo11s-helmet-detector") is False


def test_an_unreadable_registry_leaves_numbering_alone(monkeypatch):
    """If the count cannot be read, the group cannot be shown to be empty,
    so it must not be deleted."""
    class FakeSm:
        def delete_model_package_group(self, ModelPackageGroupName):
            pytest.fail("must not delete on an unknown count")

    monkeypatch.setattr(t2, "sm", FakeSm())
    def boom(group):
        raise RuntimeError("throttled")
    monkeypatch.setattr(t2, "model_registry_summary", boom)
    assert t2.restart_group_numbering_if_empty("yolo11s-helmet-detector") is False


def test_importing_into_an_empty_registry_resets_first(importable, monkeypatch):
    order = []
    monkeypatch.setattr(t2, "restart_group_numbering_if_empty",
                        lambda g: order.append(("reset", g)) or True)
    real_ensure = t2.ensure_model_package_group
    monkeypatch.setattr(t2, "ensure_model_package_group",
                        lambda g, description="": order.append(("ensure", g)) or True)
    t2.register_uploaded_base_model(
        key="external_models/uploaded/yolo11s/abc/model.pt",
        detector_slug="vehicle", architecture="yolo11s",
    )
    # Reset BEFORE ensure, or the group would be recreated and then deleted.
    assert order == [("reset", "yolo11s-vehicle-detector"),
                     ("ensure", "yolo11s-vehicle-detector")]
    assert real_ensure is not None


# --- the group must survive the numbering reset --------------------------------

def test_the_reset_waits_for_the_delete_to_finish(monkeypatch):
    """DeleteModelPackageGroup returns BEFORE the group is gone. Recreating
    straight after raced the delete, so the group stayed deleted and the
    import registered into nothing -- the group simply vanished."""
    events = []

    class FakeSm:
        def __init__(self):
            self.describes = 0

        def delete_model_package_group(self, ModelPackageGroupName):
            events.append("delete")

        def describe_model_package_group(self, ModelPackageGroupName):
            self.describes += 1
            # Still there for the first two polls, as a real delete is.
            if self.describes <= 2:
                return {"ModelPackageGroupName": ModelPackageGroupName}
            raise RuntimeError("does not exist")

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"total": 0, "packages": []})
    monkeypatch.setattr(t2.time, "sleep", lambda s: events.append("waited"))
    assert t2.restart_group_numbering_if_empty("yolo11s-helmet-detector") is True
    assert events == ["delete", "waited", "waited"]


def test_the_reset_gives_up_rather_than_racing_a_slow_delete(monkeypatch):
    class FakeSm:
        def delete_model_package_group(self, ModelPackageGroupName):
            pass

        def describe_model_package_group(self, ModelPackageGroupName):
            return {"ModelPackageGroupName": ModelPackageGroupName}  # never goes

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "model_registry_summary", lambda g: {"total": 0, "packages": []})
    monkeypatch.setattr(t2.time, "sleep", lambda s: None)
    monkeypatch.setattr(t2.time, "monotonic", _ticking())
    assert t2.restart_group_numbering_if_empty("yolo11s-helmet-detector") is False


def _ticking():
    """A monotonic clock that jumps a minute per call, so the wait loop
    times out without the test actually sleeping."""
    state = {"t": 0.0}

    def clock():
        state["t"] += 60.0
        return state["t"]
    return clock


def test_registering_refuses_when_the_group_cannot_be_created(importable, monkeypatch):
    """The failure that lost the helmet registry: the group was gone, the
    artifact was uploaded anyway, and CreateModelPackage then failed with
    nothing to register into."""
    registry, s3 = importable
    monkeypatch.setattr(t2, "restart_group_numbering_if_empty", lambda g: True)
    monkeypatch.setattr(t2, "ensure_model_package_group", lambda g, description="": False)
    with pytest.raises(ValueError, match="could not be created"):
        t2.register_uploaded_base_model(
            key="external_models/uploaded/yolo11s/abc/model.pt",
            detector_slug="vehicle", architecture="yolo11s",
        )
    # Refused BEFORE the artifact upload, not after.
    assert s3.uploaded == []
    assert registry.created == []


# --- filtering a multi-class dataset down to one detector ---------------------

class FakeS3Dataset:
    """A tiny in-memory S3 holding one 6-class Ultralytics dataset."""

    def __init__(self, labels: dict[str, str], images: list[str], names: dict):
        self.objects: dict[str, bytes] = {}
        root = "data/site"
        self.objects[f"{root}/data.yaml"] = (
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in sorted(names.items()))
        ).encode()
        for key, body in labels.items():
            self.objects[f"{root}/labels/{key}"] = body.encode()
        for key in images:
            self.objects[f"{root}/images/{key}"] = b"JPEG"
        self.copied: list[tuple] = []

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise RuntimeError(f"NoSuchKey {Key}")
        import io
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body if isinstance(Body, bytes) else str(Body).encode()

    def copy_object(self, Bucket, CopySource, Key):
        self.copied.append((CopySource["Key"], Key))
        self.objects[Key] = self.objects[CopySource["Key"]]

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(objects) if k.startswith(Prefix)]}
        return Paginator()


@pytest.fixture
def site_dataset(monkeypatch):
    """6 classes; helmet is index 0. Two train images, one with a helmet."""
    names = {0: "helmet", 1: "gloves", 2: "person", 3: "working_boots",
             4: "man_cage", 5: "safe_area"}
    fake = FakeS3Dataset(
        labels={
            # a helmet box and a gloves box
            "train/a.txt": "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n",
            # no helmet at all -- a background once filtered
            "train/b.txt": "2 0.4 0.4 0.2 0.2\n",
            "val/c.txt": "0 0.6 0.6 0.1 0.1\n",
        },
        images=["train/a.jpg", "train/b.jpg", "val/c.jpg"],
        names=names,
    )
    monkeypatch.setattr(t2, "s3", fake)
    return fake


def _helmet():
    return {"slug": "helmet", "classes": {0: "helmet"},
            "group_pattern": "yolo11{size}-helmet-detector", "builtin": False}


def test_filtering_keeps_only_the_detectors_boxes(site_dataset):
    dest = t2.derive_filtered_dataset("data/site", _helmet())
    assert dest == "data/site__helmet"
    # The gloves box is gone; the helmet box stays, reindexed to 0.
    assert site_dataset.objects[f"{dest}/labels/train/a.txt"].decode() == "0 0.5 0.5 0.1 0.1\n"


def test_an_image_without_the_class_is_kept_as_a_background(site_dataset):
    """Dropping these is what makes a single-class model fire on anything
    vaguely similar -- they are negative examples, not junk."""
    dest = t2.derive_filtered_dataset("data/site", _helmet())
    assert site_dataset.objects[f"{dest}/labels/train/b.txt"] == b""
    assert f"{dest}/images/train/b.jpg" in site_dataset.objects


def test_filtering_covers_both_train_and_val(site_dataset):
    dest = t2.derive_filtered_dataset("data/site", _helmet())
    assert site_dataset.objects[f"{dest}/labels/val/c.txt"].decode() == "0 0.6 0.6 0.1 0.1\n"
    assert f"{dest}/images/val/c.jpg" in site_dataset.objects


def test_the_derived_data_yaml_declares_only_the_detectors_classes(site_dataset):
    dest = t2.derive_filtered_dataset("data/site", _helmet())
    import yaml as _yaml
    written = _yaml.safe_load(site_dataset.objects[f"{dest}/data.yaml"].decode())
    assert written["names"] == {0: "helmet"}
    assert written["train"] == "images/train"
    assert written["val"] == "images/val"


def test_images_are_copied_server_side(site_dataset):
    """A dataset is gigabytes; pulling it through this container to put it
    straight back would be absurd."""
    t2.derive_filtered_dataset("data/site", _helmet())
    assert len(site_dataset.copied) == 3
    assert all(src != dst for src, dst in site_dataset.copied)


def test_a_second_run_reuses_the_existing_copy(site_dataset):
    t2.derive_filtered_dataset("data/site", _helmet())
    before = len(site_dataset.copied)
    t2.derive_filtered_dataset("data/site", _helmet())
    assert len(site_dataset.copied) == before, "re-derived instead of reusing"


def test_a_changed_detector_forces_a_re_derive(site_dataset):
    """The marker records WHAT was derived: editing the detector's classes
    must not silently keep training on the old filtering."""
    t2.derive_filtered_dataset("data/site", _helmet())
    before = len(site_dataset.copied)
    two = {"slug": "helmet", "classes": {0: "helmet", 1: "gloves"},
           "group_pattern": "yolo11{size}-helmet-detector", "builtin": False}
    t2.derive_filtered_dataset("data/site", two)
    assert len(site_dataset.copied) > before


def test_a_dataset_that_already_matches_is_not_copied(site_dataset):
    """Nothing to derive -- copying would only make a second identical
    dataset to keep track of."""
    everything = {
        "slug": "site", "builtin": False,
        "group_pattern": "yolo11{size}-site-detector",
        "classes": {0: "helmet", 1: "gloves", 2: "person", 3: "working_boots",
                    4: "man_cage", 5: "safe_area"},
    }
    assert t2.derive_filtered_dataset("data/site", everything) == "data/site"
    assert site_dataset.copied == []


def test_filtering_refuses_when_the_class_is_absent(site_dataset):
    ship = {"slug": "ship", "classes": {0: "ship"},
            "group_pattern": "yolo11{size}-ship-detector", "builtin": False}
    # The wording changed, not the refusal: both sides of this check come
    # from data.yaml and no .txt file is opened, so "no labels for ship"
    # was a claim about a dataset it had never read. See
    # tests/test_missing_classes_blames_the_yaml.py.
    with pytest.raises(ValueError, match="does not declare ship"):
        t2.derive_filtered_dataset("data/site", ship)


def test_validation_accepts_a_superset_because_the_run_filters_it(detector_table, monkeypatch):
    t2.upsert_detector("helmet", "Helmet", {"0": "helmet"})
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    monkeypatch.setattr(t2, "dataset_classes", lambda f: {0: "helmet", 1: "gloves"})
    t2.validate_dataset_folder("data/site", used_for_evaluation=False, detector_slug="helmet")


def test_validation_still_refuses_a_dataset_missing_the_class(detector_table, monkeypatch):
    t2.upsert_detector("helmet", "Helmet", {"0": "helmet"})
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    monkeypatch.setattr(t2, "dataset_classes", lambda f: {0: "gloves", 1: "person"})
    with pytest.raises(ValueError, match="does not declare helmet"):
        t2.validate_dataset_folder("data/site", used_for_evaluation=False, detector_slug="helmet")


# --- the session probe --------------------------------------------------------
#
# Added because ANY 401 from ANY endpoint used to sign the operator out
# immediately, so one failing background poll dropped them to the sign-in
# screen mid-task. The browser now asks this route whether the session is
# actually dead before acting.

def test_the_session_probe_accepts_a_live_token(client):
    assert client.get("/auth/session").json() == {"ok": True}


def test_the_session_probe_rejects_a_dead_token():
    c = TestClient(app)
    c.headers.update({"Authorization": "Bearer not-a-real-token"})
    assert c.get("/auth/session").status_code == 401


def test_the_session_probe_rejects_no_token():
    assert TestClient(app).get("/auth/session").status_code == 401


# --- assembling a dataset from folders laid out any way ------------------------
#
# Ultralytics finds labels by swapping /images/ for /labels/ in each image
# path. Labels anywhere else are invisible to it: the run trains on
# unannotated images and reports a confident zero. These build the one
# layout it accepts.

class FakeS3Flat:
    def __init__(self, keys: list[str]):
        self.objects = {k: b"x" for k in keys}
        self.copied: list[tuple] = []
        self.put: dict[str, bytes] = {}

    def get_paginator(self, name):
        objects = self.objects

        class Paginator:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(objects) if k.startswith(Prefix)]}
        return Paginator()

    def copy_object(self, Bucket, CopySource, Key):
        self.copied.append((CopySource["Key"], Key))
        self.objects[Key] = b"x"

    # **kwargs: boto3 takes ContentType and a dozen others, and a double
    # that rejects them makes production code fail for no reason.
    def put_object(self, Bucket, Key, Body, **_kwargs):
        self.put[Key] = Body if isinstance(Body, bytes) else str(Body).encode()
        self.objects[Key] = self.put[Key]

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def scattered(monkeypatch):
    """Images and labels in unrelated prefixes -- the case the trainer
    cannot handle and this feature exists for."""
    fake = FakeS3Flat([
        "raw/pics/a.jpg", "raw/pics/b.jpg", "raw/pics/c.jpg",
        "annotations/yolo/a.txt", "annotations/yolo/b.txt",
        "annotations/yolo/ghost.txt",
        "raw/valpics/v1.jpg", "annotations/val/v1.txt",
    ])
    monkeypatch.setattr(t2, "s3", fake)
    return fake


def test_the_layout_preview_pairs_on_filename_stems(scattered):
    report = t2.inspect_dataset_layout(splits={
        "train": ("raw/pics", "annotations/yolo"),
        "val": ("raw/valpics", "annotations/val"),
    })
    train = next(s for s in report["splits"] if s["split"] == "train")
    assert train["image_count"] == 3
    assert train["label_count"] == 3
    assert train["paired"] == 2             # a, b
    assert train["unlabelled"] == 1         # c has no label
    assert train["orphan_labels"] == 1      # ghost.txt has no image
    assert report["total_paired"] == 3      # 2 train + 1 val


def test_the_preview_warns_when_nothing_pairs(scattered):
    """The failure this catches: the wrong label folder looks fine until
    the run trains on images with no annotations."""
    report = t2.inspect_dataset_layout(splits={
        "train": ("raw/pics", "annotations/val"),
        "val": ("", ""),
    })
    assert report["splits"][0]["paired"] == 0
    assert any("wrong label folder" in w for w in report["warnings"])


def test_the_preview_writes_nothing(scattered):
    t2.inspect_dataset_layout(splits={"train": ("raw/pics", "annotations/yolo")})
    assert scattered.copied == []
    assert scattered.put == {}


def test_assembling_builds_the_mirrored_layout(scattered):
    out = t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo"),
                "val": ("raw/valpics", "annotations/val")},
        classes={"0": "helmet"},
        destination="data/built",
    )
    assert out["folder"] == "data/built"
    landed = {dst for _, dst in scattered.copied}
    assert "data/built/images/train/a.jpg" in landed
    assert "data/built/labels/train/a.txt" in landed
    assert "data/built/images/val/v1.jpg" in landed
    assert "data/built/labels/val/v1.txt" in landed


def test_an_unlabelled_image_gets_an_empty_label_not_nothing(scattered):
    """A missing label file and an empty one mean different things to
    YOLO: the empty one is a background example."""
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo")},
        classes={"0": "helmet"}, destination="data/built",
    )
    assert scattered.put["data/built/labels/train/c.txt"] == b""


def test_an_orphan_label_is_not_carried_over(scattered):
    """ghost.txt has no image; copying it would add a label the trainer
    could never match to anything."""
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo")},
        classes={"0": "helmet"}, destination="data/built",
    )
    assert "data/built/labels/train/ghost.txt" not in scattered.objects


def test_the_assembled_data_yaml_points_at_the_new_layout(scattered):
    import yaml as _yaml
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo"),
                "val": ("raw/valpics", "annotations/val")},
        classes={"0": "helmet", "1": "gloves"}, destination="data/built",
    )
    manifest = _yaml.safe_load(scattered.put["data/built/data.yaml"].decode())
    assert manifest["names"] == {0: "helmet", 1: "gloves"}
    assert manifest["train"] == "images/train"
    assert manifest["val"] == "images/val"


def test_without_a_val_split_val_falls_back_to_train(scattered):
    """Ultralytics requires a val key. Pointing it at the training images
    is what the pipeline already does when no separate set is given."""
    import yaml as _yaml
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo"), "val": ("", "")},
        classes={"0": "helmet"}, destination="data/built",
    )
    manifest = _yaml.safe_load(scattered.put["data/built/data.yaml"].decode())
    assert manifest["val"] == "images/train"


def test_assembling_refuses_without_classes(scattered):
    with pytest.raises(ValueError, match="class list"):
        t2.assemble_dataset(
            splits={"train": ("raw/pics", "annotations/yolo")},
            classes={}, destination="data/built",
        )


def test_assembling_refuses_when_there_are_no_images(scattered):
    with pytest.raises(ValueError, match="no images"):
        t2.assemble_dataset(
            splits={"train": ("annotations/yolo", "annotations/yolo")},
            classes={"0": "helmet"}, destination="data/built",
        )


def test_images_are_copied_server_side_not_downloaded(scattered):
    """A dataset is gigabytes; pulling it through this container to put it
    straight back would be absurd."""
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo")},
        classes={"0": "helmet"}, destination="data/built",
    )
    assert any(src == "raw/pics/a.jpg" for src, _ in scattered.copied)


def test_the_layout_endpoint_reports_the_pairing(client, scattered):
    body = client.post("/track2/models/dataset-layout", json={
        "train_images": "raw/pics", "train_labels": "annotations/yolo",
    }).json()
    assert body["splits"][0]["paired"] == 2


def test_the_assemble_endpoint_refuses_a_bad_request(client, scattered):
    res = client.post("/track2/models/dataset-assemble", json={
        "train_images": "raw/pics", "train_labels": "annotations/yolo", "classes": {},
    })
    assert res.status_code == 400


# --- is the evaluation actually held out? -------------------------------------

def _yaml_s3(monkeypatch, body: str | None):
    class FakeS3:
        def get_object(self, Bucket, Key):
            if body is None:
                raise RuntimeError("NoSuchKey")
            import io
            return {"Body": io.BytesIO(body.encode())}
    monkeypatch.setattr(t2, "s3", FakeS3())


def test_separate_train_and_val_splits_are_held_out(monkeypatch):
    """An assembled dataset names ONE folder for both fields while the
    evaluation is genuinely held out -- the evaluator runs
    model.val(split="val"). Reporting that as "same as training" made good
    metrics look untrustworthy."""
    _yaml_s3(monkeypatch, "names:\n  0: helmet\ntrain: images/train\nval: images/val\n")
    assert t2._evaluates_on_its_training_images("data/assembled/x", True) is False
    # Same answer when no separate folder was chosen: the folder decides.
    assert t2._evaluates_on_its_training_images("data/assembled/x", False) is False


def test_val_pointing_at_train_is_not_held_out(monkeypatch):
    """What assemble_dataset writes when there is no val split."""
    _yaml_s3(monkeypatch, "names:\n  0: helmet\ntrain: images/train\nval: images/train\n")
    assert t2._evaluates_on_its_training_images("data/x", True) is True


def test_a_dataset_declaring_no_val_is_not_held_out(monkeypatch):
    _yaml_s3(monkeypatch, "names:\n  0: helmet\ntrain: images/train\n")
    assert t2._evaluates_on_its_training_images("data/x", True) is True


def test_the_valid_spelling_is_accepted(monkeypatch):
    """Roboflow exports spell it `valid`."""
    _yaml_s3(monkeypatch, "names:\n  0: helmet\ntrain: train/images\nvalid: valid/images\n")
    assert t2._evaluates_on_its_training_images("data/x", True) is False


def test_an_unreadable_manifest_falls_back_to_the_old_proxy(monkeypatch):
    """Never claim a held-out evaluation that cannot be shown."""
    _yaml_s3(monkeypatch, None)
    assert t2._evaluates_on_its_training_images("data/x", False) is True
    assert t2._evaluates_on_its_training_images("data/x", True) is False


def test_no_folder_at_all_is_not_held_out(monkeypatch):
    _yaml_s3(monkeypatch, None)
    assert t2._evaluates_on_its_training_images("", True) is True

# --- recording every file name, and exporting them -----------------------------
#
# The file names live in their own table, one row per file. A DynamoDB
# item is capped at 400KB and a 24,000-image dataset's names are ~2.4MB,
# so putting them on the run's own row would not truncate the list -- it
# would reject the write and lose the whole audit record.


class FakeFilesTable:
    """Enough of a (audit_id, file_key) table to exercise the queries."""

    def __init__(self):
        self.items: list[dict] = []

    class _Batch:
        def __init__(self, table):
            self.table = table

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def put_item(self, Item):
            self.table.items.append(Item)

        def delete_item(self, Key):
            self.table.items = [
                i for i in self.table.items
                if not (i["audit_id"] == Key["audit_id"] and i["file_key"] == Key["file_key"])
            ]

    def batch_writer(self):
        return self._Batch(self)

    def query(self, KeyConditionExpression=None, IndexName=None, Limit=None,
              ProjectionExpression=None, ExclusiveStartKey=None):
        # The fake reads the condition's rendered form rather than
        # re-implementing boto3's expression builder.
        rendered = KeyConditionExpression.get_expression()
        if IndexName == "by-file-name":
            wanted = rendered["values"][1]
            rows = [i for i in self.items if i["name"] == wanted]
        else:
            left = rendered["values"][0] if rendered["operator"] == "AND" else rendered
            audit_id = (left.get_expression()["values"][1]
                        if rendered["operator"] == "AND" else rendered["values"][1])
            rows = [i for i in self.items if i["audit_id"] == audit_id]
            if rendered["operator"] == "AND":
                prefix = rendered["values"][1].get_expression()["values"][1]
                rows = [i for i in rows if i["file_key"].startswith(prefix)]
        rows = sorted(rows, key=lambda i: i["file_key"])
        return {"Items": rows[:Limit] if Limit else rows}


@pytest.fixture
def files_table(monkeypatch) -> FakeFilesTable:
    fake = FakeFilesTable()
    monkeypatch.setattr(t2, "training_files_table", lambda: fake)
    return fake


def test_every_name_gets_its_own_row(files_table):
    """No 400KB ceiling: 2,500 names is 2,500 rows, not a rejected write."""
    names = [f"images/train/img{i:05d}.jpg" for i in range(2500)]
    assert t2.write_file_rows("abc123", "train", names) == 2500
    assert len(files_table.items) == 2500


def test_a_row_records_the_folder_it_was_relative_to(files_table):
    """"images/train/x.jpg" alone cannot say WHERE it lived, so the same
    dataset copied to a second location is indistinguishable from the
    first. The folder and the full S3 uri are recorded beside it."""
    t2.write_file_rows("abc123", "train", ["images/train/x.jpg"], "test_delete/ds")
    row = files_table.items[0]
    assert row["folder"] == "test_delete/ds"
    assert row["s3_uri"] == f"s3://{t2.BUCKET}/test_delete/ds/images/train/x.jpg"
    assert row["file_name"] == "x.jpg"


def test_the_split_is_recorded_because_role_does_not_give_it(files_table):
    """`role` is "the folder used for training" vs "the folder used for
    evaluation", and each lists the WHOLE dataset -- so without `split` a
    row cannot answer whether the file was trained on or held out."""
    t2.write_file_rows("abc123", "train",
                       ["images/train/a.jpg", "images/val/b.jpg"], "data/ds")
    assert [r["split"] for r in files_table.items] == ["train", "val"]


def test_name_is_left_alone_so_reverse_lineage_still_resolves(files_table):
    """The by-file-name GSI is keyed on `name`; changing it to a full path
    would silently break runs_using_file() for every existing row."""
    t2.write_file_rows("run-one", "train", ["images/train/shared.jpg"], "data/ds")
    assert t2.read_file_rows("run-one", "train") == ["images/train/shared.jpg"]
    assert t2.runs_using_file("images/train/shared.jpg") == ["run-one"]


def test_label_files_get_rows_too(files_table):
    """A .txt IS the supervision the run learned from -- two runs over the
    same images with different labels must not look identical here."""
    t2.write_file_rows("abc123", "train",
                       ["images/train/a.jpg", "labels/train/a.txt"], "data/ds")
    kinds = {r["name"]: r["kind"] for r in files_table.items}
    assert kinds == {"images/train/a.jpg": "image", "labels/train/a.txt": "label"}


def test_images_and_labels_are_separately_queryable(files_table):
    t2.write_file_rows("abc123", "train",
                       ["images/train/a.jpg", "images/train/b.jpg",
                        "labels/train/a.txt", "labels/train/b.txt"], "data/ds")
    assert t2.read_file_rows("abc123", "train", "image") == [
        "images/train/a.jpg", "images/train/b.jpg"]
    assert t2.read_file_rows("abc123", "train", "label") == [
        "labels/train/a.txt", "labels/train/b.txt"]
    assert len(t2.read_file_rows("abc123", "train")) == 4


def test_each_kind_is_indexed_from_zero_so_order_survives(files_table):
    """Padded per kind, not across both -- otherwise the labels' indices
    continue the images' and the two interleave when read back."""
    names = [f"images/train/img{i}.jpg" for i in range(12)]
    names += [f"labels/train/img{i}.txt" for i in range(12)]
    t2.write_file_rows("abc123", "train", names, "data/ds")
    assert t2.read_file_rows("abc123", "train", "image") == names[:12]
    assert t2.read_file_rows("abc123", "train", "label") == names[12:]


def test_rows_read_back_in_listing_order(files_table):
    """The index is zero-padded so the sort order IS the listing order --
    unpadded, img10 would sort before img9."""
    names = [f"img{i}.jpg" for i in range(12)]
    t2.write_file_rows("abc123", "train", names)
    assert t2.read_file_rows("abc123", "train") == names


def test_train_and_eval_rows_are_separable(files_table):
    t2.write_file_rows("abc123", "train", ["a.jpg", "b.jpg"])
    t2.write_file_rows("abc123", "eval", ["v.jpg"])
    assert t2.read_file_rows("abc123", "train") == ["a.jpg", "b.jpg"]
    assert t2.read_file_rows("abc123", "eval") == ["v.jpg"]
    assert sorted(t2.read_file_rows("abc123")) == ["a.jpg", "b.jpg", "v.jpg"]


def test_reverse_lineage_answers_which_runs_used_an_image(files_table):
    """The whole reason these are rows rather than a blob of names."""
    t2.write_file_rows("run-one", "train", ["shared.jpg", "only-one.jpg"])
    t2.write_file_rows("run-two", "train", ["shared.jpg"])
    assert sorted(t2.runs_using_file("shared.jpg")) == ["run-one", "run-two"]
    assert t2.runs_using_file("only-one.jpg") == ["run-one"]
    assert t2.runs_using_file("never-used.jpg") == []


def test_a_run_appears_once_however_many_of_its_files_match(files_table):
    t2.write_file_rows("run-one", "train", ["a.jpg"])
    t2.write_file_rows("run-one", "eval", ["a.jpg"])
    assert t2.runs_using_file("a.jpg") == ["run-one"]


def test_deleting_a_run_takes_its_file_rows(files_table):
    """Thousands of rows pointing at a deleted run would be invisible and
    never collected."""
    t2.write_file_rows("abc123", "train", [f"img{i}.jpg" for i in range(30)])
    t2.write_file_rows("other", "train", ["keep.jpg"])
    assert t2.delete_file_rows("abc123") == 30
    assert t2.read_file_rows("abc123") == []
    assert t2.read_file_rows("other") == ["keep.jpg"]


def test_a_failed_write_does_not_lose_the_run(monkeypatch):
    class Exploding:
        def batch_writer(self):
            raise RuntimeError("AccessDenied")
    monkeypatch.setattr(t2, "training_files_table", lambda: Exploding())
    assert t2.write_file_rows("abc123", "train", ["a.jpg"]) == 0


def test_the_audit_block_keeps_a_sample_and_the_rows_keep_everything(files_table, monkeypatch):
    class FakeS3:
        def get_paginator(self, name):
            class Paginator:
                def paginate(self, Bucket, Prefix):
                    yield {"Contents": [
                        {"Key": f"{Prefix}images/train/img{i:05d}.jpg"} for i in range(2500)
                    ]}
            return Paginator()
    monkeypatch.setattr(t2, "s3", FakeS3())
    block = t2._dataset_audit_block("data/site", "abc123", "train")
    assert block["file_count"] == 2500
    assert len(block["file_names"]) == t2.AUDIT_MAX_FILE_NAMES   # the preview
    assert block["truncated"] is True
    assert block["file_rows"] == 2500                            # the record
    assert len(t2.read_file_rows("abc123", "train")) == 2500


def test_the_csv_export_lists_every_file_not_the_sample(files_table):
    t2.write_file_rows("abc123", "train",
                       [f"images/train/img{i}.jpg" for i in range(1500)], "data/site")
    t2.write_file_rows("abc123", "eval", ["images/val/v1.jpg"], "data/site")
    audit = {
        "id": "abc123", "epochs": 50, "notes": "a run",
        "model": {"version": 3},
        "train_dataset": {"folder": "data/site", "file_count": 1500,
                          "file_names": ["images/train/img0.jpg"]},
        "eval_dataset": {"folder": "data/site", "file_count": 1, "file_names": []},
    }
    text = t2.audit_as_csv(audit)
    lines = text.splitlines()
    assert lines[0] == "TRAINING RUN RECORD"
    files = lines[lines.index("TRAINING FILES (1500 files)") + 2:]
    assert len([r for r in files[:1501] if r.startswith("data/site/images/train/")]) == 1500
    assert "epochs,50" in text
    assert "Registered model > version,3" in text
    # Qualified: "images/val/v1.jpg" on its own names a file in no
    # particular dataset, and those names recur in every pull.
    assert "\ndata/site/images/val/v1.jpg\n" in text


def test_the_csv_falls_back_for_runs_predating_the_file_table(files_table):
    audit = {
        "id": "old", "train_dataset": {"folder": "d", "file_names": ["a.jpg", "b.jpg"]},
        "eval_dataset": {},
    }
    text = t2.audit_as_csv(audit)
    # No rows for this run, so it falls back to the audit's own sample.
    # Those names are bare, not relative paths, so they go out as-is.
    assert "TRAINING FILES (2 files)\nPath\na.jpg\nb.jpg\n" in text


def test_assemble_records_where_each_image_came_from(scattered):
    """Without this the trail stops at the copy: the audit can name
    data/assembled/<date>/images/train/x.jpg and nothing about the folder
    a person actually chose."""
    import json as _json

    fake = scattered
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo")},
        classes={"0": "helmet"},
        destination="out",
    )
    written = _json.loads(fake.put["out/sources.json"].decode())["sources"]
    assert written, "no source mapping was recorded"
    assert written["images/train/a.jpg"] == "raw/pics/a.jpg"
    for dest_relative, origin in written.items():
        assert dest_relative.startswith("images/train/")
        assert origin.startswith("raw/pics/")


def test_a_row_records_the_origin_not_only_the_copy(files_table):
    t2.write_file_rows(
        "abc123", "train", ["images/train/x.jpg"], "data/assembled/20260922",
        {"images/train/x.jpg": "test_delete/escalation_dataset/images/train/x.jpg"},
    )
    row = files_table.items[0]
    assert row["source_key"] == "test_delete/escalation_dataset/images/train/x.jpg"
    # name is untouched: the by-file-name index is keyed on it.
    assert row["name"] == "images/train/x.jpg"
    assert t2.read_file_row_paths("abc123", "train") == [
        "test_delete/escalation_dataset/images/train/x.jpg"
    ]


def test_a_row_without_an_origin_still_gives_a_qualified_path(files_table):
    """A hand-built folder has no sources.json, so the assembled path is
    all there is -- still qualified, never a bare relative name."""
    t2.write_file_rows("abc123", "train", ["images/train/x.jpg"], "data/site")
    assert t2.read_file_row_paths("abc123", "train") == ["data/site/images/train/x.jpg"]


class _DeletingS3(FakeS3Flat):
    def __init__(self, keys):
        super().__init__(keys)
        self.deleted: list[str] = []

    def delete_objects(self, Bucket, Delete):
        for obj in Delete["Objects"]:
            self.deleted.append(obj["Key"])
            self.objects.pop(obj["Key"], None)
        return {}


def _assembled_audit(**overrides):
    audit = {
        "id": "run-1",
        "status": "Succeeded",
        "train_dataset": {"folder": "data/assembled/20260922-1200", "file_rows": 50},
        "eval_dataset": {},
    }
    audit.update(overrides)
    return audit


def _cleanup(monkeypatch, audit, keys, others=()):
    fake = _DeletingS3(list(keys))
    monkeypatch.setattr(t2, "s3", fake)
    monkeypatch.setattr(t2, "scan_training_audits", lambda **_kw: list(others))
    return fake, t2.cleanup_assembled_dataset(audit)


def test_a_finished_run_s_assembled_copy_is_deleted(monkeypatch):
    """The copy is scaffolding: once the run has read it, the file rows
    are the record and the duplicate is just storage."""
    keys = ["data/assembled/20260922-1200/images/train/a.jpg",
            "data/assembled/20260922-1200/labels/train/a.txt",
            "data/assembled/20260922-1200/data.yaml"]
    fake, removed = _cleanup(monkeypatch, _assembled_audit(), keys)
    assert removed == 3
    assert sorted(fake.deleted) == sorted(keys)


def test_a_folder_the_operator_chose_is_never_deleted(monkeypatch):
    """Only assembled copies. Pointing a run straight at test_delete/ or a
    pull must not end with that folder destroyed."""
    audit = _assembled_audit(
        train_dataset={"folder": "test_delete/escalation_dataset", "file_rows": 50},
    )
    fake, removed = _cleanup(
        monkeypatch, audit, ["test_delete/escalation_dataset/data.yaml"],
    )
    assert removed == 0
    assert fake.deleted == []


def test_nothing_is_deleted_while_the_run_is_still_going(monkeypatch):
    audit = _assembled_audit(status="Executing")
    fake, removed = _cleanup(
        monkeypatch, audit, ["data/assembled/20260922-1200/data.yaml"],
    )
    assert removed == 0
    assert fake.deleted == []


def test_nothing_is_deleted_when_no_lineage_was_recorded(monkeypatch):
    """Deleting the data AND having no record of it is the one outcome
    worse than keeping the copy."""
    audit = _assembled_audit(
        train_dataset={"folder": "data/assembled/20260922-1200", "file_rows": 0},
    )
    fake, removed = _cleanup(
        monkeypatch, audit, ["data/assembled/20260922-1200/data.yaml"],
    )
    assert removed == 0
    assert fake.deleted == []


def test_a_folder_another_live_run_still_points_at_is_kept(monkeypatch):
    """Assemble once, queue three runs: deleting when the first finishes
    would pull the dataset out from under the other two."""
    others = [{"id": "run-2", "status": "Executing",
               "train_dataset": {"folder": "data/assembled/20260922-1200"}}]
    fake, removed = _cleanup(
        monkeypatch, _assembled_audit(), ["data/assembled/20260922-1200/data.yaml"],
        others=others,
    )
    assert removed == 0
    assert fake.deleted == []


def test_a_folder_only_finished_runs_used_is_deleted(monkeypatch):
    others = [{"id": "run-2", "status": "Succeeded",
               "train_dataset": {"folder": "data/assembled/20260922-1200"}}]
    fake, removed = _cleanup(
        monkeypatch, _assembled_audit(), ["data/assembled/20260922-1200/data.yaml"],
        others=others,
    )
    assert removed == 1


def test_cleanup_does_not_run_twice(monkeypatch):
    audit = _assembled_audit(assembled_cleaned_at="2026-09-22T12:00:00Z")
    fake, removed = _cleanup(
        monkeypatch, audit, ["data/assembled/20260922-1200/data.yaml"],
    )
    assert removed == 0


def test_stopping_the_queue_still_clears_the_finished_entry(monkeypatch):
    """Stop Queue means "start no more", not "forget how to notice the
    current one ended". It used to return before clearing, so the card
    said Running forever with nothing executing -- and every further Stop
    Training answered 409 because there was nothing left to stop."""
    monkeypatch.setattr(t2, "training_already_running_reason", lambda _p: None)
    monkeypatch.setattr(t2, "get_training_progress", lambda *_a, **_k: None)
    t2._queue_running = {"id": "e1"}
    t2._queue_started = False          # Stop Queue was pressed
    t2._training_queue[:] = []

    t2.try_advance_queue()

    assert t2._queue_running is None


def test_a_stopped_queue_starts_nothing_new(monkeypatch):
    monkeypatch.setattr(t2, "training_already_running_reason", lambda _p: None)
    monkeypatch.setattr(t2, "get_training_progress", lambda *_a, **_k: None)
    started: list = []
    monkeypatch.setattr(t2, "start_custom_training_run",
                        lambda **kw: started.append(kw) or {"execution_arn": "a", "run_id": "r"})
    t2._queue_running = None
    t2._queue_started = False
    t2._training_queue[:] = [{"id": "e2", "custom_classes": True}]

    t2.try_advance_queue()

    assert started == []
    assert len(t2._training_queue) == 1


def test_each_line_shows_only_its_own_split(scattered, monkeypatch):
    """"Training images: .../train, .../val" claims the run trained on its
    own held-out set. Each line gets the split it actually reads."""
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo"),
                "val": ("raw/valpics", "annotations/val")},
        classes={"0": "helmet"}, destination="out",
    )
    monkeypatch.setattr(t2, "_persist_queue_entry", lambda _e: None)
    entry = {"id": "e1", "dataset_folder": "out", "eval_dataset_folder": ""}
    t2.queue_add(entry)
    t2.queue_remove("e1")

    assert entry["dataset_source_folders"] == ["raw/pics"]
    assert entry["eval_source_folders"] == ["raw/valpics"]


def test_without_a_val_split_evaluation_reports_the_training_origins(scattered, monkeypatch):
    """data.yaml points val at images/train when there is no val split, so
    the training origins are the honest answer -- not an empty line."""
    t2.assemble_dataset(
        splits={"train": ("raw/pics", "annotations/yolo")},
        classes={"0": "helmet"}, destination="out2",
    )
    monkeypatch.setattr(t2, "_persist_queue_entry", lambda _e: None)
    entry = {"id": "e2", "dataset_folder": "out2", "eval_dataset_folder": ""}
    t2.queue_add(entry)
    t2.queue_remove("e2")

    assert entry["dataset_source_folders"] == ["raw/pics"]
    assert entry["eval_source_folders"] == ["raw/pics"]


def test_the_csv_escapes_values_containing_commas(files_table):
    """Notes are free text; an unescaped comma would shift every column."""
    text = t2.audit_as_csv(
        {"id": "x", "notes": "helmet, gloves, person", "train_dataset": {}, "eval_dataset": {}},
    )
    assert '"helmet, gloves, person"' in text


# --- whose progress is the queue showing? -------------------------------------

def test_progress_reads_the_pipeline_it_is_given(monkeypatch):
    """It was hardcoded to the PRODUCTION pipeline, so while a custom run
    executed the queue showed the last production execution -- a 1-epoch
    custom run displayed "Epoch 0 / 50", numbers from a different run on a
    different pipeline that had already finished."""
    asked = []

    class FakeSm:
        def list_pipeline_executions(self, PipelineName, **kwargs):
            asked.append(PipelineName)
            return {"PipelineExecutionSummaries": [{
                "PipelineExecutionArn": "arn:exec/1",
                "PipelineExecutionStatus": "Executing",
                "StartTime": None,
            }]}

        def list_pipeline_execution_steps(self, PipelineExecutionArn):
            return {"PipelineExecutionSteps": []}

    monkeypatch.setattr(t2, "sm", FakeSm())
    t2.get_training_progress("track2-yolo-open-training-pipeline")
    t2.get_training_progress()
    assert asked == ["track2-yolo-open-training-pipeline", t2.PIPELINE_NAME]


def test_the_epoch_total_comes_from_the_job_not_a_constant(monkeypatch):
    """Which is why pointing at the right pipeline fixes the "/50" too:
    the total is the training job's own `epochs` hyperparameter."""
    class FakeSm:
        def describe_training_job(self, TrainingJobName):
            return {"HyperParameters": {"epochs": "1"}}

    monkeypatch.setattr(t2, "sm", FakeSm())
    monkeypatch.setattr(t2, "_get_metric_max", lambda job, metric: 0)
    assert t2.get_epoch_progress("job") == {"current": 0, "total": 1}


# --- downloading the .pt a run produced ---------------------------------------

def test_the_weights_link_is_presigned_not_streamed(monkeypatch):
    """A checkpoint is 19MB+ and the API container shares one small box
    with every page, so the bytes go browser-to-S3."""
    captured = {}

    class FakeS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            captured.update(op=op, params=Params, expires=ExpiresIn)
            return "https://s3.example/signed"

    monkeypatch.setattr(t2, "s3", FakeS3())
    url = t2.presign_weights_download("s3://track2-s3/models/yolo11s-helmet-detector/v2/model.pt")
    assert url == "https://s3.example/signed"
    assert captured["op"] == "get_object"
    assert captured["params"]["Bucket"] == "track2-s3"
    assert captured["params"]["Key"] == "models/yolo11s-helmet-detector/v2/model.pt"
    assert captured["expires"] == 900


def test_the_download_is_named_after_the_group_and_version(monkeypatch):
    """Otherwise every checkpoint arrives as yet another "model.pt"."""
    captured = {}

    class FakeS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            captured.update(Params)
            return "https://s3.example/signed"

    monkeypatch.setattr(t2, "s3", FakeS3())
    t2.presign_weights_download("s3://b/models/yolo11s-helmet-detector/v2/model.pt")
    assert 'filename="yolo11s-helmet-detector-v2.pt"' in captured["ResponseContentDisposition"]


def test_an_unexpected_key_shape_keeps_the_original_filename(monkeypatch):
    class FakeS3:
        def generate_presigned_url(self, op, Params, ExpiresIn):
            return Params["ResponseContentDisposition"]

    monkeypatch.setattr(t2, "s3", FakeS3())
    assert 'filename="best.pt"' in t2.presign_weights_download("s3://b/best.pt")


def test_a_non_s3_uri_is_refused():
    with pytest.raises(ValueError, match="s3://"):
        t2.presign_weights_download("https://example.com/model.pt")


def test_the_detector_list_is_read_consistently(monkeypatch):
    """A plain Scan is eventually consistent, so deleting a class list and
    refetching could still return it -- and the stale answer was then
    cached for 30s, so the row appeared to come back from the dead."""
    seen = {}

    class RecordingTable:
        def scan(self, **kwargs):
            seen.update(kwargs)
            return {"Items": []}

    monkeypatch.setattr(t2, "training_audit_table", lambda: RecordingTable())
    t2._invalidate_detector_cache()
    try:
        t2.list_detectors()
    finally:
        t2._invalidate_detector_cache()
    assert seen.get("ConsistentRead") is True


# --- assembling from several folders at once ----------------------------------
#
# A split used to be one (images, labels) pair. It can now be a list of
# them, so a dataset scattered across folders assembles in one go. Both
# shapes stay accepted, because the single-source panel still sends a pair.


@pytest.fixture
def two_sources(monkeypatch):
    """Two image folders that BOTH contain img001.jpg -- the collision the
    merge has to survive."""
    fake = FakeS3Flat([
        "siteA/images/img001.jpg", "siteA/images/img002.jpg",
        "siteA/labels/img001.txt", "siteA/labels/img002.txt",
        "siteB/images/img001.jpg", "siteB/images/img009.jpg",
        "siteB/labels/img001.txt",
    ])
    monkeypatch.setattr(t2, "s3", fake)
    return fake


def test_a_single_pair_is_still_accepted():
    assert t2._as_sources(("img", "lbl")) == [("img", "lbl")]
    assert t2._as_sources([("a", "b"), ("c", "d")]) == [("a", "b"), ("c", "d")]
    assert t2._as_sources(None) == []
    assert t2._as_sources([]) == []


def test_several_folders_merge_into_one_split(two_sources):
    out = t2.assemble_dataset(
        splits={"train": [("siteA/images", "siteA/labels"),
                          ("siteB/images", "siteB/labels")]},
        classes={"0": "helmet"}, destination="data/built",
    )
    # 2 from siteA + 2 from siteB, none lost to the name clash.
    assert out["images"] == 4
    landed = {dst for _src, dst in two_sources.copied}
    assert "data/built/images/train/img002.jpg" in landed
    assert "data/built/images/train/img009.jpg" in landed


def test_a_clashing_filename_is_kept_not_overwritten(two_sources):
    """Both folders hold img001.jpg. Overwriting would drop an image with
    no error anywhere -- the dataset would just be short."""
    t2.assemble_dataset(
        splits={"train": [("siteA/images", "siteA/labels"),
                          ("siteB/images", "siteB/labels")]},
        classes={"0": "helmet"}, destination="data/built",
    )
    landed = {dst for _src, dst in two_sources.copied}
    assert "data/built/images/train/img001.jpg" in landed        # siteA
    assert "data/built/images/train/img001__s1.jpg" in landed    # siteB
    # And each keeps its own label.
    assert "data/built/labels/train/img001.txt" in landed
    assert "data/built/labels/train/img001__s1.txt" in landed


def test_the_sources_map_records_which_folder_each_file_came_from(two_sources):
    """The only record of where a merged dataset came from."""
    t2.assemble_dataset(
        splits={"train": [("siteA/images", "siteA/labels"),
                          ("siteB/images", "siteB/labels")]},
        classes={"0": "helmet"}, destination="data/built",
    )
    import json as _json
    written = _json.loads(two_sources.put["data/built/sources.json"].decode())["sources"]
    assert written["images/train/img001.jpg"] == "siteA/images/img001.jpg"
    assert written["images/train/img001__s1.jpg"] == "siteB/images/img001.jpg"


def test_the_preview_reports_each_source_separately(two_sources):
    """One total would hide an empty folder sitting among full ones."""
    report = t2.inspect_dataset_layout(splits={
        "train": [("siteA/images", "siteA/labels"), ("siteB/images", "siteB/labels")],
    })
    assert [s["source_index"] for s in report["splits"]] == [0, 1]
    assert report["splits"][0]["image_count"] == 2
    assert report["splits"][1]["image_count"] == 2
    assert report["splits"][1]["unlabelled"] == 1     # img009 has no label
    assert report["total_paired"] == 3


def test_val_is_detected_from_a_list_of_sources(two_sources):
    """`splits["val"][0]` read the first PAIR once a split became a list,
    and raised outright on an empty one."""
    import yaml as _yaml
    t2.assemble_dataset(
        splits={"train": [("siteA/images", "siteA/labels")],
                "val": [("siteB/images", "siteB/labels")]},
        classes={"0": "helmet"}, destination="data/built",
    )
    manifest = _yaml.safe_load(two_sources.put["data/built/data.yaml"].decode())
    assert manifest["val"] == "images/val"


def test_no_val_sources_falls_back_to_the_training_images(two_sources):
    import yaml as _yaml
    t2.assemble_dataset(
        splits={"train": [("siteA/images", "siteA/labels")], "val": []},
        classes={"0": "helmet"}, destination="data/built",
    )
    manifest = _yaml.safe_load(two_sources.put["data/built/data.yaml"].decode())
    assert manifest["val"] == "images/train"


def test_discover_dataset_sources_expands_every_week(monkeypatch):
    """One week's shape teaches the rest.

    The real roots nest several segments deep
    (`pull18.../track2/escalation_dataset/images`), so a subpath is a
    path, not a folder name -- and a week that was uploaded without its
    images has to be visible as such before anything trains on it, not
    after.
    """
    weeks = {
        "week_1": 3,
        "week_2": 2,
        # Uploaded, but nothing in it yet.
        "week_3": 0,
    }

    def fake_folders(prefix):
        assert prefix == "retraining"
        return list(weeks)

    def fake_keys(prefix):
        for name, count in weeks.items():
            images = f"retraining/{name}/track2/set/images/"
            labels = f"retraining/{name}/track2/set/labels/"
            if prefix == images:
                return [f"{images}img{i}.jpg" for i in range(count)]
            if prefix == labels:
                # week_2 is missing one label -- pairing must notice.
                drop = 1 if name == "week_2" else 0
                return [f"{labels}img{i}.txt" for i in range(max(count - drop, 0))]
        return []

    monkeypatch.setattr(t2, "list_s3_folders", fake_folders)
    monkeypatch.setattr(t2, "_list_keys", fake_keys)

    out = t2.discover_dataset_sources(
        root="retraining/",
        images_subpath="track2/set/images",
        labels_subpath="track2/set/labels",
    )

    assert out["root"] == "retraining"
    assert [c["name"] for c in out["children"]] == ["week_1", "week_2", "week_3"]
    # The empty week is reported, but is not offered as a source.
    assert out["usable"] == ["week_1", "week_2"]

    by_name = {c["name"]: c for c in out["children"]}
    assert by_name["week_1"]["images_prefix"] == "retraining/week_1/track2/set/images"
    assert by_name["week_1"]["labels_prefix"] == "retraining/week_1/track2/set/labels"
    assert by_name["week_1"]["paired"] == 3
    assert by_name["week_2"]["image_count"] == 2
    assert by_name["week_2"]["paired"] == 1
    assert by_name["week_3"]["image_count"] == 0


def test_discover_dataset_sources_needs_a_root():
    with pytest.raises(ValueError):
        t2.discover_dataset_sources(root="  ")


# --- adding a generation must not orphan what is already registered ----------

def test_widening_a_legacy_pattern_keeps_every_yolo11_group_name(detector_table):
    """The safety property behind adding YOLO26 to existing detectors.

    Records written before YOLO26 stored the pattern `yolo11{size}-...`.
    They are widened on read to `yolo{generation}{size}-...` so they gain
    the 26 groups -- and that is only safe because the widened pattern
    produces the IDENTICAL string for generation 11. If it did not, every
    version already registered would still be sitting in the old group
    while the app looked in a new one, and the models would simply
    disappear from the page.
    """
    legacy = {"id": "detector#vehicle", "slug": "vehicle", "label": "Vehicle",
              "classes": {"0": "vehicle"},
              "group_pattern": "yolo11{size}-vehicle-detector"}
    widened = t2._detector_from_item(legacy)

    assert widened["group_pattern"] == "yolo{generation}{size}-vehicle-detector"
    for size in t2.REGISTRY_SIZES:
        assert t2.detector_group(widened, size, "11") == f"yolo11{size}-vehicle-detector"
    assert "26" in t2.detector_generations(widened)


def test_the_builtin_stays_yolo11_only(detector_table):
    """`yolo26s-detector` would NOT be on the exact-match list the
    approval rule watches, so a YOLO26 model registered under the
    built-in would look like production and never deploy. Better to have
    no such group at all than a convincing dead one."""
    builtin = t2.get_detector("ppe")
    assert t2.detector_generations(builtin) == ("11",)
    assert [g for _gen, _size, g in t2.detector_groups(builtin)] == [
        f"yolo11{size}-detector" for size in t2.REGISTRY_SIZES
    ]


def test_an_architecture_splits_on_the_right_generation():
    """"yolo26m" must not keep its prefix and become a size called
    "yolo26m" -- which is what stripping only the first generation did."""
    assert t2.split_architecture("yolo11s") == ("11", "s")
    assert t2.split_architecture("yolo26m") == ("26", "m")
    assert t2.split_architecture("  YOLO26X  ") == ("26", "x")
    with pytest.raises(ValueError):
        t2.split_architecture("yolo26n")        # nano is dropped by policy
    with pytest.raises(ValueError):
        t2.split_architecture("yolov8s")
