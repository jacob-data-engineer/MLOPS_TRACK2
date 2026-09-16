"""Tests for `ppe_pipeline.track2_status.list_model_versions` against a
hand-rolled fake SageMaker client — same pattern as Track2's own
`tests/test_registry.py` — plus the "no infra yet" empty-result paths
(`ClientError`, `NoRegionError`, and anything else)."""

from __future__ import annotations

from botocore.exceptions import ClientError, NoRegionError

from aiops_pipeline.models import ModelType
from aiops_pipeline.registry import model_package_group_name

import ppe_pipeline.track2_status as track2_status


class _FakeRegistry:
    def __init__(self, client):
        self.client = client


class FakeSageMakerClient:
    def __init__(self):
        self.packages: dict[str, dict] = {}
        self._next_id = 0

    def add_package(self, group_name: str, **detail_overrides):
        self._next_id += 1
        arn = f"arn:aws:sagemaker:package/{group_name}/{self._next_id}"
        detail = {
            "ModelPackageArn": arn,
            "ModelPackageGroupName": group_name,
            "ModelPackageVersion": self._next_id,
            "CreationTime": self._next_id,
            "ModelApprovalStatus": "PendingManualApproval",
            "CustomerMetadataProperties": {},
            "ModelMetrics": {},
        }
        detail.update(detail_overrides)
        self.packages[arn] = detail
        return arn

    def list_model_packages(self, ModelPackageGroupName, SortBy=None, SortOrder=None):
        matches = [p for p in self.packages.values() if p["ModelPackageGroupName"] == ModelPackageGroupName]
        matches.sort(key=lambda p: p["CreationTime"], reverse=(SortOrder == "Descending"))
        return {"ModelPackageSummaryList": [{"ModelPackageArn": p["ModelPackageArn"]} for p in matches]}

    def describe_model_package(self, ModelPackageName):
        return self.packages[ModelPackageName]


def test_list_model_versions_returns_empty_when_group_has_no_packages(monkeypatch):
    fake_client = FakeSageMakerClient()
    monkeypatch.setattr(track2_status, "ModelRegistry", lambda region=None: _FakeRegistry(fake_client))

    assert track2_status.list_model_versions(ModelType.PPE) == []


def test_list_model_versions_returns_newest_first_with_expected_fields(monkeypatch):
    fake_client = FakeSageMakerClient()
    group_name = model_package_group_name(ModelType.PPE)
    fake_client.add_package(
        group_name,
        CustomerMetadataProperties={"deployment_status": "candidate"},
        ModelMetrics={"ModelQuality": {"Statistics": {"S3Uri": "s3://bucket/v1/eval_metrics.json"}}},
    )
    fake_client.add_package(
        group_name,
        CustomerMetadataProperties={"deployment_status": "uat"},
        ModelMetrics={"ModelQuality": {"Statistics": {"S3Uri": "s3://bucket/v2/eval_metrics.json"}}},
    )
    monkeypatch.setattr(track2_status, "ModelRegistry", lambda region=None: _FakeRegistry(fake_client))

    versions = track2_status.list_model_versions(ModelType.PPE)

    assert [v["version"] for v in versions] == [2, 1]
    assert versions[0]["deployment_status"] == "uat"
    assert versions[0]["eval_metrics_s3_uri"] == "s3://bucket/v2/eval_metrics.json"
    assert versions[1]["deployment_status"] == "candidate"


def test_list_model_versions_returns_empty_on_client_error(monkeypatch):
    def _raise(*args, **kwargs):
        raise ClientError({"Error": {"Code": "AccessDeniedException", "Message": "nope"}}, "ListModelPackages")

    fake_client = FakeSageMakerClient()
    monkeypatch.setattr(fake_client, "list_model_packages", _raise)
    monkeypatch.setattr(track2_status, "ModelRegistry", lambda region=None: _FakeRegistry(fake_client))

    assert track2_status.list_model_versions(ModelType.PPE) == []


def test_list_model_versions_returns_empty_on_no_region_error(monkeypatch):
    def _raise(region=None):
        raise NoRegionError()

    monkeypatch.setattr(track2_status, "ModelRegistry", _raise)

    assert track2_status.list_model_versions(ModelType.PPE) == []


def test_list_model_versions_returns_empty_on_unexpected_error(monkeypatch):
    def _raise(region=None):
        raise RuntimeError("something unrelated broke")

    monkeypatch.setattr(track2_status, "ModelRegistry", _raise)

    assert track2_status.list_model_versions(ModelType.PPE) == []
