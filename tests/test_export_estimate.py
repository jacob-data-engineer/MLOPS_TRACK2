from types import SimpleNamespace
import pytest
from ppe_pipeline_api.data import export_estimate as ee

@pytest.fixture
def setup(monkeypatch):
    entries = [("images/train/a.jpg", "s3://b/a", {}), ("labels/train/a.txt", "s3://b/l", {}), ("", "s3://b/excluded", {})]
    monkeypatch.setattr(ee.cv, "_plan", lambda *a: (entries, {0: "helmet"}, "run-1"))
    monkeypatch.setattr(ee.cv, "_rows", lambda *a: [{"model_package_arn": "arn:1", "record": {"weights_uri": "s3://b/model.pt"}}])
    calls = []
    def head(**kw):
        calls.append(kw["Key"])
        return {"ContentLength": {"a": 1000, "l": 100, "model.pt": 5000}[kw["Key"]], "ETag": '"frozen"'}
    monkeypatch.setattr(ee.storage, "s3", SimpleNamespace(head_object=head))
    return entries, calls

def test_selected_files_only_and_no_export_writes(setup):
    _, calls = setup
    r = ee.estimate("run", "r", include_images=False, include_model=True, model_package_arn="arn:1")
    assert calls == ["model.pt"]
    assert r["source_bytes"] == r["model_bytes"] == 5000
    assert r["files"] == 1 and r["unknown_files"] == 0
    assert r["recommended_bytes"] > r["estimated_bytes"] > 5000

def test_shared_sources_count_per_exported_file_but_head_once(setup):
    entries, calls = setup
    entries.append(("eval/images/a.jpg", "s3://b/a", {}))
    r = ee.estimate("run", "r", format="folder")
    assert r["source_bytes"] == 2100
    assert calls.count("a") == 1
    assert "excluded" not in calls

def test_unknown_is_not_zero_and_blocks_complete_total(setup):
    entries, _ = setup
    entries.append(("images/missing.jpg", "s3://b/missing", {}))
    r = ee.estimate("run", "r")
    assert r["unknown_files"] == 1 and r["source_bytes"] == 1100
    assert r["estimated_bytes"] is None and r["recommended_bytes"] is None

def test_frozen_etag_mismatch_is_unknown(setup):
    entries, _ = setup
    entries[0] = ("images/train/a.jpg", "s3://b/a", {"stored": "s3://b/a", "stored_etag": "changed"})
    assert ee.estimate("run", "r")["unknown_files"] == 1

def test_wrong_checkpoint_identity_rejected(setup):
    with pytest.raises(ValueError, match="does not match"):
        ee.estimate("run", "r", include_model=True, model_package_arn="arn:other")

def test_metadata_selection_and_embedding_budget(setup, monkeypatch):
    monkeypatch.setattr(ee.catalog_metadata, "describe", lambda *a: {"groups": {"Identity": {"name": "helmet"}, "File provenance": {}}})
    monkeypatch.setattr(ee.catalog_metadata, "select_groups", lambda data, groups: {"groups": {g: data["groups"][g] for g in groups}})
    basic = ee.estimate("run", "r", include_images=False, include_model=True)
    enriched = ee.estimate("run", "r", include_images=False, include_model=True, include_metadata=True, metadata_groups=["Identity"])
    assert enriched["source_bytes"] == basic["source_bytes"]
    assert enriched["generated_bytes_estimate"] > basic["generated_bytes_estimate"]

def test_all_missing_files_are_reported_separately_from_access_errors(setup, monkeypatch):
    from botocore.exceptions import ClientError
    entries, _ = setup
    entries.extend([('images/other.jpg', 's3://b/other', {}), ('images/private.jpg', 's3://b/private', {})])
    def head(**kw):
        raise ClientError({'Error': {'Code': '403' if kw['Key'] == 'private' else '404'}}, 'HeadObject')
    monkeypatch.setattr(ee.storage, 's3', SimpleNamespace(head_object=head))
    result = ee.estimate('run', 'r')
    assert result['missing_files'] == 3
    assert result['check_failed_files'] == 1
    assert len(result['unavailable_files']) == 4
    assert {f['uri'] for f in result['unavailable_files']} == {'s3://b/a','s3://b/l','s3://b/other','s3://b/private'}


def test_changed_file_is_reported_and_checkpoint_issues_identified(setup):
    entries, _ = setup
    entries[0] = ('images/train/a.jpg', 's3://b/a', {'stored': 's3://b/a', 'stored_etag': 'changed'})
    result = ee.estimate('run', 'r', include_model=True)
    assert result['changed_files'] == 1
    assert result['missing_files'] == 0
    assert result['unavailable_files'][0]['reason'] == 'changed'

def test_missing_checkpoint_is_identified_as_model(setup, monkeypatch):
    from botocore.exceptions import ClientError
    def head(**kw):
        raise ClientError({'Error': {'Code': '404'}}, 'HeadObject')
    monkeypatch.setattr(ee.storage, 's3', SimpleNamespace(head_object=head))
    result = ee.estimate('run', 'r', include_model=True, include_images=False)
    assert result['missing_files'] == 1
    assert result['unavailable_files'][0]['role'] == 'model'
