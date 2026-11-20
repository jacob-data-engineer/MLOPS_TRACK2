from ppe_pipeline_api.routers import track2_retention as tr
from ppe_pipeline_api import dataset_versions
from ppe_pipeline_api.data import catalog_view


def test_dataset_summary_uses_permanent_run_numbers(monkeypatch):
    monkeypatch.setattr(dataset_versions, 'listing', lambda limit: [
        {'version_id': 'dv-a', 'run_id': 'run-a', 'data_yaml': 'large payload'},
        {'version_id': 'dv-b', 'run_id': 'run-b'}, {'version_id': 'dv-unused'}])
    def numbers(ids):
        assert set(ids) == {'run-a', 'run-b'}
        return {'run-a': 70, 'run-b': 68}
    monkeypatch.setattr(catalog_view, 'run_numbers', numbers)
    monkeypatch.setattr(catalog_view, 'dataset_run_numbers', lambda ids: {'dv-b': 68})
    rows = tr.list_versions(limit=100, summary=True)['versions']
    assert [row['run_number'] for row in rows] == [70, 68, None]
    assert 'data_yaml' not in rows[0]


def test_preparation_id_resolves_through_dataset(monkeypatch):
    monkeypatch.setattr(dataset_versions, 'listing', lambda limit: [
        {'version_id': 'dv-a', 'run_id': 'custom-preparation-id'}])
    monkeypatch.setattr(catalog_view, 'run_numbers', lambda ids: {})
    monkeypatch.setattr(catalog_view, 'dataset_run_numbers', lambda ids: {'dv-a': 70})
    assert tr.list_versions(summary=True)['versions'][0]['run_number'] == 70
