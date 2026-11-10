import json
import pytest
from ppe_pipeline_api.data import sql_settings as ss
from ppe_pipeline_api.routers import track2_sql as router
from fastapi import HTTPException

@pytest.fixture
def settings_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ss.pull_control, 'SETTINGS_DIR', str(tmp_path))
    return tmp_path

def test_round_trip_all_page_controls_and_suffix(settings_dir):
    values = ss.SqlSettings(database='jpui-preview', preset_class='worker_on_edge',
        preset_scope='resolved', pull_bucket='example', pull_folder='training/set1',
        train_percent=67, show_labels=False, with_labels=False,
        exploration_hazard='worker_on_edge', exploration_open=False, frames_open=False,
        export_mode='classification', detector_names=['person', 'helmet'],
        classifier_names=['no_helmet', 'helmet_worn'])
    saved = ss.save('review', values)
    assert saved['name'] == 'review_sql'
    assert json.loads((settings_dir/'sql'/'review_sql.json').read_text())['settings'] == values.model_dump()
    assert ss.load('review_sql')['settings'] == values.model_dump()
    ss.save('review_SQL', ss.SqlSettings())
    assert len(ss.list_files()['files']) == 1
    assert ss.load('review')['settings']['train_percent'] == 80
    ss.delete('review_sql')
    assert ss.list_files()['files'] == []

def test_presets_are_isolated_from_data_preparation(settings_dir):
    existing = settings_dir/'unchanged_DP.json'
    existing.write_text('{"config":{"mode":"scheduled"}}')
    ss.save('example', ss.SqlSettings())
    assert existing.read_text() == '{"config":{"mode":"scheduled"}}'
    assert [f['name'] for f in ss.list_files()['files']] == ['example_sql']
    assert not list((settings_dir/'sql').glob('*.tmp'))

@pytest.mark.parametrize('name', ['', '../escape', 'a/b', 'a\\b', '.hidden', 'a:b', '_sql'])
def test_reject_unsafe_names(settings_dir, name):
    with pytest.raises(ValueError):
        ss.save(name, ss.SqlSettings())

def test_missing_and_invalid_requests(settings_dir):
    with pytest.raises(HTTPException) as err:
        router.load_sql_settings(router.SqlSettingsRequest(name='absent'), session={})
    assert err.value.status_code == 404
    with pytest.raises(HTTPException) as err:
        router.save_sql_settings(router.SqlSettingsRequest(name='empty'), session={})
    assert err.value.status_code == 422
    with pytest.raises(ValueError):
        ss.SqlSettings(train_percent=101)

def test_latest_saved_file_is_first_even_after_overwrite(settings_dir):
    import os
    ss.save('older', ss.SqlSettings(train_percent=60))
    ss.save('newer', ss.SqlSettings(train_percent=80))
    os.utime(settings_dir/'sql'/'older_sql.json', (100, 100))
    assert ss.list_files()['files'][0]['name'] == 'newer_sql'
    ss.save('older', ss.SqlSettings(train_percent=75))
    os.utime(settings_dir/'sql'/'newer_sql.json', (100, 100))
    newest = ss.list_files()['files'][0]['name']
    assert newest == 'older_sql'
    assert ss.load(newest)['settings']['train_percent'] == 75
