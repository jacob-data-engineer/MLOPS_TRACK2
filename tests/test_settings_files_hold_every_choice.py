"""Every choice on the SQL page and on Data Preparation reaches the settings
file (2026-11-13 audit).

A control whose value is not in the file looks saved -- it is still on
screen -- and is gone on the next load. These pin the ones the audit found
missing, and the shape that keeps the SQL page's file whole.
"""
import pathlib
import re

from ppe_pipeline_api import pull_control as pc
from ppe_pipeline_api.data import retention, sql_settings
from ppe_pipeline_api.routers import track2_pull_control, track2_retention

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"


def _read(rel):
    return (WEB / rel).read_text(encoding="utf-8")


# --- SQL Database Query: the named _sql file ------------------------------------

def _keys_of(block: str) -> set:
    # `key: value`, an optional type's `key?:`, and shorthand `key,`
    return set(re.findall(r"(?:^|[\s{,;])([a-z_]+)\s*\??\s*[:,]", block))


def test_the_page_sends_every_field_the_file_keeps():
    page = _read("routes/Track2SqlPage.tsx")
    sent = page[page.index("const pageSettings: SqlPageSettings = {"):]
    sent = sent[:sent.index("};")]
    fields = set(sql_settings.SqlSettings.model_fields)
    missing = fields - _keys_of(sent)
    assert not missing, f"the SQL page never sends {sorted(missing)} to its settings file"


def test_the_page_restores_every_field_it_saves():
    page = _read("routes/Track2SqlPage.tsx")
    loader = page[page.index("function loadPageSettings"):]
    loader = loader[:loader.index("\n  }\n")]
    for field in sql_settings.SqlSettings.model_fields:
        if field in ("export_mode", "detector_names", "classifier_names"):
            continue  # fixed on this page; no control sets them
        assert f"next.{field}" in loader, f"{field} is saved but never loaded back"


def test_the_bar_type_matches_the_server_model():
    bar = _read("components/Track2SqlSettingsBar.tsx")
    typed = bar[bar.index("export type SqlPageSettings = {"):]
    typed = typed[:typed.index("};")]
    assert set(sql_settings.SqlSettings.model_fields) <= _keys_of(typed)


# --- Data Preparation: the _DP file ------------------------------------------------

def test_the_retention_forecast_choices_are_saved():
    assert {"forecast_weeks", "forecast_as_images"} <= set(retention.DEFAULTS)
    assert {"forecast_weeks", "forecast_as_images"} <= set(track2_retention.RetentionConfig.model_fields)
    card = _read("components/Track2RetentionPanel.tsx")
    assert "forecast_weeks: ahead," in card and "forecast_as_images: asImages," in card
    assert "setAhead(got.config.forecast_weeks)" in card


def test_the_pull_control_view_choices_are_saved_with_data_preparation():
    assert "pull_control_view" in pc.DEFAULTS
    assert "pull_control_view" in pc.SCOPES["dp"]
    assert "pull_control_view" in track2_pull_control.ConfigRequest.model_fields
    card = _read("components/Track2PullControlPanel.tsx")
    assert "pull_control_view: { watching, show_all_weeks: showAllWeeks }" in card
    assert "setShowAllWeeks(view.show_all_weeks)" in card


def test_the_data_preparation_copy_of_the_sql_page_is_whole():
    page = _read("routes/Track2SqlPage.tsx")
    assert "by_week: byWeek, train_after: trainAfter && byWeek, class_ids: classIds," in page
    assert "setClassIds(saved.class_ids)" in page


def test_every_config_key_a_data_preparation_card_writes_is_in_its_file():
    cards = ["Track2DatasetAssemblePanel", "Track2GoldenSetPanel", "Track2AutoTrainingPanel",
             "Track2CentralStorePanel", "Track2PullControlPanel"]
    written = set()
    for name in cards:
        code = _read(f"components/{name}.tsx")
        written |= set(re.findall(r"saveTrack2PullConfig\(\{\s*([a-z_]+)\s*:", code))
    written |= {"sql_explorer", "readiness_flow"}
    assert written, "the scan found nothing -- the pattern is stale"
    assert written <= set(pc.SCOPES["dp"]), sorted(written - set(pc.SCOPES["dp"]))
