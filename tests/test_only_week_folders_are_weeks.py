"""Only folders named like weeks are weeks.

A root held `_retained/` (where the retention lists are written) and
`helmet_test/` (a pull destination nested inside it), and both appeared
on the Retention card's grid and bars as empty weeks. One that held
`images/train` would have been trained on as the oldest week.
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import retained_lists as rl  # noqa: E402
from ppe_pipeline_api import retention as rt  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api.routers import track2_retention as tr  # noqa: E402

FOLDERS = ["_retained", "helmet_test", "week_1", "week_2", "week_10"]


def test_only_week_prefixed_names_survive():
    assert rt.only_weeks(FOLDERS) == ["week_1", "week_2", "week_10"]
    assert rt.only_weeks(["wk1", "week_1"], "wk") == ["wk1"]


def test_read_weeks_skips_the_rest(monkeypatch):
    import ppe_pipeline_api.track2_shared as t2
    monkeypatch.setattr(t2, "list_s3_folders", lambda root: list(FOLDERS))
    monkeypatch.setattr(t2, "_basenames", lambda prefix, ext: {})
    names = [w["name"] for w in rt.read_weeks("retraining")]
    assert names == ["week_1", "week_2", "week_10"]


def test_the_retention_lists_skip_it_too():
    assert "rt.only_weeks(t2.list_s3_folders(root)" in inspect.getsource(rl.select)
    assert 'retention["week_prefix"] = str(config.get("week_prefix") or "week_")' in \
        inspect.getsource(rs._run_chain_body)


def test_the_prefix_is_never_saved_into_retention():
    save = inspect.getsource(tr.save_settings)
    assert "_with_prefix" not in save


def test_the_card_opens_on_the_root_the_chain_trains_from(monkeypatch):
    monkeypatch.setattr(tr.pc, "read_config", lambda: {
        "root": "retraining/helmet_test", "retention": {"root": "retraining"}})
    assert tr.get_settings(session=None)["suggested_root"] == "retraining/helmet_test"
