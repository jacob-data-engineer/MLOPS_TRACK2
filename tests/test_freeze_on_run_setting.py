"""`freeze_on_run`: whether every run freezes a snapshot before training.

On by default, which is the behaviour before the tick existed. Off, the
chain and a queue-add skip the freeze; the run trains on the same images
and simply leaves no snapshot. The manual Freeze button is unaffected.
"""
import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api.routers import track2_models as tm  # noqa: E402
from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest  # noqa: E402

PANEL = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "components"
         / "Track2RetentionPanel.tsx").read_text(encoding="utf-8")


def test_it_is_a_saved_setting_defaulting_on():
    assert pc.DEFAULTS["freeze_on_run"] is True
    assert ConfigRequest(freeze_on_run=False).model_dump(exclude_none=True) == {"freeze_on_run": False}


def test_the_chain_skips_the_freeze_when_off():
    code = inspect.getsource(rs._run_chain_body)
    assert 'elif not config.get("freeze_on_run", True):' in code
    assert 'snapshot = {"snapshot_id": "", "file_count": None}' in code
    # Nothing downstream announces a snapshot that was never taken.
    assert 'f"{trigger} retrain (not frozen)"' in code


def test_a_queue_add_skips_it_too(monkeypatch):
    called = []
    monkeypatch.setattr(pc, "read_config", lambda: {"freeze_on_run": False})
    import ppe_pipeline_api.dataset_snapshots as ds
    monkeypatch.setattr(ds, "create", lambda **kw: called.append(kw) or {})
    tm._freeze_for_entry({"id": "x", "dataset_folder": "retraining"})
    assert called == []


def test_the_card_has_the_tick_and_keeps_the_button():
    assert "Freeze the dataset on every run" in PANEL
    assert "saveTrack2PullConfig({ freeze_on_run: on })" in PANEL
    assert "Freeze this dataset" in PANEL
