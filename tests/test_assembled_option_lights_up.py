""""Use the assembled dataset" needs nothing pressed first.

It used to wait for "Generate yaml training file" to write a data.yaml and
record `dataset_assemble.built`, and stayed greyed out reading "Nothing
assembled yet" until then. That button is gone (2026-10-26): the run builds
the yaml at Add to Queue and on the schedule, so the option is ready as
soon as one source is ticked.
"""
from __future__ import annotations

import pathlib

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx")
PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")


def _panel() -> str:
    return PANEL.read_text(encoding="utf-8")


def _page() -> str:
    return " ".join(PAGE.read_text(encoding="utf-8").split())


def test_the_write_path_and_its_refresh_counter_are_gone():
    page, panel = _page(), _panel()
    assert "assembledVersion" not in page
    assert "onAssembled" not in page and "onAssembled" not in panel
    assert "handleWrite" not in panel


def test_the_debounced_writer_is_still_there():
    """It is what saves every change on the card."""
    code = _panel()
    assert "setTimeout(() => {" in code
    # Two since 2026-11-13: the card's writer and the class-fields writer
    # on Multi-Auto Training.
    assert code.count("saveTrack2PullConfig({") == 1
