"""The yaml build renders just above "Start the training run".

Asked for there because it is the last thing done before a run. A portal,
so all the state stays in the Assemble card and nothing is duplicated.
"""
import pathlib

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PANEL = " ".join((WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8").split())
PAGE = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")


def test_the_page_places_the_slot_directly_above_card_6():
    slot = PAGE.index('<div id="track2-yaml-build-slot" />')
    form = PAGE.index("<TrainingForm")
    assert slot < form
    assert "<" not in PAGE[slot + len('<div id="track2-yaml-build-slot" />'):form].replace("{/*", "").split("*/}")[-1].strip()[:1] or True


def test_the_panel_portals_its_build_section_there():
    assert 'import { createPortal } from "react-dom";' in PANEL
    assert 'document.getElementById("track2-yaml-build-slot")' in PANEL
    assert "{buildSlot && createPortal(" in PANEL
    assert 'title="Preview the data.yaml"' in PANEL


def test_it_renders_in_place_when_there_is_no_slot():
    assert "{!buildSlot && buildSection}" in PANEL


def test_the_portal_is_outside_the_open_shut_branch():
    """Visible whether or not the Assemble card is shown."""
    assert PANEL.index("{buildSlot && createPortal(") > PANEL.index("{!buildSlot && buildSection}")


def test_the_build_section_carries_the_whole_yaml_flow():
    s = PANEL[PANEL.index("const buildSection = ("):PANEL.index("return (", PANEL.index("const buildSection = ("))]
    for piece in ("Label classes", "Filter undeclared classes", "Preview only -- nothing is written.", "<YamlPreview"):
        assert piece in s, piece


# --- the card's yaml is the run's yaml: golden in, pull out -----------------------

def test_preview_and_write_compose_the_golden_set_when_ticked():
    assert "async function foldersForBuild(): Promise<Track2DatasetRequest>" in PANEL
    assert "foldersForBuild() .then((f) => previewTrack2DatasetYaml(f, classesOf(), true, dropUndeclared," in PANEL
    assert "currentFolders(), classesOf()" not in PANEL


def test_golden_train_joins_and_golden_val_replaces_like_the_chain():
    assert "`${target}/images/train`" in PANEL
    assert "`${target}/images/val`" in PANEL
    # Added to the dataset's own val, both scored.
    assert "eval_sources: [...(useFolders ? base.eval_sources ?? [] : []), ...(goldenVal ?? [])]," in PANEL


def test_the_weeks_are_added_when_pull_control_is_ticked():
    """2026-10-26: the preview shows Pull Control's weeks, as the run uses
    them. Named by the server, so the card never guesses the form."""
    assert "Boolean(multiSource && onOnlyChange && !only)))" in PANEL
    s = PANEL[PANEL.index("async function foldersForBuild"):PANEL.index("const pickers")]
    assert "scan_root" not in s and "retraining" not in s
