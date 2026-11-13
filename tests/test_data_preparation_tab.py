"""The Data Preparation tab (2026-10-28): the screen half of the
data/training split.

Data cards on their own tab, the data.yaml preview kept on Training
beside the button that trains on it, and the notice board carrying "the
sources changed" from one to the other.
"""

import pathlib

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
DATA = (WEB / "routes" / "Track2DataPrepPage.tsx").read_text(encoding="utf-8")
MODELS = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
PANEL = (WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")
BUS = (WEB / "lib" / "bus.ts").read_text(encoding="utf-8")

DATA_CARDS = ("<Track2GoldenSetPanel", "<Track2PullControlPanel",
              "<Track2AutoTrainingPanel")


def test_the_data_cards_are_on_the_data_tab_in_working_order():
    at = [DATA.index(card) for card in ("<Track2DatasetAssemblePanel", *DATA_CARDS)]
    assert at == sorted(at), "Assemble, Golden, Pull Control, weeks"
    assert "Track2PullQueuePanel" not in DATA


def test_and_only_there():
    for card in DATA_CARDS:
        assert card not in MODELS, f"{card} is still on the Training tab"


def test_the_tab_is_routed_and_offered():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    nav = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
    assert 'path="/track2/data-preparation" element={<Track2DataPrepPage />}' in app
    assert nav.index('"/track2/data-preparation"') < nav.index('"/track2/multi-auto-training"')


def test_the_preview_stays_on_training_and_is_the_assemble_cards_own():
    """Not a second implementation that could disagree with the run."""
    assert "<Track2DatasetAssemblePanel multiSource modelSlug={modelChoice.slug} previewOnly />" in MODELS


def test_the_classes_are_set_on_multi_auto_training_not_data_preparation():
    """Moved 2026-11-13: which labelled classes a run trains, and whether
    undeclared boxes are dropped, are training decisions -- set beside
    Add to Queue. Data Preparation picks folders only."""
    assert "{!buildSlot && !onOnlyChange && buildSection}" in PANEL
    preview = PANEL[PANEL.index("const buildSection = ("):]
    assert "readOnly={previewOnly}" not in preview and "disabled={previewOnly}" not in preview
    assert "onNameTheRest={nameTheRest}" in PANEL


def test_each_tab_writes_only_its_own_fields_onto_the_saved_block():
    """`write_config` replaces `dataset_assemble` whole, and the two tabs
    now edit different parts of it. Each starts from what is saved, or the
    last one to save would put back a stale copy of the other's fields."""
    folders = PANEL[PANEL.index("// One writer for the whole card"):PANEL.index("// The class fields' writer")]
    assert "if (!settingsRead || previewOnly) return;" in folders
    assert "...saved," in folders and "...(onOnlyChange ? {} : {" in folders
    classes = PANEL[PANEL.index("// The class fields' writer"):PANEL.index("// Check each filled pair")]
    assert "if (!settingsRead || !previewOnly || !classesTouched.current) return;" in classes
    assert "...saved," in classes
    assert "train_sources" not in classes and "folders" not in classes.split("dataset_assemble")[1]


def test_every_data_card_that_saves_says_so():
    for name, sender in (("Track2DatasetAssemblePanel", '"assemble"'),
                         ("Track2AutoTrainingPanel", '"weeks"'),
                         ("Track2RetentionPanel", '"retention"')):
        code = (WEB / "components" / f"{name}.tsx").read_text(encoding="utf-8")
        assert f'post("data-sources-changed", {{ from: {sender} }})' in code, name
    golden = (WEB / "lib" / "useGoldenSync.ts").read_text(encoding="utf-8")
    assert 'post("data-sources-changed", { from: "golden-set" })' in golden


def test_training_listens_and_rereads_only_the_ticks():
    at = MODELS.index('listen("data-sources-changed"')
    block = MODELS[at:at + 600]
    assert "getTrack2PullControl()" in block and "setSources({" in block


def test_the_board_reaches_other_windows_too():
    """The Data tab in one window and Training in another is the split's
    common case; a window event alone would not cross between them."""
    assert "BroadcastChannel" in BUS
    assert '"golden-changed": boolean;' in BUS
    assert '"data-sources-changed": { from: string };' in BUS
