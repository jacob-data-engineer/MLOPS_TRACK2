"""Writing a data.yaml has to reach step 6.

"Use the assembled dataset" stayed greyed out, reading "Nothing
assembled yet. Pick folders in Assemble a dataset above and press Write
data.yaml", beside a data.yaml that had just been written.

TWO reasons, and fixing either alone leaves it broken:

1. The option is filled from `dataset_assemble.built` in the config, and
   that config read ran once on mount. Nothing re-read it when the card
   wrote, so the answer stayed as it was when the page loaded.

2. The card's own writer is debounced 600ms. So even with a re-read,
   the read lands BEFORE the value it is looking for.

Nothing on screen said "reload the page", which is what it wanted.
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
    return PAGE.read_text(encoding="utf-8")


# --- 1. step 6 re-reads ------------------------------------------------------------

def test_the_config_read_reruns_when_something_is_assembled():
    code = _page()
    assert "}, [detectorsVersion, assembledVersion]);" in code


def test_the_form_takes_the_refresh_key():
    code = _page()
    assert "assembledVersion: number;" in code
    assert "assembledVersion={assembledVersion}" in code


def test_the_prop_is_destructured_not_only_typed():
    """Declaring it in the type only is not taking it -- `tsc` says
    "Cannot find name", seven minutes into a CodeBuild."""
    code = _page()
    head = code.split("function TrainingForm({")[1]
    head = head[:head.index("}: {")]
    assert "assembledVersion" in head


def test_both_copies_of_the_card_bump_it():
    """Standard and multi tabs render the same component. Wiring one is
    how the two tabs drift."""
    assert _page().count("setAssembledVersion((n) => n + 1)") == 2


def test_the_counter_exists_on_the_page():
    assert "const [assembledVersion, setAssembledVersion] = useState(0);" in _page()


# --- 2. the value is there before the read ------------------------------------------

def test_the_write_persists_built_before_telling_the_page():
    """Awaited, and BEFORE `onAssembled`. The order is the whole fix:
    the page re-reads the config the moment that callback fires."""
    code = _panel()
    body = code.split("const out = await writeTrack2DatasetYaml(")[1][:2200]
    assert "await saveTrack2PullConfig({" in body
    assert body.index("await saveTrack2PullConfig({") < body.index("onAssembled?.(")


def test_it_writes_the_record_it_just_made():
    code = _panel()
    body = code.split("const out = await writeTrack2DatasetYaml(")[1][:2200]
    assert "built: record," in body


def test_the_immediate_write_carries_the_whole_card():
    """`write_config` merges at the TOP level, so `dataset_assemble` is
    replaced wholesale. Sending only `built` would drop every folder,
    the Model Name and the scheduling tick."""
    code = _panel()
    body = code.split("const out = await writeTrack2DatasetYaml(")[1][:2200]
    for key in ("train_sources", "eval_sources", "folders", "detector_slug",
                "overwrite", "only"):
        assert key in body, key


def test_a_failed_save_does_not_lose_the_write():
    """The yaml is already in S3 by this point. A config save that fails
    must not turn a successful write into an error."""
    code = _panel()
    body = code.split("const out = await writeTrack2DatasetYaml(")[1][:2200]
    save = body[body.index("await saveTrack2PullConfig({"):]
    assert ".catch(() => undefined);" in save[:900]


# --- what must not have broken ------------------------------------------------------

def test_the_debounced_writer_is_still_there():
    """It is what saves every other change on the card. The immediate
    write covers one moment, not the card."""
    code = _panel()
    assert "setTimeout(() => {" in code
    assert code.count("saveTrack2PullConfig({") >= 2


def test_the_selection_still_moves_to_the_assembled_folder():
    code = _page()
    assert "setSelection({ datasetFolder: folder, evalFolder: folder })" in code


def test_the_eval_folder_still_travels_separately_on_the_multi_tab():
    """A SEPARATE download -- EVAL_DATA_S3_URI, read by a different
    container. Falling back to the training folder would evaluate on the
    images it trained on and report that as held out."""
    code = " ".join(_page().split())
    assert "datasetFolder: folder, evalFolder: evalFolder || folder," in code
