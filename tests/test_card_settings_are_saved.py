"""Every card's settings reach the settings file.

The settings file is a faithful snapshot of the live config, so a card's
settings are saved if and only if the card writes them into that config.
Two cards shipped without doing so -- Assemble and Import Checkpoint --
and the symptom was the worst kind: the save reported success, because
the save was honest. It copied a config the card had never written to.

These tests pin the two ways that happens.
"""
from __future__ import annotations

import pathlib
import re

import pytest

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
COMPONENTS = WEB / "components"


def _code(path: pathlib.Path) -> str:
    """The file with comment lines dropped, so prose cannot look like code."""
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").split("\n")
        if not line.strip().startswith(("//", "*", "/*"))
    )


# --- the silent drop ------------------------------------------------------------

def test_every_key_the_ui_saves_is_declared_on_the_request_model():
    """A key the model does not declare is discarded without a word.

    Pydantic drops undeclared fields, the route still answers 200, and
    the card looks saved until the page is reloaded. This is the failure
    that is impossible to notice from the screen.
    """
    from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest

    declared = set(ConfigRequest.model_fields)
    sent: dict[str, str] = {}
    # .ts as well as .tsx: the card open/closed record is written from
    # lib/cardState.ts, which a .tsx-only sweep would not see.
    files = sorted(list(WEB.rglob("*.tsx")) + list(WEB.rglob("*.ts")))
    for path in files:
        code = _code(path)
        for call in re.finditer(r"saveTrack2PullConfig\(\{", code):
            # Walk to the matching brace and collect only the keys at the
            # TOP level of the object. Indentation cannot decide this:
            # `dataset_assemble: { only: ... }` nests keys of its own, and
            # those are not config fields.
            start = call.end()
            depth, index = 1, start
            while index < len(code) and depth:
                if code[index] == "{":
                    depth += 1
                elif code[index] == "}":
                    depth -= 1
                index += 1
            body, depth = code[start:index - 1], 0
            for token in re.finditer(r"[{}]|([a-z_][a-z0-9_]*)\s*:", body):
                if token.group(0) == "{":
                    depth += 1
                elif token.group(0) == "}":
                    depth -= 1
                elif depth == 0 and token.group(1):
                    sent.setdefault(token.group(1), path.name)

    assert sent, "no saveTrack2PullConfig calls found -- has the API been renamed?"
    missing = {k: v for k, v in sent.items() if k not in declared}
    assert not missing, (
        "these keys are POSTed by the UI but not declared on ConfigRequest, "
        f"so the server throws them away silently: {missing}"
    )


def test_the_request_model_and_the_defaults_agree():
    """A declared key with no default comes back missing on first read."""
    from ppe_pipeline_api.pull_control import DEFAULTS
    from ppe_pipeline_api.routers.track2_pull_control import ConfigRequest

    for key in ("dataset_assemble", "import_checkpoint"):
        assert key in DEFAULTS, f"{key} has no entry in DEFAULTS"
        assert key in ConfigRequest.model_fields, f"{key} is not accepted by /config"


# --- the card that never writes -------------------------------------------------

#: Panels that hold no settings. Each is here for a stated reason, so
#: adding a name is a decision rather than a way to silence the test.
NO_SETTINGS_OF_THEIR_OWN = {
    # Read-only: they render what the server already decided.
    "Track2ChainStatusBar.tsx": "status display",
    "Track2TrainingAuditPanel.tsx": "history, read-only",
    "Track2TrainingCostToday.tsx": "a figure, read-only",
    "Track2ModelProvenance.tsx": "render-only, no state",
    "Track2PullQueuePanel.tsx": "queue contents live server-side already",
    # Controlled components: their value is owned and saved by a parent.
    "Track2TrainingAdvanced.tsx": "value/onChange props; saved by Track2DetectorsPanel",
    "Track2S3FolderBrowser.tsx": "a picker; the chosen path is saved by its parent",
    "Track2BaseModelPicker.tsx": "a picker; the choice is saved by its parent",
    "Track2ModelPicker.tsx": "a picker; the choice is saved by its parent",
    # Act on the settings file rather than holding settings.
    "Track2PinnedBar.tsx": "loads/saves the settings file itself",
    # Saved through their own endpoints, not the pull-control config.
    "Track2ExportSettingsBar.tsx": "saveTrack2ExportSettings",
    "Track2MetricsExportCard.tsx": "arm/disarm + export settings endpoints",
}

#: A panel with at least this many pieces of state is making choices.
_BUSY = 8

_PERSISTS = re.compile(r"\bsave(?:Track2|Trained)\w*\(|\barm\w*\(|\bdisarm\w*\(")


@pytest.mark.parametrize(
    "panel",
    sorted(p.name for p in COMPONENTS.glob("Track2*.tsx")),
)
def test_a_panel_with_settings_saves_them(panel: str):
    """A dozen useState lines and no save call is a card whose settings
    die on reload -- while a settings file saved at the time reports
    success and does not contain them."""
    path = COMPONENTS / panel
    code = _code(path)
    states = len(re.findall(r"\buseState\b", code))
    if states < _BUSY or panel in NO_SETTINGS_OF_THEIR_OWN:
        return
    assert _PERSISTS.search(code), (
        f"{panel} holds {states} pieces of state and never writes any of them "
        "to the server, so they are gone on reload and a settings file saved "
        "while they were on screen does not contain them. Either persist them "
        "(see CLAUDE.md, 'A new card's settings go in the settings file') or "
        "add the file to NO_SETTINGS_OF_THEIR_OWN with the reason."
    )


# --- the card open/closed record ------------------------------------------------

CARD_STATE = WEB / "lib" / "cardState.ts"


def test_the_open_state_setter_takes_the_updater_form():
    """`useCardOpen` replaces `useState` at seven call sites, and every
    one of them is written `setOpen((v) => !v)`.

    A setter typed to take only a boolean compiles everywhere it is
    called with one and fails at all seven of those -- which is how this
    was found, in CodeBuild, seven minutes at a time. tsc is the real
    gate and does not run on this machine, so the shape is pinned here.
    """
    source = CARD_STATE.read_text(encoding="utf-8")
    assert "useCardOpen" in source
    assert "(previous: boolean) => boolean" in source, (
        "useCardOpen's setter must accept the functional updater form, "
        "or every `setOpen((v) => !v)` call site fails to compile"
    )


def test_every_card_using_the_shared_record_imports_it():
    """A `useCardOpen` left without its import is a build failure, and
    the panels were edited by script."""
    for path in sorted(COMPONENTS.glob("Track2*.tsx")):
        code = _code(path)
        if "useCardOpen(" not in code:
            continue
        assert "from \"../lib/cardState\"" in code, f"{path.name} never imports it"


def test_the_cards_that_toggle_open_use_the_saved_record():
    """A card still on a bare `useState` for its open state is one that
    forgets how it was left -- which is the whole point of saving it."""
    stragglers = []
    for path in sorted(COMPONENTS.glob("Track2*.tsx")):
        code = _code(path)
        if "setOpen((" not in code:
            continue
        if "useCardOpen(" not in code:
            stragglers.append(path.name)
    assert not stragglers, (
        f"these cards toggle open/closed but do not save it: {stragglers}"
    )


def test_the_shared_card_container_remembers_whether_it_is_open():
    """Nearly every card collapses through `Card`, not through its own
    `useState` -- "thirty-odd cards across nineteen files", as Card.tsx
    puts it.

    So wiring the handful of panels that run their own toggle left the
    rest forgetting how they were left, which is what was reported: a
    card expanded, the file saved, and the card shut again on reload.
    One `useCardOpen` in `Card` covers all of them.
    """
    card = _code(COMPONENTS / "Card.tsx")
    assert "useCardOpen(" in card, (
        "Card must remember its own open state, or every card that does "
        "not run its own toggle forgets how it was left"
    )
    assert "stateKey" in card, "Card needs a key for cards with no title"


def test_an_unkeyed_card_still_works():
    """`useCardOpen` is called unconditionally -- hooks have to be -- so
    it has to cope with a card that has no title and no key, rather than
    writing every such card into one shared bucket."""
    source = CARD_STATE.read_text(encoding="utf-8")
    assert "if (!key) return [local, setLocal];" in source, (
        "an unkeyed card must fall back to plain local state"
    )
