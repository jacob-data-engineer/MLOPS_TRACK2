"""The history row names the Class, not "YOLOcustom-YOLO11S".

For a custom-class run `generation` is the literal string "custom" and
`size` holds the architecture, so

    `YOLO${audit.generation}-${audit.size.toUpperCase()}`

rendered **YOLOcustom-YOLO11S** -- a label naming neither the model nor
anything a person chose. Which Class a run belongs to is what the
history is scanned for, and it was only in the expanded detail as
"Model Name: helmet".

Production runs keep the old form. They have no detector, and their
generation and size really are 11 and s.
"""
from __future__ import annotations

import pathlib

PANEL = (pathlib.Path(__file__).resolve().parents[1]
         / "web" / "src" / "components" / "Track2TrainingAuditPanel.tsx")
API = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "lib" / "api.ts")


def _code() -> str:
    return PANEL.read_text(encoding="utf-8")


def _flat() -> str:
    return " ".join(_code().split())


# --- the custom case ------------------------------------------------------------------

def test_the_row_shows_the_model_name():
    code = _flat()
    assert "{audit.detector_slug ? (" in code
    assert "{audit.detector_slug}" in code


def test_the_architecture_rides_beside_it():
    """`size` IS the architecture on a custom run -- "yolo11s", not
    "s" -- so it is shown as itself rather than folded into a template
    that assumes otherwise."""
    code = _flat()
    assert "&middot; {audit.size}" in code


def test_the_architecture_is_muted_not_equal_weight():
    """The Class is what the row is scanned for; the architecture is
    context."""
    code = _flat()
    assert '<span className="text-ink-muted"> &middot; {audit.size}</span>' in code


def test_the_old_label_is_gone_for_runs_that_have_a_detector():
    """It is still the fallback, but no longer the only branch."""
    code = _flat()
    at = code.index("{audit.detector_slug ? (")
    assert "YOLO${audit.generation}" in code[at:at + 500], (
        "the production fallback was removed with it"
    )


# --- the production case is untouched ---------------------------------------------------

def test_a_run_with_no_detector_keeps_the_generation_and_size():
    code = _flat()
    assert "`YOLO${audit.generation}-${audit.size.toUpperCase()}`" in code


def test_the_fallback_is_the_else_branch():
    code = _flat()
    detector_at = code.index("{audit.detector_slug ? (")
    fallback_at = code.index("`YOLO${audit.generation}", detector_at)
    assert ") : (" in code[detector_at:fallback_at]


# --- the field it reads -------------------------------------------------------------------

def test_the_audit_type_declares_the_detector():
    api = API.read_text(encoding="utf-8")
    block = api.split("export interface Track2TrainingAudit {")[1]
    block = block[:block.index("\n}")]
    assert "detector_slug?: string;" in block


def test_the_server_writes_it_on_every_audit():
    import inspect
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
    from ppe_pipeline_api import track2_shared as t2

    assert "detector_slug" in inspect.getsource(t2.write_training_audit)


# --- what must not have broken --------------------------------------------------------------

def test_the_rest_of_the_row_survives():
    code = _flat()
    assert "<Badge tone={statusTone(audit.status)} label={audit.status} />" in code
    assert "{formatTime(audit.submitted_at)}" in code
    assert "v{audit.model.version}" in code


def test_the_duration_added_earlier_is_still_on_the_row():
    code = _flat()
    assert "formatDuration(audit.started_at, audit.ended_at," in code


def test_the_expanded_detail_still_names_the_model():
    """Two places, one fact -- the row is for scanning, the detail is
    for reading."""
    code = _flat()
    assert 'label="Model Name"' in code
