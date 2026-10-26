"""The queue says which Model Name each run trains as.

`detector_slug` and `architecture` have always been stored on a queue
entry server-side. They were never declared client-side, so nothing
could show them -- and a run's identity was invisible while it ran.

It matters because the two halves of a run come from different places:

    registry group  <-  the Model Name chosen in step 1
    class names     <-  the dataset's own data.yaml

Nothing forces those to agree. A run queued against the wrong Model Name
trains happily, reports metrics, and files them under a taxonomy that
does not describe it. Reading the queue should be enough to catch that
before the GPU is spent.

Seen live on 2026-10-15: a run executing as `fakecad-it` (6 classes,
matching its dataset) with the NEXT entry queued as `helmet` (1 class)
against the same six-class dataset. Nothing on screen said so.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
API = (pathlib.Path(__file__).resolve().parents[1]
       / "web" / "src" / "lib" / "api.ts")


def _block() -> str:
    """The queue entry component, which renders both states."""
    code = PAGE.read_text(encoding="utf-8")
    start = code.index("function QueueEntryBlock(")
    return code[start:code.index("\nfunction ", start + 10)]


# --- the server already had it -------------------------------------------------------

def test_the_entry_is_stored_with_its_model_name():
    from ppe_pipeline_api.routers import track2_models as m

    code = inspect.getsource(m.add_to_queue)
    assert '"detector_slug": payload.detector_slug,' in code
    assert '"architecture": payload.architecture,' in code


def test_the_chain_stores_it_too():
    """Both ways of queueing, or the line is blank for half of them."""
    from ppe_pipeline_api import retrain_schedule as rs

    code = inspect.getsource(rs._run_chain_body)
    assert "detector_slug" in code and "architecture" in code


# --- the client can see it now --------------------------------------------------------

def test_the_fields_are_declared_client_side():
    """TypeScript drops nothing, but an undeclared field cannot be
    referenced -- which is why this was invisible rather than wrong."""
    code = API.read_text(encoding="utf-8")
    entry = code.split("export interface Track2QueueEntry {")[1]
    entry = entry[:entry.index("\n}")]
    assert "detector_slug?: string;" in entry
    assert "architecture?: string;" in entry


# --- and shows it ---------------------------------------------------------------------

def test_the_block_shows_the_model_name():
    assert "Model: <span" in _block()
    assert "{entry.detector_slug}" in _block()


def test_it_shows_the_architecture_beside_it():
    assert "{entry.architecture}" in _block()


def test_it_spans_the_row_rather_than_sitting_in_a_column():
    """It is the run's identity, not one of its numbers."""
    code = " ".join(_block().split())
    assert 'Model: <span' in code
    model_at = code.index("Model: <span")
    assert 'col-span-2 truncate sm:col-span-3' in code[max(0, model_at - 300):model_at]


def test_it_comes_before_the_hyperparameters():
    code = _block()
    assert code.index("Model: <span") < code.index("Epochs: {entry.epochs}")


def test_the_registry_group_is_in_the_tooltip():
    """The consequential part: that name is an exact-match list on the
    rule that swaps production weights."""
    code = " ".join(_block().split())
    assert "-detector`}" in code


def test_nothing_is_rendered_when_there_is_no_model_name():
    """Production-pipeline entries have no detector. A line reading
    "Model: ." says less than no line."""
    assert "{entry.detector_slug && (" in _block()


# --- both states, one component --------------------------------------------------------

def test_one_component_renders_running_and_queued():
    """Two renderers would be two chances for one of them to be missing
    the line."""
    code = PAGE.read_text(encoding="utf-8")
    assert code.count("<QueueEntryBlock") == 2
    assert code.count("function QueueEntryBlock(") == 1


def test_the_running_state_is_the_same_block():
    code = PAGE.read_text(encoding="utf-8")
    assert 'state="running"' in code or 'state={"running"}' in code


# --- what must not have broken ----------------------------------------------------------

def test_every_field_the_block_showed_before_is_still_there():
    code = _block()
    for field in ("Epochs:", "Image size:", "Batch:", "Seed:"):
        assert field in code, field
    # Both folder lists, now one box headed Training and Evaluation.
    assert 'title: "Training"' in code and 'title: "Evaluation"' in code


def test_the_snapshot_line_survives():
    code = _block()
    assert "snapshot_id" in code
