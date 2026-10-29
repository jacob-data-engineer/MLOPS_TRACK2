"""Add to Queue generates the data.yaml itself (2026-10-26).

With "Use the assembled dataset" chosen, the training card's Add to Queue
runs the chain -- rebuild the yaml from the Assemble card's saved
settings, freeze, queue -- instead of queueing whatever folder the
Generate button last wrote. That is also what carries the card's filter
tick to the run. `enqueue` queues without releasing the queue, which is
what Add to Queue always meant; a composed run keeps `queue` (release).
"""
import inspect
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api.routers import track2_retention as tr  # noqa: E402

PAGE = (ROOT / "web" / "src" / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
CARD = (ROOT / "web" / "src" / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")


def test_the_chain_can_queue_without_starting():
    code = inspect.getsource(rs._run_chain_body)
    assert 'if submit in ("queue", "enqueue"):' in code
    assert 'if submit == "enqueue":' in code


def test_the_route_accepts_enqueue():
    assert '("start", "queue", "enqueue")' in inspect.getsource(tr.run_retrain)


def test_add_to_queue_runs_the_chain_for_the_assembled_dataset():
    flat = " ".join(PAGE.split())
    assert 'if (dataSource === "assembled") { const got = await runTrack2Retrain(composed ? "queue" : "enqueue");' in flat
    # Not blocked on a folder the Generate button wrote earlier.
    assert '(!datasetFolder && dataSource !== "assembled")' in PAGE


def test_the_build_card_is_a_preview_with_no_button():
    """2026-10-26: the Generate button went. The run builds the yaml, so
    the card shows what it will be and writes nothing."""
    flat = " ".join(CARD.split())
    assert "Generate yaml training file" not in flat
    assert "Preview only -- nothing is written." in flat
    assert "writeTrack2DatasetYaml" not in flat


def test_enqueue_does_not_start_the_queue():
    """The enqueue branch is decided BEFORE the queue is released."""
    src = inspect.getsource(rs._run_chain_body)
    assert src.index('if submit == "enqueue":') < src.index("training.queue_start()")
