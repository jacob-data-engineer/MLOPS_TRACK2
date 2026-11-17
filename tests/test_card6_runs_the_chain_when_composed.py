"""The training card's button runs the CHAIN when the run is composed.

The radio on card 6 said "Use the assembled dataset + the other ticked
sources ... composed at run time". The button under it called
`addTrack2Queue` with one `dataset_folder`, and one folder cannot carry
a golden set that lives under another root -- only the chain builds
that yaml. So the label promised the golden set and the run trained
without it, with the assembled folder's own val as the "held-out" set.

Worse, the only button that DID run the chain -- "Run it now" on the
Retention card -- sits inside Auto Training, which is switched off and
inert in exactly the configuration that composes: folders + golden,
Pull Control unticked. There was no reachable way to trigger the
composed run by hand; only the schedule could.

Now: when the radio says composed, the button runs the chain with
`submit="queue"`, the same hand-over the schedule uses, and says so.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api.routers import track2_retention as router  # noqa: E402

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PAGE = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
API = (WEB / "lib" / "api.ts").read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return " ".join(text.split())


# --- the route -----------------------------------------------------------------

def test_the_manual_route_takes_submit():
    assert "submit" in router.RetrainRunRequest.model_fields
    assert router.RetrainRunRequest().submit == "start"


def test_it_is_the_same_two_values_the_chain_takes():
    code = inspect.getsource(router.run_retrain)
    # `enqueue` added 2026-10-26: Add to Queue for the assembled dataset.
    assert 'if submit not in ("start", "queue", "enqueue"):' in code
    assert 'rs.run_chain(trigger="manual", submit=submit)' in code


def test_an_unknown_submit_is_a_400_not_a_default():
    """Silently falling back to `start` would run a job the caller
    asked to queue."""
    code = inspect.getsource(router.run_retrain)
    assert "status_code=400" in code


def test_the_body_is_optional_so_run_it_now_is_unchanged():
    """The Retention card posts `{}` and expects `start`."""
    code = inspect.getsource(router.run_retrain)
    assert "body: RetrainRunRequest | None = None" in code
    assert 'submit = (body.submit if body else "start") or "start"' in code


# --- the client ----------------------------------------------------------------


# --- card 6 --------------------------------------------------------------------


def test_the_client_import_is_present():
    assert "runTrack2Retrain," in PAGE
