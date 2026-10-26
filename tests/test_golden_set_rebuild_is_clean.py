"""A golden-set rebuild makes the target MATCH the plan, not grow.

`build` copied and never cleared, and the target is derived from a split
that changes. Re-running an 80/20 build over a folder left by a 5/95 one
produced, on this very bucket:

    images/train   5  ->  80      correct
    images/val    95  ->  95      the old 95 stayed; 20 were overwritten

The 80 training stems are a superset of the old 5, so 75 of the 100
golden images sat in BOTH sides -- trained on, then scored on. The
held-out number would have been inflated by an unmeasured amount and
nothing on the run would have said so. A golden set exists precisely to
be the honest number.

The same clear-out removes stems dropped from the source since the last
build, which would otherwise go on being trained on forever.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import golden_set as gs  # noqa: E402


def _src() -> str:
    return inspect.getsource(gs.build)


def _flat() -> str:
    return " ".join(_src().split())


def test_the_destinations_are_worked_out_before_anything_is_copied():
    """They are also the list of what may stay, so the two cannot
    disagree about which files belong."""
    code = _flat()
    assert "wanted: dict = {}" in code
    assert code.index("wanted: dict = {}") < code.index("delete_objects")


def test_anything_not_in_the_plan_is_removed():
    code = _flat()
    assert "if key.endswith(\"/\") or key in wanted: continue" in code
    assert "pc.s3.delete_objects(" in code


def test_the_clear_out_happens_before_the_copy():
    """Otherwise a freshly copied file could be deleted again."""
    code = _flat()
    assert code.index("delete_objects") < code.index("copy_object")


def test_it_only_touches_the_targets_own_images_and_labels():
    """A key outside those two subfolders is not ours to remove."""
    code = _flat()
    assert 'prefix = f"{target_folder}/{sub}/"' in code
    assert "for sub in {images_subpath.strip(\"/\"), labels_subpath.strip(\"/\")}" in code


def test_the_source_is_still_never_touched():
    """It is the thing somebody curated by hand, and the reason this
    copies rather than moves."""
    code = _flat()
    assert 'Bucket=target_bucket' in code
    assert 'Delete={"Objects": batch' in code
    # The only bucket deleted from is the TARGET.
    assert "delete_objects(Bucket=found" not in code


def test_deletes_are_batched_to_the_api_limit():
    assert "range(0, len(stale), 1000)" in _flat()


def test_a_failed_delete_is_reported_rather_than_raised():
    """A run that stopped here would leave the target half-reconciled,
    which is worse than one that says what it could not remove."""
    code = _flat()
    assert 'errors.append({"key": batch[0]["Key"], "error": str(exc)})' in code


def test_the_count_is_returned_so_the_step_can_say_it():
    assert '"removed": removed' in _flat()


def test_the_reason_is_written_down_where_the_code_is():
    """This one is invisible in the output -- a leaking split still
    trains and still reports a number -- so the comment is the only
    thing that stops it being 'simplified' back."""
    code = _src()
    assert "trained on, and then scored on" in code
    assert "5/95" in code
