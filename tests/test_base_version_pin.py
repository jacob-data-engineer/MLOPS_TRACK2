"""A retrain continues from a version somebody chose, not always the newest.

The newest version is the right DEFAULT -- a retrain is "the next version
of this Class", so walking the lineage forward is what the chain is for.
It is the wrong rule when the newest is not the best: a run that overfit,
or one trained on a week that turned out to be mislabelled, becomes the
parent of everything after it, and the only way back is to start again
from a known-good version.
"""
from __future__ import annotations

import importlib

import pytest

rs = importlib.import_module("src.ppe_pipeline_api.retrain_schedule")

GROUP = "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-helmet-detector"
PACKAGES = [
    {"ModelPackageArn": f"{GROUP}/6", "ModelApprovalStatus": "Rejected"},
    {"ModelPackageArn": f"{GROUP}/5", "ModelApprovalStatus": "PendingManualApproval"},
    {"ModelPackageArn": f"{GROUP}/4", "ModelApprovalStatus": "Approved"},
]


def test_a_bare_version_number_finds_it():
    assert rs.find_pinned_version(PACKAGES, "4")["ModelPackageArn"] == f"{GROUP}/4"


def test_a_v_prefix_is_tolerated():
    """The card shows 'v4'; somebody typing what they see is not wrong."""
    assert rs.find_pinned_version(PACKAGES, "v4")["ModelPackageArn"] == f"{GROUP}/4"


def test_a_full_arn_finds_it():
    """What the dropdown actually stores."""
    assert rs.find_pinned_version(PACKAGES, f"{GROUP}/5")["ModelPackageArn"] == f"{GROUP}/5"


def test_an_unknown_version_is_None_not_the_newest():
    """The whole point. Falling back to newest would produce exactly the
    model the pin exists to prevent, and nothing would say so."""
    assert rs.find_pinned_version(PACKAGES, "99") is None


def test_blank_is_None_so_the_caller_takes_the_default_path():
    assert rs.find_pinned_version(PACKAGES, "") is None
    assert rs.find_pinned_version(PACKAGES, "   ") is None


def test_an_unapproved_version_can_be_pinned():
    """Approval governs the DEFAULT. An explicit pin is a person saying
    they know; a Class's early versions are routinely unapproved and
    refusing them would make the second run impossible."""
    assert rs.find_pinned_version(PACKAGES, "5")["ModelApprovalStatus"] == "PendingManualApproval"


def test_the_default_is_still_newest_approved():
    """Unpinned behaviour must not change: v6 is newer but Rejected."""
    chosen = next((p for p in PACKAGES if p.get("ModelApprovalStatus") == "Approved"), None)
    assert chosen["ModelPackageArn"].endswith("/4")


def test_the_setting_exists_and_defaults_to_empty():
    assert rs.DEFAULTS["base_version"] == ""


def test_the_chain_refuses_a_pin_it_cannot_find(monkeypatch):
    """Recorded as a FAILED base step that names what is available --
    not a silent fallback, and not a bare traceback."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert "find_pinned_version" in source
    # The refusal returns rather than carrying on to submit.
    pinned_block = source.split("pinned = ")[1].split("# 3. submit")[0]
    assert "is not in" in pinned_block
    assert '"ok": False' in pinned_block


@pytest.mark.parametrize("missing", [None, []])
def test_no_packages_at_all_is_handled(missing):
    assert rs.find_pinned_version(missing, "4") is None


def test_scratch_is_a_valid_pin():
    """A brand-new Class has nothing in its registry, so the chain
    stopped at the base step and its FIRST unattended run could never
    happen -- "create a Class, schedule it, wait" was a dead end."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert 'pinned.lower() == "scratch"' in source
    # It must set base_model_scratch on the queued entry, not just skip
    # the lookup -- the queue runner refuses an entry with no source at
    # all, and that refusal happens one step past where the chain looks.
    assert '"base_model_scratch": scratch' in source


def test_the_version_comes_from_step_1_not_from_this_card():
    """There was briefly a second "Continue from" dropdown on the
    retention card. Two controls for one decision, and they disagreed in
    production: step 1 said version 2 while the chain took the newest
    Approved one, and the queued entry carried step 1's note -- "This
    trains from version 2" -- over a run that started from v7."""
    import inspect
    source = inspect.getsource(rs.settings)
    assert 'config.get("default_version")' in source
    assert 'default_source' in source
    # And the card can no longer set it.
    from ppe_pipeline_api.routers import track2_retention as router
    assert "base_version" not in router.RetrainScheduleRequest.model_fields


def test_training_from_scratch_in_step_1_reaches_the_chain():
    """Step 1 expresses it as a SOURCE rather than a version, and it has
    to land on the same field or "from scratch" would be silently
    ignored by the unattended path."""
    plan = rs.settings({"default_source": "scratch",
                        "default_version": "arn:.../yolo11s-helmet-detector/4"})
    assert plan["base_version"] == "scratch"


def test_step_1s_version_wins_over_anything_stored_on_the_schedule():
    """A settings file written while the old dropdown existed still
    carries a `base_version`. It must not outrank the live choice."""
    plan = rs.settings({
        "default_version": "arn:.../yolo11s-helmet-detector/2",
        "retrain_schedule": {"base_version": "arn:.../yolo11s-helmet-detector/9"},
    })
    assert plan["base_version"].endswith("/2")


def test_no_choice_in_step_1_means_the_newest_approved():
    """Blank is still "walk the lineage forward", which is what the
    chain did before any of this existed."""
    assert rs.settings({})["base_version"] == ""


def test_the_empty_registry_message_offers_the_way_out():
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert "Train from scratch" in source
