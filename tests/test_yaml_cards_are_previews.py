"""The data.yaml cards show what the run will build; they build nothing.

Asked for on 2026-10-26. The run builds its data.yaml at Add to Queue and
on the schedule -- the retention lists first when the policy is on -- so a
second build-and-save on the cards was a file that could disagree with
what trained. And Pull Control saves as it goes, so its Apply settings
button only re-sent what had already been sent.
"""
import pathlib

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
RETENTION = (WEB / "components" / "Track2RetentionPanel.tsx").read_text(encoding="utf-8")
PULL = (WEB / "components" / "Track2PullControlPanel.tsx").read_text(encoding="utf-8")
FLAT_RETENTION = " ".join(RETENTION.split())


# --- Retention: data.yaml for training -------------------------------------------

def test_the_retention_card_has_no_build_or_save():
    # Code, not words: a comment says what used to be here.
    for gone in ('withBusy("Reading..."', 'withBusy("Writing..."', "saveTrack2DatasetYaml",
                 "setApplyRetention", "onUseForTraining", "setClassNames"):
        assert gone not in RETENTION, gone


def test_it_previews_through_the_route_that_writes_nothing():
    assert "buildTrack2DatasetYaml({ root, apply_retention: enabled })" in FLAT_RETENTION


def test_the_preview_follows_the_policy_switch_the_chain_reads():
    """`enabled` is the retention policy's own switch -- the one the chain
    reads -- not a second tick that only this card saw."""
    assert "}, [yamlOpen, root, enabled, policyJson]);" in RETENTION


def test_it_says_retention_is_applied_first():
    assert "The retention policy is on, so it is applied first:" in FLAT_RETENTION
    assert "Preview only -- nothing is written." in FLAT_RETENTION


def test_it_draws_a_tree_like_the_assemble_card():
    assert "function yamlTree(built: Track2DatasetYaml, root: string): string {" in RETENTION
    assert "{yamlTree(built, root)}" in RETENTION


# --- Pull control: Apply settings --------------------------------------------------

def test_apply_settings_is_gone():
    assert "Apply settings{dirty" not in PULL
    assert 'withBusy("Applying...",' not in PULL


def test_a_landed_save_clears_the_pending_mark():
    """Apply was the only thing clearing `dirty`, which the toolbar reads
    to say a card's changes did not reach a settings file."""
    flat = " ".join(PULL.split())
    assert "if (latestConfigRef.current === sent) setDirty(false);" in flat
    assert "latestConfigRef.current = configJson;" in flat
