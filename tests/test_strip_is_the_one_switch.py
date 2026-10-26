"""The PULL, BUILD, QUEUE, TRAIN strip is the one switch for unattended runs.

Asked for on 2026-10-26: the Retention card's "Do it automatically"
section goes, and a scheduled (or arrival) pull builds, queues and starts
a run by itself, from the ticked sources.
"""
import pathlib

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
RETENTION = (WEB / "components" / "Track2RetentionPanel.tsx").read_text(encoding="utf-8")
STRIP = " ".join((WEB / "components" / "Track2PinnedBar.tsx").read_text(encoding="utf-8").split())


def test_the_retention_section_is_gone():
    for gone in ("Run it now", "Which pulls chain", "Run after every scheduled pull",
                 "saveTrack2QueueAfterPull", "runTrack2Retrain"):
        assert gone not in RETENTION, gone


def test_the_strip_says_what_a_scheduled_pull_does():
    assert "builds, queues and trains after each scheduled pull, from the ticked sources" in STRIP


def test_arrival_is_never_offered_without_its_guards():
    """CLAUDE.md: arrival needs both guards before it is ticked."""
    assert "saveTrack2QueueAfterPull({ min_files: value })" in STRIP
    assert "saveTrack2QueueAfterPull({ cooldown_minutes: value })" in STRIP
    assert "every upload will start a GPU run" in STRIP


def test_no_step_4b_tick_is_pointed_at_any_more():
    assert "under <em>Run after every pull</em> in step 4b" not in STRIP


def test_the_queue_card_lists_folders_one_per_line_and_scrolls():
    """2026-10-26: the Training images line truncated after two folders
    once the week folders joined it."""
    page = " ".join((WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8").split())
    # ONE box for both sides, a count beside each folder, "N folders".
    assert "function FolderBox(" in page and "FolderList" not in page
    assert page.count("<FolderBox") == 1
    assert "max-h-[7.5rem] overflow-auto" in page
    assert 'plural(side.folders.length, "folder", "folders")' in page
    assert 'plural(side.counts[folder], "image", "images")' in page


def test_the_metrics_export_can_make_a_new_folder():
    """2026-10-26: Review & metrics' S3 destination offers New folder."""
    card = " ".join((WEB / "components" / "Track2MetricsExportCard.tsx").read_text(encoding="utf-8").split())
    assert "onNavigate={(next) => set({ prefix: next })} allowCreate />" in card
