"""The Retention card no longer lists frozen snapshots or runs.

Decided 2026-10-20: snapshots are loaded and run from the "Use a frozen
snapshot" radio on the run card, and the History card lists what each
run was sent. Two lists of the same snapshots, each with its own Train
buttons, was one too many. The records themselves are untouched.
"""
import pathlib

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PANEL = (WEB / "components" / "Track2RetentionPanel.tsx").read_text(encoding="utf-8")


def test_the_lists_are_gone():
    for gone in ("Compare with now", "What was actually sent", "trainOnTrack2Snapshot",
                 "restoreTrack2Snapshot", "deleteTrack2Snapshot", "listTrack2Snapshots"):
        assert gone not in PANEL, gone


def test_freezing_still_works_from_here():
    assert "takeTrack2Snapshot(root, snapshotNote)" in PANEL
