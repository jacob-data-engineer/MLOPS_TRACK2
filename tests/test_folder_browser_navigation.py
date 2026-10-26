"""Clicking a folder twice must not build a path that never existed.

Reported as: open `latest_approved_model_weights`, click it again, and
the breadcrumb reads

    bucket root / latest_approved_model_weights / latest_approved_model_weights

with "Nothing here." underneath.

**It is a race, not a double-click.** `folders` is a prop. Every caller
does the same two things at two different moments: `onNavigate` sets a
prefix, and an effect notices and fetches the listing for it. Between
them the browser re-renders with the NEW prefix and the OLD rows, and a
row's destination is `prefix + "/" + name` computed at click time. So any
click in that window joins a row belonging to the previous folder onto
the new prefix -- the same row gives the doubled path above, and a
different row gives an equally wrong one, which is why a plain
double-click guard would not have fixed it.

These are source assertions because the component is TSX and there is no
JS test runner here (no Node on this machine; CodeBuild is the only
TypeScript gate). They pin the SHAPE of the fix, which is what would be
lost in a later edit: one navigation path, a guard that hides the rows,
and a backstop that cannot leave the browser empty for ever.
"""
from __future__ import annotations

import pathlib
import re

BROWSER = (pathlib.Path(__file__).resolve().parents[1]
           / "web" / "src" / "components" / "Track2S3FolderBrowser.tsx")
WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"

SOURCE = BROWSER.read_text(encoding="utf-8")


def test_every_navigation_goes_through_one_function():
    """The breadcrumb and the bucket-root button change the prefix too,
    and leave exactly the same stale listing behind. A guard on the rows
    alone would fix one third of the ways in."""
    assert SOURCE.count("onNavigate(") == 1, \
        "onNavigate is called somewhere other than navigate()"
    assert "function navigate(path: string, label: string)" in SOURCE
    for entry in ('navigate("", "the bucket root")',
                  "navigate(pathUpTo(i), seg)",
                  "navigate(fullPath, name)"):
        assert entry in SOURCE, entry


def test_the_rows_are_hidden_while_they_belong_somewhere_else():
    """Hidden, not disabled. A tick box beside a stale row selects
    `currentPrefix + "/" + oldName` -- the same wrong path the click
    produced, but quieter and harder to notice."""
    assert "{opening === null && filteredFolders.map(" in SOURCE
    assert "{opening === null && filteredFiles.map(" in SOURCE


def test_the_listing_is_compared_by_value_not_identity():
    """Two callers hand over a brand new array on every render --
    `folders={listing[key] ?? []}` -- and an identity check would read
    that as the new listing arriving before the request had left."""
    assert "const listing = useMemo(" in SOURCE
    assert "folders.join(" in SOURCE
    assert "[folders, files]" in SOURCE


def test_there_is_a_backstop_so_the_browser_cannot_stay_empty():
    """A caller whose effect does not refetch -- or a folder holding
    exactly what its parent held, so the listing never changes -- would
    otherwise sit behind the guard for ever. A short pause is
    recoverable; an empty browser is not."""
    assert "setTimeout(() => setOpening(null), 1500)" in SOURCE
    assert "clearTimeout" in SOURCE


def test_the_inline_array_caller_this_was_written_for_still_exists():
    """If this stops being true the value-comparison above can be
    relaxed to an identity check -- and this test is the reminder to
    check rather than assume."""
    assert any("?? []}" in path.read_text(encoding="utf-8")
               for path in WEB.rglob("*.tsx")
               if "Track2S3FolderBrowser" in path.read_text(encoding="utf-8")
               and path.name != "Track2S3FolderBrowser.tsx")


def test_every_folder_picker_in_the_product_uses_this_one_component():
    """The fix is only system-wide if there is one browser. A second,
    bespoke folder list would have the same race and none of the guard.
    """
    callers = [p for p in WEB.rglob("*.tsx")
               if "Track2S3FolderBrowser" in p.read_text(encoding="utf-8")
               and p.name != "Track2S3FolderBrowser.tsx"]
    assert len(callers) >= 10, f"only found {len(callers)} callers"
    # Nothing else rolls its own: a component rendering a prefix
    # breadcrumb would be the giveaway.
    for path in WEB.rglob("*.tsx"):
        if path.name == "Track2S3FolderBrowser.tsx":
            continue
        text = path.read_text(encoding="utf-8")
        assert "bucket root" not in text, f"{path.name} draws its own breadcrumb"


def test_each_caller_refetches_when_the_prefix_changes():
    """The guard clears when a new listing arrives. A caller that never
    fetched again would fall through to the 1.5s backstop every time and
    be back where it started."""
    expected = {
        "Track2ImportCheckpointPanel.tsx": "[source, prefix]",
        "Track2DetectorsPanel.tsx": "prefix",
    }
    for name, dep in expected.items():
        text = (WEB / "components" / name).read_text(encoding="utf-8")
        effects = re.findall(r"useEffect\([\s\S]*?\}, \[([^\]]*)\]\);", text)
        assert any(dep.strip("[]") in e for e in effects), \
            f"{name} has no effect keyed on the prefix"
