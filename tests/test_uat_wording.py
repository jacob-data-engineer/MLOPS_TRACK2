"""Wording the 2026-11-04 UAT found confusing or out of date."""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"


def read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_pull_control_points_at_the_schedule_where_it_now_lives():
    panel = read("web/src/components/Track2PullControlPanel.tsx")
    assert "toolbar pinned at the top of this page" not in panel
    assert 'href="/track2/multi-auto-training"' in panel


def test_assemble_says_nothing_is_copied_from_the_first_line():
    panel = read("web/src/components/Track2DatasetAssemblePanel.tsx")
    assert "this copies them into the images/ + labels/ layout" not in panel
    assert "Nothing is copied: the run's data.yaml names these folders where they already are." in panel


def test_the_sql_pull_explains_what_test_grade_and_the_split_promise():
    page = read("web/src/routes/Track2SqlPage.tsx")
    assert "the split is" in page and "approximate on a small set" in page
    assert "Test-grade means a person drew the boxes that exist" in page


def test_the_user_guide_matches_the_accepted_baseline():
    guide = read("documentation.md")
    assert "## 5. Schedules — both on Multi-Auto Training" in guide
    assert "### Pull queue" not in guide and "Run: Manual / Scheduled" not in guide
    assert "copied from DynamoDB every 5" not in guide
    assert "`helmet_DP`" in guide
    table = guide[guide.index("## 3. The tabs, in the sidebar"):guide.index("## 4.")]
    order = [table.index(f"| Track2 {t} |") for t in
             ("SQL Database Query", "Data Preparation", "Multi-Auto Training", "Data Catalog")]
    assert order == sorted(order)
