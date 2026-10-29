"""The sidebar says how finished each tab is (2026-10-28): green in use,
yellow prototype, red kept in view."""

import pathlib
import re

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
NAV = (WEB / "components" / "Layout.tsx").read_text(encoding="utf-8")
CSS = (WEB / "index.css").read_text(encoding="utf-8")


def _status(route: str):
    line = next((l for l in NAV.splitlines() if f'to: "{route}"' in l), None)
    assert line is not None, f"{route} is not in the nav"
    found = re.search(r'status: "(\w+)"', line)
    return found.group(1) if found else None


def test_each_tab_has_its_colour():
    assert _status("/track2/data-preparation") == "ready"
    assert _status("/track2/multi-auto-training") == "ready"
    assert _status("/track2/scheduled") == "prototype"
    assert _status("/track2/metrics") == "prototype"
    assert _status("/track2/notifications") is None
    assert _status("/track2/help") is None


def test_sql_comes_first_then_data_then_training():
    order = re.findall(r'to: "(/track2/[\w-]+)"', NAV)
    assert order[:3] == ["/track2/sql", "/track2/data-preparation",
                         "/track2/multi-auto-training"]


def test_deploy_and_inference_is_off_the_nav_but_still_routed():
    """Removed from the sidebar 2026-10-29; the page stays reachable by
    its address, like Evaluation and Synthetic."""
    assert 'to: "/track2/deploy"' not in NAV
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    assert 'path="/track2/deploy" element={<Track2DeployPage />}' in app


def test_the_colours_are_defined():
    for token in ("--color-status-ready", "--color-status-prototype", "--color-status-kiv"):
        assert token in CSS


def test_one_text_colour_per_entry():
    """Status replaces the neutral ink; both on one element would be
    settled by stylesheet order."""
    assert 'status ? STATUS_STYLE[status].className : isActive ? "text-ink" : "text-ink-secondary"' in NAV
