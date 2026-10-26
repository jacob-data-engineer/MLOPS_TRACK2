"""The retention bars load on their own whenever the card is open.

They used to wait for a Preview press, so the card opened with no answer
to "what would a retrain use today" until somebody remembered the button.
"""
import pathlib

PANEL = (pathlib.Path(__file__).resolve().parents[1] / "web" / "src" / "components"
         / "Track2RetentionPanel.tsx").read_text(encoding="utf-8")


def test_the_bars_load_without_a_press():
    assert "if (!open || !root) return;" in PANEL
    assert "if (!open || !plan) return;" not in PANEL


def test_there_is_no_preview_button():
    assert ">\n              Preview\n            </button>" not in PANEL
    assert "Reading the week folders..." in PANEL
