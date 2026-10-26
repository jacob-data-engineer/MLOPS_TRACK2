"""The guided flow and the standard page must read the same data.yaml.

`pull_control._classes_from_yaml` is hand-rolled because that module IS
the Lambda handler file and may not import PyYAML. That is a real
constraint; accepting only one of the four shapes Ultralytics allows was
not. The Help page documents the list form, the standard page parses it
with `yaml.safe_load`, and the guided flow returned `{}` -- which
surfaced as "No data.yaml was found to read class names from", blaming a
missing file for one that is present and valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import pull_control as pc  # noqa: E402


def _reference(text: str) -> dict:
    """What the STANDARD page sees -- a real YAML parser."""
    names = (yaml.safe_load(text) or {}).get("names")
    if isinstance(names, dict):
        return {str(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {str(i): str(v) for i, v in enumerate(names)}
    return {}


DOCUMENTS = {
    "block mapping": "path: .\ntrain: images/train\nnames:\n  0: helmet\n  1: gloves\n",
    "block sequence": "path: .\nnames:\n  - helmet\n  - gloves\n",
    "flow sequence quoted": 'names: ["helmet", "gloves"]\n',
    "flow sequence bare": "names: [helmet, gloves]\n",
    "flow mapping": "names: {0: helmet, 1: gloves}\n",
    "multiline flow": 'names: [\n  "helmet",\n  "gloves",\n]\n',
    "single quotes": "names:\n  0: 'helmet'\n  1: 'gloves'\n",
    "trailing key": "names:\n  0: helmet\n  1: gloves\nnc: 2\npath: .\n",
    "six classes": "names:\n  - a\n  - b\n  - c\n  - d\n  - e\n  - f\n",
    "no names at all": "path: .\ntrain: images/train\n",
}


@pytest.mark.parametrize("label", sorted(DOCUMENTS))
def test_it_agrees_with_a_real_yaml_parser(label):
    text = DOCUMENTS[label]
    assert pc._classes_from_yaml(text) == _reference(text), label


class TestTheFormsThatUsedToReturnNothing:
    def test_a_block_sequence(self):
        assert pc._classes_from_yaml("names:\n  - helmet\n  - gloves\n") == {
            "0": "helmet", "1": "gloves",
        }

    def test_an_inline_list(self):
        assert pc._classes_from_yaml('names: ["helmet", "gloves"]\n') == {
            "0": "helmet", "1": "gloves",
        }

    def test_an_inline_mapping(self):
        assert pc._classes_from_yaml("names: {0: helmet, 1: gloves}\n") == {
            "0": "helmet", "1": "gloves",
        }


class TestItStillRefusesToGuess:
    def test_a_non_numeric_index_is_skipped_not_renumbered(self):
        # Renumbering would silently shift every class after it, which on a
        # trained model means the wrong name on the wrong box.
        assert pc._classes_from_yaml("names:\n  helmet: 0\n  gloves: 1\n") == {}

    def test_a_file_with_no_names_gives_nothing(self):
        assert pc._classes_from_yaml("path: .\ntrain: images/train\n") == {}

    def test_an_empty_list_gives_nothing(self):
        assert pc._classes_from_yaml("names: []\n") == {}

    def test_the_block_ends_at_the_next_top_level_key(self):
        assert pc._classes_from_yaml("names:\n  0: helmet\nnc: 1\npath: /x\n") == {
            "0": "helmet",
        }


class TestIndicesSurviveRoundTrip:
    def test_sparse_indices_are_kept_as_written(self):
        # A dataset whose data.yaml skips an index is unusual but legal,
        # and renumbering it would remap every label file silently.
        assert pc._classes_from_yaml("names:\n  0: helmet\n  2: boots\n") == {
            "0": "helmet", "2": "boots",
        }
