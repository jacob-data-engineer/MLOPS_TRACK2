"""`nc` is labelled in every yaml the system writes (asked for, 2026-10-26).

The key stays `nc` -- it is Ultralytics' own name for it -- so the label is
a YAML comment, which every loader ignores.
"""
import pathlib
import sys

import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy  # noqa: E402


def test_the_line_says_what_nc_is():
    text = dy._dump({"path": "s3://b", "train": ["a"], "nc": 6, "names": {0: "helmet"}})
    assert "nc: 6  # number of classes" in text


def test_it_still_loads_as_the_same_number():
    text = dy._dump({"train": ["a"], "nc": 1, "names": {0: "helmet"}})
    assert yaml.safe_load(text)["nc"] == 1


def test_nothing_else_is_touched():
    doc = {"train": ["a"], "val": ["b"], "names": {0: "x"}}
    assert dy._dump(doc) == yaml.dump(doc, Dumper=dy._NoAliasDumper, sort_keys=False,
                                       default_flow_style=False)
