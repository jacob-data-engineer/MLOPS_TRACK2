"""The data/training split, enforced (2026-10-28).

    data/      turns files into a dataset version. Imports neither side.
    training/  trains on a dataset version. Reaches the data side ONLY
               through its front door, `ppe_pipeline_api.data`.
    chain/     the arrow between them. Uses both front doors.

The top-level modules (`pull_control`, `notify`, `auth`, ...) are shared
by both. The top-level FORWARDERS left at the old paths (`retention.py`,
`track2_shared.py`, ...) exist for code outside the package and old
imports; inside a side they would be a back door round these rules, so
using one from data/, training/ or chain/ fails here too.

Without this test the split lasts until the first deadline: one
`from ..training import track2_shared` in a data module and the two sides
are one again, with nothing anywhere saying so.
"""

from __future__ import annotations

import ast
import pathlib

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "src" / "ppe_pipeline_api"
SIDES = ("data", "training", "chain")
FORWARDERS = {
    path.stem for path in PACKAGE.glob("*.py")
    if "sys.modules[__name__] = _moved" in path.read_text(encoding="utf-8")
}


def _modules_in(side: str) -> list[str]:
    return sorted(p.stem for p in (PACKAGE / side).glob("*.py") if p.stem != "__init__")


def _imports(path: pathlib.Path, side: str):
    """(line, target) for every relative import, as a dotted path from
    the package root: `data.dataset_yaml`, `training`, `pull_control`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.level == 0:
            continue
        base = [] if node.level >= 2 else [side]
        if node.module:
            base = base + node.module.split(".")
        for alias in node.names:
            yield node.lineno, ".".join(base + [alias.name])


def _violations() -> list[str]:
    found = []
    for side in SIDES:
        for path in sorted((PACKAGE / side).glob("*.py")):
            for line, target in _imports(path, side):
                head, _, rest = target.partition(".")
                where = f"{side}/{path.name}:{line} imports {target}"
                if head in FORWARDERS:
                    found.append(f"{where} -- an old-path forwarder; import from its side")
                elif head == side or head not in SIDES:
                    continue
                elif side == "data":
                    found.append(f"{where} -- the data side imports neither training nor chain")
                elif head == "chain":
                    found.append(f"{where} -- nothing imports the chain")
                elif head == "data" and rest.split(".")[0] in _modules_in("data"):
                    found.append(f"{where} -- through the front door: `from ..data import <name>`")
                elif head == "training" and side == "training":
                    continue
                elif head == "training" and rest.split(".")[0] in _modules_in("training"):
                    found.append(f"{where} -- through the front door: `from ..training import <name>`")
    return found


def test_each_side_uses_the_other_only_through_its_front_door():
    assert _violations() == []


def test_the_front_doors_say_what_they_offer():
    """A front door is the list of what a side promises; the rest is free
    to change. `__all__` is that list, and every name in it must exist."""
    import importlib

    for side in ("data", "training"):
        door = importlib.import_module(f"ppe_pipeline_api.{side}")
        offered = getattr(door, "__all__", None)
        assert offered, f"ppe_pipeline_api.{side} has no __all__"
        for name in offered:
            assert hasattr(door, name), f"{side}.__all__ names {name}, which it lacks"


def test_every_old_path_still_imports_as_the_moved_module():
    import importlib

    for name in FORWARDERS:
        old = importlib.import_module(f"ppe_pipeline_api.{name}")
        assert old.__name__.split(".")[-2] in SIDES, name
