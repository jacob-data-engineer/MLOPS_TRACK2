"""Moved to `data/golden_set.py` (2026-10-28, the data/training split).

This name stays so every existing `from . import golden_set`, test patch and
the queue Lambda's handler path keep working: importing it hands back the
moved module itself -- one module object, not a copy -- so a patch on
either name patches both.
"""
import sys

from .data import golden_set as _moved

sys.modules[__name__] = _moved
