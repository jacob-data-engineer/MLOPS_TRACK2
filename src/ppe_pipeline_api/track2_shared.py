"""Moved to `training/track2_shared.py` (2026-10-28, the data/training split).

This name stays so every existing `from . import track2_shared`, test patch and
the queue Lambda's handler path keep working: importing it hands back the
moved module itself -- one module object, not a copy -- so a patch on
either name patches both.
"""
import sys

from .training import track2_shared as _moved

sys.modules[__name__] = _moved
