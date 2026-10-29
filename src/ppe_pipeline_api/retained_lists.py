"""Moved to `data/retained_lists.py` (2026-10-28, the data/training split).

This name stays so every existing `from . import retained_lists`, test patch and
the queue Lambda's handler path keep working: importing it hands back the
moved module itself -- one module object, not a copy -- so a patch on
either name patches both.
"""
import sys

from .data import retained_lists as _moved

sys.modules[__name__] = _moved
