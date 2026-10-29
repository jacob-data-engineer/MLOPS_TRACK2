"""How a side's front door hands out what it offers.

`through(...)` returns a function that looks the real one up at CALL
time, not import time. Two reasons:

- No import cycles. A front door imports nothing until it is used, so a
  side can be imported by the chain, and the chain by the tick, in any
  order.
- One thing to patch. A test that replaces `dataset_yaml.build` changes
  what `data.build_yaml` calls too, because the front door asks
  `dataset_yaml` for `build` every time. A copy taken at import would
  keep calling the original.
"""

from __future__ import annotations

import importlib


def through(package: str, module: str, name: str):
    def call(*args, **kwargs):
        return getattr(importlib.import_module(f"{package}.{module}"), name)(*args, **kwargs)

    call.__name__ = call.__qualname__ = name
    call.__doc__ = f"`{module}.{name}`, offered through the `{package.rsplit('.', 1)[-1]}` front door."
    return call


def value(package: str, module: str, name: str):
    """A constant, read when asked for (see the side's `__getattr__`)."""
    return getattr(importlib.import_module(f"{package}.{module}"), name)
