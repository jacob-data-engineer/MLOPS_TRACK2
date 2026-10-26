"""The one schema for the extra Ultralytics training arguments.

Every run that carries hyperparameters -- the manual form, a queue entry,
the unattended chain, a saved `default_training.hyperparams` -- reaches
`track2_shared.encode_hyperparams`, which calls `normalize_hyperparams`
here. So this is the single place a value is typed and checked before a
paid job is submitted.

WHY IT EXISTS. The card used to send every value as a string
(`lr0: "0.01"`, `cos_lr: "false"`, `freeze: "10"`), and the trainer
passed them to `model.train` unchanged. Ultralytics' `check_cfg` raises
TypeError for a string `lr0`, and silently ignores a string `freeze`
(it accepts only an int or a list), so a ticked box with nothing edited
either failed the job or froze nothing while the audit said otherwise.

The trainer image cannot import this module, so
`model_training/training-container/train.py` carries a small mirror of
it (`_HYPERPARAM_KINDS`) to type legacy string values defensively. Keep
the two in step; `tests/test_hyperparam_types.py` checks they agree.

Kinds:

    float   finite number, optional [min, max]
    int     whole number ("10", 10, 10.0 -- never "2.5"), optional min
    bool    true/false/1/0/yes/no, case-insensitive -- never bool("false")
    choice  one of a fixed set of strings, matched case-insensitively
    amp     a bool, or one of Ultralytics' precision names
    freeze  a non-negative int (first N layers) or a list of
            non-negative ints (those layer indices); "0,1,2" -> [0, 1, 2];
            "none" or "" means not set

Keys not in the schema are passed through untouched (blank ones dropped):
the trainer validates them against the Ultralytics it actually has, which
is the design that lets a new knob be exposed without an image rebuild.
"""
from __future__ import annotations

import math
import re
from typing import Any

#: Decided by the pipeline, the form and the dataset -- mirrors train.py's
#: RESERVED_HYPERPARAMS. Refused here so the job is never submitted
#: rather than failing once it reaches the GPU host.
RESERVED = frozenset({
    "data", "model", "epochs", "imgsz", "batch", "seed", "project", "name",
    "exist_ok", "resume", "device", "workers", "save_dir", "classes",
    "single_cls", "patience",
})

_FRACTION = {"kind": "float", "min": 0.0, "max": 1.0}
_NON_NEGATIVE = {"kind": "float", "min": 0.0}

#: Key -> kind and limits. The key list is Track2TrainingAdvanced.tsx's
#: GROUPS table; the kinds and ranges are Ultralytics 8.4.130's own
#: (CFG_FRACTION_KEYS are [0, 1], CFG_INT_KEYS whole numbers, ...).
SCHEMA: dict[str, dict[str, Any]] = {
    # Optimiser
    "optimizer": {"kind": "choice", "choices": (
        "auto", "SGD", "Adam", "Adamax", "AdamW", "NAdam", "RAdam",
        "RMSprop", "MuSGD")},
    "lr0": _FRACTION,
    "lrf": _FRACTION,
    "momentum": _FRACTION,
    "weight_decay": _FRACTION,
    "warmup_epochs": _NON_NEGATIVE,
    "warmup_momentum": _FRACTION,
    "warmup_bias_lr": _FRACTION,
    "cos_lr": {"kind": "bool"},
    # Fine-tuning
    "freeze": {"kind": "freeze"},
    "dropout": _FRACTION,
    "multi_scale": _FRACTION,
    "rect": {"kind": "bool"},
    "amp": {"kind": "amp"},
    # A ratio of the dataset in (0, 1]. Ultralytics also takes a count or
    # a [train, val] pair; the card offers the ratio only.
    "fraction": {"kind": "float", "min": 0.0, "max": 1.0, "exclusive_min": True},
    # Loss weights
    "box": _NON_NEGATIVE,
    "cls": _NON_NEGATIVE,
    "dfl": _NON_NEGATIVE,
    "nbs": {"kind": "int", "min": 1},
    # Augmentation
    "hsv_h": _FRACTION,
    "hsv_s": _FRACTION,
    "hsv_v": _FRACTION,
    "degrees": {"kind": "float", "min": -360.0, "max": 360.0},
    "translate": _FRACTION,
    "scale": _FRACTION,
    "shear": {"kind": "float", "min": -180.0, "max": 180.0},
    "perspective": _FRACTION,
    "flipud": _FRACTION,
    "fliplr": _FRACTION,
    "mosaic": _FRACTION,
    "close_mosaic": {"kind": "int", "min": 0},
    "mixup": _FRACTION,
    "cutmix": _FRACTION,
    "copy_paste": _FRACTION,
    "erasing": _FRACTION,
    "bgr": _FRACTION,
    "auto_augment": {"kind": "choice", "choices": (
        "randaugment", "autoaugment", "augmix")},
}

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}
_AMP_MODES = {"fp16", "bf16", "fp32"}
_NUMBER = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")
_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


class HyperparamError(ValueError):
    """A hyperparameter that cannot be sent. The message names the key
    and the value, so it can be shown to the operator as it stands."""


def _bad(key: str, value: Any, why: str) -> HyperparamError:
    return HyperparamError(f"Training parameter {key}={value!r} is invalid: {why}.")


def _is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value.strip() == "")


def _number(key: str, value: Any) -> float:
    if isinstance(value, bool):
        raise _bad(key, value, "expected a number, got true/false")
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str) and _NUMBER.match(value.strip()):
        number = float(value.strip())
    else:
        raise _bad(key, value, "expected a number")
    if not math.isfinite(number):
        raise _bad(key, value, "must be a finite number")
    return number


def _whole(key: str, value: Any) -> int:
    number = _number(key, value)
    if not number.is_integer():
        raise _bad(key, value, "must be a whole number")
    return int(number)


def _bool(key: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in _TRUE:
            return True
        if token in _FALSE:
            return False
    raise _bad(key, value, "expected true or false")


def _freeze(key: str, value: Any) -> int | list[int] | None:
    given = value                       # named in any refusal, as typed
    if isinstance(value, str):
        text = value.strip()
        if text.lower() == "none" or text == "":
            return None
        if "," in text or text.startswith("["):
            parts = [p.strip() for p in text.strip("[]").split(",") if p.strip()]
            if not parts:
                raise _bad(key, given, "an empty layer list")
            value = parts
        else:
            value = text
    if isinstance(value, (list, tuple)):
        layers = [_whole(key, part) for part in value]
        if any(n < 0 for n in layers):
            raise _bad(key, given, "layer indices must be 0 or more")
        if not layers:
            raise _bad(key, given, "an empty layer list")
        return layers
    count = _whole(key, value)
    if count < 0:
        raise _bad(key, given, "must be 0 or more layers")
    return count


def _one(key: str, value: Any) -> Any:
    """Typed value for `key`, or None meaning "not set"."""
    spec = SCHEMA[key]
    kind = spec["kind"]
    if kind == "freeze":
        return _freeze(key, value)
    if kind == "bool":
        return _bool(key, value)
    if kind == "amp":
        if isinstance(value, str) and value.strip().lower() in _AMP_MODES:
            return value.strip().lower()
        return _bool(key, value)
    if kind == "choice":
        if not isinstance(value, str):
            raise _bad(key, value, f"must be one of {', '.join(spec['choices'])}")
        canonical = {c.lower(): c for c in spec["choices"]}.get(value.strip().lower())
        if canonical is None:
            raise _bad(key, value, f"must be one of {', '.join(spec['choices'])}")
        return canonical
    if kind == "int":
        number: float | int = _whole(key, value)
    else:
        number = _number(key, value)
    low, high = spec.get("min"), spec.get("max")
    if low is not None and (number < low or (spec.get("exclusive_min") and number == low)):
        bound = "above" if spec.get("exclusive_min") else "at least"
        raise _bad(key, value, f"must be {bound} {low:g}")
    if high is not None and number > high:
        raise _bad(key, value, f"must be at most {high:g}")
    return number


def normalize_hyperparams(raw: Any) -> dict[str, Any]:
    """Typed, validated copy of a hyperparameter mapping.

    {} for None or an empty mapping -- a run that sets nothing behaves
    exactly as before. Blank values are dropped ("not set"). Raises
    HyperparamError naming the key and value for anything that cannot be
    sent. Accepts legacy all-string values (what the card used to send
    and what older settings files hold) and returns real types.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise HyperparamError(
            f"Training parameters must be a mapping of name to value, got {type(raw).__name__}.")
    out: dict[str, Any] = {}
    for raw_key, value in raw.items():
        key = str(raw_key).strip()
        if not _KEY.match(key):
            raise HyperparamError(f"Training parameter name {raw_key!r} is not a valid name.")
        if key in RESERVED:
            raise HyperparamError(
                f"Training parameter {key} is decided by the pipeline, not here -- "
                "setting it would make the audit row and the run disagree.")
        if _is_blank(value):
            continue
        if key in SCHEMA:
            typed = _one(key, value)
            if typed is not None:
                out[key] = typed
            continue
        # Not one the card offers: passed through for the trainer, which
        # checks it against the Ultralytics it has installed.
        if isinstance(value, str):
            value = value.strip()
        elif isinstance(value, float) and not math.isfinite(value):
            raise _bad(key, value, "must be a finite number")
        elif not isinstance(value, (bool, int, float, list)):
            raise _bad(key, value, "expected a number, true/false, text or a list")
        out[key] = value
    return out
