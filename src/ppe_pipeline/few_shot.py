"""Few-shot calibration examples for the judge.

Real, human-labeled crops loaded from a JSON manifest and shown to the
judge model as prior example turns before the crop actually being judged.
Nothing here is invented — until a manifest + example crops are supplied,
`load_few_shot_library` isn't called and the judge runs zero-shot
(judge_prompt.SYSTEM_PROMPT only).

Messages built here are provider-neutral: content blocks are either
`{"type": "text", "text": ...}` or `{"type": "image", "media_type": ...,
"data": <base64>}`. judge_client.py adapts this shape to the Bedrock
chat-completions wire format — this module doesn't know or care.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import storage
from .config import PipelineConfig, get_config
from .judge_prompt import REASON_CODES
from .models import ViolationType


@dataclass
class FewShotExample:
    violation_type: ViolationType
    image_path: Path
    verdict: Literal["TP", "FP", "uncertain"]
    reason: str
    confidence: Literal["high", "medium", "low"]
    explanation: str

    def __post_init__(self) -> None:
        valid_reasons = REASON_CODES[self.violation_type]
        if self.reason not in valid_reasons:
            raise ValueError(
                f"reason {self.reason!r} is not valid for {self.violation_type.value}; "
                f"expected one of {valid_reasons}"
            )


FewShotLibrary = dict[ViolationType, list[FewShotExample]]


def add_example_to_library(
    manifest: dict,
    annotations: dict,
    violation_type: ViolationType,
    image_rel_path: str,
    verdict: str,
    reason: str,
    confidence: str,
    explanation: str,
) -> None:
    """Append one example to a raw `manifest.json`-shaped dict (keyed by
    violation-type string) and mirror it into a raw `annotations.json`-shaped
    dict (keyed by the image's own filename) — the two-write pattern those
    files always keep in sync (see `load_few_shot_library`'s docstring).

    Mutates `manifest` and `annotations` in place; callers still own
    persisting them (e.g. via `storage.get_storage().update_json`, as the
    `/fewshot` screen's "Add example" action and
    `scripts/promote_gold_cases_to_fewshot.py` both do).
    """
    entries = manifest.setdefault(violation_type.value, [])
    entries.append(
        {
            "image": image_rel_path,
            "verdict": verdict,
            "reason": reason,
            "confidence": confidence,
            "explanation": explanation,
        }
    )
    annotations[Path(image_rel_path).name] = {
        "reason": reason, "confidence": confidence, "explanation": explanation,
    }


def load_few_shot_library(manifest_path: Path) -> FewShotLibrary:
    """Load a manifest shaped like:

        {
          "glove": [
            {"image": "glove_tp_01.jpg", "verdict": "TP",
             "reason": "clear_bare_hand", "confidence": "high",
             "explanation": "Bare skin visible on both hands, no glove material."},
            ...
          ],
          "helmet": [...],
          "dangerous_plane": [...],
          "too_near_forklift": [...]
        }

    Image paths are resolved relative to the manifest's own directory. Aim
    for ~5 examples per violation type, spanning TP/FP/uncertain.

    An entry may carry `image_key` INSTEAD of `image`, and that is a storage
    key used exactly as given. It exists for a sampled library
    (`few_shot_sampling`), whose entries topped up from the committed set
    point back into `few_shot/` rather than copying ~33 images on every
    build. A relative `../few_shot/...` cannot do that job: `validate_key`
    refuses upward traversal in a key, which is what stops a caller-supplied
    id from escaping the data directory.
    """
    raw, _ = storage.get_storage().read_json(manifest_path.as_posix())
    base_dir = manifest_path.parent
    library: FewShotLibrary = {}

    for type_str, entries in raw.items():
        violation_type = ViolationType(type_str)
        library[violation_type] = [
            FewShotExample(
                violation_type=violation_type,
                image_path=(
                    Path(entry["image_key"]) if entry.get("image_key") else base_dir / entry["image"]
                ),
                verdict=entry["verdict"],
                reason=entry["reason"],
                confidence=entry["confidence"],
                explanation=entry["explanation"],
            )
            for entry in entries
        ]

    return library


def _media_type_for(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise ValueError(f"unsupported few-shot image type: {path}")


def _example_to_turns(example: FewShotExample) -> tuple[dict, dict]:
    image_bytes = storage.get_storage().get_bytes(example.image_path.as_posix())
    media_type = _media_type_for(example.image_path)
    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    user_turn = {
        "role": "user",
        "content": [
            {"type": "image", "media_type": media_type, "data": b64_image},
            {"type": "text", "text": f"Example: judge {example.violation_type.value} against this crop."},
        ],
    }
    assistant_text = json.dumps(
        {
            "person_id": "example",
            "results": [
                {
                    "violation_type": example.violation_type.value,
                    "verdict": example.verdict,
                    "reason": example.reason,
                    "confidence": example.confidence,
                    "explanation": example.explanation,
                }
            ],
        }
    )
    assistant_turn = {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]}
    return user_turn, assistant_turn


def build_few_shot_messages(
    library: FewShotLibrary,
    violation_types: list[ViolationType],
    max_per_type: int | None = None,
    config: PipelineConfig | None = None,
) -> list[dict]:
    """Example turns for only the violation types this call actually needs,
    so a person checked for one violation type doesn't pay for all four
    types' worth of example images.

    This preamble is byte-identical for every call with the same
    violation-type set — exactly the "shared prefix, varying suffix"
    pattern prompt caching is built for; judge_client.py hands it to
    Bedrock as one stable prefix.

    `max_per_type` overrides `config.few_shot.max_examples_per_type` (and
    any per-violation-type override in `config.few_shot.max_examples_per_type_override`
    — config.toml / built-in default — see config.py) for every violation
    type in this call.

    ORDER DECIDES WHAT A LIMIT KEEPS. Truncation is from the front, so
    `config.few_shot.move_to_back_override` rotates the named number of
    examples from the front to the back FIRST, and the limit then falls on
    what is left at the front.
    """
    config = config or get_config()

    messages: list[dict] = []
    for violation_type in violation_types:
        if max_per_type is not None:
            limit = max_per_type
        else:
            limit = config.few_shot.max_examples_per_type_override.get(
                violation_type.value, config.few_shot.max_examples_per_type
            )
        # ROTATE, THEN LIMIT, and in that order. The limit truncates from the
        # front, so moving examples to the back is how a limit is made to drop
        # them -- see `config.FewShotConfig.move_to_back_override`.
        examples = library.get(violation_type, [])
        move_back = config.few_shot.move_to_back_override.get(violation_type.value, 0)
        if move_back > 0:
            examples = examples[move_back:] + examples[:move_back]
        for example in examples[:limit]:
            user_turn, assistant_turn = _example_to_turns(example)
            messages.append(user_turn)
            messages.append(assistant_turn)

    return messages
