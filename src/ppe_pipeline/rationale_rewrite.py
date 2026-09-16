"""Rewrite human reviewer rationales into one consistent, professional
paragraph — the shared text normaliser behind gold-set annotations and the
few-shot library's `explanation` field.

Why this exists: reviewer notes are written fast and terse ("worker is close
to the forklift", "white hard helmet barely visible, contrast visible"), and
`gold_set._merge_reviewer_rationales` concatenates two of them into
"Reviewer A: ... Reviewer B: ...". That reads as two fragments stapled
together, and it is what both the few-shot prompt's `explanation` and the
reasoning-alignment scorer's `ground_truth_rationale` are built from. This
module turns any of those into a single paragraph with a consistent voice.

STRICTLY TEXT-ONLY. The rewriter never sees the image, and is prompted hard
against inventing visual detail: it may only re-express what the reviewer
already wrote. That is a deliberate limit, not an oversight — a rewrite that
added scene description would put model-authored claims into the ground
truth that the reasoning-alignment scorer then grades the judge against, and
this codebase already refuses to fabricate ground-truth reasoning elsewhere
(`gold_set._merge_reviewer_rationales` returns None on a half-missing pair;
`rationale_sync` refuses to propagate a `cannot_determine` case). A terse
note therefore stays a short paragraph — it does not grow new facts.

Rewrites replace the reviewer's text IN PLACE, per the project decision to
keep one rationale field rather than parallel *_rewritten ones. To stop every
save from re-polishing the previous rewrite forever, the SHA-256 of each
rewrite this module produces is recorded in the entry's
`rationale_rewrite_hashes` sidecar dict. On a later pass, a field whose text
still hashes to the recorded value is skipped as already-rewritten; a field
whose text differs (a human edited it) is rewritten again. Those hashes are
bookkeeping, not content — they hold no original text.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

import anthropic

from .config import PipelineConfig, get_config

if TYPE_CHECKING:
    from .models import ViolationType

logger = logging.getLogger(__name__)

HASHES_FIELD = "rationale_rewrite_hashes"

# The three shapes of input this module normalises. They differ only in the
# instruction line prepended to the user turn.
KIND_REVIEWER = "reviewer"
KIND_MERGED = "merged"
KIND_ADJUDICATED = "adjudicated"

_SYSTEM = """You rewrite safety-inspection review notes for a port PPE compliance dataset.

Each note was written quickly by a trained human reviewer looking at a photo of a worker. Your job is to re-express that note as one clear, professional paragraph suitable for a formal annotation record.

ABSOLUTE CONSTRAINTS - violating any of these makes the output unusable:
- You are NOT shown the image. Describe NOTHING you were not told.
- Add no new visual detail, object, location, measurement, body part, colour, or circumstance. If the reviewer did not mention it, it does not exist.
- Do not strengthen or weaken the reviewer's certainty. "barely visible" must not become "clearly visible"; "appears to be" must not become "is".
- Do not change, question, or argue with the stated verdict. You are rewriting the reasoning, not re-deciding the case.
- If the note is very short, the paragraph stays short. Do not pad it to look thorough.
- Keep any hedging, doubt or ambiguity the reviewer expressed.

STYLE:
- One paragraph. No headings, no bullet points, no line breaks.
- Third person, present tense, impersonal register: "The worker is...", "No helmet is visible...". Never "I", "we" or "you".
- Do not name or number the reviewers, and do not write "Reviewer A", "Reviewer B" or "the reviewer".
- Plain professional English. Expand obvious shorthand, fix grammar and spelling, and use the full term on first use.
- Lead with the observation that determines the verdict, then any qualifying detail.

Output ONLY the rewritten paragraph. No preamble, no quotation marks, no commentary."""

_KIND_INSTRUCTION = {
    KIND_REVIEWER: "This is one reviewer's own note. Rewrite it as a single paragraph.",
    KIND_MERGED: (
        "This is two reviewers' notes concatenated, and they agreed on the verdict. "
        "Merge them into ONE paragraph in a single consistent voice. Where they say the "
        "same thing, say it once. Where each adds something the other did not, keep both "
        "observations. Do not attribute anything to a specific reviewer."
    ),
    KIND_ADJUDICATED: (
        "This is an adjudicator's ruling on a case where the two reviewers disagreed. "
        "Rewrite it as a single paragraph, preserving the reasoning that settled the "
        "disagreement."
    ),
}


def text_hash(text: str) -> str:
    """Fingerprint used to recognise this module's own output on a later pass."""
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()


def build_client(config: PipelineConfig | None = None) -> anthropic.AnthropicBedrockMantle:
    """Claude on Bedrock, via the Messages API.

    Deliberately NOT the OpenAI-compatible `bedrock-mantle` chat-completions
    endpoint `judge_client.py` uses for qwen: Bedrock rejects the Claude
    models on that path outright ("The model 'anthropic.claude-opus-5' does
    not support the '/v1/chat/completions' API"), so they are reachable only
    through the Messages API client.

    The region is its own setting rather than `config.judge.bedrock.region`
    for the same practical reason — the judge runs in us-west-2, which
    carries only claude-haiku-4-5, while the full Claude lineup is in
    us-east-1. Credentials resolve from the environment exactly as the
    judge's do (AWS_BEARER_TOKEN_BEDROCK).
    """
    config = config or get_config()
    return anthropic.AnthropicBedrockMantle(aws_region=config.rewriter.region)


def rewrite(
    text: str,
    *,
    kind: str = KIND_REVIEWER,
    violation_type: ViolationType | str | None = None,
    final_label: str | None = None,
    reason_code: str | None = None,
    client: object | None = None,
    config: PipelineConfig | None = None,
) -> str:
    """Return `text` rewritten as one professional paragraph.

    `violation_type`, `final_label` and `reason_code` are supplied as context
    so the paragraph reads coherently against the verdict — not as licence to
    add detail, which the system prompt forbids regardless.

    Raises whatever the client raises. Callers on a reviewer's save path must
    treat this as best-effort and keep the human's text on failure — see
    `rewrite_entry_fields`.
    """
    config = config or get_config()
    text = (text or "").strip()
    if not text:
        raise ValueError("nothing to rewrite: text is empty")

    client = client or build_client(config)

    violation_value = getattr(violation_type, "value", violation_type)
    context = "\n".join(
        line
        for line in (
            f"Violation type under review: {violation_value}" if violation_value else None,
            f"Verdict reached: {final_label}" if final_label else None,
            f"Reason code: {reason_code}" if reason_code else None,
        )
        if line
    )

    user_content = (
        f"{_KIND_INSTRUCTION.get(kind, _KIND_INSTRUCTION[KIND_REVIEWER])}\n\n"
        f"{context}\n\nNote to rewrite:\n{text}"
    )

    # `output_config` is sent only when an effort level is configured.
    # Effort is a control on the current Opus/Sonnet line; a model without
    # it (claude-haiku-4-5, the plausible cheaper/faster swap here) rejects
    # the request that carries it. Since `rewrite_entry_fields` deliberately
    # swallows per-field failures so a reviewer never loses a save, sending
    # it to a model that refuses it fails silently — every rewrite quietly
    # not happening. Hence `effort = ""` to omit, rather than a hardcoded
    # per-model table that would drift as models are added.
    extra = {"output_config": {"effort": config.rewriter.effort}} if config.rewriter.effort else {}

    response = client.messages.create(
        model=config.rewriter.model,
        max_tokens=config.rewriter.max_tokens,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
        **extra,
    )
    if getattr(response, "stop_reason", None) == "refusal":
        raise RuntimeError(f"rewrite refused: {getattr(response, 'stop_details', None)}")

    out = "".join(block.text for block in response.content if block.type == "text").strip()
    if not out:
        raise RuntimeError("rewrite returned no text")
    return out


# Which annotation fields get rewritten, and as which kind. `final_rationale`
# is listed once here but its kind depends on how the case resolved, so
# `rewrite_entry_fields` picks that per entry rather than reading it here.
REVIEWER_FIELDS = ("reviewer_a_rationale", "reviewer_b_rationale")
FINAL_FIELD = "final_rationale"


def _needs_rewrite(entry: dict, field: str) -> bool:
    text = (entry.get(field) or "").strip()
    if not text:
        return False
    recorded = (entry.get(HASHES_FIELD) or {}).get(field)
    return recorded != text_hash(text)


def rewrite_entry_fields(
    entry: dict,
    *,
    fields: tuple[str, ...] | None = None,
    violation_type: ViolationType | str | None = None,
    client: object | None = None,
    config: PipelineConfig | None = None,
) -> tuple[dict, list[str], list[str]]:
    """Rewrite the rationale fields of one `gold_annotations.json` entry.

    Returns `(updated_entry, rewritten_fields, failed_fields)`. Never raises
    for an individual field: a field whose rewrite fails is left exactly as
    the human wrote it and named in `failed_fields`, so a reviewer's save is
    never lost to an API problem and a later backfill can retry it. `entry`
    is not mutated; a copy is returned.

    Fields already holding this module's own output (matching hash) are
    skipped, so repeated saves cost nothing and do not drift the text.

    `final_rationale` is rewritten as KIND_ADJUDICATED when the entry was
    actually adjudicated, and as KIND_MERGED otherwise — the auto-resolved
    case, where `_merge_reviewer_rationales` glued both reviewers together
    and the merge is what needs smoothing into one voice.
    """
    config = config or get_config()
    entry = dict(entry)
    candidates = fields if fields is not None else (*REVIEWER_FIELDS, FINAL_FIELD)
    pending = [f for f in candidates if _needs_rewrite(entry, f)]
    if not pending:
        return entry, [], []

    client = client or build_client(config)
    hashes = dict(entry.get(HASHES_FIELD) or {})
    rewritten: list[str] = []
    failed: list[str] = []

    for field in pending:
        if field == FINAL_FIELD:
            kind = (
                KIND_ADJUDICATED
                if entry.get("adjudication_status") == "adjudicated"
                else KIND_MERGED
            )
        else:
            kind = KIND_REVIEWER
        try:
            new_text = rewrite(
                entry[field],
                kind=kind,
                violation_type=violation_type,
                final_label=entry.get("final_label"),
                reason_code=entry.get("reason_code"),
                client=client,
                config=config,
            )
        except Exception:
            logger.exception("rationale rewrite failed for %s; keeping the human's text", field)
            failed.append(field)
            continue
        entry[field] = new_text
        hashes[field] = text_hash(new_text)
        rewritten.append(field)

    if rewritten:
        entry[HASHES_FIELD] = hashes
    return entry, rewritten, failed
