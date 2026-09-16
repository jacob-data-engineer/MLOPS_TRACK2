"""Batched judge over Amazon Bedrock's OpenAI-compatible `bedrock-mantle`
endpoint (not boto3/the Converse API).

Verdicts only. Locating the helmet on a cleared case used to be a second
grounding call from this client; it is now a local YOLO11s detector — see
helmet_detector.py.

Bedrock is the only provider. The direct-Anthropic, OpenRouter and on-GPU
`local` paths this class used to carry were removed once Bedrock became the
sole deployed judge — see the README's "The model stack".

One image, one call, one verdict per applicable violation type, optionally
primed with few-shot examples. See judge_prompt.py for the system prompt
and schema, few_shot.py for the (provider-neutral) example turns.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import time
from dataclasses import dataclass
from typing import Callable, Literal, Protocol

import openai
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .config import PipelineConfig, RetryConfig, get_config
from .few_shot import FewShotLibrary, build_few_shot_messages
from .judge_prompt import (
    build_batch_output_schema,
    build_batch_output_schema_labels,
    build_batch_user_text,
    REASONING_ALIGNMENT_SCHEMA,
    REASONING_ALIGNMENT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_output_schema,
    build_reasoning_alignment_user_text,
    build_user_text,
)
from .models import ViolationType

logger = logging.getLogger(__name__)

BEDROCK_PROVIDER = "bedrock"
BEDROCK_API_KEY_ENV_VAR = "AWS_BEARER_TOKEN_BEDROCK"


class JudgeResult(BaseModel):
    violation_type: ViolationType
    verdict: Literal["TP", "FP", "uncertain"]
    reason: str
    confidence: Literal["high", "medium", "low"]
    # The model's own numeric probability estimate (0.0-1.0) that its
    # verdict is correct, alongside the categorical `confidence` word above
    # — additive, optional (`None` when a provider/model doesn't return it,
    # e.g. older recorded fixtures or a local model that ignores the
    # instruction) so every existing caller that only reads
    # verdict/confidence/explanation/reason is unaffected.
    confidence_score: float | None = None
    # Where the judge says the helmet is, when it clears a helmet detection
    # as "helmet_worn" — [x1, y1, x2, y2] normalized to the image the judge
    # was shown (see judge_prompt.HELMET_BOX_SCHEMA). `None` for every other
    # violation type and verdict, and for any model or recorded fixture that
    # doesn't return it, so callers reading only the verdict are unaffected.
    helmet_box: list[float] | None = None
    explanation: str

    @field_validator("confidence_score")
    @classmethod
    def _validate_confidence_score_range(cls, value: float | None) -> float | None:
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"confidence_score must be between 0 and 1, got {value}")
        return value

    @field_validator("helmet_box")
    @classmethod
    def _drop_unusable_helmet_box(cls, value: list[float] | None) -> list[float] | None:
        """Discard a malformed box instead of raising.

        Unlike confidence_score, this field decorates a verdict rather than
        carrying one: it exists so a reviewer can see what the judge looked
        at. Raising would send the whole case back through
        `_call_and_validate_with_retry` — re-judging a sound verdict, at
        full token cost, over coordinates that only drive a rectangle. A
        judge that returns nonsense here loses the rectangle; the verdict,
        reason and explanation stand on their own.
        """
        if value is None:
            return None
        if len(value) != 4 or not all(isinstance(v, (int, float)) for v in value):
            return None
        x1, y1, x2, y2 = (float(v) for v in value)
        if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)):
            return None
        # Degenerate or inverted — nothing drawable.
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @model_validator(mode="after")
    def _helmet_box_belongs_to_helmet_only(self) -> JudgeResult:
        """Drop a helmet box reported against any other violation type.

        `build_output_schema` only offers the field on the helmet variant, so
        a schema-constrained provider couldn't produce this — but Bedrock's
        chat-completions path is only steered by a schema pasted into the
        prompt, and a model that free-associates a box onto a glove verdict
        would have a green "helmet (judge)" rectangle drawn on a glove crop.
        Enforced here rather than at each draw site so no future caller has
        to remember the rule.
        """
        if self.helmet_box is not None and self.violation_type is not ViolationType.HELMET:
            object.__setattr__(self, "helmet_box", None)
        return self


class ReasoningAlignmentScore(BaseModel):
    """1-5 rubric score for how well an LLM judge's `explanation` aligns
    with a ground-truth reviewer's rationale about the same case — see
    `JudgeClient.score_reasoning` and judge_prompt.REASONING_ALIGNMENT_SYSTEM_PROMPT.
    Grades agreement on the underlying reasoning, not whether the two
    verdicts match."""

    score: int = Field(strict=True)
    critique: str

    @field_validator("score")
    @classmethod
    def _validate_score_range(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError(f"score must be 1-5, got {value}")
        return value


class JudgeResponse(BaseModel):
    person_id: str
    results: list[JudgeResult]

    @field_validator("person_id", mode="before")
    @classmethod
    def _coerce_person_id_to_str(cls, value: object) -> object:
        # Local models occasionally echo person_id back as a bare JSON
        # number (e.g. `"person_id": 1`) instead of a quoted string, even
        # when the prompt shows it quoted — coerce rather than reject,
        # since the value is just an echoed label, not something whose
        # numeric-vs-string type carries meaning here.
        return str(value) if isinstance(value, (int, float)) else value


# Qwen (and most VL models with a grounding head) emit box coordinates as
# integers normalized to 0-1000. A model asked for 0-1 fractions answers in
# 0-1000 anyway, so both have to be decoded — the magnitudes say which is
# which unambiguously, since a fraction can never exceed 1.
_THOUSANDTHS_SCALE = 1000.0


@dataclass
class JudgeUsage:
    """Token accounting for one judge call — the basis for the eval
    harness's cropped-vs-uncropped and cost comparisons."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: JudgeUsage) -> JudgeUsage:
        return JudgeUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=self.cache_creation_input_tokens + other.cache_creation_input_tokens,
            cache_read_input_tokens=self.cache_read_input_tokens + other.cache_read_input_tokens,
        )


class PersonJudgeResponse(BaseModel):
    """One person's answer inside a batched reply."""

    person_label: str
    #: The model's own description of THIS crop. The only in-band evidence
    #: that the label was matched to the right picture: a transposed answer
    #: usually describes a different scene.
    crop_summary: str = ""
    results: list[JudgeResult]


class BatchJudgeResponse(BaseModel):
    people: list[PersonJudgeResponse]


@dataclass
class BatchJudgeCallResult:
    """A batched call's answers, keyed by the label that was SENT.

    Keyed, not ordered. A caller must look its crop up by label and must never
    fall back to position: attaching a correct answer to the wrong crop is the
    one failure this whole design is organised around, and positional recovery
    is exactly how it happens.
    """

    by_label: dict[str, JudgeResponse]
    summaries: dict[str, str]
    usage: JudgeUsage
    #: Labels that were sent and came back with no entry. Their crops are
    #: failures, never verdicts.
    missing: list[str]


@dataclass
class JudgeCallResult:
    response: JudgeResponse
    usage: JudgeUsage


class SupportsJudge(Protocol):
    def judge(
        self,
        image_bytes: bytes,
        media_type: str,
        person_id: str,
        violation_types: list[ViolationType],
        extra_context: str | None = None,
    ) -> JudgeCallResult: ...


# --- provider-neutral -> wire-format adapters -------------------------------
# few_shot.build_few_shot_messages() returns messages with content blocks
# shaped {"type": "text", "text": ...} | {"type": "image", "media_type":
# ..., "data": <base64>}, independent of which provider ends up calling it.


def _apply_uncertain_overrides(response: JudgeResponse, config: PipelineConfig) -> JudgeResponse:
    """Resolve "uncertain" verdicts to "FP" for whichever violation types
    `config.judge.uncertain_as_fp` lists — applied once here so every
    caller (the live pipeline's verdict bucketing, the eval harness's
    scoring) sees the resolved verdict uniformly, regardless of provider."""
    override_types = set(config.judge.uncertain_as_fp)
    if not override_types:
        return response
    resolved = [
        r.model_copy(update={"verdict": "FP"}) if r.verdict == "uncertain" and r.violation_type in override_types else r
        for r in response.results
    ]
    return response.model_copy(update={"results": resolved})


def _json_schema_hint(schema: dict) -> str:
    """Appended to the user text: Bedrock's OpenAI-compatible endpoint
    offers only broadly-compatible JSON mode, not schema-constrained
    decoding, so the required shape has to be spelled out in the prompt."""
    return (
        "\n\nRespond with a single JSON object matching exactly this JSON "
        f"Schema (no markdown fencing, no extra text):\n{json.dumps(schema)}"
    )


def _strip_json_fence(text: str) -> str:
    """Some models still wrap their answer in a markdown code fence even
    when `response_format={"type": "json_object"}` is requested. Strip it if
    present so `json.loads` sees bare JSON either way."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.removeprefix("```json").removeprefix("```")
        stripped = stripped.removesuffix("```")
        stripped = stripped.strip()
    return stripped


def _is_retryable_status_code(status_code: int) -> bool:
    return status_code in (408, 409, 429) or status_code >= 500


def _retry_delay_seconds(attempt: int, retry_config: RetryConfig) -> float:
    jitter = 0.5 + random.random()
    delay = retry_config.base_delay_seconds * 2**attempt * jitter
    return min(delay, retry_config.max_delay_seconds)


def _call_with_retry(
    fn: Callable[[], object],
    *,
    connection_error_types: tuple[type[Exception], ...],
    status_error_types: tuple[type[Exception], ...],
    retry_config: RetryConfig,
    sleep: Callable[[float], None],
) -> object:
    """Retries `fn()` — expected to wrap only the raw network `.create(...)`
    call, nothing else — on transient connection errors or retryable status
    errors (see `_is_retryable_status_code`), with exponential + jittered
    backoff (see `_retry_delay_seconds`). Any other exception (e.g. a JSON
    parse or schema validation failure from code outside `fn`) is never
    caught here, by construction — it always propagates on first occurrence."""
    attempt = 0
    while True:
        try:
            return fn()
        except connection_error_types:
            if attempt >= retry_config.max_attempts - 1:
                raise
        except status_error_types as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code is None or not _is_retryable_status_code(status_code):
                raise
            if attempt >= retry_config.max_attempts - 1:
                raise
        sleep(_retry_delay_seconds(attempt, retry_config))
        attempt += 1


# Bounded reroll count for a model returning JSON that fails to parse or
# fails JudgeResponse validation (e.g. a near-miss enum value like
# "dangerous_plain" hallucinated by a smaller/weaker model) — a genuinely
# separate mechanism from `_call_with_retry`'s HTTP-level 503 handling above:
# no config knob, no backoff, since a fresh model call either fixes the
# content noise immediately or it doesn't, and waiting can't help either way.
_VALIDATION_RETRY_MAX_ATTEMPTS = 3


def _call_and_validate_with_retry(fn: Callable[[], object]) -> object:
    """Retries the entire call+parse+validate unit `fn()` — a fresh model
    call each attempt, not just a reparse of the same text — when it raises
    `json.JSONDecodeError` or `pydantic.ValidationError`. Any other exception
    (e.g. a retryable/non-retryable HTTP status error surfaced from inside
    `fn`) propagates immediately, unretried by this wrapper."""
    last_exc: json.JSONDecodeError | ValidationError | None = None
    for _ in range(_VALIDATION_RETRY_MAX_ATTEMPTS):
        try:
            return fn()
        except (json.JSONDecodeError, ValidationError) as exc:
            last_exc = exc
    raise last_exc


def _call_openai_with_retry(
    fn: Callable[[], object], *, retry_config: RetryConfig, sleep: Callable[[float], None]
) -> object:
    return _call_with_retry(
        fn,
        connection_error_types=(openai.APIConnectionError,),
        status_error_types=(openai.APIStatusError,),
        retry_config=retry_config,
        sleep=sleep,
    )


def _to_openai_messages(neutral_messages: list[dict]) -> list[dict]:
    converted = []
    for message in neutral_messages:
        blocks = []
        for block in message["content"]:
            if block["type"] == "image":
                url = f"data:{block['media_type']};base64,{block['data']}"
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            else:
                blocks.append({"type": "text", "text": block["text"]})
        converted.append({"role": message["role"], "content": blocks})
    return converted


def _usage_of(response) -> JudgeUsage:
    """Token usage off one chat completion, in one place.

    Read the same way for every kind of call this client makes, so a cost
    comparison between them is a comparison and not two conventions.
    """
    raw_usage = response.usage
    cached_tokens = 0
    details = getattr(raw_usage, "prompt_tokens_details", None)
    if details is not None:
        cached_tokens = getattr(details, "cached_tokens", 0) or 0
    return JudgeUsage(
        input_tokens=raw_usage.prompt_tokens,
        output_tokens=raw_usage.completion_tokens,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=cached_tokens,
    )


#: The first-pass gate's system prompt, and its THREE-WAY answer.
#:
#: The first version asked a yes/no question and ended "If you cannot tell,
#: say a person IS present", because dropping a bent-over worker loses a
#: violation nothing downstream recovers. Measured on alarm 00e90bfd -- 12
#: detections, 2 real workers -- it answered yes 12 times, with reasons like
#: "blob shape suggests human form possibly bent or obscured" on a 43x43
#: crop. A yes/no question on an unreadable crop can only be answered by
#: guessing, and the tie-break made every guess a yes.
#:
#: So the model is allowed to say it cannot tell. Measured on the same twelve
#: crops, this prompt returns:
#:
#:     p07  182x359  person   "two figures with visible helmets and torso"
#:     the other 11  unclear  "no identifiable body parts", "indistinct"
#:
#: which is exactly the frame's truth: both workers are inside p07's box.
#:
#: `unclear` is NOT `no_person`, and the difference decides two things
#: differently:
#:
#: * JUDGING -- ask the helmet question about `person` AND `unclear`. A crop
#:   too blurred to read may still hold a bent worker, and a missed violation
#:   is not recoverable. Only `no_person` is skipped.
#: * LABELLING -- only `person` may be labelled. A box nobody can make out is
#:   not a training label, and a frame containing one cannot be fully
#:   labelled, so `training_export.build_frame_export` refuses it.
PERSON_GATE_SYSTEM_PROMPT = (
    "You are shown a small crop cut from a CCTV frame at a port. Decide whether it "
    "contains a PERSON you can actually see -- head, limbs, or torso you could point "
    "to. Do NOT answer yes because a blob is roughly upright, roughly person-sized, "
    "or could be a bent-over worker: a container edge, a bollard, a light and a ship "
    "rail all look like that at this size. Answer yes only if you can name what you "
    "see on the person. If the crop is too small or too blurred to make out a body, "
    "that is unclear, not yes."
)

PERSON_GATE_USER_TEXT = (
    'Reply with JSON only: {"answer": "person" | "unclear" | "no_person", '
    '"what_i_see": "<one short clause>"}'
)

#: The answers that let the helmet question be asked. `unclear` is in it on
#: purpose: see the prompt's note.
PERSON_GATE_JUDGE_ANSWERS = ("person", "unclear")
#: The only answer that certifies a box for a training label.
PERSON_GATE_LABEL_ANSWER = "person"


def _sampling(config) -> dict:
    """`temperature` as call kwargs, or nothing when it is not configured.

    Sent on EVERY call this client makes -- the verdict and the reasoning
    score -- because a measurement is only comparable if every call in it was
    sampled the same way. Omitted entirely when `None`, so a provider that
    rejects the parameter still works.
    """
    temperature = getattr(config.judge, "temperature", None)
    return {} if temperature is None else {"temperature": temperature}


class JudgeClient:
    """Thin wrapper over Amazon Bedrock's (`bedrock-mantle`)
    OpenAI-compatible chat completions API.

    Accepts an injected `client` (an `openai.OpenAI` instance) so tests and
    callers that want a different transport don't need to touch this class.
    `provider`/`model`/`max_tokens` default to `config.judge` (config.toml /
    built-in defaults — see config.py) when not given explicitly.

    `provider` survives as a parameter because it still travels on the API
    request payloads and the web UI's judge controls, but "bedrock" is the
    only accepted value — anything else raises rather than silently judging
    over a transport this deployment has no credentials for.
    """

    def __init__(
        self,
        client: OpenAI | None = None,
        provider: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        few_shot_library: FewShotLibrary | None = None,
        config: PipelineConfig | None = None,
        retry_sleep: Callable[[float], None] | None = None,
    ):
        config = config or get_config()
        self._provider = provider if provider is not None else config.judge.provider
        self._model = model if model is not None else config.judge.model
        self._max_tokens = max_tokens if max_tokens is not None else config.judge.max_tokens
        self._few_shot_library = few_shot_library or {}
        self._config = config
        self._retry_sleep = retry_sleep or time.sleep

        if self._provider != BEDROCK_PROVIDER:
            raise ValueError(
                f"unsupported judge provider {self._provider!r}; "
                f"{BEDROCK_PROVIDER!r} is the only provider this deployment supports"
            )

        if client is not None:
            self._client = client
        else:
            region = self._config.judge.bedrock.region
            self._client = OpenAI(
                api_key=os.environ.get(BEDROCK_API_KEY_ENV_VAR),
                base_url=f"https://bedrock-mantle.{region}.api.aws/v1",
            )

    def judge(
        self,
        image_bytes: bytes,
        media_type: str,
        person_id: str,
        violation_types: list[ViolationType],
        extra_context: str | None = None,
    ) -> JudgeCallResult:
        schema = build_output_schema(violation_types)
        few_shot_messages = build_few_shot_messages(
            self._few_shot_library, violation_types, config=self._config
        )

        b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")
        result = self._judge_bedrock(
            b64_image, media_type, person_id, violation_types, schema, few_shot_messages, extra_context
        )

        result.response = _apply_uncertain_overrides(result.response, self._config)
        return result

    def judge_batch(
        self,
        crops: list[tuple[str, bytes, str]],
        violation_types: list[ViolationType],
        extra_context: str | None = None,
        max_tokens: int | None = None,
    ) -> BatchJudgeCallResult:
        """Judge SEVERAL crops of one frame in one call.

        MEASURED AND REJECTED, 1 Sep. Nothing in the pipeline calls this; it
        is kept as the artefact of the measurement, not as a dormant feature,
        and `_judge_one_frame` was deliberately never wired to it.

        Two studies, both through the real judge (50-example prefix,
        temperature 0), against a per-crop control re-run at the same settings:

            10 stratified frames, cap 3
                crop agreement 84.1%, frame verdict 90%, saving 61%
                fired-set match 1/1 -- one violation frame, so worthless

            8 VIOLATION frames, cap 6                      <- the real test
                crop agreement            81.8%
                control vs ITSELF         95.5%   the judge's own floor
                frame verdict             4/8 = 50%
                EXACT fired-set match     0/2 = 0%   against an 85% bar
                saving                    79.5%

        The 95.5% self-agreement is what makes this decisive: the judge is
        stable at 0 temperature, so the ~14 points batching loses are
        batching's. And on violation frames -- the only ones where fired-set
        attribution matters -- it flipped the frame verdict half the time and
        never once reproduced which crop fired.

        That is the failure the deferral predicted: `fired_candidates` feeds
        `training_export.audited_crop_label`, which labels ONLY the fired crop
        on a violation frame, so a wrong fired set writes `violation` onto a
        worker visibly wearing a helmet.

        Agreement did NOT degrade with batch size -- 12-crop frames scored 89%
        and 4-crop frames 50% -- so a smaller cap does not fix it. It tracks
        crop ambiguity, like everything else measured on this footage.

        Do not enable this without re-running both studies on whatever model
        replaces qwen3-vl; the mechanism is sound and the model is not.
        

        `crops` is `(label, image_bytes, media_type)`. The few-shot prefix,
        the system prompt and the schema are sent ONCE for the whole frame --
        which is the entire saving: measured, the prefix is 11,427 tokens of a
        ~14,370-token call and the endpoint does not cache it.

        LABELS ARE VALIDATED AS A SET against what was sent. A label that was
        not sent invalidates the whole response (the retry wrapper rerolls); a
        sent label with no entry comes back in `missing` and its crop is a
        failure. There is deliberately NO positional fallback -- see
        `BatchJudgeCallResult`.
        """
        if not crops:
            return BatchJudgeCallResult({}, {}, JudgeUsage(), [])

        labels = [label for label, _, _ in crops]
        if len(set(labels)) != len(labels):
            raise ValueError("batched crop labels must be unique within a frame")

        schema = build_batch_output_schema(violation_types)
        few_shot_messages = build_few_shot_messages(
            self._few_shot_library, violation_types, config=self._config
        )
        schema_hint = _json_schema_hint(schema) + build_batch_output_schema_labels(labels)

        # LABEL, THEN ITS IMAGE, interleaved in one user message: the label
        # sits in the token stream immediately before the picture it names, so
        # the model does not have to read pixels to know an id.
        parts: list[dict] = []
        for label, image_bytes, media_type in crops:
            b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
            parts.append({"type": "text", "text": f"{label}:"})
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                }
            )
        parts.append(
            {
                "type": "text",
                "text": build_batch_user_text(labels, violation_types, extra_context) + schema_hint,
            }
        )

        sent = set(labels)

        def _call_and_parse() -> tuple[object, BatchJudgeResponse]:
            response = _call_openai_with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    # Its OWN budget. The single-crop path's 2048 is for one
                    # verdict; N verdicts in one reply need more, and raising
                    # the shared value would change every other call.
                    max_tokens=max_tokens or self._config.judge.batch_max_tokens,
                    response_format={"type": "json_object"},
                    **_sampling(self._config),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *_to_openai_messages(few_shot_messages),
                        {"role": "user", "content": parts},
                    ],
                ),
                retry_config=self._config.judge.retry,
                sleep=self._retry_sleep,
            )
            text = response.choices[0].message.content
            parsed = BatchJudgeResponse.model_validate(json.loads(_strip_json_fence(text)))
            unknown = {person.person_label for person in parsed.people} - sent
            if unknown:
                # A label nobody sent means the model invented an identity, so
                # nothing in this reply can be trusted to belong to a crop.
                raise ValueError(f"batched reply used labels that were not sent: {sorted(unknown)}")
            return response, parsed

        response, parsed = _call_and_validate_with_retry(_call_and_parse)

        by_label: dict[str, JudgeResponse] = {}
        summaries: dict[str, str] = {}
        for person in parsed.people:
            by_label[person.person_label] = _apply_uncertain_overrides(
                JudgeResponse(person_id=person.person_label, results=person.results),
                self._config,
            )
            summaries[person.person_label] = person.crop_summary
        missing = [label for label in labels if label not in by_label]
        return BatchJudgeCallResult(
            by_label=by_label,
            summaries=summaries,
            usage=_usage_of(response),
            missing=missing,
        )

    def score_reasoning(
        self,
        violation_type: ViolationType,
        llm_explanation: str,
        ground_truth_rationale: str,
    ) -> ReasoningAlignmentScore:
        """Text-only call, separate from `judge()`'s image-based verdict
        schema: rate 1-5 how well `llm_explanation` (the judge's own stated
        reasoning) aligns with `ground_truth_rationale` (a human gold-set
        reviewer's rationale) about the same underlying facts — see
        judge_prompt.REASONING_ALIGNMENT_SYSTEM_PROMPT. Reuses this client's
        model/auth, since it's the same Bedrock account making a different
        kind of call, not a different transport."""
        user_text = build_reasoning_alignment_user_text(violation_type, llm_explanation, ground_truth_rationale)

        return self._score_reasoning_bedrock(user_text)

    def _score_reasoning_bedrock(self, user_text: str) -> ReasoningAlignmentScore:
        schema_hint = _json_schema_hint(REASONING_ALIGNMENT_SCHEMA)
        response = _call_openai_with_retry(
            lambda: self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                response_format={"type": "json_object"},
                **_sampling(self._config),
                messages=[
                    {"role": "system", "content": REASONING_ALIGNMENT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_text + schema_hint},
                ],
            ),
            retry_config=self._config.judge.retry,
            sleep=self._retry_sleep,
        )
        text = response.choices[0].message.content
        return ReasoningAlignmentScore.model_validate(json.loads(_strip_json_fence(text)))

    def _judge_bedrock(
        self,
        b64_image: str,
        media_type: str,
        person_id: str,
        violation_types: list[ViolationType],
        schema: dict,
        few_shot_messages: list[dict],
        extra_context: str | None = None,
    ) -> JudgeCallResult:
        # Bedrock's OpenAI-compatible endpoint does not support strict
        # JSON-schema-constrained output — request the broadly-supported
        # JSON mode instead, and spell out the required shape in the
        # prompt text so the model still has it to aim for.
        schema_hint = _json_schema_hint(schema)

        def _call_and_parse() -> tuple[object, JudgeResponse]:
            response = _call_openai_with_retry(
                lambda: self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    response_format={"type": "json_object"},
                    **_sampling(self._config),
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        *_to_openai_messages(few_shot_messages),
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:{media_type};base64,{b64_image}"},
                                },
                                {
                                    "type": "text",
                                    "text": build_user_text(person_id, violation_types, extra_context) + schema_hint,
                                },
                            ],
                        },
                    ],
                ),
                retry_config=self._config.judge.retry,
                sleep=self._retry_sleep,
            )
            text = response.choices[0].message.content
            return response, JudgeResponse.model_validate(json.loads(_strip_json_fence(text)))

        response, parsed = _call_and_validate_with_retry(_call_and_parse)

        return JudgeCallResult(response=parsed, usage=_usage_of(response))
