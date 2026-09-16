import json
from dataclasses import dataclass, field
from typing import Callable

import pytest

from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.judge_client import (
    JudgeClient,
    JudgeResult,
    _to_openai_messages,
)
from ppe_pipeline.models import ViolationType

VALID_RESPONSE_TEXT = json.dumps(
    {
        "person_id": "1",
        "results": [
            {
                "violation_type": "helmet",
                "verdict": "TP",
                "reason": "bare_head",
                "confidence": "high",
                "explanation": "x",
            }
        ],
    }
)


# --- fakes shaped like the real anthropic.Anthropic() response surface -----


# --- fakes shaped like the real openai.OpenAI() response surface -----------
# Bedrock is reached over its OpenAI-compatible `bedrock-mantle`
# chat-completions endpoint, so the openai SDK's response shape is the
# only one the judge client sees.


@dataclass
class FakeOpenAIMessage:
    content: str


@dataclass
class FakeOpenAIChoice:
    message: FakeOpenAIMessage


@dataclass
class FakePromptTokensDetails:
    cached_tokens: int


@dataclass
class FakeOpenAIUsage:
    prompt_tokens: int
    completion_tokens: int
    prompt_tokens_details: FakePromptTokensDetails | None = None


@dataclass
class FakeOpenAIResponse:
    choices: list
    usage: FakeOpenAIUsage


class FakeCompletions:
    def __init__(
        self,
        response_text=None,
        usage=None,
        fail_times: int = 0,
        exception_factory: Callable[[], Exception] | None = None,
        response_texts: list[str] | None = None,
    ):
        # `response_texts`, if given, is a per-call sequence of response
        # bodies (the last entry repeats for any further call) — used to
        # simulate a model returning malformed/hallucinated content on its
        # first attempt(s) and a valid body on a later one, distinct from
        # `fail_times`/`exception_factory`, which simulate a transport-level
        # failure (no response body at all).
        self._response_text = response_text
        self._response_texts = response_texts
        self._usage = usage
        self._fail_times = fail_times
        self._exception_factory = exception_factory
        self.last_kwargs = None
        self.call_count = 0

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.call_count < self._fail_times:
            self.call_count += 1
            raise self._exception_factory()
        if self._response_texts is not None:
            index = min(self.call_count, len(self._response_texts) - 1)
            text = self._response_texts[index]
        else:
            text = self._response_text
        self.call_count += 1
        return FakeOpenAIResponse(choices=[FakeOpenAIChoice(message=FakeOpenAIMessage(content=text))], usage=self._usage)


@dataclass
class FakeChat:
    completions: FakeCompletions


class FakeOpenAIClient:
    def __init__(
        self,
        response_text=None,
        usage=None,
        fail_times: int = 0,
        exception_factory: Callable[[], Exception] | None = None,
        response_texts: list[str] | None = None,
    ):
        self.chat = FakeChat(
            FakeCompletions(
                response_text,
                usage,
                fail_times=fail_times,
                exception_factory=exception_factory,
                response_texts=response_texts,
            )
        )


# --- adapter unit tests ------------------------------------------------------


def test_to_openai_messages_converts_image_blocks_to_data_uri():
    neutral = [{"role": "user", "content": [{"type": "image", "media_type": "image/png", "data": "BBB"}]}]
    converted = _to_openai_messages(neutral)
    assert converted[0]["content"][0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,BBB"}}


# --- JudgeClient dispatch tests ----------------------------------------------


def test_judge_client_passes_extra_context_into_user_text():
    usage = FakeOpenAIUsage(prompt_tokens=120, completion_tokens=40)
    fake_client = FakeOpenAIClient(VALID_RESPONSE_TEXT, usage)
    judge = JudgeClient(client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct")

    judge.judge(
        image_bytes=b"fakejpeg",
        media_type="image/jpeg",
        person_id="1",
        violation_types=[ViolationType.HELMET],
        extra_context="Geometric check note: this person's bounding box is 92% contained.",
    )

    user_message = fake_client.chat.completions.last_kwargs["messages"][-1]
    text_block = next(b for b in user_message["content"] if b["type"] == "text")
    assert "Geometric check note: this person's bounding box is 92% contained." in text_block["text"]


def test_judge_client_strips_markdown_json_fence():
    # JSON mode is a request, not a guarantee — a model can still wrap its
    # answer in a ```json fence even when json_object mode is asked for.
    fenced_text = f"```json\n{VALID_RESPONSE_TEXT}\n```"
    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50)
    fake_client = FakeOpenAIClient(fenced_text, usage)
    judge = JudgeClient(client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct")

    result = judge.judge(
        image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
    )

    assert result.response.person_id == "1"
    assert result.response.results[0].verdict == "TP"


def test_judge_client_without_cached_tokens_defaults_to_zero():
    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50, prompt_tokens_details=None)
    fake_client = FakeOpenAIClient(VALID_RESPONSE_TEXT, usage)
    judge = JudgeClient(client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct")

    result = judge.judge(image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET])
    assert result.usage.cache_read_input_tokens == 0


def test_judge_client_bedrock_provider_calls_chat_completions():
    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50, prompt_tokens_details=FakePromptTokensDetails(cached_tokens=30))
    fake_client = FakeOpenAIClient(VALID_RESPONSE_TEXT, usage)
    judge = JudgeClient(client=fake_client, provider="bedrock", model="qwen.qwen3-vl-235b-a22b-instruct")

    result = judge.judge(
        image_bytes=b"fakejpeg", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
    )

    assert result.response.person_id == "1"
    assert result.usage.input_tokens == 200
    assert result.usage.output_tokens == 50
    assert result.usage.cache_read_input_tokens == 30

    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs["model"] == "qwen.qwen3-vl-235b-a22b-instruct"
    assert kwargs["response_format"] == {"type": "json_object"}

    user_message = kwargs["messages"][-1]
    text_block = next(b for b in user_message["content"] if b["type"] == "text")
    assert "JSON Schema" in text_block["text"]
    image_block = next(b for b in user_message["content"] if b["type"] == "image_url")
    assert image_block["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_judge_client_rejects_any_provider_other_than_bedrock():
    """The direct-Anthropic, OpenRouter and on-GPU `local` providers were
    removed. A stale `provider` on a request payload or config must fail
    loudly rather than quietly falling through to Bedrock."""
    import pytest

    usage = FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1)
    for stale in ("anthropic", "openrouter", "local"):
        with pytest.raises(ValueError, match="unsupported judge provider"):
            JudgeClient(
                client=FakeOpenAIClient(VALID_RESPONSE_TEXT, usage),
                provider=stale,
                model="whatever",
            )


def test_judge_client_bedrock_provider_builds_base_url_from_region_config(monkeypatch):
    from ppe_pipeline.config import PipelineConfig

    # The openai SDK refuses to construct a client with no credentials
    # anywhere (env var or explicit arg) — set a fake one so this test
    # doesn't depend on AWS_BEARER_TOKEN_BEDROCK already being set in the
    # ambient environment (e.g. it isn't in CI, only in a local .env).
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-bedrock-token")

    # Deliberately not "us-west-2" (BedrockConfig's own default) — using a
    # different region than the default is what actually proves this reads
    # config.judge.bedrock.region instead of a hardcoded literal.
    config = PipelineConfig()
    config.judge.bedrock.region = "eu-west-1"
    judge = JudgeClient(provider="bedrock", model="qwen.qwen3-vl-235b-a22b-instruct", config=config)

    assert str(judge._client.base_url) == "https://bedrock-mantle.eu-west-1.api.aws/v1/"


def test_reasoning_alignment_score_rejects_non_strict_int():
    import pytest
    from pydantic import ValidationError

    from ppe_pipeline.judge_client import ReasoningAlignmentScore

    with pytest.raises(ValidationError):
        ReasoningAlignmentScore(score=True, critique="x")
    with pytest.raises(ValidationError):
        ReasoningAlignmentScore(score=3.0, critique="x")

    assert ReasoningAlignmentScore(score=3, critique="x").score == 3


def test_score_reasoning_routes_through_chat_completions():
    score_text = json.dumps({"score": 4, "critique": "Mostly aligned."})
    usage = FakeOpenAIUsage(prompt_tokens=50, completion_tokens=10)
    fake_client = FakeOpenAIClient(score_text, usage)
    judge = JudgeClient(client=fake_client, provider="bedrock", model="qwen.qwen3-vl-235b-a22b-instruct")

    result = judge.score_reasoning(ViolationType.HELMET, "bare head observed", "reviewer saw a bare head")

    assert result.score == 4
    assert result.critique == "Mostly aligned."

    kwargs = fake_client.chat.completions.last_kwargs
    assert kwargs["response_format"] == {"type": "json_object"}


def test_judge_client_defaults_provider_and_model_from_config():
    from ppe_pipeline.config import PipelineConfig

    usage = FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1)
    fake_client = FakeOpenAIClient(VALID_RESPONSE_TEXT, usage)
    config = PipelineConfig()
    judge = JudgeClient(client=fake_client, config=config)

    judge.judge(image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET])
    assert fake_client.chat.completions.last_kwargs["model"] == config.judge.model


# --- uncertain_as_fp override -------------------------------------------------


def _uncertain_response_text(violation_type: str) -> str:
    return json.dumps(
        {
            "person_id": "1",
            "results": [
                {
                    "violation_type": violation_type,
                    "verdict": "uncertain",
                    "reason": "obscured_hand" if violation_type == "glove" else "head_occluded",
                    "confidence": "low",
                    "explanation": "x",
                }
            ],
        }
    )


def test_uncertain_glove_verdict_is_resolved_to_fp_by_default():
    from ppe_pipeline.config import PipelineConfig

    usage = FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1)
    fake_client = FakeOpenAIClient(_uncertain_response_text("glove"), usage)
    judge = JudgeClient(client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct", config=PipelineConfig())

    result = judge.judge(
        image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.GLOVE]
    )

    assert result.response.results[0].verdict == "FP"


def test_uncertain_helmet_verdict_is_left_alone_by_default():
    from ppe_pipeline.config import PipelineConfig

    usage = FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1)
    fake_client = FakeOpenAIClient(_uncertain_response_text("helmet"), usage)
    judge = JudgeClient(client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct", config=PipelineConfig())

    result = judge.judge(
        image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
    )

    assert result.response.results[0].verdict == "uncertain"


def test_uncertain_as_fp_override_disabled_when_config_list_is_empty():
    from ppe_pipeline.config import PipelineConfig

    usage = FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1)
    fake_client = FakeOpenAIClient(_uncertain_response_text("glove"), usage)
    config = PipelineConfig()
    config.judge.uncertain_as_fp = []
    judge = JudgeClient(client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct", config=config)

    result = judge.judge(
        image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.GLOVE]
    )

    assert result.response.results[0].verdict == "uncertain"


# --- Bedrock/AWS 503 outer retry layer ---------------------------------------


def _status_error_factory(exception_cls, status_code: int, message: str = "boom") -> Callable[[], Exception]:
    import httpx

    def factory() -> Exception:
        response = httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://example.test"))
        return exception_cls(message, response=response, body=None)

    return factory


class FakeSleep:
    def __init__(self):
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def test_judge_client_retries_transient_503_then_succeeds():
    """Bedrock's on-demand-capacity 503s need tens of seconds to clear — the
    outer retry layer must ride them out rather than failing the run."""
    import openai

    from ppe_pipeline.config import PipelineConfig

    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50)
    fake_client = FakeOpenAIClient(
        VALID_RESPONSE_TEXT,
        usage,
        fail_times=2,
        exception_factory=_status_error_factory(openai.InternalServerError, 503),
    )
    fake_sleep = FakeSleep()
    judge = JudgeClient(
        client=fake_client,
        model="qwen.qwen3-vl-235b-a22b-instruct",
        retry_sleep=fake_sleep,
    )

    result = judge.judge(
        image_bytes=b"fakejpeg", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
    )

    assert result.response.person_id == "1"
    assert fake_client.chat.completions.call_count == 3
    assert len(fake_sleep.calls) == 2

    # Delays are jittered (0.5x-1.5x of base * 2**attempt), so their ranges
    # overlap between attempts — assert each falls in its own expected
    # pre-cap range rather than asserting a strict ordering between them.
    base_delay_seconds = PipelineConfig().judge.retry.base_delay_seconds
    for attempt, delay in enumerate(fake_sleep.calls):
        expected = base_delay_seconds * 2**attempt
        assert expected * 0.5 <= delay <= expected * 1.5


def test_judge_client_gives_up_after_exhausting_max_attempts():
    import openai
    import pytest

    from ppe_pipeline.config import PipelineConfig

    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50)
    fake_client = FakeOpenAIClient(
        VALID_RESPONSE_TEXT,
        usage,
        fail_times=10,
        exception_factory=_status_error_factory(openai.InternalServerError, 503),
    )
    fake_sleep = FakeSleep()
    config = PipelineConfig()
    config.judge.retry.max_attempts = 3
    judge = JudgeClient(
        client=fake_client,
        model="qwen.qwen3-vl-235b-a22b-instruct",
        config=config,
        retry_sleep=fake_sleep,
    )

    with pytest.raises(openai.InternalServerError):
        judge.judge(
            image_bytes=b"fakejpeg", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
        )

    assert fake_client.chat.completions.call_count == 3
    assert len(fake_sleep.calls) == 2


def test_judge_client_non_retryable_status_error_raises_immediately():
    """A 400 is the caller's fault, not transient capacity — retrying it just
    burns the retry budget and delays the error."""
    import openai
    import pytest

    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50)
    fake_client = FakeOpenAIClient(
        VALID_RESPONSE_TEXT,
        usage,
        fail_times=1,
        exception_factory=_status_error_factory(openai.BadRequestError, 400),
    )
    fake_sleep = FakeSleep()
    judge = JudgeClient(
        client=fake_client,
        model="qwen.qwen3-vl-235b-a22b-instruct",
        retry_sleep=fake_sleep,
    )

    with pytest.raises(openai.BadRequestError):
        judge.judge(
            image_bytes=b"fakejpeg", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
        )

    assert fake_client.chat.completions.call_count == 1
    assert fake_sleep.calls == []


def test_judge_client_malformed_response_retries_then_raises_after_exhausting_attempts():
    # A response that never parses/validates gets a bounded number of
    # fresh-call rerolls before giving up, since the failure is
    # model-output noise (e.g. a hallucinated near-miss enum value), not
    # something reparsing the same text could ever fix.
    import json as json_module

    import pytest

    from ppe_pipeline.judge_client import _VALIDATION_RETRY_MAX_ATTEMPTS

    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50)
    fake_client = FakeOpenAIClient(response_text="not json at all", usage=usage)
    fake_sleep = FakeSleep()
    judge = JudgeClient(
        client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct", retry_sleep=fake_sleep
    )

    with pytest.raises(json_module.JSONDecodeError):
        judge.judge(
            image_bytes=b"fakejpeg", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
        )

    assert fake_client.chat.completions.call_count == _VALIDATION_RETRY_MAX_ATTEMPTS
    # This retry layer is near-instant rerolls, not backoff — it must never
    # touch the same `retry_sleep` used by the transport-level 503 retry.
    assert fake_sleep.calls == []


def test_judge_client_malformed_response_retries_then_succeeds():
    usage = FakeOpenAIUsage(prompt_tokens=200, completion_tokens=50)
    fake_client = FakeOpenAIClient(
        response_texts=[
            json.dumps({"person_id": "1", "results": [{"violation_type": "dangerous_plain", "verdict": "TP", "reason": "unprotected_at_height", "confidence": "high", "explanation": "x"}]}),
            VALID_RESPONSE_TEXT,
        ],
        usage=usage,
    )
    fake_sleep = FakeSleep()
    judge = JudgeClient(
        client=fake_client, model="qwen.qwen3-vl-235b-a22b-instruct", retry_sleep=fake_sleep
    )

    result = judge.judge(
        image_bytes=b"fakejpeg", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET]
    )

    assert result.response.results[0].verdict == "TP"
    assert fake_client.chat.completions.call_count == 2
    assert fake_sleep.calls == []


def test_judge_client_reads_retry_config_from_config_judge_retry():
    # Deliberately not the RetryConfig defaults — using different values than
    # the default is what actually proves this reads config.judge.retry
    # instead of hardcoded constants (mirrors
    # test_judge_client_bedrock_provider_builds_base_url_from_region_config).
    import openai
    import pytest

    from ppe_pipeline.config import PipelineConfig

    config = PipelineConfig()
    config.judge.retry.max_attempts = 4
    config.judge.retry.base_delay_seconds = 0.01
    config.judge.retry.max_delay_seconds = 0.02

    usage = FakeOpenAIUsage(prompt_tokens=1, completion_tokens=1)
    fake_client = FakeOpenAIClient(
        VALID_RESPONSE_TEXT,
        usage,
        fail_times=10,
        exception_factory=_status_error_factory(openai.InternalServerError, 503),
    )
    fake_sleep = FakeSleep()
    judge = JudgeClient(
        client=fake_client,
        model="qwen.qwen3-vl-235b-a22b-instruct",
        config=config,
        retry_sleep=fake_sleep,
    )

    with pytest.raises(openai.InternalServerError):
        judge.judge(image_bytes=b"x", media_type="image/jpeg", person_id="1", violation_types=[ViolationType.HELMET])

    assert fake_client.chat.completions.call_count == config.judge.retry.max_attempts
    assert len(fake_sleep.calls) == config.judge.retry.max_attempts - 1
    for delay in fake_sleep.calls:
        assert delay <= config.judge.retry.max_delay_seconds


def test_helmet_box_survives_a_well_formed_box():
    result = JudgeResult(
        violation_type=ViolationType.HELMET,
        verdict="FP",
        reason="helmet_worn",
        confidence="high",
        confidence_score=0.95,
        helmet_box=[0.1, 0.2, 0.5, 0.6],
        explanation="Hard hat with a domed shell.",
    )

    assert result.helmet_box == [0.1, 0.2, 0.5, 0.6]


def test_helmet_box_defaults_to_none_so_older_results_still_parse():
    result = JudgeResult(
        violation_type=ViolationType.HELMET,
        verdict="TP",
        reason="bare_head",
        confidence="high",
        explanation="No head covering.",
    )

    assert result.helmet_box is None


@pytest.mark.parametrize(
    "bad_box",
    [
        [0.5, 0.6, 0.1, 0.2],  # inverted
        [0.1, 0.2, 0.1, 0.6],  # zero width
        [0.1, 0.2, 1.5, 0.6],  # outside the normalized range
        [0.1, 0.2, 0.5],  # too few coordinates
        [0.1, 0.2, 0.3, 0.4, 0.5],  # too many
    ],
)
def test_an_unusable_helmet_box_is_dropped_rather_than_failing_the_verdict(bad_box):
    """A malformed box must not raise: raising would send a sound verdict
    back through the whole call-and-validate retry loop over coordinates
    that only drive a rectangle in the UI."""
    result = JudgeResult(
        violation_type=ViolationType.HELMET,
        verdict="FP",
        reason="helmet_worn",
        confidence="high",
        confidence_score=0.95,
        helmet_box=bad_box,
        explanation="Hard hat with a domed shell.",
    )

    assert result.helmet_box is None
    assert result.verdict == "FP"  # the verdict itself is untouched


def test_a_helmet_box_on_another_violation_type_is_dropped():
    """Only the helmet variant's schema offers the field, but the
    Bedrock path is steered by a prompt-pasted schema, not
    constrained decoding — a box hallucinated onto a glove verdict would
    draw a "helmet (judge)" rectangle on a glove crop."""
    result = JudgeResult(
        violation_type=ViolationType.GLOVE,
        verdict="FP",
        reason="gloved",
        confidence="high",
        confidence_score=0.9,
        helmet_box=[0.1, 0.2, 0.5, 0.6],
        explanation="Both hands gloved.",
    )

    assert result.helmet_box is None
    assert result.verdict == "FP"
