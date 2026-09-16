"""Central, editable configuration for every tunable pipeline parameter.

Edit `config.toml` at the project root to experiment with thresholds,
padding factors, the batching window, etc. — no code changes needed.
Sections/keys missing from the file fall back to the defaults below, so a
partial override file (or no file at all) both work.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .models import ViolationType

# Loaded once, at import time — picks up AWS_BEARER_TOKEN_BEDROCK (and the
# PPE_* settings) from a .env file at (or above) the current working
# directory, if present.
# Never overwrites a variable already set in the real environment.
load_dotenv()

CONFIG_PATH_ENV_VAR = "PPE_PIPELINE_CONFIG"
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.toml"


class MergeConfig(BaseModel):
    # Containment-ratio (intersection / smaller-box-area) threshold for
    # treating two detector boxes as the same physical person.
    overlap_threshold: float = 0.3


class CroppingConfig(BaseModel):
    # Padding factor applied to each violation's own crop box.
    # Glove/Helmet pad the hand/head region box; DangerousPlane and
    # SuspendedLoad pad the full person box (no tight object box available
    # for either); TooNearForklift pads the person+forklift union.
    glove_helmet_pad: float = 0.3
    dangerous_plane_pad: float = 3.5
    suspended_load_pad: float = 3.5
    forklift_union_pad: float = 0.25


class ForkliftRuleConfig(BaseModel):
    # Percentage box expansion and coverage threshold for the geometric
    # proximity proxy, plus the containment ratio above which a person is
    # flagged as a possible forklift operator (and sent to the judge with
    # extra context, rather than excluded).
    expand_pct: float = Field(default=0.2, ge=0.0)
    overlap_threshold: float = Field(default=0.3, ge=0.0)
    operator_containment_threshold: float = Field(default=0.8, ge=0.0)


class BatchingConfig(BaseModel):
    # Session-window gap (seconds): frames from the same camera within
    # this gap of the previous frame are merged into one processing batch.
    gap_seconds: float = 2.0


class FewShotConfig(BaseModel):
    # Max few-shot example crops shown per violation type per judge call.
    max_examples_per_type: int = 5
    # Per-violation-type override of max_examples_per_type above, keyed by
    # ViolationType value (e.g. {"helmet": 10}) — only violation types
    # listed here deviate from the global default; everything else still
    # uses max_examples_per_type.
    max_examples_per_type_override: dict[str, int] = Field(default_factory=dict)
    # Examples moved from the FRONT of a type's list to the BACK, before the
    # limit above is applied. Keyed by ViolationType value, e.g.
    # {"helmet": 10}.
    #
    # CEO, 1 Sep: "Push the first 10 few shot prompt to the back, then limit
    # few shot to first 50." Order matters because `build_few_shot_messages`
    # truncates from the front, so whatever leads the list is what survives a
    # limit -- and helmet's list leads with the 10 curated examples from
    # `scripts/promote_gold_cases_to_fewshot.py`, with the 62 balanced ones
    # behind them.
    #
    # Done here rather than by rewriting the manifest: the manifest records
    # WHERE each example came from, and reordering the file would destroy
    # that. This is a prompt-shaping decision, so it lives in the prompt's
    # configuration.
    move_to_back_override: dict[str, int] = Field(default_factory=dict)


class BedrockConfig(BaseModel):
    region: str = "us-west-2"


class RetryConfig(BaseModel):
    # A longer-patience retry layer stacked OUTSIDE the openai SDK's
    # own built-in retries (which default to max_retries=2,
    # sub-second backoff) — this one exists specifically for AWS Bedrock's
    # on-demand-capacity 503s, which need tens of seconds to clear, not
    # milliseconds. Not a substitute for throttling the eval harness's call
    # rate — that's a separate, deferred concern.
    max_attempts: int = 5
    base_delay_seconds: float = 10.0
    max_delay_seconds: float = 120.0


class HelmetBoxConfig(BaseModel):
    """Stage-2 helmet localization (see `helmet_detector.py`), run only on
    a detection the judge already cleared as "helmet_worn".

    A local single-class YOLO11s detector. It answers *where* the helmet
    is and nothing else — it cannot change a verdict.
    """

    # False falls back to the `helmet_box` field on the verdict itself,
    # which needs no detector but returns markedly looser boxes.
    enabled: bool = True
    # Detector weights. Relative paths resolve against the repo root, so
    # the default works whatever directory the app is started from.
    weights: str = "weights/stage2_helmet_yolo11s_f1ba29e683661e8a.pt"
    # Detections below this score are discarded outright — nothing is
    # drawn and the inline box is used instead.
    #
    # NOT RE-VALIDATED SINCE THE MODEL SWAP. 0.25 was tuned against the
    # YOLOv8s checkpoint this replaced. The two are interface-identical
    # (task=detect, names={0: "helmet"}) so the swap is a clean drop-in, but
    # a different architecture does not necessarily put its scores on the
    # same scale, and this threshold also feeds `review_confidence` below —
    # so a shift changes how many boxes a human is asked to check, not just
    # how many are drawn. Re-measure against a labelled set before trusting
    # either number; deliberately left alone rather than retuned by guess.
    min_confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    # Confidence floor for treating a box as a retraining-label candidate
    # rather than something a human should place. A box below this is
    # still shown and still does NOT affect the verdict — the UI just
    # flags it as likely needing human review.
    review_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Passed to ultralytics `predict(device=...)`. "cpu" is deliberate:
    # the deployed container has no GPU, and torch is pinned to CPU
    # wheels (see pyproject.toml).
    device: str = "cpu"


class HelmetCvConfig(BaseModel):
    """Emulating the production helmet gate over the raw alarm-clip corpus
    (see `helmet_cv/`), so CV and the judge can be scored against one human
    decision on the same frame.

    Nothing here touches live alarms. It reads extracted frames off local
    disk and writes candidate records; the judge pass over those records is
    the ordinary `track1_adapter` path with a `cv` block attached.
    """

    # NO PERSON DETECTOR. Removed 2 Sep: person boxes come from the vendor
    # CV's own annotated snapshots (`helmet_cv/snapshot_boxes.py`), which are
    # the boxes that actually raised each alarm. `weights`, `od_confidence`,
    # `imgsz` and `device` lived here and are gone with it -- rows scored
    # before that date still carry them in their own provenance, which is
    # where a historical setting belongs.
    #
    # `helmet_box` above is UNAFFECTED. It finds helmets, not people.
    # Where `helmet_cv.cli extract` put its frames, and where the CV records
    # and the API's frame server look for them. A local directory, not a
    # storage key — the corpus is ~1.6M jpegs and never goes through
    # `storage`.
    frames_dir: str = "eval/helmet_frames"
    # 1 in N candidates go to the judge, chosen by md5(candidate_id) so the
    # selection is stable across runs and independent of whether the gate
    # fired. At ~200 candidates for the pilot this is set per run from the
    # CLI; the default is "everything" so a small corpus is not silently
    # thinned.
    sample_rate: int = Field(default=1, ge=1)
    # Frames judged concurrently in the judge pass. Threads, not processes:
    # the time goes on waiting for Bedrock. Bounded rather than unbounded
    # because a fan-out at the judge's rate limit turns a slow pass into a
    # failed one, and the work is grouped by frame so this is a count of
    # frames in flight, not of judge calls.
    max_workers: int = Field(default=4, ge=1)
    # Crops of ONE frame judged at once. `max_workers` above counts FRAMES in
    # flight, which does nothing for the live feed: one alarm is one frame, so
    # its crops were judged strictly one after another.
    #
    # MEASURED on the deployed feed, 1 Sep: the first alarm took about four
    # minutes -- 9 judge calls at ~25s each, sequential -- while the clip
    # download and the detector together took 5.1 seconds. The judge calls ARE
    # the wall clock, and they were the one part not running concurrently.
    #
    # Bounded rather than unbounded, for the reason `max_workers` gives: a
    # fan-out at the judge's rate limit turns a slow pass into a failed one.
    # 4 x 4 is 16 calls in flight at most, which is the same ceiling the
    # offline pass has always run at.
    #
    # This saves NO tokens -- it is wall clock only. Batching is the token
    # lever and it is a separate, unshipped decision.
    crop_workers: int = Field(default=4, ge=1)
    # FIRST-PASS PERSON GATE (CEO, 1 Sep). Before the helmet question, ask
    # the VLM the one cheap question "is there a person in this crop at
    # all?", and drop the crop if there is not. The detector runs at 0.25
    # confidence so it finds bent-over workers, and at that threshold it also
    # finds ships and lights: 12 detections and 2 workers on the frame this
    # was measured on.
    #
    # A geometry or confidence filter was measured and REJECTED for this --
    # the two real workers in that frame are bent over, so they are squat
    # too, and one sits below the squat mean confidence. Any threshold that
    # removes the blobs removes them. Asking is the only filter that can
    # tell the difference.
    #
    # Off by default: it changes what reaches the judge, so a run that
    # predates it stays reproducible, and the pilot corpus is not silently
    # rescored. See `judge_client.PERSON_GATE_SYSTEM_PROMPT`.
    # NO FIRST-PASS PERSON GATE. Removed 2 Sep on the CEO's call ("can you
    # remove the part where VLM asked if crop has human in it first? just
    # direct send crops for judge now. i think the bounding box extraction is
    # accurate enough now").
    #
    # It had been OFF since 1 Sep anyway, so removing it saved no calls. What
    # it did do was block the full-frame dataset: `build_frame_export`
    # required every box to carry a gate answer before it would certify a
    # frame, so with the gate off the export shipped NOTHING -- measured on
    # the live corpus, 0 of 37 frames, every one skipped for boxes "not
    # checked for a person".
    #
    # The certification it provided is now provided better. The gate existed
    # because a DETECTOR's boxes might hold a ship or a background light;
    # boxes now come from the vendor CV's own annotation, which is a claim
    # that a person is there, and an auditor can strike an individual crop out
    # of the training data on the audit screen.


class JudgeConfig(BaseModel):
    # Amazon Bedrock's OpenAI-compatible `bedrock-mantle` endpoint
    # (AWS_BEARER_TOKEN_BEDROCK) — not boto3/the Converse API. The only
    # provider: the direct-Anthropic, OpenRouter and on-GPU "local" paths
    # were removed once Bedrock became the sole deployed judge. Kept as a
    # single-valued field rather than deleted because it still travels on
    # the API payloads and the web UI's judge controls.
    provider: Literal["bedrock"] = "bedrock"
    # A bedrock-mantle model ID (e.g. "qwen.qwen3-vl-235b-a22b-instruct")
    # — note this can differ from the bedrock-runtime/Converse model ID
    # for the same model.
    model: str = "qwen.qwen3-vl-235b-a22b-instruct"
    # Response length budget.
    max_tokens: int = 2048
    # Response budget for a BATCHED call, which carries N verdicts in one
    # reply. Its own value, deliberately: raising `max_tokens` above would
    # change the single-crop path, and one verdict with its explanation runs
    # ~150-250 output tokens, so 8.5 crops sit right at the 2048 edge and 17
    # crops -- the pilot's busiest frame -- would truncate every time. A
    # truncated reply is a parse failure, which reads as "the model is bad at
    # batching" rather than as a budget.
    batch_max_tokens: int = 8192
    # Sampling temperature, sent on EVERY judge call.
    #
    # MEASURED 1 Sep, and this is why it exists. Unset, every call ran at the
    # provider default, and the judge asked twice about the SAME crop -- one
    # crop per call, no batching -- answered differently on 3 of 6 crops of an
    # ambiguous frame: TP then FP, FP then uncertain. On an unambiguous frame
    # it was 5 of 5 stable. So the judge is a coin flip precisely where the
    # crop is unreadable, which on this footage is most of them.
    #
    # That floor makes several things unmeasurable rather than merely noisy:
    # any agreement study (batched versus per-crop, prefix size versus
    # accuracy) is measuring sampling noise until this is pinned, and a
    # re-judged corpus cannot be compared with an earlier one.
    #
    # 0.0 because a safety verdict is not a place for creative variety. Note
    # what it does NOT fix: a deterministic wrong answer on an unreadable crop
    # is still wrong. It makes the number reproducible, not correct.
    #
    # None sends no temperature at all, for a provider that rejects it.
    temperature: float | None = 0.0
    # Violation types where an "uncertain" verdict is auto-resolved to "FP"
    # instead of going to human review — for types where an ambiguous crop
    # is judged more likely a false alarm than a missed real violation.
    uncertain_as_fp: list[ViolationType] = Field(default_factory=lambda: [ViolationType.GLOVE])
    bedrock: BedrockConfig = Field(default_factory=BedrockConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    # Starter model list for the UI dropdown, keyed by provider — edit
    # freely; Bedrock's catalog changes over time, and any model ID can
    # still be typed directly via the UI's "custom model" field even if
    # it's not listed here. Still a per-provider dict, not a flat list,
    # because the web UI looks options up by the selected provider.
    model_options: dict[str, list[str]] = Field(
        default_factory=lambda: {
            "bedrock": ["qwen.qwen3-vl-235b-a22b-instruct", "nvidia.nemotron-nano-12b-v2"],
        }
    )


class RewriterConfig(BaseModel):
    """Claude-on-Bedrock settings for `rationale_rewrite`, which normalises
    reviewer rationales into one professional paragraph.

    Separate from `judge` on purpose, and not just for tidiness:
      * The judge reaches Bedrock through the OpenAI-compatible
        `bedrock-mantle` chat-completions endpoint. Bedrock rejects the
        Claude models on that path ("does not support the
        '/v1/chat/completions' API"), so the rewriter must use the Messages
        API client instead.
      * `judge.bedrock.region` is us-west-2, which carries only
        claude-haiku-4-5. The full Claude lineup lives in us-east-1.
      * Keeping the rewriter's model distinct from the judge's means the
        model being evaluated is not also writing the ground-truth reasoning
        it gets scored against.
    """

    region: str = "us-east-1"
    model: str = "anthropic.claude-opus-5"
    max_tokens: int = 1024
    # Rewriting a two-sentence note is not a reasoning-heavy task; "low"
    # keeps the per-save latency and cost down. See the effort scale in the
    # Claude API docs (low | medium | high | xhigh | max).
    #
    # MUST be empty for a model that has no effort control. `output_config`
    # is only sent when this is non-empty, and Bedrock rejects the request
    # outright for a model that does not support it — which, because
    # `rewrite_entry_fields` swallows per-field failures to protect a
    # reviewer's save, would show up as every rewrite silently not
    # happening rather than as an error. claude-haiku-4-5 is the case that
    # matters here: it predates effort and needs `effort = ""`.
    effort: str = "low"
    # Whether a reviewer's save triggers a rewrite at all. Turning this off
    # leaves the human text untouched and defers everything to the backfill
    # script.
    rewrite_on_save: bool = True


class PersonDetectorConfig(BaseModel):
    """Finding people in a clean alarm frame, as an ALTERNATIVE to the
    vendor CV's own boxes (see `person_detector.py`).

    OFF BY DEFAULT in the sense that nothing uses it unless a run asks for
    it: the run screen offers both sources and records which produced each
    row. A detector was removed from this repo on 2 Sep and the reasons are
    in `[helmet_cv]` above; this is a different model answering the same
    question, and which one is better is a measurement nobody has taken.
    """

    #: A SINGLE-CLASS person detector: yolo11m, 60 epochs, trained at imgsz
    #: 1536 -- built for this footage's small workers rather than adapted to
    #: them, which is what the removed checkpoint had been.
    weights: str = "weights/best.pt"
    #: Inference resolution. 1536 because that is what the model was TRAINED
    #: at, and ultralytics' default of 640 would scale this corpus's ~83px
    #: workers to ~35px. Inferring at a different size than training is its
    #: own silent penalty, on top of the one small workers already pay.
    imgsz: int | None = Field(default=1536, ge=32, le=4096)
    #: Deliberately not tuned. Nothing here has been measured against this
    #: model yet, so this is a starting point and every row records it.
    confidence: float = Field(default=0.25, ge=0.0, le=1.0)
    #: No GPU in the deployed container; torch is pinned to CPU wheels. At
    #: imgsz 1536 on a CPU this is the slow option, which is a reason to
    #: measure it rather than switch to it by default.
    device: str = "cpu"


class PipelineConfig(BaseModel):
    merge: MergeConfig = Field(default_factory=MergeConfig)
    cropping: CroppingConfig = Field(default_factory=CroppingConfig)
    forklift: ForkliftRuleConfig = Field(default_factory=ForkliftRuleConfig)
    batching: BatchingConfig = Field(default_factory=BatchingConfig)
    few_shot: FewShotConfig = Field(default_factory=FewShotConfig)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)
    rewriter: RewriterConfig = Field(default_factory=RewriterConfig)
    # Top-level rather than under [judge]: stage-2 helmet localization is
    # a local detector, not a judge call (it was one until the VLM
    # grounding path was replaced -- see helmet_detector.py).
    helmet_box: HelmetBoxConfig = Field(default_factory=HelmetBoxConfig)
    # Its own block rather than nested under helmet_box: that one is stage-2
    # localization on a crop the judge already cleared, this one is an
    # offline emulation of the CV gate that runs BEFORE any judge at all.
    helmet_cv: HelmetCvConfig = Field(default_factory=HelmetCvConfig)
    # Its own block, and not folded into helmet_cv: that block is the offline
    # gate emulation, while this is a live alternative to the vendor's boxes.
    person_detector: PersonDetectorConfig = Field(default_factory=PersonDetectorConfig)


def _resolve_path(path: Path | None) -> Path | None:
    if path is not None:
        return path
    env_path = os.environ.get(CONFIG_PATH_ENV_VAR)
    if env_path:
        return Path(env_path)
    if DEFAULT_CONFIG_PATH.is_file():
        return DEFAULT_CONFIG_PATH
    return None


def load_config(path: Path | None = None) -> PipelineConfig:
    """Load `PipelineConfig`, applying only the sections/keys present in
    the resolved TOML file over the defaults above.

    Resolution order: explicit `path` > `PPE_PIPELINE_CONFIG` env var >
    `config.toml` at the project root > built-in defaults (no file needed).
    """
    resolved = _resolve_path(path)
    if resolved is None:
        return PipelineConfig()
    with resolved.open("rb") as f:
        raw = tomllib.load(f)
    return PipelineConfig.model_validate(raw)


@lru_cache(maxsize=1)
def get_config() -> PipelineConfig:
    """Process-wide cached config — the file is read once per process."""
    return load_config()


def reset_config_cache() -> None:
    """Test helper: drop the cache so the next get_config() call re-reads."""
    get_config.cache_clear()
