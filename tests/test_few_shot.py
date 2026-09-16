import base64
import io
import json

import pytest
from botocore.exceptions import ClientError
from PIL import Image

from ppe_pipeline import few_shot
from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.few_shot import (
    FewShotExample,
    _example_to_turns,
    add_example_to_library,
    build_few_shot_messages,
    load_few_shot_library,
)
from ppe_pipeline.models import ViolationType
from ppe_pipeline.storage import S3Storage


def _write_manifest(tmp_path):
    for name in ["glove_01.jpg", "glove_02.jpg", "helmet_01.jpg"]:
        Image.new("RGB", (32, 32), color="blue").save(tmp_path / name)

    manifest = {
        "glove": [
            {
                "image": "glove_01.jpg",
                "verdict": "TP",
                "reason": "clear_bare_hand",
                "confidence": "high",
                "explanation": "Bare skin clearly visible on both hands.",
            },
            {
                "image": "glove_02.jpg",
                "verdict": "FP",
                "reason": "gloved",
                "confidence": "high",
                "explanation": "Thin skin-toned glove, texture visible under zoom.",
            },
        ],
        "helmet": [
            {
                "image": "helmet_01.jpg",
                "verdict": "TP",
                "reason": "bare_head",
                "confidence": "medium",
                "explanation": "No headwear of any kind visible.",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path


def test_load_few_shot_library_resolves_images_relative_to_manifest(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    library = load_few_shot_library(manifest_path)

    assert set(library.keys()) == {ViolationType.GLOVE, ViolationType.HELMET}
    assert len(library[ViolationType.GLOVE]) == 2
    assert library[ViolationType.GLOVE][0].image_path == tmp_path / "glove_01.jpg"
    assert library[ViolationType.GLOVE][1].verdict == "FP"


def test_invalid_reason_code_is_rejected(tmp_path):
    bad_manifest = {
        "glove": [
            {
                "image": "glove_01.jpg",
                "verdict": "TP",
                "reason": "not_a_real_reason_code",
                "confidence": "high",
                "explanation": "x",
            }
        ]
    }
    Image.new("RGB", (32, 32)).save(tmp_path / "glove_01.jpg")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(bad_manifest))

    with pytest.raises(ValueError, match="not_a_real_reason_code"):
        load_few_shot_library(manifest_path)


def test_build_few_shot_messages_only_includes_requested_violation_types(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    library = load_few_shot_library(manifest_path)

    messages = build_few_shot_messages(library, [ViolationType.GLOVE])

    # 2 glove examples -> 2 user/assistant pairs -> 4 messages; helmet's
    # single example must not appear since it wasn't requested.
    assert len(messages) == 4
    assert all(m["role"] in ("user", "assistant") for m in messages)
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_messages_use_provider_neutral_content_blocks(tmp_path):
    # Cache breakpoints and provider-specific image shapes are added later
    # by judge_client.py's adapters — this module only builds neutral
    # {"type": "text"|"image", ...} blocks.
    manifest_path = _write_manifest(tmp_path)
    library = load_few_shot_library(manifest_path)

    messages = build_few_shot_messages(library, [ViolationType.GLOVE, ViolationType.HELMET])

    user_turn = messages[0]
    assert user_turn["content"][0]["type"] == "image"
    assert "media_type" in user_turn["content"][0]
    assert "data" in user_turn["content"][0]
    assert user_turn["content"][1]["type"] == "text"

    assistant_turn = messages[1]
    assert assistant_turn["content"] == [{"type": "text", "text": assistant_turn["content"][0]["text"]}]
    assert "cache_control" not in assistant_turn["content"][0]


def test_max_per_type_caps_examples_included():
    library = {}
    messages = build_few_shot_messages(library, [ViolationType.GLOVE], max_per_type=0)
    assert messages == []


def test_no_examples_for_type_returns_empty_list():
    messages = build_few_shot_messages({}, [ViolationType.DANGEROUS_PLANE])
    assert messages == []


class _FakeS3ClientForBytes:
    """Minimal fake just for `get_object`, keyed by exact full key, to prove
    a call routes through `storage.get_storage()` rather than a raw
    `Path.read_bytes()`/`Path.read_text()`."""

    def __init__(self, objects: dict[str, bytes]):
        self._objects = objects

    def get_object(self, Bucket, Key):
        if Key not in self._objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "Not Found"}}, "GetObject")
        return {"Body": io.BytesIO(self._objects[Key]), "ETag": '"etag"'}


def test_load_few_shot_library_reads_manifest_through_storage(tmp_path, monkeypatch):
    manifest = {
        "glove": [
            {
                "image": "glove_01.jpg",
                "verdict": "TP",
                "reason": "clear_bare_hand",
                "confidence": "high",
                "explanation": "Bare skin clearly visible on both hands.",
            },
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    # Deliberately never written to local disk — if load_few_shot_library
    # still did a raw Path.read_text() under the hood, this would raise
    # FileNotFoundError instead of reading from the fake S3 backend.
    fake_client = _FakeS3ClientForBytes({manifest_path.as_posix(): json.dumps(manifest).encode("utf-8")})
    fake_backend = S3Storage(bucket="fake-bucket", client=fake_client)
    monkeypatch.setattr(few_shot.storage, "get_storage", lambda: fake_backend)

    library = load_few_shot_library(manifest_path)

    assert set(library.keys()) == {ViolationType.GLOVE}
    assert library[ViolationType.GLOVE][0].image_path == tmp_path / "glove_01.jpg"


def test_example_to_turns_reads_image_bytes_through_storage(tmp_path, monkeypatch):
    image_path = tmp_path / "glove_01.jpg"
    # Deliberately never written to local disk — proves _example_to_turns
    # no longer calls Path.read_bytes() directly.
    fake_client = _FakeS3ClientForBytes({image_path.as_posix(): b"fake-jpeg-bytes"})
    fake_backend = S3Storage(bucket="fake-bucket", client=fake_client)
    monkeypatch.setattr(few_shot.storage, "get_storage", lambda: fake_backend)

    example = FewShotExample(
        violation_type=ViolationType.GLOVE,
        image_path=image_path,
        verdict="TP",
        reason="clear_bare_hand",
        confidence="high",
        explanation="Bare skin clearly visible on both hands.",
    )

    user_turn, _assistant_turn = _example_to_turns(example)

    assert base64.standard_b64decode(user_turn["content"][0]["data"]) == b"fake-jpeg-bytes"


def test_add_example_to_library_appends_to_manifest_and_annotations():
    manifest: dict = {}
    annotations: dict = {}

    add_example_to_library(
        manifest, annotations, ViolationType.HELMET, "crops/helmet_01.jpg",
        "TP", "bare_head", "high", "No headwear visible.",
    )

    assert manifest["helmet"] == [
        {
            "image": "crops/helmet_01.jpg",
            "verdict": "TP",
            "reason": "bare_head",
            "confidence": "high",
            "explanation": "No headwear visible.",
        }
    ]
    assert annotations["helmet_01.jpg"] == {
        "reason": "bare_head", "confidence": "high", "explanation": "No headwear visible.",
    }


def test_add_example_to_library_appends_to_existing_entries_for_same_type():
    manifest = {"helmet": [{"image": "crops/existing.jpg", "verdict": "FP", "reason": "helmet_worn",
                             "confidence": "high", "explanation": "already here"}]}
    annotations: dict = {}

    add_example_to_library(
        manifest, annotations, ViolationType.HELMET, "crops/new_one.jpg",
        "TP", "bare_head", "high", "New example.",
    )

    assert len(manifest["helmet"]) == 2
    assert manifest["helmet"][1]["image"] == "crops/new_one.jpg"


def test_max_examples_per_type_override_takes_priority_over_global_default(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    library = load_few_shot_library(manifest_path)
    config = PipelineConfig()
    config.few_shot.max_examples_per_type_override = {"glove": 1}

    messages = build_few_shot_messages(library, [ViolationType.GLOVE], config=config)

    # 1 glove example -> 1 user/assistant pair -> 2 messages, even though
    # the manifest has 2 glove examples and the global default is 5.
    assert len(messages) == 2


def test_max_examples_per_type_override_does_not_affect_other_types(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    library = load_few_shot_library(manifest_path)
    config = PipelineConfig()
    config.few_shot.max_examples_per_type_override = {"helmet": 10}

    messages = build_few_shot_messages(library, [ViolationType.GLOVE], config=config)

    # helmet's override shouldn't change glove's own behavior (global
    # default of 5, capped by the manifest's 2 actual glove examples).
    assert len(messages) == 4


def test_explicit_max_per_type_still_overrides_the_config_override(tmp_path):
    manifest_path = _write_manifest(tmp_path)
    library = load_few_shot_library(manifest_path)
    config = PipelineConfig()
    config.few_shot.max_examples_per_type_override = {"glove": 5}

    messages = build_few_shot_messages(library, [ViolationType.GLOVE], max_per_type=1, config=config)

    assert len(messages) == 2


# --- order decides what a limit keeps ------------------------------------


def _numbered_library(count: int, tmp_path):
    """`count` helmet examples whose explanations are their own index, so the
    order that reaches the prompt is readable from the messages."""
    from ppe_pipeline.models import ViolationType

    return {
        ViolationType.HELMET: [
            FewShotExample(
                violation_type=ViolationType.HELMET,
                image_path=tmp_path / "helmet_01.jpg",
                verdict="TP",
                reason="bare_head",
                confidence="high",
                explanation=f"example {index}",
            )
            for index in range(count)
        ]
    }


def _explanations(messages):
    return [
        part["text"]
        for message in messages
        if message["role"] == "assistant"
        for part in ([{"text": message["content"]}] if isinstance(message["content"], str) else message["content"])
    ]


def test_examples_can_be_rotated_to_the_back_before_the_limit(tmp_path):
    """CEO, 1 Sep: "Push the first 10 few shot prompt to the back, then limit
    few shot to first 50." Truncation is from the FRONT, so rotating is how a
    limit is made to drop the leading examples -- helmet's list leads with the
    10 curated ones and the 62 balanced ones sit behind them."""
    from ppe_pipeline.config import PipelineConfig
    from ppe_pipeline.models import ViolationType

    Image.new("RGB", (32, 32), color="blue").save(tmp_path / "helmet_01.jpg")
    config = PipelineConfig()
    config.few_shot.max_examples_per_type_override = {"helmet": 4}
    config.few_shot.move_to_back_override = {"helmet": 2}

    messages = build_few_shot_messages(
        _numbered_library(6, tmp_path), [ViolationType.HELMET], config=config
    )

    # 0 and 1 rotated behind 2..5, then the limit of 4 keeps 2,3,4,5.
    assert len(messages) == 8
    body = " ".join(str(m["content"]) for m in messages)
    for kept in ("example 2", "example 3", "example 4", "example 5"):
        assert kept in body
    for dropped in ("example 0", "example 1"):
        assert dropped not in body


def test_no_rotation_leaves_the_manifest_order_alone(tmp_path):
    from ppe_pipeline.config import PipelineConfig
    from ppe_pipeline.models import ViolationType

    Image.new("RGB", (32, 32), color="blue").save(tmp_path / "helmet_01.jpg")
    config = PipelineConfig()
    config.few_shot.max_examples_per_type_override = {"helmet": 2}

    messages = build_few_shot_messages(
        _numbered_library(4, tmp_path), [ViolationType.HELMET], config=config
    )

    body = " ".join(str(m["content"]) for m in messages)
    assert "example 0" in body and "example 1" in body
    assert "example 2" not in body


def test_the_deployment_shows_every_helmet_example():
    """The shipped decision: all 72, after a 50 cap was found to be dropping
    the 10 curated examples and helmets were being missed. `move_to_back`
    still applies but at a limit of 72 it only changes order, not content."""
    from pathlib import Path as _Path

    from ppe_pipeline.config import get_config
    from ppe_pipeline.few_shot import build_few_shot_messages, load_few_shot_library
    from ppe_pipeline.models import ViolationType

    config = get_config()
    assert config.few_shot.max_examples_per_type_override["helmet"] == 72

    library = load_few_shot_library(_Path("few_shot/manifest.json"))
    shown = build_few_shot_messages(library, [ViolationType.HELMET], config=config)
    # Every example in the library reaches the prompt: one user turn and one
    # assistant turn each.
    assert len(shown) == 2 * len(library[ViolationType.HELMET])
