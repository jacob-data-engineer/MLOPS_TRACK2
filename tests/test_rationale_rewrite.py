"""Tests for `rationale_rewrite` and the gold-set save-path hook.

No test here reaches the network: every case injects a fake client, the same
way `JudgeClient` tests inject theirs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from ppe_pipeline import rationale_rewrite as rr
from ppe_pipeline import storage
from ppe_pipeline.config import PipelineConfig
from ppe_pipeline.eval import gold_set
from ppe_pipeline.models import ViolationType


class FakeClient:
    """Stands in for AnthropicBedrockMantle. Records every request and
    returns a canned paragraph, so tests assert on what we sent."""

    def __init__(self, text="The worker is not wearing a safety helmet.", stop_reason="end_turn"):
        self.text = text
        self.stop_reason = stop_reason
        self.calls: list[dict] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=None,
            content=[SimpleNamespace(type="text", text=self.text)],
        )


class BrokenClient:
    def __init__(self):
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        raise RuntimeError("bedrock is down")


CONFIG = PipelineConfig()


def test_rewrite_returns_the_models_paragraph():
    client = FakeClient(text="The worker is standing at an unprotected edge.")
    out = rr.rewrite("worker at edge", client=client, config=CONFIG)
    assert out == "The worker is standing at an unprotected edge."


def test_rewrite_sends_the_configured_model_and_effort():
    client = FakeClient()
    rr.rewrite("worker at edge", client=client, config=CONFIG)
    call = client.calls[0]
    assert call["model"] == CONFIG.rewriter.model
    assert call["max_tokens"] == CONFIG.rewriter.max_tokens
    assert call["output_config"] == {"effort": CONFIG.rewriter.effort}


def test_rewrite_omits_output_config_for_a_model_without_effort():
    """A blank effort must send no `output_config` at all.

    Effort is a control on the current Opus/Sonnet line; claude-haiku-4-5
    rejects a request carrying it. Because `rewrite_entry_fields` swallows
    per-field failures, sending it anyway would make every rewrite fail
    silently rather than loudly.
    """
    config = PipelineConfig.model_validate(
        {"rewriter": {"model": "anthropic.claude-haiku-4-5", "effort": ""}}
    )
    client = FakeClient()
    rr.rewrite("worker at edge", client=client, config=config)
    call = client.calls[0]
    assert "output_config" not in call
    assert call["model"] == "anthropic.claude-haiku-4-5"


def test_rewrite_passes_verdict_context_without_the_image():
    client = FakeClient()
    rr.rewrite(
        "no helmet", violation_type=ViolationType.HELMET,
        final_label="violation", reason_code="bare_head",
        client=client, config=CONFIG,
    )
    sent = json.dumps(client.calls[0]["messages"])
    assert "helmet" in sent and "violation" in sent and "bare_head" in sent
    # Text-only by construction: nothing image-shaped is ever sent.
    assert "image" not in sent and "base64" not in sent


def test_rewrite_kind_changes_the_instruction():
    merged, adjudicated = FakeClient(), FakeClient()
    rr.rewrite("a. b.", kind=rr.KIND_MERGED, client=merged, config=CONFIG)
    rr.rewrite("ruling", kind=rr.KIND_ADJUDICATED, client=adjudicated, config=CONFIG)
    assert "two reviewers" in json.dumps(merged.calls[0]["messages"])
    assert "adjudicator" in json.dumps(adjudicated.calls[0]["messages"])


def test_rewrite_refuses_empty_text():
    with pytest.raises(ValueError):
        rr.rewrite("   ", client=FakeClient(), config=CONFIG)


def test_rewrite_raises_on_a_refusal_stop_reason():
    with pytest.raises(RuntimeError, match="refused"):
        rr.rewrite("x", client=FakeClient(stop_reason="refusal"), config=CONFIG)


def test_rewrite_entry_fields_rewrites_all_three_and_records_hashes():
    entry = {
        "reviewer_a_rationale": "no helmet",
        "reviewer_b_rationale": "bare head",
        "final_rationale": "Reviewer A: no helmet. Reviewer B: bare head.",
        "final_label": "violation",
    }
    client = FakeClient()
    updated, done, failed = rr.rewrite_entry_fields(entry, client=client, config=CONFIG)

    assert sorted(done) == ["final_rationale", "reviewer_a_rationale", "reviewer_b_rationale"]
    assert failed == []
    for field in done:
        assert updated[field] == client.text
        assert updated[rr.HASHES_FIELD][field] == rr.text_hash(client.text)
    # the caller's dict is left alone
    assert entry["reviewer_a_rationale"] == "no helmet"


def test_rewrite_entry_fields_is_idempotent():
    client = FakeClient()
    entry = {"reviewer_a_rationale": "no helmet"}
    once, _, _ = rr.rewrite_entry_fields(entry, client=client, config=CONFIG)
    twice, done, _ = rr.rewrite_entry_fields(once, client=client, config=CONFIG)
    assert done == []
    assert len(client.calls) == 1
    assert twice == once


def test_rewrite_entry_fields_picks_up_human_edits_after_a_rewrite():
    client = FakeClient()
    entry, _, _ = rr.rewrite_entry_fields(
        {"reviewer_a_rationale": "no helmet"}, client=client, config=CONFIG
    )
    entry["reviewer_a_rationale"] = "actually the helmet is visible"
    _, done, _ = rr.rewrite_entry_fields(entry, client=client, config=CONFIG)
    assert done == ["reviewer_a_rationale"]


def test_rewrite_entry_fields_skips_blank_and_missing_fields():
    client = FakeClient()
    _, done, _ = rr.rewrite_entry_fields(
        {"reviewer_a_rationale": "  ", "final_rationale": "real text"},
        client=client, config=CONFIG,
    )
    assert done == ["final_rationale"]


def test_rewrite_entry_fields_keeps_human_text_when_the_api_fails():
    entry = {"reviewer_a_rationale": "no helmet"}
    updated, done, failed = rr.rewrite_entry_fields(entry, client=BrokenClient(), config=CONFIG)
    assert done == []
    assert failed == ["reviewer_a_rationale"]
    assert updated["reviewer_a_rationale"] == "no helmet"
    assert rr.HASHES_FIELD not in updated


def test_final_rationale_uses_the_adjudicated_kind_only_when_adjudicated():
    for status, expected in (("adjudicated", "adjudicator"), ("not_needed", "two reviewers")):
        client = FakeClient()
        rr.rewrite_entry_fields(
            {"final_rationale": "text", "adjudication_status": status},
            client=client, config=CONFIG,
        )
        assert expected in json.dumps(client.calls[0]["messages"])


# --- the gold_set save-path hook -------------------------------------------


def _annotations_file(tmp_path: Path, entry: dict) -> Path:
    path = tmp_path / "gold_annotations.json"
    path.write_text(json.dumps({"c1": entry}))
    return path


def test_rewrite_case_rationales_persists_the_rewrite(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_storage", lambda: storage.LocalStorage(tmp_path))
    path = _annotations_file(tmp_path, {"reviewer_a_rationale": "no helmet"})
    client = FakeClient()
    monkeypatch.setattr(rr, "build_client", lambda config=None: client)

    result = gold_set.rewrite_case_rationales(Path("gold_annotations.json"), "c1", config=CONFIG)

    assert result["c1"]["reviewer_a_rationale"] == client.text
    on_disk = json.loads(path.read_text())
    assert on_disk["c1"]["reviewer_a_rationale"] == client.text


def test_rewrite_case_rationales_never_raises_when_the_api_is_down(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_storage", lambda: storage.LocalStorage(tmp_path))
    path = _annotations_file(tmp_path, {"reviewer_a_rationale": "no helmet"})
    monkeypatch.setattr(rr, "build_client", lambda config=None: BrokenClient())

    assert gold_set.rewrite_case_rationales(Path("gold_annotations.json"), "c1", config=CONFIG) is None
    # the reviewer's own words survive untouched
    assert json.loads(path.read_text())["c1"]["reviewer_a_rationale"] == "no helmet"


def test_rewrite_case_rationales_respects_rewrite_on_save_false(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_storage", lambda: storage.LocalStorage(tmp_path))
    _annotations_file(tmp_path, {"reviewer_a_rationale": "no helmet"})
    monkeypatch.setattr(
        rr, "build_client",
        lambda config=None: pytest.fail("must not build a client when rewrite_on_save is off"),
    )
    config = PipelineConfig()
    config.rewriter.rewrite_on_save = False

    assert gold_set.rewrite_case_rationales(Path("gold_annotations.json"), "c1", config=config) is None


def test_rewrite_case_rationales_yields_to_a_concurrent_human_save(tmp_path, monkeypatch):
    """If someone saves new text while the rewrite is in flight, theirs wins."""
    monkeypatch.setattr(storage, "get_storage", lambda: storage.LocalStorage(tmp_path))
    path = _annotations_file(tmp_path, {"reviewer_a_rationale": "no helmet"})

    class RacingClient(FakeClient):
        def _create(self, **kwargs):
            # a second reviewer saves over the field mid-rewrite
            path.write_text(json.dumps({"c1": {"reviewer_a_rationale": "newer human text"}}))
            return super()._create(**kwargs)

    monkeypatch.setattr(rr, "build_client", lambda config=None: RacingClient())
    gold_set.rewrite_case_rationales(Path("gold_annotations.json"), "c1", config=CONFIG)

    assert json.loads(path.read_text())["c1"]["reviewer_a_rationale"] == "newer human text"


def test_rewrite_case_rationales_no_op_for_an_unknown_case(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "get_storage", lambda: storage.LocalStorage(tmp_path))
    _annotations_file(tmp_path, {"reviewer_a_rationale": "no helmet"})
    monkeypatch.setattr(
        rr, "build_client",
        lambda config=None: pytest.fail("must not build a client for a missing case"),
    )
    assert gold_set.rewrite_case_rationales(Path("gold_annotations.json"), "nope", config=CONFIG) is None
