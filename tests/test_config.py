from pathlib import Path

from ppe_pipeline.config import DEFAULT_CONFIG_PATH, PipelineConfig, load_config
from ppe_pipeline.models import ViolationType


def test_defaults_apply_when_no_file_is_given(tmp_path):
    missing = tmp_path / "does_not_exist.toml"
    # load_config only reads a file when one resolves; a path that doesn't
    # exist should never be passed to tomllib in the first place here,
    # so exercise the "no override" path directly instead.
    config = PipelineConfig()
    assert config.merge.overlap_threshold == 0.3
    assert config.cropping.glove_helmet_pad == 0.3
    assert config.batching.gap_seconds == 2.0
    assert config.judge.model == "qwen.qwen3-vl-235b-a22b-instruct"
    assert config.judge.uncertain_as_fp == [ViolationType.GLOVE]
    assert not missing.exists()


def test_partial_override_falls_back_to_defaults_for_missing_keys(tmp_path):
    override = tmp_path / "config.toml"
    override.write_text(
        """
        [merge]
        overlap_threshold = 0.5

        [batching]
        gap_seconds = 5.0
        """
    )
    config = load_config(override)

    assert config.merge.overlap_threshold == 0.5
    assert config.batching.gap_seconds == 5.0
    # untouched sections keep their built-in defaults
    assert config.cropping.glove_helmet_pad == 0.3
    assert config.forklift.expand_pct == 0.2
    assert config.judge.model == "qwen.qwen3-vl-235b-a22b-instruct"


def test_project_config_toml_matches_builtin_defaults():
    # The human-editable config.toml at the project root mirrors the
    # PipelineConfig() defaults for every section except [forklift] and
    # [few_shot]. [forklift] was retuned
    # against eval/combined_eval_dataset.json's ground truth (see config.toml's
    # own comment), so it's expected to diverge from the code's built-in
    # 0.2/0.3 default; [few_shot] overrides helmet's max_examples_per_type up
    # to 72 (its promoted, reasoning-calibrated library — the 10 from
    # scripts/promote_gold_cases_to_fewshot.py plus the 62 appended by
    # scripts/promote_helmet_62_to_fewshot.py) while leaving the global
    # default and every other violation type untouched. This test catches
    # drift in every other section/key.
    assert DEFAULT_CONFIG_PATH.is_file(), f"expected {DEFAULT_CONFIG_PATH} to exist"
    config = load_config(DEFAULT_CONFIG_PATH)
    defaults = PipelineConfig()
    assert config.merge == defaults.merge
    assert config.cropping == defaults.cropping
    assert config.forklift.expand_pct == 0.3
    assert config.forklift.overlap_threshold == 0.06
    assert config.forklift.operator_containment_threshold == defaults.forklift.operator_containment_threshold
    assert config.batching == defaults.batching
    assert config.few_shot.max_examples_per_type == defaults.few_shot.max_examples_per_type
    # CEO, 2 Sep: all 72 again, after helmets were being missed -- the 50 cap
    # had dropped the 10 curated examples entirely, since truncation is from
    # the front and they had been rotated to the back. Asserted rather than
    # described, because the number IS the decision.
    assert config.few_shot.max_examples_per_type_override == {"helmet": 72}
    # CEO, 2 Sep: the manifest's own order -- curated 10 first, as built.
    assert config.few_shot.move_to_back_override == {}
    # [rewriter] mirrors the built-in defaults exactly (Claude on Bedrock in
    # us-east-1) - it exists in config.toml for discoverability and the long
    # comment explaining why it is separate from [judge], not to override.
    assert config.rewriter == defaults.rewriter
    assert config.judge.provider == "bedrock"
    assert config.judge.model == "qwen.qwen3-vl-235b-a22b-instruct"
    assert config.judge.max_tokens == defaults.judge.max_tokens
    assert config.judge.model_options == defaults.judge.model_options
    assert config.judge.uncertain_as_fp == defaults.judge.uncertain_as_fp


def test_empty_file_yields_all_defaults(tmp_path):
    empty = tmp_path / "config.toml"
    empty.write_text("")
    assert load_config(empty) == PipelineConfig()
