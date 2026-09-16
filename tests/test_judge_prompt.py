"""The judge's structured-output schema, which is what actually constrains
what a schema-aware provider is allowed to return."""

from ppe_pipeline.judge_prompt import SYSTEM_PROMPT, build_output_schema
from ppe_pipeline.models import ViolationType


def _variants(violation_types):
    return {
        variant["properties"]["violation_type"]["const"]: variant
        for variant in build_output_schema(violation_types)["properties"]["results"]["items"]["anyOf"]
    }


def test_helmet_results_carry_a_helmet_box():
    variant = _variants([ViolationType.HELMET])["helmet"]

    assert variant["properties"]["helmet_box"]["minItems"] == 4
    assert variant["properties"]["helmet_box"]["maxItems"] == 4
    # Nullable, not optional: a provider running the schema strictly needs
    # every property in "required", and every verdict that isn't a cleared
    # helmet still has to be able to say "no box".
    assert "null" in variant["properties"]["helmet_box"]["type"]
    assert "helmet_box" in variant["required"]


def test_no_other_violation_type_is_asked_for_a_helmet_box():
    variants = _variants(list(ViolationType))

    for name, variant in variants.items():
        if name == "helmet":
            continue
        assert "helmet_box" not in variant["properties"], name
        assert "helmet_box" not in variant["required"], name


def test_a_mixed_call_asks_for_the_box_on_the_helmet_variant_only():
    variants = _variants([ViolationType.HELMET, ViolationType.GLOVE])

    assert "helmet_box" in variants["helmet"]["properties"]
    assert "helmet_box" not in variants["glove"]["properties"]


def test_the_prompt_ties_the_box_to_a_cleared_helmet():
    assert "helmet_box" in SYSTEM_PROMPT
    # The rubric must scope it to the cleared-helmet case, not ask for it
    # on every helmet verdict.
    assert 'return "helmet_box": null' in SYSTEM_PROMPT
