import json
from pathlib import Path

import pytest

from ppe_pipeline.eval.gold_set import (
    GOLD_SET_REVIEWERS,
    apply_auto_resolution,
    compute_agreement,
    freeze_gold_set,
    load_gold_annotations,
    role_for,
    save_gold_annotations,
    to_ground_truth,
    update_case_annotation,
)


def test_load_gold_annotations_returns_empty_dict_when_missing(tmp_path):
    assert load_gold_annotations(tmp_path / "gold_annotations.json") == {}


def test_save_and_load_gold_annotations_roundtrip(tmp_path):
    path = tmp_path / "gold_annotations.json"
    annotations = {"abc": {"final_label": "violation"}}

    save_gold_annotations(path, annotations)

    assert load_gold_annotations(path) == annotations


def test_compute_agreement_agree():
    entry = {"reviewer_a_label": "violation", "reviewer_b_label": "violation"}
    assert compute_agreement(entry) == "agree"


def test_compute_agreement_disagree():
    entry = {"reviewer_a_label": "violation", "reviewer_b_label": "no_violation"}
    assert compute_agreement(entry) == "disagree"


def test_compute_agreement_single_review_when_one_label_missing():
    entry = {"reviewer_a_label": "violation", "reviewer_b_label": ""}
    assert compute_agreement(entry) == "single_review"


def test_compute_agreement_single_review_when_both_labels_missing():
    assert compute_agreement({}) == "single_review"


def test_to_ground_truth_violation_maps_to_tp():
    assert to_ground_truth("violation") == "TP"


def test_to_ground_truth_no_violation_maps_to_fp():
    assert to_ground_truth("no_violation") == "FP"


def test_to_ground_truth_cannot_determine_maps_to_none():
    assert to_ground_truth("cannot_determine") is None


def test_to_ground_truth_invalid_label_raises():
    with pytest.raises(ValueError, match="not_a_real_label"):
        to_ground_truth("not_a_real_label")


def _case(case_id: str, image: str = "some_image.jpg") -> dict:
    return {
        "case_id": case_id,
        "image": image,
        "violation_type": "glove",
        "person_box": [0, 0, 10, 10],
    }


def test_freeze_gold_set_excludes_case_with_no_annotation():
    base_cases = [_case("no_annotation")]
    output, warnings = freeze_gold_set(base_cases, {}, {})
    assert output == []
    assert warnings == []


def test_freeze_gold_set_excludes_cannot_determine_case():
    base_cases = [_case("cd_case")]
    annotations = {"cd_case": {"final_label": "cannot_determine"}}
    output, warnings = freeze_gold_set(base_cases, annotations, {})
    assert output == []


def test_freeze_gold_set_includes_violation_case_with_correct_ground_truth():
    base_cases = [_case("violation_case")]
    annotations = {
        "violation_case": {
            "final_label": "violation",
            "final_rationale": "clear bare hand",
            "reason_code": "clear_bare_hand",
            "evidence_sufficiency": "sufficient",
            "difficulty": "easy",
        }
    }
    output, warnings = freeze_gold_set(base_cases, annotations, {})

    assert len(output) == 1
    case = output[0]
    assert case["case_id"] == "violation_case"
    assert case["ground_truth"] == "TP"
    assert case["reason_code"] == "clear_bare_hand"
    assert case["evidence_sufficiency"] == "sufficient"
    assert case["difficulty"] == "easy"
    assert case["final_label"] == "violation"
    assert case["final_rationale"] == "clear bare hand"


def test_freeze_gold_set_includes_no_violation_case_with_correct_ground_truth():
    base_cases = [_case("fp_case")]
    annotations = {"fp_case": {"final_label": "no_violation"}}
    output, warnings = freeze_gold_set(base_cases, annotations, {})

    assert len(output) == 1
    assert output[0]["ground_truth"] == "FP"


def _complete_annotation(final_label: str = "violation") -> dict:
    return {
        "final_label": final_label,
        "reason_code": "clear_bare_hand",
        "evidence_sufficiency": "sufficient",
        "difficulty": "easy",
    }


def test_freeze_gold_set_warns_on_few_shot_image_overlap():
    base_cases = [_case("overlap_case", image="images/shared.jpg")]
    annotations = {"overlap_case": _complete_annotation()}
    few_shot_manifest = {"glove": [{"image": "crops/shared.jpg"}]}

    output, warnings = freeze_gold_set(base_cases, annotations, few_shot_manifest)

    assert len(output) == 1
    assert len(warnings) == 1
    assert "overlap_case" in warnings[0]
    assert "shared.jpg" in warnings[0]


def test_freeze_gold_set_no_warning_when_no_overlap():
    base_cases = [_case("no_overlap_case", image="images/unique.jpg")]
    annotations = {"no_overlap_case": _complete_annotation()}
    few_shot_manifest = {"glove": [{"image": "crops/other.jpg"}]}

    output, warnings = freeze_gold_set(base_cases, annotations, few_shot_manifest)

    assert warnings == []


@pytest.mark.parametrize("missing_field", ["reason_code", "evidence_sufficiency", "difficulty"])
def test_freeze_gold_set_warns_but_still_includes_case_missing_a_completeness_field(missing_field):
    base_cases = [_case("incomplete_case")]
    annotation = _complete_annotation()
    annotation[missing_field] = ""
    annotations = {"incomplete_case": annotation}

    output, warnings = freeze_gold_set(base_cases, annotations, {})

    assert len(output) == 1
    assert output[0]["case_id"] == "incomplete_case"
    assert len(warnings) == 1
    assert "incomplete_case" in warnings[0]
    assert missing_field in warnings[0]


def test_freeze_gold_set_no_completeness_warning_when_all_fields_present():
    base_cases = [_case("complete_case")]
    annotations = {"complete_case": _complete_annotation()}

    _, warnings = freeze_gold_set(base_cases, annotations, {})

    assert warnings == []


def test_role_for_assigns_exactly_one_role_per_person_per_case():
    roles = {role_for(person, "c0") for person in GOLD_SET_REVIEWERS}
    assert roles == {"reviewer_1", "reviewer_2", "adjudicator", None}


def test_role_for_covers_all_four_roles_across_several_cases():
    seen_roles = set()
    for i in range(20):
        for person in GOLD_SET_REVIEWERS:
            seen_roles.add(role_for(person, f"c{i}"))
    assert seen_roles == {"reviewer_1", "reviewer_2", "adjudicator", None}


def test_role_for_is_deterministic_for_same_person_and_case():
    first = role_for("Jacob", "c5")
    second = role_for("Jacob", "c5")
    assert first == second


def test_role_for_is_unaffected_by_other_case_ids_present():
    """Regression test for the rotation-instability bug: a case's role
    assignment used to be derived from its *sort position* within the full
    set of case_ids in the dataset file, so adding a new case_id that
    happened to sort earlier shifted every already-annotated case's index
    and silently reassigned everyone's roles. `role_for` now hashes the
    case_id itself, so it must give the same answer regardless of what
    other case_ids exist (or don't) anywhere else in the dataset.
    """
    case_ids = [f"case_{i:03d}" for i in range(50)]
    baseline_roles = {
        case_id: {person: role_for(person, case_id) for person in GOLD_SET_REVIEWERS}
        for case_id in case_ids
    }

    # Simulate the old bug's trigger: add a brand-new case_id that sorts
    # before everything else (and shuffle the rest, to prove list order/
    # membership have zero bearing on any of these results).
    extra_case_ids = ["aaaa_new_case", *reversed(case_ids), "zzzz_another_new_case"]

    for case_id in case_ids:
        for person in GOLD_SET_REVIEWERS:
            assert role_for(person, case_id) == baseline_roles[case_id][person]

    # role_for doesn't even take a surrounding case list anymore, but assert
    # explicitly that the extra/reordered ids in scope changes nothing.
    assert len(extra_case_ids) == 52
    for case_id in case_ids:
        for person in GOLD_SET_REVIEWERS:
            assert role_for(person, case_id) == baseline_roles[case_id][person]


def test_role_for_uses_people_list_size_not_hardcoded_four():
    people = ["Solo"]
    assert role_for("Solo", "any_case", people) == "reviewer_1"


def test_apply_auto_resolution_agree_sets_final_label_and_not_needed():
    entry = {"reviewer_a_label": "violation", "reviewer_b_label": "violation"}
    result = apply_auto_resolution(entry)
    assert result["final_label"] == "violation"
    assert result["adjudication_status"] == "not_needed"


def test_apply_auto_resolution_agree_synthesizes_final_rationale_from_reviewers():
    entry = {
        "reviewer_a_label": "violation",
        "reviewer_b_label": "violation",
        "reviewer_a_rationale": "Bare hands clearly visible",
        "reviewer_b_rationale": "No gloves worn.",
    }
    result = apply_auto_resolution(entry)
    assert result["final_rationale"] == (
        "Reviewer A: Bare hands clearly visible. Reviewer B: No gloves worn."
    )


def test_apply_auto_resolution_agree_with_missing_reviewer_rationale_leaves_final_rationale_unset():
    entry = {
        "reviewer_a_label": "violation",
        "reviewer_b_label": "violation",
        "reviewer_a_rationale": "Bare hands clearly visible",
    }
    result = apply_auto_resolution(entry)
    assert "final_rationale" not in result


def test_apply_auto_resolution_agree_does_not_overwrite_existing_final_rationale():
    entry = {
        "reviewer_a_label": "violation",
        "reviewer_b_label": "violation",
        "reviewer_a_rationale": "Bare hands clearly visible",
        "reviewer_b_rationale": "No gloves worn.",
        "final_rationale": "A human already wrote this.",
    }
    result = apply_auto_resolution(entry)
    assert result["final_rationale"] == "A human already wrote this."


def test_apply_auto_resolution_agree_on_cannot_determine_does_not_synthesize_final_rationale():
    entry = {
        "reviewer_a_label": "cannot_determine",
        "reviewer_b_label": "cannot_determine",
        "reviewer_a_rationale": "Image too blurry to tell",
        "reviewer_b_rationale": "Can't see the hands clearly.",
    }
    result = apply_auto_resolution(entry)
    assert result["final_label"] == "cannot_determine"
    assert result["adjudication_status"] == "not_needed"
    assert "final_rationale" not in result


def test_apply_auto_resolution_disagree_with_blank_final_label_sets_pending():
    entry = {"reviewer_a_label": "violation", "reviewer_b_label": "no_violation"}
    result = apply_auto_resolution(entry)
    assert result["adjudication_status"] == "pending"
    assert not result.get("final_label")


def test_apply_auto_resolution_disagree_with_final_label_already_set_is_unchanged():
    entry = {
        "reviewer_a_label": "violation",
        "reviewer_b_label": "no_violation",
        "final_label": "violation",
        "adjudication_status": "adjudicated",
    }
    result = apply_auto_resolution(dict(entry))
    assert result == entry


def test_apply_auto_resolution_single_review_is_unchanged():
    entry = {"reviewer_a_label": "violation", "reviewer_b_label": ""}
    result = apply_auto_resolution(dict(entry))
    assert result == entry


def test_update_case_annotation_merges_without_clobbering_unrelated_fields(tmp_path):
    path = tmp_path / "gold_annotations.json"
    save_gold_annotations(path, {"c1": {"reviewer_a_name": "Aidil", "reviewer_a_label": "violation"}})

    result = update_case_annotation(path, "c1", {"reviewer_b_label": "violation", "reviewer_b_name": "Jacob"})

    assert result["c1"]["reviewer_a_name"] == "Aidil"
    assert result["c1"]["reviewer_a_label"] == "violation"
    assert result["c1"]["reviewer_b_label"] == "violation"
    assert result["c1"]["reviewer_b_name"] == "Jacob"


def test_update_case_annotation_applies_auto_resolution(tmp_path):
    path = tmp_path / "gold_annotations.json"
    save_gold_annotations(path, {"c1": {"reviewer_a_label": "violation"}})

    result = update_case_annotation(path, "c1", {"reviewer_b_label": "violation"})

    assert result["c1"]["final_label"] == "violation"
    assert result["c1"]["adjudication_status"] == "not_needed"


def test_update_case_annotation_writes_to_disk(tmp_path):
    path = tmp_path / "gold_annotations.json"

    update_case_annotation(path, "c1", {"reviewer_a_label": "violation", "reviewer_a_name": "Aidil"})

    reloaded = load_gold_annotations(path)
    assert reloaded["c1"]["reviewer_a_label"] == "violation"
    assert reloaded["c1"]["reviewer_a_name"] == "Aidil"


def test_update_case_annotation_rejects_mismatched_role_keys(tmp_path):
    path = tmp_path / "gold_annotations.json"

    with pytest.raises(ValueError, match="adjudication_status"):
        update_case_annotation(
            path,
            "c1",
            {"reviewer_a_label": "violation", "adjudication_status": "adjudicated"},
            role="reviewer_1",
        )

    assert load_gold_annotations(path) == {}


def test_update_case_annotation_accepts_matching_role_keys(tmp_path):
    path = tmp_path / "gold_annotations.json"

    result = update_case_annotation(
        path, "c1", {"reviewer_a_label": "violation", "reviewer_a_name": "Aidil"}, role="reviewer_1",
    )

    assert result["c1"]["reviewer_a_label"] == "violation"
    assert result["c1"]["reviewer_a_name"] == "Aidil"


def test_update_case_annotation_no_role_skips_validation_and_allows_everything(tmp_path):
    path = tmp_path / "gold_annotations.json"

    result = update_case_annotation(
        path,
        "c1",
        {"reviewer_a_label": "violation", "adjudication_status": "adjudicated"},
        role=None,
    )

    assert result["c1"]["reviewer_a_label"] == "violation"
    assert result["c1"]["adjudication_status"] == "adjudicated"
