"""Tests for the audit queue and its agreement rate."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ppe_pipeline.audit import (
    AUDIT_DECISIONS,
    AUDIT_LIMIT,
    CORRECTED_VERDICTS,
    _audit_is_stale,
    audit_confusion,
    audit_summary,
    shadow_mode_agreement_from_audits,
    build_audit_queue,
    load_audits,
    save_audit,
)
from ppe_pipeline import storage
from ppe_pipeline.config import PipelineConfig


def _row(**overrides) -> dict:
    row = {
        "alarm_id": "a1",
        "hazard_type": "no_helmet",
        "outcome": "CONFIRM_FP",
        "reason_code": "helmet_worn",
        "confidence": 0.95,
        "timestamp": "2026-08-27T10:00:00+00:00",
        "image_key": "eval/alarm_images/a1.jpg",
        "crop_box": {"x1": 0, "y1": 0, "x2": 200, "y2": 300},
        "helmet_box_source": "cv",
        "helmet_box_confidence": 0.9,
        "judge_model": "qwen.qwen3-vl-235b-a22b-instruct",
    }
    row.update(overrides)
    return row


# --- who is auditable -----------------------------------------------------


def test_a_pipeline_settled_case_is_auditable():
    """These had no human oversight at all, which is why they need some."""
    items = build_audit_queue([_row()], {}, {}, config=PipelineConfig())

    assert [i.alarm_id for i in items] == ["a1"]
    assert items[0].settled_by == "pipeline"
    assert items[0].settled_verdict == "no_violation"


def test_a_human_settled_case_is_auditable_and_marked_as_such():
    """Auditing this audits the reviewer, not the model."""
    rows = [_row(outcome="ESCALATE")]
    reviews = {"a1": {"verdict": "violation", "reviewed_by": "Han"}}

    items = build_audit_queue(rows, reviews, {}, config=PipelineConfig())

    assert items[0].settled_by == "human"
    assert items[0].settled_verdict == "violation"
    assert items[0].reviewed_by == "Han"


def test_a_case_still_awaiting_a_human_is_not_auditable():
    """Nothing has been settled, so there is nothing to check."""
    assert build_audit_queue([_row(outcome="ESCALATE")], {}, {}, config=PipelineConfig()) == []


def test_a_case_with_no_frame_is_not_auditable():
    """An auditor has to be able to look at it."""
    rows = [_row(image_key="streamlit-legacy.jpg")]

    assert build_audit_queue(rows, {}, {}, config=PipelineConfig()) == []


@pytest.mark.parametrize(
    "review",
    [{"verdict": "cannot_determine"}, {"helmet_box_status": "no_helmet_visible"}],
)
def test_a_deliberately_unlabelled_case_is_not_auditable(review):
    """Both were excluded from the dataset on purpose, so there is nothing to
    check them against."""
    rows = [_row(outcome="ESCALATE")]

    assert build_audit_queue(rows, {"a1": review}, {}, config=PipelineConfig()) == []


def test_the_settled_verdict_prefers_the_human_over_the_judge():
    """A human decision is what settling MEANS, so that is what gets audited
    -- and the judge's own outcome stays visible beside it."""
    rows = [_row(outcome="ESCALATE", reason_code="head_occluded")]
    reviews = {"a1": {"verdict": "violation"}}

    item = build_audit_queue(rows, reviews, {}, config=PipelineConfig())[0]

    assert item.settled_verdict == "violation"
    assert item.judge_outcome == "ESCALATE"


def test_filtering_by_origin():
    rows = [_row(alarm_id="auto"), _row(alarm_id="hum", outcome="ESCALATE")]
    reviews = {"hum": {"verdict": "violation"}}

    pipeline = build_audit_queue(rows, reviews, {}, config=PipelineConfig(), settled_by="pipeline")
    human = build_audit_queue(rows, reviews, {}, config=PipelineConfig(), settled_by="human")

    assert [i.alarm_id for i in pipeline] == ["auto"]
    assert [i.alarm_id for i in human] == ["hum"]


def test_exported_is_flagged_so_findings_can_be_tied_to_the_dataset():
    items = build_audit_queue(
        [_row()], {}, {}, config=PipelineConfig(), exported={"a1": {"batch_id": "b"}}
    )

    assert items[0].exported is True


# --- the queue and its cap ------------------------------------------------


def test_the_queue_is_capped_and_oldest_first():
    """A cap on workload, not a sample. Oldest-first keeps membership stable:
    a new case joins the end rather than displacing something someone may be
    part-way through."""
    rows = [
        _row(alarm_id=f"a{i:03d}", timestamp=f"2026-08-{(i % 27) + 1:02d}T00:00:00+00:00")
        for i in range(30)
    ]

    items = build_audit_queue(rows, {}, {}, config=PipelineConfig(), limit=10)

    assert len(items) == 10
    timestamps = [i.timestamp for i in items]
    assert timestamps == sorted(timestamps)


def test_fewer_cases_than_the_cap_yields_all_of_them():
    rows = [_row(alarm_id=f"a{i}") for i in range(3)]

    assert len(build_audit_queue(rows, {}, {}, config=PipelineConfig())) == 3


def test_the_default_cap_is_400():
    assert AUDIT_LIMIT == 400


def test_an_audited_case_drops_out_so_finishing_some_lets_more_in():
    rows = [_row(alarm_id="a1"), _row(alarm_id="a2")]
    audits = {"a1": {"decision": "agree"}}

    pending = build_audit_queue(rows, {}, audits, config=PipelineConfig())
    everything = build_audit_queue(
        rows, {}, audits, config=PipelineConfig(), include_audited=True
    )

    assert [i.alarm_id for i in pending] == ["a2"]
    assert len(everything) == 2


def test_a_half_written_audit_entry_does_not_count_as_audited():
    rows = [_row()]

    assert len(build_audit_queue(rows, {}, {"a1": {"note": "hmm"}}, config=PipelineConfig())) == 1


# --- the agreement rate ---------------------------------------------------


def test_agreement_rate_is_none_before_anything_is_audited():
    """0% agreement and "no data yet" are very different claims."""
    summary = audit_summary([_row()], {}, {}, config=PipelineConfig())

    assert summary["overall"]["audited"] == 0
    assert summary["overall"]["agreement_rate"] is None
    assert summary["overall"]["pending"] == 1


def test_agreement_rate_counts_agreements_over_audited():
    rows = [_row(alarm_id=f"a{i}") for i in range(4)]
    audits = {
        "a0": {"decision": "agree"},
        "a1": {"decision": "agree"},
        "a2": {"decision": "agree"},
        "a3": {"decision": "disagree", "corrected_verdict": "violation"},
    }

    overall = audit_summary(rows, {}, audits, config=PipelineConfig())["overall"]

    assert overall["audited"] == 4
    assert overall["agreed"] == 3
    assert overall["disagreed"] == 1
    assert overall["agreement_rate"] == pytest.approx(0.75)


def test_the_summary_separates_pipeline_from_human_settled():
    """They measure different things -- one checks the model, the other the
    reviewer -- and averaging them hides both."""
    rows = [_row(alarm_id="auto"), _row(alarm_id="hum", outcome="ESCALATE")]
    reviews = {"hum": {"verdict": "violation"}}
    audits = {
        "auto": {"decision": "disagree", "corrected_verdict": "violation"},
        "hum": {"decision": "agree"},
    }

    summary = audit_summary(rows, reviews, audits, config=PipelineConfig())

    assert summary["pipeline_settled"]["agreement_rate"] == 0.0
    assert summary["human_settled"]["agreement_rate"] == 1.0
    assert summary["overall"]["agreement_rate"] == pytest.approx(0.5)


def test_the_summary_reports_the_exported_subset_separately():
    """An audit checks what was PREPARED, so findings on unexported cases say
    nothing about the dataset."""
    rows = [_row(alarm_id="kept"), _row(alarm_id="other")]
    audits = {
        "kept": {"decision": "agree"},
        "other": {"decision": "disagree", "corrected_verdict": "violation"},
    }

    summary = audit_summary(
        rows, {}, audits, config=PipelineConfig(), exported={"kept": {"batch_id": "b"}}
    )

    assert summary["exported"]["auditable"] == 1
    assert summary["exported"]["agreement_rate"] == 1.0


# --- staleness ------------------------------------------------------------


def test_an_audit_retires_when_the_verdict_it_judged_changes():
    """An alarm id identifies the IMAGE, so re-running a case supersedes the
    old row. An "agree" recorded against the old verdict must not silently
    read as agreeing with the new, opposite one."""
    rows = [_row(outcome="CONFIRM_TP")]  # re-run now says violation
    audits = {"a1": {"decision": "agree", "audited_verdict": "no_violation"}}

    items = build_audit_queue(rows, {}, audits, config=PipelineConfig())

    assert [i.alarm_id for i in items] == ["a1"]  # back in the queue
    assert items[0].audited is False
    assert items[0].stale_audit is True


def test_an_audit_still_counts_while_the_verdict_it_judged_stands():
    rows = [_row(outcome="CONFIRM_FP")]
    audits = {"a1": {"decision": "agree", "audited_verdict": "no_violation"}}

    everything = build_audit_queue(
        rows, {}, audits, config=PipelineConfig(), include_audited=True
    )

    assert everything[0].audited is True
    assert everything[0].stale_audit is False
    assert build_audit_queue(rows, {}, audits, config=PipelineConfig()) == []


def test_an_entry_written_before_the_field_existed_is_taken_at_face_value():
    """There is no way to know what it agreed with, and guessing would be
    worse than trusting it."""
    rows = [_row(outcome="CONFIRM_TP")]
    audits = {"a1": {"decision": "agree"}}

    assert build_audit_queue(rows, {}, audits, config=PipelineConfig()) == []


def test_a_stale_audit_is_excluded_from_the_agreement_rate_and_counted():
    """Otherwise the rate would report agreement with a verdict no longer in
    effect -- and a drop in `audited` would look like lost work."""
    rows = [_row(alarm_id="moved", outcome="CONFIRM_TP"), _row(alarm_id="stood")]
    audits = {
        "moved": {"decision": "agree", "audited_verdict": "no_violation"},
        "stood": {"decision": "agree", "audited_verdict": "no_violation"},
    }

    overall = audit_summary(rows, {}, audits, config=PipelineConfig())["overall"]

    assert overall["audited"] == 1
    assert overall["stale"] == 1
    assert overall["pending"] == 1
    assert overall["agreement_rate"] == 1.0


def test_a_human_verdict_change_also_retires_the_audit():
    """A reviewer correcting their own earlier decision is the other way the
    audited verdict moves."""
    rows = [_row(outcome="ESCALATE")]
    reviews = {"a1": {"verdict": "violation"}}
    audits = {"a1": {"decision": "agree", "audited_verdict": "no_violation"}}

    items = build_audit_queue(rows, reviews, audits, config=PipelineConfig())

    assert items[0].stale_audit is True


# --- persistence ----------------------------------------------------------


def test_save_audit_records_the_decision_and_who_made_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    when = datetime(2026, 8, 27, 12, 0, tzinfo=timezone.utc)

    entry = save_audit(
        "a1",
        "disagree",
        corrected_verdict="violation",
        auditor="Han",
        note="bare head",
        path="audits.json",
        now=when,
    )

    assert entry["decision"] == "disagree"
    assert entry["corrected_verdict"] == "violation"
    assert entry["audited_by"] == "Han"
    assert entry["audited_at"] == when.isoformat()
    assert entry["note"] == "bare head"


def test_save_audit_records_the_verdict_the_decision_was_about(tmp_path, monkeypatch):
    """So the decision can retire itself if that verdict later changes."""
    monkeypatch.chdir(tmp_path)

    entry = save_audit(
        "a1", "agree", audited_verdict="no_violation", path="audits.json"
    )

    assert entry["audited_verdict"] == "no_violation"


def test_reading_the_audits_back_on_s3_sees_the_save_that_just_happened(monkeypatch):
    """`load_audits` is one of the four reads that revalidates its bytes, and
    `routers/escalations.record_audit` re-reads it DELIBERATELY after saving
    so `_sync_struck_crops` acts on the new strike-outs rather than the old
    ones. A cache that served the pre-save copy there would pull the wrong
    crops out of the training set.

    The rest of the suite runs on the local backend, which caches nothing, so
    this is the only place that path is exercised with the cache live."""
    from test_storage import FakeS3Client

    backend = storage.S3Storage(bucket="safety-model-dataset", client=FakeS3Client())
    monkeypatch.setattr(storage, "get_storage", lambda: backend)

    save_audit("a1", "agree", excluded_people=["a1__p00"], path="audits.json")
    assert load_audits("audits.json")["a1"]["excluded_people"] == ["a1__p00"]

    save_audit("a1", "agree", excluded_people=["a1__p00", "a1__p01"], path="audits.json")

    assert load_audits("audits.json")["a1"]["excluded_people"] == [
        "a1__p00",
        "a1__p01",
    ]


def test_a_save_that_says_nothing_about_strike_outs_leaves_them_alone(tmp_path, monkeypatch):
    """Every save merges over the previous entry, so a caller that has never
    heard of `excluded_people` would otherwise erase an auditor's work by
    saying nothing about it. A box reviewer and a helmet-CV reviewer both do
    exactly that."""
    monkeypatch.chdir(tmp_path)
    save_audit("a1", "agree", excluded_people=["a1__p00"], path="audits.json")

    save_audit("a1", "disagree", corrected_verdict="violation", source="box_review",
               path="audits.json")

    assert load_audits("audits.json")["a1"]["excluded_people"] == ["a1__p00"]


def test_an_empty_list_still_clears_the_strike_outs(tmp_path, monkeypatch):
    """`[]` is a reviewer un-checking everything, which is a real
    instruction; only `None` means "not touching them"."""
    monkeypatch.chdir(tmp_path)
    save_audit("a1", "agree", excluded_people=["a1__p00"], path="audits.json")

    save_audit("a1", "agree", excluded_people=[], path="audits.json")

    assert load_audits("audits.json")["a1"]["excluded_people"] == []


def test_a_recorded_null_verdict_is_not_the_same_as_a_missing_one(tmp_path, monkeypatch):
    """An entry with no `audited_verdict` KEY predates the field and is taken
    at face value. One recorded as null was recorded against a case that had
    no verdict, and must retire once the case gets one -- treating the two
    the same made such an audit permanently unretirable."""
    monkeypatch.chdir(tmp_path)
    save_audit("a1", "agree", audited_verdict=None, path="audits.json")
    recorded = load_audits("audits.json")["a1"]

    assert _audit_is_stale(recorded, None) is False
    assert _audit_is_stale(recorded, "violation") is True
    assert _audit_is_stale({"decision": "agree"}, "violation") is False


def test_audits_are_per_case_and_do_not_clobber_each_other(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_audit("a1", "agree", path="audits.json")
    save_audit("a2", "disagree", corrected_verdict="violation", path="audits.json")

    audits = load_audits("audits.json")

    assert audits["a1"]["decision"] == "agree"
    assert audits["a2"]["corrected_verdict"] == "violation"


def test_the_vocabularies_are_what_the_ui_expects():
    assert AUDIT_DECISIONS == ["agree", "disagree", "box_wrong"]
    assert CORRECTED_VERDICTS == ["violation", "no_violation", "cannot_determine"]


# --- the confusion matrix audits produce -----------------------------------


def test_audits_produce_a_confusion_matrix_against_the_judge():
    """The point of auditing live data: no gold set exists, so the auditor's
    confirmation IS the ground truth, and the same matrix the offline eval
    reports falls out of it."""
    rows = [
        # judge said violation, auditor agreed -> a real catch
        _row(alarm_id="tp", outcome="CONFIRM_TP"),
        # judge said no violation, auditor agreed -> a real clear
        _row(alarm_id="tn", outcome="CONFIRM_FP"),
        # judge cleared it, auditor says it WAS a violation -> a miss
        _row(alarm_id="fn", outcome="CONFIRM_FP"),
        # judge confirmed it, auditor says it wasn't -> a false alarm kept
        _row(alarm_id="fp", outcome="CONFIRM_TP"),
    ]
    audits = {
        "tp": {"decision": "agree", "audited_verdict": "violation"},
        "tn": {"decision": "agree", "audited_verdict": "no_violation"},
        "fn": {
            "decision": "disagree",
            "corrected_verdict": "violation",
            "audited_verdict": "no_violation",
        },
        "fp": {
            "decision": "disagree",
            "corrected_verdict": "no_violation",
            "audited_verdict": "violation",
        },
    }

    overall = audit_confusion(rows, {}, audits, config=PipelineConfig())["overall"]

    assert overall["counts"] == {
        "true_positive": 1,
        "false_negative": 1,
        "false_positive": 1,
        "true_negative": 1,
        "uncertain": 0,
    }
    assert overall["precision"] == pytest.approx(0.5)
    assert overall["recall"] == pytest.approx(0.5)
    assert overall["f1"] == pytest.approx(0.5)
    assert overall["accuracy"] == pytest.approx(0.5)
    assert overall["n"] == 4


def test_the_distribution_says_where_the_errors_are():
    """A single accuracy number cannot say whether the judge is missing real
    violations or keeping false alarms."""
    rows = [
        _row(alarm_id="a", outcome="CONFIRM_TP"),
        _row(alarm_id="b", outcome="CONFIRM_FP"),
        _row(alarm_id="c", outcome="CONFIRM_FP"),
    ]
    audits = {
        "a": {"decision": "agree", "audited_verdict": "violation"},
        # both really were violations; the judge cleared them
        "b": {
            "decision": "disagree",
            "corrected_verdict": "violation",
            "audited_verdict": "no_violation",
        },
        "c": {
            "decision": "disagree",
            "corrected_verdict": "violation",
            "audited_verdict": "no_violation",
        },
    }

    overall = audit_confusion(rows, {}, audits, config=PipelineConfig())["overall"]

    assert overall["ground_truth_violation_total"] == 3
    assert overall["ground_truth_no_violation_total"] == 0
    assert overall["violation_verdict_distribution"] == {"hold": 1, "cleared": 2, "escalate": 0}


def test_an_escalation_counts_as_uncertain_not_as_a_miss():
    """The judge declining to answer is a different failure from answering
    wrongly, and folding them together would hide both."""
    rows = [_row(alarm_id="esc", outcome="ESCALATE")]
    reviews = {"esc": {"verdict": "violation"}}
    audits = {"esc": {"decision": "agree", "audited_verdict": "violation"}}

    overall = audit_confusion(rows, reviews, audits, config=PipelineConfig())["overall"]

    assert overall["counts"]["uncertain"] == 1
    assert overall["counts"]["false_negative"] == 0
    assert overall["violation_verdict_distribution"] == {"hold": 0, "cleared": 0, "escalate": 1}
    # Accuracy is over DECIDED cases, so an escalation cannot drag it down.
    assert overall["decided_total"] == 0


def test_an_unaudited_or_undecidable_case_contributes_no_ground_truth():
    rows = [_row(alarm_id="pending"), _row(alarm_id="shrug")]
    audits = {
        "shrug": {
            "decision": "disagree",
            "corrected_verdict": "cannot_determine",
            "audited_verdict": "no_violation",
        }
    }

    overall = audit_confusion(rows, {}, audits, config=PipelineConfig())["overall"]

    assert overall["n"] == 0
    assert overall["sample_is_thin"] is True


def test_a_stale_audit_contributes_no_ground_truth_either():
    """It judged a verdict that no longer stands."""
    rows = [_row(alarm_id="moved", outcome="CONFIRM_TP")]
    audits = {"moved": {"decision": "agree", "audited_verdict": "no_violation"}}

    assert audit_confusion(rows, {}, audits, config=PipelineConfig())["overall"]["n"] == 0


def test_the_matrix_is_broken_out_per_hazard_type():
    rows = [
        _row(alarm_id="h", hazard_type="no_helmet", outcome="CONFIRM_TP"),
        _row(alarm_id="v", hazard_type="too_close_to_vehicle", outcome="CONFIRM_TP"),
    ]
    audits = {
        "h": {"decision": "agree", "audited_verdict": "violation"},
        "v": {
            "decision": "disagree",
            "corrected_verdict": "no_violation",
            "audited_verdict": "violation",
        },
    }

    per_type = audit_confusion(rows, {}, audits, config=PipelineConfig())["per_hazard_type"]

    assert per_type["no_helmet"]["precision"] == 1.0
    assert per_type["too_close_to_vehicle"]["precision"] == 0.0


# --- the benchmark, now measured against audits ----------------------------


def _confusion_for(rows, audits, reviews=None):
    return audit_confusion(rows, reviews or {}, audits, config=PipelineConfig())["overall"]


def test_the_benchmark_keeps_its_old_definition_with_a_new_label_source():
    """Of the cases the judge COMMITTED on, how often did it match the human
    -- the same question the gold-set version asked. Only where the human
    answer comes from changed, which is what keeps the two comparable."""
    rows = [_row(alarm_id=f"a{i}", outcome="CONFIRM_TP") for i in range(8)]
    rows += [_row(alarm_id=f"b{i}", outcome="CONFIRM_FP") for i in range(4)]
    audits = {f"a{i}": {"decision": "agree", "audited_verdict": "violation"} for i in range(8)}
    # Two of the clears were wrong.
    audits.update(
        {f"b{i}": {"decision": "agree", "audited_verdict": "no_violation"} for i in range(2)}
    )
    audits.update(
        {
            f"b{i}": {
                "decision": "disagree",
                "corrected_verdict": "violation",
                "audited_verdict": "no_violation",
            }
            for i in (2, 3)
        }
    )

    agreement = shadow_mode_agreement_from_audits(_confusion_for(rows, audits))

    assert agreement["insufficient_data"] is False
    assert agreement["matched_count"] == 12
    assert agreement["agreement_count"] == 10
    assert agreement["agreement_rate"] == pytest.approx(10 / 12)


def test_the_benchmark_is_withheld_on_a_thin_sample():
    """Same treatment the gold-set version gave it: a rate on three cases is
    not a measurement."""
    rows = [_row(alarm_id="a", outcome="CONFIRM_TP")]
    audits = {"a": {"decision": "agree", "audited_verdict": "violation"}}

    agreement = shadow_mode_agreement_from_audits(_confusion_for(rows, audits))

    assert agreement["insufficient_data"] is True
    assert agreement["matched_count"] == 1
    assert "agreement_rate" not in agreement


def test_an_escalation_is_outside_the_benchmark_entirely():
    """The judge deferring is not a verdict to agree or disagree with, so it
    belongs in neither the numerator nor the denominator."""
    rows = [_row(alarm_id="esc", outcome="ESCALATE")]
    reviews = {"esc": {"verdict": "violation"}}
    audits = {"esc": {"decision": "agree", "audited_verdict": "violation"}}

    agreement = shadow_mode_agreement_from_audits(_confusion_for(rows, audits, reviews))

    assert agreement["matched_count"] == 0


def test_the_benchmark_is_the_confusion_matrix_read_out_loud():
    """Not a second, subtly different statistic reported beside it -- if these
    ever diverged, the page would be making two claims about one thing."""
    rows = [_row(alarm_id=f"a{i}", outcome="CONFIRM_TP") for i in range(12)]
    audits = {f"a{i}": {"decision": "agree", "audited_verdict": "violation"} for i in range(10)}
    audits.update(
        {
            f"a{i}": {
                "decision": "disagree",
                "corrected_verdict": "no_violation",
                "audited_verdict": "violation",
            }
            for i in (10, 11)
        }
    )
    confusion = _confusion_for(rows, audits)

    agreement = shadow_mode_agreement_from_audits(confusion)

    assert agreement["agreement_rate"] == pytest.approx(confusion["accuracy"])


def test_an_audit_item_carries_the_box_the_pipeline_settled_on():
    """Judging "was this verdict right" from the crop alone, with the evidence
    the judge acted on left undrawn, is asking someone to check work they have
    not been shown."""
    rows = [_row(helmet_box={"x1": 10, "y1": 20, "x2": 40, "y2": 50},
                 person_box={"x1": 0, "y1": 0, "x2": 100, "y2": 200})]

    item = build_audit_queue(rows, {}, {}, config=PipelineConfig())[0]

    assert item.helmet_box == [10.0, 20.0, 40.0, 50.0]
    assert item.person_box == [0.0, 0.0, 100.0, 200.0]
    # Same frame-pixel space as the crop it is drawn over.
    assert item.crop_box == [0.0, 0.0, 200.0, 300.0]


def test_a_case_with_no_recorded_box_says_so_rather_than_guessing():
    item = build_audit_queue([_row(helmet_box=None)], {}, {}, config=PipelineConfig())[0]

    assert item.helmet_box is None
