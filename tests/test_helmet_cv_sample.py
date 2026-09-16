"""Tests for choosing which candidates the judge is asked about.

The important one is `test_gate_fired_and_cleared_candidates_are_sampled_at
_indistinguishable_rates`. The predicate takes a candidate id and a rate and
nothing else, and that omission is the whole module: sampling on the gate's
own answer turns the pilot into "of the cases CV flagged, how many did the
judge keep", which is a number about the sample rather than about the gate —
and one that would still render, still say "precision", and still be wrong.
"""

from __future__ import annotations

import hashlib

import pytest

from ppe_pipeline.helmet_cv.sample import is_sampled, rate_for_target, sample_hash, select


def _ids(count: int, prefix: str = "cand") -> list[str]:
    return [f"{prefix}-{index:05d}" for index in range(count)]


# --- determinism ----------------------------------------------------------


def test_the_same_candidate_is_always_in_or_always_out():
    """A resumed pass must pick the same sample. Python's built-in `hash()`
    is randomised per process for strings, so it would not."""
    picked = [is_sampled("unk__unk__x__alarm__f000010__p00", 7) for _ in range(5)]

    assert len(set(picked)) == 1


def test_the_hash_is_md5_of_the_candidate_id():
    """Pinned so a future refactor cannot quietly reshuffle a sample that has
    already been half judged."""
    assert sample_hash("abc") == int(hashlib.md5(b"abc").hexdigest(), 16)


def test_adding_new_candidates_does_not_move_the_existing_ones():
    """Selection is a pure function of the id, not of position in a list, so
    a thousand new frames cannot displace a candidate already judged."""
    first = set(select(_ids(200), 5))
    second = set(select(_ids(1200), 5))

    assert first <= second


def test_rate_one_keeps_everything():
    assert list(select(_ids(50), 1)) == _ids(50)


def test_a_rate_below_one_is_refused():
    with pytest.raises(ValueError):
        is_sampled("x", 0)


# --- the rate -------------------------------------------------------------


def test_the_rate_lands_near_the_target():
    """The pilot is ~200 candidates."""
    rate = rate_for_target(20_000, 200)

    assert rate == 100
    assert 150 <= len(list(select(_ids(20_000), rate))) <= 260


def test_a_corpus_smaller_than_the_target_is_taken_whole():
    assert rate_for_target(50, 200) == 1


def test_the_rate_rounds_down_so_the_sample_is_not_undersized():
    """Overshooting is the cheap error: a slightly larger pilot costs a few
    judge calls, while a short one can only be widened by a rate that divides
    the old (see below), which is not always available."""
    rate = rate_for_target(999, 200)

    assert rate == 4
    assert len(list(select(_ids(999), rate))) >= 200


# --- the predicate is blind to the gate ----------------------------------


def test_the_predicate_takes_nothing_but_an_id_and_a_rate():
    """A structural guard: if someone adds a `record` or `gate_fired`
    parameter, this fails before the statistical test below has to."""
    import inspect

    parameters = list(inspect.signature(is_sampled).parameters)

    assert parameters == ["candidate_id", "rate"]


def test_gate_fired_and_cleared_candidates_are_sampled_at_indistinguishable_rates():
    """The one that catches a gate-aware predicate.

    Candidate ids are split into two arms by a property that has nothing to
    do with the id's hash -- here, the parity of the index, standing in for
    "the gate fired" -- and both arms must be sampled at the same rate. A
    predicate that consulted `gate_fired` would push one arm to 100% and the
    other to 0%, and this would fail by a mile rather than by noise.
    """
    ids = _ids(20_000)
    gate_fired = [cid for index, cid in enumerate(ids) if index % 2 == 0]
    gate_cleared = [cid for index, cid in enumerate(ids) if index % 2 == 1]

    rate = 10
    fired_rate = sum(is_sampled(cid, rate) for cid in gate_fired) / len(gate_fired)
    cleared_rate = sum(is_sampled(cid, rate) for cid in gate_cleared) / len(gate_cleared)

    # Both arms are 10,000 draws at p=0.1, so one standard error is ~0.003.
    # 0.02 is ~6 sigma: wide enough never to flake, narrow enough that any
    # real dependence on the arm shows up.
    assert abs(fired_rate - cleared_rate) < 0.02
    assert abs(fired_rate - 0.1) < 0.02


def test_the_overall_rate_is_close_to_one_in_n():
    ids = _ids(20_000)

    assert abs(len(list(select(ids, 20))) / len(ids) - 0.05) < 0.01


def test_a_pilot_can_be_widened_by_a_rate_that_divides_the_old_one():
    """`h % 10 == 0` implies `h % 5 == 0`, so going from 1-in-10 to 1-in-5
    KEEPS every already-judged candidate and adds more. Worth knowing: it is
    the only safe way to widen a pilot mid-flight."""
    five = set(select(_ids(2_000), 5))
    ten = set(select(_ids(2_000), 10))

    assert ten < five


def test_a_rate_that_does_not_divide_the_old_one_re_draws_the_sample():
    """1-in-7 is not a superset of 1-in-10. Changing to one of these
    discards judge calls already paid for, so it is not a widening."""
    seven = set(select(_ids(2_000), 7))
    ten = set(select(_ids(2_000), 10))

    assert not ten <= seven


# --- sampling must select WHOLE FRAMES ------------------------------------


def _candidate(frame_index: int, person_index: int) -> dict:
    name = f"unk__unk__20260427T070224Z__aaaa__f{frame_index:06d}.jpg"
    return {
        "candidate_id": f"{name[:-4]}__p{person_index:02d}",
        "frame_name": name,
    }


def test_sampling_selects_whole_frames_never_part_of_one():
    """The judge's per-crop answers are OR'd into ONE frame verdict, so a
    partially sampled frame computes that OR over an incomplete set: a frame
    whose violating worker was not sampled reads as "no violation".

    Measured on the real corpus, candidate-level sampling left 270 of 283
    frames partial -- 95%. At 8.54 people per alarm frame this is not an edge
    case, it is nearly every frame, and it understates the VLM's violations
    while distorting the CV arm through the shared human decision.
    """
    from ppe_pipeline.helmet_cv.cli import _sample_whole_frames

    candidates = [_candidate(f, p) for f in range(40) for p in range(5)]

    picked, frames, by_frame, rate = _sample_whole_frames(candidates, target=10, rate=None)

    chosen = {c["frame_name"] for c in picked}
    for name in chosen:
        in_corpus = [c for c in candidates if c["frame_name"] == name]
        in_sample = [c for c in picked if c["frame_name"] == name]
        assert len(in_sample) == len(in_corpus), f"{name} was sampled partially"
    assert len(frames) == len(chosen)
    assert picked, "the sample must not be empty"


def test_the_sampling_target_counts_frames_not_crops():
    """`--target 100` means 100 FRAMES. Counting crops would silently shrink
    the measured sample by the crowding factor, which is 8.54 here."""
    from ppe_pipeline.helmet_cv.cli import _sample_whole_frames

    candidates = [_candidate(f, p) for f in range(100) for p in range(8)]

    _, frames, by_frame, _ = _sample_whole_frames(candidates, target=50, rate=None)

    assert len(by_frame) == 100
    # roughly half the FRAMES, not roughly half the 800 crops
    assert 25 <= len(frames) <= 75
