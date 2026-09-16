"""Which candidates the judge is asked about.

THE PREDICATE MUST NOT LOOK AT THE GATE

`is_sampled` takes a candidate id and a rate, and that is the whole input.
It cannot see `gate_fired`, `state`, or the person's height, and that
omission is the entire point of the module.

The clip corpus is already skewed positive — every clip in `raw_video/`
exists because production CV fired on it. Drawing the sample blind to the
gate is what keeps the rest measurable anyway.

RECALL DEPENDS ON THIS, ENTIRELY

Because the sample ignores `gate_fired`, the audited set contains
candidates CV CLEARED as well as ones it flagged — and a human verdict of
"violation" on a cleared candidate is a real false negative. That is the
only source of false negatives the pilot has. Sample only the gated
candidates and recall loses its denominator outright, while the comparison
degenerates into "of the cases CV flagged, how many did the judge keep":
a number about the sample, not about the gate. Worse, it would be
invisible — the summary would still render, still say "precision", and
still be wrong.

So the rate applies uniformly, and there is a test asserting that gate-fired
and non-gate-fired candidates are sampled at statistically indistinguishable
rates. If someone later adds a `if record.gate_fired` here, that test is
what catches it.

WHY MD5 AND NOT random

Same reason `gold_set.case_rotation` uses it: Python's built-in `hash()` is
randomised per process for strings, so a resumed or re-run pass would pick a
different sample and the two halves could not be joined. `md5(candidate_id)`
is a pure function of the candidate, stable across runs, machines and
orderings — a candidate that was in the sample yesterday is in it today, and
adding a thousand new frames does not reshuffle the ones already judged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator


def sample_hash(candidate_id: str) -> int:
    """The candidate's own stable hash. Exposed for tests, and so a caller
    can bucket by it without re-deriving the digest."""
    return int(hashlib.md5(candidate_id.encode()).hexdigest(), 16)


def is_sampled(candidate_id: str, rate: int) -> bool:
    """Whether this candidate goes to the judge, at 1-in-`rate`.

    `rate` of 1 means everything. Nothing about the candidate other than its
    id is consulted — see the module docstring.
    """
    if rate < 1:
        raise ValueError(f"sample rate must be >= 1, got {rate}")
    if rate == 1:
        return True
    return sample_hash(candidate_id) % rate == 0


def rate_for_target(total: int, target: int) -> int:
    """The 1-in-N rate that lands nearest `target` candidates out of `total`.

    Rounds down, so the sample comes out at or ABOVE the target rather than
    below. Widening one later is only safe when the new rate DIVIDES the old
    (1-in-10 to 1-in-5 keeps every already-judged candidate and adds more,
    because `h % 10 == 0` implies `h % 5 == 0`); any other change re-draws
    the sample and discards judge calls already paid for.
    """
    if target < 1:
        raise ValueError(f"target must be >= 1, got {target}")
    if total <= target:
        return 1
    return max(1, total // target)


def select(candidate_ids: Iterable[str], rate: int) -> Iterator[str]:
    """The sampled subset of `candidate_ids`, in the order given."""
    return (cid for cid in candidate_ids if is_sampled(cid, rate))
