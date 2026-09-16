"""person -> violations dictionary, and its inverse for reporting."""

from __future__ import annotations

from collections import defaultdict

from .models import PersonCluster, ViolationType


def build_person_violation_dict(
    clusters: list[PersonCluster],
) -> dict[str, list[str]]:
    return {c.person_id: sorted(v.value for v in c.violations) for c in clusters}


def invert_violations(
    person_to_violations: dict[str, list[str]],
) -> dict[str, list[str]]:
    """violation type -> list of affected people, for reporting/dashboards."""
    violation_to_people: dict[str, list[str]] = defaultdict(list)
    for person_id, violations in person_to_violations.items():
        for v in violations:
            violation_to_people[v].append(person_id)
    return dict(violation_to_people)


__all__ = ["ViolationType", "build_person_violation_dict", "invert_violations"]
