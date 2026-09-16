"""Deterministic PPE replay contracts; no AWS calls or training side effects.

Ratios are fractions, times are milliseconds, and each frame has a content
SHA-256 identity. Detection metrics must come from independent box labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

SOURCE_CLASSES = {0: "helmet", 1: "gloves", 2: "person", 3: "working_boots", 4: "man_cage", 5: "safe_area"}
MODEL_CLASSES = {"hero": {0: "person", 1: "helmet"}, "localization": {0: "helmet"}}
REMAP = {"hero": {2: 0, 0: 1}, "localization": {0: 0}}
VERDICTS = {"violation", "no_violation"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def remap_labels(text, model):
    """Validate all source boxes before filtering; preserve negative images."""
    mapping = REMAP[model]
    result = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"line {number}: expected class and four coordinates")
        cid = int(values[0])
        x, y, w, h = map(float, values[1:])
        if cid not in SOURCE_CLASSES or not all(math.isfinite(v) for v in (x, y, w, h)):
            raise ValueError(f"line {number}: invalid class or nonfinite coordinate")
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f"line {number}: invalid box")
        if min(x-w/2, y-h/2) < -1e-6 or max(x+w/2, y+h/2) > 1+1e-6:
            raise ValueError(f"line {number}: box exceeds image")
        if cid in mapping:
            result.append(f"{mapping[cid]} " + " ".join(values[1:]))
    return "\n".join(result) + ("\n" if result else "")


def truth(row):
    # An explicit inconclusive human review must not fall back to model truth.
    value = row.get("human_verdict") if row.get("human_verdict") is not None else row.get("vlm_verdict")
    return value if value in VERDICTS else None


def select_cohort(rows, training_hashes, now, weekly_cap=1000):
    """Four complete Monday-Sunday Singapore weeks; stable content sampling.

    Input rows: frame_id, sha256, captured_at (timezone-aware ISO timestamp),
    vlm_verdict, optional human_verdict. Training hashes include all sources.
    """
    if now.tzinfo is None or not 1 <= weekly_cap <= 1000:
        raise ValueError("timezone-aware now and weekly cap 1..1000 required")
    local = now.astimezone(ZoneInfo("Asia/Singapore"))
    end = local.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=local.weekday())
    start = end - timedelta(weeks=4)
    excluded = set(training_hashes)
    unique, ids = {}, set()
    for row in rows:
        sha = row["sha256"]
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise ValueError("frame sha256 must be a lowercase content hash")
        captured = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
        if captured.tzinfo is None:
            raise ValueError("captured_at needs a timezone")
        if not start <= captured < end or sha in excluded or truth(row) is None:
            continue
        if sha in unique:
            if truth(unique[sha]) != truth(row):
                raise ValueError("conflicting truth for duplicate frame content")
            # Deterministic identity even when source ordering changes.
            if row["frame_id"] >= unique[sha]["frame_id"]:
                continue
        unique[sha] = dict(row)
    weeks = [[] for _ in range(4)]
    for row in unique.values():
        if row["frame_id"] in ids:
            raise ValueError("frame ID refers to multiple content hashes")
        ids.add(row["frame_id"])
        captured = datetime.fromisoformat(row["captured_at"].replace("Z", "+00:00"))
        weeks[(captured.astimezone(local.tzinfo).date()-start.date()).days // 7].append(row)
    selected = []
    for week in weeks:
        selected.extend(sorted(week, key=lambda r: digest([start.isoformat(), r["sha256"]]))[:weekly_cap])
    manifest = {"schema_version": "1.0.0", "window_start": start.isoformat(), "window_end_exclusive": end.isoformat(),
                "training_hashes_digest": digest(sorted(excluded)), "frames": sorted(selected, key=lambda r: r["frame_id"])}
    manifest["cohort_id"] = "cohort-" + digest(manifest)
    return manifest


def ratio(n, d):
    return n/d if d else None


def score(cohort, predictions):
    """Strict paired scoring: failed, extra, duplicate or missing frames fail."""
    by_id = {}
    for row in predictions:
        key = row["frame_id"]
        if key in by_id or row.get("status") != "ok" or row.get("image_verdict") not in VERDICTS:
            raise ValueError("duplicate or invalid prediction")
        ms = row.get("runtime_ms")
        if isinstance(ms, bool) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
            raise ValueError("positive measured runtime_ms required")
        by_id[key] = row
    frames = cohort["frames"]
    if len({r["frame_id"] for r in frames}) != len(frames) or len({r["sha256"] for r in frames}) != len(frames):
        raise ValueError("cohort must contain unique full frames")
    if set(by_id) != {r["frame_id"] for r in frames}:
        raise ValueError("prediction IDs differ from frozen cohort")
    counts = dict(tp=0, fp=0, fn=0, tn=0)
    flags = 0
    for frame in frames:
        prediction = by_id[frame["frame_id"]]
        if prediction.get("sha256") != frame["sha256"]:
            raise ValueError("prediction content differs from frozen frame")
        positive = prediction["image_verdict"] == "violation"
        flags += positive
        target = truth(frame)
        if target is None:
            continue
        key = ("tp" if target == "violation" else "fp") if positive else ("fn" if target == "violation" else "tn")
        counts[key] += 1
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {**counts, "alarm_frames": len(frames), "violation_frames": flags,
            "scored_frames": tp+fp+fn+tn, "precision": ratio(tp, tp+fp), "recall": ratio(tp, tp+fn),
            "f1": ratio(2*tp, 2*tp+fp+fn), "violation_rate": ratio(flags, len(frames)),
            "alarm_reduction": ratio(len(frames)-flags, len(frames)),
            "runtime_ms": ratio(sum(r["runtime_ms"] for r in by_id.values()), len(frames))}


def evaluate(cohort, candidate, baseline, evidence):
    """Fail closed when required measured evidence is absent or undefined.

    Evidence is supplied by validated adapter/replay workers, never a UI user.
    Class metrics need an independent held-out box-labelled set; binary VLM
    verdicts cannot produce per-object person/helmet precision and recall.
    """
    new, old = score(cohort, candidate), score(cohort, baseline)
    checks = {}
    def check(name, passed, reason):
        checks[name] = {"passed": bool(passed), "reason": reason}
    def finite(value):
        return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)
    def compare(a, b, tolerance):
        return finite(a) and finite(b) and a + 1e-12 >= b - tolerance
    check("dataset_integrity", evidence.get("snapshot_complete") is True and evidence.get("leakage_checked") is True
          and evidence.get("class_names") == {"0": "person", "1": "helmet"}, "completed snapshot, verified leakage exclusion and hero class map required")
    check("paired_replay", bool(evidence.get("candidate_version") and evidence.get("baseline_version"))
          and evidence.get("candidate_version") != evidence.get("baseline_version"),
          "all frozen frame IDs and hashes matched; distinct model version identities required")
    check("cohort_size", new["scored_frames"] >= 1000 and new["tp"]+new["fn"] >= 100 and new["fp"]+new["tn"] >= 100,
          "requires 1000 scored frames and 100 of each truth verdict")
    for metric, tolerance in (("recall", .02), ("precision", 0), ("f1", 0)):
        check("violation_"+metric, compare(new[metric], old[metric], tolerance), f"candidate >= baseline minus {tolerance}")
    reduction = ratio(old["violation_frames"]-new["violation_frames"], old["violation_frames"])
    check("alarm_reduction", finite(reduction) and reduction >= .05-1e-12, "requires at least 5% fewer alerts; undefined baseline cannot pass")
    for label in ("person", "helmet"):
        for metric in ("precision", "recall", "f1"):
            a = evidence.get("candidate_detection", {}).get(label, {}).get(metric)
            b = evidence.get("baseline_detection", {}).get(label, {}).get(metric)
            valid = finite(a) and finite(b) and 0 <= a <= 1 and 0 <= b <= 1
            check(f"{label}_{metric}", valid and compare(a, b, .02), "independent box metrics required; maximum 2 percentage-point loss")
    check("operational_health", evidence.get("artifacts_verified") is True and evidence.get("same_runtime_config") is True,
          "artifacts and identical replay configuration must be verified")
    check("performance", finite(new["runtime_ms"]) and finite(old["runtime_ms"]) and new["runtime_ms"] <= old["runtime_ms"]*1.1+1e-12,
          "candidate runtime <= 110% of baseline")
    passed = all(item["passed"] for item in checks.values())
    return {"schema_version": "1.0.0", "cohort_id": cohort["cohort_id"],
            "candidate_version": evidence.get("candidate_version"), "baseline_version": evidence.get("baseline_version"), "candidate_metrics": new,
            "baseline_metrics": old, "alerts_reduced_vs_current": reduction,
            "gate_result": {"passed": passed, "checks": checks}, "recommended_status": "uat" if passed else "candidate"}
