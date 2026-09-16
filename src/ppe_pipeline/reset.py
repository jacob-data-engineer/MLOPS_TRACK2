"""Clearing everything one offline test run produced, so the next one starts clean.

WHY THIS EXISTS

While the pipeline is being exercised against the offline eval datasets,
every run leaves state behind on purpose: judged rows in the audit log,
persisted frames, human decisions in the review and audit sidecars, and a
YOLO dataset built out of the cases that settled. That accumulation is
exactly right in production and exactly wrong while testing, where a
half-finished previous run silently becomes the denominator of the next
one's metrics.

WHAT IT DELIBERATELY DOES NOT TOUCH

Only *produced* state is cleared. Anything a human authored or that is an
input to the pipeline survives, because losing it would mean losing work
that no re-run can regenerate:

* `eval/gold_annotations.json` -- reviewers' gold labels, the thing
  `compute_shadow_mode_agreement` scores against. Hand-made, irreplaceable.
* the eval datasets themselves (the case lists under `eval/`)
* the few-shot library
* saved eval-run reports under `PPE_EVAL_RUNS_DIR` -- those belong to the
  Eval screen, are named per run, and are not what accumulates here

A reset is therefore safe to repeat, and safe in the sense that matters: it
can destroy a test run's output, never a person's input.

IDEMPOTENCE

Running it twice is not an error -- the second run simply reports zeros.
Deletion continues past an individual failure and the failures are
reported, because a reset that aborted halfway would leave exactly the
mixed state it exists to prevent.
"""

from __future__ import annotations

from pathlib import Path

from . import storage, training_export

#: The word a caller must send to actually destroy anything. A reset wipes a
#: run's entire output, so it must not be reachable by a stray POST -- the
#: API refuses without it.
RESET_CONFIRMATION = "RESET"

_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _keys_under(prefix: str, suffixes: tuple[str, ...]) -> list[str]:
    """`list_keys` is non-recursive in both backends, so every directory that
    holds files gets asked for by name rather than assuming one sweep of the
    top-level prefix finds the images and labels nested below it."""
    try:
        return storage.get_storage().list_keys(prefix, suffixes=suffixes)
    except Exception:  # noqa: BLE001 - a missing prefix is "nothing to clear"
        return []


def _export_keys(prefix: str | None = None) -> list[str]:
    """Every key `training_export.write_export` can create, in one list."""
    root = (prefix or training_export.DEFAULT_EXPORT_PREFIX).rstrip("/")
    # Both splits: the dataset is held out by scene, so cases live under
    # train/ or val/ and a reset that swept only one would leave half behind.
    return [
        *(
            key
            for split in training_export.SPLITS
            for key in _keys_under(f"{root}/images/{split}", _IMAGE_SUFFIXES)
        ),
        *(
            key
            for split in training_export.SPLITS
            for key in _keys_under(f"{root}/labels/{split}", (".txt",))
        ),
        *_keys_under(f"{root}/batches", (".json",)),
        f"{root}/data.yaml",
        f"{root}/index.json",
        f"{root}/metadata.json",
    ]


def _exists(key: str) -> bool:
    try:
        storage.get_storage().get_bytes(key)
        return True
    except Exception:  # noqa: BLE001 - absent, unreadable: either way, nothing to delete
        return False


def _count_rows(key: str) -> int:
    value, _ = storage.get_storage().read_json(key, default=[])
    return len(value) if isinstance(value, (list, dict)) else 0


def preview_reset(
    *,
    audit_log_path: str,
    escalation_reviews_path: str,
    audit_reviews_path: str,
    alarm_image_dir: str,
    export_prefix: str | None = None,
) -> dict:
    """What a reset would destroy, counted but untouched.

    Exists so the confirmation can name real numbers -- "12 judged cases, 3
    reviews, 40 exported images" -- rather than asking someone to approve an
    abstraction.
    """
    export_keys = [k for k in _export_keys(export_prefix) if _exists(k)]
    return {
        "judged_cases": _count_rows(audit_log_path),
        "escalation_reviews": _count_rows(escalation_reviews_path),
        "audit_decisions": _count_rows(audit_reviews_path),
        "alarm_images": len(_keys_under(alarm_image_dir, _IMAGE_SUFFIXES)),
        "exported_files": len(export_keys),
        "confirmation": RESET_CONFIRMATION,
        # Named in the response rather than only in this docstring, so the UI
        # can tell someone what survives without duplicating the list.
        "preserved": [
            "gold annotations (reviewers' labels)",
            "eval datasets and the few-shot library",
            "saved eval-run reports",
        ],
    }


def reset_offline_state(
    *,
    audit_log_path: str,
    escalation_reviews_path: str,
    audit_reviews_path: str,
    alarm_image_dir: str,
    export_prefix: str | None = None,
    frame_export_prefix: str | None = None,
    live_feed_state_key: str | None = None,
) -> dict:
    """Clear one test run's output. Returns what actually went.

    The JSON sidecars are emptied rather than deleted: every reader of them
    already handles an empty collection, while a missing key is a case some
    of them would have to special-case, and a reset should not be the one
    thing that produces a state the rest of the app never otherwise sees.

    TWO THINGS THIS USED TO LEAVE BEHIND, both of which made "fresh run" not
    fresh:

    * the FULL-FRAME dataset. `export_prefix` only ever covered the crop
      dataset, so after a reset the second dataset still held frames whose
      judged rows had just been deleted -- labels with no case behind them.
    * the LIVE FEED's cursor. `processed_keys` and the watermark survived, so
      a "fresh" feed silently skipped every alarm the old run had consumed and
      then sat at zero looking broken.

    Both are cleared here, and both are reported in `cleared` so a reset says
    what it did rather than implying it.
    """
    backend = storage.get_storage()
    cleared = {
        "judged_cases": _count_rows(audit_log_path),
        "escalation_reviews": _count_rows(escalation_reviews_path),
        "audit_decisions": _count_rows(audit_reviews_path),
        "alarm_images": 0,
        "exported_files": 0,
        "frame_dataset_files": 0,
        "live_feed_reset": False,
    }
    failures: list[dict] = []

    for key, empty in (
        (audit_log_path, []),
        (escalation_reviews_path, {}),
        (audit_reviews_path, {}),
    ):
        try:
            # `update_json`, not a bare `write_json_cas`: the latter's
            # `expected_token=None` means "expect this key NOT to exist", so
            # against S3 it would raise on every file a reset actually has
            # something to clear. This takes the read-modify-write path with
            # its retries, and empties the file whatever token it is on.
            backend.update_json(
                Path(key).as_posix(), lambda _current, new=empty: new, default=empty
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": str(exc)})

    for key in _keys_under(alarm_image_dir, _IMAGE_SUFFIXES):
        try:
            if backend.delete(key):
                cleared["alarm_images"] += 1
        except Exception as exc:  # noqa: BLE001 - keep going; report at the end
            failures.append({"key": key, "error": str(exc)})

    for key in _export_keys(export_prefix):
        try:
            if backend.delete(key):
                cleared["exported_files"] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": str(exc)})

    # The full-frame dataset, which the crop prefix never covered.
    from .training_export import FRAME_EXPORT_PREFIX

    for key in _export_keys(frame_export_prefix or FRAME_EXPORT_PREFIX):
        try:
            if backend.delete(key):
                cleared["frame_dataset_files"] += 1
        except Exception as exc:  # noqa: BLE001
            failures.append({"key": key, "error": str(exc)})

    # The live feed's cursor. Emptied, not deleted, for the same reason as the
    # sidecars: every reader handles an empty state, and `status: "none"` is a
    # state the screen already knows how to draw.
    from .helmet_cv.live_feed import DEFAULT_STATE_KEY

    feed_key = live_feed_state_key or DEFAULT_STATE_KEY
    try:
        current, _ = backend.read_json(feed_key, default={})
        if current:
            backend.update_json(feed_key, lambda _c: {}, default={})
            cleared["live_feed_reset"] = True
    except Exception as exc:  # noqa: BLE001
        failures.append({"key": feed_key, "error": str(exc)})

    return {"cleared": cleared, "failures": failures}
