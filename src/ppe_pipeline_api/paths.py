"""Shared, env-overridable filesystem/storage-key locations
(`EVAL_DATA_DIR`, `FEWSHOT_DATA_DIR`, `EVAL_RUNS_DIR`,
`TRACK1_AUDIT_LOG_PATH`). Kept in one place so every router agrees on the
same paths.

Almost everything here is a STORAGE KEY, resolved through
`ppe_pipeline.storage` so the same name works against the local backend and
against S3. The helmet-CV frames are the exception and are resolved as real
filesystem paths — see `helmet_cv_frame_path`.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

EVAL_RUNS_DIR = Path(os.environ.get("PPE_EVAL_RUNS_DIR", "eval_runs"))
EVAL_DATA_DIR = os.environ.get("PPE_EVAL_DATA_DIR", "eval")
FEWSHOT_DATA_DIR = os.environ.get("PPE_FEWSHOT_DATA_DIR", "few_shot")
# A STORAGE KEY, not a filesystem path, so it is joined with "/" like every
# sibling below. `os.path.join` produced "eval	rack1_audit_log.json" on
# Windows, and S3 treats that as a DIFFERENT OBJECT from "eval/...": the API
# read the backslash key and got nothing while the CLI wrote the forward-slash
# one from its own default. The pilot's 643 judged rows were on disk and the
# ledger was empty, with no error anywhere.
TRACK1_AUDIT_LOG_PATH = f"{EVAL_DATA_DIR}/track1_audit_log.json"
GOLD_ANNOTATIONS_PATH_FOR_METRICS = f"{EVAL_DATA_DIR}/gold_annotations.json"
# Human decisions on escalated alarms and unusable helmet boxes. Kept
# beside the audit log it is a view over, never merged into it: the audit
# log is append-only history of what the models decided.
ESCALATION_REVIEWS_PATH = f"{EVAL_DATA_DIR}/escalation_reviews.json"
# Audit decisions ON settled cases -- kept apart from the reviews above
# because an audit judges a decision (including ones no reviewer made),
# and folding the two together would conflate "what was decided" with
# "whether that was right".
AUDIT_REVIEWS_PATH = f"{EVAL_DATA_DIR}/audit_reviews.json"
# Judged frames, persisted so a case can be reviewed and exported later.
ALARM_IMAGE_DIR = f"{EVAL_DATA_DIR}/alarm_images"

# Frames the helmet-CV pilot scored. A REAL DIRECTORY, not a storage key,
# and deliberately so: the helmet corpus at a 1s stride is ~1.6M jpegs
# written by `helmet_cv.cli extract` straight to local disk, and pushing
# them through `storage` would mean either uploading all of them or teaching
# the local backend a second root. Every other image this API serves is a
# storage key; this one is not, which is why it gets a resolver of its own
# rather than being interpolated into a key string at the call site.
HELMET_CV_DIR = Path(os.environ.get("PPE_HELMET_CV_DIR", os.path.join(EVAL_DATA_DIR, "helmet_frames")))

# A frame name is `site__camera__stamp__alarmid__fNNNNNN.jpg`, and the
# charset `extract.safe_field` produces is exactly this. Anything outside it
# cannot be one of our frames, so it is refused before it can be joined onto
# a path -- `frame_name` arrives as a URL path segment from the browser.
_SAFE_FRAME_NAME = re.compile(r"^[A-Za-z0-9._-]+$")
_FRAME_NAME_FIELDS = 5
_FRAME_NAME_ALARM_FIELD = 3


def helmet_cv_frame_path(frame_name: str) -> Path | None:
    """The on-disk path for one extracted frame, or `None` if `frame_name`
    is not a well-formed frame name.

    The alarm id is read out of the name rather than passed alongside it:
    frames are stored per alarm (`frames/<alarm_id>/<name>`) and the name
    already carries the id, so requiring both would let a caller pair a name
    with the wrong directory.

    Two guards, because this takes a browser-supplied path segment: the
    charset check rejects separators and `..` outright, and the containment
    check catches anything the charset check somehow lets through.
    """
    if not _SAFE_FRAME_NAME.match(frame_name):
        return None
    stem = frame_name.rsplit(".", 1)[0]
    fields = stem.split("__")
    if len(fields) != _FRAME_NAME_FIELDS:
        return None

    root = HELMET_CV_DIR.resolve()
    candidate = (root / "frames" / fields[_FRAME_NAME_ALARM_FIELD] / frame_name).resolve()
    if root not in candidate.parents:
        return None
    return candidate


def helmet_cv_extract_sidecar(alarm_id: str) -> Path | None:
    """`extract`'s per-alarm sidecar, which is where a frame's index, offset,
    UTC and pixel size live. The CV records do not repeat them."""
    if not _SAFE_FRAME_NAME.match(alarm_id):
        return None
    root = HELMET_CV_DIR.resolve()
    candidate = (root / "index" / f"{alarm_id}.json").resolve()
    if root not in candidate.parents:
        return None
    return candidate


SHADOW_MODE_AGREEMENT_TARGET = 0.85
LABEL_VOLUME_WEEKLY_TARGET = 400
