"""Export settled cases as a YOLO training dataset for Track2.

WHAT IS EXPORTED, AND WHY IT IS A CROP

Each exported sample is the *person crop* the judge was shown, not the
original frame. That is not a convenience: 91.3% of the frames in the source
dataset contain more than one person (7716 of 8455 label files), and this
repo only ever kept the largest `person` box per case -- the rest of each
source label file was discarded at prepare time. Exporting a full frame with
only the alarm subject labelled would therefore teach the model that every
other worker in shot is background, making four classes worse to improve one.
Inside a person crop there is exactly one person (the alarm subject, whose
box we have) and at most one helmet (the one a reviewer drew or the detector
found), so nothing plausible is left unlabelled.

The crop is a rendering decision made HERE. Reviews and audit rows store
boxes in original frame pixels, so moving to full frames later, or to a
different crop rule, needs no migration and no review redone.

LABEL FORMAT

Ultralytics YOLO, which is one format across YOLOv5/v8/v11 -- so a dataset
written for v8 trains v11 unchanged. One `.txt` per image, one line per box:

    <class> <x_center> <y_center> <width> <height>

all normalized 0-1 against the image (here, the crop). Verified against the
real dataset's own labels: five fields, no confidence column, no polygons.

Class indices are the SIX-class map from the real `data.yaml`, even though
only two ever appear here:

    0 helmet   1 gloves   2 person   3 working_boots   4 man_cage   5 safe_area

Keeping all six names, and helmet at 0 with person at 2, means these labels
stay index-compatible with the real dataset. Renumbering to a compact
0=helmet/1=person would silently corrupt any later merge.

BOX PROVENANCE

Every box records whether a human or a model produced it. A person box comes
from the CV detector that raised the alarm, so it is a pseudo-label:
reusing a model's own output as its training target reinforces what it
already believes. That is an acceptable trade for a class we are not trying
to fix (person is the best-represented class in the dataset), but it must not
be mistaken for verified ground truth -- hence `source` on every box in the
metadata, so Track2 can weight or re-check them separately.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from . import storage
from .config import PipelineConfig, get_config
from .cropping import crop_bytes
from .escalations import (
    crop_verdict,
    frame_available,
    is_exportable,
    needs_box,
    needs_verdict,
    _entry_for,
)
from .models import BoundingBox
from .track1_adapter import NO_PERSON_DETECTED_REASON, NO_PERSON_REASON

logger = logging.getLogger(__name__)

#: Where the dataset is written. Its OWN prefix, deliberately not merged into
#: the real training dataset: these are person crops with two classes
#: labelled, and dropping them among fully-labelled full frames would mix two
#: incompatible kinds of sample in one tree.
DEFAULT_EXPORT_PREFIX = "track2/escalation_dataset"

#: The real dataset's class map, from its own data.yaml. Kept whole so
#: indices stay compatible even though only helmet and person appear here.
CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "person",
    3: "working_boots",
    4: "man_cage",
    5: "safe_area",
}
HELMET_CLASS = 0
PERSON_CLASS = 2

#: Everything goes to `train`. These are corrections meant to be ADDED to a
#: dataset that has its own val split; holding some back would only shrink
#: the correction signal.
SPLIT = "train"
VAL_SPLIT = "val"
SPLITS = (SPLIT, VAL_SPLIT)

#: Share of SCENES held out for validation, not share of frames. See
#: `split_for`.
VAL_FRACTION = 0.2

#: Matches a frame number at the end of a case id, so the clip it came from
#: can be recovered: "Helmet_Hatch_21_frame_0099" -> "Helmet_Hatch_21".
_FRAME_SUFFIX = re.compile(r"^(.*?)[_ ]?frame[_ ]?[0-9]+$", re.IGNORECASE)


def scene_of(alarm_id: str) -> str:
    """The clip an alarm's frame came from, or the id itself if it has none.

    Frames from one clip are near-identical: the same worker, the same helmet,
    seconds apart.
    """
    match = _FRAME_SUFFIX.match(str(alarm_id))
    return match.group(1) if match else str(alarm_id)


def split_for(
    alarm_id: str, *, val_fraction: float = VAL_FRACTION, image_id: str | None = None
) -> str:
    """Which split a case belongs to. Deterministic, and held out by SCENE.

    Two properties matter, and a per-frame random split has neither.

    By scene, because ~91% of the source frames are multi-person sequences and
    consecutive frames of one clip are near-duplicates. Splitting per frame
    puts the same worker in both halves, so the val score measures
    memorisation: it reads highest exactly when the model has generalised
    least, which is worse than having no val split at all.

    From a hash rather than assigned at write time, because cases are exported
    ONE AT A TIME as they settle. There is no moment when the whole set is in
    hand to divide, so the assignment has to be a pure function of the case:
    stable across runs, insensitive to order, and identical whether a case is
    published alone or in a full re-sync.

    `image_id` OVERRIDES the alarm id, and exists because the alarm id is not
    the image's identity. The vendor re-alarms on one stale frame with a fresh
    id each time: measured over the 31,281 helmet snapshots the prefix held
    at the time, 3,120 images were alarmed more than once, and **1,402 of those
    duplicate groups straddled train and val** -- the same bytes on both sides
    of the split, which is exactly the leakage the scene rule exists to
    prevent, arriving by another door. Keyed on the image's content hash they
    cannot. The prefix now holds 75,242 keys and none of these counts have been
    re-measured against it; what they establish is that the leak is real, not
    how large it is today.

    Rows written before the hash was recorded have none, and fall back to the
    alarm id. Their split therefore does not move, which is deliberate: a case
    already published under one split must not silently change sides.
    """
    digest = hashlib.sha256((image_id or scene_of(alarm_id)).encode("utf-8")).hexdigest()
    return VAL_SPLIT if (int(digest[:8], 16) % 1000) < val_fraction * 1000 else SPLIT



@dataclass
class ExportedBox:
    class_id: int
    #: Normalized to the CROP: cx, cy, w, h in 0-1. This is what the .txt
    #: label file carries, because it is what a YOLO trainer reads.
    yolo: tuple[float, float, float, float]
    #: The same box in ORIGINAL FRAME pixels, [x1, y1, x2, y2]. Carried in the
    #: metadata as well because the frame is the canonical space -- the crop is
    #: only a rendering decision, so a consumer that wants to re-crop
    #: differently, draw the box on the full frame, or compare against the
    #: source dataset's own labels needs frame coordinates, not crop-relative
    #: ones.
    frame: tuple[float, float, float, float]
    #: The same box in CROP pixels, [x1, y1, x2, y2] -- the space the
    #: reviewer drew in and the space the exported image is in. Redundant with
    #: `frame` and `yolo` by construction, and included anyway: a consumer
    #: should never have to re-derive a coordinate transform to answer "where
    #: is this box on the image you gave me", and every re-derivation is a
    #: chance to get the rounding or the origin wrong.
    crop: tuple[float, float, float, float]
    #: "human" for a box a reviewer drew, "cv" for a model's own output --
    #: which is a pseudo-label, not verified ground truth.
    source: str


@dataclass
class ExportedCase:
    alarm_id: str
    image_name: str
    boxes: list[ExportedBox] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


@dataclass
class ExportResult:
    batch_id: str
    prefix: str
    cases: list[ExportedCase] = field(default_factory=list)
    #: Cases that were settled but could not be exported, with the reason --
    #: reported rather than silently dropped, so a small export is explicable.
    skipped: list[dict] = field(default_factory=list)

    @property
    def written(self) -> int:
        return len(self.cases)


def to_yolo(box: list[float], crop: BoundingBox) -> tuple[float, float, float, float]:
    """A frame-pixel box as YOLO `cx cy w h`, normalized to `crop`.

    Normalized against the crop's ROUNDED extent, because `crop_bytes` cuts
    on the rounded tuple -- so that is the exported image's real pixel size.
    Using the unrounded width would offset every label by up to a pixel, in a
    direction that varies per case.
    """
    cx1, cy1, cx2, cy2 = crop.as_tuple()
    width, height = cx2 - cx1, cy2 - cy1
    if width <= 0 or height <= 0:
        raise ValueError(f"crop has no area: {crop}")
    x1, y1, x2, y2 = box
    return (
        ((x1 + x2) / 2 - cx1) / width,
        ((y1 + y2) / 2 - cy1) / height,
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def to_crop_pixels(
    box: list[float], crop: BoundingBox
) -> tuple[float, float, float, float]:
    """A frame-pixel box expressed in the crop's own pixels.

    `crop_bytes` cuts without resizing, so this is a translation by the crop
    origin, not a scale -- and it uses the ROUNDED origin, because that is
    where the cut actually happened.
    """
    cx1, cy1, _, _ = crop.as_tuple()
    x1, y1, x2, y2 = box
    return (x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1)


def _clamped(values: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """YOLO coordinates must sit in 0-1. A label outside that is accepted
    silently by the trainer and learned as garbage, so drop the box instead."""
    if any(v != v for v in values):  # NaN
        return None
    cx, cy, w, h = values
    if w <= 0 or h <= 0:
        return None
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    return (
        min(max(cx, 0.0), 1.0),
        min(max(cy, 0.0), 1.0),
        min(w, 1.0),
        min(h, 1.0),
    )


def label_text(boxes: list[ExportedBox]) -> str:
    """The `.txt` body: one `class cx cy w h` line per box, 6dp.

    6 decimal places matches the source dataset's own labels, which is enough
    for sub-pixel precision on a 4K frame and avoids a diff that churns on
    float formatting.
    """
    lines = [
        f"{b.class_id} " + " ".join(f"{v:.6f}" for v in b.yolo)
        for b in boxes
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def data_yaml() -> str:
    """`data.yaml` in the real dataset's exact shape.

    Verified against the real file: `names` is an int-keyed MAP (not a list --
    Track2's `load_class_names` does `.items()` on it and would break on a
    list), plus `train`/`val`. No `nc:`, no `path:`, no `test:`.

    `val` points at the same directory as `train`. There is no val split here
    by design, but ultralytics requires the key, and pointing it at an empty
    or missing directory fails the run outright.
    """
    names = "\n".join(f"  {i}: {name}" for i, name in sorted(CLASS_NAMES.items()))
    return f"names:\n{names}\ntrain: images/{SPLIT}\nval: images/{VAL_SPLIT}\n"


def _settled_cases(
    rows: list[dict], reviews: dict[str, dict], config: PipelineConfig
) -> list[tuple[dict, dict]]:
    """One (row, review) pair per settled alarm, newest row winning.

    Settled means: nothing is waiting on a human. Either a reviewer decided
    it, or the pipeline settled it alone -- a confident verdict with a helmet
    box good enough that no human was asked.
    """
    newest: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        alarm_id = row.get("alarm_id")
        if not alarm_id or not isinstance(alarm_id, str):
            continue
        previous = newest.get(alarm_id)
        if previous is None or str(row.get("timestamp") or "") >= str(
            previous.get("timestamp") or ""
        ):
            newest[alarm_id] = row

    crowd: dict[str, int] = {}
    for row in newest.values():
        crowd[_case_key_of(row)] = crowd.get(_case_key_of(row), 0) + 1

    settled = []
    for alarm_id, row in newest.items():
        # BY THE ROW'S OWN ID, OR BY ITS IMAGE'S.
        #
        # An ordinary alarm is one row and one review, both under the same id.
        # A helmet-CV row is one PERSON, and the human decision about it is
        # filed against the IMAGE -- that is what the escalation queue keys
        # by, so a crop looked up by its own id would never find the verdict
        # a reviewer just gave for the picture it belongs to, and would sit
        # unsettled and unexported forever.
        own = _entry_for(reviews, alarm_id)
        case = _entry_for(reviews, _case_key_of(row))
        decided = own or case
        # THIS CROP'S OWN ANSWER, WHICH IS THE ONE THIS ROW IS ABOUT. A frame
        # answer settles the crops it covers; a crop answer settles exactly
        # one person, and settles them whatever the frame says. Read from the
        # review sidecar rather than the audit one for this reason: the gate
        # below is what decides whether a crop is exportable at all, so an
        # answer stored anywhere this cannot see is an answer that never
        # reaches the dataset.
        mine = crop_verdict(decided, row)
        if needs_verdict(row) and not (mine or decided.get("verdict")):
            continue
        if needs_box(row, config, decided) and not decided.get("helmet_box_status"):
            continue
        # A VERDICT ABOUT AN IMAGE IS ABOUT EVERYONE IN IT. A BOX IS NOT.
        #
        # "Nobody here was in violation" is true of every person in the
        # picture, which is why the settledness above may be answered by the
        # image's review. A drawn helmet box is one person's head, so carrying
        # an image-level box onto a second crop would label one worker's
        # helmet on another. It travels only when the image holds one person,
        # where the two are the same thing.
        entry = own or (case if crowd.get(_case_key_of(row), 1) <= 1 else {})
        # The crop's own answer rides on the entry, so the label, the
        # exportability filter and `human_verdict` in the metadata all name
        # what a human said about THIS person rather than about the picture.
        settled.append((row, {**entry, "verdict": mine} if mine else entry))
    return settled


def _box_list(box: object) -> list[float] | None:
    if isinstance(box, dict):
        try:
            return [float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])]
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(box, (list, tuple)) and len(box) == 4:
        try:
            return [float(v) for v in box]
        except (TypeError, ValueError):
            return None
    return None


def _helmet_for(row: dict, entry: dict) -> tuple[list[float] | None, str | None]:
    """The helmet box to export and who produced it.

    A reviewer's box wins over the model's: that is the whole point of the
    review. `no_helmet_visible` means a human looked and found none, so no
    helmet box is exported -- exporting a guessed one is exactly what that
    option exists to prevent.

    AND NO HELMET SHIPS ON A CROP ANSWERED `violation`, whatever box is on the
    row. "This person is in violation" is "this person has no helmet on their
    head", so a helmet box beside that label is a sample that contradicts
    itself. It became reachable when a crop answer stopped having to agree
    with the frame's: a crop the judge CLEARED, with a detector box on it,
    answered `violation` by a human exported as a violation carrying the
    detector's helmet -- box class 0 drawn on a head the label says is bare.
    Same rule and same reason as `no_helmet_present` below, and the same rule
    `build_frame_export` already applies ("a helmet box is taken only from a
    person the judge CLEARED"); a reviewer's own drawn box goes too, because
    their later answer is the one that stands.
    """
    if entry.get("verdict") == "violation":
        return None, None
    status = entry.get("helmet_box_status")
    if status in ("no_helmet_visible", "no_helmet_present"):
        # No helmet box either way. `no_helmet_present` still EXPORTS -- its
        # review carries a `violation` verdict, so the case ships as a person
        # with no helmet, which is exactly the negative a helmet detector
        # needs. What must not ship is the model's box: a reviewer has just
        # said there is no helmet, so drawing one from the row would train on
        # the very hallucination they rejected.
        return None, None
    human = _box_list(entry.get("helmet_box"))
    if status == "drawn" and human:
        return human, "human"
    if status == "accepted_cv":
        # The detector's own coordinates, with a human's endorsement behind
        # them. Reported as its own source rather than folded into either
        # "human" or "cv": it is more trustworthy than an unreviewed
        # pseudo-label and less precise than a hand-drawn box, and a
        # retraining run that wants to weight those differently can only do
        # so if the export says which it is.
        return _box_list(row.get("helmet_box")), "cv_accepted"
    return _box_list(row.get("helmet_box")), row.get("helmet_box_source") or None


#: An audit decision is about the IMAGE, and a training sample is one PERSON
#: in it. `audited_crop_label` is the whole of what can honestly be carried
#: across that gap.
#:
#: * The human CLEARED the image. Nobody in it was in violation, so that is
#:   true of every person in it. Every crop of the image inherits it.
#: * The human called the image a VIOLATION. At least one person in it was --
#:   not all of them, and the human never said which. The judge's attribution
#:   is the only pointer to who, and attribution is exactly what per-crop
#:   calls buy and what a batched call would lose. So the crops the judge
#:   fired on inherit "violation"; the rest inherit NOTHING and stay out of
#:   the dataset. A worker standing beside a violator is not thereby a
#:   violator, and is not thereby fine either.
#:
#: The alternative -- pushing the image's verdict onto every crop -- would
#: label every bystander in every violation frame as a violation. On the
#: pilot's measured 8.54 people per alarm frame that is up to seven wrong
#: labels per right one.
#:
#: NONE OF WHICH APPLIES WHEN THE HUMAN ANSWERED THE CROP ITSELF. Then there
#: is no gap to carry anything across, and this is the whole product argument
#: for `escalations.CROP_VERDICTS`: a reviewer resolving a hedge on a crowded
#: violation frame used to produce NO training sample for the person they had
#: just looked at, because that person did not fire and the frame's verdict
#: was not about them.
def audited_crop_label(
    image_verdict: str | None,
    crop_fired: bool,
    people_in_image: int = 1,
    crop_verdict: str | None = None,
) -> str | None:
    """What one person's crop inherits from the human's verdict on its image.

    ONE PERSON IN THE IMAGE AND THERE IS NO GAP: the verdict is about them.
    That is every live alarm and every gold case today, which is why this
    changes nothing for either. The gap opens only when several people share
    one image -- the helmet-CV arm, at a measured 8.54 of them per frame.

    `crop_verdict` is a human's answer about THIS person, and it inherits
    nothing: it is the label, whether or not the judge fired on the crop and
    whatever the image's verdict says. Including `cannot_determine`, which is
    an answer and not a label -- it is returned as itself so the caller skips
    it as unlabelled work rather than as a bystander nobody spoke for.
    """
    if crop_verdict:
        return crop_verdict
    if image_verdict is None:
        return None
    if people_in_image <= 1:
        return image_verdict
    if image_verdict == "no_violation":
        return "no_violation"
    return "violation" if crop_fired else None


def _is_excluded(row: dict, audits: dict[str, dict]) -> bool:
    """Whether an auditor struck this crop out of the training data.

    Keyed by candidate id, which is the only identifier a crop has that
    survives a re-run of its frame. A row with no candidate id (every live
    alarm) can never be excluded, which is right: the strike-out is offered
    on the crop strip, and a one-person alarm has no strip.
    """
    from .audit import excluded_people

    candidate = (row.get("cv") or {}).get("candidate_id")
    if not candidate:
        return False
    return str(candidate) in excluded_people(_entry_for(audits, _case_key_of(row)))


def _image_verdict(row: dict, audits: dict[str, dict]) -> str | None:
    """The human's verdict on the IMAGE this row's crop came from, if any."""
    from .audit import AUDIT_DECISIONS, reviewer_verdict

    entry = _entry_for(audits, _case_key_of(row))
    if entry.get("decision") not in AUDIT_DECISIONS:
        return None
    # The reviewer's own words, recovered from the agree/disagree encoding by
    # the function that owns that encoding -- decoding it a second time here
    # would be a second definition of what a human said.
    verdict = reviewer_verdict(entry)
    return verdict if verdict in ("violation", "no_violation") else None


def _image_id_of(row: dict) -> str | None:
    """The content identity of the image a row came from, if it was recorded.

    `None` for rows written before it was, and they fall back to the alarm id
    so a case already published does not silently change split.
    """
    lineage = row.get("lineage")
    etag = (lineage or {}).get("snapshot_etag") if isinstance(lineage, dict) else None
    return str(etag) if etag else None


def _case_key_of(row: dict) -> str:
    """The image key an audit decision about this row is filed under.

    The same rule as `audit.case_key`, read off the raw row rather than an
    `AuditItem` -- the export works in rows and importing the queue builder
    here would be a cycle. If one changes, so must the other; the audit
    module's version is the definition.
    """
    cv = row.get("cv") if isinstance(row.get("cv"), dict) else {}
    if cv:
        frame = cv.get("frame_name")
        if frame:
            from .helmet_cv.records import frame_stem

            return frame_stem(frame) or str(row.get("alarm_id") or "")
    lineage = row.get("lineage") if isinstance(row.get("lineage"), dict) else {}
    return str(lineage.get("frame_key") or row.get("alarm_id") or "")


def _crop_fired(row: dict) -> bool:
    """Whether the judge called THIS person a violation.

    The attribution per-crop calls exist to buy. A batched call could return
    the right frame answer attached to the wrong crop, which is why batching
    was deferred and why this is readable at all.
    """
    return row.get("outcome") == "CONFIRM_TP"


def build_export(
    rows: list[dict],
    reviews: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    now: datetime | None = None,
    only: set[str] | None = None,
    audits: dict[str, dict] | None = None,
) -> ExportResult:
    """Work out what to export, without writing anything.

    Separated from the write so the decisions are testable without storage,
    and so a caller can show what a run WOULD do before doing it.

    `audits` is the audit sidecar, keyed by CASE -- the image. It is how the
    live feed and the offline gold run converge here: their human decision is
    recorded against the image, not against each person in it, so without it
    a crop from either arm exports on the judge's word alone while a human
    answer about it sits unread. What a crop may inherit from an image is
    `audited_crop_label`, and a crop that may inherit nothing is skipped.

    Optional, and omitting it changes nothing for an ordinary alarm: those
    carry their human answer in `reviews`, keyed by the alarm itself.

    `only` restricts the work to specific alarm ids. Saving a verdict or a box
    changes exactly ONE case, and re-cropping and re-uploading every case
    already in the dataset to publish it makes each save slower than the last
    -- the cost grows with the dataset the tool exists to build. Because
    images and labels are written under stable per-case names, exporting a
    subset is not a partial export: it updates those cases and leaves the rest
    exactly as they were.
    """
    config = config or get_config()
    # Timestamp plus a uniqueness suffix. Second resolution alone collides:
    # two exports in the same second produce the same batch id, and the
    # second silently overwrites the first batch manifest -- losing the record
    # of what went out. The same collision the eval-run checkpoint filenames
    # already guard against, for the same reason.
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    result = ExportResult(
        batch_id=f"{stamp}-{uuid.uuid4().hex[:8]}", prefix=DEFAULT_EXPORT_PREFIX
    )

    audits = audits or {}
    settled = _settled_cases(rows, reviews, config)
    # How many people share each image, over the settled rows. The audit's
    # verdict is about the image, so how far it reaches depends on how many
    # people are standing in it.
    people_per_image: dict[str, int] = {}
    for row, _entry in settled:
        key = _case_key_of(row)
        people_per_image[key] = people_per_image.get(key, 0) + 1

    for row, entry in settled:
        alarm_id = row["alarm_id"]
        if only is not None and alarm_id not in only:
            continue

        def skip(reason: str) -> None:
            result.skipped.append({"alarm_id": alarm_id, "reason": reason})

        # STRUCK OUT BY THE AUDITOR. Checked before anything else about this
        # crop is considered: the auditor looked at this exact picture and
        # said it should not be trained on, which outranks everything the
        # pipeline can work out about it.
        if _is_excluded(row, audits):
            skip("excluded by the auditor")
            continue

        # Only where the gap is real: an image several people share, whose
        # human verdict names none of them. A one-person image is settled by
        # the ordinary review path above and is left exactly as it was.
        crowd = people_per_image.get(_case_key_of(row), 1)
        if crowd > 1:
            image_verdict = _image_verdict(row, audits)
            if image_verdict is not None and audited_crop_label(
                image_verdict,
                _crop_fired(row),
                crowd,
                # `_settled_cases` put this crop's own answer here when a
                # human gave it one, so it is the same string the metadata
                # will report as `human_verdict`.
                crop_verdict=entry.get("verdict"),
            ) is None:
                skip(
                    "a human called this image a violation and this person was not the one "
                    "the judge fired on, so there is no honest label for this crop"
                )
                continue

        if not frame_available(row.get("image_key")):
            skip("no stored frame: judged before alarm frames were persisted")
            continue
        crop_raw = _box_list(row.get("crop_box"))
        if not crop_raw:
            skip("no recorded crop region: judged before the crop was recorded")
            continue
        if entry and not is_exportable(entry) and (
            entry.get("verdict") or entry.get("helmet_box_status")
        ):
            skip("settled without a label (cannot determine / no helmet visible)")
            continue

        try:
            crop = BoundingBox(*crop_raw)
        except (TypeError, ValueError):
            skip("crop region is not a usable box")
            continue

        person = _box_list(row.get("person_box"))
        helmet, helmet_source = _helmet_for(row, entry)

        boxes: list[ExportedBox] = []
        if helmet:
            yolo = _clamped(to_yolo(helmet, crop))
            if yolo:
                boxes.append(
                    ExportedBox(
                        HELMET_CLASS,
                        yolo,
                        tuple(helmet),
                        to_crop_pixels(helmet, crop),
                        helmet_source if helmet_source in ("human", "cv_accepted") else "cv",
                    )
                )
        if person:
            yolo = _clamped(to_yolo(person, crop))
            if yolo:
                # The CV detector that raised the alarm produced this, so it
                # is a pseudo-label. Kept because omitting it would train the
                # person class to see this worker as background, which is
                # worse than reinforcing a box the model already believes.
                boxes.append(
                    ExportedBox(
                        PERSON_CLASS, yolo, tuple(person), to_crop_pixels(person, crop), "cv"
                    )
                )

        if not boxes:
            skip("no box survived normalization into the crop")
            continue

        result.cases.append(
            ExportedCase(
                alarm_id=alarm_id,
                image_name=f"{alarm_id}.jpg",
                boxes=boxes,
                metadata=_metadata_for(row, entry, crop, boxes),
            )
        )
    return result


#: The FULL-FRAME dataset's own prefix, beside the crop dataset in the same
#: bucket and deliberately not merged with it: one holds person crops with
#: two classes labelled, the other holds whole frames with every person in
#: them labelled. Same class indices, incompatible geometry -- mixed in one
#: tree they would train a model on two different definitions of "image".
FRAME_EXPORT_PREFIX = "track2/escalation_dataset_full_frame"


def build_frame_export(
    rows: list[dict],
    reviews: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    now: datetime | None = None,
    audits: dict[str, dict] | None = None,
    only: set[str] | None = None,
) -> ExportResult:
    """The FULL-FRAME dataset: one sample per image, in frame coordinates.

    WHY THIS IS NOW POSSIBLE, AND WAS NOT BEFORE

    This module's docstring argues at length that a sample must be a person
    CROP, because "this repo only ever kept the largest `person` box per
    case" -- so a full frame with one person labelled would teach the model
    that every other worker in shot is background, making four classes worse
    to improve one.

    That objection was about the DATA, and the data changed. The helmet-CV
    path runs a full-frame detector over the frame and asks the judge about
    EVERY person it finds, so a frame's rows now carry every person box in
    it, not the largest one. Nothing plausible is left unlabelled, which is
    the condition the crop rule existed to guarantee. Both exports stand:
    crops for the corrections a reviewer makes one person at a time, whole
    frames for a detector that has to find them all.

    WHAT IS IN IT (CEO, 1 Sep)

    * every person box in the frame, in FULL FRAME pixels;
    * MINUS the boxes the first-pass person gate emptied -- a box the VLM
      says holds no person is a box that would teach the detector to find
      ships (see `track1_adapter.NO_PERSON_REASON`);
    * plus the helmet boxes.

    A helmet box is taken only from a person the judge CLEARED. On a person
    it called a violation there is no helmet to box, and a model-drawn one
    there would contradict the verdict it shipped with.

    IT DOES NOT WAIT FOR A HUMAN, and it must not. A person box is the
    detector's output, so an escalated crop still contributes one -- and
    dropping those was measured to leave 105 of 128 frames labelling fewer
    people than the detector found. What it DOES wait for is completeness: a
    frame with a person who has no row at all is skipped, because an
    unlabelled worker in a whole-frame sample is a worker taught as
    background.

    CV ROWS ONLY. A live alarm carries one person box for a frame that may
    hold eight people, so exporting it as a full frame reintroduces exactly
    the unlabelled-worker problem -- those cases belong in the crop dataset,
    which is still there and still fed by `build_export`.

    WHAT AN AUDIT CHANGES (CEO: "auto changed if the audit decision differs
    from what was exported for that image")

    A human verdict that CONTRADICTS the judge's frame verdict drops the
    frame's helmet labels and keeps its person boxes. The asymmetry is the
    honest one: a person is a person whatever the verdict, but if a human
    says a frame holds a violation where the judge cleared everyone, then at
    least one of the helmet boxes it drew is wrong and nothing says which.
    Person boxes are also most of this dataset's value.

    `only` restricts the work to specific case ids, so settling one case
    republishes one frame rather than the whole set.
    """
    from .audit import excluded_people

    config = config or get_config()
    audits = audits or {}
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    result = ExportResult(
        batch_id=f"{stamp}-{uuid.uuid4().hex[:8]}", prefix=FRAME_EXPORT_PREFIX
    )

    # EVERY CV row of the frame, not only the settled ones.
    #
    # A person box comes from the DETECTOR, not from a verdict, so it does not
    # wait on a human. Built from `_settled_cases` this export dropped every
    # escalated crop: measured on the real log, 105 of 128 frames would have
    # shipped labelling FEWER people than the detector found, and alarm
    # 00e90bfd would have shipped 1 person out of 12 -- eleven workers
    # labelled background, which is the exact failure the crop rule exists to
    # prevent. Only the HELMET box depends on the verdict.
    newest: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not (row.get("cv") or {}):
            continue
        alarm_id = row.get("alarm_id")
        if not alarm_id or not isinstance(alarm_id, str):
            continue
        previous = newest.get(alarm_id)
        if previous is None or str(row.get("timestamp") or "") >= str(
            previous.get("timestamp") or ""
        ):
            newest[alarm_id] = row

    frames: dict[str, list[tuple[dict, dict]]] = {}
    for row in newest.values():
        frames.setdefault(_case_key_of(row), []).append(
            (row, _entry_for(reviews, row["alarm_id"]))
        )

    for case_id, members in sorted(frames.items()):
        if only is not None and case_id not in only:
            continue

        def skip(reason: str) -> None:
            result.skipped.append({"alarm_id": case_id, "reason": reason})

        # WHAT CERTIFIES A BOX AS HOLDING A PERSON.
        #
        # A first-pass VLM question used to, and was a hard PRECONDITION here:
        # a frame whose boxes carried no gate answer was refused, because the
        # boxes came from a DETECTOR and some of them held ships or background
        # lights. Exporting those would put a ship in the dataset labelled
        # `person`.
        #
        # THE GATE WAS REMOVED ON 3 SEP AND THIS REFUSED EVERYTHING. It had
        # been switched off since 1 Sep, so no row carried an answer, and this
        # check silently emptied the whole dataset -- measured on the live
        # corpus, 0 of 37 frames exported, every one skipped for boxes "not
        # checked for a person", pointing at a CLI command that no longer
        # exists. The full-frame dataset had been shipping nothing.
        #
        # Two things replaced it, and together they are stronger:
        #
        #  * boxes are no longer a detector's guess. They are recovered from
        #    the vendor CV's own annotation -- a rectangle it drew because it
        #    asserted a person there -- and `snapshot_boxes` confirms each one
        #    is a drawn rectangle rather than a red object;
        #  * an auditor can strike an individual crop out of the training data
        #    on the audit screen, which is a human answering the same question
        #    the gate was asked, on the crops that actually reach a person.
        #
        # `people_gated` and the rows the gate settled are still honoured
        # below, because rows written before 2 Sep carry them.
        # A BOX THE OLD GATE COULD NOT MAKE OUT is still not a `person` label.
        # Only rows written while the gate ran carry this, and no new row can:
        # kept because those rows are still in the corpus, and an `unclear`
        # box means the frame cannot be fully labelled -- an unlabelled worker
        # is a worker taught as background. MEASURED on alarm 00e90bfd: eleven
        # of twelve boxes came back that way at 37x40 to 67x39 pixels.
        unreadable = [
            r for r, _ in members if (r.get("cv") or {}).get("gate_answer") == "unclear"
        ]
        if unreadable:
            skip(
                f"{len(unreadable)} of {len(members)} boxes are too small or too blurred for "
                "the check to make out a person, so they cannot be labelled and the frame "
                "cannot be labelled without them"
            )
            continue

        # A FRAME MISSING PEOPLE IS NOT A FULL-FRAME SAMPLE. A judge call that
        # failed, or a live feed stopped between people, leaves a person with
        # no row at all -- and an unlabelled person in a whole-frame sample
        # teaches the detector that a worker is background. A crop the gate
        # emptied is NOT missing: it was answered, and its box is dropped on
        # purpose.
        detected = max(
            int((r.get("cv") or {}).get("people_in_frame") or 0) for r, _ in members
        )
        if detected and len(members) < detected:
            skip(
                f"the detector found {detected} people here and only {len(members)} were "
                "judged, so labelling this frame would leave a worker as background"
            )
            continue

        width = next((r.get("image_width") for r, _ in members if r.get("image_width")), None)
        height = next((r.get("image_height") for r, _ in members if r.get("image_height")), None)
        if not width or not height:
            skip("no recorded frame size, so a box cannot be normalized")
            continue
        frame_key = next(
            (r.get("image_key") for r, _ in members if frame_available(r.get("image_key"))), None
        )
        if not frame_key:
            skip("no stored frame")
            continue

        # What a human said about this image, against what the judge said.
        # Only a DISAGREEMENT changes the labels -- see the docstring.
        human_verdict = _image_verdict(members[0][0], audits)
        judged = {row.get("outcome") for row, _ in members}
        judge_verdict = (
            "violation"
            if "CONFIRM_TP" in judged
            else "no_violation" if "CONFIRM_FP" in judged else None
        )
        # ONE DIRECTION, not both, and the asymmetry is the whole point.
        #
        # A helmet box exists only on a crop the judge CLEARED as helmet_worn.
        # So which disagreement puts those boxes in doubt depends on which way
        # it runs:
        #
        #   judge cleared everyone, human says VIOLATION -> at least one crop
        #     the judge cleared is actually a violation, so its "helmet_worn"
        #     and the box drawn on it are wrong, and nothing says which crop.
        #     Every helmet box on the frame is suspect. Drop them all.
        #
        #   judge fired on someone, human says NO VIOLATION -> the human is
        #     overruling the crop that FIRED, and a fired crop carries no
        #     helmet box in the first place. The cleared crops are untouched
        #     by that finding, and the human agreeing nobody was in violation
        #     is consistent with their helmets being worn. Keep them.
        #
        # Treating both directions the same cost real labels: measured on
        # frame 1821747e, seven helmet boxes were dropped because a human
        # overruled the ONE crop that fired, which never had a box.
        helmets_disputed = bool(
            human_verdict == "violation" and judge_verdict == "no_violation"
        )

        # The whole image is the normalization target, so the same `to_yolo`
        # every crop label goes through does the arithmetic here too.
        whole = BoundingBox(0.0, 0.0, float(width), float(height))
        boxes: list[ExportedBox] = []
        gated = 0
        struck = excluded_people(_entry_for(audits, case_id))
        for row, entry in members:
            if row.get("reason_code") in (NO_PERSON_REASON, NO_PERSON_DETECTED_REASON):
                # The VLM looked and found no person. Labelling this box
                # would teach the detector to find ships.
                gated += 1
                continue
            # STRUCK OUT BY THE AUDITOR. Same effect as the gate and for the
            # same reason -- a box a human says holds nothing worth training
            # on teaches the detector something false. Counted with the gated
            # ones so a frame that loses every box still reports why.
            if str((row.get("cv") or {}).get("candidate_id") or "") in struck:
                gated += 1
                continue
            person = _box_list(row.get("person_box"))
            if person:
                yolo = _clamped(to_yolo(person, whole))
                if yolo:
                    boxes.append(
                        ExportedBox(
                            PERSON_CLASS, yolo, tuple(person), to_crop_pixels(person, whole), "cv"
                        )
                    )
            # A helmet box only where the judge CLEARED this person, and not
            # at all when a human has contradicted the frame's verdict.
            if helmets_disputed or row.get("outcome") != "CONFIRM_FP":
                continue
            helmet, helmet_source = _helmet_for(row, entry)
            if helmet:
                yolo = _clamped(to_yolo(helmet, whole))
                if yolo:
                    boxes.append(
                        ExportedBox(
                            HELMET_CLASS,
                            yolo,
                            tuple(helmet),
                            to_crop_pixels(helmet, whole),
                            helmet_source if helmet_source in ("human", "cv_accepted") else "cv",
                        )
                    )

        if not boxes:
            skip("every box in this frame was gated, struck out, or unusable")
            continue

        result.cases.append(
            ExportedCase(
                alarm_id=case_id,
                image_name=f"{case_id}.jpg",
                boxes=boxes,
                metadata={
                    "alarm_id": case_id,
                    "frame_key": frame_key,
                    "image_width": width,
                    "image_height": height,
                    "unit": "full_frame",
                    "people_labelled": sum(1 for b in boxes if b.class_id == PERSON_CLASS),
                    "helmets_labelled": sum(1 for b in boxes if b.class_id == HELMET_CLASS),
                    # Recorded, not silently dropped: how many of the
                    # detector's boxes the VLM emptied is the measurement that
                    # says whether the gate is worth its calls.
                    "boxes_gated": gated,
                    # True when a human verdict disagreed with the judge's, so
                    # the helmet labels came out and the person boxes stayed.
                    "helmets_disputed": helmets_disputed,
                    "human_verdict": human_verdict,
                    "split": split_for(case_id, image_id=_image_id_of(members[0][0])),
                    "boxes": [
                        {
                            "class_id": b.class_id,
                            "class_name": CLASS_NAMES[b.class_id],
                            "yolo": list(b.yolo),
                            "frame_box": list(b.frame),
                            "source": b.source,
                        }
                        for b in boxes
                    ],
                },
            )
        )
    return result


def write_frame_export(
    result: ExportResult,
    *,
    prefix: str | None = None,
    backend=None,
    on_case=None,
    partial: bool = False,
) -> dict:
    """Write the full-frame dataset. The image goes out WHOLE, uncropped.

    Same tree as the crop dataset and same `data.yaml`, under its own prefix:
    `images/{split}/`, `labels/{split}/`, a batch manifest and a cumulative
    index.

    `partial` has the same meaning as in `write_export`: one case being
    republished because somebody saved a decision, which is not a batch.
    """
    backend = backend or storage.get_storage()
    prefix = (prefix or result.prefix or FRAME_EXPORT_PREFIX).rstrip("/")

    total = len(result.cases)
    for done, case in enumerate(result.cases, start=1):
        frame = storage.frame_bytes(case.metadata["frame_key"], backend)
        split = case.metadata.get("split") or split_for(case.alarm_id)
        stem = Path(case.image_name).stem
        jobs: list[Callable[[], object]] = [
            lambda: backend.put_bytes(
                f"{prefix}/images/{split}/{case.image_name}", frame, content_type="image/jpeg"
            ),
            lambda: backend.put_bytes(
                f"{prefix}/labels/{split}/{stem}.txt",
                label_text(case.boxes).encode("utf-8"),
                content_type="text/plain",
            ),
        ]
        _in_parallel(
            jobs,
            [
                (lambda key=stale: backend.delete(key))
                for other in (sp for sp in SPLITS if sp != split)
                for stale in (
                    f"{prefix}/images/{other}/{case.image_name}",
                    f"{prefix}/labels/{other}/{stem}.txt",
                )
            ],
        )
        if on_case is not None:
            on_case(done, total)

    backend.put_bytes(f"{prefix}/data.yaml", data_yaml().encode("utf-8"), content_type="text/yaml")
    manifest = {
        "batch_id": result.batch_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "unit": "full_frame",
        "classes": {str(k): v for k, v in sorted(CLASS_NAMES.items())},
        "cases": [case.metadata for case in result.cases],
        "skipped": result.skipped,
    }
    if not partial:
        backend.put_bytes(
            f"{prefix}/batches/{result.batch_id}.json",
            json.dumps(manifest, indent=2).encode("utf-8"),
            content_type="application/json",
        )

    def merge(current: dict) -> dict:
        cases = dict(current.get("cases") or {})
        for case in result.cases:
            cases[case.alarm_id] = {
                "batch_id": result.batch_id,
                "image_name": case.image_name,
                "class_ids": sorted({b.class_id for b in case.boxes}),
                "people_labelled": case.metadata["people_labelled"],
                "helmets_labelled": case.metadata["helmets_labelled"],
            }
        batches = list(current.get("batches") or [])
        if not partial:
            batches.append({"batch_id": result.batch_id, "case_count": len(result.cases)})
        return {"cases": cases, "batches": batches}

    index = backend.update_json(f"{prefix}/index.json", merge, default={})

    # THE FULL RECORD PER CASE, cumulative, exactly as the crop dataset keeps
    # one. It used to live only in the batch manifests, so when a single-case
    # save stopped writing those (a save is not a batch) the geometry of
    # everything published that way had nowhere left to live. One round trip,
    # and it is what "show me this sample and its boxes" reads.
    def merge_metadata(current: dict) -> dict:
        cases = dict(current.get("cases") or {})
        for case in result.cases:
            cases[case.alarm_id] = {**case.metadata, "batch_id": result.batch_id}
        return {"cases": cases}

    backend.update_json(f"{prefix}/metadata.json", merge_metadata, default={})
    return {
        "prefix": prefix,
        "batch_id": result.batch_id,
        "written": len(result.cases),
        "skipped": len(result.skipped),
        "exported_total": len(index.get("cases") or {}),
    }


def _metadata_for(
    row: dict, entry: dict, crop: BoundingBox, boxes: list[ExportedBox]
) -> dict:
    """The per-case record Track2 gets alongside the labels.

    Everything needed to trace a label back to what produced it: which
    models, which human, what the judge said, and what the frame and crop
    were -- because a label whose provenance is unknown cannot be audited or
    retracted later.
    """
    human_verdict = entry.get("verdict")
    return {
        "alarm_id": row.get("alarm_id"),
        "image_name": f"{row.get('alarm_id')}.jpg",
        # Which split this case belongs to, derived from its scene. Recorded so
        # Track2 can see it without recomputing, and so the writer and the
        # manifest agree with each other.
        "split": split_for(str(row.get("alarm_id") or ""), image_id=_image_id_of(row)),
        "source_detection_id": row.get("source_detection_id"),
        "hazard_type": row.get("hazard_type"),
        "timestamp": row.get("timestamp"),
        # What the models concluded...
        "judge_outcome": row.get("outcome"),
        "judge_reason": row.get("reason_code"),
        "judge_reasoning": row.get("reasoning"),
        "judge_confidence": row.get("confidence"),
        "judge_provider": row.get("judge_provider"),
        "judge_model": row.get("judge_model"),
        "helmet_detector_weights": row.get("helmet_detector_weights"),
        "helmet_box_source": row.get("helmet_box_source"),
        "helmet_box_confidence": row.get("helmet_box_confidence"),
        # ...and what a human concluded, if one was asked. Both are recorded
        # so a disagreement between them stays visible downstream.
        "human_verdict": human_verdict,
        "human_box_status": entry.get("helmet_box_status"),
        "reviewed_by": entry.get("reviewed_by"),
        "reviewed_at": entry.get("reviewed_at"),
        "settled_by": "human" if (human_verdict or entry.get("helmet_box_status")) else "pipeline",
        # Geometry, in original frame pixels -- the canonical space.
        "frame_key": row.get("image_key"),
        "frame_width": row.get("image_width"),
        "frame_height": row.get("image_height"),
        # The crop's placement on the frame, and its own pixel size -- the
        # two facts needed to move between the spaces above in either
        # direction.
        "crop_box": list(crop.as_tuple()),
        "crop_width": crop.as_tuple()[2] - crop.as_tuple()[0],
        "crop_height": crop.as_tuple()[3] - crop.as_tuple()[1],
        "boxes": [
            {
                "class_id": b.class_id,
                "class_name": CLASS_NAMES[b.class_id],
                # Who produced it. "cv" is a pseudo-label -- the model's own
                # output reused as its training target -- not verified truth.
                "source": b.source,
                # The same box in every space it exists in, so nothing
                # downstream has to re-derive a transform:
                #   frame -- original full-frame pixels, [x1,y1,x2,y2]
                #   crop  -- pixels of the exported image, [x1,y1,x2,y2]
                #   yolo  -- crop-normalized cx/cy/w/h, i.e. the .txt line
                "box_frame": [round(v, 2) for v in b.frame],
                "box_crop": [round(v, 2) for v in b.crop],
                "box_yolo": [round(v, 6) for v in b.yolo],
            }
            for b in boxes
        ],
    }


def _in_parallel(
    required: list[Callable[[], object]],
    optional: list[Callable[[], object]] = (),
) -> None:
    """Run independent storage writes at once.

    A publish is a handful of round trips to DIFFERENT keys, and against this
    bucket each costs 110-260ms. Sequentially that is most of what a save
    spends. Nothing here reads what another writes, so the only ordering that
    matters is that the index update comes after, and it still does.

    `required` RAISES. Writing the image or the label is the publish, and a
    helper that swallowed those would turn a failed upload into a case that
    silently never reached the dataset while the screen said it had. The
    caller already turns an exception into a reported sync failure.

    `optional` does not. Deleting a stale copy from the other split is
    expected to 404 on almost every case, because almost every case has only
    ever been in one split.
    """
    jobs = list(required) + list(optional)
    if not jobs:
        return
    with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        futures = [(pool.submit(job), index < len(required)) for index, job in enumerate(jobs)]
        failure = None
        for future, is_required in futures:
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001 - see above
                if is_required:
                    failure = failure or exc
                else:
                    logger.debug("a stale copy could not be removed", exc_info=True)
        if failure is not None:
            raise failure


def write_export(
    result: ExportResult,
    *,
    prefix: str | None = None,
    backend=None,
    on_case: Callable[[int, int], None] | None = None,
    partial: bool = False,
) -> dict:
    """Write the dataset tree, the metadata JSON and a batch manifest.

    Additive by construction: images and labels are written per case under a
    stable name, and the manifest lists what THIS batch contained. Nothing
    already at the prefix is removed, so exporting twice tops the dataset up
    rather than replacing it.

    `partial` says this is ONE CASE being republished because somebody saved a
    decision, not a publish of the dataset. Such a save is not a batch: it
    writes no batch manifest and appends nothing to the index's batch list,
    which had grown to 92 entries for 72 cases, each one making the index
    slower to read and write. What it DOES write is the case's own image,
    label and index entry, which is the whole of what changed.
    """
    backend = backend or storage.get_storage()
    prefix = (prefix or result.prefix).rstrip("/")

    # `on_case(done, total)` after each case. A full publish is ~3 storage
    # round trips per case, so a 130-case re-sync runs for minutes: without a
    # tick there is nothing to show but a spinner, which reads as hung.
    total = len(result.cases)
    for done, case in enumerate(result.cases, start=1):
        frame_key = case.metadata["frame_key"]
        crop = BoundingBox(*case.metadata["crop_box"])
        frame = storage.frame_bytes(frame_key, backend)
        cropped, _ = crop_bytes(frame, crop)
        split = case.metadata.get("split") or split_for(case.alarm_id)
        stem = Path(case.image_name).stem
        jobs: list[Callable[[], object]] = [
            lambda: backend.put_bytes(
                f"{prefix}/images/{split}/{case.image_name}", cropped, content_type="image/jpeg"
            ),
            lambda: backend.put_bytes(
                f"{prefix}/labels/{split}/{stem}.txt",
                label_text(case.boxes).encode("utf-8"),
                content_type="text/plain",
            ),
        ]
        # Remove any copy sitting in the OTHER split. Re-publishing is routine
        # (a correction, a re-sync, a re-run), and a case that has ever been
        # written under a different split -- including everything exported
        # before this file had a val split at all -- would otherwise appear in
        # both, which is precisely the leakage the split exists to prevent.
        _in_parallel(
            jobs,
            [
                (lambda key=stale: backend.delete(key))
                for other in (sp for sp in SPLITS if sp != split)
                for stale in (
                    f"{prefix}/images/{other}/{case.image_name}",
                    f"{prefix}/labels/{other}/{stem}.txt",
                )
            ],
        )
        if on_case is not None:
            on_case(done, total)

    # Written every time, cheaply and unconditionally. A process-level "already
    # written" cache saved one 110ms round trip and outlived a storage backend
    # swap, which is a correctness hazard for 110ms.
    backend.put_bytes(f"{prefix}/data.yaml", data_yaml().encode("utf-8"), content_type="text/yaml")

    manifest = {
        "batch_id": result.batch_id,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "classes": {str(k): v for k, v in sorted(CLASS_NAMES.items())},
        "splits": sorted(
            {case.metadata.get("split") or SPLIT for case in result.cases}
        ),
        "case_count": result.written,
        "cases": [case.metadata for case in result.cases],
        "skipped": result.skipped,
    }
    if not partial:
        backend.put_bytes(
            f"{prefix}/batches/{result.batch_id}.json",
            json.dumps(manifest, indent=2).encode("utf-8"),
            content_type="application/json",
        )

    # A cumulative index, so the audit tab can ask "what has ever been
    # exported" without listing and parsing every batch file.
    def merge(current: dict) -> dict:
        cases = dict(current.get("cases") or {})
        for case in result.cases:
            cases[case.alarm_id] = {
                "batch_id": result.batch_id,
                "image_name": case.image_name,
                "class_ids": sorted({b.class_id for b in case.boxes}),
            }
        batches = list(current.get("batches") or [])
        if not partial:
            batches.append({"batch_id": result.batch_id, "case_count": result.written})
        return {"cases": cases, "batches": batches}

    index = backend.update_json(f"{prefix}/index.json", merge, default={})

    # One file with the full record for every case ever exported. The batch
    # files are the history of what went out when; this is the thing to read
    # when the question is "everything you know about this dataset" -- without
    # it a consumer has to list and merge every batch file to answer that.
    def merge_metadata(current: dict) -> dict:
        cases = dict(current.get("cases") or {})
        for case in result.cases:
            cases[case.alarm_id] = {**case.metadata, "batch_id": result.batch_id}
        return {
            "classes": {str(k): v for k, v in sorted(CLASS_NAMES.items())},
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(cases),
            "cases": cases,
        }

    metadata = backend.update_json(f"{prefix}/metadata.json", merge_metadata, default={})
    return {
        "manifest": manifest,
        "exported_total": len(index.get("cases") or {}),
        "metadata_case_count": metadata.get("case_count", 0),
    }


def retract_case(alarm_id: str, *, prefix: str | None = None, backend=None) -> bool:
    """Pull one case out of the dataset: its image, its label, its metadata.

    Needed because an audit can prove an exported label wrong. Leaving it in
    place would keep training on a sample a human has since said is
    incorrect, which is worse than having no sample -- so a correction that
    cannot produce a right label removes the wrong one instead.

    Returns whether anything was there to remove. Idempotent: retracting a
    case twice is not an error, since the caller's intent (it must not be in
    the dataset) is satisfied either way.
    """
    backend = backend or storage.get_storage()
    prefix = (prefix or DEFAULT_EXPORT_PREFIX).rstrip("/")
    image_name = f"{alarm_id}.jpg"

    removed = False
    # Both splits: a case's split is derived, and retraction must not depend on
    # recomputing it correctly.
    for key in (
        *(f"{prefix}/images/{sp}/{image_name}" for sp in SPLITS),
        *(f"{prefix}/labels/{sp}/{Path(image_name).stem}.txt" for sp in SPLITS),
    ):
        try:
            # `delete` reports whether the key was actually there, which is
            # what the caller is told; setting this unconditionally made the
            # return value always True.
            if backend.delete(key):
                removed = True
        except Exception:  # noqa: BLE001 - absent is the desired end state
            logger.info("nothing to retract at %s", key)

    def drop(current: dict) -> dict:
        cases = dict(current.get("cases") or {})
        cases.pop(alarm_id, None)
        return {**current, "cases": cases, "case_count": len(cases)}

    backend.update_json(f"{prefix}/metadata.json", drop, default={})
    backend.update_json(f"{prefix}/index.json", drop, default={})
    backend.update_json(f"{prefix}/metadata.json", drop, default={})
    return removed


def retract_frame(case_id: str, *, prefix: str | None = None, backend=None) -> bool:
    """Pull one FRAME out of the full-frame dataset. Idempotent."""
    backend = backend or storage.get_storage()
    prefix = (prefix or FRAME_EXPORT_PREFIX).rstrip("/")
    image_name = f"{case_id}.jpg"

    removed = False
    for key in (
        *(f"{prefix}/images/{sp}/{image_name}" for sp in SPLITS),
        *(f"{prefix}/labels/{sp}/{Path(image_name).stem}.txt" for sp in SPLITS),
    ):
        try:
            if backend.delete(key):
                removed = True
        except Exception:  # noqa: BLE001 - absent is the desired end state
            logger.info("nothing to retract at %s", key)

    def drop(current: dict) -> dict:
        cases = dict(current.get("cases") or {})
        cases.pop(case_id, None)
        return {**current, "cases": cases}

    backend.update_json(f"{prefix}/index.json", drop, default={})
    return removed


def sync_frame(
    case_id: str,
    rows: list[dict],
    reviews: dict[str, dict],
    *,
    audits: dict[str, dict] | None = None,
    config: PipelineConfig | None = None,
    prefix: str | None = None,
    backend=None,
) -> dict:
    """Publish ONE frame, or retract it if it no longer yields a label.

    The full-frame dataset's half of "the data is auto exported if the VLM
    settled, and after the escalation and helmet box settled, and also auto
    changed if the audit decision differs from what was exported for that
    image" (CEO). Reached from `sync_case`, so every existing settle-and-
    publish path feeds both datasets, and from the audit endpoint, where a
    decision can change what a frame's labels should be.

    RETRACTS when the rebuild yields nothing. A frame that used to export and
    now cannot -- every box gated, the frame gone -- must come OUT rather than
    keep the version it had, which is the whole point of re-syncing on a
    decision.

    Never raises: a failed publish must not lose the decision that triggered
    it, and the full re-sync button republishes everything.
    """
    try:
        result = build_frame_export(
            rows, reviews, config=config, audits=audits, only={case_id}
        )
        if not result.cases:
            removed = retract_frame(case_id, prefix=prefix, backend=backend)
            return {"synced": 0, "case_id": case_id, "retracted": removed}
        written = write_frame_export(result, prefix=prefix, backend=backend, partial=True)
        return {
            "synced": written["written"],
            "case_id": case_id,
            "exported_total": written["exported_total"],
        }
    except Exception as exc:  # noqa: BLE001 - never lose a decision to a sync
        logger.exception("full-frame sync of %s failed", case_id)
        return {"synced": 0, "case_id": case_id, "error": str(exc)}


def load_export_index(prefix: str | None = None, backend=None) -> dict[str, dict]:
    """Every alarm ever exported, keyed by alarm id. `{}` before the first."""
    backend = backend or storage.get_storage()
    prefix = (prefix or DEFAULT_EXPORT_PREFIX).rstrip("/")
    value, _ = backend.read_json(f"{prefix}/index.json", default={}, revalidate=True)
    cases = value.get("cases") if isinstance(value, dict) else None
    return cases if isinstance(cases, dict) else {}


def _frame_result(future) -> dict | None:
    """The full-frame publish's result, waited for.

    `sync_frame` never raises, so this cannot either; the guard is for a
    future that failed before it ever ran.
    """
    if future is None:
        return None
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 - a sound decision outweighs a sync
        logger.exception("full-frame sync failed")
        return {"synced": 0, "error": str(exc)}


def sync_case(
    alarm_id: str,
    rows: list[dict],
    reviews: dict[str, dict],
    *,
    config: PipelineConfig | None = None,
    prefix: str | None = None,
    backend=None,
    audits: dict[str, dict] | None = None,
) -> dict:
    """Publish ONE case, if it is settled and yields a label.

    The single definition of "a case reaches the dataset as soon as it is
    settled". Settling happens in two places, and only one of them used to
    publish: a human saving a verdict or a box, and the JUDGE itself, when a
    case needs no human at all (a confirmed violation, or a clearance whose
    detector box is confident enough that nobody is asked). The second kind is
    the majority of cases, and it reached the dataset only as a side effect of
    the next human save republishing everything. Once that republish was
    narrowed to the case that changed, the side effect went with it and
    auto-settled cases stopped being published at all.

    Scoped to one case so the cost of a save stays flat as the dataset grows.
    Never raises: a failed publish must not lose the decision that triggered
    it, and `POST /escalations/export` re-publishes everything.
    """
    frame_synced = None
    row = next(
        (r for r in rows if isinstance(r, dict) and r.get("alarm_id") == alarm_id), None
    )
    # THE ID MAY NAME AN IMAGE RATHER THAN A ROW. The escalation queue shows
    # one item per image and keys its decisions by the frame, so what arrives
    # here after a verdict is saved is a case key, matching no row's
    # `alarm_id` at all -- and this used to publish nothing for it.
    members = [
        r
        for r in rows
        if isinstance(r, dict) and (r.get("cv") or {}) and _case_key_of(r) == alarm_id
    ]
    if row is None and members:
        row = members[0]
    only = {r["alarm_id"] for r in members if r.get("alarm_id")} or {alarm_id}
    # THE TWO DATASETS PUBLISH CONCURRENTLY.
    #
    # The same settle feeds the FULL-FRAME dataset. Done here rather than at
    # each call site so every path that publishes a crop on settle -- the
    # judge's own, a verdict, a box -- publishes the frame too, and no new one
    # can forget to.
    #
    # Run one after the other this was the bulk of a save: measured on the
    # deployed box, the frame publish alone took 2.8s of a 3.8s sync, almost
    # all of it S3 round trips at 110-150ms each. They touch different
    # prefixes and share nothing, so waiting for one before starting the other
    # bought nothing but latency.
    frame_future = None
    pool = None
    if row is not None and (row.get("cv") or {}):
        pool = ThreadPoolExecutor(max_workers=1)
        frame_future = pool.submit(
            sync_frame, _case_key_of(row), rows, reviews, audits=audits, config=config
        )
    try:
        result = build_export(rows, reviews, config=config, only=only, audits=audits)
        if not result.cases:
            # Not settled yet, or settled without a label. Both are normal.
            return {"synced": 0, "alarm_id": alarm_id, "frame": _frame_result(frame_future)}
        written = write_export(result, prefix=prefix, backend=backend, partial=True)
        return {
            "synced": result.written,
            "alarm_id": alarm_id,
            "frame": _frame_result(frame_future),
            "exported_total": written["exported_total"],
        }
    except Exception as exc:  # noqa: BLE001 - a sound decision outweighs a sync
        logger.exception("dataset sync for %s failed", alarm_id)
        return {
            "synced": 0,
            "alarm_id": alarm_id,
            "error": str(exc),
            "frame": _frame_result(frame_future),
        }
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
