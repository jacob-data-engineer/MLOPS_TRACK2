"""Where the people are in one frame, so the VLM can be asked about each.

THE DETECTOR DECIDES NOTHING

It is a crop provider. It says where the people are; the judge says whether
each of them is in violation, and the human says who was right. There is no
gate here and no helmet-colour answer -- the ported JPSAI cascade was removed
once measurement showed it could not run on this footage at all (70% of its
candidates had no pose, because the workers are ~83px tall).

One consequence is worth stating plainly, because it bounds everything
downstream: **the detector's recall is now a ceiling on the VLM's recall.** A
person it does not find is a person the judge is never shown, and no metric
computed later can see that person. This is why `imgsz` is recorded.

THE UNIT IS ONE PERSON IN ONE FRAME

Not a clip, and not a frame. A 52-second `helmet_violation` clip contains
frames where the violation is plain, frames where the worker has walked out
of shot, and frames with three other people in them who are all wearing
helmets. Scoring at clip level would credit an answer nobody gave; scoring at
frame level would collapse four people into one verdict. So the atom is a
candidate:

    unk__unk__20260427T070224Z__726f88cb-...-676__f000450__p00
    └──────────── the frame that produced it ─────────────┘ └┘
                                                        person index

The person index is the detector's own output order for that frame, zero
padded to two digits. It is stable for a given run over given weights and
means nothing across runs -- there is no tracker, so it is an index, not an
identity.

Per-person verdicts are aggregated up to a frame answer by OR (any person in
violation makes the frame a violation), and the aggregation keeps hold of
WHICH person fired. That attribution is the reason the crops are judged one
call each rather than batched -- see `runner.judge_candidates`.

TWO FRAME ROLES, AND ONLY ONE OF THEM CV SPOKE ABOUT

`frame_role == "alarm"` is the frame production CV fired on. It is the only
frame CV ever claimed anything about, so it is the only frame where one human
verdict scores both arms. `clip` frames carry no CV claim and feed the VLM arm
alone, which is what lets the VLM have recall while CV has only precision.

The alarm frame comes from `raw_video/`, NOT from `snapshot/`. The snapshot
still has production's own box and the literal word "violation" drawn on it, so
feeding it to the judge would hand over the answer in text and showing it to a
reviewer would defeat the blind. It is used only to LOCATE the matching clean
frame -- see `extract._locate_alarm_frame`.

PROVENANCE TRAVELS WITH EVERY ROW

Each record carries the weights that produced it, the confidence it ran at,
and the inference resolution. Two runs whose rows are indistinguishable would
be two runs nobody can compare. `od_confidence` is marked as OUR value rather
than a ported one, because production's own inference confidence is not
recorded anywhere in the reference repo.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import re

from .thresholds import REFERENCE_PERSON_OD_CONFIDENCE_THRESHOLD

#: Bumped from 2 when the helmet cascade was removed: these records carry
#: person geometry and no gate decision, so a v2 sidecar and a v3 one are
#: not the same thing even where their keys overlap.
SCHEMA_VERSION = 3
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: The frame production CV fired on: a CLEAN frame out of `raw_video/`, located
#: by matching against the annotated `snapshot/` still. Named `alarm` rather
#: than `snapshot` on purpose -- the snapshot itself has CV's box and the word
#: "violation" drawn on it and must never reach the judge or a reviewer.
FRAME_ROLE_ALARM = "alarm"
#: Any other frame decoded out of the alarm's clip. Carries no CV claim; feeds
#: the VLM arm only.
FRAME_ROLE_CLIP = "clip"

#: Where `od_confidence` came from. Recorded as a string rather than left
#: implicit so nobody reads 0.25 off a record and cites it as production's.
OD_CONFIDENCE_SOURCE = "unknown_production_value"


def frame_stem(frame_name: str) -> str:
    """A frame name without its image suffix."""
    lowered = frame_name.lower()
    for suffix in (".jpg", ".jpeg", ".png"):
        if lowered.endswith(suffix):
            return frame_name[: -len(suffix)]
    return frame_name


def candidate_id(frame_name: str, person_index: int) -> str:
    """The id for one person in one frame — `<frame stem>__pNN`.

    `__` is the frame name's own field separator and the charset stays
    `[A-Za-z0-9._-]`, so a candidate id is still a legal `alarm_id` for
    `track1_adapter.safe_image_stem` and a legal storage key. That matters:
    the judge pass reuses the ordinary audit-log row, keyed by this.
    """
    return f"{frame_stem(frame_name)}__p{person_index:02d}"


def validate_cv_run_id(cv_run_id: str) -> str:
    if not RUN_ID_PATTERN.fullmatch(cv_run_id):
        raise ValueError(
            "cv_run_id must be 1-128 filesystem-safe characters: letters, digits, '.', '_' or '-'"
        )
    return cv_run_id


def execution_id(cv_run_id: str, ground_truth_id: str) -> str:
    """Identity of one model execution; labels remain keyed by ground truth."""
    digest = hashlib.sha256(ground_truth_id.encode("utf-8")).hexdigest()
    return f"{validate_cv_run_id(cv_run_id)}__g{digest}"


@dataclass(frozen=True)
class CvProvenance:
    """What one detection pass ran under.

    Only the detector remains. The pose checkpoint, the helmet-colour ratio,
    the limb-length ratio and both minimum person heights are gone with the
    cascade -- none of them influences a person box, which is all this pass
    now produces.
    """

    cv_run_id: str
    schema: int = SCHEMA_VERSION
    #: Where the person boxes came from. `vendor_cv` is the only value a new
    #: pass writes; `detector` appears on rows scored before 2 Sep, when we
    #: ran our own YOLO over the clean frame.
    box_source: str = "vendor_cv"
    #: The checkpoint that produced the boxes. Empty since 2 Sep: no detector
    #: runs, so naming one would be a fiction.
    detector_weights: str = ""
    #: Inference resolution the detector ran at. `None` means imgsz was never
    #: passed, i.e. ultralytics' 640. It matters more than it looks: the
    #: detector is now the VLM's only source of crops, so a person it misses
    #: at 640 is a person the judge is never asked about. Two runs at
    #: different resolutions are not comparable, and without this field
    #: nothing on the record would say which was which.
    imgsz: int | None = None
    od_confidence: float = 0.0
    #: Why that confidence is not a ported constant. See the module docstring.
    od_confidence_source: str = OD_CONFIDENCE_SOURCE
    #: The person threshold ACTUALLY APPLIED, which now follows `od_confidence`.
    #: Recorded because it changed on 31 Aug: rows scored before then were
    #: filtered at the reference's 0.5 and hold about a third fewer people.
    person_confidence_threshold: float = 0.0
    #: The reference's own 0.5, no longer applied. Kept so a row says plainly
    #: which regime produced it rather than leaving it to be inferred.
    reference_person_confidence_threshold: float = REFERENCE_PERSON_OD_CONFIDENCE_THRESHOLD

    @classmethod
    def from_config(cls, config, *, cv_run_id: str) -> CvProvenance:
        """What produced this pass's boxes.

        Every detector field is left at its empty default, and that is the
        honest answer since 2 Sep: no detector ran. Person boxes are recovered
        from the vendor CV's own annotated snapshot, so there is no checkpoint,
        no inference resolution and no confidence threshold to record -- the
        vendor publishes a box, not a score.

        The FIELDS stay because rows scored before that date carry real values
        in them, and dropping them would make an old row unreadable.
        """
        del config  # nothing here is configured any more
        return cls(cv_run_id=validate_cv_run_id(cv_run_id), box_source="vendor_cv")


@dataclass(frozen=True)
class CandidateRecord:
    """One person the detector found in one frame, and where they are.

    NOT a decision. The detector is a crop provider here, nothing more: it
    says where the people are so the VLM can be asked about each of them.
    Whether a person is in violation is the judge's answer and the human's,
    and neither lives on this record.

    A frame the detector found nobody in still produces exactly one of these,
    with `person_box` `None`, so the frame stays visible to the audit instead
    of silently leaving the corpus. That candidate is judged on the whole
    frame -- see `runner._judge_one_frame`.
    """

    candidate_id: str
    ground_truth_id: str
    execution_id: str
    frame_name: str
    relative_path: str
    alarm_id: str
    clip_key: str
    frame_utc: str
    #: Which frame of the alarm this is. `alarm` is the frame production CV
    #: actually fired on, so it is the ONLY frame CV made a claim about and the
    #: only one where one human verdict scores both arms. `clip` frames carry
    #: no CV claim and feed the VLM arm alone.
    frame_role: str
    person_index: int
    person_box: list[float] | None
    person_confidence: float | None
    #: Frame-level context, so a candidate can be read without reopening the
    #: frame: how crowded it was, and how many helmets were on offer. The
    #: helmet count is the detector's, and is context only -- nothing here
    #: pairs a helmet to a head any more.
    people_in_frame: int = 0
    #: `None` means NOT MEASURED, which is what the shipped person detector
    #: gives: stock COCO has no helmet class. `0` still means "the detector
    #: looked and found none in shot".
    #:
    #: The helmet box a verdict rests on never came from here anyway -- it
    #: comes from the stage-2 crop model (`helmet_box.weights`), per person.
    #: This was always frame-level context.
    helmets_in_frame: int | None = 0
    #: How many of this frame's people the first-pass person gate discarded.
    #: Carried so `people_in_frame` minus the judged rows can be told apart
    #: from a frame the judge never finished -- a gated crop is accounted
    #: for, a missing one is not. See `judge_client.PERSON_GATE_SYSTEM_PROMPT`.
    people_gated: int = 0
    #: Whether the first-pass person gate was asked about THIS crop.
    #:
    #: A positive marker, because `people_gated == 0` is ambiguous: it means
    #: both "the gate ran and emptied nothing" and "no gate ran at all". The
    #: full-frame export needs to tell those apart -- without it, a frame from
    #: a pre-gate run exports its ships and background lights as `person`
    #: boxes, which is the opposite of what the gate exists to prevent.
    gate_checked: bool = False
    #: The gate's own answer: `person`, `unclear`, or `no_person`. The
    #: difference between the last two decides two things differently -- an
    #: `unclear` crop is still JUDGED (it may hold a bent worker) and is never
    #: LABELLED (nobody can make out a body to label). See
    #: `judge_client.PERSON_GATE_SYSTEM_PROMPT`.
    gate_answer: str = ""
    #: Where this person's box came from: `detector` (our own model) or
    #: `vendor_cv` (recovered from the vendor's annotated snapshot -- the box
    #: that actually raised the alarm).
    #:
    #: Recorded because the two are not the same measurement and must never be
    #: pooled: "our detector versus the judge" and "the vendor's CV versus the
    #: judge" are different questions, and a corpus mixing them silently
    #: answers neither.
    box_source: str = "detector"
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    def cv_block(self) -> dict:
        """The `row["cv"]` block for the shared audit log.

        NESTED, never flattened into the row. The judge's own row already
        carries `confidence`, `helmet_box`, `helmet_box_source` and
        `helmet_box_confidence`; a flat box from here would overwrite the
        judge's and nobody would be able to tell which model drew it.

        Carries `cv_run_id` so these rows stay filterable out of a log that is
        otherwise live alarms -- see `audit.is_cv_row`.
        """
        return {
            "cv_run_id": self.provenance.get("cv_run_id"),
            "candidate_id": self.candidate_id,
            "ground_truth_id": self.ground_truth_id,
            "execution_id": self.execution_id,
            "frame_name": self.frame_name,
            "clip_alarm_id": self.alarm_id,
            "frame_role": self.frame_role,
            "person_index": self.person_index,
            "person_box": self.person_box,
            "person_confidence": self.person_confidence,
            "people_in_frame": self.people_in_frame,
            "helmets_in_frame": self.helmets_in_frame,
            "people_gated": self.people_gated,
            "gate_checked": self.gate_checked,
            "gate_answer": self.gate_answer,
            "box_source": self.box_source,
            "provenance": dict(self.provenance),
        }


def build_record(
    *,
    frame: dict,
    person_index: int,
    person_box: list[float] | None,
    person_confidence: float | None,
    people_in_frame: int,
    helmets_in_frame: int | None,
    provenance: CvProvenance,
    box_source: str = "detector",
) -> CandidateRecord:
    """Assemble one record from a detected person and its frame's manifest row."""
    frame_name = frame["frame_name"]
    stable_id = candidate_id(frame_name, person_index)
    return CandidateRecord(
        candidate_id=stable_id,
        ground_truth_id=stable_id,
        execution_id=execution_id(provenance.cv_run_id, stable_id),
        frame_name=frame_name,
        relative_path=frame.get("relative_path", ""),
        alarm_id=frame.get("alarm_id", ""),
        clip_key=frame.get("clip_key", ""),
        frame_utc=frame.get("frame_utc", ""),
        frame_role=frame.get("frame_role", FRAME_ROLE_CLIP),
        person_index=person_index,
        person_box=person_box,
        person_confidence=person_confidence,
        people_in_frame=people_in_frame,
        helmets_in_frame=helmets_in_frame,
        box_source=box_source,
        provenance=asdict(provenance),
    )
