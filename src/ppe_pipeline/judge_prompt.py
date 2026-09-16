"""System prompt, reason codes, and the structured-output schema for the
batched LLM judge. One image, one call per person — the schema is built
per-request so `results` only asks about the violation types that actually
apply to that person.
"""

from __future__ import annotations

from .models import ViolationType

# Shared with the review UI so annotators judging a case see the literal
# same written policy the judge prompt applies — see VIOLATION_RULE_TEXT
# below.
DANGEROUS_PLANE_RULE_TEXT = """\
DANGEROUS PLANE — OPEN EDGES AND VOID GAPS AT ELEVATED HEIGHT
This is spelled "plane" as in a flat surface/height term, not "plain" —
always respond with exactly "dangerous_plane" in the violation_type field.

An open edge or void gap is a dangerous zone if the vertical drop below
it looks like it would reach roughly waist-height or higher on a person
standing at the bottom (a visual stand-in for "more than 0.5 metres,"
which can't be measured directly from a 2D crop) — for a void gap, this
only applies if the gap itself is wider than roughly a footprint, not a
narrow seam or expansion joint. NOT dangerous: an edge or gap backed by a
wall, bulkhead, or solid barrier rising to at least the person's torso
height — only exposed edges/gaps count, no matter how elevated the
surface is.

Anyone positioned within roughly one adult body-length of a dangerous
zone (a stand-in for "within 1.8 metres") must be wearing a harness with
a lifeline visibly connected to a fixed structural anchor point (a
railing, structural padeye, etc.) — not just clipped to something
portable or to their own equipment. Work through this in order:
  1. Not actually elevated / not near any open edge or void gap at all:
     "FP", reason "not_elevated".
  2. The edge/gap is wall-backed per above: "FP", reason
     "wall_backed_edge_exempt".
  3. A harness+lifeline is visibly worn AND visibly anchored to a fixed
     point: "FP", reason "adequate_fall_protection".
  4. Otherwise, within reach of an unprotected dangerous zone with no
     harness, or a harness not visibly anchored to a fixed point: "TP",
     reason "unprotected_at_height".
  5. Elevation, the edge/gap geometry, or the presence/anchoring of fall
     protection can't be told from the crop: "uncertain", reason
     "height_ambiguous".
"""

TOO_NEAR_FORKLIFT_RULE_TEXT = """\
TOO NEAR FORKLIFT — NEAR-VEHICLE REASONING
Do not attempt to judge distance directly — the crop's 2D perspective
doesn't support that. Instead work through this in order, for the person
being checked:
  1. Is this figure actually a real person — not a mannequin, poster,
     painted figure, reflection, or person-shaped equipment? If not:
     "FP", reason "not_a_real_person".
  2. Are they the forklift's own operator, or inside another vehicle? If
     so: "FP", reason "operator_in_cab".
  3. Is the vehicle stationary AND is no driver/operator visible inside
     it (empty cab, or clearly parked — chocked, cones/barriers placed,
     engine/lights off)? If so there is no proximity restriction at all,
     no matter how close the person is: "FP", reason
     "vehicle_unattended".
  4. Are they standing on a raised deck, platform, or stack of cargo at a
     different level from the vehicle? If so: "FP", reason
     "different_level".
  5. Is there a fence, wall, railing, or container stack physically
     between them and the vehicle? If so: "FP", reason
     "barrier_between".
  6. Otherwise, compare the gap between them and the nearest part of the
     vehicle — including any load, forks, or boom it is carrying — to
     the vehicle's own length. A vehicle that's visibly moving (mid-
     travel, forks/boom actively raising or in motion — look for motion
     blur, kicked-up dust/spray) needs a clearly larger gap than one
     that's stationary with a visible operator; if you can't tell
     whether it's moving, judge by the stricter (moving) standard rather
     than guessing:
       - smaller than the applicable margin → "TP", reason
         "unsafe_proximity".
       - clearly larger → "FP", reason "safe_distance".
       - about the same, or motion state itself is unclear and it
         affects the call → "uncertain", reason "distance_ambiguous".
"""

# Only the two violation types with a written rubric so far — others fall
# back to no rule panel in the UI. Add entries here as more violation
# types get a reviewed rule text.
VIOLATION_RULE_TEXT: dict[ViolationType, str] = {
    ViolationType.DANGEROUS_PLANE: DANGEROUS_PLANE_RULE_TEXT,
    ViolationType.TOO_NEAR_FORKLIFT: TOO_NEAR_FORKLIFT_RULE_TEXT,
}

SYSTEM_PROMPT = f"""\
You are verifying detections from a computer vision safety-monitoring
system covering five violation types: missing glove, missing helmet,
working at height without fall protection ("dangerous plane"), standing
under a suspended crane/hoist load ("suspended load"), and unsafe
proximity to a forklift ("too near forklift"). For each detection you will
be shown a cropped image and the violation type(s) to check.

Judge only what's shown. Do not speculate about what's outside the crop,
and do not try to correct or relocate any detection. If a violation's
status is ambiguous, say so rather than guessing — use "uncertain" and low
confidence for that violation.

For every result, alongside the categorical "confidence" (high/medium/low),
also give "confidence_score": a number from 0.0 to 1.0 that is your own
probability estimate that your verdict is correct. The two must stay
consistent with each other — "high" confidence should correspond to a
confidence_score well above 0.85, "medium" to something in the middle, and
"low" to a number closer to 0.0.

THE ACTIVITY GATE (applies to helmet and glove)
Only treat missing PPE as a violation when the crop shows the person
actively engaged with the hazard the detection concerns — handling
material, operating equipment, working at height, or working near
machinery. If they are clearly at rest, in a break/admin area, or not
engaged with any hazard, note what you actually observe but do not call it
a violation on activity grounds alone — use "FP" with the relevant reason,
or "uncertain" if you can't tell whether they're engaged in work from the
crop.

HELMET — WHAT'S ON THEIR HEAD
Classify the head covering as one of: hard helmet with domed shell / soft
cap or beanie / hood / bare head / not visible / too small or blurred to
tell. A hard helmet is compliant ("FP", reason "helmet_worn"). A soft cap,
beanie, or hood is not a helmet and is a confident violation ("TP", reason
"soft_headwear") — do not call this "ambiguous_headwear", which is for
genuine visual uncertainty about the headwear itself, not for headwear
you can clearly see is the wrong kind. A bare head is "TP", reason
"bare_head". If the head isn't visible or is too small/blurred to
classify, use "uncertain".

When you answer "FP" with reason "helmet_worn" — and only then — also
return "helmet_box": where the helmet you can see actually is, given as
[x1, y1, x2, y2] normalized to the image you were shown, with 0.0, 0.0 at
its top-left corner and 1.0, 1.0 at its bottom-right. Fit it tightly to
the helmet's own shell — not the whole head, and not the person. This box
is the evidence for clearing the alarm: a reviewer sees it drawn on the
image, so it must point at the helmet you actually judged. For every other
verdict and reason, return "helmet_box": null.

GLOVE — WHAT'S ON THEIR HANDS
For each visible hand, classify it as gloved, bare, or not visible. Look
for a cuff line and a colour or texture change at the wrist to tell
gloved from bare. If a hand is in a pocket, behind the body, or under
roughly 15 pixels across, call it "not visible" rather than guessing.
Hands being "not visible" should resolve to "uncertain" — expect this
often; do not force a guess.
Two specific exemptions, both meaning "FP" rather than a violation, if
you can tell them from the image itself:
  - the person is a forklift/lifting operator (e.g. an SLO/Lifting
    Supervisor) and the only activity shown is a forklift or lifting
    operation — reason "exempt_forklift_operator_role".
  - the person is standing inside a clearly marked holding/exclusion area
    (painted lines, signage, barriers marking it as such) — reason
    "exempt_holding_area".
Gloves is the hardest category by a wide margin — hands are small, often
occluded, and often holding something. From a wide shot where hands
aren't clearly resolved, prefer "uncertain" over an alarm; only commit to
a confident glove verdict where the crop actually has the resolution to
support it.

{DANGEROUS_PLANE_RULE_TEXT}
SUSPENDED LOAD — WORKING UNDER A SUSPENDED CRANE/HOIST LOAD
[DRAFT — needs sign-off from whoever owns judge-prompt safety content before
this is trusted in production scoring; not yet reviewed by a domain expert.]
This flags a person positioned where a crane- or hoist-suspended load
(cargo, a sling load, a man-cage, etc.) could strike them if it fell or
swung — not any lifting activity in general. Work through this in order:
  1. Is a load actually visibly suspended in the air (off the deck/ground,
     hanging from a crane hook, sling, or cable) somewhere in the frame?
     If no suspended load is visible at all: "FP", reason
     "no_suspended_load".
  2. Is the person positioned under the load itself, or under its
     hook/cable line — i.e. directly in its fall path? If they're clearly
     to the side, outside the load's footprint and swing arc: "FP", reason
     "outside_load_path".
  3. Is the person an authorized rigger/signaller actively guiding the
     load with a tag line from the side (not underneath), per standard
     rigging practice? If so: "FP", reason "exempt_tagline_rigger".
  4. Otherwise, if the person is under the load or within its swing path:
     "TP", reason "under_suspended_load".
  5. If the load's position, the person's position, or the fall/swing
     path can't be told from the crop: "uncertain", reason
     "position_ambiguous".

{TOO_NEAR_FORKLIFT_RULE_TEXT}
Respond only with the JSON schema provided. No extra text.\
"""

REASON_CODES: dict[ViolationType, list[str]] = {
    ViolationType.GLOVE: [
        "clear_bare_hand",
        "gloved",
        "no_hand_in_box",
        "hand_occluded",
        "ambiguous_material",
        "exempt_forklift_operator_role",
        "exempt_holding_area",
    ],
    ViolationType.HELMET: [
        "bare_head",
        "helmet_worn",
        "no_head_in_box",
        "head_occluded",
        "ambiguous_headwear",
        "soft_headwear",
    ],
    ViolationType.DANGEROUS_PLANE: [
        "unprotected_at_height",
        "adequate_fall_protection",
        "not_elevated",
        "no_person_in_box",
        "height_ambiguous",
        "wall_backed_edge_exempt",
    ],
    ViolationType.TOO_NEAR_FORKLIFT: [
        "unsafe_proximity",
        "safe_distance",
        "operator_in_cab",
        "forklift_not_present",
        "person_not_present",
        "distance_ambiguous",
        "not_a_real_person",
        "different_level",
        "barrier_between",
        "vehicle_unattended",
    ],
    # DRAFT — see the SUSPENDED LOAD section of SYSTEM_PROMPT: needs
    # safety-domain sign-off before production use.
    ViolationType.SUSPENDED_LOAD: [
        "under_suspended_load",
        "outside_load_path",
        "exempt_tagline_rigger",
        "no_suspended_load",
        "no_person_in_box",
        "position_ambiguous",
    ],
}

VERDICTS = ["TP", "FP", "uncertain"]
CONFIDENCE_LEVELS = ["high", "medium", "low"]

# Where the judge says the helmet is, when it clears a helmet alarm as
# "helmet_worn" — [x1, y1, x2, y2] normalized to the image the judge was
# shown (the crop, or the full image for an uncropped run), so the UI can
# draw it back on whichever of those it displays without knowing which the
# judge saw. Nullable rather than optional: a provider running the schema
# in strict mode needs every property listed in "required", and "no box"
# has to stay expressible for every verdict that isn't a cleared helmet.
HELMET_BOX_SCHEMA = {
    "type": ["array", "null"],
    "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    "minItems": 4,
    "maxItems": 4,
}


def _result_variant_schema(violation_type: ViolationType) -> dict:
    properties = {
        "violation_type": {"const": violation_type.value},
        "verdict": {"type": "string", "enum": VERDICTS},
        "reason": {"type": "string", "enum": REASON_CODES[violation_type]},
        "confidence": {"type": "string", "enum": CONFIDENCE_LEVELS},
        "confidence_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "explanation": {"type": "string"},
    }
    required = ["violation_type", "verdict", "reason", "confidence", "confidence_score", "explanation"]

    # Only HELMET carries it: no other violation type has a single object
    # whose location clears the alarm, and adding a nullable field to the
    # other variants would just be another thing for a weaker model to get
    # wrong on every call.
    if violation_type is ViolationType.HELMET:
        properties["helmet_box"] = HELMET_BOX_SCHEMA
        required.append("helmet_box")

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def build_output_schema(violation_types: list[ViolationType]) -> dict:
    """JSON schema for `output_config.format`, scoped to this person's
    applicable violation types."""
    return {
        "type": "object",
        "properties": {
            "person_id": {"type": "string"},
            "results": {
                "type": "array",
                "items": {"anyOf": [_result_variant_schema(v) for v in violation_types]},
            },
        },
        "required": ["person_id", "results"],
        "additionalProperties": False,
    }


def build_batch_output_schema(violation_types: list[ViolationType]) -> dict:
    """JSON schema for a call that judges SEVERAL crops of one frame.

    `_result_variant_schema` is reused byte-identically, so a batched result
    validates as the same `JudgeResult` a single call produces -- same
    verdicts, same reason codes, same `helmet_box` semantics. Only the
    envelope is new.

    `person_label` is the join key and it is required. `crop_summary` is the
    only in-band evidence that a label was matched to the right picture: a
    transposed answer usually describes a different scene, and the summary is
    what a human or a scorer can compare against the crop.
    """
    return {
        "type": "object",
        "properties": {
            "people": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "person_label": {"type": "string"},
                        "crop_summary": {"type": "string"},
                        "results": {
                            "type": "array",
                            "items": {
                                "anyOf": [_result_variant_schema(v) for v in violation_types]
                            },
                        },
                    },
                    "required": ["person_label", "crop_summary", "results"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["people"],
        "additionalProperties": False,
    }


def build_batch_user_text(
    labels: list[str],
    violation_types: list[ViolationType],
    extra_context: str | None = None,
) -> str:
    """The instruction for a batched call.

    Says the labels out loud, in order, and demands one entry per label. It
    also asks for brevity: at 8.5 crops a verdict's explanation is generated
    8.5 times in one response, and `[judge] max_tokens` is a single budget for
    all of them -- a truncated response is a parse failure, which reads as
    "the model is bad at batching".
    """
    names = ", ".join(v.value for v in violation_types)
    listed = ", ".join(labels)
    text = (
        f"This frame holds {len(labels)} person crops, each introduced by its own label: "
        f"{listed}. Judge EVERY one of them against these violation type(s): {names}. "
        "Return exactly one entry per label, using the labels exactly as given, in the "
        "same order. Keep each explanation to one sentence: the reply must hold every "
        "person's answer."
    )
    if extra_context:
        text = f"{text} {extra_context}"
    return text


def build_batch_output_schema_labels(labels: list[str]) -> str:
    """A one-line reminder of the labels, appended after the schema hint."""
    return "\nOne entry per label, no extras, no omissions: " + ", ".join(labels) + "."


def build_user_text(
    person_id: str,
    violation_types: list[ViolationType],
    extra_context: str | None = None,
) -> str:
    names = ", ".join(v.value for v in violation_types)
    text = (
        f"Detection for person_id {person_id}. Check the following violation "
        f"type(s) against the attached crop: {names}."
    )
    if extra_context:
        text = f"{text} {extra_context}"
    return text


def build_forklift_operator_context(containment_ratio: float) -> str:
    return (
        f"Geometric check note: this person's bounding box is {containment_ratio:.0%} "
        "contained within the forklift's bounding box, which usually indicates the "
        "vehicle's own operator — but could also be a person standing on or hanging "
        "off the vehicle in a hazardous position. Verify from the image which is the "
        "case before ruling on unsafe proximity."
    )


REASONING_ALIGNMENT_SYSTEM_PROMPT = """\
You are grading how well one reviewer's stated reasoning aligns with a
second, ground-truth reviewer's reasoning about the same PPE safety
detection — not whether they reached the same verdict. A judge can reach
the right verdict for the wrong reason, or the wrong verdict for reasoning
that's actually sound given what it describes; grade the reasoning itself.

Rate 1-5 how well the two reasonings agree about the underlying facts of
the scene (what's on the person's hands/head, their activity, their
distance from any vehicle, elevation and fall protection, etc.) and about
which of those facts drove the conclusion:
  5 — reasonings describe the same key facts and the same basis for the
      conclusion.
  3 — reasonings partly overlap, or reach a similar conclusion via
      different observed facts.
  1 — reasonings describe contradictory facts, or share no meaningful
      overlap.

Respond only with the JSON schema provided. No extra text.\
"""


def build_reasoning_alignment_user_text(
    violation_type: ViolationType, llm_explanation: str, ground_truth_rationale: str
) -> str:
    return (
        f"Violation type: {violation_type.value}\n\n"
        f"LLM judge's reasoning: {llm_explanation}\n\n"
        f"Ground-truth reviewer's reasoning: {ground_truth_rationale}"
    )


REASONING_ALIGNMENT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [1, 2, 3, 4, 5]},
        "critique": {"type": "string"},
    },
    "required": ["score", "critique"],
    "additionalProperties": False,
}
