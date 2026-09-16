"""The two detector numbers this pass still turns on.

This file used to hold every constant of the ported JPSAI helmet gate --
camera intrinsics, both contradictory minimum person heights, the HSV helmet
colour bands, the colour-to-state table, the gating colours. All of it went
with the cascade. What follows is only what a person-box pass needs.

WHY THE CASCADE WENT, RECORDED HERE BECAUSE THIS IS WHERE SOMEONE WILL LOOK

The emulation ran. Measured over 20 clips / 1394 frames / 2120 candidates
(2026-08-31, `scripts/helmet_cv_density.py`), it could not operate on this
footage: 69.9% of candidates reached `no_pose_for_person`, because the workers
are ~83px tall (median) and the pose checkpoint finds almost nobody at that
size even at `imgsz=1920`. No pose means no head keypoint, which means no head
box, which means rungs 2-4 of the cascade never ran. The gate fired on 0.38%
of candidates, so a 200-candidate pilot would have held about one routed case.

The reference implementation and its defects are still described in the git
history of `cascade.py`, `pose.py`, `color.py` and `geometry.py`, and in
`HANDOFF_helmet_cv.md`. Nothing here reproduces them any more: the detector
now provides crops, the VLM decides, and a human scores both.
"""

from __future__ import annotations

#: `PPEViolationDetection.py:38`. The reference filters PERSON detections at
#: 0.5 and applies NO threshold at all to the helmet class.
#:
#: **NO LONGER APPLIED. Kept as the reference's value, for provenance.** It was
#: right for the reference, where a person detection fed a gate decision
#: directly, and wrong here, where the detector only supplies crops and the VLM
#: decides. Measured at `imgsz=1920`: filtering at 0.5 discarded 34% of the
#: people found on alarm frames and 50% on clip frames, and caused more than
#: half of the alarm frames that found nobody at all (28% -> 11% at 0.25). A
#: 0.3-confidence person is a crop the judge can rule on for the price of one
#: call; a discarded person is invisible to every arm forever, and detector
#: recall is the ceiling on VLM recall.
REFERENCE_PERSON_OD_CONFIDENCE_THRESHOLD = 0.5

#: Our inference confidence, not production's. The reference is handed
#: detection results produced elsewhere and records no confidence anywhere,
#: so this is a choice and every record says so in its provenance
#: (`records.OD_CONFIDENCE_SOURCE`). It now gates the person class as well as
#: inference -- see `detect`'s module docstring -- so it is the only person
#: threshold in force. Helmets are still never filtered.
DEFAULT_OD_CONFIDENCE = 0.25
