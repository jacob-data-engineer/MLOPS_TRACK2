"""Emulating the JPSAI post-processing helmet gate over the raw UAT clip
corpus, so CV and the judge can be scored against the same human decision.

The reference is `pj.jpsai.postprocessing_logic`'s
`postprocess/PPE/Helmet/v1_0`. This package reproduces its decisions,
including the ones that are wrong: an emulation that quietly improves on
production cannot attribute its own disagreements.

Bottom up:

    clips       what an alarm-clip S3 key means, and which clips to take
    extract     clips -> named frames on local disk, one sidecar per clip

    thresholds  every constant, with the reference line it came from
    geometry    is_intersect / point_in_bbox / the two person heights / head box
    color       the HSV helmet-colour classifier, ported verbatim
    pose        COCO-17 -> the reference's AIC-14, and pose<->person matching
    snapshot_boxes  the vendor CV's own person boxes, recovered from the
                    red rectangle it drew on its annotated still

    cascade     THE GATE: four rungs, in order. Everything else is plumbing.
    records     one person in one frame, plus the provenance to reproduce it
    runner      the gate over a corpus, and the judge over a sample of it
    sample      which candidates the judge is asked about, blind to the gate
    cli         the whole thing from a terminal

WHAT IS DELIBERATELY ABSENT

No tracker. Production associates a pose with a person by shared track id and
smooths its answer across frames with `StatusManager_v2`; neither is
reproduced (CEO decision), so poses are matched to people geometrically and
every record stands alone as one frame. The reference's smoothing is buggy in
any case — its deque is pre-filled with 200 copies of `("No Helmet", -1)`, so
frames 0-4 report a violation whatever the detector saw — and emulating that
would mean explaining it away in every result.
"""
