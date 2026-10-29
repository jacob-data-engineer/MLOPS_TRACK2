"""Recency weighting: how much of each week still counts."""

from __future__ import annotations

import pytest

from ppe_pipeline_api import retention as rt
from _sides import patch_everywhere


def week(name: str, count: int) -> dict:
    return {"name": name, "stems": [f"{name}_img{i:04d}" for i in range(count)]}


def test_one_percent_a_week_compounds():
    """The newest week is whole; a year back is 59%, not 48%."""
    assert rt.retention_fraction(0) == 1.0
    assert rt.retention_fraction(1) == pytest.approx(0.99)
    assert rt.retention_fraction(52) == pytest.approx(0.99 ** 52, rel=1e-9)
    assert rt.retention_fraction(52) == pytest.approx(0.5930, abs=5e-4)


def test_exponential_never_reaches_zero_and_linear_does():
    """Why compounding is the default.

    At 1% a week, linear decay deletes week 1 outright at week 100 and
    would go negative after it. A curve that needs a floor to stay sane is
    the wrong curve.
    """
    assert rt.retention_fraction(100, {"curve": "exponential"}) > 0
    assert rt.retention_fraction(100, {"curve": "linear"}) == 0.0
    # Clamped, not negative.
    assert rt.retention_fraction(200, {"curve": "linear"}) == 0.0


def test_the_rate_is_settable():
    assert rt.retention_fraction(1, {"weekly_rate": 0.25}) == pytest.approx(0.75)
    assert rt.retention_fraction(2, {"weekly_rate": 0.25}) == pytest.approx(0.5625)
    # Zero decay is a valid setting and means "keep everything".
    assert rt.retention_fraction(52, {"weekly_rate": 0.0}) == 1.0


def test_selection_is_deterministic():
    """The same week at the same rate must give the same images.

    Otherwise every rebuild is a different dataset, the audit cannot be
    checked, and no past run can be reproduced.
    """
    stems = [f"img{i:04d}" for i in range(400)]
    first = rt.select_stems(stems, 0.5)
    second = rt.select_stems(list(reversed(stems)), 0.5)
    assert first == second
    assert len(first) == 200


def test_a_smaller_rate_keeps_a_superset():
    """Decay must be nested, not a reshuffle.

    Relaxing the rate should ADD images, not swap them: a run trained at
    10% and one at 20% otherwise share nothing, and comparing them
    measures the shuffle rather than the extra data.
    """
    stems = [f"img{i:04d}" for i in range(400)]
    tighter = set(rt.select_stems(stems, 0.25))
    looser = set(rt.select_stems(stems, 0.75))
    assert tighter < looser


def test_a_week_never_vanishes():
    """Floors, so an old week contributes history rather than nothing."""
    stems = [f"img{i:04d}" for i in range(1000)]
    # 1% of 1000 would be 10; the 10% floor holds it at 100.
    assert len(rt.select_stems(stems, 0.01, {"floor": 0.10})) == 100
    # And the absolute minimum wins when the percentage floor is smaller.
    assert len(rt.select_stems(stems, 0.0, {"floor": 0.0, "min_images": 25})) == 25
    # A week smaller than the minimum is kept whole, not padded.
    assert len(rt.select_stems([f"i{n}" for n in range(9)], 0.0, {"min_images": 25})) == 9


def test_nothing_is_exempt_from_the_decay():
    """The per-class exemption is gone, replaced by the golden set.

    It guessed which images mattered from a class count. A golden set is
    a folder somebody chose -- a judgement rather than a heuristic, and
    one visible as a folder rather than inferred at assembly time.
    """
    stems = [f"img{i:04d}" for i in range(100)]
    kept = rt.select_stems(stems, 0.10, {"floor": 0.0, "min_images": 0})
    assert len(kept) == 10
    # Deterministic: the same week at the same rate gives the same images,
    # which is what lets a past dataset be rebuilt exactly.
    assert kept == rt.select_stems(stems, 0.10, {"floor": 0.0, "min_images": 0})


def test_the_settings_no_longer_carry_a_class_exemption():
    assert "protect_rare_classes" not in rt.DEFAULTS
    assert "rare_class_threshold" not in rt.DEFAULTS
    assert not hasattr(rt, "mark_rare_class_images")


def test_age_is_counted_back_from_the_newest_week():
    """A gap in collection must not age the data around it oddly.

    Age is a position in the sequence of weeks, not a wall-clock distance
    from today, so a fortnight when nobody collected anything does not
    suddenly halve the week before it.
    """
    plan = rt.plan([week("week_1", 100), week("week_2", 100), week("week_3", 100)],
                   {"enabled": True, "weekly_rate": 0.10, "floor": 0.0, "min_images": 0})
    rows = {r["week"]: r for r in plan["weeks"]}
    assert rows["week_3"]["age_weeks"] == 0 and rows["week_3"]["keep"] == 100
    assert rows["week_2"]["age_weeks"] == 1 and rows["week_2"]["keep"] == 90
    assert rows["week_1"]["age_weeks"] == 2 and rows["week_1"]["keep"] == 81
    # Newest first, which is the order the card reads in.
    assert [r["week"] for r in plan["weeks"]] == ["week_3", "week_2", "week_1"]


def test_disabled_keeps_everything():
    plan = rt.plan([week("week_1", 50), week("week_2", 50)], {"enabled": False})
    assert plan["keep"] == 100 and plan["dropped"] == 0


def test_the_plan_totals_agree_with_its_rows():
    plan = rt.plan([week("week_1", 200), week("week_2", 200), week("week_3", 200)],
                   {"enabled": True, "weekly_rate": 0.05, "floor": 0.0, "min_images": 0})
    assert plan["keep"] == sum(r["keep"] for r in plan["weeks"])
    assert plan["available"] == 600
    assert plan["dropped"] == plan["available"] - plan["keep"]


def test_assembly_honours_the_allowlist(monkeypatch):
    """A thinned week is copied thin, not copied whole and trimmed after.

    The week folder is the record and must not be touched; only the
    assembled dataset is smaller.
    """
    from ppe_pipeline_api import track2_shared as t2

    copied = []
    patch_everywhere(monkeypatch, "_basenames", lambda prefix, exts: (
        {f"img{i}": f"{prefix}/img{i}.jpg" for i in range(6)}
        if "images" in prefix
        else {f"img{i}": f"{prefix}/img{i}.txt" for i in range(6)}
    ))
    monkeypatch.setattr(t2.s3, "copy_object",
                        lambda **kw: copied.append(kw["Key"]))
    monkeypatch.setattr(t2.s3, "put_object", lambda **kw: None)

    t2.assemble_dataset(
        splits={"train": [("w1/images", "w1/labels")], "val": []},
        classes={"0": "helmet"},
        destination="out",
        keep_stems={"w1/images": {"img0", "img3"}},
    )
    images = sorted(k for k in copied if "/images/" in k)
    assert images == ["out/images/train/img0.jpg", "out/images/train/img3.jpg"]


def test_the_curve_endpoint_spans_two_years_and_shows_the_floor(monkeypatch):
    """The chart must describe what happens, not just the formula.

    A 1%-a-week curve over one year barely moves; two years shows it
    going somewhere. And past the point where the floor takes over, the
    curve is decorative -- every older week keeps the same share -- so the
    floor has to be in the data the chart draws from.
    """
    from ppe_pipeline_api.routers import track2_retention as router

    points = router._curve({"weekly_rate": 0.05, "curve": "linear", "floor": 0.10})
    assert len(points) == router.CURVE_WEEKS + 1 == 105
    assert points[0]["fraction"] == 1.0
    # 5% linear reaches zero at week 20...
    assert points[20]["fraction"] == 0.0
    # ...but the floor holds the effective share at 10% from there on.
    assert points[20]["effective"] == 0.10
    assert points[104]["effective"] == 0.10
    # A week before the floor bites, the two agree.
    assert points[5]["effective"] == points[5]["fraction"] == pytest.approx(0.75)


def test_the_curve_endpoint_uses_unsaved_settings():
    """It is drawn from the controls, not from what was last saved.

    The bug this closes: the chart was captioned "ending at 59%" (1%
    compounding, the saved value) while the inputs read 5% linear.
    """
    from ppe_pipeline_api.routers import track2_retention as router

    slow = router._curve({"weekly_rate": 0.01, "curve": "exponential", "floor": 0.0})
    fast = router._curve({"weekly_rate": 0.05, "curve": "exponential", "floor": 0.0})
    assert slow[52]["fraction"] > fast[52]["fraction"]
    assert slow[52]["fraction"] == pytest.approx(0.5930, abs=5e-4)
    # Rounded to 4dp for transport, so compared at that precision.
    assert fast[52]["fraction"] == pytest.approx(0.95 ** 52, abs=1e-4)


# --------------------------------------------------------------------------
# The full-weight window, and the schedule as a grid
# --------------------------------------------------------------------------

def test_one_full_week_is_the_old_behaviour():
    """Adding the window must not change a policy that already exists.

    `full_weeks = 1` has to give `steps == age`, or every saved config
    silently starts decaying one week later than it was set up to.
    """
    for age in range(0, 20):
        assert rt.decay_steps(age, {"full_weeks": 1}) == age
        assert rt.retention_fraction(age, {"weekly_rate": 0.1, "curve": "linear"}) \
            == pytest.approx(rt.retention_fraction(age, {
                "weekly_rate": 0.1, "curve": "linear", "full_weeks": 1}))


def test_the_newest_weeks_are_untouched():
    """With a window of four, ages 0-3 take no decay and age 4 takes one."""
    config = {"full_weeks": 4}
    assert [rt.decay_steps(age, config) for age in range(8)] == [0, 0, 0, 0, 1, 2, 3, 4]


def test_a_window_of_zero_is_read_as_one():
    """Zero weeks at full weight is not a policy, it is a typo. It has to
    land somewhere sane rather than making the first step negative."""
    assert rt.decay_steps(0, {"full_weeks": 0}) == 0
    assert rt.decay_steps(3, {"full_weeks": 0}) == 3


def test_the_grid_matches_the_schedule_it_was_specified_from():
    """Ten percent a week, linear, four weeks at full weight.

    Deployment week down the side, collection week across:

        wk5   90 100 100 100 100
        wk6   80  90 100 100 100 100
        wk7   70  80  90 100 100 100 100
        wk8   60  70  80  90 100 100 100 100
    """
    config = {"enabled": True, "curve": "linear", "weekly_rate": 0.10,
              "full_weeks": 4}
    grid = rt.schedule_matrix([f"wk{n}" for n in range(1, 9)], config)

    rows = {row["week"]: [round(c["fraction"] * 100) for c in row["cells"]]
            for row in grid["rows"]}
    assert rows["wk1"] == [100]
    assert rows["wk4"] == [100, 100, 100, 100]
    assert rows["wk5"] == [90, 100, 100, 100, 100]
    assert rows["wk6"] == [80, 90, 100, 100, 100, 100]
    assert rows["wk7"] == [70, 80, 90, 100, 100, 100, 100]
    assert rows["wk8"] == [60, 70, 80, 90, 100, 100, 100, 100]


def test_a_row_holds_only_the_weeks_that_existed_by_then():
    """A retrain cannot train on data collected after it. The triangle is
    the point -- a full rectangle would be claiming it could."""
    grid = rt.schedule_matrix(["a", "b", "c"], {"enabled": True})
    assert [len(row["cells"]) for row in grid["rows"]] == [1, 2, 3]
    assert [c["week"] for c in grid["rows"][2]["cells"]] == ["a", "b", "c"]


def test_the_grid_runs_past_the_weeks_that_exist():
    """The question is what a retrain will use in two months, so the rows
    have to carry on past the data that has been collected."""
    grid = rt.schedule_matrix(["week_1", "week_2"], {"enabled": True}, ahead=3)
    assert grid["weeks"] == ["week_1", "week_2", "week_3", "week_4", "week_5"]
    assert [row["existing"] for row in grid["rows"]] == [True, True, False, False, False]


def test_invented_week_names_keep_the_prefix_and_the_padding():
    """They are shown beside the real ones, so they have to look like
    them -- a `week_08` series must not continue `week_9`."""
    assert rt.next_week_names(["week_1", "week_2"], 2) == ["week_3", "week_4"]
    assert rt.next_week_names(["week_08", "week_09"], 2) == ["week_10", "week_11"]
    assert rt.next_week_names(["wk7"], 1) == ["wk8"]
    assert rt.next_week_names([], 1) == ["week_1"]
    assert rt.next_week_names(["week_1"], 0) == []


def test_the_golden_column_appears_only_when_one_is_switched_on():
    with_gold = rt.schedule_matrix(["a", "b"], {
        "enabled": True, "use_golden": True, "golden_root": "golden/core"})
    assert with_gold["golden"] is True
    assert with_gold["golden_root"] == "golden/core"
    assert all(row["golden"] == 1.0 for row in with_gold["rows"])

    without = rt.schedule_matrix(["a", "b"], {"enabled": True})
    assert without["golden"] is False
    assert all(row["golden"] is None for row in without["rows"])


def test_the_golden_folder_is_remembered_while_the_switch_is_off():
    """Two keys rather than one: switching the golden set off must not
    mean finding the path again to switch it back on."""
    off = rt.schedule_matrix(["a"], {
        "enabled": True, "use_golden": False, "golden_root": "golden/core"})
    assert off["golden"] is False
    assert off["golden_root"] == "golden/core"


def test_the_floors_can_be_switched_off():
    """With no floor the curve alone decides, and an old enough week can
    reach zero. With one, nothing falls below it."""
    floored = {"floor": 0.10, "min_images": 0, "use_floor": True}
    bare = {"floor": 0.10, "min_images": 0, "use_floor": False}
    assert rt.keep_count(100, 0.02, floored) == 10
    assert rt.keep_count(100, 0.02, bare) == 2
    assert rt.keep_count(100, 0.0, bare) == 0
    # The minimum count goes with it.
    assert rt.keep_count(100, 0.0, {"min_images": 25, "use_floor": True}) == 25
    assert rt.keep_count(100, 0.0, {"min_images": 25, "use_floor": False}) == 0


def test_floors_on_is_the_default_and_unchanged():
    """Adding the switch must not quietly remove the floor from a policy
    that already relies on it."""
    assert rt.DEFAULTS["use_floor"] is True
    assert rt.keep_count(100, 0.0, {"floor": 0.10, "min_images": 0}) == 10


def test_the_golden_set_never_decays():
    """Whatever the rate, however far down the schedule. It is the core
    the detector is not allowed to forget."""
    grid = rt.schedule_matrix([f"w{n}" for n in range(20)], {
        "enabled": True, "weekly_rate": 0.5, "curve": "linear",
        "use_golden": True, "golden_root": "golden/core",
    })
    assert grid["rows"][-1]["golden"] == 1.0
    # ...while the oldest real week has long since bottomed out.
    assert grid["rows"][-1]["cells"][0]["fraction"] == 0.0


def test_a_disabled_policy_is_a_grid_of_ones():
    """Off means every week counts in full, and the grid has to say so
    rather than drawing the curve it would apply if it were on."""
    grid = rt.schedule_matrix(["a", "b", "c"], {
        "enabled": False, "weekly_rate": 0.5, "full_weeks": 1})
    assert all(c["fraction"] == 1.0 for row in grid["rows"] for c in row["cells"])
    assert grid["enabled"] is False


def test_the_curve_endpoint_draws_the_same_shape_as_the_grid():
    """The chart and the grid are two views of one policy. If the chart
    ignored the window it would decay from week 1 while the grid held the
    newest weeks flat, and one of them would be lying."""
    from ppe_pipeline_api.routers import track2_retention as router

    config = {"weekly_rate": 0.10, "curve": "linear", "floor": 0.0, "full_weeks": 4}
    points = router._curve(config)
    grid = rt.schedule_matrix([f"wk{n}" for n in range(1, 9)],
                              {**config, "enabled": True})
    for age in range(8):
        assert points[age]["fraction"] == pytest.approx(
            grid["rows"][age]["cells"][0]["fraction"])


def test_the_window_is_rejected_below_one():
    """Zero weeks at full weight would decay the week just collected,
    which is not a policy anybody means."""
    from fastapi import HTTPException

    from ppe_pipeline_api.routers import track2_retention as router

    with pytest.raises(HTTPException) as raised:
        router.save_settings(router.RetentionConfig(full_weeks=0), session=None)
    assert raised.value.status_code == 400


def test_the_golden_column_follows_the_switch_not_the_path():
    """Ticking the box has to put the column there.

    Gating it on the folder as well meant ticking "use a golden set" did
    nothing visible until a path had been typed, which reads as a broken
    checkbox rather than as an incomplete setting.
    """
    on = rt.schedule_matrix(["a", "b"], {"enabled": True, "use_golden": True})
    assert on["golden"] is True
    assert on["golden_root"] == ""
    assert all(row["golden"] == 1.0 for row in on["rows"])


def test_the_grid_reports_how_far_apart_the_two_curves_are():
    """Whole-percent cells hide the curve choice at low rates.

    At 1% a week compounding and linear are a fifth of a point apart over
    eight weeks, so both render as the same number and the control looks
    dead. The gap is reported so the card can show enough decimals.
    """
    names = [f"wk{n}" for n in range(1, 9)]
    slow = rt.schedule_matrix(names, {
        "enabled": True, "curve": "exponential", "weekly_rate": 0.01, "full_weeks": 1})
    fast = rt.schedule_matrix(names, {
        "enabled": True, "curve": "exponential", "weekly_rate": 0.10, "full_weeks": 1})

    assert slow["other_curve"] == "linear"
    assert slow["curve_gap"] < 0.01          # under a point: needs decimals
    assert fast["curve_gap"] > 0.10          # plainly visible in whole percents
    assert slow["curve_gap"] == pytest.approx(0.0021, abs=5e-4)

    # Off, there is no curve in play and nothing to compare.
    assert rt.schedule_matrix(names, {"enabled": False})["curve_gap"] == 0.0


def test_the_curve_choice_reaches_the_grid():
    """The maths differs, so the grid must differ. This is the assertion
    that would have caught the control genuinely not being wired."""
    names = [f"wk{n}" for n in range(1, 9)]
    config = {"enabled": True, "weekly_rate": 0.10, "full_weeks": 1}
    compounding = rt.schedule_matrix(names, {**config, "curve": "exponential"})
    linear = rt.schedule_matrix(names, {**config, "curve": "linear"})
    oldest = [round(c["fraction"] * 100) for c in compounding["rows"][-1]["cells"]]
    assert oldest == [48, 53, 59, 66, 73, 81, 90, 100]
    assert [round(c["fraction"] * 100) for c in linear["rows"][-1]["cells"]] \
        == [30, 40, 50, 60, 70, 80, 90, 100]


def test_the_grid_shows_the_floor_not_the_bare_curve():
    """A policy set to keep at least 20% of every week must not display
    2.81%.

    The floor was always applied when images were selected -- it is
    `keep_count` that does it -- but the grid reported
    `retention_fraction` alone, so the card contradicted the policy it
    was drawing. Reported, not spotted: the numbers looked like a broken
    floor rather than an unfloored display.
    """
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.20,
              "full_weeks": 1, "use_floor": True, "floor": 0.20, "min_images": 25}
    grid = rt.schedule_matrix([f"week_{n}" for n in range(1, 18)], config)
    oldest = grid["rows"][-1]["cells"][0]

    assert oldest["fraction"] == pytest.approx(0.0281, abs=5e-4)   # the curve
    assert oldest["effective"] == pytest.approx(0.20)              # what survives
    assert oldest["floored"] is True
    assert grid["floor"] == pytest.approx(0.20)
    assert grid["min_images"] == 25

    # The newest week is set by the curve, not the floor.
    newest = grid["rows"][-1]["cells"][-1]
    assert newest["effective"] == 1.0
    assert newest["floored"] is False


def test_the_grid_drops_the_floor_when_it_is_switched_off():
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.20,
              "full_weeks": 1, "use_floor": False, "floor": 0.20}
    grid = rt.schedule_matrix([f"week_{n}" for n in range(1, 18)], config)
    oldest = grid["rows"][-1]["cells"][0]
    assert oldest["effective"] == pytest.approx(oldest["fraction"])
    assert oldest["floored"] is False
    assert grid["floor"] == 0.0


def test_both_floors_are_applied_when_images_are_chosen():
    """The percentage and the image count, whichever is the larger.

    Reported as "the never fall below a certain percentage is not
    working" -- it was, and so was the image minimum. At 20% a week, a
    hundred-image week sixteen weeks old keeps 25 images, because the
    25-image minimum beats the 20% that the floor would have given.
    """
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.20,
              "full_weeks": 1, "use_floor": True, "floor": 0.20, "min_images": 25}
    curve_at_16 = rt.retention_fraction(16, config)
    assert curve_at_16 == pytest.approx(0.0281, abs=5e-4)

    assert rt.keep_count(100, curve_at_16, config) == 25    # the count wins
    assert rt.keep_count(500, curve_at_16, config) == 100   # 20% of 500 wins
    # A week smaller than the minimum keeps all of itself, never more.
    assert rt.keep_count(10, curve_at_16, config) == 10


def test_a_week_reports_the_share_it_actually_kept():
    """The bars printed the curve beside a count the floor had already
    lifted, so the two disagreed on the same row."""
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.20,
              "full_weeks": 1, "use_floor": True, "floor": 0.20, "min_images": 0}
    weeks = [week(f"week_{n}", 100) for n in range(1, 18)]
    out = rt.plan(weeks, config)
    oldest = out["weeks"][-1]                    # rows come back newest first
    assert oldest["week"] == "week_1"
    assert oldest["fraction"] == pytest.approx(0.0281, abs=5e-4)
    assert oldest["keep"] == 20
    assert oldest["effective"] == pytest.approx(0.20)


def test_real_sizes_bring_the_image_minimum_into_the_grid():
    """The half of the floor the grid could not show.

    The percentage floor depends only on the policy, so the grid always
    could apply it. The IMAGE minimum depends on how big a week is -- and
    when it is the binding half, a grid that only knew the percentage
    understated the policy. At 20% a week with a 25-image minimum, a
    hundred-image week keeps 25, which is 25%, not the 20% the
    percentage alone would have shown.
    """
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.20,
              "full_weeks": 1, "use_floor": True, "floor": 0.20, "min_images": 25}
    names = [f"week_{n}" for n in range(1, 18)]

    blind = rt.schedule_matrix(names, config)
    assert blind["sized"] is False
    assert blind["rows"][-1]["cells"][0]["effective"] == pytest.approx(0.20)

    sized = rt.schedule_matrix(names, config, sizes={n: 100 for n in names})
    assert sized["sized"] is True
    oldest = sized["rows"][-1]["cells"][0]
    assert oldest["keep"] == 25
    assert oldest["available"] == 100
    assert oldest["effective"] == pytest.approx(0.25)
    assert oldest["floored"] is True
    assert oldest["real"] is True


def test_the_grid_agrees_with_what_would_actually_be_built():
    """One arithmetic, not two. The cell calls `keep_count`, which is
    what the assembly calls, so the grid cannot promise a number the
    build would not produce."""
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.15,
              "full_weeks": 2, "use_floor": True, "floor": 0.05, "min_images": 10}
    names = [f"week_{n}" for n in range(1, 10)]
    sizes = {name: 40 + n * 7 for n, name in enumerate(names)}
    grid = rt.schedule_matrix(names, config, sizes=sizes)

    for cell in grid["rows"][-1]["cells"]:
        expected = rt.keep_count(sizes[cell["week"]],
                                 rt.retention_fraction(cell["age_weeks"], config),
                                 config)
        assert cell["keep"] == expected


def test_a_week_not_collected_yet_is_assumed_and_marked():
    """A guess must not wear the look of a measurement.

    The median, not the mean: week 1 is usually the whole backlog, and a
    mean would land somewhere no real week has ever been.
    """
    names = ["week_1", "week_2", "week_3"]
    sizes = {"week_1": 5000, "week_2": 80, "week_3": 100}
    grid = rt.schedule_matrix(names, {"enabled": True}, ahead=2, sizes=sizes)

    assert grid["assumed_size"] == 100           # median of 80, 100, 5000
    last = {c["week"]: c for c in grid["rows"][-1]["cells"]}
    assert last["week_1"]["real"] is True
    assert last["week_4"]["real"] is False
    assert last["week_4"]["available"] == 100


def test_counts_are_absent_rather_than_invented_when_nothing_was_read():
    grid = rt.schedule_matrix(["a", "b"], {"enabled": True})
    assert grid["sized"] is False
    assert grid["assumed_size"] == 0
    assert all(c["keep"] == 0 and c["real"] is False
               for row in grid["rows"] for c in row["cells"])


def test_a_week_that_exists_and_holds_nothing_is_a_counted_zero():
    """Not a guess. Tying `real` to a non-zero count marked the empty
    weeks as assumed, which said "we do not know" about the one thing
    that had been measured -- they were read, and they were empty."""
    names = ["week_1", "week_2", "week_3"]
    sizes = {"week_1": 26, "week_2": 0, "week_3": 119}
    grid = rt.schedule_matrix(names, {"enabled": True}, ahead=1, sizes=sizes)
    cells = {c["week"]: c for c in grid["rows"][-1]["cells"]}

    assert cells["week_2"]["real"] is True
    assert cells["week_2"]["empty"] is True
    assert cells["week_2"]["available"] == 0
    assert cells["week_1"]["empty"] is False
    # The projected week is still a guess.
    assert cells["week_4"]["real"] is False
    assert cells["week_4"]["empty"] is False


def test_an_empty_week_does_not_age_the_weeks_before_it():
    """Age counts weeks of COLLECTION, not weeks of calendar.

    Two empty weeks between week 3 and week 6 used to push week 3 three
    steps down the curve by the time week 6 was trained on, when only one
    further week of data had actually arrived -- the model was being told
    the world had moved on while nothing had been collected to say so.
    Same reason age is counted back from the newest week, not from today.
    """
    config = {"enabled": True, "curve": "linear", "weekly_rate": 0.10,
              "full_weeks": 1, "use_floor": False, "floor": 0.0, "min_images": 0}
    names = [f"week_{n}" for n in range(1, 8)]
    sizes = {"week_1": 26, "week_2": 392, "week_3": 203,
             "week_4": 0, "week_5": 0, "week_6": 8, "week_7": 119}
    grid = rt.schedule_matrix(names, config, sizes=sizes)
    cells = {c["week"]: c for c in grid["rows"][-1]["cells"]}

    # Seven calendar weeks, five of them holding anything.
    assert cells["week_7"]["age_weeks"] == 0
    assert cells["week_6"]["age_weeks"] == 1
    assert cells["week_3"]["age_weeks"] == 2          # not 4
    assert cells["week_2"]["age_weeks"] == 3          # not 5
    assert cells["week_1"]["age_weeks"] == 4          # not 6

    # The empty ones contribute nothing at all, rather than a share of
    # images that are not there.
    for name in ("week_4", "week_5"):
        assert cells[name]["empty"] is True
        assert cells[name]["keep"] == 0
        assert cells[name]["effective"] == 0.0
        assert cells[name]["floored"] is False


def test_the_plan_steps_over_empty_weeks_too():
    """The grid and the thing it describes must agree -- the dataset is
    chosen by `plan`, so the rule has to live there as well."""
    config = {"enabled": True, "curve": "linear", "weekly_rate": 0.10,
              "full_weeks": 1, "use_floor": False, "floor": 0.0, "min_images": 0}
    weeks = [week("week_1", 100), week("week_2", 100), {"name": "week_3", "stems": []},
             {"name": "week_4", "stems": []}, week("week_5", 100)]
    rows = {r["week"]: r for r in rt.plan(weeks, config)["weeks"]}

    assert rows["week_5"]["age_weeks"] == 0
    assert rows["week_2"]["age_weeks"] == 1           # not 3
    assert rows["week_1"]["age_weeks"] == 2           # not 4
    assert rows["week_1"]["keep"] == 80               # 1 - 0.10 * 2
    assert rows["week_3"]["available"] == 0


def test_the_floor_marker_means_a_floor_decided_not_that_rounding_did():
    """On a small week the two look identical from the outside.

    26 images at 99% is 25.74, which rounds to 26 -- the week reads as
    untouched without any floor being involved. Marking that as
    floor-held says the policy is doing something it is not.
    """
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.01,
              "full_weeks": 1, "use_floor": True, "floor": 0.10, "min_images": 25}
    names = ["week_1", "week_2", "week_3"]
    grid = rt.schedule_matrix(names, config, sizes={"week_1": 26, "week_2": 26,
                                                    "week_3": 26})
    cells = {c["week"]: c for c in grid["rows"][-1]["cells"]}

    # One week back: 25.74 rounds UP to 26, so nothing was held.
    assert rt.unfloored_count(26, rt.retention_fraction(1, config)) == 26
    assert cells["week_2"]["keep"] == 26
    assert cells["week_2"]["floored"] is False

    # Two back: 25.48 rounds to 25, which is also the minimum, so the
    # curve and the floor agree and neither changed the other's answer.
    assert cells["week_1"]["keep"] == 25
    assert cells["week_1"]["floored"] is False


def test_a_week_smaller_than_the_minimum_is_kept_whole_for_ever():
    """`keep_count` never returns fewer than the week holds, so a week of
    eight images under a 25-image minimum never decays at all. Reading
    that off a column of 100%s is guesswork, so it is flagged."""
    config = {"enabled": True, "curve": "exponential", "weekly_rate": 0.01,
              "full_weeks": 1, "use_floor": True, "floor": 0.10, "min_images": 25}
    names = [f"week_{n}" for n in range(1, 40)]
    grid = rt.schedule_matrix(names, config, sizes={n: 8 for n in names})
    oldest = grid["rows"][-1]["cells"][0]

    assert oldest["age_weeks"] == 38
    assert oldest["keep"] == 8
    assert oldest["effective"] == 1.0
    assert oldest["under_minimum"] is True

    # A week larger than the minimum is not flagged, and does decay.
    bigger = rt.schedule_matrix(names, config, sizes={n: 400 for n in names})
    assert bigger["rows"][-1]["cells"][0]["under_minimum"] is False
    assert bigger["rows"][-1]["cells"][0]["effective"] < 1.0
