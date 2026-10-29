"""Recency weighting: older weeks contribute fewer images.

A detector retrained every week on everything ever collected drifts
towards its own history -- the site changes, the cameras change, the
season changes, and a year-old week still counts as much as last week's.
Decaying the older weeks lets recent data dominate while keeping enough
history that the model does not forget scenarios it rarely sees. What it
must NOT forget goes in the golden set -- a folder somebody chose, which
never decays and is always trained on in full.

NOTHING IS DELETED. The week folders are the record; this picks a subset
of them at assembly time and the assembled dataset is the derived thing.
That is deliberate, and the reasons are worth keeping written down:

  * The training audit records which files each run used. Delete the
    files and that record becomes a list of names nobody can recover --
    the run can never be reproduced or explained.
  * It cannot be undone. Storage can: the whole corpus here is under
    200 MB, and a hundred weeks of it is a few dollars a year. There is
    no cost argument that justifies destroying training data.
  * If cost ever does bite, an S3 lifecycle rule moves old weeks to
    Infrequent Access or Glacier. Still not deletion.

So a week can always be re-assembled at a different rate, and last
month's dataset can be rebuilt exactly.
"""

from __future__ import annotations

import hashlib

DEFAULTS = {
    "enabled": False,
    # Fraction lost per week of age. 0.01 is the 1% asked for.
    "weekly_rate": 0.01,
    # How many of the NEWEST weeks count in full before any decay starts.
    # 1 means only the newest, which is the original behaviour and why it
    # is the default -- an existing config keeps the curve it had.
    #
    # A window wider than one says something real about how a site
    # changes: nothing meaningful shifts in a month, so a month of data
    # is all equally current, and only past that is age worth paying
    # attention to. Decaying from the second week instead thins data that
    # is still describing today.
    "full_weeks": 1,
    # A curated folder that never decays and is always in the training
    # set, laid out like a week folder (images/train, labels/train).
    # It is the core the detector is not allowed to forget, however far
    # the recent weeks drift from it.
    #
    # Two keys, not one: the folder is remembered while the switch is
    # off, so turning it back on does not mean finding the path again.
    "golden_root": "",
    # Which bucket that folder is in. Blank is this app's own. The
    # curated sets are not always kept beside the week folders, and a
    # path with no bucket beside it is only correct by luck.
    "golden_bucket": "",
    "use_golden": False,
    # Whether `floor` and `min_images` below apply at all. True is what
    # they have always done, so an existing policy is unchanged.
    "use_floor": True,
    # exponential | linear. See retention_fraction.
    "curve": "exponential",
    # No week falls below this share, however old...
    "floor": 0.10,
    # ...nor below this many images, whichever is the larger. A week
    # reduced to four images contributes noise, not history.
    "min_images": 25,
}


def decay_steps(age_weeks: int, config: dict | None = None) -> int:
    """How many weeks of decay a week of this age has actually taken.

    The newest `full_weeks` weeks take none. The first week past that
    window takes one, and each week older takes one more:

        full_weeks = 4     age  0  1  2  3  4  5  6  7
                           step 0  0  0  0  1  2  3  4

    `full_weeks = 1` gives `step == age`, which is what this did before
    the window existed -- so an old config is unchanged by adding it.
    """
    config = {**DEFAULTS, **(config or {})}
    full = max(1, int(config.get("full_weeks") or 1))
    return max(0, int(max(0, int(age_weeks)) - full + 1))


def retention_fraction(age_weeks: int, config: dict | None = None) -> float:
    """How much of a week that is `age_weeks` old still counts.

    Exponential by default: `(1 - rate) ** steps`. It never reaches zero
    and reads as a half-life -- at 1% a week, half of a week's images are
    still in play after 69 steps.

    Linear (`1 - rate * steps`) is offered because it is easier to
    explain, but it hits exactly zero -- at 1% a week, week 1 vanishes
    entirely 100 steps later and would go negative after it. The floor
    below catches that, but a curve that needs a floor to stay sane is
    the wrong curve.

    `steps`, not `age`: the newest `full_weeks` weeks are all current and
    all count in full. See `decay_steps`.
    """
    config = {**DEFAULTS, **(config or {})}
    rate = max(0.0, min(float(config.get("weekly_rate") or 0.0), 1.0))
    steps = decay_steps(age_weeks, config)
    if config.get("curve") == "linear":
        fraction = 1.0 - rate * steps
    else:
        fraction = (1.0 - rate) ** steps
    return max(0.0, min(fraction, 1.0))


def next_week_names(names: list, count: int) -> list:
    """`count` week names continuing the series, for the weeks not
    collected yet. Derived from the last name so the invented ones match
    whatever prefix the pull control is actually writing."""
    if count <= 0:
        return []
    last = names[-1] if names else "week_0"
    digits = ""
    while last and last[-1].isdigit():
        digits = last[-1] + digits
        last = last[:-1]
    start = int(digits) if digits else 0
    width = len(digits) if digits else 0
    return [f"{last}{str(start + n).zfill(width)}" for n in range(1, count + 1)]


def schedule_matrix(week_names: list, config: dict | None = None, *,
                    ahead: int = 0, sizes: dict | None = None,
                    golden_available: int = 0) -> dict:
    """Every retrain against every week of data, as one grid.

        deployment  | gold | wk1  wk2  wk3  wk4  wk5
        wk5         | 100% |  90% 100% 100% 100% 100%

    Rows are the week a retrain is DEPLOYED; columns are the week the
    data was COLLECTED; the cell is how much of that week the retrain
    trains on. A week is absent from a row when it did not exist yet.

    This is what the single-column preview cannot show. The preview
    answers "what would a retrain use today"; a retraining policy is a
    schedule, and the question that decides whether it is the right one
    is "what will a retrain use in two months, and is week 1 still in it
    by then". The grid answers that without waiting two months.

    Pure -- it reads the config and the week NAMES, never the bucket. No
    week has to exist for a future row to be drawn, which is the point of
    `ahead`.

    `sizes` is `{week: images it holds}`, which the caller has already
    read. With it, each cell can report the IMAGE COUNT that survives,
    through `keep_count` -- and that matters for more than display: the
    minimum-image floor depends on how big a week is, so without sizes
    the grid can only apply the PERCENTAGE floor and quietly understates
    a policy whose image minimum is the one doing the work. A week
    nobody has collected yet has no size, so it is assumed to be the
    median of the ones that exist, and marked.
    """
    config = {**DEFAULTS, **(config or {})}
    names = [str(n) for n in week_names]
    names = names + next_week_names(names, max(0, int(ahead)))
    # The switch alone. A golden set that is switched on but has no
    # folder chosen yet still belongs in the grid -- the column is what
    # says "this is coming and it will never decay", and hiding it until
    # a path is typed made ticking the box look like it did nothing.
    golden = bool(config.get("use_golden"))
    enabled = bool(config.get("enabled"))

    # The percentage floor, which the grid CAN apply: it depends only on
    # the policy. The image minimum cannot be shown here -- it depends on
    # how many images a week actually holds, and this function never
    # reads the bucket. Named in `min_images` below so the card can say
    # so rather than quietly understating a small week.
    floor = (max(0.0, min(float(config.get("floor") or 0.0), 1.0))
             if config.get("use_floor", True) else 0.0)

    # Sizes, and an assumption for the weeks not collected yet. The median
    # rather than the mean: one enormous backlog week (week 1 usually is
    # one) would drag a mean somewhere no real week has ever been.
    sizes = {str(k): int(v) for k, v in (sizes or {}).items()}
    known = sorted(v for v in sizes.values() if v > 0)
    assumed = known[len(known) // 2] if known else 0
    sized = bool(known)

    # An empty week is not a week of data, so it does not age anything.
    #
    # Age is counted in weeks of COLLECTION, not weeks of calendar. Two
    # empty weeks between week 3 and week 6 used to push week 3 three
    # steps down the curve by the time week 6 was trained on, when only
    # one further week of data had actually arrived -- the model was
    # being told the world had moved on while nothing had been collected
    # to say so. It is the same reason age is counted back from the
    # newest week rather than from today.
    #
    # A week whose size is NOT known counts as holding data. Only a week
    # that was read and found empty is stepped over -- otherwise a grid
    # drawn before any folder has been listed would treat every week as
    # empty and flatten the whole curve to 100%.
    holds = [(sizes[name] > 0) if name in sizes else True for name in names]
    counted = [0]
    for has_data in holds:
        counted.append(counted[-1] + (1 if has_data else 0))

    def weeks_of_data(collected: int, deployed: int) -> int:
        """Weeks holding data collected after `collected`, up to
        `deployed`. Zero when they are the same week."""
        return counted[deployed + 1] - counted[collected + 1]

    def effective_at(age: int, settings: dict) -> float:
        if not enabled:
            return 1.0
        return max(retention_fraction(age, settings), floor)

    rows = []
    for deployed, name in enumerate(names):
        cells = []
        for collected in range(deployed + 1):
            age = weeks_of_data(collected, deployed)
            fraction = retention_fraction(age, config) if enabled else 1.0
            week_name = names[collected]
            real = week_name in sizes
            available = sizes.get(week_name, assumed)

            if available > 0:
                # The real arithmetic, both floors and the rounding --
                # the same call the assembly makes, so the grid cannot
                # disagree with what is actually built.
                keep = keep_count(available, fraction, config) if enabled else available
                effective = keep / available
                # A floor DECIDED this only if it changed the answer. The
                # test used to be `effective > fraction`, which also fires
                # on rounding up -- so 26 images at 99% (25.74, rounds to
                # 26) was marked as floor-held when nothing had held it.
                held = bool(enabled and keep > unfloored_count(available, fraction))
            elif real:
                # Read, and found empty. Nothing to take a share of --
                # reporting the curve here would put a number on a folder
                # that holds no images.
                keep = 0
                effective = 0.0
                held = False
            else:
                # Size unknown, so there is no count to give; the SHARE is
                # still well defined and is what the policy says.
                keep = 0
                effective = max(fraction, floor) if enabled else 1.0
                held = bool(enabled and effective > fraction + 1e-9)

            cells.append({
                "week": week_name,
                "age_weeks": age,
                "steps": decay_steps(age, config) if enabled else 0,
                # The curve on its own...
                "fraction": round(fraction, 4),
                # ...and what actually survives. The grid used to print
                # the first and call it the answer, so a policy set to
                # keep at least 20% of every week displayed 2.81% -- the
                # floors were working and the grid was not describing
                # them.
                "effective": round(effective, 4),
                "floored": held,
                # Smaller than the image minimum, so it is kept whole
                # however old it gets: `keep_count` never returns fewer
                # than the week holds. Worth naming -- a week of eight
                # images under a 25-image minimum never decays at all,
                # and reading that off a column of 100%s is guesswork.
                "under_minimum": bool(
                    enabled and config.get("use_floor", True)
                    and 0 < available <= int(config.get("min_images") or 0)),
                "available": available,
                "keep": keep,
                # False only where the size is the ASSUMED one, so the
                # card never presents a guess as a count.
                #
                # A week that exists and holds nothing is a counted zero,
                # not a guess. Tying this to `available > 0` marked the
                # empty weeks as assumed, which said "we do not know"
                # about the one thing that had actually been measured.
                "real": bool(real),
                "empty": bool(real and available == 0),
            })
        rows.append({
            "week": name,
            "deployed_index": deployed,
            # The golden set never decays, so it is 1.0 in every row it
            # appears in -- which is every row, since it is not collected
            # on any particular week.
            "golden": 1.0 if golden else None,
            "existing": deployed < len(week_names),
            "cells": cells,
        })
    # The two curves are close at low rates: at 1% a week, compounding
    # and linear differ by 0.2 of a point after eight weeks, which is
    # nothing once a cell is rounded to a whole percent -- so switching
    # between them changed no number on screen and read as the control
    # doing nothing. Reported so the card can show enough decimals to
    # make the difference visible, and say how small it is.
    other = "linear" if str(config.get("curve")) != "linear" else "exponential"
    alternative = {**config, "curve": other}
    oldest = max(0, len(names) - 1)
    # Measured on the EFFECTIVE shares, because those are the numbers on
    # screen. Once the floor is holding both curves at the same value
    # there is genuinely no difference left to see, and a gap quoted from
    # the bare curves would promise one.
    gap = max(
        (abs(effective_at(age, config) - effective_at(age, alternative))
         for age in range(oldest + 1)),
        default=0.0,
    ) if enabled else 0.0

    return {
        "weeks": names,
        "existing": len(week_names),
        "ahead": max(0, int(ahead)),
        "curve": str(config.get("curve") or "exponential"),
        "other_curve": other,
        "curve_gap": round(gap, 6),
        "floor": round(floor, 4),
        "use_floor": bool(config.get("use_floor", True)),
        "min_images": int(config.get("min_images") or 0),
        # Whether any real counts were in hand. Without them the grid can
        # only show shares, and only the percentage floor is in play.
        "sized": sized,
        "assumed_size": assumed,
        "golden_available": int(golden_available),
        "golden": golden,
        "golden_root": str(config.get("golden_root") or "").strip("/"),
        "golden_bucket": str(config.get("golden_bucket") or "").strip(),
        "enabled": enabled,
        "full_weeks": max(1, int(config.get("full_weeks") or 1)),
        "rows": rows,
    }


def _rank(stem: str) -> str:
    """A stable position for a filename, independent of listing order.

    Sorting by this and taking the first N makes the subset deterministic:
    the same week at the same rate always yields the same images, so an
    assembled dataset can be rebuilt exactly and an audit can be checked.
    `random.sample` would make every rebuild a different dataset and every
    past run unverifiable.
    """
    return hashlib.sha256(stem.encode("utf-8")).hexdigest()


def unfloored_count(available: int, fraction: float) -> int:
    """What the curve alone would keep, before any floor.

    Only used to tell a cell whether a FLOOR decided its value or whether
    rounding did. On a small week those look identical from the outside:
    26 images at 99% is 25.74, which rounds to 26, so the week reads as
    untouched without any floor being involved.
    """
    if available <= 0:
        return 0
    return min(int(round(available * max(0.0, min(fraction, 1.0)))), available)


def keep_count(available: int, fraction: float, config: dict | None = None) -> int:
    """How many images to keep from a week, floors applied."""
    config = {**DEFAULTS, **(config or {})}
    if available <= 0:
        return 0
    if not config.get("use_floor", True):
        # No safety net: the curve alone decides, and a week old enough
        # can reach zero. Offered because a floor is a judgement -- "no
        # week ever falls below a tenth" is a claim about what history is
        # worth, and it is not one this has any business making silently.
        return min(int(round(available * max(0.0, min(fraction, 1.0)))), available)
    floor_fraction = max(0.0, min(float(config.get("floor") or 0.0), 1.0))
    wanted = int(round(available * max(fraction, floor_fraction)))
    # Never more than exists, and never below the minimum unless the week
    # is smaller than the minimum to begin with.
    return max(min(available, int(config.get("min_images") or 0)), min(wanted, available))


def select_stems(stems, fraction: float, config: dict | None = None) -> list:
    """The stems to keep from one week, oldest-agnostic and repeatable.

    Ranked by `_rank`, so the same week at the same fraction always
    yields the same images and a past dataset can be rebuilt exactly.

    Nothing is exempt. There used to be a per-class exemption here --
    images holding a scarce class kept regardless of age -- and it is
    gone: the golden set does that job, and does it better. The exemption
    guessed which images mattered from a class count; a golden set is a
    folder somebody chose, which is a judgement rather than a heuristic,
    and it is visible as a folder rather than inferred at assembly time.
    """
    stems = sorted(set(stems))
    if not stems:
        return []
    target = keep_count(len(stems), fraction, config)
    if target >= len(stems):
        return stems
    return sorted(sorted(stems, key=_rank)[:target])


def plan(weeks: list, config: dict | None = None) -> dict:
    """What each week contributes, newest first.

    `weeks` is `[{"name": str, "stems": [...]}]` in
    chronological order. Age is counted in weeks BACK FROM THE NEWEST, not
    from today: a gap where nothing was collected should not age the data
    either side of it differently from the order it was collected in.

    An EMPTY week is such a gap. It does not age anything either, for the
    same reason -- age counts weeks of collection, not weeks of calendar,
    so two empty weeks do not push everything before them two steps down
    the curve on the strength of nothing having been collected.
    """
    config = {**DEFAULTS, **(config or {})}
    rows = []
    holding = [index for index, week in enumerate(weeks) if week.get("stems")]
    ranked = {index: position for position, index in enumerate(holding)}
    newest_rank = len(holding) - 1
    for index, week in enumerate(weeks):
        stems = list(week.get("stems") or [])
        age = newest_rank - ranked.get(index, newest_rank)
        fraction = retention_fraction(age, config) if config.get("enabled") else 1.0
        kept = select_stems(stems, fraction, config)
        rows.append({
            "week": week.get("name", ""),
            "age_weeks": age,
            # The curve...
            "fraction": round(fraction, 4),
            # ...and the share that survived it, floors and rounding
            # included. The bars used to print `fraction` beside a count
            # the floor had already lifted, so the two disagreed on the
            # same row.
            "effective": round(len(kept) / len(stems), 4) if stems else 0.0,
            "floored": bool(
                config.get("enabled")
                and len(kept) > unfloored_count(len(stems), fraction)),
            "available": len(stems),
            "keep": len(kept),
            "dropped": len(stems) - len(kept),
            "stems": kept,
        })
    rows.reverse()
    return {
        "weeks": rows,
        "available": sum(r["available"] for r in rows),
        "keep": sum(r["keep"] for r in rows),
        "dropped": sum(r["dropped"] for r in rows),
        "enabled": bool(config.get("enabled")),
    }


# --------------------------------------------------------------------------
# Reading real week folders, and building the decayed dataset
#
# Imported lazily inside the functions: the maths above is pure and worth
# keeping importable without boto3 or the rest of the app.
# --------------------------------------------------------------------------

def only_weeks(names, week_prefix: str = "week_") -> list:
    """The folder names that are WEEKS: those starting with the prefix.

    A root holds more than weeks. `_retained/` is where the retention
    lists are written, and a root can contain any other folder somebody
    put there -- a nested dataset, a test copy. Listed as weeks, each one
    showed up on the card as a week with no images, and one that DID
    hold `images/train` would have been trained on as the oldest week.
    The pull files weeks under `week_prefix` and only those are weeks.
    """
    prefix = (week_prefix or "").strip()
    return [n for n in names if not prefix or str(n).startswith(prefix)]


def week_order(name: str) -> tuple:
    """`week_10` after `week_9`, not between `week_1` and `week_2` --
    which decides which week counts as newest, and so the whole curve."""
    digits = "".join(ch for ch in str(name) if ch.isdigit())
    return (int(digits) if digits else 0, str(name))


def week_names(root: str, week_prefix: str = "week_") -> list:
    """The week folders under a root, oldest first. The one definition of
    which folders are weeks and what order they age in."""
    from .. import storage

    root = (root or "").strip().strip("/")
    if not root:
        raise ValueError("Pick the folder that holds the week folders.")
    return sorted(only_weeks(storage.list_s3_folders(root), week_prefix), key=week_order)


def read_inventory(root: str, *, images_subpath: str = "images",
                   week_prefix: str = "week_") -> list:
    """Every week's TRAIN and VAL images, oldest first:
    `[{"name", "train": {id: key}, "val": {id: key}}]`.

    THE inventory. The card's bars, the schedule grid and the list the
    trainer reads (`retained_lists.select`) all count from this, because
    they used not to: the card and grid sized each week from the whole
    `images/` subtree, val included, while the trainer decayed
    `images/train` alone -- so the card said "100 of 200" over a week the
    run took 50 of 100 from, and a validation-only week aged every week
    before it on the card while the trainer stepped over it as empty.

    An `id` is the image's path under its split folder, extension off:
    `cam_a/f0` for `images/train/cam_a/f0.jpg`. Not the bare stem, which
    collapses `cam_a/f0.jpg` and `cam_b/f0.jpg` -- two images with two
    labels -- into one and hides the other. For a flat folder the id IS
    the stem, so the decay ranks (and keeps) exactly what it did before.
    The extension is dropped because the label is: `x.jpg` and `x.png`
    beside each other both read `x.txt`, so they are one labelled image
    as far as Ultralytics is concerned; the later key wins, as before.

    Images directly under `images/` (no split) are in neither side. The
    trainer's lists never read them, so counting them would be the same
    disagreement over again.
    """
    from .. import storage

    root = (root or "").strip().strip("/")
    sub = (images_subpath or "").strip("/")
    weeks = []
    for name in week_names(root, week_prefix):
        base = f"{root}/{name}/{sub}" if sub else f"{root}/{name}"
        sides: dict = {"train": {}, "val": {}}
        # One listing per week, split by its first segment.
        for key in sorted(storage._list_keys(base + "/")):
            if not key.lower().endswith(tuple(storage.IMAGE_EXTENSIONS)):
                continue
            split, _, inside = key[len(base) + 1:].partition("/")
            if split in sides and inside:
                sides[split][inside.rsplit(".", 1)[0]] = key
        weeks.append({"name": name, **sides})
    return weeks


def plan_inventory(inventory: list, config: dict | None = None) -> dict:
    """`plan` over the TRAIN side of an inventory, with each week's
    validation count beside it.

    Validation is reported, never decayed: an evaluation set that shrinks
    week by week makes every run's numbers incomparable with the last.
    """
    planned = plan([{"name": w["name"], "stems": sorted(w["train"])}
                    for w in inventory], config)
    vals = {w["name"]: len(w["val"]) for w in inventory}
    for row in planned["weeks"]:
        row["val"] = vals.get(row["week"], 0)
    planned["val_available"] = sum(vals.values())
    return planned


def read_weeks(root: str, *, images_subpath: str = "images",
               week_prefix: str = "week_") -> list:
    """Each week's TRAINING image ids, oldest first:
    `[{"name", "stems": [...]}]`.

    The train side of `read_inventory`, kept for callers that want the
    old shape. It used to list the whole `images/` subtree, val included,
    which is what made the card disagree with the trainer.
    """
    return [{"name": w["name"], "stems": sorted(w["train"])}
            for w in read_inventory(root, images_subpath=images_subpath,
                                    week_prefix=week_prefix)]


def preview(*, root: str, config: dict | None = None, images_subpath: str = "images",
            labels_subpath: str = "labels") -> dict:
    """The plan for a real root, ready for the card.

    Counted from `read_inventory`, the same as `retained_lists.select`,
    so every number on the card is one the trainer's list will hold.

    `labels_subpath` is accepted and unused. It was read by the
    rare-class pass, which opened every label file in the corpus; that
    pass is gone, and so is the cost of it. The parameter stays because
    callers pass it by name.
    """
    del labels_subpath
    inventory = read_inventory(
        root, images_subpath=images_subpath,
        week_prefix=(config or {}).get("week_prefix") or "week_")
    if not inventory:
        return {"weeks": [], "available": 0, "keep": 0, "dropped": 0,
                "val_available": 0,
                "enabled": bool((config or {}).get("enabled")), "root": root}
    out = plan_inventory(inventory, config)
    out["root"] = root
    # The stems are for the build, not the card -- a year of weeks is a
    # lot of filenames to send to a browser that only draws a bar chart.
    for row in out["weeks"]:
        row.pop("stems", None)
    out["golden"] = golden_summary(config, images_subpath=images_subpath)
    return out


def golden_summary(config: dict | None = None, *, images_subpath: str = "images") -> dict:
    """What the golden set contributes, for the bars beside the weeks.

    Always its whole self: it has no collection week, so it has no age,
    so there is nothing for the decay to act on. Drawn at 100% in every
    week's company rather than as a week of its own, because that is what
    it is -- a constant, not the oldest entry in a series.
    """
    from .. import pull_control as pc
    from .. import storage

    merged = {**DEFAULTS, **(config or {})}
    root = str(merged.get("golden_root") or "").strip("/")
    bucket = str(merged.get("golden_bucket") or "").strip()
    on = bool(merged.get("use_golden")) and bool(root)
    if not on:
        return {"enabled": False, "root": root, "bucket": bucket, "available": 0}

    def count_images(prefix: str) -> int:
        # pull_control's lister takes a bucket; this app's own does not,
        # and the golden set is often kept away from the week folders.
        try:
            return sum(
                1 for obj in pc._list_objects(prefix.strip("/") + "/", bucket)
                if obj["Key"].lower().endswith(tuple(storage.IMAGE_EXTENSIONS))
            )
        except Exception:                                      # noqa: BLE001
            return 0

    sub = images_subpath.strip("/")
    count = count_images(f"{root}/{sub}/train")
    if not count:
        # An unsplit golden folder, written before the split existed.
        count = count_images(f"{root}/{sub}")
    return {"enabled": True, "root": root, "bucket": bucket, "available": count}


def build(*, root: str, classes: dict, config: dict | None = None, destination: str = "",
          eval_root: str = "", images_subpath: str = "images",
          labels_subpath: str = "labels") -> dict:
    """Assembles a training set with the older weeks thinned.

    The evaluation side is taken WHOLE and never decayed. A shrinking
    evaluation set makes every run's metrics incomparable with the last,
    and an improvement you cannot measure is not one you can ship.
    """
    from .. import storage
    from . import assemble

    week_prefix = (config or {}).get("week_prefix") or "week_"
    inventory = read_inventory(root, images_subpath=images_subpath,
                               week_prefix=week_prefix)
    if not inventory:
        raise ValueError(f"No week folders under {root}.")
    chosen = plan_inventory(inventory, config)

    # `assemble_dataset` narrows a source by bare STEM, so the kept ids
    # are handed over as stems. For a flat split folder those are the
    # same thing; nested duplicates are this older copy path's limit.
    keep_by_week = {row["week"]: {s.rsplit("/", 1)[-1] for s in row["stems"]}
                    for row in chosen["weeks"]}
    train_sources, keep_stems = [], {}

    # The golden set first, and whole. It is not collected on any week, so
    # it has no age and nothing to decay -- it is the core the detector is
    # not allowed to forget while the recent weeks pull it around. Laid
    # out like a week folder, which is what the Golden Set card writes.
    #
    # Its TRAIN split only. The val split is the fixed evaluation set, and
    # training on that would score the model on images it had just been
    # taught -- the exact leakage a golden set exists to avoid.
    merged = {**DEFAULTS, **(config or {})}
    golden_root = str(merged.get("golden_root") or "").strip("/")
    if not merged.get("use_golden"):
        golden_root = ""
    golden_bucket = str(merged.get("golden_bucket") or "").strip()
    if golden_root and golden_bucket and golden_bucket != storage.BUCKET:
        # `assemble_dataset` copies within one bucket. Refusing is the
        # honest answer: reading the prefix out of THIS bucket would
        # quietly assemble a dataset with no golden set in it, and the
        # run would look fine.
        raise ValueError(
            f"The golden set is in {golden_bucket}, and a dataset is assembled "
            f"inside {storage.BUCKET}. Copy it across, or clear the bucket to use "
            f"the one beside the week folders."
        )
    if golden_root:
        images = f"{golden_root}/{images_subpath.strip('/')}/train"
        labels = f"{golden_root}/{labels_subpath.strip('/')}/train"
        if assemble._basenames(images, storage.IMAGE_EXTENSIONS):
            train_sources.append((images, labels))
        else:
            # An unsplit golden folder, written before the split existed.
            images = f"{golden_root}/{images_subpath.strip('/')}"
            labels = f"{golden_root}/{labels_subpath.strip('/')}"
            if assemble._basenames(images, storage.IMAGE_EXTENSIONS):
                train_sources.append((images, labels))

    # The TRAIN split of each week, which is what the plan was counted
    # from. This used to copy the whole `images/` subtree, so a week's
    # val images whose stems the plan kept landed in the train split.
    for week in inventory:
        images = f"{root}/{week['name']}/{images_subpath.strip('/')}/train"
        labels = f"{root}/{week['name']}/{labels_subpath.strip('/')}/train"
        if not keep_by_week.get(week["name"]):
            continue
        train_sources.append((images, labels))
        keep_stems[images] = keep_by_week[week["name"]]

    eval_sources = []
    if eval_root:
        # Names only: the evaluation side is taken whole, folder by folder.
        for name in week_names(eval_root, week_prefix):
            eval_sources.append((
                f"{eval_root.strip('/')}/{name}/{images_subpath.strip('/')}",
                f"{eval_root.strip('/')}/{name}/{labels_subpath.strip('/')}",
            ))

    built = assemble.assemble_dataset(
        splits={"train": train_sources, "val": eval_sources},
        classes=classes,
        destination=destination,
        keep_stems=keep_stems,
    )
    built["retention"] = {k: v for k, v in chosen.items() if k != "weeks"}
    built["retention"]["golden_root"] = golden_root
    built["weeks"] = [{k: v for k, v in row.items() if k != "stems"}
                      for row in chosen["weeks"]]
    return built
