"""The unattended retrain: pull, rebuild, freeze, train.

Everything on the multi-auto-training page can be done by hand, and one
of the six steps -- the pull -- could already be put on a clock. The rest
could not, so "retrain weekly" meant a person pressing four buttons in
order on a Monday, and the model was only as current as somebody's
diary.

This runs the same four steps, in the same order, using the same code
paths the buttons use:

    1. sweep the watched folders into the week folders   (pull_control)
    2. rebuild data.yaml under the current policy        (dataset_yaml +
                                                          retained_lists)
    3. freeze it, so the run can be repeated             (dataset_snapshots)
    4. start the training pipeline                       (sagemaker)

Same code, not a second implementation. A scheduled retrain that filed
files differently from a manual one, or trained on a different selection,
would be a second system wearing the first one's name -- and the
difference would surface as a model nobody could explain.

WHY IT IS ITS OWN CLOCK, separate from the pull's. They answer different
questions: the pull's decides when data arrives, this decides when a
model is trained on it. Folding them into one schedule was the first
design and it is wrong -- a pull that finds nothing should not spend GPU
hours, and a retrain should be able to run on a Sunday against a week
that filled on Tuesday.

IT REFUSES MORE THAN IT DOES. A retrain is expensive and hard to undo:
it takes a GPU for hours and puts a version in a registry. So every
precondition is checked before anything starts, and a miss stops the run
with a reason rather than training on whatever happened to be there.
"""

from __future__ import annotations

import datetime as _dt
import uuid
import os

from .. import pull_control as pc

#: Where a run of this records what it did. Same table as the pull's own
#: history, so "what has this system done unattended" is one query.
KIND = "retrain_run"

#: The pipeline a scheduled retrain starts. The custom/open one, never
#: the production replay lineage -- that lineage's consumers read index 0
#: as person, and a model trained on somebody's own classes must not be
#: able to reach it.
PIPELINE_NAME = os.environ.get(
    "OPEN_PIPELINE_NAME", "track2-yolo-open-training-pipeline")

DEFAULTS = {
    "enabled": False,
    #: Chain onto the pull's schedule instead of keeping a clock of its
    #: own. True is the only mode wired today, and it is the better one:
    #: a separate clock can fire half way through a sweep and train on a
    #: week that is still filling, and no amount of picking the times
    #: apart fixes that -- a pull is as long as it is.
    #:
    #: Only SCHEDULED pulls chain. A manual "Pull now" and the
    #: file-arrival trigger do not, because those fire unpredictably
    #: often and each run of this costs GPU hours.
    "follow_pull": True,
    #: Only read when `follow_pull` is off: days between retrains on an
    #: independent clock.
    "every_days": 7,
    #: "HH:MM" Singapore time, on an independent clock.
    "at": "02:00",
    #: "YYYY-MM-DD" (Singapore): the day the independent clock's grid
    #: counts from -- every slot is anchor + k * every_days at `at`.
    #: WRITTEN BY THE SERVER (`with_anchor`), never by the card: set when
    #: the schedule is switched on or its interval/time changes, to the
    #: first occurrence still ahead, so switching on never fires on the
    #: spot. Without it the grid slid forward with the clock and a weekly
    #: schedule became due every day. An operator-chosen `from_date` wins
    #: over it; a config with neither counts from `LEGACY_EPOCH`.
    "anchor": "",
    #: NOTE: `pull_first` and `skip_if_unchanged` used to sit here. Both
    #: were settable from the UI and read by nothing -- `run_chain` does
    #: not sweep, and nothing consulted the second at all. "Do not train
    #: when the pull moved nothing" is real, and lives where it can
    #: actually be decided: `queue_after_pull.even_if_nothing_moved`.
    #:
    #: The run's shape. Held here rather than asked at fire time: a
    #: schedule that needs an answer is not unattended.
    "detector_slug": "",
    "architecture": "yolo11s",
    "epochs": 50,
    "img_size": 640,
    "batch": 16,
    "seed": 42,
    #: Early stopping. 0 means "never stop early", which is the default
    #: because a fixed epoch count is what makes two runs comparable.
    "patience": 0,
    #: Carried onto every run started by the schedule, so an unattended
    #: model is not the one in the registry with an empty Notes column.
    "notes": "",
    #: The rest of the Ultralytics arguments, as the card resolved them.
    #: Empty means "send nothing", which is what the card means by the
    #: box being unticked -- and what every schedule did before this
    #: existed.
    #:
    #: This was MISSING, and silently: the manual form carried
    #: hyperparameters and the schedule did not, so ticking the box
    #: changed what a hand-started run trained with and not what the
    #: unattended one did. Two runs, one record, and the difference only
    #: visible by reading the training log.
    "hyperparams": {},
    #: WHICH version of the Class to continue from. Blank means "the
    #: newest Approved one", which is what every run did before this
    #: existed and is still the right default for a chain meant to walk
    #: a lineage forward.
    #:
    #: NOT SET HERE. `settings()` fills it from step 1's own choice --
    #: `default_version`, or "scratch" when step 1 says to train from
    #: scratch. It stays in DEFAULTS so the key always exists.
    #:
    #: Pinning matters when the newest is not the best: a run that
    #: overfit, or one trained on a week that turned out to be
    #: mislabelled, becomes the parent of everything after it, and the
    #: only way back is to start again from a known-good version. Takes
    #: a full ModelPackageArn, a bare version number ("4"), or the
    #: literal "scratch".
    #:
    #: "scratch" exists because a BRAND NEW Class has nothing in its
    #: registry to continue from, so the chain stopped at the base step
    #: and an unattended schedule could never produce that Class's first
    #: model -- "create a Class, schedule it, wait" was a dead end.
    #: Scratch starts from the architecture's stock pretrained weights,
    #: which is what a first run has always meant.
    "base_version": "",
}


def find_pinned_version(packages: list, pinned: str):
    """The registry package a pin names, or None.

    Accepts either a full ModelPackageArn or a bare version number, so
    the UI can store whichever it has: the dropdown knows the ARN, while
    somebody typing into the field knows "4". Matching on the arn's last
    segment rather than a `version` field because that is the only form
    SageMaker guarantees across both list and describe shapes.

    None means NOT FOUND, and the caller must treat that as an error
    rather than falling back to the newest -- see where it is called.
    """
    wanted = str(pinned or "").strip()
    if not wanted:
        return None
    for package in packages or []:
        arn = str(package.get("ModelPackageArn") or "")
        if not arn:
            continue
        if arn == wanted or arn.rsplit("/", 1)[-1] == wanted.lstrip("vV"):
            return package
    return None


def settings(config: dict | None = None) -> dict:
    """The schedule's settings, defaults filled in.

    The detector and size come from `default_detector` -- the one chosen
    in step 1 -- and overwrite anything held here. ONE source, not a copy
    that can drift: a schedule still naming last month's detector while
    step 1 says otherwise is a disagreement nobody would notice until a
    model landed in the wrong registry.
    """
    config = config if config is not None else pc.read_config()
    plan = {**DEFAULTS, **(config.get("retrain_schedule") or {})}
    chosen = config.get("default_detector") or {}
    if chosen.get("slug"):
        plan["detector_slug"] = str(chosen["slug"])
    if chosen.get("architecture"):
        plan["architecture"] = str(chosen["architecture"])
    # ...and WHICH VERSION it continues from, from the same card. Step 1
    # already asks that question and stores the answer in
    # `default_version`; this card briefly asked it a second time, which
    # is two places to say one thing with nothing forcing them to agree
    # -- the exact shape of the duplication the detector and the run
    # shape were pulled out of a few lines above.
    #
    # "Training from scratch" there is the same choice expressed as a
    # source rather than a version, so it maps onto the same field.
    if str(config.get("default_source") or "") == "scratch":
        plan["base_version"] = "scratch"
    else:
        plan["base_version"] = str(config.get("default_version") or "")
    # ...and the run's shape, from the same card, for the same reason.
    shape = config.get("default_training") or {}
    for key in ("epochs", "img_size", "batch", "seed", "patience"):
        if shape.get(key) not in (None, ""):
            try:
                plan[key] = int(shape[key])
            except (TypeError, ValueError):
                # Keep the default rather than fail the schedule -- an
                # unattended retrain should still run. But SAY SO: this
                # used to be a bare `pass`, so a corrupt `epochs` meant
                # the schedule quietly trained at 50 while the card
                # showed the bad value, and nothing anywhere disagreed.
                print(f"retrain schedule: ignoring {key}={shape[key]!r} "
                      f"-- not a whole number; using {plan[key]}.")
    if shape.get("notes"):
        plan["notes"] = str(shape["notes"])
    # Only when the card says they are in play. The stored overrides are
    # SPARSE -- what somebody changed -- so the tick is what decides
    # whether a blank field means "the default, stated" or "say nothing".
    if shape.get("hyperparams_enabled"):
        given = shape.get("hyperparams") or {}
        if isinstance(given, dict):
            plan["hyperparams"] = {str(k): v for k, v in given.items()
                                   if str(v).strip() != ""}
    return plan


def _now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


#: The after-pull chain's defaults, imported rather than restated: a
#: second copy would answer "is this on?" differently from the module
#: that owns the setting.
from .queue_after_pull import DEFAULTS as DEFAULTS_AFTER_PULL  # noqa: E402


def excluded_weeks(config: dict) -> list:
    """Week folders unticked on the Auto Training card. Every reader of the
    week folders has to honour these -- see `retained_lists.select`."""
    return sorted(str(w) for w in ((config.get("auto_training") or {}).get("excluded") or []))


def pull_train_entries(config: dict, root: str, *, write: bool) -> dict:
    """Pull Control's contribution to the TRAINING side, as the run names it.

    One function for the chain and the Assemble card's preview, so the
    preview shows the weeks exactly as the run will read them:

      retention ON  -- one entry, the path list `<root>/train.txt` holding
                       exactly the images the decay keeps. Ultralytics
                       reads a .txt entry as a list of image paths.
      retention OFF -- every week folder, whole.

    `write` is the only difference. The chain writes the list (it trains
    on it); the preview only SELECTS, and reports how many the list will
    hold -- the file in S3 is whatever the last build left there.
    """
    from .. import data

    retention = {**(config.get("retention") or {}),
                 "week_prefix": str(config.get("week_prefix") or "week_"),
                 "excluded_weeks": excluded_weeks(config)}
    skip = set(retention["excluded_weeks"])
    if retention.get("enabled"):
        if write:
            chosen = data.write_retained(root=root, config=retention)
            # The FULL key. In a composed yaml the entries are full key
            # paths, because the downloader keeps full paths once there is
            # more than one prefix. The list's own lines are `./<path>`
            # and resolve against the file's parent either way.
            entry = chosen["train_entry"].strip("/")
        else:
            chosen = data.select_retained(root=root, config=retention)
            entry = f"{data.RETAINED_FOLDER}/{data.RETAINED_TRAIN_NAME}" if data.RETAINED_FOLDER else data.RETAINED_TRAIN_NAME
        images = f"{root}/{entry}" if entry else root
        return {"entries": [{"images": images, "labels": ""}],
                "used": f"{len(chosen['train'])} retained images",
                "retained": {"entry": images, "images": len(chosen["train"]),
                             "available": chosen.get("train_available"),
                             # Which week folders the list draws from, and
                             # how many of each the decay keeps -- the list
                             # file alone hides them.
                             "weeks": [{"folder": f"{root}/{w['week']}",
                                        "keep": int(w.get("keep") or 0),
                                        "available": int(w.get("available") or 0)}
                                       for w in chosen.get("weeks") or []
                                       if w.get("available") and not w.get("excluded")]}}
    weeks = [w for w in data.scan_weeks(root, bucket=config.get("source_bucket") or "")
             if w["week"] not in skip]
    return {"entries": [{"images": w["images_path"], "labels": w["labels_path"]}
                        for w in weeks],
            "used": f"{len(weeks)} week folder" + ("s" if len(weeks) != 1 else ""),
            "retained": None}


def automatic(config: dict) -> dict:
    """The config an UNATTENDED run builds from (2026-10-26).

    Always composed from the ticked sources -- the pull's week folders,
    the golden set and the Assemble card's folders, each when its card is
    ticked -- whatever step 6's Training data radio says. The radio's
    frozen-snapshot options are for a run queued by hand; an automatic
    run is there to train on what has arrived.
    """
    return {**config, "data_source": "assembled"}


def data_source(config: dict) -> str:
    """Which of the three "Training data" options is ticked.

    One reader, because the chain and the preflight have to agree about
    it -- and because "assembled" is the default for a config that has
    never had the radio touched.
    """
    chosen = str(config.get("data_source") or "").strip()
    return chosen if chosen in ("assembled", "snapshot", "pull") else "assembled"


def frozen_plan(config: dict) -> dict | None:
    """The frozen dataset a scheduled run should train on, as-is.

    `None` unless the radio is on one of the two snapshot options.

    NOTHING IS REBUILT for these. "The same images every time, whatever
    has been pulled since" is the whole promise of a frozen snapshot, and
    rebuilding the lists under the current retention policy would break
    it quietly -- the run would still succeed, on a different corpus
    than the one that was pinned.
    """
    source = data_source(config)
    if source == "assembled":
        return None
    slot = "default_dataset" if source == "snapshot" else "last_pull_dataset"
    record = config.get(slot) or {}
    folder = str(record.get("folder") or "").strip("/")
    label = ("frozen snapshot" if source == "snapshot"
             else "snapshot generated during auto pull")
    problems: list = []
    if not folder:
        problems.append({
            "level": "blocking",
            "text": f"Training data is set to the {label}, but none has been "
                    f"chosen yet. Pick one in step 6, or switch to another "
                    f"source.",
        })
    return {
        "folder": folder,
        "snapshot_id": str(record.get("snapshot_id") or ""),
        "files": record.get("files"),
        "label": label,
        "problems": problems,
    }


def assemble_plan(config: dict) -> dict | None:
    """The Assemble card's saved build, when it is the one that runs.

    `None` when the tick is off -- which is every config written before
    this existed, so the root-and-week-folders path stays the default and
    an existing settings file cannot start building in place by surprise.

    The classes come from the chosen Model Name, not from a copy saved in
    the card. A detector whose class list was corrected would otherwise
    keep training against the stale names here, and the yaml is the only
    place those names exist -- nothing downstream could disagree.
    """
    # THE RADIO DECIDES, not a second tick on the card.
    #
    # "Training data" in step 6 already answers "what does a run train
    # on". A separate "Enabled for scheduling" on the Assemble card was
    # a second answer to the same question, and nothing said which won.
    if data_source(config) != "assembled":
        return None

    saved = config.get("dataset_assemble") or {}
    # Never filled in. Falls through to the root-and-week-folders build,
    # which is what every config written before the Assemble card
    # existed still wants -- "assembled" is the DEFAULT source, so
    # treating an empty card as a misconfiguration would break them all.
    if not (saved.get("train_sources") or saved.get("eval_sources")):
        return None

    problems: list = []
    splits = {
        "train": [s for s in (saved.get("train_sources") or []) if s.get("images")],
        "val": [s for s in (saved.get("eval_sources") or []) if s.get("images")],
    }
    if not splits["train"]:
        problems.append({
            "level": "blocking",
            "text": "Assemble is set to run on a schedule, but no training "
                    "folder is picked in it.",
        })
    if not splits["val"]:
        problems.append({
            "level": "blocking",
            "text": "Assemble is set to run on a schedule, but no evaluation "
                    "folder is picked in it. Evaluating on the training images "
                    "reports metrics that are not held out.",
        })

    slug = str(saved.get("detector_slug") or "").strip()
    classes: dict = {}
    if not slug:
        problems.append({
            "level": "blocking",
            "text": "Assemble is set to run on a schedule, but no Model Name "
                    "is chosen in it, so the yaml would declare no classes.",
        })
    else:
        try:
            from .. import training

            detector = training.get_detector(slug)
            classes = {int(k): str(v)
                       for k, v in (detector.get("classes") or {}).items()}
        except Exception as exc:                               # noqa: BLE001
            problems.append({
                "level": "blocking",
                "text": f"Assemble's Model Name '{slug}' could not be read: {exc}",
            })
        if slug and not classes:
            problems.append({
                "level": "blocking",
                "text": f"Model Name '{slug}' has no classes, so the yaml "
                        "would declare nothing.",
            })

    # WHAT THE LABEL FILES MEAN, when that is more than the Model Name
    # trains.
    #
    # The Model Name says what to TRAIN -- `helmet`, one class, one
    # registry group. The label files say what was LABELLED, and on this
    # bucket that was four classes. Writing the first into the yaml over
    # the second made Ultralytics drop every image holding a box it had
    # no name for: the whole image, silently, on the training side; a
    # refusal to start, on the validation side.
    #
    # So the yaml declares the labels' taxonomy when the card has been
    # told it, and `class_filter_for` does what it was built for: a
    # one-class Model Name present in a wider yaml becomes `classes=` plus
    # `single_cls=` at load time. Every image is kept; only the Model
    # Name's boxes train. The Model Name's class must appear in the list
    # BY NAME, or there is nothing to filter to.
    labelled_raw = saved.get("label_classes") or {}
    labelled: dict = {}
    if isinstance(labelled_raw, dict):
        for key, value in labelled_raw.items():
            try:
                if str(value).strip():
                    labelled[int(key)] = str(value).strip()
            except (TypeError, ValueError):
                continue
    # Filter ON means the yaml declares what the model TRAINS and the
    # training container drops every other box -- so Label classes is
    # not read. Two alternatives, not a layering: six names AND the
    # filter gave a six-name yaml with nothing left to filter, and the
    # tick looked like it did nothing.
    if saved.get("drop_undeclared_labels"):
        labelled = {}
    if labelled and classes:
        missing = [name for name in classes.values() if name not in labelled.values()]
        if missing:
            problems.append({
                "level": "blocking",
                "text": "Label classes on the Assemble card do not include "
                        f"{', '.join(missing)} -- the Model Name's own class. "
                        "Add it at the index the label files use for it, or "
                        "the run has nothing to filter to.",
            })
        else:
            classes = labelled

    return {
        "splits": splits,
        "classes": classes,
        # Tick on the Assemble card: boxes whose class the yaml does not
        # name are dropped by the training container, S3 untouched.
        "drop_undeclared": bool(saved.get("drop_undeclared_labels")),
        "bucket": str(saved.get("bucket") or ""),
        # Unattended, so it may replace the file it wrote last time --
        # that is the whole point of running weekly. `write_in_place`
        # still refuses a data.yaml this card did not write.
        "overwrite": True,
        "problems": problems,
    }


def preflight(config: dict | None = None) -> dict:
    """Everything that would stop a scheduled retrain, checked now.

    Run from the card as well as from the Lambda, so the answer to "will
    this fire on Sunday" is visible on Saturday rather than discovered on
    Monday by its absence. A schedule whose failures are silent is worse
    than no schedule: it reads as "the model is being retrained".
    """
    config = config if config is not None else pc.read_config()
    plan = settings(config)
    problems: list = []

    root = str(config.get("root") or "").strip("/")
    frozen = frozen_plan(config)
    assembled = assemble_plan(config)
    if frozen is not None:
        # A frozen source has its own folder and its own file lists.
        # `root` is the week-folders answer and this path does not use
        # it, so demanding one would block a schedule that is correct.
        problems.extend(frozen["problems"])
    elif assembled is not None:
        # The Assemble card names its folders outright, so this path has
        # no root and no week folders. "No root folder is set" would be a
        # blocking complaint about a mechanism it does not use -- and it
        # would block every scheduled run of a card that is working.
        problems.extend(assembled["problems"])
    elif not root:
        problems.append({
            "level": "blocking",
            "text": "No root folder is set on the pull control, so there are "
                    "no week folders to train from.",
        })
    if not plan["detector_slug"]:
        problems.append({
            "level": "blocking",
            "text": "No detector chosen. Pick one in step 1, \"Register a "
                    "model and Classes\" -- it decides which registry the "
                    "trained model is versioned into, and a run cannot pick "
                    "one for itself.",
        })
    if not (config.get("picked") or {}).get("train_images"):
        problems.append({
            "level": "warn",
            "text": "No source folders are picked in Pull Control, so nothing "
                    "new will reach the week folders and each run will train "
                    "on the same images as the last.",
        })

    # The combination that can never fire.
    #
    # "Use these folders only" means no sweep moves anything, so every
    # pull reports `moved: 0` -- and `even_if_nothing_moved` is off by
    # default, which skips exactly those. The schedule then runs on time,
    # the pull succeeds, the chain declines every single one, and the
    # card shows a working schedule that will never train. Nothing in
    # that chain is broken, which is what makes it hard to see.
    # The combination that could never fire is GONE: with "Use these
    # folders only" on, the pull is no longer the trigger at all, so
    # `even_if_nothing_moved` has nothing to decide. What replaced it is
    # the check below -- a clock that is on but has no time set.
    if not follows_pull(config) and settings(config).get("enabled"):
        if not _valid_time(str(settings(config).get("at") or "")):
            problems.append({
                "level": "blocking",
                "text": "The retrain is on its own clock -- the pull is not "
                        "the trigger while \"Use these folders only\" is "
                        "ticked -- but no time of day is set for it.",
            })

    # Images that were pulled, are being paid for, and will never be
    # trained on. The data.yaml is built from the week DEFINITIONS, so a
    # folder with no definition is invisible to every other check -- each
    # component is working correctly from a list the folder is not on.
    try:
        orphans = pc.orphan_week_folders(config)
    except Exception:                                          # noqa: BLE001
        orphans = []
    if orphans:
        where = ", ".join(f"{o['name']} ({o['images']})" for o in orphans[:5])
        problems.append({
            "level": "warn",
            "text": f"Images are filed under week folders no week covers: "
                    f"{where}. They are excluded from the data.yaml, so they "
                    f"are stored but never trained on. Add a week that covers "
                    f"those dates, or delete the folders.",
        })

    # The inverse: definitions that will route files somewhere nothing
    # else reads, and date ranges no week covers at all.
    try:
        week_problems = pc.week_definition_problems(config)
    except Exception:                                          # noqa: BLE001
        week_problems = []
    for problem in week_problems[:4]:
        problems.append({"level": "warn", "text": problem["detail"]})

    # Where the numbers will come from. Not a failure -- the run is
    # correct -- but a score measured on the split that also chose
    # `best.pt` is optimistic, and a reader deserves to know before they
    # compare it with anything.
    if not str((config.get("default_dataset") or {}).get("eval_folder") or "").strip():
        problems.append({
            "level": "warn",
            "text": "No separate evaluation set, so metrics are measured on the "
                    "training corpus's own val split -- the same split "
                    "Ultralytics picks best.pt on. The numbers are optimistic "
                    "by an unmeasured amount. Point the Golden Set at a "
                    "held-out folder to fix it.",
        })

    # `follows_pull`, not the raw setting: "Use these folders only"
    # overrides it, and warning "the pull is not on a schedule" while
    # this schedule's own clock is in charge sends somebody to configure
    # a mechanism that is deliberately switched off.
    if follows_pull(config):
        # The pull's own schedule is the clock, so this one's interval and
        # time are not read at all -- checking them would block on fields
        # nobody has set and nothing uses.
        if str(config.get("mode") or "") != "schedule":
            problems.append({
                "level": "warn",
                "text": "The pull is not on a schedule, so nothing will "
                        "trigger this. Set \"On a schedule\" on the pull "
                        "control, or press Run it now when you want a run.",
            })
    else:
        every = int(plan.get("every_days") or 0)
        if every < 1:
            problems.append({
                "level": "blocking",
                "text": "The interval is less than a day. A retrain takes "
                        "hours of GPU, so anything under a day would overlap "
                        "itself.",
            })
        at = str(plan.get("at") or "")
        if not _valid_time(at):
            problems.append({
                "level": "blocking",
                "text": f"{at!r} is not a time of day -- use HH:MM, e.g. 02:00.",
            })

    return {
        "settings": plan,
        "root": root,
        "problems": problems,
        "ready": not any(p["level"] == "blocking" for p in problems),
    }


#: Every stage the chain runs, in order, whether or not it reaches them.
#: Published so a status bar can draw the whole sequence from the start
#: -- a progress display that only lists steps already taken cannot show
#: what is still to come, which is most of what somebody watching wants.
CHAIN_STAGES = [
    ("preflight", "Checks"),
    # Before the lists: the golden set's val side becomes the evaluation
    # set, so it has to exist before anything names it.
    ("golden", "Golden set"),
    ("lists", "Retention"),
    ("data.yaml", "data.yaml"),
    ("freeze", "Freeze snapshot"),
    ("publish", "Set as training data"),
    ("base", "Base model"),
    ("queue", "Queue"),
    ("start", "Start"),
]

#: Where the live progress is kept. Its own row rather than a field on the
#: settings, because it is written several times a minute while a chain
#: runs and read by a bar polling every few seconds -- merging that into
#: the config on every step would make two writers of one document.
CHAIN_STATE_KEY = "chain_state"


def chain_progress() -> dict:
    """The chain's stages and how far the last/current run got.

    Always returns the full stage list, so the bar can draw the sequence
    greyed out before anything starts.
    """
    state = pc.read_config().get(CHAIN_STATE_KEY) or {}
    done = {str(s.get("step")): s for s in (state.get("steps") or [])}
    stages = []
    for key, label in CHAIN_STAGES:
        got = done.get(key)
        stages.append({
            "step": key,
            "label": label,
            "state": ("ok" if got and got.get("ok")
                      else "failed" if got
                      else "running" if state.get("running") and not done
                      else "pending"),
            "detail": str((got or {}).get("detail") or ""),
        })
    # The stage after the last recorded one is the one in flight.
    if state.get("running"):
        for stage in stages:
            if stage["state"] == "pending":
                stage["state"] = "running"
                break
    return {
        "running": bool(state.get("running")),
        "trigger": str(state.get("trigger") or ""),
        "started_at": str(state.get("started_at") or ""),
        "finished_at": str(state.get("finished_at") or ""),
        "ok": state.get("ok"),
        "stages": stages,
    }


def _publish_progress(**fields) -> None:
    """Never raises. A chain that fell over because its progress display
    was unreachable would be a worse failure than no display."""
    try:
        pc.write_config({CHAIN_STATE_KEY: {**(pc.read_config().get(CHAIN_STATE_KEY) or {}),
                                           **fields}})
    except Exception:                                       # noqa: BLE001
        pass


def run_chain(*, config: dict | None = None, trigger: str = "manual",
              note: str = "", submit: str = "start") -> dict:
    """Runs the chain and ALWAYS closes the progress row.

    A wrapper rather than a `_publish_progress` at each of the six exits:
    one of them is reached only on an exception, and a bar left reading
    "running" forever because a rarely-taken return forgot to clear it is
    the failure mode a progress display has to avoid above all others.
    """
    try:
        out = _run_chain_body(config=config, trigger=trigger, note=note,
                              submit=submit)
    except Exception as exc:                                # noqa: BLE001
        _publish_progress(running=False, ok=False,
                          finished_at=_now().isoformat(timespec="seconds"))
        _notify_chain_stopped(trigger, [], f"{type(exc).__name__}: {exc}")
        raise
    _publish_progress(running=False, ok=bool(out.get("ok")),
                      finished_at=_now().isoformat(timespec="seconds"))
    if not out.get("ok"):
        _notify_chain_stopped(trigger, out.get("steps") or [], "")
    return out


def _notify_chain_stopped(trigger: str, steps: list, crash: str) -> None:
    """Email "Errors" subscribers that a retrain chain stopped short.

    Only a chain that STOPPED: a run queued behind a busy GPU is ok and
    sends nothing. Never raises -- see `notify.publish`.
    """
    from .. import notify

    failed = [s for s in steps if not s.get("ok")]
    lines = [f"A retrain chain ({trigger}) stopped before training started.", ""]
    for s in failed:
        lines.append(f"Stopped at: {s.get('label') or s.get('step')}")
        lines.append(f"Why:        {s.get('detail') or '(no detail)'}")
    if crash:
        lines.append(f"Crashed:    {crash}")
    done = [s.get("label") or s.get("step") for s in steps if s.get("ok")]
    if done:
        lines += ["", "Completed first: " + ", ".join(str(d) for d in done)]
    lines += ["", "Nothing was submitted to SageMaker. Check the PULL, BUILD, QUEUE, "
                  "TRAIN strip on Multi-Auto Training."]
    where = (failed[0].get("label") or failed[0].get("step")) if failed else "an error"
    notify.publish(f"Retrain chain stopped at {where}", chr(10).join(lines), ["error"])


def _run_chain_body(*, config: dict | None = None, trigger: str = "manual",
              note: str = "", submit: str = "start") -> dict:
    """Rebuild the dataset, freeze it, and submit a run. In that order.

    The same three things the buttons do, through the same functions they
    call. Not a second implementation: a scheduled retrain that rebuilt
    or selected differently from a manual one would be a second system
    wearing the first one's name, and the difference would surface much
    later as a model nobody could account for.

    The ORDER is the point. Freezing before the rebuild would pin the
    previous dataset; submitting before the freeze would train something
    with no record of what it was. Each step hands its output to the
    next, and a failure stops the rest -- half a chain is worse than
    none, because the parts that did run look like they worked.

    `submit` decides the last step only:

      * "build" -- stop once the dataset is rebuilt, frozen and published;
        train nothing (Assemble's "Scheduled").
      * "start" -- begin the pipeline execution now.
      * "queue" -- append it to the training queue AND start the queue,
        which is what "do it automatically" means: a schedule that
        stopped at a queued entry produced a to-do item, not a model.
        The first two steps are identical to "start", which is the
        point: the dataset a queued entry trains on was built by the
        same code that builds one for a run started by hand.

        Starting is safe to do unattended because the queue serialises
        on the account's single GPU slot -- `try_advance_queue` refuses
        to pop while anything is Executing, so a schedule firing during
        a run waits rather than colliding with it.

    Returns what happened at each step, whether or not it got to the end.
    """
    from .. import data

    config = config if config is not None else pc.read_config()
    plan = settings(config)
    checks = preflight(config)
    steps: list = []

    _publish_progress(running=True, trigger=trigger, ok=None, finished_at="",
                      started_at=_now().isoformat(timespec="seconds"), steps=[])

    def record(name: str, ok: bool, detail: str, **extra):
        steps.append({"step": name, "ok": ok, "detail": detail, **extra})
        # Published AS IT HAPPENS, not at the end: the whole point of a
        # progress bar is the twenty minutes before there is a result.
        _publish_progress(steps=steps)
        return steps[-1]

    if not checks["ready"]:
        blocking = "; ".join(p["text"] for p in checks["problems"]
                             if p["level"] == "blocking")
        record("preflight", False, blocking)
        return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                "ok": False, "steps": steps}

    root = checks["root"]
    retention = {**(config.get("retention") or {})}
    # Which folders are weeks. Pull Control's setting, carried in so the
    # retention lists skip `_retained/` and anything else under the root.
    retention["week_prefix"] = str(config.get("week_prefix") or "week_")
    # ...and which weeks were unticked, which every list it writes skips.
    retention["excluded_weeks"] = excluded_weeks(config)

    # 1. the file lists, then the yaml that points at them. Lists first,
    #    so there is never a moment where the yaml names a file that does
    #    not exist yet.
    # 0. The Golden Set card, if it is live and filled in.
    #
    # One of the cards "Use these folders only" does NOT grey out, so a
    # scheduled run acts on it. Built FIRST because its val side becomes
    # the evaluation set, and nothing can name a folder that does not
    # exist yet.
    golden = golden_plan(config)
    golden_val = ""
    golden_train = ""
    if golden is None:
        record("golden", True, "not in use -- the evaluation set is "
                               "whatever the data source already names")
    else:
        try:
            from .. import data

            built_golden = data.build_golden(
                source=golden["source"],
                target_folder=golden["target_folder"],
                bucket=golden["bucket"],
                target_bucket=golden["target_bucket"],
                labels_source=golden["labels_source"],
                split=golden["split"],
                train_percent=golden["train_percent"],
            )
            # BOTH sides. The val side is the held-out evaluation set;
            # the train side is trained on in full and never decays --
            # that is what the card promises, and until now only the
            # val side was ever named in a yaml.
            #
            # With the split switched off there is one folder and no
            # held-out set, so it is all training data. Naming
            # `images/val` regardless -- which is what this did --
            # pointed the evaluation at a prefix with nothing under it.
            if golden["split"]:
                golden_train = f"{golden['target_folder']}/images/train"
                golden_val = f"{golden['target_folder']}/images/val"
            else:
                golden_train = f"{golden['target_folder']}/images"
                golden_val = ""
            record("golden", True,
                   f"{golden['target_folder']} -- "
                   f"{built_golden.get('copied', '?')} files copied",
                   target=golden["target_folder"])
        except Exception as exc:                               # noqa: BLE001
            # FATAL. The golden set is the held-out evaluation set; a run
            # that silently fell back to the training corpus's own val
            # split would report optimistic numbers under a label
            # promising held-out ones.
            record("golden", False, str(exc))
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps}

    # Blank unless the build below composes several prefixes. Every
    # other path is one folder, which is what every run so far has
    # been, and these stay empty for it.
    train_data_uris = ""
    data_yaml_uri = ""
    fetch_roots: list = []
    # Val folders outside the evaluation prefix, fetched by the evaluator.
    eval_extra_uris = ""
    # What a composed run really reads, for the queue card. The dataset
    # folder of a composed run is `_generated/<hash>` -- the yaml's home,
    # not where any image is -- so the card needs the yaml's own entries.
    composed_train: list = []
    composed_val: list = []
    # The Assemble card's filter tick, read once for every branch below:
    # it decides whether a label the yaml cannot name is a refusal or a
    # box the training container drops.
    drop_undeclared = bool((config.get("dataset_assemble") or {}).get("drop_undeclared_labels"))
    # A frozen source is trained AS IT WAS FROZEN: no rebuild, no
    # re-freeze. See `frozen_plan`.
    frozen = frozen_plan(config)
    assembled = assemble_plan(config)
    if frozen is not None:
        if not frozen["folder"]:
            record("data.yaml", False,
                   f"No {frozen['label']} has been chosen yet.")
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps}
        root = frozen["folder"]
        eval_root = ""
        record("lists", True,
               f"{frozen['label']} -- its file lists are already frozen, "
               "so nothing is rebuilt")
        record("data.yaml", True,
               f"{root} as frozen"
               + (f" ({frozen['snapshot_id']})" if frozen["snapshot_id"] else ""))
    elif assembled is not None:
        # The Assemble card's own build, through the same function its
        # button calls -- not a second implementation. A schedule that
        # assembled differently from the button would be a second system
        # wearing the first one's name.
        picks = source_selection(config)
        used: list = []

        # THE UNION OF WHATEVER IS TICKED.
        #
        # Three sources, three ticks, and the training side is whichever
        # of them are on. `train.py` takes a comma-separated list of
        # prefixes, so the folders no longer have to share a parent --
        # and `dataset_yaml` decides per build whether one prefix covers
        # everything or several are needed. Nothing here has to know
        # which.
        train_entries: list = []
        if picks["folders"]:
            train_entries += list(splits_of_assembled := assembled["splits"]["train"])
            used.append(f"{len(splits_of_assembled)} picked folder"
                        + ("s" if len(splits_of_assembled) != 1 else ""))
        splits = assembled["splits"]

        # The golden set, BOTH sides. Its train side is trained on in
        # full and never decays; its val side is the held-out set.
        if golden_train:
            train_entries.append({"images": golden_train, "labels": ""})
            used.append("the golden set")

        # The pull's week folders, through whatever retention selected.
        # Retention writes a .txt of image paths and Ultralytics reads a
        # .txt as a path list, so the decay curve arrives as one entry
        # rather than as a folder that would train every week whole.
        if picks["pull"] and root:
            try:
                pulled = pull_train_entries(config, root, write=True)
                train_entries += pulled["entries"]
                used.append(pulled["used"])
            except Exception as exc:                           # noqa: BLE001
                # FATAL. Training on two of the three sources when three
                # were asked for is a quietly smaller dataset, and the
                # run would succeed and report numbers for it.
                record("lists", False,
                       "Pull Control is ticked but its week folders could "
                       f"not be read: {exc}")
                return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                        "ok": False, "steps": steps}

        if not train_entries:
            record("lists", False,
                   "Nothing is ticked to train on. Tick at least one of "
                   "\"Use these folders\", \"Use Pick Golden Set / Edge "
                   "Cases\" or \"Use Pull Control\".")
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps}

        splits = {**splits, "train": train_entries}
        if golden_val:
            # BOTH val sides are scored: the dataset's own val folder and
            # the golden set's. Decided by the operator on 2026-10-20 --
            # replacing the dataset's val left 1,693 images used by
            # nothing. Golden is added, not substituted.
            kept = list(splits.get("val") or []) if picks["folders"] else []
            splits = {**splits, "val": kept + [{"images": golden_val, "labels": ""}]}
        elif not picks["folders"]:
            # The val side came from the Assemble card, which is no
            # longer in use. Leaving it would score the run on images
            # nothing was trained on.
            splits = {**splits, "val": []}
        try:
            out = data.write_yaml_in_place(
                splits=splits, classes=assembled["classes"],
                bucket=assembled["bucket"], overwrite=assembled["overwrite"],
                # Every label file, not a sample. Unattended, with GPU
                # time riding on the answer.
                label_scan=None, drop_undeclared=drop_undeclared)
        except Exception as exc:                               # noqa: BLE001
            record("data.yaml", False, str(exc))
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps}
        # Everything downstream -- the freeze, the published folder, the
        # prefix the run downloads -- works on ONE folder, and for an
        # in-place build that is where the yaml landed. `root` is the
        # week-folders answer and this path never had one; leaving it
        # would freeze and train something the yaml does not describe.
        root = out["folder"]
        # A COMPOSED dataset: the folders named in the yaml share no
        # parent, so the run fetches several prefixes and the yaml is
        # not at the top of any of them. Both have to be named
        # explicitly from here on -- `root` is only where the yaml
        # landed, and downloading it would fetch the yaml and nothing
        # else.
        if out.get("many_roots"):
            composed_train, composed_val = list(out["train"]), list(out["val"])
            train_data_uris = out["download_prefix"]
            data_yaml_uri = out["uri"]
            fetch_roots = [(u.split("/", 3)[3].strip("/") if u.split("/", 3)[2] == out["bucket"] else u.rstrip("/"))
                           for u in out["download_prefixes"]]
            # Stop HERE if the run could not be started with it.
            #
            # The yaml is written by this point, which is cheap and
            # harmless. What follows is not: a snapshot gets frozen
            # and an entry queued, and `try_advance_queue` DROPS an
            # entry it cannot start rather than putting it back. So
            # the whole chain would run, report "queued", and the
            # queue would empty itself with nothing trained.
            try:
                declared = t2_module().pipeline_parameter_names(
                    expect="DataYamlUri")
            except Exception:                              # noqa: BLE001
                # Unknown is not missing. Letting it through to the
                # submit check is better than refusing a run because
                # one describe call failed.
                declared = set()
            if declared and "DataYamlUri" not in declared:
                record("data.yaml", False,
                       "This dataset is composed from folders under "
                       "unrelated roots, so the run has to be told where "
                       "its data.yaml is -- and the training pipeline "
                       "does not declare DataYamlUri. Update it from "
                       "model_training/custom_training/pipeline.json.")
                return {"trigger": trigger,
                        "at": _now().isoformat(timespec="seconds"),
                        "ok": False, "steps": steps}
        # The SECOND download, when the held-out set has a prefix of its
        # own. Carried to the freeze below so the images the run is
        # SCORED on are recorded, not just the ones it trains on.
        eval_root = (out.get("eval") or {}).get("folder") or ""
        eval_extra_uris = ",".join((out.get("eval") or {}).get("extra_uris") or [])
        record("lists", True, "trained on " + ", ".join(used),
               sources=len(used))
        record("data.yaml", True, out["key"],
               train=", ".join(out["train"]), val=", ".join(out["val"]),
               classes=len(assembled["classes"]))
    else:
        eval_root = ""
        try:
            entries = ("", "")
            if retention.get("enabled"):
                written = data.write_retained(root=root, config=retention)
                # From write() itself, NOT rebuilt from FOLDER + NAME. FOLDER
                # became "" when the lists moved to the root this morning, so
                # f"{FOLDER}/{TRAIN_NAME}" yields "/train.txt" -- a leading
                # slash the trainer cannot resolve. write() already computes
                # the entry the yaml needs and is the only thing that knows
                # where it actually put the file.
                entries = (written["train_entry"], written["val_entry"])
                record("lists", True,
                       f"{len(written['train'])} train, {len(written['val'])} val",
                       train=len(written["train"]), val=len(written["val"]))
            else:
                record("lists", True, "retention is off -- the yaml names the "
                                      "week folders whole")

            # The same two arguments the UI passes. Without `class_sources`
            # the search never looks where the pull read from, and the
            # unattended build failed with "No data.yaml was found to read
            # class names from" on a root the UI builds happily.
            picked = config.get("picked") or {}
            class_sources: list = []
            for key in ("train_images", "train_labels", "val_images", "val_labels"):
                for folder in picked.get(key) or []:
                    if folder and folder not in class_sources:
                        class_sources.append(folder)

            # THE GOLDEN SET ON THE PULL PATH.
            #
            # The three ticks say golden may join either source, and the
            # assembled branch honours that. This branch never did: it
            # built its yaml with `data.build_yaml(root=...)`, which knows only
            # the week folders, so a golden set ticked alongside Pull
            # Control was built, recorded as built -- and used by nothing.
            # The held-out number a run reported came from the week
            # folders' own val side, under a label promising a golden one.
            #
            # So when golden is in play this branch composes, exactly as
            # the assembled one does: the pull's contribution (retention's
            # path list, or the week folders whole) plus the golden train
            # side, with the golden val side as the evaluation set. Class
            # names come from where this path always took them -- the
            # data.yaml beside the pulled sources -- unless the Assemble
            # card has been told what the label files mean.
            if golden_val or golden_train:
                pull_entries: list = []
                if entries[0]:
                    pull_entries.append({"images": f"{root}/{entries[0].strip('/')}",
                                         "labels": ""})
                else:
                    for week in [w for w in data.scan_weeks(root, bucket=config.get("source_bucket") or "")
                                 if w["week"] not in set(retention["excluded_weeks"])]:
                        pull_entries.append({"images": week["images_path"],
                                             "labels": week["labels_path"]})
                if golden_train:
                    pull_entries.append({"images": golden_train, "labels": ""})
                if golden_val:
                    # Golden ADDED to the pull's own val, as on the
                    # assembled path.
                    val_entries = ([{"images": f"{root}/{entries[1].strip('/')}", "labels": ""}]
                                   if entries[1] else []) + [{"images": golden_val, "labels": ""}]
                elif entries[1]:
                    val_entries = [{"images": f"{root}/{entries[1].strip('/')}", "labels": ""}]
                else:
                    val_entries = []

                labelled_raw = (config.get("dataset_assemble") or {}).get("label_classes") or {}
                if drop_undeclared:
                    # Same rule as the assembled path: the filter means the
                    # yaml names what the model trains, nothing more.
                    labelled_raw = {}
                names: dict = {}
                if isinstance(labelled_raw, dict):
                    for key, value in labelled_raw.items():
                        try:
                            if str(value).strip():
                                names[int(key)] = str(value).strip()
                        except (TypeError, ValueError):
                            continue
                if not names and drop_undeclared:
                    try:
                        names = {int(k): str(v) for k, v in
                                 (t2_module().get_detector(str(plan["detector_slug"]))
                                  .get("classes") or {}).items()}
                    except Exception:                          # noqa: BLE001
                        names = {}
                if not names:
                    names = (data.find_classes(class_sources,
                                             bucket=config.get("source_bucket") or "")
                             .get("names") or {})
                if not names:
                    record("data.yaml", False,
                           "No class names: nothing beside the pulled sources "
                           "declares any, and Label classes on the Assemble card "
                           "is empty.")
                    return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                            "ok": False, "steps": steps}

                out = data.write_yaml_in_place(
                    splits={"train": pull_entries, "val": val_entries},
                    classes=names, bucket=pc.destination_bucket(), overwrite=True,
                    # Every label file: unattended, with GPU time on the answer.
                    label_scan=None, drop_undeclared=drop_undeclared)
                root = out["folder"]
                eval_root = (out.get("eval") or {}).get("folder") or ""
                eval_extra_uris = ",".join((out.get("eval") or {}).get("extra_uris") or [])
                if out.get("many_roots"):
                    composed_train, composed_val = list(out["train"]), list(out["val"])
                    train_data_uris = out["download_prefix"]
                    data_yaml_uri = out["uri"]
                    fetch_roots = [(u.split("/", 3)[3].strip("/") if u.split("/", 3)[2] == out["bucket"] else u.rstrip("/"))
                                   for u in out["download_prefixes"]]
                    try:
                        declared = t2_module().pipeline_parameter_names(
                            expect="DataYamlUri")
                    except Exception:                          # noqa: BLE001
                        declared = set()
                    if declared and "DataYamlUri" not in declared:
                        record("data.yaml", False,
                               "This dataset is composed from folders under "
                               "unrelated roots, so the run has to be told where "
                               "its data.yaml is -- and the training pipeline "
                               "does not declare DataYamlUri. Update it from "
                               "model_training/custom_training/pipeline.json.")
                        return {"trigger": trigger,
                                "at": _now().isoformat(timespec="seconds"),
                                "ok": False, "steps": steps}
                record("data.yaml", True, out["key"],
                       train=", ".join(out["train"]), val=", ".join(out["val"]),
                       classes=len(names), golden=True)
            else:
                built = data.build_yaml(root=root, retained_train=entries[0],
                                 retained_val=entries[1],
                                 class_sources=class_sources,
                                 source_bucket=config.get("source_bucket") or "",
                                 excluded_weeks=retention["excluded_weeks"])
                if not built["ready"]:
                    blocking = "; ".join(p["text"] for p in built["problems"]
                                         if p["level"] == "blocking")
                    record("data.yaml", False, blocking)
                    return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                            "ok": False, "steps": steps}
                data.save_yaml(key=built["write_to"], text=built["yaml"])
                record("data.yaml", True, built["write_to"],
                       train_images=built["train_images"], val_images=built["val_images"])
        except Exception as exc:                                   # noqa: BLE001
            record("data.yaml", False, str(exc))
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps}

    # 2. freeze it, so whatever trains next can be trained again --
    #    unless it is ALREADY frozen, which is the whole point of the two
    #    snapshot sources. Re-freezing would mint a second id for the
    #    same images and lose the link to the one that was chosen.
    if frozen is not None:
        snapshot = {"snapshot_id": frozen["snapshot_id"],
                    "file_count": frozen.get("files")}
        record("freeze", True,
               frozen["snapshot_id"] or "already frozen -- reused as chosen",
               snapshot_id=frozen["snapshot_id"], files=frozen.get("files"))
    elif not config.get("freeze_on_run", True):
        # Ticked off on the Retention card. The run trains on exactly the
        # same images; it just leaves no snapshot to pick again later.
        snapshot = {"snapshot_id": "", "file_count": None}
        record("freeze", True, "skipped -- \"Freeze the dataset on every run\" is off")
    else:
        try:
            snapshot = data.create_snapshot(root=root, note=note or f"{trigger} retrain",
                                 eval_root=eval_root,
                                 # Empty for a single-folder dataset, so
                                 # this is the same call it always was.
                                 extra_roots=fetch_roots)
            record("freeze", True, snapshot["snapshot_id"],
                   snapshot_id=snapshot["snapshot_id"], files=snapshot["file_count"])
        except Exception as exc:                               # noqa: BLE001
            # Not fatal to the run itself -- the dataset is built and would
            # train fine. Fatal to REPEATING it, which is the reason the
            # freeze exists, so it stops here rather than training something
            # there is no record of.
            record("freeze", False, str(exc))
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps}

    # 2b. Publish the dataset this run built, server-side.
    #
    # Without this the automatic path is invisible: it rebuilds the
    # lists, writes the yaml and freezes a snapshot, then the training
    # card still reads "Nothing chosen yet", because the card's folder
    # lives in a browser tab. Written here so "Choose a folder" opens
    # already pointing at what the schedule just produced -- the same
    # reason the detector and the run's shape are held server-side.
    try:
        produced = {
            "folder": root,
            "snapshot_id": snapshot["snapshot_id"],
            "files": snapshot.get("file_count"),
            "set_by": trigger,
            "at": _now().isoformat(timespec="seconds"),
        }
        # Both slots, and they are not the same question.
        #
        # `default_dataset` is "what a run would use right now" and
        # anything may overwrite it -- loading a snapshot by hand does.
        # `last_pull_dataset` is "what the pull produced", which the
        # training card offers as its own choice, so it must survive
        # every manual pick made afterwards.
        pc.write_config({"default_dataset": produced,
                         "last_pull_dataset": produced})
        record("publish", True,
               f"{root} (snapshot {snapshot['snapshot_id']})" if snapshot["snapshot_id"]
               else f"{root} (not frozen)")
    except Exception as exc:                                   # noqa: BLE001
        # Not fatal: the run is queued either way, and a queued entry
        # carries its own dataset_folder. This only decides whether the
        # card shows it.
        record("publish", False, str(exc))

    # "build" stops HERE: the dataset is rebuilt, frozen and published as
    # the one the training card opens on, and nothing is trained. It is
    # what Assemble's "Scheduled" runs after each Data Preparation pull
    # (2026-10-29) -- the data side on its own clock, the training side on
    # its own. Everything below is about a run.
    if submit == "build":
        return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                "ok": True, "steps": steps, "folder": root,
                "snapshot_id": snapshot["snapshot_id"]}

    # 2c. What it continues FROM.
    #
    # A retrain is "the next version of this Class", so the natural
    # starting point is the newest thing already in that Class's
    # registry. Nothing supplied one before and every entry was built
    # with all four source fields empty -- the queue runner then refused
    # it with "Pick exactly one base model source ... (got none)" and
    # SKIPPED it, so the chain reported "Queued." and the queue emptied
    # itself with nothing run. Silent, because the refusal happened in
    # the runner, one step past where the chain stopped looking.
    base_arn = ""
    try:
        from .. import training

        detector = training.get_detector(str(plan["detector_slug"]))
        group = training.detector_group_name(detector, str(plan["architecture"]))
        packages = training.model_registry_summary(group).get("packages") or []
        pinned = str(plan.get("base_version") or "").strip()
        if pinned.lower() == "scratch" and packages:
            # A Class with versions may not be restarted from random
            # weights -- see `scratch_refusal`. Caught here as well as at
            # submit time so the chain stops on a step that says why,
            # rather than failing inside the run.
            record("base", False, training.scratch_refusal(
                str(plan["detector_slug"]), str(plan["architecture"])))
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps,
                    "snapshot_id": snapshot["snapshot_id"]}
        if pinned.lower() == "scratch":
            # No parent. `base_model_scratch` on the queue entry is the
            # same thing the training card's "Training from scratch"
            # does, so the first unattended run of a Class is built by
            # the code a manual first run uses.
            chosen = None
            scratch = True
            record("base", True, f"from scratch -- no parent version in {group}")
        elif pinned:
            scratch = False
            chosen = find_pinned_version(packages, pinned)
            if chosen is None:
                # REFUSED, not fallen back to newest. Somebody pinned a
                # version because the newest one was wrong; quietly
                # training from it anyway would produce exactly the model
                # the pin exists to prevent, and nothing on the run would
                # say so.
                available = ", ".join(
                    "v" + p["ModelPackageArn"].rsplit("/", 1)[-1] for p in packages[:10]
                ) or "none"
                record("base", False,
                       f"Version {pinned!r} is not in {group}. Available: "
                       f"{available}. Change the pinned version, or clear it "
                       f"to continue from the newest Approved one.")
                return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                        "ok": False, "steps": steps,
                        "snapshot_id": snapshot["snapshot_id"]}
        else:
            scratch = False
            # Newest first (list_model_packages sorts Descending). Approved
            # preferred, but NOT required: a Class's first few versions are
            # routinely unapproved, and refusing to continue from them would
            # make the second run impossible.
            chosen = next((p for p in packages
                           if p.get("ModelApprovalStatus") == "Approved"), None)
            chosen = chosen or (packages[0] if packages else None)
        if scratch:
            base_arn = ""
        elif chosen:
            base_arn = chosen["ModelPackageArn"]
            record("base", True,
                   f"{group} v{base_arn.rsplit('/', 1)[-1]} "
                   f"({chosen.get('ModelApprovalStatus', '?')})"
                   + (" -- pinned" if pinned else ""))
        else:
            record("base", False,
                   f"Nothing registered in {group} to continue from. Set "
                   f"\"Continue from\" to \"Train from scratch\" for this "
                   f"Class's first run, import a checkpoint, or run it once "
                   f"by hand choosing a starting version.")
            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": False, "steps": steps,
                    "snapshot_id": snapshot["snapshot_id"]}
    except Exception as exc:                                   # noqa: BLE001
        record("base", False, str(exc))
        return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                "ok": False, "steps": steps,
                "snapshot_id": snapshot["snapshot_id"]}

    # 2d. THE DATASET VERSION. Everything the run reads, resolved to the
    # exact images NOW and written once under `_datasets/<id>/` -- so a
    # build that happens while this waits in the queue (another Add to
    # Queue, the Generate button, the next schedule) cannot change what it
    # trains on. That is how a golden-set-only run came to read a composed
    # run's evaluation yaml. Fatal when it fails: the fallback is exactly
    # the shared, overwritable slot this exists to avoid.
    version_id = ""
    try:
        from .. import data

        version = data.version_for_run(
            home=pc.destination_bucket(config), root=root,
            train_data_uris=train_data_uris, data_yaml_uri=data_yaml_uri,
            eval_root=eval_root, trigger=trigger,
            sources=[s["detail"] for s in steps if s.get("step") == "lists"][-1:],
            drop_undeclared=bool(drop_undeclared))
        version_id = version["version_id"]
        fields = data.version_run_fields(version)
        root = fields["dataset_folder"]
        train_data_uris = fields["train_data_uris"]
        data_yaml_uri = fields["data_yaml_uri"]
        eval_root = fields["eval_dataset_folder"]
        eval_extra_uris = fields["eval_extra_uris"]
        record("version", True, version_id, version_id=version_id,
               train=version["train_images"], val=version["val_images"])
    except Exception as exc:                                   # noqa: BLE001
        record("version", False, f"Could not freeze the dataset version: {exc}")
        return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                "ok": False, "steps": steps,
                "snapshot_id": snapshot["snapshot_id"]}

    # 3. submit -- or queue, and let somebody else decide when
    try:
        from .. import training

        run_notes = " -- ".join(
            part for part in (
                str(plan.get("notes") or "").strip(),
                (f"{trigger} retrain from snapshot {snapshot['snapshot_id']}"
                 if snapshot["snapshot_id"] else f"{trigger} retrain (not frozen)"),
            ) if part
        )
        # "enqueue" is "queue" without releasing the queue: what the
        # training card's Add to Queue does -- it now regenerates the yaml
        # first (2026-10-26), but adding to the queue still leaves the
        # queue as the operator left it.
        if submit in ("queue", "enqueue"):
            generation, size = training.split_architecture(str(plan["architecture"]))
            entry = {
                "id": uuid.uuid4().hex[:12],
                "generation": generation,
                "size": size,
                "epochs": int(plan["epochs"]),
                "img_size": int(plan["img_size"]),
                "batch": int(plan["batch"]),
                "seed": int(plan["seed"]),
                "patience": int(plan.get("patience") or 0),
                "dataset_folder": root,
                # Computed above for the freeze and, until now, dropped
                # here: the run was handed "" and fell back to the
                # training URI, which for a composed run is a
                # comma-separated list the evaluation step cannot read.
                "eval_dataset_folder": eval_root,
                "base_model_folder": "",
                # The open path: this trains a detector's own taxonomy,
                # not the production person/helmet one.
                "custom_classes": True,
                "base_model_package_arn": base_arn,
                "base_model_key": "",
                "base_model_scratch": scratch,
                "architecture": str(plan["architecture"]),
                "detector_slug": str(plan["detector_slug"]),
                "notes": run_notes[:2048],
                # Or a scheduled run trains on defaults while the card
                # says otherwise.
                "hyperparams": plan.get("hyperparams") or {},
                "added_at": _now().isoformat(timespec="seconds"),
                # Kept on the entry so the queue card can say where it
                # came from -- an entry nobody remembers adding is one
                # somebody deletes.
                "queued_by": trigger,
                "snapshot_id": snapshot["snapshot_id"],
                # A composed dataset's prefixes and yaml. Blank for an
                # ordinary one, where both are derived from the folder.
                "train_data_uris": train_data_uris,
                "data_yaml_uri": data_yaml_uri,
                # The training container drops boxes the yaml does not
                # name, when the card says so. Nothing in S3 is changed.
                "drop_undeclared_classes": drop_undeclared,
                "eval_extra_uris": eval_extra_uris,
                # The frozen dataset this entry trains on (2d above).
                "dataset_version_id": version_id,
            }
            # Where the images come from and how many -- from the version,
            # which resolved them; its own folder holds only lists.
            entry.update({k: v for k, v in fields.items() if k.startswith(("dataset_source",
                          "eval_source", "dataset_version_"))})
            training.queue_add(entry)
            record("queue", True, f"queued {entry['id']}", entry_id=entry["id"])
            # Kept on the version, so "Queue again" can replay the same
            # settings on the same images.
            data.mark_version(version_id, state="queued", entry_id=entry["id"],
                     entry_template={k: v for k, v in entry.items()
                                     if k not in ("id", "added_at")})

            # ...and release it. "Do it automatically" promises "rebuild,
            # freeze, then train", and until now it stopped one step
            # short: the entry sat in a queue nobody had started, so an
            # unattended schedule produced a to-do item rather than a
            # model.
            #
            # `queue_start` is the same call the button makes. It is
            # idempotent and refuses politely when the single GPU slot is
            # already busy -- `try_advance_queue` checks
            # `training_already_running_reason` before popping anything,
            # so a schedule firing mid-run queues behind it rather than
            # racing it.
            if submit == "enqueue":
                record("start", True, "queued, not started -- the queue runs it "
                                      "when it is started and the GPU is free")
            else:
                try:
                    started = training.queue_start()
                    record("start", started,
                           "queue started" if started
                           else "nothing to start -- the entry is queued and will "
                                "go when the GPU frees up")
                except Exception as exc:                       # noqa: BLE001
                    # The entry is queued and survives this: a person can
                    # press Start Queue. Not worth failing the chain for.
                    record("start", False, str(exc))

            return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                    "ok": True, "steps": steps,
                    "snapshot_id": snapshot["snapshot_id"]}

        out = training.start_custom_training_run(
            dataset_folder=root,
            eval_dataset_folder=eval_root,
            train_data_uris=train_data_uris,
            data_yaml_uri=data_yaml_uri,
            drop_undeclared_classes=drop_undeclared,
            eval_extra_uris=eval_extra_uris,
            epochs=int(plan["epochs"]), img_size=int(plan["img_size"]),
            batch=int(plan["batch"]), seed=int(plan["seed"]),
            notes=run_notes,
            base_model_package_arn=base_arn,
            architecture=str(plan["architecture"]),
            detector_slug=str(plan["detector_slug"]),
            source=f"retrain-{trigger}",
            # Both of these were being dropped on THIS path only. The
            # queued branch above builds an entry carrying patience and
            # the hyperparameters, and `try_advance_queue` passes them
            # on -- so a schedule that started its run directly trained
            # with different settings from the identical schedule that
            # queued it, and nothing said so. `patience` defaulting to 0
            # is what hid it: the common case looked right.
            patience=int(plan.get("patience") or 0),
            hyperparams=plan.get("hyperparams") or None,
        )
        record("train", True, out.get("run_id", ""), **{
            k: out[k] for k in ("run_id", "audit_id") if k in out})
        data.mark_version(version_id, state="started", run_id=out.get("run_id", ""))
    except Exception as exc:                                   # noqa: BLE001
        record("train", False, str(exc))
        return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
                "ok": False, "steps": steps, "snapshot_id": snapshot["snapshot_id"]}

    return {"trigger": trigger, "at": _now().isoformat(timespec="seconds"),
            "ok": True, "steps": steps, "snapshot_id": snapshot["snapshot_id"]}


def _valid_time(text: str) -> bool:
    parts = str(text or "").split(":")
    if len(parts) != 2:
        return False
    try:
        hour, minute = int(parts[0]), int(parts[1])
    except ValueError:
        return False
    return 0 <= hour <= 23 and 0 <= minute <= 59


def schedule_expression(plan: dict) -> str:
    """The EventBridge expression for this cadence.

    `rate(N days)` rather than a cron: the cadence is "every N days from
    a start", not "on these days of the week", and expressing the first
    as the second silently changes it the moment N is not 1 or 7.
    """
    every = max(1, int(plan.get("every_days") or 1))
    return f"rate({every} day{'s' if every != 1 else ''})"


#: What starts a training run, chosen on Multi-Auto Training's own toolbar
#: (2026-10-29), now that each tab has a schedule of its own.
TRIGGERS = ("pull", "clock", "off")


def training_trigger(config: dict | None = None) -> str:
    """"pull" -- train after each data pull (the chain, as before);
    "clock" -- train on this schedule's own `every_days` at `at`;
    "off" -- nothing trains by itself.

    Stored as `retrain_schedule.trigger`. A config saved before it existed
    has none, and reads as it always behaved: following the pull, whose
    mode was the switch.
    """
    config = config if config is not None else pc.read_config()
    stored = config.get("retrain_schedule") or {}
    trigger = str(stored.get("trigger") or "").strip().lower()
    if trigger in TRIGGERS:
        return trigger
    if stored.get("follow_pull", DEFAULTS["follow_pull"]):
        return "pull"
    return "clock" if stored.get("enabled") else "off"


def trigger_fields(trigger: str) -> dict:
    """The stored fields that make `trigger` true for every reader --
    the after-pull chain, the clock and the preflight all read these."""
    if trigger not in TRIGGERS:
        raise ValueError(f"Unknown training trigger {trigger!r}.")
    return {
        "pull": {"trigger": "pull", "enabled": True, "follow_pull": True},
        "clock": {"trigger": "clock", "enabled": True, "follow_pull": False},
        "off": {"trigger": "off", "enabled": False},
    }[trigger]


def follows_pull(config: dict | None = None) -> bool:
    """Is the PULL the clock for the unattended retrain?

    No, whenever "Use these folders only" is ticked. That switch turns
    Pull Control, Auto Training and Retention off -- the Assemble card
    names its folders outright and no sweep runs -- so chaining onto a
    pull would be hanging the schedule off a mechanism the operator has
    explicitly disabled. Every pull then reports `moved: 0`, and the
    guards that exist to pace pull-driven training (`min_files`,
    `cooldown_minutes`, `even_if_nothing_moved`) are about upload bursts
    that cannot happen.

    The reasoning recorded on `follow_pull` -- that a separate clock can
    fire half way through a sweep and train a week that is still filling
    -- is sound, and is about sweeps. With no sweep there is nothing to
    fire half way through.
    """
    config = config if config is not None else pc.read_config()
    if (config.get("dataset_assemble") or {}).get("only"):
        return False
    return training_trigger(config) == "pull"


def t2_module():
    """`track2_shared`, imported late.

    It imports this module, so a top-level import either way is a
    cycle. Every other call site in here does the same thing inline;
    this is the one that needs it from two places.
    """
    from .. import training

    return t2


def source_selection(config: dict | None = None) -> dict:
    """Which of the three sources feed the final data.yaml.

    Three independent ticks, and the yaml is the UNION of whichever are
    on:

        folders only              -> the Assemble card's folders
        folders + golden          -> both
        folders + golden + pull   -> all three

    The stored forms are deliberately the ones that already existed, so
    a settings file written before this went in still reads correctly
    and every rule downstream keeps working:

    `dataset_assemble.only` stays the stored form of the PULL switch and
    keeps its meaning -- it is what greys Pull Control, Auto Training and
    Retention out, and what `follows_pull` reads. The card shows its
    opposite, because "use Pull Control" is the question being asked and
    "Use these folders only" was an answer to a different one.

    `retention.use_golden` stays the golden set's switch, owned by the
    Golden Set card and read here.

    `dataset_assemble.use` is the only new field, and it defaults to True
    -- every configuration that existed before this was using its
    folders, and a default of False would silently empty their training
    set on the next scheduled run.
    """
    config = config if config is not None else pc.read_config()
    assemble = config.get("dataset_assemble") or {}
    retention = config.get("retention") or {}
    return {
        "folders": bool(assemble.get("use", True)),
        "golden": bool(retention.get("use_golden")),
        "pull": not bool(assemble.get("only")),
    }


def golden_plan(config: dict) -> dict | None:
    """The Golden Set card's build, when that card is live and filled in.

    `None` when the card is switched off or has no source folder yet --
    the schedule then leaves the evaluation set wherever it already
    points.

    This card is NOT greyed out by "Use these folders only": it is a
    curated folder somebody split by hand, and it has nothing to do with
    week folders. So it is one of the cards a scheduled run should act
    on -- and until now nothing consumed `golden_set` at all. It was
    saved by the card and read by no code anywhere, which is why the
    preflight kept warning that there was no separate evaluation set
    while a golden set sat configured.
    """
    saved = config.get("golden_set") or {}
    retention = config.get("retention") or {}
    if not retention.get("use_golden"):
        return None
    source = str(saved.get("source") or "").strip("/")
    target = str(saved.get("target_folder") or "").strip("/")
    if not source or not target:
        return None
    return {
        "source": source,
        "target_folder": target,
        "bucket": str(saved.get("bucket") or ""),
        "target_bucket": str(saved.get("target_bucket") or ""),
        "labels_source": str(saved.get("labels_source") or ""),
        "split": bool(saved.get("split", True)),
        "train_percent": int(saved.get("train_percent") or 80),
    }


#: How late a scheduled slot may still be started, in minutes.
#:
#: The schedule names a TIME OF DAY in Singapore. A run that starts
#: eleven hours after that time did not happen at that time by any
#: reading, and the operator who set it has no way to tell it apart
#: from one that fired correctly.
#:
#: This used to be a whole interval: with `every_days: 1`, a 15:34
#: schedule stayed "due" for the next 24 hours, so it could legitimately
#: start at any hour of the following day. On 2026-10-18 the 15:34 slot
#: started at 10:58 the next morning -- 11h24m late -- and looked from
#: the outside like a timezone fault. It was not; the SGT conversion in
#: `next_runs` was correct. It was this window.
#:
#: Not zero. What starts a run is a two-minute tick, and the container
#: it runs in is replaced by deploys. Thirty minutes covers a deploy
#: (~7 minutes), an instance restart, and a run of failed ticks, while
#: still meaning "it ran at the time I set".
#:
#: A slot older than this is LET GO, not deferred: the next occurrence
#: is the one to wait for. `overdue_training_slot` is what says so, so
#: a skipped run is reported rather than silent.
CATCH_UP_MINUTES = 30


#: Singapore, fixed +08:00 -- no DST here since 1982, so no zone to resolve.
SGT = _dt.timezone(_dt.timedelta(hours=8))

#: Where a schedule with neither `from_date` nor `anchor` counts from --
#: one saved before anchors existed. A FIXED date, so the grid does not
#: depend on when it is asked: such a schedule is stable at once, with no
#: write, and every caller (app, Lambda, a restarted container) computes
#: the same slot. Its next save pins the same grid as a real anchor.
LEGACY_EPOCH = _dt.date(1970, 1, 1)


def _grid_start(plan: dict) -> _dt.datetime | None:
    """Slot zero of the grid, in SGT, or None when `at` is unusable.

    `from_date` (the operator's own "from this date") wins, then the
    stored `anchor`, then `LEGACY_EPOCH`. Never the current time: a grid
    counted from "now" moves with the clock, which is what let a weekly
    schedule fall due daily.
    """
    if not _valid_time(str(plan.get("at") or "")):
        return None
    hour, minute = (int(part) for part in str(plan["at"]).split(":"))
    day = LEGACY_EPOCH
    for key in ("from_date", "anchor"):
        try:
            day = _dt.date.fromisoformat(str(plan.get(key) or "").strip()[:10])
            break
        except ValueError:
            continue
    return _dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=SGT)


def fresh_anchor(plan: dict, *, now: _dt.datetime | None = None) -> str:
    """The anchor for a schedule being switched on or re-timed NOW.

    The date of the first `at` strictly after `now` -- today if the time
    has not come yet, else tomorrow -- so arming a schedule can never
    make a slot that is already due.
    """
    if not _valid_time(str(plan.get("at") or "")):
        return ""
    hour, minute = (int(part) for part in str(plan["at"]).split(":"))
    moment = (now or _now()).astimezone(SGT)
    first = moment.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if first <= moment:
        first += _dt.timedelta(days=1)
    return first.date().isoformat()


def with_anchor(before: dict, after: dict, *,
                now: _dt.datetime | None = None) -> dict:
    """`after`, with the anchor it should be saved with.

    Re-anchored (to `fresh_anchor`) only on a DELIBERATE reset: the
    schedule being switched on, or its interval or time changing. Any
    other save keeps the grid -- and a schedule with no anchor yet pins
    the one it was already showing (the `LEGACY_EPOCH` grid's next date),
    so upgrading moves no run. A disabled schedule is left as it is.
    """
    out = dict(after)
    if not out.get("enabled"):
        return out

    def every(plan):
        try:
            return max(1, int(plan.get("every_days") or 1))
        except (TypeError, ValueError):
            return 1

    reset = (not before.get("enabled")
             or every(before) != every(out)
             or str(before.get("at") or "") != str(out.get("at") or ""))
    if reset:
        out["anchor"] = fresh_anchor(out, now=now)
    elif not str(out.get("anchor") or "").strip():
        upcoming = next_runs({**out, "from_date": ""}, start=now, count=1)
        out["anchor"] = upcoming[0].astimezone(SGT).date().isoformat() if upcoming else ""
    return out


def _passed_slots(plan: dict, moment: _dt.datetime) -> list:
    """The latest two occurrences at or before `moment`, oldest first.

    Counted on the same fixed grid as `next_runs`, so the slot that
    falls due is the one the card showed, and its string is the same
    for every caller at the same moment. Shared by the two readers below
    so they cannot disagree about which slot is the latest one.
    """
    start = _grid_start(plan)
    if start is None or start > moment:
        return []
    step = _dt.timedelta(days=max(1, int(plan.get("every_days") or 1)))
    latest = (moment - start) // step
    return [(start + step * k).astimezone(_dt.timezone.utc)
            for k in range(max(0, latest - 1), latest + 1)]


def overdue_training_slot(config: dict | None = None, *,
                          now: _dt.datetime | None = None) -> dict | None:
    """A slot that passed and is now too late to start, or None.

    Reported rather than acted on. A schedule that quietly does nothing
    reads exactly like a schedule that is working, which is the failure
    this whole module exists to avoid -- so when a run is let go, the
    tick says which one and by how much.
    """
    config = config if config is not None else pc.read_config()
    plan = settings(config)
    if not plan.get("enabled") or follows_pull(config):
        return None
    if not _valid_time(str(plan.get("at") or "")):
        return None
    moment = now or _now()
    passed = _passed_slots(plan, moment)
    if not passed:
        return None
    latest = passed[-1]
    late = moment - latest
    if late <= _dt.timedelta(minutes=CATCH_UP_MINUTES):
        return None
    return {
        "slot": latest.isoformat(timespec="seconds"),
        "minutes_late": int(late.total_seconds() // 60),
    }


def due_training_run(config: dict | None = None, *,
                     now: _dt.datetime | None = None) -> str | None:
    """The scheduled occurrence that is due, or None.

    Returns the occurrence's own UTC timestamp, which is what makes this
    safe to call from three places at once: dev, stable and the
    two-minute Lambda all compute the SAME string for the same slot, and
    `claim_once` lets exactly one of them act on it.

    A slot is due from its time until the next one. Nothing is caught up
    retroactively: a container that was down for a week starts one run,
    not seven.
    """
    config = config if config is not None else pc.read_config()
    plan = settings(config)
    if not plan.get("enabled"):
        return None
    if follows_pull(config):
        return None
    if not _valid_time(str(plan.get("at") or "")):
        return None

    moment = now or _now()
    passed = _passed_slots(plan, moment)
    if not passed:
        return None
    latest = passed[-1]
    # TOO LATE, so let it go and wait for the next occurrence.
    #
    # The schedule names a time of day in Singapore; starting hours
    # after it is not that time of day, and nothing on screen would
    # distinguish the late run from a correct one. See CATCH_UP_MINUTES
    # for the incident.
    if (moment - latest) > _dt.timedelta(minutes=CATCH_UP_MINUTES):
        return None
    return latest.isoformat(timespec="seconds")


def check_training_schedule(config: dict | None = None) -> dict:
    """Start a scheduled retrain if one is due. Never raises.

    Called from the same tick that advances the queue -- the app's poll
    loop and the two-minute Lambda -- so it needs no schedule of its own
    in AWS. `claim_once` is what stops the three of them starting three
    runs for one slot; it is a conditional put, and exactly one caller
    wins.
    """
    config = config if config is not None else pc.read_config()
    slot = due_training_run(config)
    if slot is None:
        missed = overdue_training_slot(config)
        if missed:
            # Named, not swallowed. "Nothing due" is true and useless
            # here: the operator set a time, it passed, and no run
            # happened -- which is the one case they need told.
            return {"started": False, "missed": missed["slot"],
                    "reason": f"The {missed['slot']} run was missed by "
                              f"{missed['minutes_late']} minutes and has been "
                              f"let go. The next occurrence will run."}
        return {"started": False, "reason": "Nothing due."}
    if not pc.claim_once("training_schedule", slot):
        return {"started": False, "reason": "Another instance has this slot."}
    try:
        out = run_chain(config=automatic(config), trigger="schedule", submit="queue")
    except Exception as exc:                                   # noqa: BLE001
        return {"started": False, "slot": slot, "reason": str(exc)}
    return {"started": bool(out.get("ok")), "slot": slot,
            "steps": out.get("steps", [])}


def next_runs(plan: dict, *, start: _dt.datetime | None = None,
              count: int = 4) -> list:
    """When the next few would fire, in UTC.

    Shown in the card because "every 7 days at 02:00" is not something
    anybody can turn into a date in their head, and the first question
    asked of a schedule is always "so when does it actually run".
    """
    # Singapore time, which is what the field says and what the EC2
    # schedule this system already runs on uses.
    #
    # ONE grid, fixed by `_grid_start` (from_date, else anchor, else the
    # legacy epoch) and never by `start`: this used to count from
    # whatever moment it was handed when no from_date was set, and
    # `_passed_slots` handed it "now minus an interval", so the grid slid
    # with the clock and a weekly schedule fell due every day (F2).
    grid = _grid_start(plan)
    if grid is None:
        return []
    every = max(1, int(plan.get("every_days") or 1))
    step = _dt.timedelta(days=every)
    moment = (start or _now()).astimezone(SGT)
    first = grid
    if first <= moment:
        first += step * (((moment - first) // step) + 1)
    return [(first + step * n).astimezone(_dt.timezone.utc)
            for n in range(max(0, int(count)))]
