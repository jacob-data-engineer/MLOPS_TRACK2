"""The unattended retrain: when it fires, and when it refuses to."""

from __future__ import annotations

import datetime as _dt

from ppe_pipeline_api import retrain_schedule as rs


def test_it_is_off_until_somebody_sets_it_up():
    """A retrain takes a GPU for hours and puts a version in a registry.
    Creating the schedule switched on would start doing that before
    anyone had said what to train."""
    assert rs.DEFAULTS["enabled"] is False
    assert rs.settings({}) == rs.DEFAULTS
    # And chained to the pull by default, not on a clock of its own: a
    # separate clock can fire half way through a sweep and train on a
    # week that is still filling.
    assert rs.DEFAULTS["follow_pull"] is True


def test_the_cadence_is_a_rate_not_a_cron():
    """The cadence is "every N days from a start", not "on these days of
    the week". Expressing the first as the second silently changes it the
    moment N is neither 1 nor 7."""
    assert rs.schedule_expression({"every_days": 7}) == "rate(7 days)"
    assert rs.schedule_expression({"every_days": 1}) == "rate(1 day)"
    assert rs.schedule_expression({"every_days": 10}) == "rate(10 days)"
    # Nonsense lands somewhere sane rather than producing an expression
    # EventBridge would reject at apply time.
    assert rs.schedule_expression({"every_days": 0}) == "rate(1 day)"


def test_the_times_are_singapore_and_come_back_as_utc():
    """The field says SGT because that is the clock the EC2 schedule this
    system already runs on uses. AWS wants UTC."""
    start = _dt.datetime(2026, 9, 29, 1, 0, tzinfo=_dt.timezone.utc)  # 09:00 SGT
    # Anchored, as every saved schedule now is: without one the grid
    # counts from a fixed epoch, not from `start` (F2).
    runs = rs.next_runs({"at": "02:00", "every_days": 7, "anchor": "2026-09-30"},
                        start=start, count=3)

    # 02:00 SGT is 18:00 UTC the day before.
    assert [m.hour for m in runs] == [18, 18, 18]
    assert runs[0] == _dt.datetime(2026, 9, 29, 18, 0, tzinfo=_dt.timezone.utc)
    assert (runs[1] - runs[0]).days == 7
    assert (runs[2] - runs[1]).days == 7


def test_a_time_already_past_today_goes_to_tomorrow():
    """Otherwise the card would show a first run in the past, which reads
    as "it should have happened and did not"."""
    # 20:00 SGT, asking for 02:00 -- today's slot has gone.
    start = _dt.datetime(2026, 9, 29, 12, 0, tzinfo=_dt.timezone.utc)
    runs = rs.next_runs({"at": "02:00", "every_days": 1}, start=start, count=1)
    assert runs[0] > start


def test_a_broken_time_yields_no_runs_rather_than_a_guess():
    assert rs.next_runs({"at": "", "every_days": 7}) == []
    assert rs.next_runs({"at": "2pm", "every_days": 7}) == []
    assert rs.next_runs({"at": "25:00", "every_days": 7}) == []


def test_preflight_blocks_on_everything_that_would_waste_a_run():
    """A retrain is expensive and hard to undo. Every precondition is
    checked BEFORE anything starts, so a miss stops it with a reason
    rather than training on whatever happened to be there."""
    out = rs.preflight({"root": "", "picked": {}, "retrain_schedule": {}})
    assert out["ready"] is False
    blocking = [p["text"] for p in out["problems"] if p["level"] == "blocking"]
    assert any("root folder" in t for t in blocking)
    assert any("detector" in t for t in blocking)


def test_preflight_passes_once_the_two_things_it_needs_are_set(monkeypatch):
    """`ready` is about BLOCKING problems only.

    Preflight also reports warnings that depend on the live estate --
    week folders holding images no week covers, week definitions that
    name a range nothing reads, metrics measured on a split that also
    chose best.pt. Those are advice, not refusals, and asserting an empty
    list made this test a statement about one S3 bucket's contents rather
    than about preflight. Stubbed here so it checks what it is named for.
    """
    monkeypatch.setattr(rs.pc, "orphan_week_folders", lambda config=None: [])
    monkeypatch.setattr(rs.pc, "week_definition_problems", lambda config=None: [])
    out = rs.preflight({
        "root": "retraining",
        "mode": "schedule",
        "picked": {"train_images": ["src/images/train"]},
        "retrain_schedule": {"detector_slug": "helmet"},
        "default_dataset": {"eval_folder": "held_out"},
    })
    assert out["ready"] is True
    assert out["problems"] == []


def test_preflight_warnings_never_make_it_unready(monkeypatch):
    """Every new advisory check must stay advisory. A stray week folder
    is not a reason to refuse to train."""
    monkeypatch.setattr(rs.pc, "orphan_week_folders",
                        lambda config=None: [{"name": "week_4", "images": 7}])
    monkeypatch.setattr(rs.pc, "week_definition_problems",
                        lambda config=None: [{"kind": "prefix", "week": "day_1",
                                              "detail": "day_1 is misnamed"}])
    out = rs.preflight({
        "root": "retraining",
        "mode": "schedule",
        "picked": {"train_images": ["src/images/train"]},
        "retrain_schedule": {"detector_slug": "helmet"},
    })
    assert out["ready"] is True
    assert [p["level"] for p in out["problems"]] == ["warn"] * len(out["problems"])
    texts = " ".join(p["text"] for p in out["problems"])
    assert "week_4" in texts and "day_1" in texts
    # And the one about where the numbers come from.
    assert "selection" in texts or "optimistic" in texts
    assert out["root"] == "retraining"


def test_following_the_pull_warns_when_the_pull_has_no_schedule():
    """Chained to the pull means the pull is the clock. If the pull is
    not on one, nothing triggers this -- and a schedule that silently
    never fires reads as "the model is being retrained"."""
    out = rs.preflight({
        "root": "retraining",
        "mode": "off",
        "picked": {"train_images": ["src"]},
        "retrain_schedule": {"detector_slug": "helmet"},
    })
    # A warning, not a block: Run it now still works, and training on
    # demand is a legitimate way to use this.
    assert out["ready"] is True
    assert any(p["level"] == "warn" and "not on a schedule" in p["text"]
               for p in out["problems"])


def test_following_the_pull_ignores_its_own_clock_fields():
    """They are not read in that mode, so validating them would block on
    fields nobody has set and nothing uses."""
    out = rs.preflight({
        "root": "retraining",
        "mode": "schedule",
        "picked": {"train_images": ["src"]},
        "retrain_schedule": {"detector_slug": "helmet", "every_days": 0,
                             "at": "nonsense"},
    })
    assert out["ready"] is True


def test_an_interval_under_a_day_is_refused():
    """A retrain takes hours of GPU, so anything under a day would
    overlap itself."""
    out = rs.preflight({
        "root": "retraining",
        "picked": {"train_images": ["src"]},
        "retrain_schedule": {"detector_slug": "helmet", "follow_pull": False,
                             "every_days": 0, "at": "02:00"},
    })
    assert out["ready"] is False
    assert any("overlap" in p["text"] for p in out["problems"])


def test_no_source_folders_warns_but_does_not_block():
    """Training on what is already filed is a legitimate thing to want,
    so this warns rather than refusing.

    It used to be gated on a `pull_first` setting. That setting was
    removed: `run_chain` does not sweep -- it rebuilds the lists, the
    yaml and the snapshot and submits -- so an option promising to pull
    first described a step that did not exist, while the warning it
    gated described a sweep this code never performs. The warning that
    remains is the true one: with no sources picked, nothing new reaches
    the week folders and each run trains on the same images."""
    out = rs.preflight({
        "root": "retraining",
        "picked": {},
        "mode": "schedule",
        "retrain_schedule": {"detector_slug": "helmet"},
    })
    assert out["ready"] is True
    assert any(p["level"] == "warn" and "same images" in p["text"]
               for p in out["problems"])


def test_the_removed_settings_are_not_quietly_accepted_again():
    """`pull_first` and `skip_if_unchanged` were settable from the UI and
    read by nothing. A field nobody reads is worse than a missing one --
    it is a promise the system does not keep."""
    import inspect
    source = inspect.getsource(rs)
    assert "pull_first" not in source.replace(
        "#: NOTE: `pull_first` and `skip_if_unchanged` used to sit here.", "")
    assert "skip_if_unchanged" not in source.replace(
        "#: NOTE: `pull_first` and `skip_if_unchanged` used to sit here.", "")


def test_it_targets_the_open_pipeline_not_the_production_lineage():
    """A model trained on somebody's own classes must not reach the
    replay lineage, whose consumers read index 0 as person."""
    assert "open" in rs.PIPELINE_NAME
    assert rs.PIPELINE_NAME != "track2-yolo-training-pipeline"


def test_the_yaml_entry_comes_from_write_not_rebuilt_from_FOLDER():
    """`retained_lists.FOLDER` became "" when the lists moved to the root
    of the downloaded prefix. The chain used to rebuild the yaml entry as
    f"{FOLDER}/{TRAIN_NAME}", which then produced "/train.txt" -- a
    LEADING SLASH the trainer cannot resolve, in the unattended path
    nobody watches.

    `write()` already computes the entry and is the only thing that knows
    where it actually put the file, so the chain takes it from there."""
    import inspect
    from ppe_pipeline_api import retained_lists as rl

    # The shape that broke it, demonstrated rather than described.
    assert rl.FOLDER == ""
    assert f"{rl.FOLDER}/{rl.TRAIN_NAME}" == "/train.txt"

    # The BODY, not `run_chain`: that is now a thin wrapper whose only
    # job is to close the progress row on every exit, including the one
    # reached by an exception. The chain itself is here.
    source = inspect.getsource(rs._run_chain_body)
    assert 'written["train_entry"]' in source
    assert 'written["val_entry"]' in source
    assert "data.RETAINED_FOLDER" not in source


def test_the_chain_publishes_the_dataset_it_built():
    """Otherwise the unattended path is invisible: it rebuilds the lists,
    writes the yaml and freezes a snapshot, and the training card still
    reads "Nothing chosen yet" because its folder lives in a browser
    tab."""
    import inspect
    from ppe_pipeline_api import pull_control as pc

    assert "default_dataset" in pc.DEFAULTS
    source = inspect.getsource(rs._run_chain_body)
    assert 'pc.write_config({"default_dataset"' in source
    # ...carrying the snapshot, so the card can name what was frozen.
    assert '"snapshot_id": snapshot["snapshot_id"]' in source


def test_the_chain_supplies_what_the_UI_supplies_to_build():
    """The unattended build failed with "No data.yaml was found to read
    class names from" on a root the UI builds happily.

    The UI passes `class_sources` (where the pull reads from) and `names`
    (its editable field). The chain passed neither, so the search looked
    nowhere useful -- and there is no operator to type names at 18:33."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert "class_sources=class_sources" in source
    assert "source_bucket=" in source


def test_build_falls_back_to_the_root_its_own_yaml():
    """With no data.yaml beside the sources and no operator to type
    names, the root's existing yaml is the record of what the label
    indices in these week folders mean."""
    import inspect
    from ppe_pipeline_api import dataset_yaml as dy
    source = inspect.getsource(dy.build)
    assert "find_classes([training_root]" in source


def test_queue_mode_also_starts_the_queue():
    """"Do it automatically" promised "rebuild, freeze, then train" and
    stopped one step short: the entry sat in a queue nobody had started,
    so an unattended schedule produced a to-do item rather than a model.

    Safe unattended because the queue serialises on the account's one GPU
    slot -- `try_advance_queue` refuses to pop while anything is
    Executing, so a schedule firing mid-run waits behind it."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert "training.queue_start()" in source
    # ...and the failure to start must not lose the queued entry.
    assert 'record("start", False, str(exc))' in source


def test_starting_is_reported_as_its_own_step():
    """A chain that queued and failed to start must say which of the two
    happened -- "ok" covering both would hide a run that never began."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert 'record("queue", True' in source
    assert 'record("start"' in source


def test_the_chain_resolves_a_base_model_before_queueing():
    """Every entry used to be built with all four source fields empty.
    The queue runner then refused it -- "Pick exactly one base model
    source ... (got none)" -- and SKIPPED it, so the chain reported
    "Queued." and the queue emptied itself with nothing run.

    Silent, because the refusal happened in the runner, one step past
    where the chain stopped looking. A retrain is "the next version of
    this Class", so it continues from the newest thing in that Class's
    registry."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert "detector_group_name" in source
    assert "model_registry_summary" in source
    assert '"base_model_package_arn": base_arn' in source
    # ...and the direct-start path takes it too, or it fails the same way.
    assert "base_model_package_arn=base_arn" in source


def test_an_empty_registry_stops_the_chain_rather_than_queueing_junk():
    """A Class with nothing registered cannot be continued from. Saying
    so is better than queueing an entry that the runner will refuse and
    discard ten seconds later."""
    import inspect
    source = inspect.getsource(rs._run_chain_body)
    assert "Nothing registered in" in source
    assert 'record("base", False' in source
