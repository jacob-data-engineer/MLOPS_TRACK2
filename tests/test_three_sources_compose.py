"""The final data.yaml is the union of whichever sources are ticked.

    these folders only            -> just those folders
    these folders + golden        -> both
    these folders + golden + pull -> all three

Three independent ticks, not a mode. They were a mode until now: the
Assemble card's "Use these folders only" turned the pull off, the Golden
Set card offered "Use a golden set / Not using one", and the chain's
if/elif/else picked exactly ONE of frozen, assembled or week folders. A
golden set configured alongside assembled folders contributed its val
side and nothing else; its train side -- "trained on in full and never
decays", in the card's own words -- was never named in any yaml.

The stored fields are deliberately the OLD ones, so a settings file
written before this still reads correctly:

  * `dataset_assemble.only` is still the pull switch and still greys the
    three cards out. The tick shows its opposite.
  * `retention.use_golden` is still the golden set's switch.
  * `dataset_assemble.use` is the only new field, and absent means True.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy      # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
ASSEMBLE = WEB / "components" / "Track2DatasetAssemblePanel.tsx"
GOLDEN = WEB / "components" / "Track2GoldenSetPanel.tsx"


def _flat(path: pathlib.Path) -> str:
    return " ".join(path.read_text(encoding="utf-8").split())


# --- which sources are on ---------------------------------------------------------

def test_all_three_sources_can_be_on_at_once():
    """Three independent ticks (2026-10-20, replacing a short-lived
    either/or): the assembled folders, the golden set and the pull all
    compose into one yaml when all three are ticked."""
    sel = rs.source_selection({"retention": {"use_golden": True},
                               "dataset_assemble": {"use": True, "only": False}})
    assert sel == {**sel, "folders": True, "golden": True, "pull": True}
    # A configuration from before the folders tick still uses its folders.
    assert rs.source_selection({})["folders"] is True
    assert rs.source_selection({"dataset_assemble": {"use": False}})["folders"] is False


def test_the_folders_tick_can_switch_them_off():
    assert rs.source_selection({"dataset_assemble": {"use": False}})["folders"] is False


def test_the_golden_tick_is_retentions_own_field():
    """One field, two cards. A copy would be a second answer."""
    assert rs.source_selection({"retention": {"use_golden": True}})["golden"] is True
    assert rs.source_selection({})["golden"] is False


def test_the_pull_tick_is_the_opposite_of_use_these_folders_only():
    """`only` keeps its stored meaning -- it is what greys Pull Control,
    Auto Training and Retention out, and what `follows_pull` reads."""
    assert rs.source_selection({})["pull"] is True
    assert rs.source_selection({"dataset_assemble": {"only": True}})["pull"] is False


def test_only_still_takes_the_pull_off_the_trigger():
    """The composition change must not quietly re-arm pull-driven
    chaining for a configuration that switched it off."""
    assert rs.follows_pull({"dataset_assemble": {"only": True},
                            "retrain_schedule": {"follow_pull": True}}) is False


# --- the golden set has two sides ---------------------------------------------------

def test_the_golden_train_side_is_named_now():
    code = inspect.getsource(rs._run_chain_body)
    assert "golden_train = f\"{golden['target_folder']}/images/train\"" in code
    assert '{"images": golden_train, "labels": ""}' in code


def test_with_the_split_off_it_is_all_training_data():
    """`golden_set.build` writes one folder and no val side when the
    split is off. Naming `images/val` regardless pointed the evaluation
    at a prefix with nothing under it."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert 'else: golden_train = f"{golden[\'target_folder\']}/images" golden_val = ""' in code


def test_the_golden_val_side_is_added_not_substituted():
    """Decided 2026-10-20: substituting left the dataset's own 1,693 val
    images used by nothing. Both are scored now."""
    code = inspect.getsource(rs._run_chain_body)
    assert 'kept = list(splits.get("val") or []) if picks["folders"] else []' in code
    assert 'splits = {**splits, "val": kept + [{"images": golden_val, "labels": ""}]}' in code


# --- the union ----------------------------------------------------------------------

def test_every_ticked_source_is_added_to_training():
    code = inspect.getsource(rs._run_chain_body)
    assert 'if picks["folders"]:' in code
    assert "if golden_train:" in code
    assert 'if picks["pull"] and root:' in code
    assert 'splits = {**splits, "train": train_entries}' in code


def test_retentions_selection_arrives_as_its_path_list():
    """The decay curve is a .txt of image paths. Naming the week folders
    instead would train every week whole and discard the policy."""
    # Named by `pull_train_entries`, shared with the Assemble preview.
    assert "pull_train_entries(config, root, write=True)" in inspect.getsource(rs._run_chain_body)
    code = inspect.getsource(rs.pull_train_entries)
    assert "chosen = data.write_retained(root=root, config=retention)" in code
    assert 'entry = chosen["train_entry"].strip("/")' in code


def test_with_retention_off_the_week_folders_go_in_whole():
    code = " ".join(inspect.getsource(rs.pull_train_entries).split())
    assert "weeks = [w for w in data.scan_weeks(root," in code
    assert '{"images": w["images_path"], "labels": w["labels_path"]}' in code


def test_nothing_ticked_stops_the_chain_rather_than_training_on_nothing():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "Nothing is ticked to train on." in code
    assert code.index("Nothing is ticked to train on.") < code.index("data.write_yaml_in_place(")


def test_a_pull_that_cannot_be_read_stops_the_chain():
    """Training on two of three sources when three were asked for is a
    quietly smaller dataset, and the run would report numbers for it."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "Pull Control is ticked but its week folders could " in code


def test_dropping_the_folders_drops_their_val_side_too():
    """It would otherwise score the run on images nothing trained on."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert 'elif not picks["folders"]: # The val side came from the Assemble card' in code


def test_the_step_says_what_went_into_it():
    code = inspect.getsource(rs._run_chain_body)
    assert 'record("lists", True, "trained on " + ", ".join(used)' in code


# --- a .txt is a source like any other ----------------------------------------------

def test_a_path_list_is_recognised():
    assert dy._is_path_list("retraining/train.txt")
    assert not dy._is_path_list("retraining/images/train")


def test_a_path_list_is_downloaded_as_the_folder_it_sits_in():
    """Its lines are relative to the file, and Ultralytics resolves them
    against its parent -- so fetching the list alone would arrive with
    every line pointing at nothing."""
    assert dy._download_dir_for("retraining/train.txt") == "retraining"
    assert dy._download_dir_for("retraining/images/train") == "retraining/images/train"


def test_a_path_list_skips_the_labels_derivation():
    """There is no folder to swap the images segment in -- each LINE's
    labels are derived from that line."""
    code = inspect.getsource(dy.build_in_place)
    assert "if _is_path_list(images):" in code
    assert code.index("if _is_path_list(images):") < code.index("expected = _labels_dir_for(images)")


def test_a_list_and_a_folder_under_unrelated_roots_are_two_downloads():
    assert sorted(dy._bases_for(["Testing_Dataset/ds/images/train",
                                 "retraining/train.txt"])) == \
        ["Testing_Dataset/ds", "retraining"]


def test_a_list_beside_its_own_folders_is_one_download():
    assert dy._bases_for(["retraining/week_1/images", "retraining/train.txt"]) == \
        ["retraining"]


# --- and the run is told where everything is -----------------------------------------

def test_the_composed_prefixes_reach_the_queue_entry():
    code = inspect.getsource(rs._run_chain_body)
    assert '"train_data_uris": train_data_uris' in code
    assert '"data_yaml_uri": data_yaml_uri' in code


def test_they_reach_a_run_started_directly_too():
    """The queued and the direct path have drifted apart before --
    patience and hyperparams were dropped on this one only."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "train_data_uris=train_data_uris, data_yaml_uri=data_yaml_uri," in code


def test_an_ordinary_single_folder_run_sends_neither():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert 'train_data_uris = "" data_yaml_uri = "" fetch_roots: list = []' in code


def test_the_freeze_records_every_prefix_not_just_the_yamls_folder():
    """The generated folder holds the yaml and nothing else. Walking it
    alone would freeze a manifest of one file and call it the dataset."""
    code = inspect.getsource(rs._run_chain_body)
    assert "extra_roots=fetch_roots" in code


# --- the three ticks are on the screen -------------------------------------------------

def test_each_tick_sits_on_the_card_it_switches():
    """They were all three in the Assemble heading to begin with, which
    put one question in two places: the Golden Set and Pull control
    cards each already had, or now have, the same switch in their own
    heading. A tick belongs with the thing it turns off."""
    assemble = _flat(ASSEMBLE)
    assert "Use these folders </label>" in assemble
    assert "Use Pick Golden Set / Edge Cases" not in assemble
    # Named in prose, pointing at where it lives -- but not a control.
    assert "checked={!only}" not in assemble

    pull = _flat(WEB / "components" / "Track2PullControlPanel.tsx")
    assert "Use Pull Control" in pull
    assert "checked={!only}" in pull

    golden = _flat(GOLDEN)
    assert "Use Pick Golden Set / Edge Cases" in golden


def test_the_assemble_card_still_says_where_the_pull_tick_is():
    """A card that says "those three are off" and does not say what
    turns them back on sends somebody hunting."""
    code = _flat(ASSEMBLE)
    assert "Use Pull Control</span> in the Pull control heading" in code


def test_the_pull_tick_is_shown_inverted():
    """`only` keeps its stored meaning; the tick asks the opposite
    question, so it is drawn from its negation."""
    code = _flat(WEB / "components" / "Track2PullControlPanel.tsx")
    assert "checked={!only}" in code
    assert "onChange={(e) => onOnlyChange(!e.target.checked)}" in code


def test_the_card_says_what_the_yaml_will_be_built_from():
    code = _flat(ASSEMBLE)
    assert "data.yaml will be built from:" in code
    assert "nothing &mdash; tick at least one" in code


def test_the_golden_tick_is_in_the_title_row():
    """Asked for on the same row as the card's name. `titleAside` is the
    slot for exactly this -- `action` is the far right, next to Hide,
    where a tick reads as belonging to Hide rather than to the title."""
    code = _flat(GOLDEN)
    assert "titleAside={" in code
    assert code.index("titleAside={") < code.index("Use Pick Golden Set / Edge Cases")
    assert code.index("Use Pick Golden Set / Edge Cases") < code.index("Step skipped.")


def test_what_the_tick_means_stays_in_the_body():
    """Too long for a title row, and only worth saying when the answer
    is no."""
    code = _flat(GOLDEN)
    assert "{inUse === false && (" in code
    assert "Training uses whichever other sources are ticked." in code


def test_the_golden_card_is_a_tick_not_a_pair_of_buttons():
    code = _flat(GOLDEN)
    assert "Use Pick Golden Set / Edge Cases" in code
    assert "Not using one" not in code
    assert '"Use a golden set"' not in code


def test_only_the_golden_card_writes_the_golden_switch():
    """One writer. The Assemble card reads it for its summary line and
    never sets it, which is what stops the two disagreeing."""
    golden = _flat(GOLDEN)
    assert "announceGolden" in golden
    assert "saveTrack2Retention({ use_golden:" in golden

    assemble = _flat(ASSEMBLE)
    assert "announceGolden" not in assemble
    assert "saveTrack2Retention" not in assemble


def test_the_summary_line_still_follows_the_other_cards_live():
    """It reads two switches set elsewhere on the same page. Without
    the listener it would be right only after a reload."""
    assert "onGoldenChanged(setUseGolden)" in _flat(ASSEMBLE)


def test_the_new_field_is_in_the_client_type():
    api = (WEB / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "use?: boolean;" in api


def test_the_folders_tick_is_saved_like_every_other_setting():
    code = _flat(ASSEMBLE)
    assert 'if (typeof saved.use === "boolean") setUseFolders(saved.use);' in code
    assert "use: useFolders," in code
    # Its own tick -- not the pull's switch.
    assert "onChange={(e) => onOnlyChange(e.target.checked)}" not in code


# --- the run has to be startable before the work is spent ----------------------------

def test_a_composed_build_checks_the_pipeline_before_freezing():
    """The yaml is written by then, which is cheap. The freeze and the
    queue entry are not -- and `try_advance_queue` DROPS an entry it
    cannot start rather than putting it back, so the chain would report
    "queued" and the queue would empty itself with nothing trained."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert 'declared = t2_module().pipeline_parameter_names( expect="DataYamlUri")' in code
    assert 'if declared and "DataYamlUri" not in declared:' in code
    assert code.index("DataYamlUri") < code.index("snapshot = data.create_snapshot(")


def test_an_unknown_answer_is_not_treated_as_missing():
    """Refusing a run because one describe call failed would be worse
    than letting it reach the check at submit time."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "declared = set()" in code


def test_a_single_folder_run_is_never_asked():
    """The check sits inside the many-roots branch, so every run that
    has ever worked skips it entirely."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert code.index('if out.get("many_roots"):') < code.index("pipeline_parameter_names(")


# --- the run card says what the run is actually built from ---------------------------

PAGE = WEB / "routes" / "Track2ModelsPage.tsx"


def test_the_training_data_option_names_every_ticked_source():
    """It said "Use the assembled dataset" and named one folder. With
    the golden set ticked that folder is one of two the run trains
    from, and nothing on the card said so."""
    code = _flat(PAGE)
    assert '"Use the assembled dataset + the other ticked sources"' in code
    assert "the golden set / edge cases" in code
    assert "Pull Control's week folders" in code


def test_it_reads_the_same_three_fields_the_chain_reads():
    code = _flat(PAGE)
    assert "folders: status.config.dataset_assemble?.use !== false," in code
    assert "golden: Boolean(status.config.retention?.use_golden)," in code
    assert "pull: !status.config.dataset_assemble?.only," in code


def test_it_says_where_the_held_out_set_comes_from():
    """Which is the part that decides whether a score means anything."""
    code = _flat(PAGE)
    assert "The golden set's val side is the held-out evaluation set." in code


def test_nothing_ticked_says_so_here_too():
    code = _flat(PAGE)
    assert "No source is ticked. Tick at least one on the cards above." in code


def test_the_folder_line_admits_it_is_not_the_whole_dataset():
    code = _flat(PAGE)
    assert "after the retention policy when Pull Control is ticked." in code


def test_the_field_the_card_reads_is_declared_on_the_type():
    """`tsc` is the gate here and it failed on exactly this: `retention`
    rides in the pull config server-side -- `pull_control.DEFAULTS` has
    it -- but the client type had never declared it."""
    api = (WEB / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "retention?: { use_golden?: boolean };" in api


def test_it_is_marked_read_only_so_nobody_tries_to_save_it():
    """`ConfigRequest` does not declare it, so a save carrying it would
    be dropped in silence -- which is the trap this repo keeps hitting."""
    api = " ".join((WEB / "lib" / "api.ts").read_text(encoding="utf-8").split())
    assert "READ ONLY from here." in api
