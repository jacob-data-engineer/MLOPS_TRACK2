"""Two gaps between the pipeline as specified and the pipeline as built.

Spec (2026-10-19): a base PT uploaded for a Model Name with no versions
is registered as a SEED; and the golden set may join EITHER data path.

1. The pull (week-folders) branch of the chain built its yaml with
   `data.build_yaml(root=...)`, which knows only the week folders. A golden set
   ticked alongside Pull Control was built, recorded as built, and used
   by nothing -- the held-out number came from the week folders' own val
   side under a label promising a golden one.

2. `register_uploaded_base_model` wrote the SEED marker only when the
   checkpoint's classes did not match the detector's. A COCO yolo11s.pt
   imported into a brand-new one-class Class got it; a helmet checkpoint
   imported into an empty helmet Class was described as a plain import,
   so the first version -- where lineage starts -- was not marked as the
   start of anything.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api import track2_shared as t2     # noqa: E402


def _chain() -> str:
    return " ".join(inspect.getsource(rs._run_chain_body).split())


# --- (1) the golden set on the pull path --------------------------------------

def test_the_pull_branch_composes_when_golden_is_ticked():
    code = _chain()
    assert "if golden_val or golden_train:" in code
    # It takes the composed route, not the week-folders-only builder.
    assert code.index("if golden_val or golden_train:") < code.index("built = data.build_yaml(root=root")


def test_retentions_path_list_is_the_pulls_contribution_when_retention_is_on():
    code = _chain()
    assert 'pull_entries.append({"images": f"{root}/{entries[0].strip(\'/\')}", "labels": ""})' in code


def test_the_week_folders_whole_are_the_contribution_when_it_is_off():
    code = _chain()
    # ...skipping weeks unticked on the Auto Training card.
    assert 'for week in [w for w in data.scan_weeks(root, bucket=config.get("source_bucket") or "") if w["week"] not in set(retention["excluded_weeks"])]: pull_entries.append({"images": week["images_path"], "labels": week["labels_path"]})' in code


def test_the_golden_train_side_joins_training_here_too():
    code = _chain()
    pull = code[code.index("if golden_val or golden_train:"):code.index("built = data.build_yaml(root=root")]
    assert 'if golden_train: pull_entries.append({"images": golden_train, "labels": ""})' in pull


def test_the_golden_val_side_is_the_evaluation_set_here_too():
    code = _chain()
    pull = code[code.index("if golden_val or golden_train:"):code.index("built = data.build_yaml(root=root")]
    # Added to the pull's own val, not substituted for it.
    assert 'if golden_val:' in pull
    assert '+ [{"images": golden_val, "labels": ""}]' in pull


def test_class_names_come_from_the_label_classes_or_the_pulled_sources_yaml():
    """Where this path always took them -- unless the Assemble card has
    been told what the label files mean."""
    code = _chain()
    pull = code[code.index("if golden_val or golden_train:"):code.index("built = data.build_yaml(root=root")]
    assert 'labelled_raw = (config.get("dataset_assemble") or {}).get("label_classes") or {}' in pull
    assert 'names = (data.find_classes(class_sources, bucket=config.get("source_bucket") or "") .get("names") or {})' in pull


def test_no_names_at_all_stops_the_chain_with_a_sentence():
    code = _chain()
    # The literal is split across two source lines; the first fragment is
    # the one that identifies it.
    assert "No class names: nothing beside the pulled sources " in code


def test_it_reads_every_label_file_like_the_other_branch():
    code = _chain()
    pull = code[code.index("if golden_val or golden_train:"):code.index("built = data.build_yaml(root=root")]
    assert "label_scan=None, drop_undeclared=drop_undeclared)" in pull


def test_a_composed_pull_run_carries_its_prefixes_and_yaml_too():
    code = _chain()
    pull = code[code.index("if golden_val or golden_train:"):code.index("built = data.build_yaml(root=root")]
    assert 'train_data_uris = out["download_prefix"]' in pull
    assert 'data_yaml_uri = out["uri"]' in pull
    assert 'expect="DataYamlUri"' in pull


def test_the_old_week_folders_build_is_untouched_when_golden_is_off():
    """Every pull-driven configuration written before this."""
    code = _chain()
    assert "built = data.build_yaml(root=root, retained_train=entries[0]," in code
    assert 'data.save_yaml(key=built["write_to"], text=built["yaml"])' in code


def test_the_step_records_that_golden_was_in_it():
    assert "classes=len(names), golden=True)" in _chain()


# --- (2) a first version is a seed -------------------------------------------

def _reg() -> str:
    return inspect.getsource(t2.register_uploaded_base_model)


def test_first_version_is_decided_from_the_summary_before_the_group_is_touched():
    code = _reg()
    assert 'first_version = not model_registry_summary(group)["total"]' in code
    assert code.index("first_version = not model_registry_summary") < code.index(
        "restart_group_numbering_if_empty(group)")


def test_not_from_the_restart_helpers_return():
    """That returns False for a group that does not exist yet -- the
    commonest first import of all."""
    code = _reg()
    assert "first_version = restart_group_numbering_if_empty" not in code


def test_unknown_is_not_first():
    """A seed marker on a Class that already had trained versions would
    be a lie in the one field whose job is saying what a version is."""
    code = " ".join(_reg().split())
    assert "except Exception: # noqa: BLE001 first_version = False" in code


def test_a_first_version_carries_the_seed_marker():
    code = _reg()
    assert 'f"{SEED_MARKER} First version of this Class' in code
    assert "if first_version else" in code


def test_a_mismatched_import_still_gets_its_own_seed_text():
    """Both are seeds; the texts differ because the situations do."""
    code = _reg()
    assert 'f"{SEED_DESCRIPTION_PREFIX} Its weights detect' in code
    assert "if mismatched else" in code


def test_a_later_matching_import_is_still_a_plain_import():
    code = _reg()
    assert 'f"Imported checkpoint, not trained here. {notes}").strip()' in code


def test_the_marker_is_the_one_everything_already_reads():
    """`is_seed_description` is what the metrics export and the base
    picker consult; a different prefix here would be invisible to them."""
    assert t2.is_seed_description(f"{t2.SEED_MARKER} First version of this Class -- x")
    assert not t2.is_seed_description("Imported checkpoint, not trained here.")
