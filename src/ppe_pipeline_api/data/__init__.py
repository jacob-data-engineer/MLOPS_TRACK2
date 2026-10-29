"""DATA SIDE: everything that turns files in the bucket into a dataset
version a run can train on -- weeks, retention, golden set, assembling,
the data.yaml and the frozen dataset version itself.

Rule: nothing in here imports `training` or `chain`. See
tests/test_side_boundaries.py.
"""

from .._door import through as _through, value as _value

#: What the data side offers. Everything else in data/ is free to change.
__all__ = [
    # the week folders
    "scan_weeks",
    # the data.yaml
    "build_yaml", "write_yaml_in_place", "save_yaml", "find_classes",
    # the retention policy's file lists
    "select_retained", "write_retained", "RETAINED_FOLDER", "RETAINED_TRAIN_NAME",
    # the golden set
    "build_golden",
    # frozen snapshots
    "create_snapshot",
    # THE HANDOVER: a dataset version, the only thing a run trains on
    "version_for_run", "version_run_fields", "mark_version", "version_card_fields",
    # assembling folders into one dataset, and what a dataset declares
    "assemble_dataset", "inspect_dataset_layout", "discover_dataset_sources",
    "dataset_source_map", "image_names_by_stem", "dataset_classes",
    "class_filter_for", "derive_filtered_dataset", "evaluates_on_its_training_images",
    "ASSEMBLED_DATASET_PREFIX", "FILTERED_DATASET_MARKER", "SOURCES_FILE",
    "YAML_IS_THE_ONLY_SOURCE_OF_NAMES",
    # what a dataset's yaml and lists select
    "read_selection",
]

scan_weeks = _through(__name__, "dataset_yaml", "scan_root")
build_yaml = _through(__name__, "dataset_yaml", "build")
write_yaml_in_place = _through(__name__, "dataset_yaml", "write_in_place")
save_yaml = _through(__name__, "dataset_yaml", "save")
find_classes = _through(__name__, "dataset_yaml", "find_classes")
select_retained = _through(__name__, "retained_lists", "select")
write_retained = _through(__name__, "retained_lists", "write")
build_golden = _through(__name__, "golden_set", "build")
create_snapshot = _through(__name__, "dataset_snapshots", "create")
version_for_run = _through(__name__, "dataset_versions", "for_run")
version_run_fields = _through(__name__, "dataset_versions", "run_fields")
mark_version = _through(__name__, "dataset_versions", "mark")
version_card_fields = _through(__name__, "dataset_versions", "card_fields")
assemble_dataset = _through(__name__, "assemble", "assemble_dataset")
inspect_dataset_layout = _through(__name__, "assemble", "inspect_dataset_layout")
discover_dataset_sources = _through(__name__, "assemble", "discover_dataset_sources")
dataset_source_map = _through(__name__, "assemble", "dataset_source_map")
image_names_by_stem = _through(__name__, "assemble", "_basenames")
dataset_classes = _through(__name__, "assemble", "dataset_classes")
class_filter_for = _through(__name__, "assemble", "class_filter_for")
derive_filtered_dataset = _through(__name__, "assemble", "derive_filtered_dataset")
evaluates_on_its_training_images = _through(__name__, "assemble", "_evaluates_on_its_training_images")
read_selection = _through(__name__, "selection", "read_selection")

_CONSTANTS = {
    "RETAINED_FOLDER": ("retained_lists", "FOLDER"),
    "RETAINED_TRAIN_NAME": ("retained_lists", "TRAIN_NAME"),
    "ASSEMBLED_DATASET_PREFIX": ("assemble", "ASSEMBLED_DATASET_PREFIX"),
    "FILTERED_DATASET_MARKER": ("assemble", "FILTERED_DATASET_MARKER"),
    "SOURCES_FILE": ("assemble", "SOURCES_FILE"),
    "YAML_IS_THE_ONLY_SOURCE_OF_NAMES": ("assemble", "_YAML_IS_THE_ONLY_SOURCE_OF_NAMES"),
}


def __getattr__(name: str):
    if name in _CONSTANTS:
        return _value(__name__, *_CONSTANTS[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
