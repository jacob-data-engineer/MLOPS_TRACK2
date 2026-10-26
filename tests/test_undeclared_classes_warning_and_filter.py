"""Labels the yaml cannot name: a warning with two answers, not a wall.

The gate from 19 Oct refused outright. The operator asked for the choice
to be theirs, in one place: a WARNING naming the folders and indices,
then either NAME THE REST (placeholders into Label classes) or FILTER --
the training container drops those boxes from its downloaded copy before
Ultralytics reads them, and the label files in S3 stay exactly as they
are. The filter is a saved checkbox; the unattended chain refuses until
one of the two has been taken.
"""
from __future__ import annotations

import inspect
import json
import os
import pathlib
import sys
import tempfile

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy      # noqa: E402
from ppe_pipeline_api import pull_control as pc      # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api import track2_shared as t2     # noqa: E402
from ppe_pipeline_api.routers import track2_models   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
PANEL = (WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")
FLAT = " ".join(PANEL.split())
API = (WEB / "lib" / "api.ts").read_text(encoding="utf-8")
TRAIN_DIR = ROOT / "model_training" / "training-container"
EVALUATOR = (ROOT / "model_training" / "custom_training" / "evaluate_any.py").read_text(encoding="utf-8")
PIPELINE = json.loads((ROOT / "model_training" / "custom_training" / "pipeline.json").read_text(encoding="utf-8"))


# --- the gate reports, the write refuses, the filter lets it through ---------

class _Body:
    def __init__(self, text: str):
        self._t = text.encode("utf-8")

    def read(self):
        return self._t


class _S3:
    def __init__(self, files: dict):
        self.files = files
        self.put = []

    def get_object(self, Bucket, Key):
        return {"Body": _Body(self.files[Key])}

    def put_object(self, **kw):
        self.put.append(kw["Key"])


@pytest.fixture
def mixed(monkeypatch):
    files = {
        "ds/labels/train/a.txt": "0 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.1 0.1\n",
        "ds/labels/train/b.txt": "3 0.5 0.5 0.1 0.1\n",
    }
    s3 = _S3(files)
    monkeypatch.setattr(pc, "s3", s3)
    monkeypatch.setattr(pc, "_list_objects",
                        lambda prefix, bucket="", **kw: [{"Key": k, "Size": 1}
                                                         for k in files if k.startswith(prefix)])
    monkeypatch.setattr(pc, "destination_bucket", lambda: "b")
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    return s3


SPLITS = {"train": [{"images": "ds/images/train", "labels": ""}],
          "val": [{"images": "ds/images/val", "labels": ""}]}


def test_the_preview_reports_the_mismatch_instead_of_dying(mixed):
    out = dy.build_in_place(splits=SPLITS, classes={0: "helmet"}, bucket="b")
    assert len(out["undeclared"]) == 1
    assert out["undeclared"][0]["entry"] == "ds/images/train"
    assert out["undeclared"][0]["indices"] == [2, 3]
    assert out["drop_undeclared"] is False


def test_the_sentence_names_both_answers(mixed):
    out = dy.build_in_place(splits=SPLITS, classes={0: "helmet"}, bucket="b")
    text = out["undeclared"][0]["text"]
    assert "name the rest in Label classes" in text
    assert "tick 'Filter undeclared classes'" in text
    assert "label files in S3 unchanged" in text


def test_the_write_refuses_until_one_answer_is_taken(mixed):
    with pytest.raises(ValueError, match="its label files use class 2, 3"):
        dy.write_in_place(splits=SPLITS, classes={0: "helmet"}, bucket="b")
    assert mixed.put == []


def test_naming_the_rest_lets_the_write_through(mixed):
    out = dy.write_in_place(splits=SPLITS, classes={0: "helmet", 2: "class_2", 3: "class_3"},
                            bucket="b")
    assert out["undeclared"] == []
    assert mixed.put


def test_the_filter_lets_the_write_through_and_records_the_mismatch(mixed):
    out = dy.write_in_place(splits=SPLITS, classes={0: "helmet"}, bucket="b",
                            drop_undeclared=True)
    assert out["drop_undeclared"] is True
    assert out["undeclared"][0]["indices"] == [2, 3]
    assert mixed.put


def test_a_many_roots_document_carries_the_same_fields(mixed, monkeypatch):
    monkeypatch.setattr(dy, "_bases_for", lambda dirs: ["a", "b"])
    out = dy.build_in_place(splits=SPLITS, classes={0: "helmet"}, bucket="b")
    assert out["many_roots"] is True
    assert "undeclared" in out and "drop_undeclared" in out


def test_the_scan_still_runs_before_the_yaml_is_shaped():
    code = inspect.getsource(dy.build_in_place)
    assert code.index("_labels_beyond(") < code.index("train_bases = ")


# --- the chain and the run ------------------------------------------------------

def test_the_chain_reads_the_tick_once_for_every_branch():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert 'drop_undeclared = bool((config.get("dataset_assemble") or {}).get("drop_undeclared_labels"))' in code
    assert code.count("drop_undeclared=drop_undeclared)") == 2


def test_the_entry_and_the_direct_start_carry_it():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert '"drop_undeclared_classes": drop_undeclared,' in code
    assert "drop_undeclared_classes=drop_undeclared," in code


def test_assemble_plan_exposes_it(monkeypatch):
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {"detector_slug": "helmet",
                                "train_sources": [{"images": "x/images/train", "labels": ""}],
                                "drop_undeclared_labels": True}}
    assert rs.assemble_plan(cfg)["drop_undeclared"] is True


def test_the_parameter_is_sent_only_when_on():
    code = " ".join(inspect.getsource(t2.start_custom_training_run).split())
    assert '"DropUndeclaredClasses": "true" if drop_undeclared_classes else "",' in code


def test_each_optional_parameter_is_checked_against_the_live_pipeline():
    code = " ".join(inspect.getsource(t2.start_custom_training_run).split())
    assert 'for optional in ("DataYamlUri", "DropUndeclaredClasses", "EvalExtraUris"):' in code
    assert "declared = pipeline_parameter_names(expect=optional)" in code


def test_the_queue_runner_passes_it_on():
    code = " ".join(inspect.getsource(t2.try_advance_queue).split())
    assert 'drop_undeclared_classes=bool(entry.get("drop_undeclared_classes"))' in code


def test_the_request_model_declares_it():
    assert "drop_undeclared" in track2_models.DatasetYamlInPlaceRequest.model_fields
    assert track2_models.DatasetYamlInPlaceRequest.model_fields["drop_undeclared"].default is False


def test_the_setting_is_documented_in_defaults():
    assert "drop_undeclared_labels" in inspect.getsource(pc)


# --- the training container -------------------------------------------------------

train = pytest.importorskip("train") if str(TRAIN_DIR) in sys.path else None
if train is None:
    sys.path.insert(0, str(TRAIN_DIR))
    train = pytest.importorskip("train")


def _dataset(tmp: str) -> str:
    root = os.path.join(tmp, "ds")
    os.makedirs(os.path.join(root, "labels", "train"))
    os.makedirs(os.path.join(root, "images", "train"))
    with open(os.path.join(root, "labels", "train", "a.txt"), "w") as f:
        f.write("0 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.1 0.1\n2 0.3 0.3 0.1 0.1\n")
    with open(os.path.join(root, "labels", "train", "b.txt"), "w") as f:
        f.write("3 0.5 0.5 0.1 0.1\n")
    with open(os.path.join(root, "train.txt"), "w") as f:
        f.write("./images/train/a.jpg\n./images/train/b.jpg\n")   # a path list, not a label
    with open(os.path.join(root, "data.yaml"), "w") as f:
        f.write("names:\n  0: helmet\ntrain: images/train\nval: images/train\n")
    return root


def test_the_trainer_drops_only_undeclared_boxes_and_keeps_the_image():
    with tempfile.TemporaryDirectory() as tmp:
        root = _dataset(tmp)
        got = train.drop_undeclared_class_boxes(root, os.path.join(root, "data.yaml"))
        assert got == {"files_seen": 2, "files_changed": 2, "dropped": {2: 2, 3: 1}}
        a = open(os.path.join(root, "labels", "train", "a.txt")).read()
        assert a == "0 0.5 0.5 0.1 0.1\n"
        # No boxes left -> an empty label file -> a background image, kept.
        assert open(os.path.join(root, "labels", "train", "b.txt")).read() == ""


def test_the_trainer_leaves_a_path_list_alone():
    with tempfile.TemporaryDirectory() as tmp:
        root = _dataset(tmp)
        train.drop_undeclared_class_boxes(root, os.path.join(root, "data.yaml"))
        assert open(os.path.join(root, "train.txt")).read().startswith("./images/train/a.jpg")


def test_the_trainer_flag_is_wired_like_single_cls():
    source = (TRAIN_DIR / "train.py").read_text(encoding="utf-8")
    assert '"--drop-undeclared-classes"' in source
    assert 'os.environ.get("DROP_UNDECLARED_CLASSES", "")' in source
    assert "args.drop_undeclared_classes = (" in source
    assert "drop_undeclared=args.drop_undeclared_classes)" in source
    assert "LABEL_FILTER:" in source


def test_the_trainer_filters_after_the_download_and_before_training():
    source = (TRAIN_DIR / "train.py").read_text(encoding="utf-8")
    assert source.index("data_yaml = download_dataset_with_progress(")         < source.index("data_yaml, _filtered = build_filtered_dataset(")         < source.index("results = model.train(")


def test_the_trainer_never_uploads():
    """The docstring SAYS nothing is uploaded; this checks the code agrees
    -- no S3 call of any kind inside the function."""
    code = inspect.getsource(train.drop_undeclared_class_boxes)
    assert "upload_file(" not in code and "put_object(" not in code
    assert "s3." not in code and "boto3" not in code


# --- the evaluator ------------------------------------------------------------------

def test_the_evaluator_applies_the_same_rule_from_its_environment():
    assert "def drop_undeclared_class_boxes(data, declared_names):" in EVALUATOR
    assert 'os.environ.get("DROP_UNDECLARED_CLASSES")' in EVALUATOR
    assert EVALUATOR.index("DROP_UNDECLARED_CLASSES") < EVALUATOR.index("candidate = detection(")


def test_the_evaluator_only_touches_label_files():
    assert 'if "labels" not in path.relative_to(data).parts:' in EVALUATOR


# --- the pipeline definition ---------------------------------------------------------

def test_the_pipeline_declares_the_parameter_with_a_safe_default():
    params = {p["Name"]: p for p in PIPELINE["Parameters"]}
    assert params["DropUndeclaredClasses"]["DefaultValue"] == "false"


def test_it_reaches_both_containers():
    steps = {s["Name"]: s["Arguments"] for s in PIPELINE["Steps"]}
    assert steps["TrainYolo"]["HyperParameters"]["drop-undeclared-classes"] == {
        "Get": "Parameters.DropUndeclaredClasses"}
    assert steps["EvaluateYolo"]["Environment"]["DROP_UNDECLARED_CLASSES"] == {
        "Get": "Parameters.DropUndeclaredClasses"}


# --- the card -------------------------------------------------------------------------

def test_the_warning_sits_in_the_preview_with_both_answers():
    assert "Labels use classes the yaml does not name." in FLAT
    assert "Name the rest as class_" in FLAT
    assert "Filter undeclared classes in the training container" in FLAT
    assert "(label files in S3 unchanged)" in FLAT


def test_the_filter_checkbox_is_always_visible_not_only_on_a_mismatch():
    """It is a saved setting. One that only renders after a preview on a
    dataset that happens to need it cannot be found on a dataset that
    does not -- which is exactly how it was first shipped."""
    # The checkbox sits under Label classes, outside the `undeclared`
    # block, and the warning points at it rather than carrying a second
    # copy -- two controls for one setting is the trap this card has.
    field = FLAT.index('placeholder="0 helmet, 1 gloves, 2 vest, 3 boots"')
    warn = FLAT.index("{undeclared.length > 0 && (")
    tick = FLAT.index("onChange={(e) => setDropUndeclared(e.target.checked)}")
    assert field < tick
    assert FLAT.count('checked={dropUndeclared}') == 1, "one checkbox, not one per place"
    assert "under Label classes above." in FLAT
    assert warn < FLAT.index("under Label classes above.")


def test_it_says_what_ultralytics_does():
    assert "discards the <em>whole image</em> for any box it has no name for" in FLAT


def test_name_the_rest_fills_placeholders_without_touching_real_names():
    assert "function nameTheRest(indices: number[])" in FLAT
    assert "merged[String(i)] = `class_${i}`" in FLAT
    assert "{ ...modelClasses(), ...parseLabelClasses(labelClassesText) }" in FLAT


def test_the_tick_is_saved_restored_and_sent():
    assert PANEL.count("drop_undeclared_labels: dropUndeclared,") == 1
    assert "setDropUndeclared(saved.drop_undeclared_labels)" in FLAT
    # Sent with the live preview; the run reads the saved tick itself.
    assert "previewTrack2DatasetYaml(f, classesOf(), true, dropUndeclared," in FLAT


def test_the_client_declares_the_new_shapes():
    assert "undeclared?: { entry: string; indices: number[]; sampled: number; total: number; text: string }[];" in API
    assert "drop_undeclared_labels?: boolean;" in API
    assert "dropUndeclared = false," in API


# --- the filter and Label classes are alternatives, not a layering -----------------

def test_with_the_filter_on_the_yaml_declares_the_model_names_classes(monkeypatch):
    """Six names typed AND the filter ticked gave a six-name yaml with
    nothing left to filter, and the tick looked like it did nothing.
    "Why does the yaml show six when I asked for helmet" was the right
    question."""
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {
        # Assemble mode: the folders are used only when Pull Control is off.
        "only": True,
        "detector_slug": "helmet",
        "train_sources": [{"images": "x/images/train", "labels": ""}],
        "label_classes": {"0": "helmet", "1": "gloves", "2": "person"},
        "drop_undeclared_labels": True,
    }}
    assert rs.assemble_plan(cfg)["classes"] == {0: "helmet"}


def test_with_the_filter_off_label_classes_still_win(monkeypatch):
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {
        # Assemble mode: the folders are used only when Pull Control is off.
        "only": True,
        "detector_slug": "helmet",
        "train_sources": [{"images": "x/images/train", "labels": ""}],
        "label_classes": {"0": "helmet", "1": "gloves", "2": "person"},
        "drop_undeclared_labels": False,
    }}
    assert rs.assemble_plan(cfg)["classes"] == {0: "helmet", 1: "gloves", 2: "person"}


def test_the_pull_golden_path_applies_the_same_rule():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "if drop_undeclared: # Same rule as the assembled path" in code
    assert "if not names and drop_undeclared:" in code


def test_the_card_sends_the_model_names_classes_when_the_filter_is_on():
    assert "if (dropUndeclared) return modelClasses();" in FLAT
    assert "Label classes is not used while this is ticked." in FLAT
    assert "Filter on: the yaml declares only" in FLAT
