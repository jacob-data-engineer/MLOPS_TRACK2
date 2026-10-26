"""A composed run hands the evaluation step ONE prefix.

Two composed runs on 2026-10-19 trained for 35 and 66 minutes on exactly
the right data -- six names, classes=0, single_cls -- and then failed in
two seconds at CreateProcessingJob:

    No S3 objects found under S3 URL
    "s3://track2-s3/golden_set/,s3://track2-s3/Testing_Dataset/..."

The trainer reads that comma-separated list; a SageMaker ProcessingInput
does not. The run was handed no evaluation folder, so the evaluation URI
fell back to the training URI. Nothing was registered; the 20-epoch
model sat unregistered in training-output/.

Three things, then: the many-roots builder writes an evaluation-side
yaml at the val folders' own prefix when they share one; the chain
passes the evaluation folder it already computed for the freeze on to
the run; and a composed run with no evaluation folder is refused before
the GPU starts, with the reason.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy      # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs  # noqa: E402
from ppe_pipeline_api import track2_shared as t2     # noqa: E402


@pytest.fixture
def quiet_s3(monkeypatch):
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    monkeypatch.setattr(dy, "_download_cost", lambda base, dirs, bucket: {
        "prefix": base, "in_named_folders": 0, "in_named_bytes": 0,
        "extra_objects": 0, "extra_bytes": 0})


NAMES = {0: "helmet", 1: "gloves"}


# --- the builder ------------------------------------------------------------------

def test_a_composed_build_offers_the_val_folders_prefix_as_the_evaluation_set(quiet_s3):
    out = dy._many_roots_document(
        train_dirs=["Testing_Dataset/ds/images/train", "golden_set/images/train"],
        val_dirs=["golden_set/images/val"],
        bases=["Testing_Dataset/ds", "golden_set"],
        names=NAMES, bucket="b", overwrite=False)
    assert out["many_roots"] is True
    assert out["eval"] is not None
    assert out["eval"]["folder"] == "golden_set"
    assert out["eval"]["key"] == "golden_set/data.yaml"


def test_the_evaluation_yaml_names_the_val_folders_relative_to_its_own_prefix(quiet_s3):
    out = dy._many_roots_document(
        train_dirs=["Testing_Dataset/ds/images/train", "golden_set/images/train"],
        val_dirs=["golden_set/images/val"],
        bases=["Testing_Dataset/ds", "golden_set"],
        names=NAMES, bucket="b", overwrite=False)
    assert out["eval"]["val"] == ["images/val"]
    assert "val:\n- images/val" in out["eval"]["text"]
    assert "0: helmet" in out["eval"]["text"]


def test_val_folders_under_several_roots_are_all_evaluated(quiet_s3):
    """Golden val AND the dataset's own val, both scored (decided
    2026-10-20). The evaluation step still gets ONE prefix -- the golden
    set's -- and fetches the other val folder itself, images and labels
    only, never the whole dataset around it."""
    out = dy._many_roots_document(
        train_dirs=["a/images/train"],
        val_dirs=["golden_set/images/val", "Testing_Dataset/ds/images/val"],
        bases=["a", "golden_set", "Testing_Dataset/ds"],
        names=NAMES, bucket="b", overwrite=False)
    assert out["eval"]["folder"] == "golden_set"
    assert out["eval"]["val"] == ["images/val", "Testing_Dataset/ds/images/val"]
    assert out["eval"]["extra_uris"] == ["s3://b/Testing_Dataset/ds/images/val/",
                                         "s3://b/Testing_Dataset/ds/labels/val/"]


def test_no_val_folders_offers_no_evaluation_set(quiet_s3):
    out = dy._many_roots_document(
        train_dirs=["a/images/train", "golden_set/images/train"], val_dirs=[],
        bases=["a", "golden_set"], names=NAMES, bucket="b", overwrite=False)
    assert out["eval"] is None


def test_a_hand_written_yaml_at_the_evaluation_prefix_is_protected(monkeypatch):
    """The second file is written too, so the same overwrite rule has to
    cover it."""
    monkeypatch.setattr(dy, "_download_cost", lambda base, dirs, bucket: {})
    monkeypatch.setattr(dy, "_existing_yaml",
                        lambda b, k: {"bytes": 10, "ours": False} if k == "golden_set/data.yaml" else {})
    out = dy._many_roots_document(
        train_dirs=["a/images/train"], val_dirs=["golden_set/images/val"],
        bases=["a", "golden_set"], names=NAMES, bucket="b", overwrite=False)
    assert out["needs_overwrite"] is True


def test_write_in_place_writes_the_evaluation_yaml_too():
    """It already loops over `(out, out.get("eval"))`; a many-roots build
    now has something in the second slot."""
    code = inspect.getsource(dy.write_in_place)
    assert 'for side in (out, out.get("eval")):' in code


def test_the_reason_is_written_down_where_the_eval_side_is_built():
    code = inspect.getsource(dy._many_roots_document)
    assert "THE EVALUATION STEP TAKES ONE PREFIX" in code
    assert "CreateProcessingJob" in code


# --- the chain ----------------------------------------------------------------------

def test_the_queue_entry_carries_the_evaluation_folder_the_chain_computed():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert '"eval_dataset_folder": eval_root,' in code
    assert '"eval_dataset_folder": "",' not in code


def test_the_direct_start_does_too():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "eval_dataset_folder=eval_root," in code


def test_both_composed_branches_read_it_from_the_build():
    code = inspect.getsource(rs._run_chain_body)
    assert code.count('eval_root = (out.get("eval") or {}).get("folder") or ""') == 2


# --- the run -------------------------------------------------------------------------

def test_a_composed_run_without_an_evaluation_folder_is_refused_before_training():
    code = " ".join(inspect.getsource(t2.start_custom_training_run).split())
    assert 'if "," in train_uri and not eval_dataset_folder:' in code
    # The message is two adjacent literals; this phrase sits inside one.
    assert "evaluation step can read only one" in code
    # Before the pipeline is started, not after.
    assert code.index('if "," in train_uri and not eval_dataset_folder:') \
        < code.index("sm.start_pipeline_execution(")


def test_a_single_prefix_run_without_an_evaluation_folder_is_still_fine():
    """Every run before composition existed: the evaluation URI falls
    back to the one training prefix, which the step can read."""
    code = " ".join(inspect.getsource(t2.start_custom_training_run).split())
    # A folder in another bucket now travels as a full s3:// URI; the
    # home-bucket and fall-back-to-train forms are unchanged.
    assert 'else f"s3://{BUCKET}/{eval_dataset_folder.strip(\'/\')}/" if eval_dataset_folder else train_uri)' in code


def test_no_yaml_this_module_writes_uses_anchors(quiet_s3):
    """`train: &id001` / `val: *id001` read as validation BEING training."""
    out = dy._many_roots_document(
        train_dirs=["a/images/train"],
        val_dirs=["golden_set/images/val", "Testing_Dataset/ds/images/val"],
        bases=["a", "golden_set", "Testing_Dataset/ds"],
        names=NAMES, bucket="b", overwrite=False)
    for text in (out["text"], out["eval"]["text"]):
        assert "&id" not in text and "*id" not in text


def test_the_evaluation_yaml_says_it_is_evaluation_only(quiet_s3):
    out = dy._many_roots_document(
        train_dirs=["a/images/train"],
        val_dirs=["golden_set/images/val", "Testing_Dataset/ds/images/val"],
        bases=["a", "golden_set", "Testing_Dataset/ds"],
        names=NAMES, bucket="b", overwrite=False)
    assert "EVALUATION ONLY -- nothing trains from this file." in out["eval"]["text"]
    assert "replaced by the evaluator's local download directory" in out["eval"]["text"]


def test_the_queue_card_is_told_a_composed_runs_real_folders():
    """The dataset folder of a composed run is `_generated/<hash>` -- the
    yaml's home, which holds no image. The card showed that as "Training
    images", and the golden set and the dataset looked missing."""
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert 'composed_train, composed_val = list(out["train"]), list(out["val"])' in code
    # Since 2026-10-26 the folders come from the run's dataset version,
    # which resolved every image -- still the real folders, never the
    # yaml's home (now `_datasets/<id>`).
    assert 'entry.update({k: v for k, v in fields.items() if k.startswith(("dataset_source",' in code
    q = " ".join(inspect.getsource(t2.queue_add).split())
    assert 'if train_folder and not entry.get("dataset_source_folders"):' in q
