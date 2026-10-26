"""The data.yaml must declare every class the label files use.

What happened on 2026-10-19 at 15:34 SGT, the first composed run:

    Label class 3 exceeds dataset class count 1. Possible class labels are 0-0
    RuntimeError: No valid images found in .../golden_set/labels/val.cache

The yaml declared `{0: helmet}` -- the Model Name's list -- and the
label files on BOTH sources used 0, 1, 2 and 3. Ultralytics marks the
whole image corrupt when any box is outside the declared range: on the
validation side that was all 20 images and a refusal to start, thirty
minutes in, after the download. On the training side it does not even
error -- the images are silently dropped -- which is what every earlier
run on this dataset had been doing.

Two things, then. The yaml declares what the LABELS mean (`label_classes`
on the Assemble card), so the Model Name's one class becomes a load-time
filter and every image is kept. And the label files are READ before any
GPU time, so a mismatch is a sentence naming the folder and the fix.
"""
from __future__ import annotations

import inspect
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import dataset_yaml as dy        # noqa: E402
from ppe_pipeline_api import pull_control as pc        # noqa: E402
from ppe_pipeline_api import retrain_schedule as rs    # noqa: E402

WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
PANEL = (WEB / "components" / "Track2DatasetAssemblePanel.tsx").read_text(encoding="utf-8")
FLAT = " ".join(PANEL.split())


# --- reading the label files -------------------------------------------------------

class _Body:
    def __init__(self, text: str):
        self._t = text.encode("utf-8")

    def read(self):
        return self._t


class _S3:
    def __init__(self, files: dict):
        self.files = files

    def get_object(self, Bucket, Key):
        return {"Body": _Body(self.files[Key])}


@pytest.fixture
def labels(monkeypatch):
    files = {
        "ds/labels/train/a.txt": "0 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.1 0.1\n",
        "ds/labels/train/b.txt": "3 0.5 0.5 0.1 0.1\n",
        "ds/labels/train/c.txt": "",
        "ds/images/train/a.jpg": "",          # not a label, must be ignored
        "ds/train.txt": "./images/train/a.jpg\n",  # a path list, not a label
    }
    monkeypatch.setattr(pc, "s3", _S3(files))
    monkeypatch.setattr(pc, "_list_objects",
                        lambda prefix, bucket="", **kw: [
                            {"Key": k, "Size": 1} for k in files if k.startswith(prefix)])
    monkeypatch.setattr(pc, "destination_bucket", lambda: "b")
    return files


def test_it_reads_the_indices_the_files_actually_use(labels):
    got = dy.label_indices_used("ds/labels/train")
    assert got["indices"] == {0, 2, 3}
    assert got["total"] == 3
    assert got["sampled"] == 3


def test_only_txt_files_under_a_labels_segment_count(labels):
    """An image is not a label, and a path list sitting at the root is
    not one either -- its first column is a filename."""
    got = dy.label_indices_used("ds")
    assert got["indices"] == {0, 2, 3}
    assert got["total"] == 3


def test_an_empty_label_file_contributes_nothing(labels):
    got = dy.label_indices_used("ds/labels/train")
    assert 1 not in got["indices"]


def test_a_sample_is_a_sample(labels, monkeypatch):
    many = {f"big/labels/train/{i}.txt": f"{i % 4} 0.5 0.5 0.1 0.1\n" for i in range(500)}
    monkeypatch.setattr(pc, "s3", _S3(many))
    monkeypatch.setattr(pc, "_list_objects",
                        lambda prefix, bucket="", **kw: [{"Key": k} for k in many])
    got = dy.label_indices_used("big/labels/train", sample=50)
    assert got["sampled"] == 50
    assert got["total"] == 500


def test_none_means_read_them_all(labels, monkeypatch):
    many = {f"big/labels/train/{i}.txt": "0 0.5 0.5 0.1 0.1\n" for i in range(300)}
    monkeypatch.setattr(pc, "s3", _S3(many))
    monkeypatch.setattr(pc, "_list_objects",
                        lambda prefix, bucket="", **kw: [{"Key": k} for k in many])
    assert dy.label_indices_used("big/labels/train", sample=None)["sampled"] == 300


# --- the check in the build ---------------------------------------------------------

def test_labels_beyond_the_names_are_refused_before_anything_is_written(labels, monkeypatch):
    """The PREVIEW reports the mismatch (so the card can offer its two
    answers); the WRITE is what refuses. Nothing reaches S3."""
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    with pytest.raises(ValueError) as caught:
        dy.write_in_place(
            splits={"train": [{"images": "ds/images/train", "labels": ""}], "val": []},
            classes={0: "helmet"}, bucket="b")
    text = str(caught.value)
    assert "'ds/images/train'" in text
    assert "use class 2, 3" in text
    assert "declare only 0: helmet" in text
    assert "Label classes on the Assemble card" in text


def test_the_refusal_says_what_ultralytics_actually_does():
    """The whole image, not the box -- which is why it matters."""
    code = inspect.getsource(dy._labels_beyond)
    assert "the whole image, not the box" in code


def test_names_that_cover_the_labels_pass(labels, monkeypatch):
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    out = dy.build_in_place(
        splits={"train": [{"images": "ds/images/train", "labels": ""}], "val": []},
        classes={0: "helmet", 1: "gloves", 2: "vest", 3: "boots"}, bucket="b")
    assert out["names"] == {0: "helmet", 1: "gloves", 2: "vest", 3: "boots"}


def test_zero_skips_the_scan(labels, monkeypatch):
    monkeypatch.setattr(dy, "_existing_yaml", lambda b, k: {})
    out = dy.build_in_place(
        splits={"train": [{"images": "ds/images/train", "labels": ""}], "val": []},
        classes={0: "helmet"}, bucket="b", label_scan=0)
    assert out["names"] == {0: "helmet"}


def test_a_listing_that_fails_refuses_nothing(monkeypatch):
    """Unknown is not broken."""
    def boom(*a, **k):
        raise RuntimeError("no bucket")
    monkeypatch.setattr(pc, "_list_objects", boom)
    monkeypatch.setattr(pc, "destination_bucket", lambda: "b")
    assert dy._labels_beyond(["ds/images/train"], {0: "helmet"}, "b", 200) == []


def test_write_in_place_passes_the_scan_depth_through():
    code = inspect.getsource(dy.write_in_place)
    assert "label_scan=label_scan" in code


def test_the_check_runs_before_the_yaml_is_shaped():
    """Refusing after `_bases_for` would mean a many-roots document was
    built for folders that cannot be trained together anyway."""
    code = inspect.getsource(dy.build_in_place)
    assert code.index("_labels_beyond(") < code.index("train_bases = ")


# --- the chain reads every file and declares what the labels mean ----------------

def test_the_chain_reads_every_label_file_not_a_sample():
    code = " ".join(inspect.getsource(rs._run_chain_body).split())
    assert "label_scan=None, drop_undeclared=drop_undeclared)" in code


def test_label_classes_become_the_yamls_names(monkeypatch):
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    from ppe_pipeline_api import track2_shared as t2
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {
        # Assemble mode: the folders are used only when Pull Control is off.
        "only": True,
        "detector_slug": "helmet",
        "train_sources": [{"images": "ds/images/train", "labels": "ds/labels/train"}],
        "eval_sources": [{"images": "ds/images/val", "labels": "ds/labels/val"}],
        "label_classes": {"0": "helmet", "1": "gloves", "2": "vest", "3": "boots"},
    }}
    plan = rs.assemble_plan(cfg)
    assert plan["classes"] == {0: "helmet", 1: "gloves", 2: "vest", 3: "boots"}
    assert not [p for p in plan["problems"] if p["level"] == "blocking"]


def test_without_label_classes_the_model_names_list_is_used(monkeypatch):
    """Every configuration written before this existed."""
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    from ppe_pipeline_api import track2_shared as t2
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {
        # Assemble mode: the folders are used only when Pull Control is off.
        "only": True,
        "detector_slug": "helmet",
        "train_sources": [{"images": "ds/images/train", "labels": "ds/labels/train"}],
        "eval_sources": [{"images": "ds/images/val", "labels": "ds/labels/val"}],
    }}
    assert rs.assemble_plan(cfg)["classes"] == {0: "helmet"}


def test_label_classes_missing_the_model_names_class_are_refused(monkeypatch):
    """A list with nothing to filter to would train on nothing, silently."""
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    from ppe_pipeline_api import track2_shared as t2
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {
        # Assemble mode: the folders are used only when Pull Control is off.
        "only": True,
        "detector_slug": "helmet",
        "train_sources": [{"images": "ds/images/train", "labels": "ds/labels/train"}],
        "eval_sources": [{"images": "ds/images/val", "labels": "ds/labels/val"}],
        "label_classes": {"0": "person", "1": "gloves"},
    }}
    plan = rs.assemble_plan(cfg)
    blocking = [p["text"] for p in plan["problems"] if p["level"] == "blocking"]
    assert any("do not include helmet" in t for t in blocking)
    assert plan["classes"] == {0: "helmet"}


def test_junk_keys_in_label_classes_are_dropped_not_fatal(monkeypatch):
    monkeypatch.setattr(rs, "data_source", lambda c: "assembled")
    from ppe_pipeline_api import track2_shared as t2
    monkeypatch.setattr(t2, "get_detector", lambda slug: {"classes": {0: "helmet"}})
    cfg = {"dataset_assemble": {
        # Assemble mode: the folders are used only when Pull Control is off.
        "only": True,
        "detector_slug": "helmet",
        "train_sources": [{"images": "ds/images/train", "labels": "ds/labels/train"}],
        "label_classes": {"0": "helmet", "x": "bad", "2": "  "},
    }}
    assert rs.assemble_plan(cfg)["classes"] == {0: "helmet"}


def test_the_filter_then_does_what_it_was_built_for(monkeypatch):
    """A one-class Model Name present in a wider yaml is `classes=` plus
    `single_cls=` at load time -- every image kept, only helmet trains.
    This is the path that was unreachable while the yaml equalled the
    Model Name's list."""
    from ppe_pipeline_api import track2_shared as t2
    monkeypatch.setattr(t2, "dataset_classes",
                        lambda folder: {0: "helmet", 1: "gloves", 2: "vest", 3: "boots"})
    got = t2.class_filter_for("_generated/x", {"classes": {0: "helmet"}})
    assert got == {"classes": [0], "single_cls": True}


# --- the setting ---------------------------------------------------------------------

def test_the_key_is_documented_in_defaults():
    code = inspect.getsource(pc)
    assert "label_classes" in code
    assert "what the LABEL FILES mean" in code


def test_the_client_type_declares_it():
    api = (WEB / "lib" / "api.ts").read_text(encoding="utf-8")
    assert "label_classes?: Record<string, string>;" in api


# --- the card -----------------------------------------------------------------------

def test_the_card_has_the_field_under_the_model_names_classes():
    assert "Label classes" in FLAT
    # Anchor on the field's own hint: the words "Label classes" also appear
    # earlier, in the preview's warning that points at this field.
    assert FLAT.index("(from your Model Name, in index order)") < FLAT.index("(every class the label files use, in index order")
    assert 'placeholder="0 helmet, 1 gloves, 2 vest, 3 boots"' in FLAT


def test_it_is_saved_in_both_writers_and_restored_on_load():
    assert PANEL.count("label_classes: parseLabelClasses(labelClassesText),") == 1
    assert "setLabelClassesText(formatLabelClasses(saved.label_classes));" in FLAT
    assert "labelClassesText," in FLAT  # in the debounced writer's deps


def test_the_yaml_is_sent_the_labels_taxonomy_when_it_covers_the_model_name():
    assert "return labelledClasses() ?? modelClasses();" in FLAT
    assert "return own.every((n) => have.has(n)) ? typed : null;" in FLAT


def test_the_card_says_what_will_actually_train():
    assert "keeps every image and trains only" in FLAT
    assert "filtered at load time." in FLAT


def test_a_list_missing_the_model_names_class_is_called_out_on_the_card():
    assert "would have nothing to filter to" in FLAT


def test_the_parser_is_forgiving_about_punctuation():
    """Typed once, by hand, from a labelling tool's class list."""
    assert "export function parseLabelClasses" in PANEL
    assert "split(/[,\\n;]+/)" in PANEL
    assert "[:=\\-]?" in PANEL


# --- the scan is parallel, and must stay so --------------------------------------

def test_the_label_files_are_read_in_parallel():
    """Serial, the chain's full scan of this bucket's 8,455 label files
    took 751 seconds -- twelve and a half minutes on every unattended
    run. Sixteen at a time brought the same scan to under a minute."""
    code = inspect.getsource(dy.label_indices_used)
    assert "concurrent.futures.ThreadPoolExecutor(max_workers=16)" in code
    assert "pool.map(_indices_in, keys)" in code


def test_an_unreadable_file_is_counted_not_fatal_in_parallel_too(labels, monkeypatch):
    class _Flaky(_S3):
        def get_object(self, Bucket, Key):
            if Key.endswith("b.txt"):
                raise RuntimeError("gone")
            return super().get_object(Bucket, Key)
    monkeypatch.setattr(pc, "s3", _Flaky(labels))
    got = dy.label_indices_used("ds/labels/train")
    assert got["unreadable"] == 1
    assert got["indices"] == {0, 2}
