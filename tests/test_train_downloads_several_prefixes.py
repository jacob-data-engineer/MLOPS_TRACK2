"""`train.py` can fetch a training set from more than one prefix.

The yaml could always name folders anywhere -- Ultralytics takes a
directory, a list of directories, or a .txt of image paths, and checks
`is_dir()`/`is_file()` PER ENTRY, so a mixed list works. What pinned a
run to one place was this repo's own downloader: one prefix, everything
under it, prefix stripped.

So combining sources that live under unrelated roots was refused by our
code, not by Ultralytics:

    assemble only              -> Testing_Dataset/object_detection_dataset
    assemble + golden (train)  -> REFUSED: no parent -- 'Testing_Dataset',
                                  'golden_set'
    all three                  -> REFUSED: ... 'retraining'

TWO MODES, and the first must not change. Every data.yaml this system
has ever written says `images/train`, which only resolves if the single
prefix is stripped. With several, stripping would collide unrelated
trees into one, so each key keeps its full path and the yaml names it
the same way.

The yaml itself then has no single root to be found at, so it is named
explicitly -- `--data-yaml-s3-uri`.

NOT DEPLOYED BY deploy.sh. This file lives in the track2-yolo-train
image and needs its own `track2-yolo-train-images` CodeBuild, plus the
pipeline repointed at the new digest.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile

import pytest

TRAIN_DIR = (pathlib.Path(__file__).resolve().parents[1]
             / "model_training" / "training-container")
sys.path.insert(0, str(TRAIN_DIR))

train = pytest.importorskip("train")


OBJECTS = {
    "Testing_Dataset/ds/": ["Testing_Dataset/ds/data.yaml",
                            "Testing_Dataset/ds/images/train/a.jpg",
                            "Testing_Dataset/ds/labels/train/a.txt"],
    "golden_set/": ["golden_set/images/val/g.jpg",
                    "golden_set/labels/val/g.txt"],
}


class _Paginator:
    def paginate(self, Bucket, Prefix):
        yield {"Contents": [{"Key": k, "Size": 10} for k in OBJECTS.get(Prefix, [])]}


class _S3:
    def get_paginator(self, _):
        return _Paginator()

    def download_file(self, bucket, key, dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with open(dest, "w", encoding="utf-8") as handle:
            handle.write("names:\n  0: helmet\ntrain: images/train\nval: images/train\n")


@pytest.fixture
def bucket(monkeypatch):
    monkeypatch.setattr(train.boto3, "client", lambda *a, **k: _S3())
    # The 20s wait exists so CloudWatch's metric filter can attach before
    # the progress lines are printed. Nothing here reads those metrics.
    monkeypatch.setattr(train.time, "sleep", lambda n: None)


def _fetch(uri: str, yaml_uri: str = "") -> list:
    with tempfile.TemporaryDirectory() as into:
        train.download_dataset_with_progress(uri, "ap-southeast-1", into,
                                             data_yaml_s3_uri=yaml_uri)
        return sorted(
            os.path.relpath(os.path.join(root, name), into).replace("\\", "/")
            for root, _, names in os.walk(into) for name in names
        )


# --- parsing ---------------------------------------------------------------------

def test_one_uri_is_a_list_of_one():
    assert train._split_uris("s3://b/a/") == ["s3://b/a/"]


def test_several_are_comma_separated_and_trimmed():
    assert train._split_uris(" s3://b/a/ , s3://b/c/ ") == ["s3://b/a/", "s3://b/c/"]


def test_nothing_is_an_empty_list():
    assert train._split_uris("") == []
    assert train._split_uris(None) == []


def test_an_empty_uri_is_refused_rather_than_downloading_the_bucket(bucket):
    with pytest.raises(ValueError, match="is empty"):
        _fetch("")


# --- one prefix: unchanged -----------------------------------------------------------

def test_one_prefix_still_strips_it(bucket):
    """Every data.yaml this system has written says `images/train`."""
    assert _fetch("s3://b/Testing_Dataset/ds/") == [
        "data.yaml", "images/train/a.jpg", "labels/train/a.txt",
    ]


def test_one_prefix_still_finds_the_yaml_in_the_download(bucket):
    assert "data.yaml" in _fetch("s3://b/Testing_Dataset/ds/")


# --- several prefixes ------------------------------------------------------------------

def test_several_keep_their_full_paths(bucket):
    """Stripping would collide `Testing_Dataset/ds/images` and
    `golden_set/images` into one tree."""
    got = _fetch("s3://b/Testing_Dataset/ds/,s3://b/golden_set/",
                 yaml_uri="s3://b/runs/x/data.yaml")
    assert "Testing_Dataset/ds/images/train/a.jpg" in got
    assert "golden_set/images/val/g.jpg" in got


def test_nothing_from_either_source_is_lost(bucket):
    got = _fetch("s3://b/Testing_Dataset/ds/,s3://b/golden_set/",
                 yaml_uri="s3://b/runs/x/data.yaml")
    # Five source objects plus the run's own data.yaml at the root.
    assert len(got) == 6


def test_the_yaml_is_taken_from_its_own_uri(bucket):
    got = _fetch("s3://b/Testing_Dataset/ds/,s3://b/golden_set/",
                 yaml_uri="s3://b/runs/x/data.yaml")
    assert "data.yaml" in got


def test_several_prefixes_without_a_yaml_uri_say_what_is_missing(bucket):
    """There is no single dataset root to find one at, and a run that
    guessed would train on whichever source happened to carry one."""
    with pytest.raises(FileNotFoundError, match="data-yaml-s3-uri"):
        _fetch("s3://b/Testing_Dataset/ds/,s3://b/golden_set/")


def test_a_prefix_with_nothing_under_it_names_every_uri_tried(bucket):
    with pytest.raises(FileNotFoundError, match="s3://b/nowhere/"):
        _fetch("s3://b/nowhere/")


# --- the argument is wired ---------------------------------------------------------------

def test_the_flag_exists_and_reads_the_environment():
    source = (TRAIN_DIR / "train.py").read_text(encoding="utf-8")
    assert '"--data-yaml-s3-uri"' in source
    assert 'os.environ.get("DATA_YAML_S3_URI", "")' in source


def test_the_call_site_passes_it():
    source = (TRAIN_DIR / "train.py").read_text(encoding="utf-8")
    assert "data_yaml_s3_uri=args.data_yaml_s3_uri" in source


def test_the_manifest_filter_still_matches_on_the_local_path():
    """A pinned run lists `images/train/a.jpg`, not the full key."""
    source = (TRAIN_DIR / "train.py").read_text(encoding="utf-8")
    assert 'present = {obj["Local"] for obj in objects}' in source


# --- forward compatibility -------------------------------------------------------------

def test_an_unknown_hyperparameter_does_not_kill_the_run(monkeypatch):
    """SageMaker turns every pipeline hyperparameter into `--key value`,
    so `parse_args` made this image refuse to start the moment the
    pipeline learned a parameter it did not know.

    That was an ordering trap with no safe direction: add the parameter
    first and every run dies at startup on the old image; ship the image
    first and the app cannot send it yet.
    """
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--train-data-s3-uri", "s3://b/a/",
        "--base-model-s3-uri", "s3://b/w.pt",
        "--a-parameter-from-the-future", "7",
    ])
    args = train.parse_args()
    assert args.train_data_s3_uri == "s3://b/a/"


def test_it_says_which_arguments_it_ignored(capsys, monkeypatch):
    """This also swallows a misspelled flag. A run that trained on the
    default because somebody typed `--img-sizes` deserves a line in the
    log saying so."""
    monkeypatch.setattr(sys, "argv", [
        "train.py",
        "--train-data-s3-uri", "s3://b/a/",
        "--base-model-s3-uri", "s3://b/w.pt",
        "--img-sizes", "640",
    ])
    train.parse_args()
    assert "--img-sizes" in capsys.readouterr().out


def test_the_known_ones_are_still_parsed_and_validated(monkeypatch):
    """Ignoring the unknown must not loosen the real checks."""
    monkeypatch.setattr(sys, "argv", [
        "train.py", "--train-data-s3-uri", "s3://b/a/", "--unknown", "x",
    ])
    with pytest.raises(SystemExit):
        # Exactly one of the two base-model flags is required.
        train.parse_args()
