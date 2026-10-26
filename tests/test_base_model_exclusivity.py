"""Exactly one base model source reaches the server.

Originally written for a bug: "Add to queue failed: Pick exactly one base
model source ... (got 2: scratch, package)" on every submission with
"train from scratch" ticked. The backend check was right; the card sent
both, because the version chosen in step 1 is restored into the form and
KEPT there, so ticking scratch left the package ARN set alongside it.

**The scratch card has since been removed from the page** -- a
from-scratch run needs far more data and many more epochs than
fine-tuning, and takes correspondingly longer on the one GPU slot
everything queues behind. So the card can no longer send two sources,
and the first test now pins that rather than the old conditional.

The SERVER checks are unchanged and still tested. They are the ones that
matter: the queue replays entries that carry `base_model_scratch`, the
retrain chain builds its own arguments, and the route is reachable
without the page.
"""
from __future__ import annotations

import importlib
import pathlib
import re

import pytest

PAGE = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "routes" / "Track2ModelsPage.tsx")
t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")


def test_the_card_sends_exactly_one_source_and_it_is_never_scratch():
    """With the scratch card gone there is one source to send, so the
    conditional that used to clear the ARN has nothing to clear. What has
    to stay true is that the page sends `base_model_scratch` as a literal
    false rather than dropping the field: the backend's "exactly one
    source" check reads it, and the queue replays entries carrying it."""
    source = PAGE.read_text(encoding="utf-8")
    assert "base_model_package_arn: basePackageArn," in source
    assert "base_model_scratch: false," in source
    # No mode state left to disagree with it.
    assert "modelSource" not in re.sub(r"//.*", "", source)
    assert "Track2ScratchTrainingPanel" not in source


def test_the_server_still_refuses_two_sources():
    """The card must not become the only thing preventing it: the queue
    replays entries, and the chain builds its own."""
    with pytest.raises(ValueError, match="exactly one"):
        t2.resolve_base_model_source(
            scratch=True, scratch_architecture="yolo26s",
            base_model_package_arn="arn:aws:sagemaker:::model-package/g/1",
        )


def test_the_server_still_refuses_none():
    with pytest.raises(ValueError, match="got none"):
        t2.resolve_base_model_source()


def test_scratch_alone_resolves_to_the_seed(monkeypatch):
    monkeypatch.setattr(t2, "BUCKET", "track2-s3")
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    uri, source = t2.resolve_base_model_source(scratch=True, scratch_architecture="yolo26s")
    assert source == "scratch"
    assert uri.endswith("external_models/scratch/yolo26s/model.pt")


def test_a_package_alone_still_works(monkeypatch):
    monkeypatch.setattr(t2, "resolve_package_model_artifact", lambda arn: "s3://b/model.tar.gz")
    uri, source = t2.resolve_base_model_source(
        base_model_package_arn="arn:aws:sagemaker:::model-package/g/1")
    assert source == "package"


def test_the_refusal_renders_last_in_the_form():
    """Immediately above the Training queue -- where a refused add is
    looked for, and where the entry would have appeared had it worked.
    At the top of the form it was several cards up and invisible."""
    source = PAGE.read_text(encoding="utf-8")
    banner = source.index("{error && (")
    # The LAST error banner in the form closes the fragment.
    tail = source[banner:]
    assert re.search(r"\{error && \(.*?\)\}\s*\n\s*</>", tail, re.S), \
        "the banner must be the last thing before the fragment closes"


def test_a_scratch_run_for_every_generation_has_a_seed_name():
    """yolo26 included -- the architectures come from GENERATION_SIZES,
    so a new generation gets a seed path without any code change."""
    assert "yolo26s" in t2.KNOWN_ARCHITECTURES
    assert "yolo11s" in t2.KNOWN_ARCHITECTURES


def test_the_server_can_still_train_from_scratch(monkeypatch):
    """Removed from the page, not from the platform. The random-init
    seeds are real objects in S3, the route is reachable, and the rule
    that keeps a v4 continuing v3 is a registry fact rather than a
    property of a card."""
    # Stubbed rather than called: the seed lookup is a real head_object,
    # and a test suite that reaches S3 fails for whoever has no
    # credentials rather than for the reason it was written.
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    uri, source = t2.resolve_base_model_source(
        scratch=True, scratch_architecture="yolo11s")
    assert source == "scratch"
    assert uri.endswith("external_models/scratch/yolo11s/model.pt")
    assert callable(t2.scratch_refusal)
