"""A Class that already has a version may not be trained from scratch.

A registered Class has a LINEAGE: v3 continues v2 continues v1, and that
is the whole meaning of the number. Training it from random weights
registers a v4 that continues nothing -- same group, same name, unrelated
model -- and every later comparison between v3 and v4 compares two
different things, with nothing in the registry to show it.

Scratch is for a Class's FIRST model. To restart deliberately, make a new
Class: it costs nothing and keeps both histories readable.
"""
from __future__ import annotations

import importlib

import pytest

t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")


@pytest.fixture
def registry(monkeypatch):
    def use(packages, group="yolo11s-helmet-detector"):
        monkeypatch.setattr(t2, "get_detector", lambda slug: {"slug": slug, "classes": {}})
        monkeypatch.setattr(t2, "detector_group_name", lambda detector, arch: group)
        monkeypatch.setattr(t2, "model_registry_summary",
                            lambda g: {"packages": packages})
    return use


def test_a_class_with_versions_is_refused(registry):
    registry([{"ModelPackageArn": "arn:.../yolo11s-helmet-detector/9"},
              {"ModelPackageArn": "arn:.../yolo11s-helmet-detector/8"}])
    reason = t2.scratch_refusal("helmet", "yolo11s")
    assert "not permitted" in reason
    assert "newest v9" in reason
    assert "2 registered versions" in reason


def test_one_version_reads_as_singular(registry):
    registry([{"ModelPackageArn": "arn:.../g/1"}])
    assert "1 registered version (" in t2.scratch_refusal("helmet", "yolo11s")


def test_an_empty_class_is_allowed(registry):
    """The case scratch exists for -- a Class's first model."""
    registry([])
    assert t2.scratch_refusal("helmet", "yolo26s") == ""


def test_no_detector_is_allowed(registry):
    """The open groups, as before detectors existed."""
    registry([{"ModelPackageArn": "arn:.../g/1"}])
    assert t2.scratch_refusal("", "yolo11s") == ""


def test_a_registry_that_cannot_be_read_does_not_refuse(monkeypatch):
    """Refusing over a listing failure would make an AWS hiccup look like
    a policy, and the message would name a rule that did not apply."""
    def boom(slug):
        raise RuntimeError("throttled")
    monkeypatch.setattr(t2, "get_detector", boom)
    assert t2.scratch_refusal("helmet", "yolo11s") == ""


def test_the_submit_path_enforces_it():
    """Every custom run passes through here -- the card, the queue
    replaying an entry queued before a version existed, and the chain."""
    import inspect
    source = inspect.getsource(t2.start_custom_training_run)
    assert "scratch_refusal(detector_slug, architecture)" in source
    # The CODE, not the docstring -- which names `class_filter_for` while
    # explaining why the dataset gets narrowed.
    body = source.split('"""', 2)[-1]
    assert body.index("scratch_refusal") < body.index("class_filter_for("),         "the refusal must come before any work is done"


def test_queue_add_refuses_at_add_time():
    """Not at run time: a queued entry can sit for hours, and refusing it
    then means the queue skips it with nothing on screen saying why."""
    import inspect
    models = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
    source = inspect.getsource(models.add_to_queue)
    assert "scratch_refusal" in source
    assert source.index("scratch_refusal") < source.index("t2.queue_add(entry)")


def test_the_chain_stops_on_a_step_that_says_why():
    """Rather than failing inside the run, where the reason is a
    traceback in a training log."""
    import inspect
    rs = importlib.import_module("src.ppe_pipeline_api.retrain_schedule")
    source = inspect.getsource(rs._run_chain_body)
    assert "scratch_refusal" in source
    assert 'record("base", False' in source


def test_scratch_still_works_for_a_brand_new_class(registry, monkeypatch):
    """The rule must not close the one path a new Class has."""
    registry([])
    monkeypatch.setattr(t2, "BUCKET", "track2-s3")
    monkeypatch.setattr(t2, "_s3_object_exists", lambda key: True)
    uri, source = t2.resolve_base_model_source(scratch=True, scratch_architecture="yolo26s")
    assert source == "scratch"
    assert t2.scratch_refusal("helmet", "yolo26s") == ""


def test_the_card_asks_the_server_rather_than_the_registries_it_has():
    """The page loads the four PRODUCTION groups --
    yolo11{s,m,l,x}-detector -- and a Class registers into
    `yolo11m-helmet-detector`, which is not among them. Looking the group
    up in that list found nothing and offered scratch for exactly the
    Classes the rule protects, while the server refused the submission.

    One endpoint, returning the server's own sentence, so what the card
    says is what the run would be refused with."""
    import inspect
    models = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
    source = inspect.getsource(models.scratch_check)
    assert "t2.scratch_refusal(detector_slug, architecture)" in source


def test_the_brief_the_page_loads_is_production_only_by_default():
    """The fact the bug rested on: `serialize_registries_brief` takes
    `include_open=False`, so the detector groups a Class registers into
    are opt-in and the page does not have them. Pinned so a future change
    does not silently re-enable the lookup that failed."""
    import inspect
    signature = inspect.signature(t2.serialize_registries_brief)
    assert signature.parameters["include_open"].default is False


def test_the_message_names_the_remedy(registry):
    """A refusal that does not say what to do instead sends somebody to
    the logs, or to guessing."""
    registry([{"ModelPackageArn": "arn:.../yolo11m-helmet-detector/1"}],
             group="yolo11m-helmet-detector")
    reason = t2.scratch_refusal("helmet", "yolo11m")
    assert "not permitted" in reason
    assert "Delete the existing version" in reason
    assert "another Class, model or size" in reason
    # And it names the group, so "which one?" is answered too.
    assert "yolo11m-helmet-detector" in reason


def test_each_size_is_its_own_registry_group(registry):
    """`yolo11s-helmet-detector` having twelve versions says nothing
    about `yolo11m-helmet-detector`. The check is re-asked on the model
    and the size, not only the Class."""
    registry([], group="yolo11m-helmet-detector")
    assert t2.scratch_refusal("helmet", "yolo11m") == ""
