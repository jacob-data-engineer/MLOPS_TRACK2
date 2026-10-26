"""Group names must carry the generation they were trained at.

Both of these were single hardcoded `yolo11` strings that stayed correct
for as long as YOLO11 was the only generation. Shipping YOLO26 made one
of them hide models and the other mislabel files, and neither failed --
the run trained, registered Approved, and simply was not there.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ppe_pipeline_api import track2_shared as t2  # noqa: E402


def _old_stem(group_name: str) -> str:
    """The implementation this replaced, kept to prove nothing moved."""
    size = group_name.removeprefix("yolo11").removesuffix("-detector")
    return f"yolo11-{size}"


class TestExportFilename:
    def test_a_yolo26_export_is_not_named_yolo11(self):
        # The bug: removeprefix("yolo11") is a no-op on a yolo26 group, so
        # the size came out as the whole name and the prefix was bolted on
        # anyway -- "yolo11-yolo26s-open".
        assert t2._export_stem("yolo26s-open-detector") == "yolo26-s-open"
        assert "yolo11" not in t2._export_stem("yolo26s-open-detector")

    def test_every_generation_11_name_is_unchanged(self):
        # The whole safety argument for this change: exports already
        # sitting in a folder keep their names, so the archive-on-collision
        # check still finds them.
        for size in t2.REGISTRY_SIZES:
            for group in (
                f"yolo11{size}-detector",
                f"yolo11{size}-open-detector",
                f"yolo11{size}-helmet-detector",
            ):
                assert t2._export_stem(group) == _old_stem(group), group

    def test_the_generation_is_read_not_assumed(self):
        assert t2._export_stem("yolo26m-detector") == "yolo26-m"
        assert t2._export_stem("yolo11m-detector") == "yolo11-m"

    def test_an_unrecognised_group_keeps_its_own_name(self):
        # Rather than being handed a yolo11 prefix it never earned, which
        # is what the old code did to the legacy custom group.
        assert t2._export_stem("track2-yolo-custom-detector") == "track2-yolo-custom"


class TestBuiltinOpenGroups:
    def test_every_known_architecture_has_an_open_group(self):
        groups = {g for _gen, _size, g in t2.builtin_open_groups()}
        for arch in t2.KNOWN_ARCHITECTURES:
            assert f"{arch}-open-detector" in groups, arch

    def test_yolo26_open_groups_are_enumerated(self):
        # The bug: trained, registered, and on no page anywhere.
        groups = {g for _gen, _size, g in t2.builtin_open_groups()}
        assert "yolo26s-open-detector" in groups
        assert "yolo26x-open-detector" in groups

    def test_the_generation_11_groups_are_still_there_and_still_first(self):
        listed = [g for _gen, _size, g in t2.builtin_open_groups()]
        eleven = [g for g in listed if g.startswith("yolo11")]
        assert eleven == [f"yolo11{s}-open-detector" for s in t2.REGISTRY_SIZES]
        assert listed[: len(eleven)] == eleven

    def test_no_open_group_is_a_production_group(self):
        # The four production names are an exact-match list on the rule
        # that fires dump_approved_weights, which overwrites production
        # weights. An open group landing in that list would auto-deploy
        # every unreviewed experiment over production.
        production = {f"yolo11{s}-detector" for s in t2.REGISTRY_SIZES}
        for _gen, _size, group in t2.builtin_open_groups():
            assert group not in production, group

    def test_every_open_group_still_matches_the_iam_grant(self):
        # The grant is model-package/yolo*-detector/* -- a name outside it
        # fails at registration time, in a training job, hours in.
        for _gen, _size, group in t2.builtin_open_groups():
            assert group.startswith("yolo"), group
            assert group.endswith("-detector"), group


class TestParseGroupName:
    def test_a_production_group(self):
        assert t2.parse_group_name("yolo11s-detector") == ("11", "s", "")

    def test_an_open_group_keeps_its_suffix(self):
        assert t2.parse_group_name("yolo26s-open-detector") == ("26", "s", "-open")

    def test_a_named_detectors_group(self):
        assert t2.parse_group_name("yolo26m-vehicle-detector") == ("26", "m", "-vehicle")

    def test_an_unrecognised_name_reports_nothing_rather_than_guessing(self):
        # A caller that gets a generation back must be able to trust it.
        assert t2.parse_group_name("track2-yolo-custom-detector") == (None, None, None)
        assert t2.parse_group_name("something-else") == (None, None, None)

    def test_every_group_this_estate_creates_parses(self):
        for _gen, _size, group in t2.builtin_open_groups():
            assert t2.parse_group_name(group)[0] is not None, group
        for size in t2.REGISTRY_SIZES:
            assert t2.parse_group_name(f"yolo11{size}-detector")[0] == "11"


class TestDeployLabel:
    """The Model tag the Deploy page reads back as "what is serving"."""

    # Calls the function deploy_model calls, not a copy of it -- a test
    # that mirrors the branch it is testing cannot notice the branch
    # changing underneath it.
    _label = staticmethod(t2.registry_source_desc)

    def test_a_yolo11_package_is_still_labelled_yolo11(self):
        arn = "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo11s-detector/7"
        assert self._label(arn, "s") == "YOLO11-S v7"

    def test_a_yolo26_package_is_not_labelled_yolo11(self):
        arn = "arn:aws:sagemaker:ap-southeast-1:1:model-package/yolo26s-open-detector/3"
        assert self._label(arn, "s") == "YOLO26-S v3"

    def test_an_unparseable_group_says_what_it_is_rather_than_inventing_a_generation(self):
        arn = "arn:aws:sagemaker:ap-southeast-1:1:model-package/track2-yolo-custom-detector/2"
        assert self._label(arn, "s") == "track2-yolo-custom-detector v2"


class TestEveryLabelPathDerivesTheGeneration:
    """There were four of these, written independently, all saying 11.

    Behaviour tests cover `registry_source_desc` itself; these pin that the
    callers still go through it. A fifth copy of the f-string is the whole
    failure mode -- CONTINUE.md calls this "one job, not seven" -- and it
    would pass every behavioural test while labelling the queue wrong.
    """

    @staticmethod
    def _source(func) -> str:
        """The function's CODE, with comments stripped.

        Reading the raw source would have the guard trip over the comment
        explaining the bug it guards against, which is an argument for
        deleting the explanation -- exactly backwards.
        """
        import ast
        import inspect
        import textwrap

        return ast.unparse(ast.parse(textwrap.dedent(inspect.getsource(func))))

    def test_the_deploy_path_goes_through_it(self):
        src = self._source(t2.deploy_inference_component)
        assert "registry_source_desc(" in src
        assert "YOLO11-" not in src

    def test_the_eval_queue_goes_through_it(self):
        src = self._source(t2.queue_eval_run)
        assert "registry_source_desc(" in src
        assert "YOLO11-" not in src

    def test_the_inference_queue_goes_through_it(self):
        src = self._source(t2.queue_inference_run)
        assert "registry_source_desc(" in src
        assert "YOLO11-" not in src

    def test_the_export_path_goes_through_the_parser(self):
        src = self._source(t2.export_weights_to_folder)
        assert "_export_stem(" in src
        assert 'removeprefix("yolo11")' not in src
