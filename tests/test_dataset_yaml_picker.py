"""Picking the data.yaml, the way the checkpoint picker picks a .pt.

The dataset browser offered FOLDERS with tick boxes and read
`<folder>/data.yaml` from whichever was ticked. Two problems, and the
second is the one that bites:

  * a dataset whose file is `dataset.yaml`, or which carries several,
    was unreachable;
  * a folder tick box says nothing about which file it is about to read,
    so "Read data.yaml" on a folder holding three of them was a guess
    presented as a choice.

The checkpoint picker had already solved this -- it lists `.pt` files at
the level being browsed and selects one -- so this is that browser used
the same way, not a new one.
"""
from __future__ import annotations

import importlib
import pathlib
import re

t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")
router = importlib.import_module("src.ppe_pipeline_api.routers.track2_models")
WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"


class FakeS3:
    def __init__(self, keys, bodies=None):
        self.keys = keys
        self.bodies = bodies or {}

    def list_objects_v2(self, **kw):
        prefix = kw.get("Prefix", "")
        contents = [{"Key": k} for k in self.keys
                    if k.startswith(prefix) and "/" not in k[len(prefix):]]
        folders = sorted({
            k[len(prefix):].split("/")[0] for k in self.keys
            if k.startswith(prefix) and "/" in k[len(prefix):]
        })
        return {"Contents": contents,
                "CommonPrefixes": [{"Prefix": f"{prefix}{f}/"} for f in folders]}

    def get_object(self, Bucket, Key):  # noqa: N803
        class Body:
            def __init__(self, data):
                self.data = data

            def read(self):
                return self.data

        return {"Body": Body(self.bodies[Key])}


def test_both_yaml_spellings_are_offered(monkeypatch):
    """Ultralytics writes `.yaml`; plenty of exports write `.yml`."""
    monkeypatch.setattr(t2, "s3", FakeS3([
        "retraining/data.yaml",
        "retraining/dataset.yml",
        "retraining/notes.txt",
        "retraining/images/x.jpg",
    ]))
    folders, yamls = t2.list_s3_folders_and_yaml_files("retraining")
    assert folders == ["images"]
    assert sorted(yamls) == ["data.yaml", "dataset.yml"]


def test_a_checkpoint_is_not_a_dataset_file(monkeypatch):
    monkeypatch.setattr(t2, "s3", FakeS3(["weights/model.pt", "weights/data.yaml"]))
    _folders, yamls = t2.list_s3_folders_and_yaml_files("weights")
    assert yamls == ["data.yaml"]


def test_reading_is_keyed_on_the_exact_file(monkeypatch):
    """Picking `dataset.yaml` reads that file, not the `data.yaml` that
    may or may not sit beside it."""
    monkeypatch.setattr(t2, "s3", FakeS3(
        ["r/data.yaml", "r/dataset.yaml"],
        {"r/data.yaml": b"names:\n  0: wrong\n",
         "r/dataset.yaml": b"names:\n  0: right\n"},
    ))
    assert t2.classes_from_yaml_key("r/dataset.yaml") == {0: "right"}
    assert t2.classes_from_yaml_key("r/data.yaml") == {0: "wrong"}


def test_the_list_form_is_read_as_well(monkeypatch):
    """`names: [person, helmet]` is as standard as the mapping form -- it
    is what a Roboflow export writes."""
    monkeypatch.setattr(t2, "s3", FakeS3(
        ["r/data.yaml"], {"r/data.yaml": b"names: [person, helmet]\n"}))
    assert t2.classes_from_yaml_key("r/data.yaml") == {0: "person", 1: "helmet"}


def test_something_that_is_not_a_yaml_is_refused(monkeypatch):
    import pytest

    monkeypatch.setattr(t2, "s3", FakeS3([]))
    with pytest.raises(ValueError, match="not a .yaml"):
        t2.classes_from_yaml_key("r/model.pt")


def test_the_route_still_takes_a_folder():
    """The browser picks a file; the training form and the assemble
    picker still have only a folder in hand, and both call this."""
    import inspect

    source = inspect.getsource(router.dataset_classes)
    assert 'folder: str = ""' in source and 'key: str = ""' in source
    assert "f\"{folder.strip('/')}/data.yaml\"" in source


def test_the_browser_is_used_the_same_way_as_the_checkpoint_one():
    """One browser, one behaviour. The dataset half passed
    `onSelect`/`selected`, which ticks a FOLDER."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    for text in (panel, importer):
        assert "onSelectFile=" in text
        assert "selectedFile=" in text
    block = panel[panel.index("<Track2S3FolderBrowser"):]
    block = block[:block.index("/>")]
    assert "files={yamlFiles}" in block
    assert "onSelect=" not in block and "selected=" not in block


# --- the same yaml, from a laptop -------------------------------------

def test_one_parser_serves_both_sources():
    """A YAML reader in TypeScript would be the THIRD hand-rolled reader
    of this key in the codebase (`pull_control._classes_from_yaml` is the
    second, and cannot import PyYAML because it is the Lambda handler).
    The third is the one that drifts, and the failure is a silently
    shifted class index."""
    import inspect

    assert "classes_from_yaml_text(" in inspect.getsource(t2.classes_from_yaml_key)
    assert "t2.classes_from_yaml_text(" in inspect.getsource(router.dataset_classes_from_text)
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    # Nothing parses yaml in the browser.
    assert "yaml.safe_load" not in panel
    assert "js-yaml" not in panel


def test_text_and_key_agree_on_the_same_document(monkeypatch):
    doc = b"names:\n  0: helmet\n  1: gloves\n"
    monkeypatch.setattr(t2, "s3", FakeS3(["r/data.yaml"], {"r/data.yaml": doc}))
    assert t2.classes_from_yaml_key("r/data.yaml") == t2.classes_from_yaml_text(doc)


def test_an_empty_or_enormous_file_is_refused():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as empty:
        router.dataset_classes_from_text(
            router.YamlTextRequest(text="   ", filename="data.yaml"), session=None)
    assert empty.value.status_code == 422
    with pytest.raises(HTTPException) as huge:
        router.dataset_classes_from_text(
            router.YamlTextRequest(text="a" * 1_000_001), session=None)
    assert huge.value.status_code == 422


def test_something_that_is_not_yaml_says_so():
    import pytest
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        router.dataset_classes_from_text(
            router.YamlTextRequest(text="\x00\x01 not: [yaml", filename="x.yaml"),
            session=None)
    assert exc.value.status_code == 400
    assert "x.yaml" in exc.value.detail


def test_nothing_is_uploaded_to_read_class_names():
    """Reading six class names is not a reason to leave a file in the
    bucket."""
    import inspect

    source = inspect.getsource(router.dataset_classes_from_text)
    assert "put_object" not in source and "upload" not in source


def test_the_card_offers_both_and_filters_to_yaml():
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert '"From my computer"' in panel and '"From the bucket"' in panel
    assert 'accept=".yaml,.yml' in panel
    # Both land in one place, so a yaml off a laptop and one in the
    # bucket cannot be interpreted differently: the declaration plus
    # exactly one call per source.
    assert panel.count("applyReadClasses(") == 3
    assert "function applyReadClasses(" in panel


# --- browsing any bucket, not just this app's --------------------------

def test_the_yaml_browser_takes_a_bucket():
    """It was bound to this app's own, which made "From the bucket" mean
    "from the one bucket we happen to write to" -- and the datasets are
    as often in another."""
    import inspect

    pc = importlib.import_module("src.ppe_pipeline_api.pull_control")
    for fn in (pc.list_folders_and_yaml_files, pc.read_yaml_object):
        assert "bucket" in inspect.signature(fn).parameters, fn.__name__
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "listTrack2YamlFiles(yamlBucket, prefix)" in panel
    assert "readTrack2YamlClasses(yamlBucket, key)" in panel


def test_only_this_level_and_only_yaml(monkeypatch):
    """A nested key would render as a name with slashes in it and select
    something the browser is not showing."""
    pc = importlib.import_module("src.ppe_pipeline_api.pull_control")

    class Paged:
        def list_objects_v2(self, **kw):
            prefix = kw["Prefix"]
            return {
                "CommonPrefixes": [{"Prefix": f"{prefix}images/"}],
                "Contents": [
                    {"Key": f"{prefix}data.yaml"},
                    {"Key": f"{prefix}dataset.yml"},
                    {"Key": f"{prefix}model.pt"},
                    {"Key": f"{prefix}images/nested.yaml"},
                ],
                "IsTruncated": False,
            }

    monkeypatch.setattr(pc, "s3", Paged())
    out = pc.list_folders_and_yaml_files(bucket="other-bucket", prefix="sets")
    assert out["bucket"] == "other-bucket"
    assert out["folders"] == ["images"]
    assert out["yaml_files"] == ["data.yaml", "dataset.yml"]


def test_reading_checks_the_extension_before_the_fetch(monkeypatch):
    """This is handed a key the browser chose, and "read whatever key you
    are given" is a wider door than reading class names needs."""
    import pytest

    pc = importlib.import_module("src.ppe_pipeline_api.pull_control")

    class Boom:
        def get_object(self, **kw):
            raise AssertionError("fetched before checking the extension")

    monkeypatch.setattr(pc, "s3", Boom())
    with pytest.raises(ValueError, match="not a .yaml"):
        pc.read_yaml_object(bucket="b", key="secrets/credentials.json")


def test_a_new_bucket_resets_the_path():
    """Keeping the old prefix would walk to a path that does not exist in
    the new bucket and read as an empty bucket."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    block = panel[panel.index("setYamlBucket(e.target.value);"):]
    block = block[:block.index("}}")]
    assert 'setPrefix("")' in block and 'setPicked("")' in block


def test_all_three_sources_parse_the_same_way():
    """A yaml read from a laptop, from this app's bucket and from
    somebody else's all have to mean the same thing."""
    import inspect

    router_pc = importlib.import_module("src.ppe_pipeline_api.routers.track2_pull_control")
    assert "t2.classes_from_yaml_text(" in inspect.getsource(router_pc.yaml_classes)
    assert "t2.classes_from_yaml_text(" in inspect.getsource(router.dataset_classes_from_text)
    assert "classes_from_yaml_text(" in inspect.getsource(t2.classes_from_yaml_key)


# --- the card reads as two decisions ------------------------------------

def test_the_card_is_two_titled_parts():
    """It ran Name, then a collapsed "Read from a data.yaml", then the
    class list as one undifferentiated form -- so "name the Class" and
    "where do its class names come from" read as one decision rather
    than two, and the second was behind a button."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "1 &middot; Name the Model Name" in panel
    assert "2 &middot; Read its classes from a data.yaml" in panel
    assert panel.index("1 &middot; Name") < panel.index("2 &middot; Read its classes")


def test_the_yaml_picker_is_not_behind_a_toggle():
    """Reading the names IS the way to fill this in -- a label file
    carries "0 0.51 0.22 ..." and no name at all -- and typing them is
    the fallback. A collapsed toggle made the correct route the one you
    had to go looking for."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "readOpen" not in panel
    # Comments stripped: the history of the change is allowed to name the
    # control it removed, and this is about what somebody READS. It
    # caught the card's own "?" help still pointing at the old button.
    visible = re.sub(r"\{/\*.*?\*/\}", "", panel, flags=re.S)
    visible = re.sub(r"//.*", "", visible)
    assert "Read from a data.yaml" not in visible
    # Both sources still offered, in the open.
    assert '"From my computer"' in panel and '"From the bucket"' in panel


def test_the_listings_follow_the_card_being_in_play():
    """They used to gate on the toggle. With it gone they would run on
    every page load, including while the card is dimmed and nothing can
    be chosen."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "if (bodyDimmed || buckets.length) return;" in panel
    assert 'if (bodyDimmed || yamlSource !== "bucket" || !yamlBucket) return;' in panel
    # And it is declared BEFORE them: a dependency array is evaluated
    # during render, so a const declared after would be in its temporal
    # dead zone.
    assert panel.index("const bodyDimmed") < panel.index("if (bodyDimmed || buckets.length)")


def test_a_successful_read_still_says_so():
    """The confirmation lived in the block the toggle opened."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "{readNote && (" in panel


# --- the week prefix is fixed, and shown -------------------------------

def test_the_week_prefix_cannot_be_edited():
    """`plan_routing` and `week_at` skip any folder that does not start
    with it, so changing it does not RENAME anything -- it makes every
    week folder already in the bucket invisible to the system at once."""
    panel = (WEB / "components" / "Track2PullControlPanel.tsx").read_text(encoding="utf-8")
    block = panel[panel.index("Week prefix"):]
    block = block[:block.index("New week every")]
    # Comments stripped -- the comment here explains WHY `readOnly` and
    # not `disabled`, and naming the thing it rejected should not fail
    # the assertion about what the markup does.
    block = re.sub(r"\{/\*.*?\*/\}", "", block, flags=re.S)
    assert "readOnly" in block
    assert "onChange" not in block
    # `readOnly`, not `disabled`: a disabled input is skipped by the
    # keyboard and reads as unavailable, when the value is the thing
    # worth reading.
    assert "disabled" not in block


def test_it_shows_what_is_actually_configured():
    """A read-only field that hides the real value is just a field that
    lies: `week_3` on screen and `week_3` in the bucket have to be the
    same string."""
    panel = (WEB / "components" / "Track2PullControlPanel.tsx").read_text(encoding="utf-8")
    assert 'setWeekPrefix(c.week_prefix || "week_")' in panel
    # Not blanked to let a placeholder speak -- that was for the
    # editable version.
    assert 'c.week_prefix === "week_" ? ""' not in panel


def test_the_server_still_decides_the_default():
    """The card no longer offers a way to change it, but the stored
    config is still what every reader consults."""
    pc = importlib.import_module("src.ppe_pipeline_api.pull_control")
    assert pc.DEFAULTS["week_prefix"] == "week_"
