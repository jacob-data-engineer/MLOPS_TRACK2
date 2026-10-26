"""One field, and what it becomes.

"Define a new Class" and the import card both asked for a Slug and a
Name side by side -- the same question twice, where one of the two
silently decided the registry group every version of that Class would be
filed under for the rest of its life, and the other was a label. There is
one Name field now and the identifier is derived.

That makes the derivation load-bearing in a way a free-text field never
was, in two ways these tests pin:

  * it has to satisfy the SERVER's rule. `upsert_detector` requires
    `[a-z0-9]([a-z0-9-]{0,30})` because the result is interpolated into
    `yolo{generation}{size}-{slug}-detector`. Anything the browser
    derives that fails it produces a 422 naming a field nobody can see
    any more.
  * there has to be exactly ONE of it. Two cards create Classes, and two
    derivations that disagreed would file the same typed name under two
    different group names depending on which card was used.

The derivation itself is TypeScript, so these read the rule out of it
and re-implement it here -- the point is the AGREEMENT between the two
rules, which is what would silently rot.
"""
from __future__ import annotations

import importlib
import pathlib
import re

SLUG = (pathlib.Path(__file__).resolve().parents[1]
        / "web" / "src" / "lib" / "slug.ts")
WEB = pathlib.Path(__file__).resolve().parents[1] / "web" / "src"
t2 = importlib.import_module("src.ppe_pipeline_api.track2_shared")

SOURCE = SLUG.read_text(encoding="utf-8")

#: The TypeScript, as Python. Kept beside the real thing rather than
#: imported, because there is no JS runtime here -- the tests below
#: assert the two stay in step.
def derive(name: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", (name or "").lower())
    out = out.strip("-")[:31]
    return out.rstrip("-")


SERVER_RULE = re.compile(r"[a-z0-9]([a-z0-9-]{0,30})")


def test_the_two_rules_still_say_the_same_thing():
    """If `upsert_detector` tightens its pattern, this fails rather than
    the next person typing a name and getting a 422 about a field that is
    no longer on screen."""
    import inspect

    source = inspect.getsource(t2.upsert_detector)
    assert r"[a-z0-9]([a-z0-9-]{0,30})" in source
    assert "SLUG_MAX = 31" in SOURCE


def test_the_browser_mirrors_the_server_pattern():
    assert 'replace(/[^a-z0-9]+/g, "-")' in SOURCE
    assert 'replace(/^-+|-+$/g, "")' in SOURCE
    assert ".slice(0, SLUG_MAX)" in SOURCE
    # Truncating at 31 can land on a hyphen, which the server rejects.
    assert 'replace(/-+$/, "")' in SOURCE


def test_everything_a_person_would_type_passes_the_server():
    for typed in [
        "Vehicle detection",
        "Ship",
        "fall zone",
        "Fall zone / vehicle",
        "PPE -- helmets & gloves",
        "crane_2",
        "  padded  ",
        "A" * 80,
        "ship---detection",
        "2026 vehicles",
        "Ünïcôde name",
        "a" * 30 + "-tail",
    ]:
        slug = derive(typed)
        assert slug, typed
        assert SERVER_RULE.fullmatch(slug), (typed, slug)
        assert len(slug) <= 31, (typed, slug)


def test_a_name_with_nothing_usable_in_it_derives_to_nothing():
    """Empty has to stay a real answer. Inventing something like
    "class-1" would file a model under a name nobody chose and nobody
    could find again -- so the card refuses instead."""
    for typed in ["!!!", "   ", "---", "", "/+/"]:
        assert derive(typed) == ""
    assert "slugProblem" in SOURCE
    assert "at least one letter or digit" in SOURCE


def test_a_truncated_name_never_ends_in_a_hyphen():
    """31 characters lands mid-word often enough to matter, and the
    server rejects a trailing hyphen."""
    slug = derive("a" * 30 + " tail")
    assert SERVER_RULE.fullmatch(slug)
    assert not slug.endswith("-")


def test_the_group_name_is_built_the_same_way_on_both_sides():
    import inspect

    assert '`yolo${generation}${size}-${slug}-detector`' in SOURCE
    assert '"yolo{generation}{size}-" + f"{slug}-detector"' \
        in inspect.getsource(t2.upsert_detector)


def test_a_derived_slug_can_never_be_a_production_group():
    """`yolo11{s,m,l,x}-detector` is an exact-match list on the rule that
    fires `dump_approved_weights` -- it overwrites production weights and
    swaps the live inference component. A Class slug is always inserted
    between the size and `-detector`, so the only way to produce one of
    those four names is an EMPTY slug, which both sides refuse."""
    for size in ("s", "m", "l", "x"):
        production = f"yolo11{size}-detector"
        assert not SERVER_RULE.fullmatch("")
        # The shortest a real slug can be is one character, and even that
        # cannot collapse the name back to the production form.
        assert f"yolo11{size}-a-detector" != production


def test_there_is_exactly_one_derivation_in_the_frontend():
    """Two cards create Classes. Two derivations that drifted would file
    one typed name under two different registry groups."""
    importers = [p for p in WEB.rglob("*.tsx")
                 if "slugFromName" in p.read_text(encoding="utf-8")]
    assert len(importers) >= 2, "expected both cards to use the helper"
    for path in importers:
        text = path.read_text(encoding="utf-8")
        assert 'from "../lib/slug"' in text, path.name
        # No card rolls its own.
        assert 'replace(/[^a-z0-9-]+/g' not in text, f"{path.name} derives its own slug"


def test_neither_card_still_asks_for_a_slug():
    for name in ("Track2DetectorsPanel.tsx", "Track2ImportCheckpointPanel.tsx"):
        text = (WEB / "components" / name).read_text(encoding="utf-8")
        assert 'placeholder="slug, e.g. ship"' not in text, name
        assert "Slug{\" \"}" not in text, name


def test_editing_a_class_cannot_move_its_identifier():
    """The slug is in the registry group name. Renaming the group would
    strand every version already filed under the old one, and the server
    would create a second Class rather than rename the first."""
    text = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "const slug = editingSlug || slugFromName(label);" in text
    assert "Keeps its original identifier" in text


# --- the either/or between the file's classes and a hand-written one ----

def test_the_checkbox_is_exclusive_with_define_a_new_class():
    """Two answers to one question -- what does this model detect -- and
    the registry records only one. Both live at once is the state that
    files a version under a taxonomy nobody chose."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    page = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")

    assert "Use the classes in this file" in importer
    # One piece of state, on the page: the two controls live in two
    # components and a copy in each is a pair that can disagree.
    assert "onPendingImportChange={setPendingImport}" in page
    assert "defineDimmed={!pendingImport}" in page
    assert 'bodyDimmed ? "pointer-events-none opacity-40" : ""' in panel


def test_editing_an_existing_class_is_still_reachable():
    """Changing what a Model Name detects is done by typing its name into
    the define form and saving: the server upserts, and the card warns
    that it will replace.

    There was an Edit button on the row that loaded it into the form;
    it was removed deliberately. `editingSlug` is what it set, and the
    `!editingSlug` clause below is the un-dimming that went with it --
    kept because the slug and clash checks still read the flag, but
    nothing sets it any more.
    """
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "const bodyDimmed = (defineDimmed && !editingSlug) || lockedByPick;" in panel
    assert 'bodyDimmed ? "pointer-events-none opacity-40" : ""' in panel
    assert "{!bodyDimmed && (" in panel
    # The upsert is what makes retyping a name an edit.
    assert "saveTrack2Detector(slug, label, classes)" in panel


def test_registering_cannot_create_a_class_while_the_box_is_off():
    """Off means the Class is being defined by hand. Creating one from
    the file's names anyway would be the card doing the thing the
    operator just switched off."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "!useFileClasses || !newSlug || !!newLabelProblem" in importer


def test_the_renamed_labels_are_in_place():
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "Available classes in the .pt file" in importer
    assert "Input a name for the model with these classes" in importer
    assert "Register New Model Name" in importer
    # And the wording each replaced is gone.
    assert "This checkpoint detects" not in importer
    assert "No Class has these class names" not in importer
    assert "Create detector and register" not in importer
    assert "Give the new Class a name -- a version has to belong to one." not in importer


# --- registering a checkpoint into a Class it does not detect -----------

def test_a_class_mismatch_is_still_refused_by_default():
    """The guard is the reason a ship checkpoint cannot be filed in the
    helmet registry. Relaxing it for the seed case must not relax it for
    everything."""
    import inspect

    source = inspect.getsource(t2.register_uploaded_base_model)
    assert "if mismatched and not allow_class_mismatch:" in source
    assert "allow_class_mismatch: bool = False" in source


def test_a_seed_says_what_it_is_in_the_field_every_page_shows():
    """Until the first training run the weights detect one thing and the
    registry says another, and Export Control's `newest` criterion will
    hand that .pt to somebody. The description is what the metrics page,
    the base-model picker and the registry console all already read."""
    import inspect

    source = inspect.getsource(t2.register_uploaded_base_model)
    assert "SEED_DESCRIPTION_PREFIX" in source
    assert "NOT A DETECTOR FOR THIS CLASS YET" in t2.SEED_DESCRIPTION_PREFIX
    assert "The head is rebuilt on the first" in source


def test_a_seed_is_registered_pending_not_approved():
    """It is not an approved detector for this Class -- it is waiting for
    the run that makes it one. Pending keeps it out of everything that
    filters on Approved, including `export_weights_to_folder`, while
    leaving it usable as a starting point."""
    import inspect

    source = inspect.getsource(t2.register_uploaded_base_model)
    assert 'ModalApprovalStatus' not in source
    assert '"PendingManualApproval" if mismatched else "Approved"' in source


def test_the_registry_records_both_class_lists():
    """`classes` was the DETECTOR's list, which on a seed is exactly the
    half that is not true of the weights."""
    import inspect

    source = inspect.getsource(t2.register_uploaded_base_model)
    assert '"checkpoint_classes"' in source
    assert '"class_mismatch"' in source


def test_the_button_saves_the_class_before_registering_into_it():
    """There is nothing to register into until the Class exists, and a
    failure between the two leaves a saved Class and no version -- which
    is visible and recoverable, unlike a version filed under a Class that
    was never written."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    body = panel[panel.index("async function handleSaveAndRegister"):]
    body = body[:body.index("async function handleSave(")]
    assert body.index("handleSave({ keepForm: true })") < body.index("registerTrack2BaseModel(")
    assert "if (!saved) return;" in body


def test_there_is_one_button_not_two_when_a_checkpoint_is_waiting():
    """Register saves the Class and then files the checkpoint into it,
    so a second button doing only the first half is a worse version of
    what the one beside it already does -- and the obvious mistake is
    pressing it and wondering where the version went."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    # Anchored on the handler, because `{pendingImport ? (` now also
    # opens the collapsed card's summary further up the file.
    at = panel.index("onClick={handleSaveAndRegister}")
    branch = panel[panel.rindex("{pendingImport ? (", 0, at):]
    branch = branch[:branch.index(") : (")]
    assert "Register New Model Name" in branch
    assert "Save classes" not in branch
    # And without a checkpoint, Save is the only thing there is.
    other = panel[panel.index(") : (", at):]
    other = other[:other.index("</div>")]
    assert "Save classes" in other
    assert "Register New Model Name" not in other


def test_the_waiting_box_states_a_count_not_eighty_names():
    """A warning box is not the place to read a taxonomy. The import
    card above lists them in full, in a table; eighty names wrapping
    here buried the sentence underneath them."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    block = panel[panel.index("Its weights currently detect"):]
    block = block[:block.index("head is rebuilt")]
    assert "checkpointClasses.length" in block
    assert "slice(0, 4)" in block
    assert "checkpointClasses.join(\", \")" not in block


def test_the_button_only_appears_when_a_checkpoint_is_waiting():
    """With the box ticked the import card registers the checkpoint
    itself. A second button doing the same thing is two ways to perform
    one action, which is how they drift."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "inspection && !useFileClasses" in importer
    assert "{pendingImport && (" in panel


def test_the_card_states_what_the_weights_actually_detect():
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "Its weights currently detect" in panel
    assert "the head is rebuilt on the first training" in panel.lower()
    assert "Do not" in panel and "deploy it as it stands" in panel


def test_a_waiting_checkpoint_registers_into_one_group_not_eight():
    """A Class OWNS a group at every generation and size, because a
    future run can train it at any of them. But the press files the
    checkpoint into exactly one -- the one its architecture belongs to --
    and listing all eight under a button about to use one of them reads
    as if it landed in all eight."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "const pendingGroup = pendingArch && slug && !nameProblem" in panel
    assert "groupNameFor(slug, pendingArch[1], pendingArch[2])" in panel
    assert "files\n                    this checkpoint into" in panel
    # The others stay listed, as what the Class owns for later runs.
    assert "for runs at a" in panel


def test_the_architecture_is_parsed_not_assumed():
    """`yolo11s` -> generation 11, size s. Stripping a literal "yolo11"
    is a no-op on yolo26 and has produced a size of "yolo26m" here
    before."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert r"/^yolo(\d+)([smlx])$/" in panel


def test_the_builtin_name_is_refused_in_the_card_not_by_a_422():
    """With the identifier derived and no longer on screen, "PPE"
    colliding with the built-in `ppe` is otherwise a server error about
    a field nobody can see. The server refuses it either way."""
    import inspect

    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "clash.builtin ?" in panel
    assert "already exists" in panel and "unable to comply" in panel
    assert "Boolean(clash?.builtin)" in panel
    # The card and the server say the same thing, so hitting the refusal
    # through the API reads the same as being stopped before it.
    assert "already exists -- unable to comply" \
        in inspect.getsource(t2.upsert_detector)


# --- training from a version that already exists ------------------------

def test_the_base_model_picker_has_no_tick():
    """There was briefly one meaning "I am training, not importing",
    which dimmed the import card above -- removed as not needed. This
    card never switches off anyway: it is the only way a run gets its
    starting weights, so a box that could disable it would be a control
    that breaks the page."""
    picker = (WEB / "components" / "Track2BaseModelPicker.tsx").read_text(encoding="utf-8")
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    page = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    assert '<p className="mb-2 text-sm font-medium text-ink">Train on an existing model</p>' in picker
    assert "onlyThis" not in picker
    # And nothing dims the import card any more.
    assert "importDimmed" not in panel and "importDimmed" not in page
    assert "trainingFromExisting" not in page


def test_a_registered_version_lands_where_it_can_be_trained():
    """The base-model picker filters on step 1's Class, model and size.
    An import that left step 1 pointing elsewhere put the new version in
    a list nobody was looking at -- which to the operator is the same as
    the import having failed."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    page = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")

    # It reports what it registered...
    assert "onImported?.({ slug, architecture, arn: registered.model_package_arn })" in importer
    # ...the page keeps it...
    assert "setJustRegistered(registered)" in page
    assert "selectVersion={justRegistered}" in page
    # ...and the panel moves all three dropdowns and preselects it.
    assert "void chooseDetector(selectVersion.slug, parsed[2], parsed[1])" in panel
    assert "onBaseVersion?.(selectVersion.arn)" in panel
    assert "setSelection({ basePackageArn: arn })" in page


def test_the_landing_marks_itself_restored():
    """`onChoose(choice, restored)` clears the version when somebody
    PICKS a different Class -- right, because a version belongs to one
    group. This is not that: the version was created for exactly this
    Class a moment ago, so clearing it would undo the whole point."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    body = panel[panel.index("if (!selectVersion) return;"):]
    body = body[:body.index("async function chooseDetector")]
    assert "}, true);" in body


def test_a_delete_lands_on_nothing():
    """There is no new version to select after one is removed."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    body = importer[importer.index("await deleteTrack2Model("):]
    body = body[:body.index("} catch")]
    assert "onImported?.();" in body


# --- a seed says it is a seed, everywhere a status is shown -------------

def test_one_marker_on_both_sides():
    """The browser reads seed-ness off the description the server wrote.
    A longer sentence drifting on one side would silently stop matching
    on the other, and the version would quietly go back to reading
    'PendingManualApproval'."""
    slug_ts = SLUG.read_text(encoding="utf-8")
    assert t2.SEED_MARKER == "SEED --"
    assert f'export const SEED_PREFIX = "{t2.SEED_MARKER}";' in slug_ts
    assert t2.SEED_DESCRIPTION_PREFIX.startswith(t2.SEED_MARKER)


def test_seed_ness_is_not_version_one():
    """A v1 produced by a training run is a real model. Labelling it a
    seed would be a lie in the one column whose job is saying what a
    version is."""
    assert t2.is_seed_description(t2.SEED_DESCRIPTION_PREFIX + " anything")
    assert t2.is_seed_description("   " + t2.SEED_DESCRIPTION_PREFIX)
    assert not t2.is_seed_description("Imported checkpoint, not trained here.")
    assert not t2.is_seed_description("")
    assert not t2.is_seed_description(None)
    # Nothing anywhere keys the label on the version number.
    for name in ("Track2BaseModelPicker.tsx", "Track2ModelPicker.tsx"):
        text = (WEB / "components" / name).read_text(encoding="utf-8")
        assert 'version === "1"' not in text, name


def test_every_place_a_version_shows_a_status_knows_about_seeds():
    base = (WEB / "components" / "Track2BaseModelPicker.tsx").read_text(encoding="utf-8")
    metrics = (WEB / "components" / "Track2ModelPicker.tsx").read_text(encoding="utf-8")
    export = (WEB / "components" / "Track2MetricsExportCard.tsx").read_text(encoding="utf-8")

    # Both go through the shared `statusLabel`, so "Approved" reads
    # "Ready" in one place rather than three that can drift.
    assert 'isSeedVersion(row.description) ? "Seed" : statusLabel(row.status)' in base
    assert 'isSeedVersion(row.description) ? "Seed" : statusLabel(row.status)' in metrics
    assert 'pick.is_seed' in export
    # The raw status is still reachable, not replaced outright.
    assert "title={isSeedVersion(row.description)" in base


def test_the_export_pick_carries_it():
    """`criterion="newest"` will pick a seed -- nothing filters on
    status, by choice -- so the fact has to travel with the pick rather
    than being left in a description the card does not show."""
    import inspect

    assert '"is_seed": t2.is_seed_description(' in inspect.getsource(mx_describe_pick_source())


def mx_describe_pick_source():
    import importlib

    return importlib.import_module("src.ppe_pipeline_api.metrics_export").describe_pick


# --- the vocabulary on the dropdowns ------------------------------------

def test_the_three_dropdowns_read_the_same_way_everywhere():
    """One row -- what is it called, which network, which size -- appears
    on three cards. Two vocabularies for one question is the confusion
    the rename was for."""
    for name, path in [
        ("Track2DetectorsPanel.tsx", WEB / "components" / "Track2DetectorsPanel.tsx"),
        ("Track2ModelPicker.tsx", WEB / "components" / "Track2ModelPicker.tsx"),
        ("Track2MetricsExportCard.tsx", WEB / "components" / "Track2MetricsExportCard.tsx"),
    ]:
        text = path.read_text(encoding="utf-8")
        assert "Model Name" in text, name
        assert "Model type" in text, name


def test_deleting_asks_about_the_model_by_name():
    """The question names the thing. It is asked in the dialog's title
    now rather than in a `window.confirm` -- see
    test_destructive_actions_ask_properly.py -- but it still has to say
    WHICH Model Name, because the Delete button is a row away from the
    picker that chose it."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert 'Delete "${deleteTarget?.label}"?' in panel
    # And the refusal names it too, since that dialog is a different
    # question: not "shall I" but "it has versions, still?".
    assert '"${deleteTarget?.label}" has registered versions' in panel
    assert 'Delete the "${detector.label}" Class?' not in panel


# --- how a status reads, and the stock weights library ------------------

def test_approved_reads_as_ready_in_one_place():
    """`Approved` is SageMaker's stored value and stays that way --
    `export_weights_to_folder` and the approval rule both key on it. What
    it MEANS here is ready to deploy or train from, so that is what it
    says, decided once rather than in each of three cards."""
    slug_ts = SLUG.read_text(encoding="utf-8")
    assert 'if (status === "Approved") return "Ready";' in slug_ts
    for name in ("Track2BaseModelPicker.tsx", "Track2ModelPicker.tsx",
                 "Track2MetricsExportCard.tsx"):
        text = (WEB / "components" / name).read_text(encoding="utf-8")
        assert "statusLabel" in text, name
        assert 'from "../lib/slug"' in text, name
    # No card keeps its own copy of the rule.
    metrics = (WEB / "components" / "Track2ModelPicker.tsx").read_text(encoding="utf-8")
    assert "function statusLabel" not in metrics


def test_the_server_still_stores_approved():
    """The label is a display decision. Changing the stored value would
    break the approval rule that fires `dump_approved_weights` and the
    Approved-only check in `export_weights_to_folder`."""
    import inspect

    assert 'ModelApprovalStatus="PendingManualApproval" if mismatched else "Approved"' \
        in inspect.getsource(t2.register_uploaded_base_model)
    assert '!= "Approved"' in inspect.getsource(t2.export_weights_to_folder)


def test_the_weights_library_lists_what_is_there_not_what_should_be():
    """The folder is seeded by a script. A card promising `yolo26l.pt`
    when its upload failed sends somebody looking for a file that does
    not exist."""
    import inspect

    source = inspect.getsource(t2.list_yolo_weights_library)
    assert "list_objects_v2" in source
    assert "KNOWN_ARCHITECTURES" in source
    assert t2.YOLO_WEIGHTS_LIBRARY_PREFIX == "track2-yolo-weights"


def test_the_library_lives_in_the_bucket_that_is_already_granted():
    """A bucket of its own would need an IAM grant plus a change to the
    platform stack, which has drift."""
    import inspect

    source = inspect.getsource(t2.list_yolo_weights_library)
    assert "Bucket\": BUCKET" in source or '"Bucket": BUCKET' in source


def test_the_mirror_script_uses_a_tag_that_has_both_generations():
    """`v8.3.0` carries yolo11 and NOT yolo26 -- every yolo26 asset 404s
    against it -- and `latest` is not a resolvable tag on that repo.
    Checked against GitHub, not assumed: half the library would have
    mirrored and the other half skipped with a 404 nobody reads as a
    wrong tag."""
    import importlib.util
    import pathlib as _pathlib

    path = _pathlib.Path(__file__).resolve().parents[1] / "scripts" / "mirror_yolo_weights.py"
    spec = importlib.util.spec_from_file_location("mirror_yolo_weights", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.RELEASE_TAG == "v8.4.0"
    assert module.SIZES == ("s", "m", "l")
    assert module.wanted() == [
        "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
        "yolo26s.pt", "yolo26m.pt", "yolo26l.pt",
    ]
    # Detection only: -seg, -pose, -cls and -obb are different tasks, and
    # one offered beside the others would train and register as if it
    # were a detector.
    assert all("-" not in name for name in module.wanted())


def test_the_checkpoints_classes_are_a_table_not_a_comma_run():
    """A COCO checkpoint carries eighty classes, and
    "0:person, 1:bicycle, 2:car, ..." wrapped over six lines is
    unreadable in exactly the place the index matters most -- a label row
    carries the NUMBER and never the name, so "index 2 means car" is the
    fact being checked here."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    block = importer[importer.index("Available classes in the .pt file"):]
    block = block[:block.index("Architecture not recorded in the file")]
    assert "<table" in block
    assert ">No<" in block and ">Class<" in block
    # Sorted numerically: as text, 10 sorts before 2 and the list is
    # silently wrong from the eleventh class on.
    assert "Number(a[0]) - Number(b[0])" in block
    # Long lists scroll instead of pushing the card off screen.
    assert "max-h-56 overflow-y-auto" in block
    assert "sticky top-0" in block


# --- the import card stays editable after a file is read ----------------

def test_nothing_above_the_reading_is_frozen_by_it():
    """Everything locked the moment a checkpoint was read, so changing
    the source, the size or the release meant pressing Start over and
    doing it again."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "disabled={!!inspection}" not in importer
    assert "Start over" not in importer


def test_changing_the_file_clears_what_was_read():
    """A panel of classes belonging to a file you are no longer pointing
    at is the worst of the three states this could be in."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    # Every control that picks a different FILE.
    for anchor in ("setSource(value);", "setFile(e.target.files?.[0] ?? null);",
                   "setBucketKey(e.target.value);", "onSelectFile={(key) => {"):
        block = importer[importer.index(anchor):]
        block = block[:block.index("}}")]
        assert "setInspection(null)" in block, anchor


def test_changing_the_size_does_not_clear_it():
    """The size is a DECLARATION about the file, not a choice of file.
    Clearing the reading would throw away a correct answer because the
    question it is compared against moved."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    block = importer[importer.index("value={architecture}"):]
    block = block[:block.index("</select>")]
    assert "setInspection(null)" not in block


def test_the_mismatch_is_recomputed_not_read_off_the_reading():
    """`architecture_matches` was baked server-side against the size as
    it stood when the file was read. With the dropdown now editable, a
    stale boolean would keep saying "matches" about a comparison nobody
    is making any more."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "inspection.detected_architecture !== architecture" in importer
    assert "inspection.architecture_matches" not in importer


def test_a_wrong_size_cannot_be_registered():
    """Freezing the dropdown was the only thing preventing this before:
    read yolo11s.pt, switch the size to yolo11m, and the checkpoint lands
    in the yolo11m group -- a mislabelling that follows every model
    fine-tuned from it."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    # The CONDITION, not its line breaks. Pinning the exact string broke
    # the moment a third clause was added and the `||` wrapped -- a test
    # failing over whitespace says nothing about whether the guard works.
    guard = importer.split("onClick={handleRegister}")[0].rsplit("disabled={", 1)[-1]
    assert "architectureMismatch" in guard
    assert "!!busy" in guard
    # A file that does not RECORD its architecture is not a mismatch.
    assert "Boolean(\n    inspection?.detected_architecture" in importer


# --- Define a new Class starts shut ------------------------------------

def test_define_a_new_class_starts_shut_with_no_extra_control():
    """It read as a second thing to fill in beside the import card. Shut
    at the start, opened with the card's own Show button -- there was
    briefly a tick beside the title doing this, and it was one control
    too many: Show/Hide already says whether a card is in play."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    page = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    block = panel[panel.index('title="Define a new Model Name"'):]
    block = block[:block.index("help={")]
    assert "defaultOpen={false}" in block
    assert "Define a Class myself" not in panel
    assert "defineActive" not in panel and "defineActive" not in page


def test_a_shut_card_still_says_a_checkpoint_is_waiting():
    """The one moment it matters is when a checkpoint needs a Class that
    has not been named yet, which would otherwise be invisible behind a
    closed card."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    block = panel[panel.index("summary={pendingImport ? ("):]
    block = block[:block.index("help={")]
    assert "A checkpoint is waiting for a Model Name" in block
    assert "Press <em>Show</em>" in block


def test_only_the_body_is_dimmed():
    """The card's own Show/Hide stays usable, so it can be looked at
    while it is out of play."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert 'bodyDimmed ? "pointer-events-none opacity-40" : ""' in panel


def test_the_define_card_is_dimmed_in_every_state_but_one():
    """The only moment defining a Class by hand is the next thing to do
    is a checkpoint read with its own class names declined. Every other
    state has an answer somewhere else: no checkpoint means there is
    nothing for the Class to hold, and a checkpoint whose classes are
    accepted has already named them."""
    page = (WEB / "routes" / "Track2ModelsPage.tsx").read_text(encoding="utf-8")
    assert "defineDimmed={!pendingImport}" in page


def test_no_button_is_offered_while_the_card_is_dimmed():
    """The dimming already stops the press, but a greyed-out button
    still reads as the thing to reach for -- and the answer there is
    upstairs rather than here."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    at = panel.index("onClick={handleSaveAndRegister}")
    guard = panel.rindex("{!bodyDimmed && (", 0, at)
    # Immediately around the button row, not somewhere far above it.
    assert 0 < at - guard < 1200


# --- seeding an EXISTING Class ------------------------------------------

def test_every_class_is_offered_not_only_the_matching_ones():
    """Filtering the dropdown to Classes whose class names matched the
    file made the whole point of a seed unreachable: a COCO checkpoint
    detects eighty things and matches nothing, so the only offer was
    "create a new eighty-class Class" -- and giving an EXISTING Class its
    first checkpoint to train from, which is what a seed IS, could not be
    done at all."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "detectors.filter((d) => !d.builtin).map" in importer
    # The old filter is gone entirely, not merely bypassed.
    assert "const matching = detectors.filter" not in importer


def test_a_mismatch_is_labelled_in_the_option_itself():
    """Before the press, in the list, not only after it."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "-- same classes" in importer
    assert "SEED, it detects" in importer
    assert "This will register as a seed." in importer


def test_the_seed_flag_follows_the_chosen_class():
    """A NEW Class is built FROM this file's classes, so it never
    mismatches and must not be waved through as though it might."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "const seedForChosen = Boolean(" in importer
    assert "chosenDetector && classSignature(chosenDetector.classes) !== foundSignature" in importer
    block = importer[importer.index("const registered = await registerTrack2BaseModel("):]
    block = block[:block.index(");")]
    assert "seedForChosen" in block


def test_nothing_is_preselected():
    """Preselecting the first match filed a checkpoint into a Class
    nobody had looked at -- and now that every Class is offered, the same
    slip would reach one with different classes entirely."""
    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert 'setDetectorSlug("");' in importer
    assert "setDetectorSlug(fits[0]?.slug ?? \"\")" not in importer


def test_the_builtin_is_not_offered():
    """`register_uploaded_base_model` refuses it outright -- it owns the
    production groups, which the replay pipeline feeds."""
    import inspect

    importer = (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(encoding="utf-8")
    assert "!d.builtin" in importer
    assert "builtin" in inspect.getsource(t2.register_uploaded_base_model)


def test_the_chosen_class_shows_what_it_detects():
    """The row said "PPE2 (1 classes)" and where its versions are filed,
    and nothing about the taxonomy -- which is the thing every dataset is
    checked against and the thing a training run inherits."""
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "const chosenClasses = Object.entries(" in panel
    assert "{chosenClasses.length > 0 && (" in panel
    assert ">Detects{" in panel.replace(" ", "") or "Detects{" in panel
    # In INDEX order, numerically: as text 10 sorts before 2 and the list
    # goes wrong at the eleventh class.
    block = panel[panel.index("const chosenClasses"):]
    block = block[:block.index(";")]
    assert "Number(a[0]) - Number(b[0])" in block
    # The index is shown, because that is what a label file carries.
    assert "{index}" in panel


def test_one_class_is_not_one_classes():
    panel = (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")
    assert "{Object.keys(d.classes).length === 1 ? \"\" : \"es\"})" in panel


# --- picking some of the file's classes ---------------------------------
#
# A COCO checkpoint carries eighty and you rarely want eighty. The table
# that lists them is now selectable, and what follows from a SUBSET is the
# part worth pinning: the Model Name declares only those, renumbered from
# zero, and the weights still predict all eighty -- so it is a seed.

def _importer() -> str:
    return (WEB / "components" / "Track2ImportCheckpointPanel.tsx").read_text(
        encoding="utf-8")


def test_each_class_row_has_a_tick_and_the_header_selects_all():
    importer = _importer()
    assert "setPickedClasses" in importer
    # The header box reports a partial selection rather than guessing.
    assert "indeterminate" in importer


def test_a_subset_is_renumbered_from_zero():
    """A Model Name's indices are its own. Keeping COCO's numbering would
    declare a taxonomy with holes -- {0: person, 7: truck} -- and every
    label file written against it would have to know 1..6 are missing."""
    importer = _importer()
    block = importer.split("const found: Record<string, string>")[1][:400]
    assert "pickedClasses.includes(index)" in block
    assert "String(position)" in block, "renumbered, not the original index"


def test_a_subset_is_sliced_rather_than_registered_as_a_seed():
    """It WAS a seed: the registry said three classes and the weights
    still predicted eighty, with only a training run to reconcile them.

    That is the right answer when the subset is a starting point for new
    data, and the wrong one when the weights are already good and the
    extra classes are simply unwanted -- then there is nothing to learn,
    and an hour of GPU would reproduce what the file already does.

    So the head is cut instead, and the version is an ordinary one."""
    importer = _importer()
    assert "const subsetChosen" in importer
    # The picked ORIGINAL indices go to the server, which does the cut.
    assert "subsetChosen ? pickedClasses.map(Number)" in importer
    # And a subset no longer drags in the mismatch exception.
    seed = importer.split("const seedForChosen")[1][:220]
    assert "subsetChosen" not in seed
    assert "not a seed" in importer


def test_registering_nothing_is_refused_and_says_why():
    importer = _importer()
    assert "pickedClasses.length === 0" in importer
    assert "Nothing selected" in importer


def test_the_selection_resets_when_a_different_file_is_read():
    """Three sources feed this card -- upload, bucket, stock release -- so
    the reset lives on the inspection, not beside each one."""
    importer = _importer()
    # The effect that owns the reset, found by its dependency rather than
    # by one exact line: it reads the inspection's classes and runs
    # whenever the inspection changes.
    block = importer.split("Object.keys(inspection?.classes")[1][:600]
    assert "setPickedClasses" in block
    assert "[inspection]" in block


def test_a_restored_selection_survives_the_reset():
    """Reloading re-reads the checkpoint, which produces a new inspection
    -- and the reset above would then tick every class, losing the very
    selection the reload was meant to bring back.

    So a restore carries its ticks in a ref that the reset consumes once.
    Reported from the page: classes picked, file saved, and the card came
    back with everything ticked and Read classes waiting to be pressed.
    """
    importer = _importer()
    assert "restoredPicks" in importer
    block = importer.split("Object.keys(inspection?.classes")[1][:600]
    assert "restoredPicks.current" in block, (
        "the reset must consult the restored ticks before overwriting them"
    )
    # Cleared as it is consumed, or the next genuine file read would also
    # be treated as a restore.
    assert "restoredPicks.current = null" in block


def test_a_restored_tick_is_intersected_with_what_the_file_now_has():
    """A checkpoint can be replaced at the same key by one with different
    classes. Ticking an index it no longer has would train against a
    class that is not there."""
    importer = _importer()
    block = importer.split("Object.keys(inspection?.classes")[1][:600]
    assert "available.includes" in block


# --- cutting the head when you type your own class names ----------------
#
# The same outcome as picking from the table, reached the other way. What
# decides it is whether the names you typed are classes the file ALREADY
# detects: those can be kept, and anything else cannot. No amount of head
# surgery turns a COCO checkpoint into a forklift detector -- it has no
# forklift output to keep.

def _detectors() -> str:
    return (WEB / "components" / "Track2DetectorsPanel.tsx").read_text(encoding="utf-8")


def test_typed_names_are_matched_against_the_files_own_classes():
    panel = _detectors()
    assert "const matchedIndices" in panel
    assert "fileLower.indexOf(name.toLowerCase())" in panel, "case-insensitive"


def test_the_typed_order_is_what_is_sent_not_numeric_order():
    """Your first line is class 0. Sending sorted indices would file every
    label under the wrong name, and nothing would error."""
    panel = _detectors()
    assert "canCutHead ? matchedIndices : []" in panel
    assert "sort" not in panel.split("canCutHead ? matchedIndices")[0][-300:]


def test_a_name_the_file_does_not_have_blocks_the_cut():
    panel = _detectors()
    assert "unmatchedNames" in panel
    assert "everyNameInFile" in panel
    # And says so, rather than leaving "seed" to be inferred.
    assert "never learned" in panel


def test_cutting_does_not_also_claim_the_mismatch_exception():
    """A cut head matches by construction. Passing the exception too would
    wave a genuine mismatch through alongside it."""
    panel = _detectors()
    assert "!canCutHead," in panel


def test_asking_for_exactly_what_is_there_cuts_nothing():
    """A byte-copy under a new key is a second artefact saying the same
    thing."""
    panel = _detectors()
    assert "sliceWouldChangeIt" in panel
    assert "matchedIndices.some((original, position) => original !== position)" in panel
