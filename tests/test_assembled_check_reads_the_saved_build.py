"""Step 6's class check, on the assembled source, reads what WILL be built.

Reported 2026-11-06: a six-class vendor run was blocked with "The dataset
has no labels for gloves, person, working_boots, man_cage, safe_area ...
It declares helmet", with the golden set ticked or not. The check was
reading `dataset_assemble.built.folder` -- on production `golden_set`,
whose yaml the Assemble card wrote on 26 October with the one-class
Helmet Model Name. Add to Queue never trains that file: it reruns the
chain, which rewrites the yaml from the card's SAVED settings
(`retrain_schedule.assemble_plan`). So the check compared the run against
a file the run does not use.
"""
from pathlib import Path

PAGE = Path(__file__).resolve().parents[1] / "web/src/routes/Track2ModelsPage.tsx"
SRC = PAGE.read_text(encoding="utf-8")


def test_assembled_source_compares_against_the_saved_build():
    assert "readAssembleSaved(status.config.dataset_assemble)" in SRC
    # The comparison runs on the derived classes, not the folder's yaml.
    assert "const checkedClasses" in SRC
    assert "assembledBuild\n    ? assembledBuild.classes" in SRC
    assert "detectorNames.filter((name) => !datasetNames.includes(name))" in SRC
    assert "chosenDetector && datasetClasses\n" not in SRC


def test_mirrors_assemble_plan_label_classes_rule():
    # Filter ON: label classes are not read (assemble_plan clears them).
    assert "assembleSaved.dropUndeclared ? {} : assembleSaved.labelClasses" in SRC
    # Label classes win only when they contain every Model Name class.
    assert ".every((n) => labelledNames.includes(n))" in SRC


def test_the_refusal_does_not_say_no_labels():
    # Class names live only in data.yaml; no label file is opened here.
    assert "has no labels for" not in SRC
