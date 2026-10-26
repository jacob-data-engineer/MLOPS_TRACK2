"""The rest of Ultralytics' training arguments, carried end to end.

The chain this pins, because every link was a place it could die
silently:

    card -> default_training.hyperparams -> TrainingParams
         -> Hyperparams pipeline parameter -> train.py --hyperparams
         -> model.train(**extra)

ONE blob, not a parameter per knob. The alternative meant a SageMaker
pipeline parameter, a coordinator change and a training-image rebuild
every time somebody wanted another dial -- three releases across two
stacks for one number.

OVERRIDES ONLY. A key absent means "whatever Ultralytics does by
default", so a run that sets none of this trains exactly as it did
before any of this existed.
"""
from __future__ import annotations

import base64
import json

import pytest

from ppe_pipeline_api import track2_shared as t2


# --- the blob ------------------------------------------------------------------

def test_nothing_set_sends_nothing():
    """The guarantee the whole design rests on: an untouched card changes
    no run."""
    assert t2.encode_hyperparams(None) == ""
    assert t2.encode_hyperparams({}) == ""


def test_blank_values_are_dropped_not_sent_as_blanks():
    """A field the operator left alone means "default", not "set it to
    empty" -- and `lr0: ""` would be a different statement entirely."""
    assert t2.encode_hyperparams({"lr0": "", "optimizer": None}) == ""


def test_what_is_set_survives_the_encoding():
    blob = t2.encode_hyperparams({"optimizer": "AdamW", "lr0": 0.001})
    assert json.loads(base64.b64decode(blob).decode()) == {
        "optimizer": "AdamW", "lr0": 0.001}


def test_it_is_encoded_rather_than_sent_as_raw_json():
    """It travels as a SageMaker pipeline parameter and then as a command
    line argument. A JSON object full of quotes and braces through those
    layers is a quoting problem that presents as the parameter being
    silently ignored."""
    blob = t2.encode_hyperparams({"optimizer": "AdamW"})
    assert "{" not in blob and '"' not in blob and " " not in blob


# --- the pipeline has somewhere to put it ---------------------------------------

def test_the_open_pipeline_declares_the_parameter():
    """The coordinator can only pass parameters the pipeline DECLARES, so
    a value with no parameter to land in is dropped without a word."""
    import pathlib

    definition = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "model_training" / "custom_training" / "pipeline_open.json")
        .read_text(encoding="utf-8"))
    names = [p["Name"] for p in definition["Parameters"]]
    assert "Hyperparams" in names
    assert "Patience" in names, "early stopping was already wired; do not lose it"

    train = [s for s in definition["Steps"] if s["Name"] == "TrainYolo"][0]
    hp = train["Arguments"]["HyperParameters"]
    assert hp["hyperparams"] == {"Get": "Parameters.Hyperparams"}
    # train.py's own flag name, or the value arrives under a key it does
    # not read.
    assert "patience" in hp


def test_the_blank_default_means_an_unchanged_run():
    import pathlib

    definition = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "model_training" / "custom_training" / "pipeline_open.json")
        .read_text(encoding="utf-8"))
    param = [p for p in definition["Parameters"] if p["Name"] == "Hyperparams"][0]
    assert param["DefaultValue"] == ""


# --- the request carries it -----------------------------------------------------

def test_training_params_accepts_it_and_defaults_to_empty():
    from ppe_pipeline_api.routers.track2_models import TrainingParams

    params = TrainingParams(generation="11", size="s", dataset_folder="d")
    assert params.hyperparams == {}
    with_it = TrainingParams(generation="11", size="s", dataset_folder="d",
                             hyperparams={"optimizer": "AdamW"})
    assert with_it.hyperparams == {"optimizer": "AdamW"}


def test_a_queued_run_carries_what_it_was_queued_with():
    """The card may have been changed since. An entry should run what it
    was queued with, like the snapshot it already captures."""
    import inspect

    src = inspect.getsource(t2.try_advance_queue)
    assert 'hyperparams=entry.get("hyperparams")' in src


# --- the trainer refuses what it cannot honour ----------------------------------

@pytest.fixture(scope="module")
def trainer():
    """train.py, loaded directly -- it is not an importable package."""
    import importlib.util
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "model_training" / "training-container" / "train.py")
    spec = importlib.util.spec_from_file_location("track2_train", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


def test_the_trainer_takes_nothing_as_nothing(trainer):
    assert trainer._validated_hyperparams(None) == {}
    assert trainer._validated_hyperparams("") == {}


def test_the_trainer_accepts_both_encoded_and_plain(trainer):
    """Encoded from the app; plain so a run can be driven by hand from the
    console without encoding anything first."""
    blob = base64.b64encode(json.dumps({"lr0": 0.001}).encode()).decode()
    assert trainer._validated_hyperparams(blob) == {"lr0": 0.001}
    assert trainer._validated_hyperparams('{"cos_lr": true}') == {"cos_lr": True}


def test_a_misspelled_key_fails_the_job_rather_than_being_ignored(trainer):
    """The failure this exists to prevent: "optimiser" dropped in silence,
    the run training on defaults, and the audit recording the operator's
    intent. Two different runs with one record."""
    with pytest.raises(SystemExit, match="has no optimiser"):
        trainer._validated_hyperparams('{"optimiser": "AdamW"}')


def test_a_parameter_the_pipeline_owns_is_refused(trainer):
    """Letting the blob set epochs would mean the audit row and the run
    disagreeing about what was trained."""
    for key in ("epochs", "data", "seed", "patience"):
        with pytest.raises(SystemExit, match="decided by the pipeline"):
            trainer._validated_hyperparams(json.dumps({key: 1}))


def test_malformed_input_is_refused_clearly(trainer):
    with pytest.raises(SystemExit, match="neither JSON nor base64"):
        trainer._validated_hyperparams("not-json-or-base64!!")
    with pytest.raises(SystemExit, match="must be a JSON object"):
        trainer._validated_hyperparams("[1, 2]")


def test_every_field_the_card_offers_is_one_ultralytics_knows():
    """The card's list and the installed Ultralytics drifting apart is how
    a field comes to do nothing. Checked against the real config rather
    than a copy of it."""
    pytest.importorskip("ultralytics")
    import pathlib
    import re

    from ultralytics.cfg import DEFAULT_CFG_DICT

    panel = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "src" / "components" / "Track2TrainingAdvanced.tsx"
             ).read_text(encoding="utf-8")
    keys = re.findall(r'\{ key: "([a-z0-9_]+)"', panel)
    assert len(keys) > 25, "the parameter table should have been found"
    unknown = sorted(k for k in keys if k not in DEFAULT_CFG_DICT)
    assert unknown == [], f"the card offers {unknown}, which Ultralytics has not"


def test_every_default_shown_is_the_default_ultralytics_actually_uses():
    """The card prints a default beside each field, and those are typed
    into a table by hand.

    Checking only that the KEYS exist -- which is all the test above did
    -- leaves the numbers free to drift: an Ultralytics upgrade that
    changes `mosaic` from 1.0 to 0.8 would leave the card quietly
    advertising the old one, and a blank field sends nothing, so the run
    would train on 0.8 while the screen said 1.0. Nothing would error.
    """
    pytest.importorskip("ultralytics")
    import pathlib
    import re

    from ultralytics.cfg import DEFAULT_CFG_DICT

    panel = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "src" / "components" / "Track2TrainingAdvanced.tsx"
             ).read_text(encoding="utf-8")
    rows = re.findall(r'\{ key: "([a-z0-9_]+)",[^}]*?fallback: "([^"]*)"',
                      panel, re.S)
    assert len(rows) > 25, "the parameter table should have been found"

    wrong = []
    for key, shown in rows:
        actual = DEFAULT_CFG_DICT[key]
        actual_text = "none" if actual is None else str(actual).lower()
        shown_text = shown.strip().lower()
        if actual_text == shown_text:
            continue
        # 7.5 and 7.50 are the same number; "1.0" and "1" are not worth
        # failing over either.
        if ("." in actual_text and "." in shown_text
                and actual_text.rstrip("0").rstrip(".")
                == shown_text.rstrip("0").rstrip(".")):
            continue
        wrong.append(f"{key}: card says {shown!r}, ultralytics says {actual!r}")
    assert not wrong, (
        "these fields advertise a default Ultralytics does not use:\n  "
        + "\n  ".join(wrong)
    )


def test_the_defaults_do_not_depend_on_the_generation():
    """The card shows ONE default per field, and the page trains both
    YOLO11 and YOLO26. That is only honest if the model contributes no
    training hyperparameters -- which it does not: a model YAML carries
    architecture, and the hyperparameters come from one global config.
    """
    pytest.importorskip("ultralytics")
    import pathlib
    import re

    import yaml as _yaml
    from ultralytics.utils import ROOT

    panel = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "src" / "components" / "Track2TrainingAdvanced.tsx"
             ).read_text(encoding="utf-8")
    keys = set(re.findall(r'\{ key: "([a-z0-9_]+)"', panel))

    checked = 0
    for name in ("yolo11.yaml", "yolo26.yaml"):
        found = list(ROOT.rglob(name))
        if not found:
            continue
        checked += 1
        document = _yaml.safe_load(found[0].read_text(encoding="utf-8"))
        overlap = sorted(set(document) & keys)
        assert not overlap, (
            f"{name} sets {overlap}, so the card's single default is wrong "
            f"for at least one generation"
        )
    assert checked, "neither model YAML was found -- has the layout changed?"


# --- the switch that decides whether any of this is sent -------------------------

def test_the_card_offers_a_switch_and_it_defaults_to_off():
    """Every settings file written before the switch existed has no such
    key, and those runs must go on behaving exactly as they did."""
    import pathlib

    panel = (pathlib.Path(__file__).resolve().parents[1]
             / "web" / "src" / "components" / "Track2DetectorsPanel.tsx"
             ).read_text(encoding="utf-8")
    assert "const [hyperparamsOn, setHyperparamsOn] = useState(false)" in panel
    assert "hyperparams_enabled: hyperparamsOn" in panel


def test_off_sends_nothing_and_on_sends_everything():
    """The whole contract of the tick, read off the helper the card and
    the form both call."""
    import pathlib
    import re

    source = (pathlib.Path(__file__).resolve().parents[1]
              / "web" / "src" / "components" / "Track2TrainingAdvanced.tsx"
              ).read_text(encoding="utf-8")
    body = source.split("export function effectiveHyperparams")[1][:900]
    assert "if (!enabled) return {};" in body, "unticked must send nothing"
    assert "ULTRALYTICS_DEFAULTS" in body, "ticked must fill the blanks in"
    # `none` is how the table writes "not set" for freeze; sending the
    # string would be a value.
    assert '"none"' in body


def test_the_unattended_retrain_carries_them_too():
    """The manual form carried hyperparameters and the schedule did not,
    so ticking the box changed what a hand-started run trained with and
    not what the unattended one did."""
    from ppe_pipeline_api import retrain_schedule as rs

    assert "hyperparams" in rs.DEFAULTS, "the plan has nowhere to put them"


def test_the_schedule_only_sends_them_when_the_box_is_ticked():
    import inspect

    from ppe_pipeline_api import retrain_schedule as rs

    source = inspect.getsource(rs)
    assert 'shape.get("hyperparams_enabled")' in source, (
        "the sparse overrides alone cannot say whether a blank field means "
        "'the default, stated' or 'say nothing'"
    )


def test_both_submit_paths_send_the_same_run():
    """A schedule that started its run directly used to drop patience and
    the hyperparameters, while the identical schedule that queued it kept
    them. `patience` defaulting to 0 is what hid it."""
    import inspect

    from ppe_pipeline_api import retrain_schedule as rs

    source = inspect.getsource(rs)
    # To the closing paren of that call, rather than a fixed window: the
    # argument list carries a long comment and a character count silently
    # stopped short of the two arguments being checked for.
    after = source.split("start_custom_training_run(")[1]
    call = after[:after.index("\n        )")]
    assert "patience=" in call, "the direct-start path drops patience"
    assert "hyperparams=" in call, "the direct-start path drops hyperparameters"


def test_patience_zero_reaches_the_trainer_as_zero(trainer):
    """0 means NEVER stop early, and Ultralytics implements exactly that:
    `self.patience = patience or float("inf")`.

    The trainer used to withhold 0 on the belief that it meant "stop at
    the first epoch that does not improve". Withholding it handed the run
    Ultralytics' own default of 100, so "Patience 0" quietly meant "stop
    after 100 epochs without improvement" -- invisible on a 50-epoch run,
    which cannot reach 100, and live on anything longer.
    """
    import inspect

    source = inspect.getsource(trainer)
    assert 'extra = {"patience": args.patience}' in source, (
        "patience must be passed unconditionally; 0 is a meaningful value"
    )
    assert 'if args.patience > 0 else {}' not in source


def test_zero_really_does_mean_no_limit_in_this_ultralytics():
    """Pinned against the library rather than trusted, because the whole
    fix rests on it."""
    pytest.importorskip("ultralytics")
    from ultralytics.utils.torch_utils import EarlyStopping

    assert EarlyStopping(0).patience == float("inf")
    assert EarlyStopping(10).patience == 10
