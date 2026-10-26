"""Advanced training parameters reach Ultralytics as real types.

The card used to send every value as a string. With the box ticked and
nothing edited that was `lr0: "0.01"`, `cos_lr: "false"`, `freeze: "10"`
-- and the trainer passed them to `model.train` unchanged. Ultralytics'
`check_cfg` raises TypeError on a string `lr0`, and quietly ignores a
string `freeze` (it accepts an int or a list only), so the run either
died or froze nothing while the audit recorded otherwise.

Now: ONE schema (`ppe_pipeline_api.hyperparams.SCHEMA`) types and
checks every value at the shared boundary every submit path crosses,
before a paid job exists; the frontend emits numbers and booleans; and
the trainer re-types legacy strings defensively with its own mirror.
"""
from __future__ import annotations

import base64
import importlib.util
import inspect
import json
import pathlib
import re

import pytest

from ppe_pipeline_api import hyperparams as hp
from ppe_pipeline_api import track2_shared as t2

ROOT = pathlib.Path(__file__).resolve().parents[1]
PANEL = ROOT / "web" / "src" / "components" / "Track2TrainingAdvanced.tsx"


@pytest.fixture(scope="module")
def trainer():
    path = ROOT / "model_training" / "training-container" / "train.py"
    spec = importlib.util.spec_from_file_location("track2_train_types", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except SystemExit:
        pass
    return module


def _card_defaults() -> dict[str, str]:
    """The card's own table, read off the source: key -> shown default."""
    rows = re.findall(r'\{ key: "([a-z0-9_]+)",[^}]*?fallback: "([^"]*)"',
                      PANEL.read_text(encoding="utf-8"), re.S)
    assert len(rows) > 25
    return dict(rows)


def _legacy_frontend_payload() -> dict[str, str]:
    """EXACTLY what the old `effectiveHyperparams` produced with the box
    ticked and nothing edited: every default, as a string, `none` skipped.
    The same shape older settings files and queue entries still hold."""
    return {k: v for k, v in _card_defaults().items() if v and v != "none"}


def _decode(blob: str) -> dict:
    return json.loads(base64.b64decode(blob).decode())


# --- the frontend-shaped payload, end to end -----------------------------------

def test_the_ticked_untouched_card_reaches_the_trainer_typed(trainer):
    blob = t2.encode_hyperparams(_legacy_frontend_payload())
    sent = _decode(blob)
    kept = trainer._validated_hyperparams(blob)

    for typed in (sent, kept):
        assert type(typed["lr0"]) is float and typed["lr0"] == 0.01
        assert type(typed["warmup_epochs"]) is float
        assert typed["amp"] is True
        assert typed["cos_lr"] is False
        assert typed["rect"] is False
        assert type(typed["close_mosaic"]) is int and typed["close_mosaic"] == 10
        assert type(typed["nbs"]) is int and typed["nbs"] == 64
        assert typed["optimizer"] == "auto"
        assert typed["auto_augment"] == "randaugment"
        # "none" is the card's way of writing "not set".
        assert "freeze" not in typed


def test_ultralytics_own_validation_accepts_the_result(trainer):
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import check_cfg

    blob = t2.encode_hyperparams({**_legacy_frontend_payload(), "freeze": "10"})
    for typed in (_decode(blob), trainer._validated_hyperparams(blob)):
        check_cfg(dict(typed), hard=True)          # raises on a wrong type


def test_the_old_strings_really_were_refused_by_ultralytics():
    """Pinned so the test above is proving something: the raw strings
    the card sent are what Ultralytics rejects."""
    pytest.importorskip("ultralytics")
    from ultralytics.cfg import check_cfg

    with pytest.raises(TypeError):
        check_cfg({"lr0": "0.01"}, hard=True)


@pytest.mark.parametrize("given, expected", [
    ("10", 10), (10, 10), ("0", 0),
    ("0,1,2", [0, 1, 2]), ("[3, 4]", [3, 4]), ([0, 1], [0, 1]),
])
def test_freeze_takes_a_count_or_a_list(trainer, given, expected):
    blob = t2.encode_hyperparams({"freeze": given})
    assert _decode(blob)["freeze"] == expected
    assert trainer._validated_hyperparams(blob)["freeze"] == expected
    assert trainer._validated_hyperparams(json.dumps({"freeze": given}))["freeze"] == expected


@pytest.mark.parametrize("blank", ["none", "None", "", "  ", None])
def test_freeze_none_means_not_set(blank):
    assert t2.encode_hyperparams({"freeze": blank}) == ""


@pytest.mark.parametrize("token, expected", [
    ("true", True), ("TRUE", True), ("1", True), ("yes", True), (True, True),
    ("false", False), ("False", False), ("0", False), ("no", False), (False, False),
])
def test_booleans_by_exact_token_never_bool_of_a_string(trainer, token, expected):
    assert hp.normalize_hyperparams({"cos_lr": token}) == {"cos_lr": expected}
    assert trainer._validated_hyperparams(json.dumps({"cos_lr": token})) == {"cos_lr": expected}


# --- refused before anything is submitted --------------------------------------

BAD = [
    ("lr0", "abc"), ("lr0", "nan"), ("lr0", "inf"), ("lr0", "2"),
    ("amp", "maybe"), ("cos_lr", "2"),
    ("close_mosaic", "2.5"), ("nbs", "0"),
    ("freeze", "-1"), ("freeze", "1,-2"), ("freeze", "1.5"),
    ("optimizer", "Adagrad"), ("fraction", "0"),
]


@pytest.mark.parametrize("key, value", BAD)
def test_an_invalid_value_is_refused_naming_the_key_and_value(key, value):
    with pytest.raises(hp.HyperparamError) as caught:
        t2.encode_hyperparams({key: value})
    assert key in str(caught.value) and repr(value) in str(caught.value)


@pytest.mark.parametrize("key, value", BAD)
def test_the_trainer_refuses_them_too(trainer, key, value):
    with pytest.raises(SystemExit, match=key):
        trainer._validated_hyperparams(json.dumps({key: value}))


def test_the_api_refuses_them_at_the_request(monkeypatch):
    """A 400 with a plain-string detail naming the key, on all three
    routes, before anything is checked, queued or submitted."""
    from fastapi import HTTPException

    from ppe_pipeline_api.routers import track2_models as tm

    def boom(*_a, **_k):
        raise AssertionError("got past the hyperparameter check")

    monkeypatch.setattr(t2, "get_training_progress", boom)
    monkeypatch.setattr(t2, "validate_training_params", boom)
    routes = (tm.start_training, tm.start_custom_training, tm.add_to_queue)
    for key, value in [("lr0", "abc"), ("amp", "maybe"),
                       ("close_mosaic", "2.5"), ("freeze", "-1")]:
        for route in routes:
            payload = tm.TrainingParams(generation="11", size="s",
                                        dataset_folder="d",
                                        hyperparams={key: value})
            with pytest.raises(HTTPException) as caught:
                route(payload, session=None)
            assert caught.value.status_code == 400
            assert isinstance(caught.value.detail, str)
            assert key in caught.value.detail and repr(value) in caught.value.detail


def test_the_api_types_legacy_strings_before_queueing(monkeypatch):
    from ppe_pipeline_api.routers import track2_models as tm

    captured = {}
    monkeypatch.setattr(t2, "validate_training_params",
                        lambda *a, **k: (1, 640, 16, 42))
    monkeypatch.setattr(tm, "_stamp_snapshot", lambda entry: None)
    monkeypatch.setattr(tm, "_freeze_for_entry", lambda entry: None)
    monkeypatch.setattr(t2, "queue_add", lambda entry: captured.update(entry))
    payload = tm.TrainingParams(
        generation="11", size="s", dataset_folder="d",
        hyperparams={"lr0": "0.01", "cos_lr": "false", "freeze": "10",
                     "close_mosaic": "10", "amp": "true"})
    tm.add_to_queue(payload, session=None)
    assert captured["hyperparams"] == {
        "lr0": 0.01, "cos_lr": False, "freeze": 10, "close_mosaic": 10, "amp": True}


def test_a_reserved_key_is_refused_before_submission():
    with pytest.raises(hp.HyperparamError, match="decided by the pipeline"):
        t2.encode_hyperparams({"epochs": 5})


def test_a_direct_start_refuses_before_touching_aws(monkeypatch):
    """Both start functions normalise FIRST, so a bad value never reaches
    a filtered-copy, a CodeBuild start or a pipeline execution."""
    def boom(*_a, **_k):
        raise AssertionError("reached AWS with an invalid hyperparameter")

    monkeypatch.setattr(t2, "validate_training_params", boom)
    monkeypatch.setattr(t2, "get_detector", boom)
    with pytest.raises(hp.HyperparamError, match="lr0"):
        t2.start_custom_training_run(
            dataset_folder="d", eval_dataset_folder="", epochs=1, img_size=640,
            batch=1, seed=1, notes="", detector_slug="x",
            hyperparams={"lr0": "abc"})
    with pytest.raises(hp.HyperparamError, match="lr0"):
        t2.start_training_run("11", "s", 1, "d", "", "", hyperparams={"lr0": "abc"})


# --- nothing sent when the box is off ------------------------------------------

def test_disabled_sends_nothing():
    assert t2.encode_hyperparams({}) == ""
    assert t2.encode_hyperparams(None) == ""
    body = PANEL.read_text(encoding="utf-8").split(
        "export function effectiveHyperparams")[1][:1500]
    assert "if (!enabled) return {};" in body


# --- saved legacy settings, queued and scheduled runs ---------------------------

def test_saved_legacy_string_settings_are_normalised_on_the_schedule_path():
    """`default_training.hyperparams` was saved as strings. The scheduled
    chain reads it straight into its plan; the shared boundary types it."""
    from ppe_pipeline_api import retrain_schedule as rs

    plan = rs.settings({
        "default_training": {
            "hyperparams": {"lr0": "0.005", "cos_lr": "true",
                            "close_mosaic": "5", "freeze": "3"},
            "hyperparams_enabled": True,
        },
    })
    sent = _decode(t2.encode_hyperparams(plan["hyperparams"]))
    assert sent == {"lr0": 0.005, "cos_lr": True, "close_mosaic": 5, "freeze": 3}


def test_manual_queued_and_scheduled_paths_share_one_normalisation():
    """Every path ends in one of the two start functions, and both type
    the values on their first line -- so a queue entry or a chain plan
    holding strings is typed exactly as the manual form is."""
    for fn in (t2.start_training_run, t2.start_custom_training_run):
        body = inspect.getsource(fn).split('"""', 2)[2]
        first = next(line.strip() for line in body.splitlines()
                     if line.strip() and not line.strip().startswith("#"))
        assert first == "hyperparams = normalize_hyperparams(hyperparams)", fn.__name__
    assert "normalize_hyperparams(" in inspect.getsource(t2.encode_hyperparams)
    # The queue replays what the entry holds through the custom start.
    assert 'hyperparams=entry.get("hyperparams")' in inspect.getsource(t2.try_advance_queue)


def test_a_queue_add_keeps_the_hyperparameters():
    """The card's queue-add built its entry without them, so a queued run
    trained on defaults while the form showed otherwise."""
    from ppe_pipeline_api.routers import track2_models as tm

    assert '"hyperparams": payload.hyperparams' in inspect.getsource(tm.add_to_queue)


# --- the trainer's mirror agrees with the app's schema ---------------------------

def test_the_trainer_mirror_matches_the_schema(trainer):
    assert trainer._HYPERPARAM_KINDS == {k: s["kind"] for k, s in hp.SCHEMA.items()}
    assert trainer.RESERVED_HYPERPARAMS == set(hp.RESERVED)


def test_the_schema_covers_every_field_the_card_offers():
    assert set(_card_defaults()) == set(hp.SCHEMA)


def test_every_default_shown_passes_the_schema():
    for key, shown in _card_defaults().items():
        hp.normalize_hyperparams({key: shown})


# --- the frontend emits typed values ---------------------------------------------

def test_the_card_emits_numbers_and_booleans_not_strings():
    source = PANEL.read_text(encoding="utf-8")
    body = source.split("export function effectiveHyperparams")[1][:2500]
    signature = body[:body.index("{")]
    assert "): Track2Hyperparams" in signature, "it must return typed values"
    assert 'kind === "bool"' in body or "typedValue(" in body
    helper = source.split("function typedValue")[1][:1500]
    assert 'raw.toLowerCase() === "true"' in helper
    assert "Number(" in helper
    assert 'kind: "freeze"' in source, "freeze needs its own kind: an int or a list"
