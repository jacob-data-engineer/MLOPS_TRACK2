"""Keep the browser's parameter ranges aligned with server validation."""
import json
from pathlib import Path

from ppe_pipeline_api.training.hyperparams import SCHEMA


def test_browser_rules_match_backend_hyperparameters():
    rules = json.loads((Path(__file__).parents[1] / 'web/src/lib/trainingParameterRules.json').read_text())
    assert {key: rules[key] for key in SCHEMA} == json.loads(json.dumps(SCHEMA))
    assert rules['epochs'] == {'kind': 'int', 'min': 1, 'max': 1000}
    assert rules['img_size'] == {'kind': 'int', 'min': 32, 'max': 4096}
    assert rules['batch'] == {'kind': 'int', 'min': 1, 'max': 512}
