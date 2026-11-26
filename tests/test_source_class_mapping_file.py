import copy
import json
from ppe_pipeline_api import pull_control as pc


def test_source_edits_roundtrip_in_named_file(monkeypatch, tmp_path):
    mapping = {
        's3://bucket/a': {'source_classes': {'0':'helmets'}, 'ids': {'0':'0'}, 'target_classes': {'0':'helmet'}},
        's3://bucket/b': {'source_classes': {'0':'hardhat'}, 'ids': {'0':'0'}, 'target_classes': {'0':'helmet'}},
    }
    live = {'root':'r', 'source_class_mappings': copy.deepcopy(mapping)}
    monkeypatch.setattr(pc, 'SETTINGS_DIR', str(tmp_path))
    monkeypatch.setattr(pc, 'read_config', lambda: copy.deepcopy(live))
    def write(patch):
        live.update(copy.deepcopy(patch))
        return copy.deepcopy(live)
    monkeypatch.setattr(pc, 'write_config', write)
    monkeypatch.setattr(pc, 'pull_in_use', lambda body: False)
    pc.save_settings_file(name='helmet_sit', scope='dp')
    live['source_class_mappings'] = {}
    pc.load_settings_file(name='helmet_sit_DP', scope='dp')
    assert live['source_class_mappings'] == mapping
    (tmp_path/'old_DP.json').write_text(json.dumps({'config':{'root':'old'}}))
    pc.load_settings_file(name='old_DP', scope='dp')
    assert live['source_class_mappings'] == {}
