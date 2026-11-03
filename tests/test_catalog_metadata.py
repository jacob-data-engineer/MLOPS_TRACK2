import io
import json
import zipfile
import pytest
from ppe_pipeline_api.data import catalog_metadata as cm, catalog_view as cv


def sample():
    return {'schema':cm.SCHEMA,'label':'run-1','groups':{'Evaluation':{},'Training history':{}},'records':{},'notes':[]}


def test_embed_preserves_checkpoint_records_and_reads_without_pickle():
    source=io.BytesIO()
    with zipfile.ZipFile(source,'w') as z:
        z.writestr('archive/data.pkl',b'not executable by this viewer')
        z.writestr('archive/data/0',b'weights')
    original=source.getvalue()
    enriched=cm.embed(original,sample(),'path,sha256\na.jpg,123\n')
    with zipfile.ZipFile(io.BytesIO(original)) as before, zipfile.ZipFile(io.BytesIO(enriched)) as after:
        for name in before.namelist(): assert before.read(name)==after.read(name)
    result=cm.inspect_file(io.BytesIO(enriched),'model.pt')
    assert result['schema']==cm.SCHEMA
    assert 'a.jpg' in result['file_manifest']
    with pytest.raises(ValueError,match='already enriched'): cm.embed(enriched,sample(),'')


def test_metadata_parser_limits_and_redaction():
    value=sample();value['records']={'password':'hidden','nested':{'aws_access_key_id':'hidden','url':'https://host/path?X-Amz-Signature=hidden'}}
    result=cm.inspect_file(io.BytesIO(json.dumps(value).encode()),'metadata.json')
    assert 'hidden' not in json.dumps(result)
    with pytest.raises(ValueError): cm.inspect_file(io.BytesIO(b'{}'),'metadata.json')
    with pytest.raises(ValueError): cm.embed(b'legacy pickle',sample(),'')
    payload=io.BytesIO()
    with zipfile.ZipFile(payload,'w',zipfile.ZIP_DEFLATED) as z: z.writestr('metadata.json',b' '*(cm.MAX_JSON+1))
    with pytest.raises(ValueError,match='exceeds'): cm.inspect_file(io.BytesIO(payload.getvalue()),'x.zip')


def test_metadata_groups_preserve_zero_and_missing(monkeypatch):
    monkeypatch.setattr(cv,'_rows',lambda *a:[{'run_id':'r','run_number':58,'record':{'metrics':{'mAP50':0},'parameters':{'epochs':10},'password':'secret'}}])
    result=cm.describe('run','r')
    assert result['groups']['Evaluation']['metrics']['mAP50']==0
    assert 'Training history' in result['unavailable']
    assert result['records']['selected']['record']['password']=='[redacted]'


def test_real_torch_checkpoint_still_loads():
    torch=pytest.importorskip('torch')
    source=io.BytesIO()
    torch.save({'weights':torch.arange(6),'epoch':4},source)
    enriched=cm.embed(source.getvalue(),sample(),'path\na.jpg\n')
    loaded=torch.load(io.BytesIO(enriched),map_location='cpu',weights_only=True)
    assert loaded['epoch']==4
    assert torch.equal(loaded['weights'],torch.arange(6))
