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


def test_selected_categories_do_not_leak_raw_records(monkeypatch):
    metadata=sample()
    metadata.update(captured_at='now',unavailable=[],records={'private_unselected':123})
    metadata['groups']={'Evaluation':{'metric':0.8},'Training settings':{'epochs':10},'File provenance':{'inventory':'manifest.csv'}}
    selected=cm.select_groups(metadata,['Evaluation'])
    assert selected['groups']=={'Evaluation':{'metric':0.8}}
    assert 'records' not in selected
    assert 'epochs' not in json.dumps(selected)
    with pytest.raises(ValueError,match='Unknown metadata'): cm.select_groups(metadata,['Unknown'])
    original=io.BytesIO()
    with zipfile.ZipFile(original,'w') as z: z.writestr('archive/data.pkl',b'opaque')
    data=cm.embed(original.getvalue(),selected,None)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert not any('manifest.csv' in name for name in z.namelist())
    assert 'file_manifest' not in cm.inspect_file(io.BytesIO(data),'model.pt')


def test_direct_model_export_embeds_only_selected_groups(monkeypatch):
    original=io.BytesIO()
    with zipfile.ZipFile(original,'w') as z: z.writestr('archive/data.pkl',b'opaque')
    class S3:
        def __init__(self): self.written=[]
        def head_object(self,**kw): return {}
        def put_object(self,**kw): self.written.append(kw)
        def generate_presigned_url(self,*a,**kw): return 'signed'
    s3=S3();monkeypatch.setattr(cv.storage,'s3',s3)
    monkeypatch.setattr(cv,'_rows',lambda *a:[{'run_number':54,'record':{'weights_uri':'s3://source/model.pt','parameters':{'epochs':10},'metrics':{'mAP50':0.8}}}])
    monkeypatch.setattr(cv,'_fetch',lambda uri:original.getvalue())
    monkeypatch.setattr(cv,'_export_bucket',lambda:('local-bucket','exports'))
    for target in ('local','s3'):
        result=cv.export_run_weights('run54',destination=target,bucket='chosen-bucket',prefix='folder',metadata_groups=['Evaluation'])
        assert ('url' if target=='local' else 'uri') in result
        loaded=cm.inspect_file(io.BytesIO(s3.written[-1]['Body']),'model.pt')
        assert list(loaded['groups'])==['Evaluation']
        assert 'records' not in loaded and 'file_manifest' not in loaded
    count=len(s3.written)
    assert cv.export_run_weights('run54',metadata_groups=[])=={'url':'signed'}
    assert len(s3.written)==count
