import io
import pickle
import zipfile
import pytest
from ppe_pipeline_api.data import pt_view

class Model:
    def __init__(self):
        self.names={0:'helmet',1:'person'}
        self.yaml={'nc':2}

def checkpoint(value):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        z.writestr('model/data.pkl',pickle.dumps(value,protocol=4))
    return out

def test_classes_and_training_metadata():
    got=pt_view.inspect(checkpoint({'model':Model(),'epoch':10,'train_args':{'imgsz':640}}),'best.pt')
    assert got['classes']==[{'id':0,'name':'helmet'},{'id':1,'name':'person'}]
    assert got['metadata']['epoch']==10
    assert got['metadata']['train_args']=={'imgsz':640}

def test_unknown_reducer_never_executes(tmp_path):
    target=tmp_path/'executed'
    class Malicious:
        def __reduce__(self):
            return (eval,(f"__import__('pathlib').Path({str(target)!r}).write_text('bad')",))
    result=pt_view.inspect(checkpoint({'model':Malicious()}),'model.pt')
    assert not target.exists()
    assert result['classes']==[]

def test_weights_only_does_not_invent_classes():
    assert 'not recorded' in pt_view.inspect(checkpoint({'state_dict':{}}),'model.pt')['notice']

@pytest.mark.parametrize('body,name',[(b'','model.pt'),(b'bad','model.pt'),(b'bad','model.exe')])
def test_invalid_upload(body,name):
    with pytest.raises(ValueError):
        pt_view.inspect(io.BytesIO(body),name)

def test_registered_archive_is_read_without_extracting(monkeypatch):
    import tarfile
    from ppe_pipeline_api.routers import track2_pt_view as route
    raw=checkpoint({'model':Model()}).getvalue()
    archive=io.BytesIO()
    with tarfile.open(fileobj=archive,mode='w:gz') as tar:
        info=tarfile.TarInfo('../../best.pt');info.size=len(raw)
        tar.addfile(info,io.BytesIO(raw))
    monkeypatch.setattr(route.t2.sm,'describe_model_package',lambda **kw:{'InferenceSpecification':{'Containers':[{'ModelDataUrl':'s3://bucket/model.tar.gz'}]}})
    monkeypatch.setattr(route.t2.s3,'get_object',lambda **kw:{'Body':io.BytesIO(archive.getvalue()),'ContentLength':len(archive.getvalue())})
    got=route.registered(route.Registered(arn='arn:model'))
    assert len(got['classes'])==2
    assert got['registry']['arn']=='arn:model'
