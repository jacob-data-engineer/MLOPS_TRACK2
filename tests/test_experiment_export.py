import copy
from types import SimpleNamespace
import pytest
from ppe_pipeline_api.data import experiment_export as ex, catalog_view as cv, export_estimate as ee
from ppe_pipeline_api.chain import training_variants as tv

@pytest.fixture
def experiment(monkeypatch):
    job={'id':'exp','run_number':75,'baseline_arn':'baseline','baseline':{'uri':'s3://b/base.pt'},
         'dataset_version_id':'source','working_dataset_version_id':'experiment-data',
         'variants':[{'index':0,'name':'One','run_id':'child','model_package_arn':'variant','state':'Succeeded','settings':{'epochs':1}}]}
    monkeypatch.setattr(tv,'read',lambda ident:copy.deepcopy(job))
    monkeypatch.setattr(tv,'public',lambda job:job)
    monkeypatch.setattr(cv,'_rows',lambda *a:[{'record':{'weights_uri':'s3://b/variant.pt'}}])
    return job

def test_baseline_export_uses_experiment_data_not_baseline_history(experiment):
    result=ex.context('exp','baseline')
    assert result['dataset']=='experiment-data'
    assert result['weights_uri']=='s3://b/base.pt'
    assert result['label']=='run-75-baseline'

def test_variant_export_is_pinned_to_selected_checkpoint(experiment):
    result=ex.context('exp','variant')
    assert result['weights_uri']=='s3://b/variant.pt'
    assert result['dataset']=='experiment-data'
    assert result['member']['settings']['epochs']==1

@pytest.mark.parametrize('arn',['unrelated','',None])
def test_foreign_model_cannot_export_with_experiment(experiment,arn):
    with pytest.raises(ValueError,match='Select the baseline'):
        ex.context('exp',arn)

def test_experiment_without_dataset_is_rejected(experiment):
    experiment['dataset_version_id']='';experiment['working_dataset_version_id']=''
    with pytest.raises(ValueError,match='no preserved dataset'):
        ex.context('exp','baseline')

def test_estimate_uses_selected_baseline_weights(experiment,monkeypatch):
    monkeypatch.setattr(cv,'_plan',lambda *a:([],{0:'helmet'},'run-75'))
    monkeypatch.setattr(ee.storage,'s3',SimpleNamespace(head_object=lambda **k:{'ContentLength':123,'ETag':'x'}))
    result=ee.estimate('experiment','exp',include_model=True,model_package_arn='baseline')
    assert result['model_bytes']==123 and result['unknown_files']==0

def test_shared_dataset_plan_recurses_only_to_recorded_version(experiment,monkeypatch):
    original=cv._plan
    def plan(kind,ident):
        if kind=='experiment': return original(kind,ident)
        assert (kind,ident)==('version','experiment-data')
        return [('images/train/a.jpg','s3://saved/a',{})],{0:'helmet'},'dataset'
    monkeypatch.setattr(cv,'_plan',plan)
    entries,names,label=plan('experiment','exp')
    assert label=='run-75' and entries[0][1]=='s3://saved/a'

@pytest.mark.parametrize("bundle", [False, True])
def test_export_build_receives_shared_files_and_selected_baseline(experiment,monkeypatch,bundle):
    import threading
    monkeypatch.setattr(cv.catalog,'ensure_schema_once',lambda:None)
    statements=[]
    cursor=SimpleNamespace(execute=lambda *args:statements.append(args))
    monkeypatch.setattr(cv.catalog,'connect',lambda:SimpleNamespace(cursor=lambda:cursor,commit=lambda:None,close=lambda:None))
    monkeypatch.setattr(cv,'_destination',lambda *a:('dest','out'))
    monkeypatch.setattr(cv,'_plan',lambda *a:([('images/train/a.jpg','s3://preserved/a',{})],{0:'helmet'},'run-75'))
    monkeypatch.setattr(cv.storage,'s3',SimpleNamespace(head_object=lambda **kw:{}))
    monkeypatch.setattr(cv,'_building',threading.BoundedSemaphore(2))
    launched=[]
    monkeypatch.setattr(cv.threading,'Thread',lambda **kw:SimpleNamespace(start=lambda:launched.append(kw)))
    cv.start_export('experiment','exp',include_model=True,model_package_arn='baseline', model_package_arns=['baseline','variant','baseline'] if bundle else None)
    args=launched[0]['args']
    assert args[3]==('run-75-experiment' if bundle else 'run-75-baseline')
    assert args[4][0][1]=='s3://preserved/a'
    assert args[4][1][1]=='s3://b/base.pt'
    assert len(args[4])==(3 if bundle else 2)
    if bundle: assert args[4][2][1]=='s3://b/variant.pt'
    assert statements[0][1][1]=='experiment-data'


def test_bundle_estimate_counts_shared_images_once(experiment,monkeypatch):
    monkeypatch.setattr(cv,'_plan',lambda *a:([('images/train/a.jpg','s3://b/a',{})],{0:'helmet'},'run-75'))
    monkeypatch.setattr(ee.storage,'s3',SimpleNamespace(head_object=lambda **k:{'ContentLength':100,'ETag':'x'}))
    result=ee.estimate('experiment','exp',include_model=True,model_package_arns=['baseline','variant','baseline'])
    assert result['files']==3
    assert result['source_bytes']==300
    assert result['model_bytes']==200


def test_bundle_rejects_foreign_model(experiment):
    with pytest.raises(ValueError,match='Select the baseline'):
        ex.bundle('exp',['baseline','unrelated'])
