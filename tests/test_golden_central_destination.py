from types import SimpleNamespace
import pytest
from ppe_pipeline_api.routers import track2_pull_control as router
from ppe_pipeline_api.data import combined_store as cs

def test_central_target_overrides_legacy_destination(monkeypatch):
 monkeypatch.setattr(router.pc,'read_config',lambda:{'central_store':{'bucket':'store','folder':'root/'}})
 payload=router.SplitRequest(use_central_store=True,target_bucket='old',target_folder='old')
 router._central_split_target(payload)
 assert (payload.target_bucket,payload.target_folder)==('store','root/golden')

def test_central_target_requires_configuration(monkeypatch):
 monkeypatch.setattr(router.pc,'read_config',lambda:{})
 with pytest.raises(ValueError,match='Central data store'):
  router._central_split_target(router.SplitRequest(use_central_store=True))

def test_combine_never_copies_golden_onto_itself(monkeypatch):
 def forbidden(**kwargs): raise AssertionError('Self-copy attempted')
 monkeypatch.setattr(cs.storage,'s3',SimpleNamespace(copy_object=forbidden))
 out=cs._copy([('store','root/golden/a.jpg','root/golden/a.jpg')],'store',skip_existing=set())
 assert out['copied']==0 and out['already']==1

def test_combine_builds_golden_before_combining(monkeypatch):
 from ppe_pipeline_api.chain import retrain_schedule as rs
 from ppe_pipeline_api import data
 calls=[]
 monkeypatch.setattr(rs,'source_selection',lambda c:{'folders':False,'pull':False})
 monkeypatch.setattr(rs,'assemble_plan',lambda c:None)
 monkeypatch.setattr(rs,'excluded_weeks',lambda c:[])
 monkeypatch.setattr(data,'build_golden',lambda **kw:calls.append(('build',kw)) or {'errors':[]})
 monkeypatch.setattr(data,'combine_store',lambda **kw:calls.append(('combine',kw)) or {'failed':0,'slugs':{}})
 monkeypatch.setattr(data,'rebuild_decayed',lambda **kw:calls.append(('decay',kw)) or {})
 config={'retention':{'use_golden':True},'central_store':{'bucket':'store','folder':'root'},'golden_set':{'source':'source','bucket':'source-bucket','target_folder':'old'}}
 rs.combine_and_decay(config,for_run=False)
 assert [c[0] for c in calls]==['build','combine','decay']
 assert calls[0][1]['target_folder']=='root/golden'
 assert calls[1][1]['golden_folder']=='root/golden'
