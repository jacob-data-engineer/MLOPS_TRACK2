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
