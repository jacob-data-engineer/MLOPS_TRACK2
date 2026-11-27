from ppe_pipeline_api.routers import track2_models as r
from ppe_pipeline_api.chain import training_variants as e

def test_run_order_before_limit(monkeypatch):
    monkeypatch.setattr(r.t2,'scan_training_audits',lambda limit:[{'id':'new','submitted_at':'2026-11-27'},{'id':'old','submitted_at':'2026-11-25'}])
    monkeypatch.setattr(r.catalog_view,'run_numbers',lambda ids:{'new':81,'old':74})
    monkeypatch.setattr(e,'list_summaries',lambda:[{'run_number':77,'variants':[{'run_id':'new'}]},{'run_number':82,'variants':[{'run_id':'old'}]}])
    result=r.training_audit(limit=1,session={},order='run_number')
    assert [a['id'] for a in result['audits']]==['old']
