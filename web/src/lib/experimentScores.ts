import type {Track2MetricsRow,Track2TrainingAudit,VariantRunRecord} from './api';
export function experimentScores(run:VariantRunRecord,models:Track2MetricsRow[],audits:Track2TrainingAudit[]){
 const items=[...(run.baseline_arn?[{arn:run.baseline_arn,runId:undefined as string|undefined,label:'Original baseline'}]:[]),...run.variants.map(v=>({arn:v.model_package_arn,runId:v.run_id,label:v.name}))];
 return items.map((item,index)=>{
  const audit=audits.find(a=>item.runId?a.id===item.runId:!!item.arn&&a.model?.model_package_arn===item.arn);
  const arn=item.arn||audit?.model?.model_package_arn;
  const registry=arn?models.find(m=>m.model_package_arn===arn):undefined;
  const values:Record<string,number|null>={};
  for(const key of ['mAP50-95','mAP50','precision','recall','f1']){
   const candidates=[registry?.metrics?.[key as keyof Track2MetricsRow['metrics']],audit?.metrics?.[key]];
   values[key]=candidates.find((v):v is number=>typeof v==='number'&&Number.isFinite(v))??null;
  }
  return {...item,key:item.runId||arn||String(index),values};
 });
}
