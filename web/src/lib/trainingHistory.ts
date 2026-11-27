import type {Track2TrainingAudit,VariantRunRecord} from './api';
export function groupTrainingHistory(audits:Track2TrainingAudit[],runs:VariantRunRecord[]){
 const owners=new Map(runs.flatMap(run=>run.variants.filter(v=>v.run_id).map(v=>[v.run_id!,run] as const)));
 const seen=new Set<string>();
 return audits.flatMap<{id:string;audits:Track2TrainingAudit[];run?:VariantRunRecord}>(audit=>{
  const run=owners.get(audit.id);
  if(!run)return [{id:audit.id,audits:[audit],run:undefined}];
  if(seen.has(run.id))return [];seen.add(run.id);
  const ids=new Set(run.variants.map(v=>v.run_id));
  return [{id:run.id,run,audits:audits.filter(a=>ids.has(a.id))}];
 }).sort((a,b)=>(b.run?.run_number??b.audits[0]?.run_number??-1)-(a.run?.run_number??a.audits[0]?.run_number??-1));
}
export function splitCommonSettings(settings:Record<string,unknown>[]){
 const flat=settings.map(s=>Object.fromEntries(Object.entries(s).flatMap(([key,value])=>key==='hyperparams'&&value&&typeof value==='object'?Object.entries(value).map(([k,v])=>[`hyperparams.${k}`,v]):[[key,value]])));
 const keys=[...new Set(flat.flatMap(s=>Object.keys(s)))].sort();
 const same=(a:unknown,b:unknown):boolean=>JSON.stringify(a)===JSON.stringify(b);
 const common=keys.filter(k=>flat.every(s=>Object.hasOwn(s,k)&&same(s[k],flat[0][k])));
 return {common:Object.fromEntries(common.map(k=>[k,flat[0][k]])),different:keys.filter(k=>!common.includes(k)),flat};
}
