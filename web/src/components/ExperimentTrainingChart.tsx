import {useState} from 'react';
import {experimentScores} from '../lib/experimentScores';
import type {Track2MetricsRow,Track2TrainingAudit,VariantRunRecord} from '../lib/api';
const metrics=['mAP50-95','mAP50','precision','recall','f1'] as const;
const colors=['#2563eb','#d97706','#059669','#7c3aed','#db2777'];
export function ExperimentTrainingChart({run,models,audits=[]}:{run:VariantRunRecord;models:Track2MetricsRow[];audits?:Track2TrainingAudit[]}){
 const [excluded,setExcluded]=useState<string[]>([]);
 const available=experimentScores(run,models,audits);
 const items=available.filter(item=>!excluded.includes(item.key));
 return <section aria-label="Recorded training score charts" className="my-4 space-y-3"><h3 className="font-semibold">Recorded training scores</h3>
 <p className="text-xs">These are each model's recorded training validation scores, not a shared baseline evaluation. Missing scores are shown as Not measured, not zero. No baseline differences are calculated here.</p>
 <fieldset className="flex flex-wrap gap-4 text-xs"><legend>Models to compare (baseline optional)</legend>{available.map(item=><label key={item.key}><input type="checkbox" checked={!excluded.includes(item.key)} onChange={e=>setExcluded(current=>e.target.checked?current.filter(key=>key!==item.key):[...current,item.key])}/> {item.label}</label>)}</fieldset><div className="flex flex-wrap gap-4 text-xs">{items.map((item,i)=><span key={i} className="flex items-center gap-2"><span className="h-3 w-3" style={{backgroundColor:colors[i%colors.length]}}/>{item.label}</span>)}</div>
 <div className="overflow-x-auto"><table aria-label="Variant training score comparison" className="w-full text-left text-xs"><thead><tr><th className="p-2">Model</th>{metrics.map(metric=><th key={metric} className="p-2">{metric}</th>)}</tr></thead><tbody>{items.map((item,i)=><tr key={i} className="border-t border-hairline"><th className="p-2">{item.label}</th>{metrics.map(metric=>{const score=item.values[metric];return <td key={metric} className="p-2">{typeof score==='number'&&Number.isFinite(score)?`${(score*100).toFixed(2)}%`:'Not measured'}</td>;})}</tr>)}</tbody></table></div>
 <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{metrics.map(metric=><section key={metric} className="border border-hairline p-3"><h4 className="mb-2 font-semibold">{metric} <span className="text-xs font-normal">(higher is better)</span></h4>{items.map((item,i)=>{
 const value=item.values[metric];const known=typeof value==='number'&&Number.isFinite(value);
 return <div key={i} className="mb-3 text-xs"><p>{item.label}</p><div className="flex items-center gap-2">{known?<><div className="h-5 flex-1 rounded bg-black/5" role="img" aria-label={`${item.label}, ${metric}: ${(value*100).toFixed(2)}%`}><div className="h-full rounded" style={{width:`${Math.max(0,Math.min(100,value*100))}%`,backgroundColor:colors[i%colors.length]}}/></div><span>{(value*100).toFixed(2)}%</span></>:<span className="text-ink-muted">Not measured</span>}</div></div>;
 })}<p className="text-xs text-ink-muted">Scale: 0–100%</p></section>)}</div></section>;
}
