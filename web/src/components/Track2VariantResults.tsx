import {useState} from 'react';
import {Track2BaselineChart} from "./Track2BaselineChart";
import type {VariantRunRecord,ComparisonMetricRow} from '../lib/api';

const flatten=(settings:Record<string,unknown>)=>{
  const {hyperparams,...main}=settings;
  return {...main,...(hyperparams&&typeof hyperparams==='object'?hyperparams:{})} as Record<string,unknown>;
};
const display=(value:unknown)=>value===undefined?'—':JSON.stringify(value);
export function VariantSettings({experiment}:{experiment:VariantRunRecord}){
 const shared=flatten(experiment.shared), variants=experiment.variants.map(v=>flatten(v.settings));
 const keys=[...new Set([...Object.keys(shared),...variants.flatMap(Object.keys)])];
 return <details className="rounded border p-3"><summary className="cursor-pointer">Compare training parameters</summary><p className="my-2 text-xs">Highlighted cells differ from the first selected variant. These are the new variants’ settings, not the baseline’s historical training settings.</p><div className="max-h-96 overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Parameter</th><th>First selected variant</th>{experiment.variants.map(v=><th key={v.index}>{v.name}</th>)}</tr></thead><tbody>{keys.map(k=><tr key={k} className="border-t"><th className="p-2">{k}</th><td>{display(shared[k])}</td>{variants.map((v,i)=><td key={i} className={display(v[k])!==display(shared[k])?'bg-amber-100 p-2':'p-2'}>{display(v[k])}</td>)}</tr>)}</tbody></table></div></details>;
}
export function VariantResults({rows}:{rows:ComparisonMetricRow[]}){
 const [metric,setMetric]=useState<keyof Pick<ComparisonMetricRow,'mAP50-95'|'mAP50'|'precision'|'recall'|'f1'|'fp'|'fn'>>('mAP50-95');
 const ordered=[...rows].sort((a,b)=>metric==='fp'||metric==='fn'?a[metric]-b[metric]:b[metric]-a[metric]);
 return <><Track2BaselineChart rows={rows}/><label>Rank comparison by <select aria-label="Rank comparison by" className="border bg-page p-2" value={metric} onChange={e=>setMetric(e.target.value as typeof metric)}>{['mAP50-95','mAP50','precision','recall','f1','fp','fn'].map(k=><option key={k} value={k}>{k==='fp'?'False detections (fewer is better)':k==='fn'?'Missed objects (fewer is better)':k}</option>)}</select></label><p className="text-xs">Differences are percentage points from the baseline. All models use the same recorded validation images and evaluation settings.</p><div className="overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Model</th>{['mAP50-95','mAP50','precision','recall','f1','False detections','Missed objects'].map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{ordered.map(r=><tr key={r.id} className="border-t"><td className="p-2">{r.label}{r.role==='baseline'?' (Baseline)':''}</td>{(['mAP50-95','mAP50','precision','recall','f1'] as const).map(k=><td key={k}>{(r[k]*100).toFixed(2)}% ({r.delta_pp[k]>=0?'+':''}{r.delta_pp[k].toFixed(2)} pp)</td>)}<td>{r.fp}</td><td>{r.fn}</td></tr>)}</tbody></table></div><details><summary className="cursor-pointer">Per-class results</summary>{ordered.map(r=><div key={r.id} className="my-3 overflow-auto"><h4>{r.label}</h4><table className="w-full text-left text-xs"><thead><tr>{['Class','mAP50-95','Precision','Recall','False detections','Missed objects'].map(k=><th key={k}>{k}</th>)}</tr></thead><tbody>{r.per_class.map(c=><tr key={c.class} className="border-t"><td>{c.class}</td><td>{c['mAP50-95']==null?'—':(c['mAP50-95']*100).toFixed(2)+'%'}</td><td>{(c.precision*100).toFixed(2)}%</td><td>{(c.recall*100).toFixed(2)}%</td><td>{c.fp}</td><td>{c.fn}</td></tr>)}</tbody></table></div>)}</details></>;
}
