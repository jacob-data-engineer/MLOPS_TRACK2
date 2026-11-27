import {RunStatusBadge} from './RunStatusBadge';
import type {ReactNode} from "react";
import type {Track2TrainingAudit,VariantRunRecord} from '../lib/api';
import {splitCommonSettings} from '../lib/trainingHistory';
import {formatDayTime} from '../lib/format';
const display=(v:unknown)=>v==null?'—':typeof v==='object'?JSON.stringify(v):String(v);
function imageCount(audits:Track2TrainingAudit[],role:'train_dataset'|'eval_dataset'){
 const counts=[...new Set(audits.map(a=>a[role]?.file_count).filter((n):n is number=>typeof n==='number'))];
 return counts.length===1?counts[0].toLocaleString():counts.length?'Varies by model':'Not recorded';
}export function ExperimentHistory({run,audits,children}:{run:VariantRunRecord;audits:Track2TrainingAudit[];children?:ReactNode}){
 const settings=run.variants.map(v=>{const a=audits.find(a=>a.id===v.run_id);return a?{...v.settings,...a.parameters,hyperparams:a.hyperparams??v.settings.hyperparams}:v.settings;});
 const {common,different,flat}=splitCommonSettings(settings);
 const baseline=run.comparison?.result?.rows?.find(row=>row.role==='baseline');
 const datasets=[...new Map(audits.map(a=>[a.dataset_version_id||a.id,a])).values()];
 return <details className="rounded border border-hairline bg-surface p-3 text-xs">
  <summary className="cursor-pointer font-semibold"><span className="inline-flex w-[calc(100%-1rem)] items-center gap-3 align-middle"><span className="shrink-0"><RunStatusBadge status={run.state}/></span><span>Run {run.run_number??'pending'} · {run.name} · {run.variants.length} models · {formatDayTime(run.created_at)}<span className="ml-3 font-normal">Training: {imageCount(audits,'train_dataset')} images · Validation: {imageCount(audits,'eval_dataset')} images</span></span></span></summary>
  <p className="my-2 break-all">Starting model: {(run.starting_model_arn??run.baseline_arn).split('/').slice(-2).join(' / ')} · Comparison: {run.comparison?.status??'Not available'}</p>
  {run.comparison?.error&&<p className="text-critical">{run.comparison.error}</p>}
  <div className="my-3 rounded border border-hairline p-2"><strong>Common training parameters</strong><div className="mt-2 grid gap-2 sm:grid-cols-3 lg:grid-cols-5">{Object.entries(common).map(([k,v])=><span key={k}>{k.replace('hyperparams.','')}: <b>{display(v)}</b></span>)}</div></div>
  <div className="overflow-x-auto"><table className="w-full text-left"><caption className="text-left font-semibold">Models and differing parameters</caption><thead><tr>{['Model / variant','Status',...different.map(k=>k.replace('hyperparams.','')),'Started','Ended','mAP50','mAP50-95'].map((h,i)=><th className="p-2" key={i}>{h}</th>)}</tr></thead><tbody>{run.baseline_arn&&<tr className="border-t border-hairline bg-surface"><td className="p-2"><strong>Baseline comparison model</strong><div>{run.baseline_arn.split('/').slice(-2).join(' / ')}</div><div>Comparison scores</div></td><td><RunStatusBadge status={baseline?'Evaluated':run.comparison?.status??'Not evaluated'}/></td>{different.map(k=><td key={k} className="p-2">N/A</td>)}<td>N/A</td><td>N/A</td><td>{baseline?display(baseline.mAP50):'Not measured'}</td><td>{baseline?display(baseline['mAP50-95']):'Not measured'}</td></tr>}{run.variants.map((v,i)=>{const a=audits.find(a=>a.id===v.run_id);return <tr className="border-t border-hairline" key={v.index}><td className="p-2">{v.name}<div>{a?.model?`${a.model.group} / ${a.model.version}`:v.model_package_arn?.split('/').slice(-2).join(' / ')||'Not registered'}</div></td><td><RunStatusBadge status={a?.status??v.training_status??v.state}/>{!a&&<div>Audit not loaded</div>}</td>{different.map(k=><td key={k} className="bg-amber-50 p-2">{display(flat[i][k])}</td>)}<td>{a?.started_at?formatDayTime(a.started_at):'—'}</td><td>{a?.ended_at?formatDayTime(a.ended_at):'—'}</td><td>{display(a?.metrics?.mAP50)}</td><td>{display(a?.metrics?.['mAP50-95'])}</td></tr>;})}</tbody></table></div>
  <p className="mt-2 text-muted">Variant scores are recorded training results; baseline scores come from the comparison evaluation. Common training parameters apply to variants only. The baseline is not trained in this run.</p><details className="mt-3"><summary>Dataset and source details (shared where recorded)</summary>{datasets.map(a=><div className="my-2 break-all" key={a.dataset_version_id||a.id}><strong>{a.dataset_version_id||'Dataset ID not recorded'}</strong><p>Training: {a.train_dataset.file_count} images · {a.train_dataset.source_folders?.join(', ')||a.train_dataset.folder}</p><p>Validation: {a.eval_dataset.file_count} images · {a.eval_dataset.source_folders?.join(', ')||a.eval_dataset.folder}</p></div>)}</details>
  <details className="mt-3"><summary>Individual audit records</summary>{children}</details>
 </details>;
}
