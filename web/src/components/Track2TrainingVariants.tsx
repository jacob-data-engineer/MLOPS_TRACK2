import {useEffect,useState} from "react";
import {Card} from "./Card";
import {effectiveHyperparams} from "./Track2TrainingAdvanced";
import {VariantSettings,VariantResults} from "./Track2VariantResults";
import {Track2ComparisonExport} from "./Track2ComparisonExport";
import {ApiError,listTrainingVariantDrafts,saveTrainingVariantDraft,listVariantRuns,startVariantRun,retryVariantRun,getVariantReport,getTrack2Metrics,getTrack2RunsByModel,
 type TrainingVariantDraft,type SavedTrainingVariant,type VariantSettingsSnapshot,type VariantRunRecord,type Track2MetricsRow,type Track2RunForModel,type Track2ExportSettings} from "../lib/api";

const uuid=()=>globalThis.crypto?.randomUUID?.()??'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.floor(Math.random()*16);return(c==='x'?r:(r&3)|8).toString(16);});
const message=(e:unknown)=>e instanceof ApiError?e.detail:e instanceof Error?e.message:'Request failed.';
const flatten=(s:VariantSettingsSnapshot):Record<string,unknown>=>{const {hyperparams,...base}=s;return {...base,...hyperparams};};
// Settings and drafts are JSON API values; do not require secure-context structuredClone.
const copyJson=<T,>(value:T):T=>JSON.parse(JSON.stringify(value));
const show=(v:unknown)=>v===undefined?'—':JSON.stringify(v);

export function Track2TrainingVariants({detector,architecture,baseline,dataset,settings}:{detector:string;architecture:string;baseline:string;dataset:string;settings:VariantSettingsSnapshot}){
 const [draft,setDraft]=useState<TrainingVariantDraft|null>(null),[loaded,setLoaded]=useState(false);
 const [variantName,setVariantName]=useState(''),[editing,setEditing]=useState('');
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState(''),[review,setReview]=useState(false);
 const [runId,setRunId]=useState(uuid),[batchName,setBatchName]=useState('');
 const [runs,setRuns]=useState<VariantRunRecord[]>([]),[selectedRun,setSelectedRun]=useState('');
 const [models,setModels]=useState<Track2MetricsRow[]>([]),[modelRuns,setModelRuns]=useState<Record<string,Track2RunForModel>>({});
 const [exportSettings,setExportSettings]=useState<Track2ExportSettings>({registry:'',generation:'',size:'',group_name:'',criterion:'mAP50',destination:'s3',bucket:'',prefix:'',retention:'keep_all',include_metadata:true,metadata_groups:[]});
 const ready=Boolean(detector&&architecture&&baseline&&dataset);
 const context={detector_slug:detector,architecture,baseline_arn:baseline,dataset_version_id:dataset};
 const matches=(row:{baseline_arn:string;dataset_version_id:string})=>row.baseline_arn===baseline&&row.dataset_version_id===dataset;
 async function refreshDraft(){setLoaded(false);try{const result=await listTrainingVariantDrafts();setDraft(result.drafts.find(matches)??null);setLoaded(true);setError('');}catch(e){setError(message(e));}}
 async function refreshRuns(){try{setRuns((await listVariantRuns()).runs.filter(matches));}catch(e){setError(message(e));}}
 useEffect(()=>{void refreshDraft();void refreshRuns();const timer=setInterval(()=>void refreshRuns(),8000);return()=>clearInterval(timer);},[]);
 const variants=draft?.variants??[], selected=variants.filter(v=>v.selected);
 const saveBlocked=!detector||!architecture?'Choose a model name and model type above.':!baseline?'Choose a registered starting model version above.':!dataset?'Choose Saved SQL dataset above and select a dataset before saving a variant.':!loaded?'Saved variants have not loaded. Use Refresh saved variants to retry.':!variantName.trim()?'Enter a variant name.':'';
 const run=runs.find(r=>r.id===selectedRun)??runs[0];
 useEffect(()=>{if(!run)return;let alive=true;Promise.all([getTrack2Metrics(),getTrack2RunsByModel()]).then(([m,r])=>{if(alive){setModels(m.rows);setModelRuns(r.runs);}}).catch(e=>{if(alive)setError(message(e));});return()=>{alive=false;};},[run?.id,run?.state]);
 const frozen:VariantSettingsSnapshot={...settings,hyperparams:{...effectiveHyperparams({},true),...settings.hyperparams}};
 async function persist(rows:SavedTrainingVariant[]){
  setBusy(true);setError('');setNotice('');setReview(false);
  try{const saved=await saveTrainingVariantDraft({...context,revision:draft?.revision??0,variants:rows});setDraft(saved);setNotice('Variants saved in SQL. No training has started.');return true;}catch(e){setError(message(e));return false;}finally{setBusy(false);}
 }
 async function save(asNew:boolean){
  if(saveBlocked){setError(saveBlocked);return;}
  try{
  const row={id:asNew?uuid():editing,name:variantName.trim(),settings:copyJson(frozen),selected:true};
  if(await persist(asNew?[...variants,row]:variants.map(v=>v.id===editing?{...row,selected:v.selected}:v))){setEditing(row.id);setVariantName(row.name);}
  }catch(e){setError(message(e));}
 }
 function edit(v:SavedTrainingVariant){
  setEditing(v.id);setVariantName(v.name);setReview(false);
  window.dispatchEvent(new CustomEvent('track2-edit-training-variant',{detail:v.settings}));
  setNotice(`Editing ${v.name}. Change the training settings above, then Update variant or Save as new variant.`);
 }
 function duplicate(v:SavedTrainingVariant){
  let name=v.name+' copy',n=2;while(variants.some(row=>row.name===name))name=v.name+' copy '+n++;
  void persist([...variants,{...copyJson(v),id:uuid(),name}]);
 }
 const payload={id:runId,name:batchName.trim()||`${detector} variants`,...context,shared:selected[0]?.settings??frozen,
  evaluation:{imgsz:selected[0]?.settings.img_size??640,confidence:.25,ap_conf:.001,nms_iou:.7,max_det:300},
  variants:selected.map(v=>({name:v.name,overrides:v.settings}))};
 async function start(){setBusy(true);setError('');try{const result=await startVariantRun(payload);setSelectedRun(result.id);setRunId(uuid());setReview(false);setNotice('Selected variants submitted. The background worker continues when this page is closed.');await refreshRuns();}catch(e){setError(message(e));}finally{setBusy(false);}}
 async function retry(index?:number){if(!run)return;setBusy(true);try{await retryVariantRun(run.id,index);await refreshRuns();}catch(e){setError(message(e));}finally{setBusy(false);}}
 async function report(){if(!run)return;try{const doc=await getVariantReport(run.id);const url=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`training-variants-${run.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){setError(message(e));}}
 const first:Record<string,unknown>=variants[0]?flatten(variants[0].settings):{}, keys=[...new Set(variants.flatMap(v=>Object.keys(flatten(v.settings))))];
 const changedKeys=keys.filter(k=>variants.some(v=>show(flatten(v.settings)[k])!==show(first[k])));
 const columns=[...new Set(['epochs','img_size','batch','seed','patience','lr0','optimizer',...changedKeys])];
 const exportArns=new Set(run?[run.baseline_arn,...run.variants.map(v=>v.model_package_arn)]:[]);
 const exportRows=models.filter(m=>exportArns.has(m.model_package_arn));
 return <section aria-label="Saved training variants" className="mt-5 space-y-4 rounded border border-hairline p-4">
  <h3 className="font-semibold">Save training setups as variants</h3>
  <p className="text-xs">Pick a model and baseline above, choose one saved SQL dataset, then change the normal training settings and save each setup here. Every variant starts from that same baseline and uses the same train/validation images.</p>
  {!ready&&<p role="status" className="text-xs">You can enter names now. To save or run variants, choose a registered starting version and a Saved SQL dataset above.</p>}
  <button type="button" disabled={busy} className="underline text-xs" onClick={()=>void refreshDraft()}>Refresh saved variants</button>
  <fieldset disabled={busy} className="min-w-0 space-y-3">
   <p className="break-all text-xs">Baseline: {baseline||'Not selected'}<br/>Dataset: {dataset||'Not selected'}</p>
   <label className="block">Variant name <input aria-label="Variant name" maxLength={100} className="rounded border p-2" value={variantName} onChange={e=>setVariantName(e.target.value)}/></label>
   <div className="flex flex-wrap gap-3">
    <button type="button" disabled={!ready||!loaded||!variantName.trim()||variants.length>=20} className="rounded border px-3 py-2 disabled:opacity-40 disabled:cursor-not-allowed" onClick={()=>void save(true)}>{busy?'Saving...':editing?'Save as new variant':'Save as variant'}</button>
    {editing&&variants.some(v=>v.id===editing)&&<button type="button" disabled={!ready||!loaded||!variantName.trim()} className="rounded border px-3 py-2" onClick={()=>void save(false)}>Update variant</button>}
   </div>
   {saveBlocked&&<p role="status" className="text-xs">{saveBlocked}</p>}
   {error&&<p role="alert" className="text-critical">{error}</p>}
   {notice&&<p role="status" className="text-xs">{notice}</p>}
   <details><summary>Current settings to save</summary><pre className="max-h-72 overflow-auto text-xs">{JSON.stringify(frozen,null,2)}</pre></details>
   <p className="text-xs">{variants.length}/20 saved variants. Highlighted values differ from the first saved variant. All displayed training defaults are captured. Saving does not start training.</p>
   <div className="overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Run?</th><th>Variant</th>{columns.map(k=><th key={k} className="px-2">{k}</th>)}<th>Actions</th></tr></thead><tbody>{variants.map(v=><tr key={v.id} className="border-t"><td><input aria-label={`Run ${v.name}`} type="checkbox" checked={v.selected} onChange={e=>void persist(variants.map(row=>row.id===v.id?{...row,selected:e.target.checked}:row))}/></td><th className="p-2">{v.name}</th>{columns.map(k=><td key={k} className={`p-2 ${show(flatten(v.settings)[k])!==show(first[k])?'bg-amber-100':''}`}>{show(flatten(v.settings)[k])}</td>)}<td className="whitespace-nowrap space-x-3"><button type="button" className="underline" onClick={()=>edit(v)}>Edit</button><button type="button" disabled={variants.length>=20} className="underline" onClick={()=>duplicate(v)}>Duplicate</button><button type="button" className="underline" onClick={()=>void persist(variants.filter(row=>row.id!==v.id))}>Remove</button></td></tr>)}</tbody></table></div>
   <details><summary>All recorded variant settings</summary><pre className="max-h-72 overflow-auto text-xs">{JSON.stringify(variants,null,2)}</pre></details>
   <label>Batch name <input aria-label="Variant batch name" className="rounded border p-2" value={batchName} onChange={e=>{setBatchName(e.target.value);setReview(false);}} placeholder={`${detector} variants`}/></label>
   <button type="button" disabled={!ready||!loaded||!selected.length} className="ml-3 rounded border px-3 py-2" onClick={()=>setReview(true)}>Review selected variants</button>
  </fieldset>
  {review&&<div className="rounded border p-3"><p>{selected.length} training jobs followed by one comparison with the original baseline. AWS charges apply. This uses the saved variants, not unsaved edits in the settings above.</p><ul>{selected.map(v=><li key={v.id}>{v.name}: {v.settings.epochs} epochs</li>)}</ul><button type="button" disabled={busy} className="mt-2 rounded bg-ink px-3 py-2 text-white" onClick={()=>void start()}>Run selected variants</button></div>}
  {!!runs.length&&<div className="space-y-3"><h3 className="font-semibold">Variant runs and comparisons</h3><div className="overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Batch</th><th>Status</th><th>Created</th><th/></tr></thead><tbody>{runs.map(r=><tr key={r.id}><td>{r.name}</td><td>{r.state}</td><td>{r.created_at}</td><td><button type="button" className="underline" onClick={()=>setSelectedRun(r.id)}>View results</button></td></tr>)}</tbody></table></div>
   {run&&<><p role="status">{run.name}: {run.state} — {run.stage}</p>{run.error&&<p role="alert">{run.error}</p>}<div className="overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th>Variant</th><th>Run</th><th>Status</th><th>Action</th></tr></thead><tbody>{run.variants.map(v=><tr key={v.index}><td>{v.name}</td><td>{v.run_number?`Run ${v.run_number}`:v.run_id??'Not submitted'}</td><td>{v.state==='Queued'?(v.training_status??v.state):v.state}{v.error&&<p>{v.error}</p>}</td><td>{!['Succeeded','Queued'].includes(v.state)&&!['Pending','Preparing','Training','Comparing'].includes(run.state)&&<button type="button" disabled={busy} onClick={()=>void retry(v.index)}>Retry variant</button>}</td></tr>)}</tbody></table></div>
   {['Needs attention','Comparison failed'].includes(run.state)&&<button type="button" disabled={busy} onClick={()=>void retry()}>Resume / retry comparison</button>}
   {run.comparison?.error&&<p role="alert">{run.comparison.error}</p>}<VariantSettings experiment={run}/>{run.comparison?.result?.rows&&<VariantResults rows={run.comparison.result.rows}/>}
   <button type="button" className="rounded border px-3 py-2" onClick={()=>void report()}>Download variant report (JSON)</button>
   <Card title="Export variant models"><p className="mb-2 text-xs">Choose the original baseline and trained variants, package contents and destination. The report above contains the complete comparison.</p>{exportRows.length?<Track2ComparisonExport key={run.id} rows={exportRows} runs={modelRuns} settings={exportSettings} onChange={setExportSettings} autoSelect/>:<p className="text-xs">Registered models will appear when available.</p>}</Card></>}
  </div>}
 </section>;
}
