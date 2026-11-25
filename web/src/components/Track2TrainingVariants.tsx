import {usePageSave} from "../lib/pageSave";
import {usePageSnapshot} from "../lib/pageSnapshot";
import {useEffect,useState,useImperativeHandle,type Ref} from "react";
import {Card} from "./Card";
import {TrainingParameterInput} from "./TrainingParameterInput";
import {Track2TrainingAdvanced,effectiveHyperparams} from "./Track2TrainingAdvanced";
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

export function Track2TrainingVariants({active=true,detector,architecture,baseline,dataset,dataMode='saved',settings,queueRef,onQueueState}:{active?:boolean;detector:string;architecture:string;baseline:string;dataset:string;dataMode?:"saved"|"prepared";settings:VariantSettingsSnapshot;queueRef?:Ref<{start:()=>Promise<void>}>;onQueueState?:(state:{count:number;blocked:boolean;name:string})=>void}){
 const [batchDirty,setBatchDirty]=useState(false);
 const [draft,setDraft]=useState<TrainingVariantDraft|null>(null),[loaded,setLoaded]=useState(false);
 const [variantName,setVariantName]=useState(''),[editing,setEditing]=useState('');
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
 const [runId,setRunId]=useState(uuid),[batchName,setBatchName]=useState('');
 const [runs,setRuns]=useState<VariantRunRecord[]>([]),[selectedRun,setSelectedRun]=useState('');
 const [models,setModels]=useState<Track2MetricsRow[]>([]),[modelRuns,setModelRuns]=useState<Record<string,Track2RunForModel>>({});
 const [exportSettings,setExportSettings]=useState<Track2ExportSettings>({registry:'',generation:'',size:'',group_name:'',criterion:'mAP50',destination:'s3',bucket:'',prefix:'',retention:'keep_all',include_metadata:true,metadata_groups:[]});
 const [localSettings,setLocalSettings]=useState(()=>copyJson(settings));
 useEffect(()=>setLocalSettings(copyJson(settings)),[JSON.stringify(settings)]);
  usePageSnapshot(`variant-editor/${detector}/${architecture}/${baseline}/${dataMode}/${dataset}`,{variantName,editing,batchName,localSettings,selectedRun,exportSettings,variants:draft?.variants??[]},v=>{setVariantName(v.variantName);setEditing(v.editing);setBatchName(v.batchName);setLocalSettings(v.localSettings);setSelectedRun(v.selectedRun);setExportSettings(v.exportSettings);setDraft(current=>({...context,...current,variants:v.variants??[],revision:current?.revision??0,batch_name:v.batchName,id:current?.id??""}));setBatchDirty(true);},loaded);
 const ready=Boolean(detector&&architecture&&baseline&&(dataMode==='prepared'||dataset));
 const context={detector_slug:detector,architecture,baseline_arn:baseline,dataset_version_id:dataset,data_mode:dataMode};
 const matches=(row:{baseline_arn:string;dataset_version_id:string;data_mode?:string})=>row.baseline_arn===baseline&&(row.data_mode??'saved')===dataMode&&(dataMode==='prepared'||row.dataset_version_id===dataset);
 async function refreshDraft(){setLoaded(false);try{const result=await listTrainingVariantDrafts();const found=result.drafts.find(matches)??null;setDraft(found);setBatchName(found?.batch_name??'');setLoaded(true);setError('');}catch(e){setError(message(e));}}
 async function refreshRuns(){try{setRuns((await listVariantRuns()).runs.filter(matches));}catch(e){setError(message(e));}}
 useEffect(()=>{void refreshDraft();void refreshRuns();const timer=setInterval(()=>void refreshRuns(),8000);return()=>clearInterval(timer);},[]);
 const variants=draft?.variants??[], selected=variants.filter(v=>v.selected);
 const saveBlocked=!detector||!architecture?'Choose a model name and model type above.':!baseline?'Choose a registered starting model version above.':dataMode==='saved'&&!dataset?'Choose Saved SQL dataset above and select a dataset before saving a variant.':!loaded?'Saved variants have not loaded. Use Refresh saved variants to retry.':!variantName.trim()?'Enter a variant name.':'';
 const run=runs.find(r=>r.id===selectedRun)??runs[0];
 useEffect(()=>{if(!run)return;let alive=true;Promise.all([getTrack2Metrics(),getTrack2RunsByModel()]).then(([m,r])=>{if(alive){setModels(m.rows);setModelRuns(r.runs);}}).catch(e=>{if(alive)setError(message(e));});return()=>{alive=false;};},[run?.id,run?.state]);
 const frozen:VariantSettingsSnapshot={...localSettings,hyperparams:{...effectiveHyperparams({},true),...localSettings.hyperparams}};
 async function persist(rows:SavedTrainingVariant[],name=draft?.batch_name??''){
  setBusy(true);setError('');setNotice('');
  try{const saved=await saveTrainingVariantDraft({...context,revision:draft?.revision??0,batch_name:name,variants:rows});setDraft(saved);setBatchDirty(false);setNotice('Variants saved in SQL. No training has started.');return true;}catch(e){setError(message(e));return false;}finally{setBusy(false);}
 }
 async function save(asNew:boolean){
  if(saveBlocked){setError(saveBlocked);return;}
  try{
  const row={id:asNew?uuid():editing,name:variantName.trim(),settings:copyJson(frozen),selected:true};
  if(await persist(asNew?[...variants,row]:variants.map(v=>v.id===editing?{...row,selected:v.selected}:v))){setEditing(row.id);setVariantName(row.name);}
  }catch(e){setError(message(e));}
 }
 function edit(v:SavedTrainingVariant){
  setEditing(v.id);setVariantName(v.name);
  setLocalSettings(copyJson(v.settings));
  setNotice(`Editing ${v.name}. Change the Training Settings below, then Update variant or Save as new variant.`);
 }
 function duplicate(v:SavedTrainingVariant){
  let name=v.name+' copy',n=2;while(variants.some(row=>row.name===name))name=v.name+' copy '+n++;
  changeRows([...variants,{...copyJson(v),id:uuid(),name}]);
 }
 function changeRows(rows:SavedTrainingVariant[]){setDraft(current=>({...context,...current,variants:rows,revision:current?.revision??0,batch_name:current?.batch_name??'',id:current?.id??''}));setBatchDirty(true);}
 usePageSave('Variant batch',async()=>{
  if(!active)return;
  if(!loaded)throw new Error('Saved variants have not loaded.');
  if(!variants.length&&!batchName.trim())return;
  if(!ready)throw new Error('Choose a starting model and dataset before saving variants.');
  if(!(await persist(variants,batchName.trim())))throw new Error('Could not save the variant batch. See the variants card.');
 });
 const payload={id:runId,name:draft?.batch_name||`${detector} variants`,...context,shared:selected[0]?.settings??frozen,
  evaluation:{imgsz:selected[0]?.settings.img_size??640,confidence:.25,ap_conf:.001,nms_iou:.7,max_det:300},
  variants:selected.map(v=>({name:v.name,overrides:v.settings}))};
 const batchBlocked=batchDirty||!ready||!loaded||busy||!selected.length||!draft?.batch_name||batchName.trim()!==draft.batch_name;
 useImperativeHandle(queueRef,()=>({start}));
 useEffect(()=>{onQueueState?.({count:variants.length,blocked:batchBlocked,name:draft?.batch_name??''});},[variants.length,batchBlocked,draft?.batch_name,onQueueState]);
 async function start(){if(!active||batchBlocked){setError('Save the batch name and select at least one variant first.');return;}setBusy(true);setError('');try{const result=await startVariantRun(payload);setSelectedRun(result.id);setRunId(uuid());setNotice('Selected variants submitted. The background worker continues when this page is closed.');await refreshRuns();}catch(e){setError(message(e));}finally{setBusy(false);}}
 async function retry(index?:number){if(!run)return;setBusy(true);try{await retryVariantRun(run.id,index);await refreshRuns();}catch(e){setError(message(e));}finally{setBusy(false);}}
 async function report(){if(!run)return;try{const doc=await getVariantReport(run.id);const url=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`training-variants-${run.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){setError(message(e));}}
 const first:Record<string,unknown>=variants[0]?flatten(variants[0].settings):{}, keys=[...new Set(variants.flatMap(v=>Object.keys(flatten(v.settings))))];
 const changedKeys=keys.filter(k=>variants.some(v=>show(flatten(v.settings)[k])!==show(first[k])));
 const columns=[...new Set(['epochs','img_size','batch','seed','patience','lr0','optimizer',...changedKeys])];
 const exportArns=new Set(run?[run.baseline_arn,...run.variants.map(v=>v.model_package_arn)]:[]);
 const exportRows=models.filter(m=>exportArns.has(m.model_package_arn));
 return <section aria-label="Saved training variants" className="mt-5 space-y-4 rounded border border-hairline p-4">
  <h3 className="font-semibold">Save training setups as variants</h3>
  <p className="text-xs">Pick a model and baseline above, choose newly prepared data or a saved SQL dataset, then adjust Training Settings below and save each setup here. Every variant starts from that same baseline and uses the same train/validation images.</p>
  {!ready&&<p role="status" className="text-xs">You can enter names now. To save or run variants, choose a registered starting version and either newly prepared data or a saved SQL dataset above.</p>}
  <button type="button" disabled={busy} className="underline text-xs" onClick={()=>void refreshDraft()}>Refresh saved variants</button>
  <fieldset disabled={busy} className="min-w-0 space-y-3">
   <p className="break-all text-xs">Baseline: {baseline||'Not selected'}<br/>Dataset: {dataMode==='prepared'?'Freshly prepared once when this batch starts; shared by all variants':dataset||'Not selected'}</p>
   <div className="rounded-md border border-black/15 p-3" aria-label="Variant Training Settings">
    <h4 className="mb-2 text-xs font-medium text-ink">Training Settings</h4>
    <p className="mb-2 text-xs text-ink-muted">Edit this variant here, then Save as variant. These edits do not change the main training settings or other saved variants.</p>
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
     {([['Epochs','epochs'],['Image size','img_size'],['Batch','batch'],['Seed','seed'],['Patience','patience']] as const).map(([label,key])=><label key={key} className="block"><span className="mb-1 block text-xs text-ink-muted">{label}</span><TrainingParameterInput parameter={key} label={`Variant ${label}`} value={localSettings[key]} onChange={raw=>setLocalSettings(s=>({...s,[key]:Number(raw)}))} onCommit={()=>{}}/></label>)}
    </div>
    <label className="mt-2 block"><span className="mb-1 block text-xs text-ink-muted">Notes (saved with this variant)</span><input aria-label="Variant notes" className="w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-sm" value={localSettings.notes??''} onChange={e=>setLocalSettings(s=>({...s,notes:e.target.value}))}/></label>
    <p className="mt-1.5 text-xs text-ink-muted">Patience 0 means never stop early. A non-zero value stops once the score has not improved for that many epochs.</p>
    <Track2TrainingAdvanced pinned value={Object.fromEntries(Object.entries(localSettings.hyperparams).map(([k,v])=>[k,typeof v==='string'?v:JSON.stringify(v)]))} enabled={true} onEnabledChange={()=>{}} onChange={next=>setLocalSettings(s=>({...s,hyperparams:effectiveHyperparams(next,true)}))} onCommit={()=>{}}/>
   </div>
   <label className="block">Variant name <input aria-label="Variant name" maxLength={100} className="rounded border p-2" value={variantName} onChange={e=>setVariantName(e.target.value)}/></label>
   <div className="flex flex-wrap gap-3">
    <button type="button" disabled={!ready||!loaded||!variantName.trim()||variants.length>=20} className="rounded border px-3 py-2 disabled:opacity-40 disabled:cursor-not-allowed" onClick={()=>void save(true)}>{busy?'Saving...':editing?'Save as new variant':'Save as variant'}</button>
    {editing&&variants.some(v=>v.id===editing)&&<button type="button" disabled={!ready||!loaded||!variantName.trim()} className="rounded border px-3 py-2" onClick={()=>void save(false)}>Update variant</button>}
   </div>
   {saveBlocked&&<p role="status" className="text-xs">{saveBlocked}</p>}
   {error&&<p role="alert" className="text-critical">{error}</p>}
   {notice&&<p role="status" className="text-xs">{notice}</p>}
   <p className="text-xs">{variants.length}/20 saved variants. Highlighted values differ from the first saved variant. All displayed training defaults are captured. Saving does not start training.</p>
   <div className="overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Run?</th><th>Variant</th>{columns.map(k=><th key={k} className="px-2">{k}</th>)}<th>Actions</th></tr></thead><tbody>{variants.map(v=><tr key={v.id} className="border-t"><td><input aria-label={`Run ${v.name}`} type="checkbox" checked={v.selected} onChange={e=>changeRows(variants.map(row=>row.id===v.id?{...row,selected:e.target.checked}:row))}/></td><th className="p-2">{v.name}</th>{columns.map(k=><td key={k} className={`p-2 ${show(flatten(v.settings)[k])!==show(first[k])?'bg-amber-100':''}`}>{show(flatten(v.settings)[k])}</td>)}<td className="whitespace-nowrap space-x-3"><button type="button" className="underline" onClick={()=>edit(v)}>Edit</button><button type="button" disabled={variants.length>=20} className="underline" onClick={()=>duplicate(v)}>Duplicate</button><button type="button" className="underline" onClick={()=>changeRows(variants.filter(row=>row.id!==v.id))}>Remove</button></td></tr>)}</tbody></table></div>
   <details><summary>All recorded variant settings</summary><pre className="max-h-72 overflow-auto text-xs">{JSON.stringify(variants,null,2)}</pre></details>
   <label>Batch name <input aria-label="Variant batch name" className="rounded border p-2" value={batchName} onChange={e=>{setBatchName(e.target.value);}} placeholder={`${detector} variants`}/></label>
   <button type="button" disabled={!ready||!loaded||!selected.length||!batchName.trim()} className="ml-3 rounded border px-3 py-2 disabled:opacity-40" onClick={()=>void persist(variants,batchName.trim())}>Save batch</button>
   <p className="text-xs">Save batch stores its name and selected variants without starting training. Add to Queue below starts all selected saved variants, followed by their baseline comparison. Unsaved edits in the settings form are not included.</p>
   {selected.length>0&&<p className="text-xs">{selected.length} selected: {selected.map(v=>`${v.name} (${v.settings.epochs} epochs)`).join(', ')}</p>}
   {variants.length>0&&batchBlocked&&<p className="text-xs">Save the batch name and select at least one variant before adding it to the queue.</p>}
  </fieldset>
  {!!runs.length&&<div className="space-y-3"><h3 className="font-semibold">Variant runs and comparisons</h3><div className="overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Batch</th><th>Status</th><th>Created</th><th/></tr></thead><tbody>{runs.map(r=><tr key={r.id}><td>{r.name}</td><td>{r.state}</td><td>{r.created_at}</td><td><button type="button" className="underline" onClick={()=>setSelectedRun(r.id)}>View results</button></td></tr>)}</tbody></table></div>
   {run&&<><p role="status">{run.name}: {run.state} — {run.stage}</p>{run.error&&<p role="alert">{run.error}</p>}<div className="overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th>Variant</th><th>Run</th><th>Status</th><th>Action</th></tr></thead><tbody>{run.variants.map(v=><tr key={v.index}><td>{v.name}</td><td>{v.run_number?`Run ${v.run_number}`:v.run_id??'Not submitted'}</td><td>{v.state==='Queued'?(v.training_status??v.state):v.state}{v.error&&<p>{v.error}</p>}</td><td>{!['Succeeded','Queued'].includes(v.state)&&!['Pending','Preparing','Training','Comparing'].includes(run.state)&&<button type="button" disabled={busy} onClick={()=>void retry(v.index)}>Retry variant</button>}</td></tr>)}</tbody></table></div>
   {['Needs attention','Comparison failed'].includes(run.state)&&<button type="button" disabled={busy} onClick={()=>void retry()}>Resume / retry comparison</button>}
   {run.comparison?.error&&<p role="alert">{run.comparison.error}</p>}<VariantSettings experiment={run}/>{run.comparison?.result?.rows&&<VariantResults rows={run.comparison.result.rows}/>}
   <button type="button" className="rounded border px-3 py-2" onClick={()=>void report()}>Download variant report (JSON)</button>
   <Card title="Export variant models"><p className="mb-2 text-xs">Choose the original baseline and trained variants, package contents and destination. The report above contains the complete comparison.</p>{exportRows.length?<Track2ComparisonExport key={run.id} stateKey={`variant/${run.id}`} rows={exportRows} runs={modelRuns} settings={exportSettings} onChange={setExportSettings} autoSelect/>:<p className="text-xs">Registered models will appear when available.</p>}</Card></>}
  </div>}
 </section>;
}
