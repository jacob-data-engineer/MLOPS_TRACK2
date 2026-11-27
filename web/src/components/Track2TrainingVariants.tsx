import {BaselineClassWarning} from "./BaselineClassWarning";
import {usePageSave} from "../lib/pageSave";
import {usePageSnapshot} from "../lib/pageSnapshot";
import {useEffect,useState,useImperativeHandle,type Ref} from "react";
import {TrainingParameterInput} from "./TrainingParameterInput";
import {Track2TrainingAdvanced,effectiveHyperparams} from "./Track2TrainingAdvanced";
import {ApiError,listTrainingVariantDrafts,saveTrainingVariantDraft,startVariantRun,
 type TrainingVariantDraft,type SavedTrainingVariant,type VariantSettingsSnapshot} from "../lib/api";

const uuid=()=>globalThis.crypto?.randomUUID?.()??'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.floor(Math.random()*16);return(c==='x'?r:(r&3)|8).toString(16);});
const message=(e:unknown)=>e instanceof ApiError?e.detail:e instanceof Error?e.message:'Request failed.';
const flatten=(s:VariantSettingsSnapshot):Record<string,unknown>=>{const {hyperparams,...base}=s;return {...base,...hyperparams};};
// Settings and drafts are JSON API values; do not require secure-context structuredClone.
const copyJson=<T,>(value:T):T=>JSON.parse(JSON.stringify(value));
const show=(v:unknown)=>v===undefined?'—':JSON.stringify(v);

export function Track2TrainingVariants({active=true,expectedClasses,detector,architecture,baseline,comparisonBaseline,dataset,dataMode='saved',settings,queueRef,onQueueState}:{active?:boolean;expectedClasses?:Record<string,string>;detector:string;architecture:string;baseline:string;comparisonBaseline?:string;dataset:string;dataMode?:"saved"|"prepared";settings:VariantSettingsSnapshot;queueRef?:Ref<{start:()=>Promise<string>}>;onQueueState?:(state:{count:number;blocked:boolean;name:string})=>void}){
 const [draft,setDraft]=useState<TrainingVariantDraft|null>(null),[loaded,setLoaded]=useState(false);
 const [variantName,setVariantName]=useState(''),[editing,setEditing]=useState('');
 const [busy,setBusy]=useState(false),[error,setError]=useState(''),[notice,setNotice]=useState('');
 const [classBlocked,setClassBlocked]=useState(true);
 const [runId,setRunId]=useState(uuid);
 const [localSettings,setLocalSettings]=useState(()=>copyJson(settings));
 useEffect(()=>setLocalSettings(copyJson(settings)),[JSON.stringify(settings)]);
  usePageSnapshot(`variant-editor/${detector}/${architecture}/${baseline}/${dataMode}/${dataset}`,{variantName,editing,localSettings,variants:draft?.variants??[]},v=>{setVariantName(v.variantName);setEditing(v.editing);setLocalSettings(v.localSettings);setDraft(current=>({...context,...current,variants:v.variants??[],revision:current?.revision??0,batch_name:current?.batch_name??"",id:current?.id??""}));},loaded);
 const ready=Boolean(detector&&architecture&&baseline&&(dataMode==='prepared'||dataset));
 const context={detector_slug:detector,architecture,baseline_arn:baseline,dataset_version_id:dataset,data_mode:dataMode};
 const matches=(row:{baseline_arn:string;dataset_version_id:string;data_mode?:string})=>row.baseline_arn===baseline&&(row.data_mode??'saved')===dataMode&&(dataMode==='prepared'||row.dataset_version_id===dataset);
 async function refreshDraft(){setLoaded(false);try{const result=await listTrainingVariantDrafts();const found=result.drafts.find(matches)??null;setDraft(found);setLoaded(true);setError('');}catch(e){setError(message(e));}}
 useEffect(()=>{void refreshDraft();},[]);
 const variants=draft?.variants??[], selected=variants.filter(v=>v.selected);
 usePageSnapshot('scheduled-experiment',{enabled:active,...context,comparison_baseline_arn:comparisonBaseline||null,variants:selected},()=>{},loaded);
 const saveBlocked=!detector||!architecture?'Choose a model name and model type above.':!baseline?'Choose a registered starting model version above.':dataMode==='saved'&&!dataset?'Choose Saved SQL dataset above and select a dataset before saving a variant.':!loaded?'Saved variants have not loaded. Use Refresh saved variants to retry.':!variantName.trim()?'Enter a variant name.':'';
 const frozen:VariantSettingsSnapshot={...localSettings,hyperparams:{...effectiveHyperparams({},true),...localSettings.hyperparams}};
 async function persist(rows:SavedTrainingVariant[],name=draft?.batch_name??''){
  setBusy(true);setError('');setNotice('');
  try{const saved=await saveTrainingVariantDraft({...context,revision:draft?.revision??0,batch_name:name,variants:rows});setDraft(saved);setNotice('Variants saved in SQL. No training has started.');return true;}catch(e){setError(message(e));return false;}finally{setBusy(false);}
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
 function changeRows(rows:SavedTrainingVariant[]){setDraft(current=>({...context,...current,variants:rows,revision:current?.revision??0,batch_name:current?.batch_name??'',id:current?.id??''}));}
 usePageSave('Variant batch',async()=>{
  if(!active)return;
  if(!loaded)throw new Error('Saved variants have not loaded.');
  if(!variants.length)return;
  if(!ready)throw new Error('Choose a starting model and dataset before saving variants.');
  if(!(await persist(variants)))throw new Error('Could not save the variant batch. See the variants card.');
 });
 const runName=`${detector} variants ${runId.slice(0,8)}`;
 const payload={id:runId,name:runName,...context,comparison_baseline_arn:comparisonBaseline||null,shared:selected[0]?.settings??frozen,
  evaluation:{imgsz:selected[0]?.settings.img_size??640,confidence:.25,ap_conf:.001,nms_iou:.7,max_det:300},
  variants:selected.map(v=>({name:v.name,overrides:v.settings}))};
 const batchBlocked=(!!comparisonBaseline&&classBlocked)||!ready||!loaded||busy||!selected.length;
 useImperativeHandle(queueRef,()=>({start}));
 useEffect(()=>{onQueueState?.({count:variants.length,blocked:batchBlocked,name:runName});},[variants.length,batchBlocked,runName,onQueueState]);
 async function start(){if(!active||batchBlocked){throw new Error('Select at least one saved variant and a base model first.');}setBusy(true);setError('');try{await startVariantRun(payload);setRunId(uuid());setNotice('Selected variants submitted. The background worker continues when this page is closed.');return `Submitted ${selected.length} variants. The background worker will prepare the shared data, train each variant, then compare with the base model. Training entries appear in the queue after preparation.`;}catch(e){setError(message(e));throw e;}finally{setBusy(false);}}
 const first:Record<string,unknown>=variants[0]?flatten(variants[0].settings):{}, keys=[...new Set(variants.flatMap(v=>Object.keys(flatten(v.settings))))];
 const changedKeys=keys.filter(k=>variants.some(v=>show(flatten(v.settings)[k])!==show(first[k])));
 const columns=[...new Set(['epochs','img_size','batch','seed','patience','lr0','optimizer',...changedKeys])];
 return <section aria-label="Saved training variants" className="mt-5 space-y-4 rounded border border-hairline p-4">
  <h3 className="font-semibold">Save training setups as variants</h3>
  <p className="text-xs">Pick a model and baseline above, choose newly prepared data or a saved SQL dataset, then adjust Training Settings below and save each setup here. Every variant starts from the same starting model and uses the same train/validation images.</p>
  {!ready&&<p role="status" className="text-xs">You can enter names now. To save or run variants, choose a registered starting version and either newly prepared data or a saved SQL dataset above.</p>}
  <BaselineClassWarning key={comparisonBaseline} arns={comparisonBaseline?[comparisonBaseline]:[]} expected={expectedClasses} onChecked={setClassBlocked}/>
  {classBlocked&&comparisonBaseline&&<p role="status" className="text-xs">Add to Queue is unavailable until baseline comparison classes are verified and compatible.</p>}
  <button type="button" disabled={busy} className="underline text-xs" onClick={()=>void refreshDraft()}>Refresh saved variants</button>
  <fieldset disabled={busy} className="min-w-0 space-y-3">
   <p className="break-all text-xs">Starting model: {baseline||'Not selected'}<br/>Dataset: {dataMode==='prepared'?'Freshly prepared once when this batch starts; shared by all variants':dataset||'Not selected'}</p>
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
   <p className="text-xs">For scheduled experiments, select the variants and use Save at the top of this page. Each scheduled occurrence pulls fresh data once and prepares a new shared dataset for those saved variants.</p>
   <p className="text-xs">Add to Queue automatically trains the selected variants, then compares the trained models with the selected comparison baseline on the same validation images. The background worker continues when this page is closed. Save or update each variant before queueing; unsaved edits in the settings form are not included.</p>
   {selected.length>0&&<p className="text-xs">{selected.length} selected: {selected.map(v=>`${v.name} (${v.settings.epochs} epochs)`).join(', ')}</p>}
   {variants.length>0&&batchBlocked&&<p className="text-xs">Choose a base model and select at least one saved variant before adding to the queue.</p>}
  </fieldset>
 </section>;
}
