import {useEffect,useMemo,useState} from 'react';
import {Card} from '../components/Card';
import {ExperimentSettings,ExperimentResults} from '../components/Track2ExperimentResults';
import {Track2ComparisonExport} from '../components/Track2ComparisonExport';
import {Track2TrainingAdvanced,effectiveHyperparams} from '../components/Track2TrainingAdvanced';
import {ApiError,listPtRegistered,getTrack2Detectors,listTrack2DatasetVersions,getTrack2Metrics,getTrack2RunsByModel,
 listExperiments,startExperiment,retryExperiment,getExperimentReport,
 type ExperimentRecord,type Track2Detector,type Track2DatasetVersion,type Track2MetricsRow,type Track2RunForModel,type Track2ExportSettings} from '../lib/api';
const defaults={epochs:20,img_size:640,batch:16,seed:42,patience:0};
const exportDefaults:Track2ExportSettings={registry:'',generation:'',size:'',group_name:'',criterion:'mAP50',destination:'s3',bucket:'',prefix:'',retention:'keep_all',include_metadata:true,metadata_groups:[]};
const newId=()=>globalThis.crypto?.randomUUID?.()??'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g,c=>{const r=Math.floor(Math.random()*16);return(c==='x'?r:(r&3)|8).toString(16);});
type Variant={name:string;values:Partial<typeof defaults>;params:Record<string,string>};
const message=(e:unknown)=>e instanceof ApiError?e.detail:e instanceof Error?e.message:'Request failed.';
export function Track2ExperimentsPage(){
 const [name,setName]=useState('');const [id,setId]=useState(newId);
 const [detectors,setDetectors]=useState<Track2Detector[]>([]),[datasets,setDatasets]=useState<Track2DatasetVersion[]>([]);
 const [models,setModels]=useState<{arn:string;label:string;status:string}[]>([]);
 const [detector,setDetector]=useState(''),[architecture,setArchitecture]=useState(''),[baseline,setBaseline]=useState(''),[dataset,setDataset]=useState('');
 const [shared,setShared]=useState(defaults),[params,setParams]=useState<Record<string,string>>({optimizer:'AdamW'});
 const [variants,setVariants]=useState<Variant[]>([{name:'Variant A',values:{},params:{lr0:'0.01'}},{name:'Variant B',values:{},params:{lr0:'0.005'}}]);
 const [evaluation,setEvaluation]=useState({imgsz:640,confidence:.25,ap_conf:.001,nms_iou:.7,max_det:300});
 const [experiments,setExperiments]=useState<ExperimentRecord[]>([]),[selected,setSelected]=useState('');
 const [error,setError]=useState(''),[loadError,setLoadError]=useState(''),[busy,setBusy]=useState(false),[review,setReview]=useState(false);
 const [rows,setRows]=useState<Track2MetricsRow[]>([]),[runs,setRuns]=useState<Record<string,Track2RunForModel>>({}),[exportSettings,setExportSettings]=useState(exportDefaults);
 async function load(){try{setExperiments((await listExperiments()).experiments);setLoadError('');}catch(e){setLoadError(message(e));}}
 useEffect(()=>{Promise.all([getTrack2Detectors(),listTrack2DatasetVersions(200,true),listPtRegistered()]).then(([d,v,m])=>{setDetectors(d.detectors);setDatasets(v.versions);setModels(m.models);}).catch(e=>setError(message(e)));void load();const timer=setInterval(()=>void load(),8000);return()=>clearInterval(timer);},[]);
 const experiment=experiments.find(e=>e.id===selected);
 useEffect(()=>{if(!selected)return;Promise.all([getTrack2Metrics(),getTrack2RunsByModel()]).then(([m,r])=>{setRows(m.rows);setRuns(r.runs);}).catch(e=>setError(message(e)));},[selected,experiment?.state]);
 const definition=detectors.find(d=>d.slug===detector);
 const architectures=Object.entries(definition?.groups_by_generation??{'11':definition?.groups??{}}).flatMap(([g,sizes])=>Object.entries(sizes).map(([s,group])=>({value:`yolo${g}${s}`,group})));
 const group=architectures.find(a=>a.value===architecture)?.group;
 const candidates=models.filter(m=>m.arn.split('/').at(-2)===group);
 const effective=variants.map(v=>({...shared,...v.values,hyperparams:effectiveHyperparams({...params,...v.params},true)}));
 const payload={id,name,detector_slug:detector,architecture,baseline_arn:baseline,dataset_version_id:dataset,shared:{...shared,hyperparams:effectiveHyperparams(params,true)},evaluation,variants:variants.map((v,i)=>({name:v.name,overrides:{...v.values,hyperparams:effective[i].hyperparams}}))};
 const selectedArns=useMemo(()=>new Set(experiment?[experiment.baseline_arn,...experiment.variants.map(v=>v.model_package_arn).filter(Boolean)]:[]),[experiment]);
 const exportRows=rows.filter(r=>selectedArns.has(r.model_package_arn));
 async function submit(){setBusy(true);setError('');try{const result=await startExperiment(payload);setSelected(result.id);setId(newId());setReview(false);await load();}catch(e){setError(message(e));}finally{setBusy(false);}}
 async function retry(index?:number){if(!experiment)return;setBusy(true);try{await retryExperiment(experiment.id,index);await load();}catch(e){setError(message(e));}finally{setBusy(false);}}
 async function report(){if(!experiment)return;try{const doc=await getExperimentReport(experiment.id);const url=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=`experiment-${experiment.id}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}catch(e){setError(message(e));}}
 const numeric=(key:keyof typeof defaults,value:number,onChange:(n:number)=>void)=><label key={key} className="text-xs">{key}<input aria-label={key} type="number" className="block w-28 rounded border border-hairline bg-page p-2" value={value} onChange={e=>onChange(Number(e.target.value))}/></label>;
 const updateVariant=(i:number,value:Variant)=>{setVariants(old=>old.map((v,n)=>n===i?value:v));setReview(false);};
 return <div className="space-y-5"><h1 className="text-xl font-semibold">Experiments</h1><p className="text-sm">Train variants from one fixed baseline and saved dataset, then compare every successful variant on the same validation images. The backend continues when this page is closed.</p>
 {error&&<p role="alert" className="text-critical">{error}</p>}
 <Card title="New experiment"><fieldset disabled={busy} className="min-w-0 space-y-4" onChange={()=>setReview(false)}>
 <label className="block">Experiment name<input className="ml-3 rounded border p-2" value={name} onChange={e=>setName(e.target.value)}/></label>
 <div className="grid gap-3 md:grid-cols-2"><label>Model name<select aria-label="Model name" className="block w-full border bg-page p-2" value={detector} onChange={e=>{setDetector(e.target.value);setArchitecture('');setBaseline('');}}><option value="">Choose a model name</option>{detectors.map(d=><option key={d.slug} value={d.slug}>{d.label}</option>)}</select></label>
 <label>Architecture<select aria-label="Architecture" className="block w-full border bg-page p-2" value={architecture} onChange={e=>{setArchitecture(e.target.value);setBaseline('');}}><option value="">Choose architecture</option>{architectures.map(a=><option key={a.value}>{a.value}</option>)}</select></label>
 <label>Baseline and starting checkpoint<select aria-label="Baseline and starting checkpoint" className="block w-full border bg-page p-2" value={baseline} onChange={e=>setBaseline(e.target.value)}><option value="">Choose an exact version</option>{candidates.map(m=><option key={m.arn} value={m.arn}>{m.label}</option>)}</select></label>
 <label>Saved dataset<select aria-label="Saved dataset" className="block w-full border bg-page p-2" value={dataset} onChange={e=>setDataset(e.target.value)}><option value="">Choose a saved SQL dataset</option>{datasets.map(d=><option key={d.version_id} value={d.version_id}>{d.run_number?`Run ${d.run_number} · `:''}{d.version_id} · {d.train_images} train / {d.val_images} validation</option>)}</select></label></div>
 <h3 className="font-semibold">Shared training settings</h3><div className="flex flex-wrap gap-3">{(Object.keys(shared) as (keyof typeof defaults)[]).map(k=>numeric(k,shared[k],n=>setShared({...shared,[k]:n})))}</div>
 <Track2TrainingAdvanced value={params} onChange={setParams} onCommit={()=>{}} enabled fixed onEnabledChange={()=>{}}/>
 <h3 className="font-semibold">Variants ({variants.length}/20)</h3><p className="text-xs">Every variant starts from the baseline. Changed values are highlighted. Duplicate a variant or edit its training parameters.</p>
 <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr><th>Variant</th><th>Epochs</th><th>Initial learning rate</th><th>Batch</th><th>Actions</th></tr></thead><tbody>{variants.map((v,i)=><tr key={i} className="border-t border-hairline"><td><input aria-label={`Variant ${i+1} name`} value={v.name} onChange={e=>updateVariant(i,{...v,name:e.target.value})} className="w-36 border p-2"/></td>
 {(['epochs','batch'] as const).slice(0,1).map(k=><td key={k}><input aria-label={`${v.name} ${k}`} type="number" value={v.values[k]??shared[k]} onChange={e=>updateVariant(i,{...v,values:{...v.values,[k]:Number(e.target.value)}})} className={`w-24 border p-2 ${v.values[k]!=null&&v.values[k]!==shared[k]?'bg-amber-100':''}`}/></td>)}
 <td><input aria-label={`${v.name} learning rate`} type="number" step="any" value={v.params.lr0??params.lr0??'0.01'} onChange={e=>updateVariant(i,{...v,params:{...v.params,lr0:e.target.value}})} className={`w-28 border p-2 ${effective[i].hyperparams.lr0!==effectiveHyperparams(params,true).lr0?'bg-amber-100':''}`}/></td>
 <td><input aria-label={`${v.name} batch`} type="number" value={v.values.batch??shared.batch} onChange={e=>updateVariant(i,{...v,values:{...v.values,batch:Number(e.target.value)}})} className={`w-24 border p-2 ${v.values.batch!=null&&v.values.batch!==shared.batch?'bg-amber-100':''}`}/></td>
 <td><button type="button" disabled={variants.length>=20} className="mr-3 underline" onClick={()=>setVariants([...variants,{...v,name:v.name+' copy'}])}>Duplicate</button><button type="button" disabled={variants.length===1} className="underline" onClick={()=>setVariants(variants.filter((_,n)=>n!==i))}>Remove</button><details><summary className="cursor-pointer">All settings</summary><div className="flex flex-wrap gap-2">{(['img_size','seed','patience'] as const).map(k=>numeric(k,v.values[k]??shared[k],n=>updateVariant(i,{...v,values:{...v.values,[k]:n}})))}</div><Track2TrainingAdvanced value={{...params,...v.params}} onChange={p=>updateVariant(i,{...v,params:p})} onCommit={()=>{}} enabled fixed onEnabledChange={()=>{}}/></details></td></tr>)}</tbody></table></div>
 <button type="button" disabled={variants.length>=20} className="rounded border px-3 py-2" onClick={()=>setVariants([...variants,{name:`Variant ${variants.length+1}`,values:{},params:{}}])}>Add variant</button>
 <details><summary className="cursor-pointer">Common evaluation settings</summary><div className="flex flex-wrap gap-3">{Object.entries(evaluation).map(([k,v])=><label key={k}>{k}<input type="number" step="any" className="block w-28 border p-2" value={v} onChange={e=>setEvaluation({...evaluation,[k]:Number(e.target.value)})}/></label>)}</div></details>
 <button type="button" disabled={!name||!baseline||!dataset||variants.some(v=>!v.name)} className="rounded border px-3 py-2" onClick={()=>setReview(true)}>Review experiment</button>
 </fieldset>
 {review&&<div className="mt-4 rounded border p-4"><p>{variants.length} training jobs, followed by one evaluation of the baseline and every successful variant. These jobs use billed AWS resources.</p><p className="my-2 break-all text-xs">Baseline: {baseline}<br/>Dataset: {dataset}</p><details><summary>Effective settings for every variant</summary><pre className="max-h-80 overflow-auto text-xs">{JSON.stringify(payload,null,2)}</pre></details><button disabled={busy} className="mt-3 rounded bg-ink px-4 py-2 text-white" onClick={()=>void submit()}>{busy?'Saving experiment…':'Run experiment'}</button></div>}
 </Card>
 <Card title="Saved experiments">{loadError&&<p role="alert">{loadError}</p>}<button className="mb-2 underline" onClick={()=>void load()}>Refresh experiments</button><select aria-label="Saved experiment" className="block w-full border bg-page p-2" value={selected} onChange={e=>setSelected(e.target.value)}><option value="">Choose an experiment</option>{experiments.map(e=><option key={e.id} value={e.id}>{e.name} · {e.state} · {e.created_at}</option>)}</select>
 {experiment&&<div className="mt-3 space-y-3"><p role="status">{experiment.state} — {experiment.stage}</p>{experiment.error&&<p className="text-critical">{experiment.error}</p>}<p className="text-xs break-all">Baseline: {experiment.baseline_arn}<br/>Dataset: {experiment.dataset_version_id}</p>
 <div className="overflow-x-auto"><table className="w-full text-left text-sm"><thead><tr><th>Variant</th><th>Run</th><th>Status</th><th>Recorded settings</th><th>Action</th></tr></thead><tbody>{experiment.variants.map(v=><tr key={v.index} className="border-t"><td>{v.name}</td><td>{v.run_number?`Run ${v.run_number}`:v.run_id??'Not submitted'}</td><td>{v.state==='Queued'?(v.training_status??v.state):v.state}{v.error&&<p className="text-critical">{v.error}</p>}</td><td><details><summary>View settings</summary><pre className="max-h-60 overflow-auto text-xs">{JSON.stringify(v.settings,null,2)}</pre></details></td><td>{!['Succeeded','Queued'].includes(v.state)&&!['Pending','Preparing','Training','Comparing'].includes(experiment.state)&&<button disabled={busy} className="underline" onClick={()=>void retry(v.index)}>Retry variant</button>}</td></tr>)}</tbody></table></div>
 {['Needs attention','Comparison failed'].includes(experiment.state)&&<button disabled={busy} className="underline" onClick={()=>void retry()}>Resume / retry comparison</button>}
 {experiment.comparison?.error&&<p role="alert">{experiment.comparison.error}</p>}
 <ExperimentSettings experiment={experiment}/>
 {experiment.comparison?.result?.rows&&<ExperimentResults rows={experiment.comparison.result.rows}/>}
 <button className="rounded border px-3 py-2" onClick={()=>void report()}>Download experiment report (JSON)</button>
 </div>}
 </Card>
 {experiment&&<Card title="Export experiment models"><p className="mb-3 text-xs">Select the baseline and successful variants to export. Download the experiment report above for the complete configuration and comparison results.</p>{exportRows.length?<Track2ComparisonExport key={experiment.id} rows={exportRows} runs={runs} settings={exportSettings} onChange={setExportSettings} autoSelect/>:<p>Loading available registered models. Refresh the page if the registry is still updating.</p>}</Card>}
 </div>;
}
