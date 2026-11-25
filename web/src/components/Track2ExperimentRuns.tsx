import {useEffect,useState} from 'react';
import {usePageSnapshot} from '../lib/pageSnapshot';
import {ApiError,listVariantRuns,getTrack2Metrics,getTrack2RunsByModel,type VariantRunRecord,type Track2MetricsRow,type Track2RunForModel,type Track2ExportSettings} from '../lib/api';
import {VariantSettings,VariantResults} from './Track2VariantResults';
import {Track2ComparisonExport} from './Track2ComparisonExport';
import {formatDayTime} from '../lib/format';

const activeStates=['Pending','Preparing','Training','Comparing'];
const terminalStates=['Succeeded','Failed','Stopped'];
const status=(v:VariantRunRecord['variants'][number])=>v.state==='Queued'?(v.training_status??'Queued'):v.state;
export function ExperimentSummary({run}:{run:VariantRunRecord}){
 const finished=run.variants.filter(v=>terminalStates.includes(v.state)).length;
 return <section aria-label={`Experiment ${run.name}`} className="my-3 space-y-2 rounded border border-hairline p-3">
  <h3 className="font-semibold">{run.name}</h3>
  <p role="status" className="text-sm">{run.state} Â· {finished} / {run.variants.length} variants finished Â· {run.variants.filter(v=>v.state==='Succeeded').length} models trained</p>
  {run.stage&&<p className="text-xs">{run.stage}</p>}
  <p className="break-all text-xs">Base model: {run.baseline_arn.split('/').slice(-2).join(' / ')} Â· Dataset: {run.dataset_version_id||'Shared dataset will be prepared once'}</p>
  {run.error&&<p role="alert" className="text-critical">{run.error}</p>}
  <div className="overflow-x-auto"><table aria-label="Experiment variants" className="w-full text-left text-xs"><thead><tr>{['Order','Variant','Run','Epochs','Batch','Status','Registered model'].map(h=><th className="p-2" key={h}>{h}</th>)}</tr></thead><tbody>{run.variants.map(v=><tr key={v.index} className="border-t border-hairline"><td className="p-2">{v.index+1}</td><th className="p-2">{v.name}</th><td>{v.run_number!=null?`Run ${v.run_number}`:v.run_id?'Run number pending':'Not started'}</td><td>{String(v.settings.epochs??'â€”')}</td><td>{String(v.settings.batch??'â€”')}</td><td>{status(v)}{v.error&&<p className="text-critical">{v.error}</p>}</td><td>{v.model_package_arn?.split('/').slice(-2).join(' / ')||'Not available yet'}</td></tr>)}</tbody></table></div>
  <p className="text-xs">Final baseline comparison: {run.comparison?.status??'Waiting for variant training to finish'}. Each variant has its own run number; all belong to this experiment and share the same dataset.</p>
  {run.comparison?.error&&<p role="alert" className="text-critical">{run.comparison.error}</p>}
 </section>;
}
export function Track2ExperimentRuns({queue=false}:{queue?:boolean}){
 const [runs,setRuns]=useState<VariantRunRecord[]>([]),[selected,setSelected]=useState(''),[error,setError]=useState('');
 const [models,setModels]=useState<Track2MetricsRow[]>([]),[modelRuns,setModelRuns]=useState<Record<string,Track2RunForModel>>({});
 const [modelError,setModelError]=useState('');
 const [settings,setSettings]=useState<Track2ExportSettings>({registry:'',generation:'',size:'',group_name:'',criterion:'mAP50',destination:'s3',bucket:'',prefix:'',retention:'keep_all',include_metadata:true,metadata_groups:[]});
 usePageSnapshot(queue?'experiment-queue':'experiment-review',{selected,settings},v=>{setSelected(v.selected);setSettings(v.settings);});
 useEffect(()=>{let alive=true;async function load(){try{const result=await listVariantRuns();if(alive){setRuns(result.runs);setError('');}}catch(e){if(alive)setError(e instanceof ApiError?e.detail:'Could not load experiments.');}}void load();const timer=setInterval(load,5000);return()=>{alive=false;clearInterval(timer);};},[]);
 const run=runs.find(r=>r.id===selected)??runs[0];
 const modelKey=run?.variants.map(v=>v.model_package_arn??'').join('|');
 useEffect(()=>{if(queue||!run)return;let alive=true;async function load(){try{const [m,r]=await Promise.all([getTrack2Metrics(),getTrack2RunsByModel()]);if(alive){setModels(m.rows);setModelRuns(r.runs);setModelError('');}}catch(e){if(alive)setModelError(e instanceof ApiError?e.detail:'Could not load models for export.');}}void load();const timer=setInterval(load,15000);return()=>{alive=false;clearInterval(timer);};},[queue,run?.id,modelKey]);
 const active=runs.filter(r=>activeStates.includes(r.state));
 const arns=new Set(run?[run.baseline_arn,...run.variants.map(v=>v.model_package_arn)]:[]);
 const exportRows=models.filter(m=>arns.has(m.model_package_arn));
 const resultRows=run?.comparison?.result?.rows?.map(r=>({...r,label:run.variants.find(v=>v.model_package_arn===r.id)?.name??r.label}));
 if(queue)return <div aria-label="Queued experiments"><h3 className="font-semibold">Experiment queue</h3><p className="text-xs">{active.length} active experiments Â· {active.reduce((n,r)=>n+r.variants.filter(v=>!terminalStates.includes(v.state)).length,0)} variants remaining. Variants train sequentially, followed by one baseline comparison.</p>{error&&<p role="alert">{error}</p>}{active.map(r=><ExperimentSummary key={r.id} run={r}/>)}{!active.length&&!error&&<p className="text-xs">No active experiments.</p>}{runs.filter(r=>!activeStates.includes(r.state)).length>0&&<details><summary>Finished experiments / needs attention</summary>{runs.filter(r=>!activeStates.includes(r.state)).map(r=><ExperimentSummary key={r.id} run={r}/>)}</details>}</div>;
 return <section aria-label="Experiment review" className="space-y-3 rounded border border-hairline p-4"><h2 className="font-semibold">Experiment baseline comparisons</h2><p className="text-xs">Review every variant from one submission against its original base model on the shared validation images. All experiment models are included; the manual four-model limit does not apply.</p><label>Experiment<select aria-label="Choose experiment" className="my-2 block w-full border bg-page p-2" value={run?.id??''} onChange={e=>setSelected(e.target.value)}><option value="">Choose an experiment</option>{runs.map(r=><option key={r.id} value={r.id}>{r.name} Â· {formatDayTime(r.created_at)} Â· {r.variants.length} variants Â· {r.state}</option>)}</select></label>{error&&<p role="alert">{error}</p>}{!runs.length&&!error&&<p>No experiments have been submitted yet.</p>}{run&&<><ExperimentSummary run={run}/><VariantSettings experiment={run}/>{resultRows?.length?<VariantResults rows={resultRows}/>:<p>Comparison results will appear after training and baseline evaluation finish.</p>}<h3 className="font-semibold">Export experiment models</h3><p className="text-xs">Select any or all available variants and the original baseline. Include model weights, images, labels and metadata in the export package.</p>{modelError&&<p role="alert">{modelError}</p>}{exportRows.length?<Track2ComparisonExport key={run.id} stateKey={`experiment-review/${run.id}`} rows={exportRows} runs={modelRuns} settings={settings} onChange={setSettings} autoSelect/>:<p>No registered models are available for export yet.</p>}</>}</section>;
}
