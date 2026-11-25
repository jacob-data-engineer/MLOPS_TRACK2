import {usePageSnapshot} from "../lib/pageSnapshot";
﻿import {useEffect, useState} from 'react';
import {Card} from '../components/Card';
import {ApiError, listPtRegistered, inspectPtUpload, inspectPtRegistered, type PtInspection} from '../lib/api';
export function Track2PtViewPage() {
  const [mode,setMode]=useState<'upload'|'registered'>('upload');
  const [rows,setRows]=useState<{arn:string;label:string;status:string}[]>([]);
  const [loading,setLoading]=useState(true);
  const [listError,setListError]=useState('');
  const [arn,setArn]=useState('');
  usePageSnapshot('checkpoint',{mode,arn},v=>{setMode(v.mode);setArn(v.arn);});
  const [file,setFile]=useState<File|null>(null);
  const [result,setResult]=useState<PtInspection|null>(null);
  const [error,setError]=useState('');
  const [busy,setBusy]=useState(false);
  async function loadModels() {
    setLoading(true);setListError('');
    try {setRows((await listPtRegistered()).models);}
    catch(e){setListError(e instanceof ApiError?e.detail:'Could not load registered models. Use Refresh models to retry.');}
    finally{setLoading(false);}
  }
  useEffect(()=>{void loadModels();},[]);
  async function inspect() {
    setBusy(true);setError('');setResult(null);
    try {setResult(mode==='upload' ? await inspectPtUpload(file!) : await inspectPtRegistered(arn));}
    catch(e){setError(e instanceof ApiError?e.detail:'Could not inspect this checkpoint.');}
    finally{setBusy(false);}
  }
  return <div className="space-y-5">
    <h1 className="text-xl font-semibold">PT View</h1>
    <p className="text-sm">Explore checkpoint classes and recorded metadata. Inspection does not train, register or modify the model. Uploaded files are not saved.</p>
    <Card title="Choose a checkpoint">
      <fieldset disabled={busy} className="min-w-0 space-y-4">
        <div className="flex gap-5">
          <label><input type="radio" name="pt-source" checked={mode==='upload'} onChange={()=>{setMode('upload');setResult(null);setError('');}}/> Uploaded .pt file</label>
          <label><input type="radio" name="pt-source" checked={mode==='registered'} onChange={()=>{setMode('registered');setResult(null);setError('');}}/> Registered model</label>
        </div>
        {mode==='upload' ? <label className="block">Checkpoint file (.pt, up to 512 MB)<input type="file" accept=".pt" className="mt-2 block" onChange={e=>{setFile(e.target.files?.[0]??null);setResult(null);}}/></label> :
          <div><label className="block">Registered model<select disabled={loading} className="mt-2 block w-full rounded border border-hairline bg-page p-2" value={arn} onChange={e=>{setArn(e.target.value);setResult(null);}}>
            <option value="">{loading?'Loading registered models…':'Choose a model version'}</option>{rows.map(r=><option key={r.arn} value={r.arn}>{r.label} · {r.status}</option>)}
          </select></label>
          <button type="button" className="mt-2 rounded border border-hairline px-2 py-1 text-xs" disabled={loading} onClick={()=>void loadModels()}>Refresh models</button>
          <p role="status" className="mt-2 text-xs">{loading?'Reading the model registry…':listError?'Model list unavailable.':rows.length?`${rows.length} registered model versions`:'No registered model versions found in this AWS region.'}</p>
          {listError&&<p role="alert" className="text-critical">{listError}</p>}</div>}
        <button className="rounded border border-hairline px-3 py-2 disabled:opacity-40" disabled={mode==='upload'?!file:!arn} onClick={()=>void inspect()}>{busy?'Inspecting...':'Inspect checkpoint'}</button>
      </fieldset>
      {error&&<p role="alert" className="mt-3 text-critical">{error}</p>}
    </Card>
    {result&&<>
      <Card title={`Classes (${result.classes.length})`}>
        <p className="mb-3 break-all text-xs">{result.filename} · {(result.size_bytes/1024/1024).toFixed(2)} MB</p>
        {result.notice&&<p role="status" className="mb-3 text-sm">{result.notice}</p>}
        <table className="w-full text-left text-sm"><thead><tr><th className="p-2">Class ID</th><th className="p-2">Class name</th></tr></thead><tbody>{result.classes.map(c=><tr key={c.id} className="border-t border-hairline"><td className="p-2">{c.id}</td><td className="p-2">{c.name}</td></tr>)}</tbody></table>
      </Card>
      <Card title="Recorded metadata">
        {Object.keys(result.metadata).length===0&&<p>No additional metadata was recorded.</p>}
        {Object.entries({...result.metadata,...(result.registry?{registry:result.registry}:{})}).map(([key,value])=><details key={key} className="border-b border-hairline py-2"><summary className="cursor-pointer font-medium">{key}</summary><pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-all text-xs">{JSON.stringify(value,null,2)}</pre></details>)}
      </Card>
    </>}
  </div>;
}
