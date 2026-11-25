import {usePageSnapshot} from "../lib/pageSnapshot";
import {useEffect, useState} from 'react';
import {ApiError, getTrack2Metrics, listBaselineModels, registerBaselineModel, uploadBaselineModel, deleteBaselineModel,
  type BaselineSelection, type BaselineVersion, type Track2MetricsRow} from '../lib/api';
import {ConfirmDialog} from './ConfirmDialog';

export function Track2BaselineComparison({value, onChange, disabled = false, completed = false}: {
  value: BaselineSelection; onChange: (value: BaselineSelection) => void; disabled?: boolean; completed?: boolean;
}) {
  const [modelToAdd,setModelToAdd]=useState('');
  const [modelsToRemove,setModelsToRemove]=useState<string[]>([]);
  const [versions, setVersions] = useState<BaselineVersion[]>([]);
  const [models, setModels] = useState<Track2MetricsRow[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState('');
  const [sourceKind, setSourceKind] = useState<'upload' | 'registry'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [fileKey, setFileKey] = useState(0);
  const [notice, setNotice] = useState('');
  const [label, setLabel] = useState('');
  const [reload, setReload] = useState(0);
  const [deleteTarget,setDeleteTarget] = useState<BaselineVersion|null>(null);
  const [deleteError,setDeleteError] = useState('');
  usePageSnapshot('baseline-form',{source,sourceKind,label},v=>{setSource(v.source);setSourceKind(v.sourceKind);setLabel(v.label);});
  async function removeBaseline() {
    if (!deleteTarget) return;
    setBusy(true);setDeleteError('');setNotice('');
    try {
      await deleteBaselineModel(deleteTarget.version);
      setVersions(rows=>rows.filter(row=>row.arn!==deleteTarget.arn));
      setModels(rows=>rows.filter(row=>row.model_package_arn!==deleteTarget.arn));
      onChange({...value,baseline_arn:value.baseline_arn===deleteTarget.arn?'':value.baseline_arn,
        other_arns:(value.other_arns||[]).filter(arn=>arn!==deleteTarget.arn)});
      setNotice(`Deleted baseline v${deleteTarget.version}: ${deleteTarget.label}. Choose another baseline before comparing.`);
      setDeleteTarget(null);setReload(n=>n+1);
    } catch(e) {setDeleteError(e instanceof ApiError?e.detail:'Could not delete this baseline.');}
    finally {setBusy(false);}
  }
  useEffect(() => {
    let active = true;
    setError('');
    Promise.all([listBaselineModels(), getTrack2Metrics()]).then(([registry, metrics]) => {
      if (active) {setVersions(registry.versions); setModels(metrics.rows);}
    }).catch(e => {if (active) setError(e instanceof ApiError ? e.detail : 'Could not load comparison models.');});
    return () => {active = false;};
  }, [reload]);
  async function register() {
    setBusy(true); setError(''); setNotice('');
    try {
      if (sourceKind === 'upload' && (!file || !file.name.toLowerCase().endsWith('.pt') || file.size === 0 || file.size > 512 * 1024 * 1024)) {
        setError('Choose a non-empty .pt checkpoint up to 512 MB.'); return;
      }
      const registered = sourceKind === 'upload'
        ? await uploadBaselineModel(file!, label) : await registerBaselineModel(source, label);
      setReload(n => n + 1); onChange({...value, enabled: true, baseline_arn: registered.arn});
      setLabel(''); setFile(null); setFileKey(n => n + 1);
      setNotice('Registered in the separate Baseline Model registry and selected for comparison.');
    } catch (e) {setError(e instanceof ApiError ? e.detail : 'Could not register the baseline.');}
    finally {setBusy(false);}
  }
  return <fieldset disabled={disabled || busy} className="my-3 space-y-3 rounded-md border border-hairline p-3 text-sm">
    <legend className="px-1 font-semibold">Baseline Comparison</legend>
    <label className="flex items-center gap-2"><input type="checkbox" checked={value.enabled}
      onChange={e => onChange({...value, enabled: e.target.checked})}/>Compare this training run</label>
    <details><summary className="cursor-pointer font-medium">Baseline Model registry · {versions.length} versions</summary>
      <p className="my-2 text-xs">Upload a .pt checkpoint directly to the separate Baseline Model registry, or copy an existing registered version. Uploaded baselines are stored separately from training models.</p>
      <div className="my-2 flex flex-wrap gap-4">
        <label className="flex items-center gap-2"><input type="radio" checked={sourceKind === 'upload'} onChange={() => setSourceKind('upload')}/>Upload .pt file</label>
        <label className="flex items-center gap-2"><input type="radio" checked={sourceKind === 'registry'} onChange={() => setSourceKind('registry')}/>Existing registered model</label>
      </div>
      {sourceKind === 'upload' ? <label className="block">Baseline checkpoint (.pt)
        <input key={fileKey} type="file" accept=".pt" className="my-2 block w-full rounded border border-hairline p-2"
          onChange={e => {const chosen = e.target.files?.[0] || null; setFile(chosen); setError(''); setNotice('');
            if (chosen && !label) setLabel(chosen.name.replace(/\.pt$/i, '').slice(0, 200));}}/>
        <span className="text-xs text-ink-muted">Object detection checkpoint, up to 512 MB. Its class names are checked before registration.</span>
      </label> : <label className="block">Source model version<select className="my-1 block w-full rounded border border-hairline bg-page p-2" value={source} onChange={e => setSource(e.target.value)}>
        <option value="">Choose a registered model</option>
        {models.map(m => <option key={m.model_package_arn} value={m.model_package_arn}>{m.group_name} · v{m.version}</option>)}
      </select></label>}
      <label className="block">Baseline name<input className="my-1 block w-full rounded border border-hairline bg-page p-2" value={label} maxLength={200} onChange={e => setLabel(e.target.value)}/></label>
      <button type="button" disabled={(sourceKind === 'upload' ? !file : !source) || !label.trim() || busy} onClick={register} className="rounded border border-hairline px-3 py-2 disabled:opacity-40">{busy ? (sourceKind === 'upload' ? 'Uploading and registering…' : 'Registering…') : (sourceKind === 'upload' ? 'Upload and register baseline' : 'Register baseline version')}</button>
      <ul className="mt-2 text-xs">{versions.map(v => <li key={v.arn}>v{v.version} · {v.label}</li>)}</ul>
    </details>
    {!value.enabled && <p className="text-xs text-ink-muted">No comparisons. Training will run normally.</p>}
    {value.enabled && <>
      <label className="block">Baseline Model version
        <select className="mt-1 block w-full rounded border border-hairline bg-page p-2" value={value.baseline_arn || ''}
          onChange={e => onChange({...value, baseline_arn: e.target.value})}>
          <option value="">Select a baseline version</option>
          {value.baseline_arn && !versions.some(v => v.arn === value.baseline_arn) &&
            <option value={value.baseline_arn}>Unavailable saved version: {value.baseline_arn}</option>}
          {versions.map(v => <option key={v.arn} value={v.arn}>v{v.version} · {v.label}</option>)}
        </select>
      </label>
      <button type="button" className="rounded border border-critical/40 px-3 py-2 text-critical disabled:opacity-40"
        disabled={!versions.some(v=>v.arn===value.baseline_arn)}
        onClick={()=>{setDeleteError('');setDeleteTarget(versions.find(v=>v.arn===value.baseline_arn)??null);}}>Delete baseline model</button>
      <div className="space-y-2">
        <label className="block">Other trained models (optional, up to four)
          <select aria-label="Other comparison models" className="mt-1 block w-full rounded border border-hairline bg-page p-2" value={modelToAdd} onChange={e=>setModelToAdd(e.target.value)}>
            <option value="">Choose a model version</option>
            {models.filter(m=>!(value.other_arns||[]).includes(m.model_package_arn)).map(m=><option key={m.model_package_arn} value={m.model_package_arn}>{m.group_name} / v{m.version}</option>)}
          </select>
        </label>
        <button type="button" className="rounded border px-3 py-2 disabled:opacity-40" disabled={!modelToAdd||(value.other_arns||[]).length>=4||(value.other_arns||[]).includes(modelToAdd)} onClick={()=>{onChange({...value,other_arns:[...(value.other_arns||[]),modelToAdd]});setModelToAdd('');}}>Add model</button>
        <p className="text-xs">Selected models ({(value.other_arns||[]).length}/4)</p>
        <ul aria-label="Selected comparison models" className="space-y-1">{(value.other_arns||[]).map(arn=>{const model=models.find(m=>m.model_package_arn===arn);return <li key={arn} className="rounded border p-2"><label className="flex items-center gap-2"><input type="checkbox" checked={modelsToRemove.includes(arn)} onChange={e=>setModelsToRemove(current=>e.target.checked?[...current,arn]:current.filter(a=>a!==arn))}/>{model?`${model.group_name} / v${model.version}`:`Unavailable saved version: ${arn}`}</label></li>;})}</ul>
        <button type="button" className="rounded border border-critical/40 px-3 py-2 text-critical disabled:opacity-40" disabled={!(value.other_arns||[]).some(a=>modelsToRemove.includes(a))} onClick={()=>{onChange({...value,other_arns:(value.other_arns||[]).filter(a=>!modelsToRemove.includes(a))});setModelsToRemove([]);}}>Remove selected models</button>
        <p className="text-xs text-ink-muted">Removal only changes this comparison list. Registered models are kept.</p>
      </div>
      <p className="text-xs text-ink-muted">{completed ? 'Evaluate' : 'After training, evaluate'} these fixed versions on this run’s preserved validation images and labels. This starts a separate billed evaluation job; it does not train the baseline or change the serving model.</p>
    </>}

    {error && <p role="alert" className="text-xs text-critical">{error}</p>}
    {notice && <p role="status" className="text-xs">{notice}</p>}
    <ConfirmDialog open={!!deleteTarget} title="Delete baseline model?" destructive busy={busy}
      confirmLabel="Delete baseline version" onConfirm={()=>void removeBaseline()} onCancel={()=>setDeleteTarget(null)}
      body={<><p>Delete v{deleteTarget?.version}: {deleteTarget?.label} from the Baseline Model registry?</p>
        <p className="mt-2">Stored checkpoint files and existing comparison results are retained. Saved configurations using this version must select another baseline before starting a new comparison.</p>
        {deleteError&&<p role="alert" className="mt-2 text-critical">{deleteError}</p>}</>}/>
    <button type="button" className="text-xs underline" onClick={() => setReload(n => n + 1)}>Refresh model lists</button>
  </fieldset>;
}
