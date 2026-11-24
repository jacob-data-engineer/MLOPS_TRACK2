import {useEffect, useState} from 'react';
import {ApiError, getTrack2Metrics, listBaselineModels, registerBaselineModel,
  type BaselineSelection, type BaselineVersion, type Track2MetricsRow} from '../lib/api';

export function Track2BaselineComparison({value, onChange, disabled = false, completed = false}: {
  value: BaselineSelection; onChange: (value: BaselineSelection) => void; disabled?: boolean; completed?: boolean;
}) {
  const [versions, setVersions] = useState<BaselineVersion[]>([]);
  const [models, setModels] = useState<Track2MetricsRow[]>([]);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [source, setSource] = useState('');
  const [label, setLabel] = useState('');
  const [reload, setReload] = useState(0);
  useEffect(() => {
    let active = true;
    setError('');
    Promise.all([listBaselineModels(), getTrack2Metrics()]).then(([registry, metrics]) => {
      if (active) {setVersions(registry.versions); setModels(metrics.rows);}
    }).catch(e => {if (active) setError(e instanceof ApiError ? e.detail : 'Could not load comparison models.');});
    return () => {active = false;};
  }, [reload]);
  async function register() {
    setBusy(true); setError('');
    try {
      const registered = await registerBaselineModel(source, label);
      setReload(n => n + 1); onChange({...value, enabled: true, baseline_arn: registered.arn});
      setLabel('');
    } catch (e) {setError(e instanceof ApiError ? e.detail : 'Could not register the baseline.');}
    finally {setBusy(false);}
  }
  return <fieldset disabled={disabled || busy} className="my-3 space-y-3 rounded-md border border-hairline p-3 text-sm">
    <legend className="px-1 font-semibold">Baseline Comparison</legend>
    <label className="flex items-center gap-2"><input type="checkbox" checked={value.enabled}
      onChange={e => onChange({...value, enabled: e.target.checked})}/>Compare this training run</label>
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
      <label className="block">Other trained models (optional, up to four)
        <select multiple size={4} aria-label="Other comparison models" className="mt-1 block w-full rounded border border-hairline bg-page p-2"
          value={value.other_arns || []} onChange={e => onChange({...value,
            other_arns: Array.from(e.target.selectedOptions, o => o.value).slice(0, 4)})}>
          {(value.other_arns || []).filter(a => !models.some(m => m.model_package_arn === a)).map(a =>
            <option key={a} value={a}>Unavailable saved version: {a}</option>)}
          {models.map(m => <option key={m.model_package_arn} value={m.model_package_arn}>{m.group_name} · v{m.version}</option>)}
        </select>
      </label>
      <p className="text-xs text-ink-muted">{completed ? 'Evaluate' : 'After training, evaluate'} these fixed versions on this run’s preserved validation images and labels. This starts a separate billed evaluation job; it does not train the baseline or change the serving model.</p>
    </>}
    <details><summary className="cursor-pointer font-medium">Baseline Model registry · {versions.length} versions</summary>
      <p className="my-2 text-xs">Register an existing model here as a separate baseline version. For a .pt file, first use “Import a pretrained COCO or an already-trained model” in Multi-Auto Training, then select its registered version below.</p>
      <label className="block">Source model version<select className="my-1 block w-full rounded border border-hairline bg-page p-2" value={source} onChange={e => setSource(e.target.value)}>
        <option value="">Choose a registered model</option>
        {models.map(m => <option key={m.model_package_arn} value={m.model_package_arn}>{m.group_name} · v{m.version}</option>)}
      </select></label>
      <label className="block">Baseline name<input className="my-1 block w-full rounded border border-hairline bg-page p-2" value={label} maxLength={200} onChange={e => setLabel(e.target.value)}/></label>
      <button type="button" disabled={!source || !label.trim() || busy} onClick={register} className="rounded border border-hairline px-3 py-2 disabled:opacity-40">{busy ? 'Registering…' : 'Register baseline version'}</button>
      <ul className="mt-2 text-xs">{versions.map(v => <li key={v.arn}>v{v.version} · {v.label}</li>)}</ul>
    </details>
    {error && <p role="alert" className="text-xs text-critical">{error}</p>}
    <button type="button" className="text-xs underline" onClick={() => setReload(n => n + 1)}>Refresh model lists</button>
  </fieldset>;
}
