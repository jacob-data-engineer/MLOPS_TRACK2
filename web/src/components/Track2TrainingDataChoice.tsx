import { useEffect, useRef, useState } from 'react';
import { ApiError, checkTrack2SavedDataset, getTrack2TrainingAudit, listTrack2DatasetVersions,
  type SavedDatasetCheck, type Track2DatasetVersion, type Track2TrainingAudit } from '../lib/api';

export type TrainingDataChoice = { mode: 'prepared' | 'saved'; id: string; checked: boolean };

export function Track2TrainingDataChoice({value, onChange, detectorSlug, classKey, disabled}: {
  value: TrainingDataChoice; onChange: (value: TrainingDataChoice) => void;
  detectorSlug: string; classKey: string; disabled: boolean;
}) {
  const [versions, setVersions] = useState<Track2DatasetVersion[]>([]);
  const [runs, setRuns] = useState<Track2TrainingAudit[]>([]);
  const [checked, setChecked] = useState<SavedDatasetCheck | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const request = useRef(0);
  const anchor = useRef<HTMLDivElement>(null);
  const failure = (error: unknown) => error instanceof ApiError ? error.detail : 'Could not load the saved records.';
  useEffect(() => {
    request.current++;
    setChecked(null); setMessage(''); setBusy(false);
    onChange({...value, checked: false});
  }, [value.mode, value.id, detectorSlug, classKey]);
  useEffect(() => {
    if (value.mode === 'prepared') return;
    let active = true;
    Promise.all([listTrack2DatasetVersions(100, true), getTrack2TrainingAudit(100)])
      .then(([datasets, history]) => { if (active) {setVersions(datasets.versions); setRuns(history.audits);} })
      .catch(error => {if (active) setMessage(failure(error));});
    return () => {active = false;};
  }, [value.mode]);
  useEffect(() => {
    if (value.mode === 'saved' && value.id) anchor.current?.scrollIntoView({behavior: 'smooth', block: 'center'});
  }, [value.mode, value.id]);
  const selected = versions.find(v => v.version_id === value.id);
  async function check() {
    const token = ++request.current;
    setBusy(true); setMessage('Checking every recorded image and label…');
    try {
      const result = await checkTrack2SavedDataset(value.id, detectorSlug);
      if (token !== request.current) return;
      setChecked(result); onChange({...value, checked: true});
      setMessage(result.remapped ? 'Files verified. Class IDs will be remapped to your current model’s order.' : 'Files and class mapping verified.');
    } catch (error) { if (token === request.current) setMessage(failure(error)); }
    finally {if (token === request.current) setBusy(false);}
  }
  const sourceRun = runs.find(r => r.dataset_version_id === value.id || r.train_dataset?.dataset_version_id === value.id);
  return <div ref={anchor} className="mb-3 rounded-md border border-black/15 p-3 text-xs">
    <p className="mb-2 font-medium">Training data</p>
    <fieldset disabled={disabled || busy} className="flex flex-wrap gap-4">
      {([['prepared','Use newly prepared data'],['saved','Use a saved dataset']] as const).map(([mode,label]) =>
        <label key={mode} className="flex items-center gap-2"><input type="radio" name="training-data-choice"
          checked={value.mode === mode} onChange={() => onChange({mode,id:'',checked:false})}/>{label}</label>)}
    </fieldset>
    {value.mode === 'saved' && <>
      <p className="my-2">Rebuild files from the SQL record using your current model and training settings. The temporary copy is removed once training finishes and the queue is clear.</p>
      <label>Saved dataset record<select aria-label="Saved dataset record" className="mt-1 block w-full" value={value.id} disabled={disabled || busy}
        onChange={e => onChange({...value,id:e.target.value,checked:false})}>
        <option value="">Choose a dataset</option>
        {value.id && !selected && <option value={value.id}>{value.id}</option>}
        {versions.map(v => <option key={v.version_id} value={v.version_id}>{v.version_id} · {v.created_at} · {v.train_images ?? 0} train / {v.val_images ?? 0} validation</option>)}
      </select></label>
      {selected && <div className="my-2 space-y-1 break-all">
        <p>Dataset record ID: {selected.version_id}</p>
        <p>Source run: {sourceRun ? `${sourceRun.run_number ?? sourceRun.id} · ${sourceRun.submitted_at}` : selected.run_id || 'No run recorded'}</p>
        <p>{selected.train_images} training / {selected.val_images} validation images</p>
        <p>Classes: {selected.classes?.join(', ')}</p>
        <p>Original central store: {checked?.store_uri || selected.central_recipe?.store_uri || 'Not recorded — this older dataset may not support reuse.'}</p>
      </div>}
      <button type="button" className="mt-2 rounded border border-black/15 px-3 py-2 disabled:opacity-40"
        disabled={disabled || busy || !value.id || !detectorSlug} onClick={() => void check()}>{busy ? 'Checking saved files…' : 'Check saved dataset'}</button>
    </>}
    {message && <p role="status" className="mt-2">{message}</p>}
  </div>;
}
