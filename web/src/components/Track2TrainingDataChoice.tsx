import { useEffect, useRef, useState } from 'react';
import { ChevronDown } from 'lucide-react';
import { formatDatasetDateTime } from '../lib/format';
import { ApiError, checkTrack2SavedDataset, getTrack2TrainingAudit, listTrack2DatasetVersions,
  type SavedDatasetCheck, type Track2DatasetVersion, type Track2TrainingAudit } from '../lib/api';

export type TrainingDataChoice = { mode: 'prepared' | 'saved'; id: string; checked: boolean };

export function Track2TrainingDataChoice({value, onChange, detectorSlug, classKey, disabled}: {
  value: TrainingDataChoice; onChange: (value: TrainingDataChoice) => void;
  detectorSlug: string; classKey: string; disabled: boolean;
}) {
  const [versions, setVersions] = useState<Track2DatasetVersion[]>([]);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [datasetError, setDatasetError] = useState('');
  const [reloadDatasets, setReloadDatasets] = useState(0);
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
    setLoadingDatasets(true); setDatasetError(''); setVersions([]); setRuns([]);
    listTrack2DatasetVersions(100, true)
      .then(datasets => {if (active) setVersions(datasets.versions);})
      .catch(error => {if (active) setDatasetError(failure(error));})
      .finally(() => {if (active) setLoadingDatasets(false);});
    // Source-run details are optional: a slow history request must not hold up the picker.
    getTrack2TrainingAudit(100)
      .then(history => {if (active) setRuns(history.audits);})
      .catch(() => undefined);
    return () => {active = false;};
  }, [value.mode, reloadDatasets]);
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
      <label className="block font-semibold">Saved dataset record
      <span className="relative mt-2 block">
      <select aria-label="Saved dataset record" aria-busy={loadingDatasets} className="block min-h-12 w-full cursor-pointer appearance-none rounded-md border-2 border-ink-muted bg-white px-3 py-3 pr-12 text-sm font-medium text-ink shadow-sm hover:border-ink focus:border-accent focus:outline-none focus:ring-2 focus:ring-accent/30 disabled:cursor-not-allowed disabled:opacity-50" value={value.id} disabled={disabled || busy || loadingDatasets || !!datasetError || !versions.length}
        onChange={e => onChange({...value,id:e.target.value,checked:false})}>
        <option value="">{loadingDatasets ? 'Loading datasets…' : datasetError ? 'Could not load datasets' : versions.length ? 'Choose a dataset' : 'No saved datasets available'}</option>
        {value.id && !selected && <option value={value.id}>{value.id}</option>}
        {versions.map(v => <option key={v.version_id} value={v.version_id}>{formatDatasetDateTime(v.created_at)} · {v.version_id} · {v.train_images ?? 0} train / {v.val_images ?? 0} validation</option>)}
      </select>
      <ChevronDown aria-hidden="true" size={22} className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-ink" />
      </span></label>
      {loadingDatasets && <p role="status" aria-live="polite" className="mt-2 animate-pulse font-semibold text-ink motion-reduce:animate-none">Loading datasets…</p>}
      {datasetError && <div role="alert" className="mt-2 text-critical">
        <p>{datasetError}</p>
        <button type="button" className="mt-1 rounded border border-current px-3 py-2" disabled={disabled || busy}
          onClick={() => setReloadDatasets(n => n + 1)}>Retry loading datasets</button>
      </div>}
      {selected && <div className="my-2 space-y-1 break-all">
        <p>Dataset record ID: {selected.version_id}</p>
        <p>Created: {formatDatasetDateTime(selected.created_at)}</p>
        <p>Source run: {sourceRun ? `${sourceRun.run_number ?? sourceRun.id} · ${formatDatasetDateTime(sourceRun.submitted_at)}` : selected.run_id || 'No run recorded'}</p>
        <p>{selected.train_images} training / {selected.val_images} validation images</p>
        <p>Classes: {selected.classes?.join(', ')}</p>
        <p>Original central store: {checked?.store_uri || selected.central_recipe?.store_uri || 'Not recorded — this older dataset may not support reuse.'}</p>
      </div>}
      <button type="button" className="mt-2 rounded border border-black/15 px-3 py-2 disabled:opacity-40"
        disabled={disabled || busy || loadingDatasets || !!datasetError || !value.id || !detectorSlug} onClick={() => void check()}>{busy ? 'Checking saved files…' : 'Check saved dataset'}</button>
    </>}
    {message && <p role="status" className="mt-2">{message}</p>}
  </div>;
}
