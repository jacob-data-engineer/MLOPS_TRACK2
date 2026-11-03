import { useEffect, useState } from "react";
import { Card } from "./Card";
import { ApiError, getTrack2PullControl, getTrack2SettingsFiles, loadTrack2TrainingData } from "../lib/api";

export function Track2TrainingDataFile() {
  const [files, setFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [loaded, setLoaded] = useState("");
  const [recipe, setRecipe] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([getTrack2SettingsFiles(), getTrack2PullControl()]).then(([listing, status]) => {
      setFiles(listing.files.filter(f => f.scope === "dp").map(f => f.name));
      const saved = status.config.training_data_preparation;
      setLoaded(saved?.name || "");
      setRecipe(saved?.config || null);
    }).catch(() => setError("Could not read saved Data Preparation files. Refresh to retry."));
  }, []);

  async function load() {
    setBusy(true); setError("");
    try {
      await loadTrack2TrainingData(selected, true);
      window.location.reload();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not load the preparation file.");
      setBusy(false);
    }
  }

  return <Card title="Data Preparation file for training" collapsible={false}>
    <p className="mb-3 text-sm text-ink-secondary">Load a saved preparation recipe, including folders, splits, labels, golden set and week selection. Training builds from these settings when you queue a run. Pull and training schedules stay unchanged.</p>
    <div className="flex flex-wrap items-center gap-3">
      <label htmlFor="training-data-file">Data Preparation file</label>
      <select id="training-data-file" className="max-w-full rounded border border-black/15 bg-page px-3 py-2" value={selected} onChange={e => setSelected(e.target.value)} disabled={busy}>
        <option value="">{files.length ? "Choose a saved _DP file" : "No saved Data Preparation files"}</option>
        {files.map(name => <option key={name} value={name}>{name}</option>)}
      </select>
      <button className="rounded-md bg-accent px-3 py-2 font-medium text-white disabled:opacity-40" disabled={!selected || busy} onClick={() => void load()}>{busy ? "Loading…" : "Load data settings"}</button>
    </div>
    <p className="mt-3 text-sm">Loaded: {loaded || "No Data Preparation file loaded"}</p>
    <p className="mt-1 text-sm text-ink-secondary">These are shared data settings, also visible in Data Preparation. Loading does not start a job. Later edits to the data cards affect the next run.</p>
    {error && <p role="alert" className="mt-2 text-danger">{error}</p>}
    {recipe && <details className="mt-3"><summary className="cursor-pointer">View imported preparation settings</summary>
      <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap text-xs">{JSON.stringify(recipe, null, 2)}</pre>
    </details>}
  </Card>;
}
