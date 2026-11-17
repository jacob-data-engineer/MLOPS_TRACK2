import { useEffect, useState } from "react";
import { Card } from "./Card";
import {
  ApiError, getTrack2PullControl, getTrack2SettingsFiles, loadTrack2TrainingData,
  previewTrack2TrainingData, type Track2TrainingDataPreview,
} from "../lib/api";
import { formatDayTime } from "../lib/format";

export function Track2TrainingDataFile() {
  const [files, setFiles] = useState<string[]>([]);
  const [selected, setSelected] = useState("");
  const [loaded, setLoaded] = useState("");
  // When the loaded file was loaded for training, first created and last saved.
  const [loadedTimes, setLoadedTimes] = useState<{ loaded_at?: string; created_at?: string; saved_at?: string }>({});
  const [recipe, setRecipe] = useState<Record<string, unknown> | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  // The chosen (or loaded) file's classes and the data.yaml a run would
  // build from it. Re-read whenever the choice changes -- built from the
  // bucket as it is now, so it is never a stale copy.
  const [preview, setPreview] = useState<Track2TrainingDataPreview | null>(null);
  const [previewing, setPreviewing] = useState(false);
  const [previewError, setPreviewError] = useState("");
  const shown = selected || loaded;
  useEffect(() => {
    if (!shown) { setPreview(null); return; }
    let cancelled = false;
    setPreviewing(true); setPreviewError(""); setPreview(null);
    previewTrack2TrainingData(shown)
      .then((got) => { if (!cancelled) setPreview(got); })
      .catch((err) => { if (!cancelled) setPreviewError(err instanceof ApiError ? err.detail : "Could not read this file."); })
      .finally(() => { if (!cancelled) setPreviewing(false); });
    return () => { cancelled = true; };
  }, [shown]);
  useEffect(() => {
    Promise.all([getTrack2SettingsFiles(), getTrack2PullControl()]).then(([listing, status]) => {
      setFiles(listing.files.filter(f => f.scope === "dp").map(f => f.name));
      const saved = status.config.training_data_preparation;
      setLoaded(saved?.name || "");
      setLoadedTimes({ loaded_at: saved?.loaded_at, created_at: saved?.created_at, saved_at: saved?.saved_at });
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
    <p className="mt-3 text-sm">
      Loaded: {loaded || "No Data Preparation file loaded"}
      {loaded && (
        <span className="tabular text-ink-secondary">
          {" · "}loaded {loadedTimes.loaded_at ? formatDayTime(loadedTimes.loaded_at) : "(time not recorded -- load it again to record it)"}
          {" · "}file created {loadedTimes.created_at ? formatDayTime(loadedTimes.created_at) : "(not recorded -- saved before creation dates were kept)"}
          {" · "}last saved {loadedTimes.saved_at ? formatDayTime(loadedTimes.saved_at) : "(not recorded)"}
        </span>
      )}
    </p>
    <p className="mt-1 text-sm text-ink-secondary">These are shared data settings, also visible in Data Preparation. Loading does not start a job. Later edits to the data cards affect the next run.</p>
    {error && <p role="alert" className="mt-2 text-danger">{error}</p>}
    {shown && <TrainingDataPreview name={shown} preview={preview} busy={previewing} error={previewError} />}
    {recipe && <details key={`${loaded}:${shown}`} open={false} className="mt-3"><summary className="cursor-pointer">View imported preparation settings</summary>
      <pre className="mt-2 max-h-80 overflow-auto whitespace-pre-wrap text-xs">{JSON.stringify(recipe, null, 2)}</pre>
    </details>}
  </Card>;
}

/** What a Data Preparation file holds for training: its saved class list and data.yaml. */
function TrainingDataPreview({ name, preview, busy, error }: {
  name: string; preview: Track2TrainingDataPreview | null; busy: boolean; error: string;
}) {
  const saved = Object.entries(preview?.label_classes ?? {})
    .sort(([a], [b]) => Number(a) - Number(b));
  const yaml = preview?.yaml;
  const cell = "border-b border-hairline px-2 py-1";
  return (
    <div className="mt-4 border-t border-hairline pt-3 text-sm">
      <p className="font-medium text-ink">In <span className="tabular">{name}</span></p>
      {busy && <p className="mt-1 text-xs text-ink-muted" role="status">Reading the file and building its data.yaml...</p>}
      {error && <p className="mt-1 text-xs text-critical" role="alert">{error}</p>}
      {preview && (
        <div className="mt-2">
          <div>
            <p className="text-xs font-medium text-ink">Classes saved in the file</p>
            {saved.length ? (
              <table className="tabular mt-1 text-xs">
                <thead><tr><th className={`${cell} text-left`}>ID</th><th className={`${cell} text-left`}>Name</th></tr></thead>
                <tbody>{saved.map(([id, n]) => <tr key={id}><td className={cell}>{id}</td><td className={cell}>{n}</td></tr>)}</tbody>
              </table>
            ) : <p className="mt-1 text-xs text-ink-muted">No class list is saved in this file.</p>}

          </div>
        </div>
      )}
      {preview && preview.yaml_error && (
        <p className="mt-3 text-xs text-critical" role="alert">No data.yaml: {preview.yaml_error}</p>
      )}
      {yaml && (
        <div className="mt-3">
          <p className="text-xs text-ink-secondary">
            data.yaml a run would write to <span className="tabular text-ink">s3://{yaml.bucket}/{yaml.key}</span>
            {yaml.image_counts && (
              <> &middot; {(yaml.image_counts.train?.total ?? 0).toLocaleString()} training and{" "}
                {(yaml.image_counts.val?.total ?? 0).toLocaleString()} validation image(s) there now</>
            )}
          </p>
          <p className="mt-1 text-xs text-ink-muted">
            Built from the bucket as it is now; nothing is written.
            {yaml.combined_into && ` It names ${yaml.combined_into}, the decayed copy, which every run brings up to date before training -- so counts here can be lower than the run will see.`}
          </p>
          {yaml.problems.map((p, i) => (
            <p key={`p${i}`} className="mt-1 text-xs text-critical" role="alert">{p.text}</p>
          ))}
          {/* With the filter on, undeclared boxes are dropped on purpose:
              one note, not a red line per folder. Off, each is a real
              refusal and says so. */}
          {yaml.undeclared.length > 0 && (yaml.drop_undeclared ? (
            <p className="mt-1 text-xs text-ink-secondary" role="status">
              {yaml.undeclared.length} folder(s) hold boxes of class ID(s){" "}
              {[...new Set(yaml.undeclared.flatMap((u) => u.indices ?? []))].sort((a, b) => a - b).join(", ")}
              {" "}that this yaml does not name. &ldquo;Drop labels not in the list&rdquo; is on, so
              those boxes are removed at training and the images are kept; the label files in S3 are unchanged.
            </p>
          ) : yaml.undeclared.map((u, i) => (
            <p key={`u${i}`} className="mt-1 text-xs text-critical" role="alert">{u.text}</p>
          )))}
          <pre className="tabular mt-2 max-h-96 overflow-auto rounded border border-hairline bg-page p-2 text-xs">{yaml.text}</pre>
        </div>
      )}
    </div>
  );
}
