import { useEffect, useState } from "react";
import { apiGet, apiPost, ApiError } from "../lib/api";

export type SqlPageSettings = {
  export_mode: "detection" | "classification";
  detector_names: string[]; classifier_names: string[];
  database: string; preset_class: string; preset_scope: "resolved" | "ready" | "test";
  pull_bucket: string; pull_folder: string; train_percent: number;
  show_labels: boolean; with_labels: boolean;
  exploration_hazard: string; exploration_open: boolean; frames_open: boolean;
};
export const SQL_DEFAULTS: SqlPageSettings = {
  export_mode: "detection",
  detector_names: ["helmet", "gloves", "person", "working_boots", "man_cage", "safe_area"],
  classifier_names: ["helmet_worn", "no_helmet"],
  database: "violation-review", preset_class: "", preset_scope: "resolved",
  pull_bucket: "track2-s3", pull_folder: "", train_percent: 80,
  show_labels: true, with_labels: true, exploration_hazard: "all", exploration_open: true, frames_open: true,
};
export type SqlSavedPreset = { name: string; saved_at: string; settings: SqlPageSettings };

export function Track2SqlSettingsBar({ settings, onLoad, disabled, initialPreset }: {
  initialPreset: SqlSavedPreset | null;
  settings: SqlPageSettings; onLoad: (settings: SqlPageSettings) => void; disabled: boolean;
}) {
  const [files, setFiles] = useState<{name: string}[]>([]);
  const [name, setName] = useState("");
  const [selected, setSelected] = useState("");
  const [loaded, setLoaded] = useState("");
  const [savedAt, setSavedAt] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  useEffect(() => {
    if (!initialPreset) return;
    setLoaded(initialPreset.name); setSelected(initialPreset.name); setSavedAt(initialPreset.saved_at);
    setMessage(`Loaded latest saved settings: ${initialPreset.name}`);
  }, [initialPreset]);
  async function refresh() {
    const result = await apiGet<{files: {name: string}[]}>("/track2/sql/settings");
    setFiles(result.files);
  }
  useEffect(() => { void refresh().catch(() => setError("Could not list saved settings.")); }, []);
  async function work(fn: () => Promise<void>) {
    setBusy(true); setError(""); setMessage("");
    try { await fn(); } catch (err) { setError(err instanceof ApiError ? err.detail : "Could not complete the settings operation."); }
    finally { setBusy(false); }
  }
  const locked = disabled || busy;
  const saveName = name.trim() ? name.trim().replace(/_sql$/i, "") + "_sql" : loaded;
  const button = "rounded-md border border-black/15 px-2.5 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40";
  return <section className="mb-4 rounded border border-hairline bg-surface px-2.5 py-2" aria-label="Database query saved settings">
    <div className="flex flex-wrap items-center gap-2">
      <button className={button} disabled={locked} onClick={() => { onLoad(SQL_DEFAULTS); setLoaded(""); setSavedAt(""); setName(""); setMessage(""); setError(""); }}>New</button>
      <label className="text-xs">Save as <input aria-label="SQL settings name" value={name} disabled={locked} onChange={e => setName(e.target.value)} placeholder="name (saved as name_sql)" className="ml-2 w-44 rounded-sm border border-black/15 bg-page px-2 py-1 text-xs" /></label>
      <button className="rounded-md bg-accent px-2.5 py-1 text-xs font-semibold text-white disabled:opacity-40" disabled={locked || !(name.trim() || loaded)} onClick={() => void work(async () => {
        const got = await apiPost<{name: string; path: string; saved_at: string}>("/track2/sql/settings/save", {name: name.trim() || loaded, settings});
        setLoaded(got.name); setSelected(got.name); setName(""); setSavedAt(got.saved_at); setMessage(`Saved ${got.name}`); await refresh();
      })}>{name.trim() ? `Save as ${saveName}` : "Save"}</button>
      {!name.trim() && <span className="text-xs text-ink-muted">{loaded ? `saves over ${loaded}` : "type a name — no file is loaded to save over"}</span>}
      <select aria-label="Saved SQL settings" className="rounded-sm border border-black/15 bg-page px-2 py-1 text-xs" value={selected} disabled={locked} onChange={e => setSelected(e.target.value)}>
        <option value="">Choose saved settings</option>{files.map(f => <option key={f.name} value={f.name}>{f.name}</option>)}
      </select>
      <button className={button} disabled={locked || !selected} onClick={() => void work(async () => {
        const got = await apiPost<{name: string; settings: SqlPageSettings; saved_at: string}>("/track2/sql/settings/load", {name: selected});
        onLoad(got.settings); setSavedAt(got.saved_at); setLoaded(got.name); setName(""); setMessage(`Loaded ${got.name}`);
      })}>Load</button>
      <button className="px-1 text-xs text-critical underline disabled:opacity-40" disabled={locked || !selected} onClick={() => void work(async () => {
        await apiPost("/track2/sql/settings/delete", {name: selected});
        if (loaded === selected) { setLoaded(""); setSavedAt(""); } setSelected(""); setMessage(`Deleted ${selected}`); await refresh();
      })}>Delete</button>
    </div>
    <p className="mt-1.5 text-xs text-ink-muted">{savedAt ? `Saved ${new Date(savedAt).toLocaleString()}` : "Not saved yet"}<span className="ml-5 font-mono">Loaded: {loaded || "no file loaded"}</span></p>
    {message && <p role="status" className="mt-1 text-xs">{message}</p>}
    {error && <p role="alert" className="mt-1 text-xs text-critical">{error}</p>}
  </section>;
}
