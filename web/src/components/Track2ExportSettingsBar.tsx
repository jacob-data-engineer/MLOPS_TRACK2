import { useEffect, useState } from "react";
import {
  ApiError,
  deleteTrack2ExportSettings,
  listTrack2ExportSettings,
  loadTrack2ExportSettings,
  saveTrack2ExportSettings,
  type Track2ExportSettings,
  type Track2ExportSettingsFile,
} from "../lib/api";

/** The Review & Metrics page's saved-settings toolbar.
 *
 * Pinned at the top of the page rather than inside the card it governs,
 * the same way Multi-Auto Training's is: a settings file is about the
 * page, so it belongs where the page's name is. Buried in a card halfway
 * down, "which file am I in" was a question you had to scroll to answer.
 *
 * The settings themselves live on the PAGE and are passed in, because
 * the export card edits the same values. Two copies -- one here, one
 * there -- would need keeping in step, and the first time they drifted
 * Save would write something nobody was looking at.
 *
 * Its own file list, never the pull-control one. That holds a whole
 * retraining configuration and loading it rearms a schedule; this holds
 * which metrics, which versions and where they go.
 */
export function Track2ExportSettingsBar({
  settings,
  onChange,
  defaults,
}: {
  settings: Track2ExportSettings;
  onChange: (next: Track2ExportSettings) => void;
  /** What New goes back to. Held by the page so "blank" means one thing
   * wherever it is pressed. */
  defaults: Track2ExportSettings;
}) {
  const [files, setFiles] = useState<Track2ExportSettingsFile[]>([]);
  const [writable, setWritable] = useState(true);
  const [saveAs, setSaveAs] = useState("");
  const [pick, setPick] = useState("");
  const [loaded, setLoaded] = useState("");
  const [savedAt, setSavedAt] = useState("");
  const [busy, setBusy] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState("");

  function refreshFiles() {
    listTrack2ExportSettings()
      .then((r) => {
        setFiles(r.files);
        setWritable(r.writable);
        setPick((current) => current || r.files[0]?.name || "");
      })
      .catch(() => undefined);
  }
  useEffect(refreshFiles, []);

  async function withBusy(label: string, work: () => Promise<void>) {
    setBusy(label);
    setNote("");
    setError("");
    try {
      await work();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "That did not work.");
    } finally {
      setBusy("");
    }
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <button
        type="button"
        onClick={() => {
          onChange(defaults);
          setLoaded("");
          setSavedAt("");
          setNote("Blank settings. Nothing saved until you press Save.");
        }}
        className="rounded-md border border-black/15 px-2.5 py-1 font-medium text-ink-secondary"
      >
        New
      </button>

      <span className="text-ink-secondary">Save as</span>
      <input
        type="text"
        value={saveAs}
        onChange={(e) => setSaveAs(e.target.value)}
        placeholder="new file name"
        className="w-40 rounded-sm border border-black/15 bg-page px-2 py-1 text-xs text-ink placeholder:text-ink-muted"
      />
      <button
        type="button"
        disabled={!!busy || !writable || !(saveAs.trim() || loaded)}
        onClick={() => withBusy("Saving...", async () => {
          // Typing a name saves AS that name; an empty box saves over the
          // file that is loaded. A Save that silently invented a name
          // would scatter near-identical files.
          const name = saveAs.trim() || loaded;
          await saveTrack2ExportSettings(name, settings);
          setLoaded(name);
          setSaveAs("");
          setSavedAt(new Date().toLocaleString());
          setNote(`Saved ${name}`);
          refreshFiles();
        })}
        className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
      >
        Save
      </button>
      <span className="text-ink-muted">
        {saveAs.trim()
          ? `saves as ${saveAs.trim()}`
          : loaded
            ? `saves over ${loaded}`
            : "type a name -- no file is loaded to save over"}
      </span>

      <select
        value={pick}
        onChange={(e) => setPick(e.target.value)}
        disabled={files.length === 0}
        className="ml-auto min-w-[10rem] rounded-sm border border-black/15 bg-page px-2 py-1 text-xs text-ink disabled:opacity-40"
      >
        {files.length === 0 && <option value="">no saved files</option>}
        {files.map((f) => <option key={f.name} value={f.name}>{f.name}</option>)}
      </select>
      <button
        type="button"
        disabled={!!busy || !pick}
        onClick={() => withBusy("Loading...", async () => {
          const got = await loadTrack2ExportSettings(pick);
          onChange(got.settings);
          setLoaded(got.name);
          setSavedAt(got.saved_at ? new Date(got.saved_at).toLocaleString() : "");
          setNote(`Loaded ${got.name}`);
        })}
        className="rounded-md border border-black/15 px-2.5 py-1 font-medium text-ink-secondary disabled:opacity-40"
      >
        Load
      </button>
      <button
        type="button"
        disabled={!!busy || !pick || !writable}
        onClick={() => withBusy("Deleting...", async () => {
          await deleteTrack2ExportSettings(pick);
          if (loaded === pick) { setLoaded(""); setSavedAt(""); }
          setNote(`Deleted ${pick}`);
          setPick("");
          refreshFiles();
        })}
        className="rounded-md px-2.5 py-1 font-medium text-critical underline disabled:opacity-40"
      >
        Delete
      </button>

      <span className="w-full text-ink-muted">
        {savedAt ? `Saved ${savedAt}` : "Not saved yet"}
        <span className="ml-3">Loaded: {loaded || "no file loaded"}</span>
        {note && <span className="ml-3 text-ink-secondary">{note}</span>}
        {error && <span className="ml-3 text-critical">{error}</span>}
        {!writable && (
          <span className="ml-3 text-critical">
            The settings folder is not writable, so nothing can be saved.
          </span>
        )}
      </span>
    </div>
  );
}
