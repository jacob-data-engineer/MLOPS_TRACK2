import {SaveSection} from "./SaveSection";
import {saveEntirePage,usePageSave,useNamedPageSave} from "../lib/pageSave";
import { useEffect, useState } from "react";
import { formatDayTime } from "../lib/format";
import { Track2ChainStatusBar } from "./Track2ChainStatusBar";
import { Track2TrainingCostToday } from "./Track2TrainingCostToday";
import {
  ApiError,
  deleteTrack2SettingsFile,
  getTrack2PullControl,
  getTrack2Retrain,
  newTrack2Settings,
  getTrack2SettingsFiles,
  loadTrack2SettingsFile,
  saveTrack2SettingsFile,
  saveTrack2Retrain,
  type Track2SettingsFile,
  type Track2SettingsScope,
  type Track2TrainingTrigger,
} from "../lib/api";


/** Shared settings-file toolbar. Data Preparation shows file controls only;
 * Multi-Auto Training also shows its schedule, GPU status and cost.
 * Save captures all current card values within the page scope.
 */
/** Both readings on this strip -- when the settings were saved, and when the
 * schedule next fires -- are instants the SERVER recorded or computed, so
 * they are shown in the zone the server's clock is set in, with the zone
 * named. */
const savedWhen = formatDayTime;

const SUFFIX: Record<Track2SettingsScope, string> = { dp: "_DP", at: "_AT" };
const TAB: Record<Track2SettingsScope, string> = {
  dp: "Data Preparation",
  at: "Multi-Auto Training",
};

/** `name` as the server will save it: the tab's ending added once. */
function withSuffix(name: string, scope: Track2SettingsScope): string {
  const suffix = SUFFIX[scope];
  return name.toUpperCase().endsWith(suffix) ? name.slice(0, -suffix.length) + suffix : name + suffix;
}

export function Track2PinnedBar({ scope }: { scope: Track2SettingsScope }) {
  const [files, setFiles] = useState<Track2SettingsFile[]>([]);
  const [current, setCurrent] = useState("");
  const [name, setName] = useState("");
  const [picked, setPicked] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  /** New asks first. It rewrites the LIVE settings of this tab, which are
   * what the schedule and the sweep actually read -- not just what is on
   * screen. Delete asks for the same reason. */
  const [confirmNew, setConfirmNew] = useState(false);
  /** Shown only after the server has actually refused for a clash (409),
   * so the tick is an answer to something that was read rather than a
   * standing invitation to skip a check nobody has seen. */
  const [clashed, setClashed] = useState(false);
  const [allowClash, setAllowClash] = useState(false);

  const [trigger, setTrigger] = useState<Track2TrainingTrigger>("off");
  const [trainEvery, setTrainEvery] = useState(1);
  const [trainAt, setTrainAt] = useState("02:00");
  const [trainFrom, setTrainFrom] = useState("");
  const [trainNext, setTrainNext] = useState("");
  const [scheduleDirty, setScheduleDirty] = useState(false);
  const [busy, setBusy] = useState("");
  usePageSave("Training schedule",async()=>{
    if(scope==='at'&&scheduleDirty){await saveTrack2Retrain({trigger,every_days:trainEvery,at:trainAt,from_date:trainFrom});setScheduleDirty(false);}
  });
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  /** The file the settings are in, and the moment it was last written --
   * the FILE's own mtime off the listing, so it is right after a reload
   * and right for a save somebody else made. */
  const [lastSave, setLastSave] = useState<{ name: string; at: string } | null>(null);

  /** `force` re-reads the schedule fields even while they are marked
   * dirty -- straight after Apply, when the server is the truth again. */
  async function refresh(force = false) {
    try {
      const [out, status, retrain] = await Promise.all([
        getTrack2SettingsFiles(),
        getTrack2PullControl().catch(() => null),
        scope === "at" ? getTrack2Retrain().catch(() => null) : Promise.resolve(null),
      ]);
      // This tab's files ONLY. Files from before the split hold both pages
      // and are not offered on either (2026-10-29): a tab's list is that
      // tab's settings and nothing else.
      const mine = out.files.filter((f) => f.scope === scope);
      setFiles(mine);
      const live = String(status?.config[`settings_file_${scope}`] ?? "");
      setCurrent(live);

      const liveFile = mine.find((f) => f.name === live);
      setLastSave(liveFile ? { name: liveFile.name, at: liveFile.modified } : null);

      // The selection must always be a file that EXISTS. A <select>
      // whose value matches no <option> renders the FIRST option while
      // holding the other value -- the box said one name and Load sent
      // another.
      const names = mine.map((f) => f.name);
      setPicked((p) =>
        names.includes(p) ? p : names.includes(live) ? live : names[0] ?? "");

      if (retrain && (force || !scheduleDirty)) {
        setTrigger(retrain.trigger === "clock" ? "clock" : "off");
        setTrainEvery(Number(retrain.settings.every_days ?? 7) || 7);
        setTrainAt(String(retrain.settings.at ?? "02:00"));
        setTrainFrom(String(retrain.settings.from_date ?? ""));
        setTrainNext(retrain.trigger === "clock" ? retrain.next_runs?.[0] || "" : "");
      }
    } catch {
      // A bar that cannot load its own state is not worth an error
      // banner across the top of every page.
    }
  }

  useEffect(() => { void refresh(); }, []);

  async function run(label: string, work: () => Promise<void>) {
    setBusy(label);
    setError("");
    setNote("");
    try {
      await work();
    } catch (err) {
      setError(err instanceof Error ? err.message : `${label} failed.`);
    } finally {
      setBusy("");
    }
  }

  const currentExists = files.some((f) => f.name === current);

  /** Save-as versus save-over, decided by whether a name has been typed.
   * The label says which, with the ending the server will add. */
  const typed = name.trim();
  const canSave = Boolean(typed || currentExists);

  async function saveNamedSettings():Promise<string> {
    if(busy) throw new Error("A settings operation is already in progress.");
    setBusy("Save");
    try {
            const target = typed || (currentExists ? current : "");
            if (!target) {
              throw new Error("Type a name at the top -- there is no current file to save over.");
            }
            // Capture all current card values before writing the named file.
            await saveEntirePage();

            try {
              await saveTrack2SettingsFile(target, allowClash, scope);
            } catch (err) {
              if (err instanceof ApiError && err.status === 409) {
                setClashed(true);
              }
              throw err;
            }
            const saved = withSuffix(target, scope);
            setClashed(false);
            setAllowClash(false);
            setName("");
            setNote(`Saved all page settings to ${saved}. Edits are saved only when you press Save.`);
            await refresh();
      return saved;
    } finally {setBusy("");}
  }
  useNamedPageSave(saveNamedSettings);

  return (
    /* No sticky here: the page pins the whole band -- title and toolbar
       together -- and a second sticky context inside the first just
       fights it. This is a plain strip. */
    <SaveSection><div className="rounded-md border border-hairline bg-surface px-2.5 py-2">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 text-xs">
        {/* ---- start a fresh one ---- First in the row, because it is
             where a new file begins: New, then name it, then Save. */}
        {confirmNew ? (
          <>
            <span className="text-critical">
              Reset {TAB[scope]}&rsquo;s settings to defaults? The settings in
              force on this tab are replaced; the other tab is untouched.
            </span>
            <button
              type="button"
              disabled={!!busy}
              onClick={() => void run("New", async () => {
                const out = await newTrack2Settings(scope);
                setConfirmNew(false);
                setNote(out.problems?.length
                  ? out.problems.join(" ")
                  : "Blank template loaded. Name it and press Save. Refreshing...");
                // Every card read the old settings when it opened, so
                // without this they go on showing them while the server
                // holds the defaults -- the same reason Load reloads.
                window.setTimeout(() => window.location.reload(), 1800);
              })}
              className="rounded-md bg-critical px-2.5 py-1 font-medium text-white"
            >
              Yes, start fresh
            </button>
            <button type="button" onClick={() => setConfirmNew(false)}
                    className="text-ink-secondary underline">cancel</button>
          </>
        ) : (
          <button
            type="button"
            disabled={!!busy}
            onClick={() => setConfirmNew(true)}
            className="rounded-md border border-black/15 px-2.5 py-1 font-medium text-ink-secondary disabled:opacity-40"
          >
            {busy === "New" ? "Resetting..." : "New"}
          </button>
        )}

        {/* ---- the destination ---- Type a name to save a COPY under it;
             leave it empty to write back over the file you are in. */}
        <span className="font-medium text-ink">Save as</span>
        <input
          className="tabular min-w-0 w-36 flex-1 rounded border border-black/15 bg-page px-2 py-1"
          placeholder={`name (saved as name${SUFFIX[scope]})`}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button
          type="button"
          disabled={!!busy || !canSave}
          onClick={() => void run("Save", async () => {
            await saveNamedSettings();
          })}
          className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
        >
          {busy === "Save"
            ? "Saving..."
            : typed ? `Save as ${withSuffix(typed, scope)}` : "Save"}
        </button>
        {!canSave && (
          <span className="text-ink-muted">
            type a name &mdash; no file is loaded to save over
          </span>
        )}

        {files.length > 0 && (
          <>
            <select
              className="tabular min-w-0 w-44 flex-1 lg:ml-8 rounded border border-black/15 bg-page px-2 py-1"
              value={picked}
              onChange={(e) => { setPicked(e.target.value); setConfirmDelete(false); }}
            >
              {files.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.name}
                </option>
              ))}
            </select>
            <button
              type="button"
              disabled={!!busy || !picked}
              onClick={() => void run("Load", async () => {
                const out = await loadTrack2SettingsFile(picked, true, scope);
                // Say what the registry did: a detector appearing on this
                // machine because a file was loaded is worth naming, and
                // one LEFT ALONE matters just as much.
                const made = out.registry?.created ?? [];
                const kept = out.registry?.kept ?? [];
                const registry = made.length
                  ? ` Added ${made.length} detector${made.length === 1 ? "" : "s"}: ${made.join(", ")}.`
                  : "";
                const untouched = kept.length
                  ? ` ${kept.length} already here, left as ${kept.length === 1 ? "it is" : "they are"}.`
                  : "";
                setNote(`Loaded ${out.name}.${registry}${untouched} Refreshing...`);
                window.setTimeout(() => window.location.reload(), 1100);
              })}
              className="rounded-md border border-black/15 px-2.5 py-1 font-medium text-ink-secondary disabled:opacity-40"
            >
              {busy === "Load" ? "Loading..." : "Load"}
            </button>
            {confirmDelete ? (
              <>
                <span className="text-critical">delete {picked}?</span>
                <button
                  type="button"
                  disabled={!!busy}
                  onClick={() => void run("Delete", async () => {
                    await deleteTrack2SettingsFile(picked);
                    setConfirmDelete(false);
                    setNote(`Deleted ${picked}.`);
                    await refresh();
                  })}
                  className="rounded-md bg-critical px-2.5 py-1 font-medium text-white"
                >
                  Yes
                </button>
                <button type="button" onClick={() => setConfirmDelete(false)}
                        className="text-ink-secondary underline">cancel</button>
              </>
            ) : (
              <button
                type="button"
                disabled={!!busy || !picked}
                onClick={() => setConfirmDelete(true)}
                className="font-medium text-critical underline disabled:opacity-40"
              >
                Delete
              </button>
            )}
          </>
        )}

        {note && <span className="text-ink-secondary">{note}</span>}
        {error && <span className="text-critical">{error}</span>}
        {clashed && (
          <label className="flex items-center gap-1.5 text-critical">
            <input
              type="checkbox"
              className="shrink-0"
              checked={allowClash}
              onChange={(e) => setAllowClash(e.target.checked)}
            />
            <span className="font-medium">Ignore this and save anyway</span>
          </label>
        )}
      </div>

      {/* ---- STATUS LINE: what was written, and what is in force. ---- */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
        {lastSave ? (
          <span className="tabular text-ink-secondary">
            Saved <span className="font-medium text-ink">{lastSave.name}</span>
            {lastSave.at && <> &middot; {savedWhen(lastSave.at)}</>}
          </span>
        ) : (
          <span className="text-ink-muted">Not saved yet</span>
        )}
        <span className="tabular text-ink-secondary">
          Loaded:{" "}
          {currentExists ? (
            <span className="font-medium text-ink">{current}</span>
          ) : (
            <span className="text-ink-muted">no file loaded</span>
          )}
        </span>
      </div>

      {scope === "at" && <div className="mt-1.5 border-t border-hairline pt-1.5 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <span className="font-medium text-ink">TRAINING SCHEDULE</span>
          <div className="flex overflow-hidden rounded-md border border-black/15">
            {([["off", "Off"], ["clock", "On a schedule"]] as const).map(([value, label]) => (
              <button key={value} type="button" disabled={!!busy}
                onClick={() => { setTrigger(value); setScheduleDirty(true); }}
                className={`px-2.5 py-1 font-medium ${trigger === value ? "bg-accent text-white" : "bg-page text-ink-secondary"}`}>
                {label}
              </button>
            ))}
          </div>
          <span className="text-ink-muted">Pulls follow the training schedule.</span>
        </div>
        {trigger === "clock" && <div className="mt-2 grid grid-cols-1 gap-4 md:grid-cols-3">
          <label className="flex min-w-0 items-center gap-2">Every <input type="number" min={1} aria-label="Training interval in days"
            value={trainEvery} onChange={e => { setTrainEvery(Number(e.target.value) || 1); setScheduleDirty(true); }}
            className="min-w-0 flex-1 rounded border border-hairline bg-page px-2 py-1" /> day(s)</label>
          <label className="flex min-w-0 items-center gap-2">At (Singapore time) <input type="time" aria-label="Training time"
            value={trainAt} onChange={e => { setTrainAt(e.target.value); setScheduleDirty(true); }}
            className="min-w-0 flex-1 rounded border border-hairline bg-page px-2 py-1" /></label>
          <label className="flex min-w-0 items-center gap-2">Start date (optional) <input type="date" aria-label="Training start date"
            value={trainFrom} onChange={e => { setTrainFrom(e.target.value); setScheduleDirty(true); }}
            className="min-w-0 flex-1 rounded border border-hairline bg-page px-2 py-1" /></label>
        </div>}
        <div className="mt-2 flex flex-wrap items-center justify-between gap-3 text-ink-muted">
          <span>{trigger === "clock"
            ? "Pull fresh data, rebuild the decayed dataset, preserve it, then queue and train. Saving after the selected time schedules the next occurrence."
            : "Automatic pulls and training are off."}</span>
          {scheduleDirty ? <button type="button" disabled={!!busy}
            onClick={() => void run("Apply", async () => {
              await saveTrack2Retrain({ trigger, every_days: trainEvery, at: trainAt, from_date: trainFrom });
              setScheduleDirty(false);
              setNote("Training schedule saved. Pulls use this schedule too.");
              await refresh(true);
            })}
            className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40">
            {busy === "Apply" ? "Applying..." : "Apply schedule"}
          </button> : trigger === "clock" && trainNext && <span>Next run: {savedWhen(trainNext)}</span>}
        </div>
      </div>}
      {/* ---- THIRD ROW: what the chain is doing right now ---- */}
      {scope === "at" && <Track2ChainStatusBar />}
      {/* Training spend belongs with the training schedule. */}
      {scope === "at" && <Track2TrainingCostToday />}
    </div></SaveSection>
  );
}
