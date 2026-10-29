import { useEffect, useState } from "react";
import { formatDayTime } from "../lib/format";
import { Track2ChainStatusBar } from "./Track2ChainStatusBar";
import { Track2TrainingCostToday } from "./Track2TrainingCostToday";
import {
  ApiError,
  deleteTrack2SettingsFile,
  getTrack2PullControl,
  getTrack2QueueAfterPull,
  getTrack2Retrain,
  newTrack2Settings,
  getTrack2SettingsFiles,
  loadTrack2SettingsFile,
  saveTrack2SettingsFile,
  saveTrack2QueueAfterPull,
  saveTrack2Retrain,
  setTrack2PullMode,
  type Track2SettingsFile,
  type Track2SettingsScope,
  type Track2TrainingTrigger,
} from "../lib/api";
import { pendingCards } from "../lib/pendingChanges";

/** The two things you reach for from anywhere on the page, pinned: the
 * settings file and the schedule.
 *
 * ONE PER TAB (2026-10-29). Data Preparation saves `<name>_DP` and owns the
 * PULL schedule; Multi-Auto Training saves `<name>_AT` and owns the
 * TRAINING schedule. Each file holds only its own page -- the server keeps
 * the list of which settings are whose (`pull_control.SCOPES`) -- so
 * saving one tab can no longer write the other's choices into its file,
 * and loading one cannot change the other.
 *
 * It saves what is IN EFFECT, not what is typed into a card and not yet
 * applied. A card with an Apply button has to be applied first.
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

  // The PULL schedule (Data Preparation): off, on arrival, on a clock.
  const [mode, setMode] = useState<"off" | "arrival" | "schedule">("off");
  const [everyDays, setEveryDays] = useState(1);
  const [timeOfDay, setTimeOfDay] = useState("");
  // Where "every N days" counts from: today, or a chosen date whose grid
  // the pulls then keep. Saved server-side with the rest of the schedule.
  const [countFrom, setCountFrom] = useState<"today" | "date">("today");
  const [startDate, setStartDate] = useState("");
  // The TRAINING schedule (Multi-Auto Training): after each pull, its own
  // clock, or nothing.
  const [trigger, setTrigger] = useState<Track2TrainingTrigger>("pull");
  const [trainEvery, setTrainEvery] = useState(7);
  const [trainAt, setTrainAt] = useState("02:00");
  const [pullNext, setPullNext] = useState("");
  const [trainNext, setTrainNext] = useState("");
  const [scheduleDirty, setScheduleDirty] = useState(false);
  // The arrival trigger's guards: smallest batch worth a run, and the
  // shortest gap between two runs. Training settings, so they sit with
  // the training schedule.
  const [minFiles, setMinFiles] = useState(1);
  const [cooldown, setCooldown] = useState(0);

  const [busy, setBusy] = useState("");
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
      const [out, status, after, retrain] = await Promise.all([
        getTrack2SettingsFiles(),
        getTrack2PullControl().catch(() => null),
        getTrack2QueueAfterPull().catch(() => null),
        getTrack2Retrain().catch(() => null),
      ]);
      setMinFiles(Number(after?.settings.min_files ?? 1) || 1);
      setCooldown(Number(after?.settings.cooldown_minutes ?? 0) || 0);
      // This tab's files, and the ones from before the split -- those
      // hold both pages, and loading one here takes only this tab's half.
      const mine = out.files.filter((f) => f.scope === scope || !f.scope);
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

      if (status && (force || !scheduleDirty)) {
        const c = status.config;
        setMode((c.mode as typeof mode) || "off");
        setEveryDays(Number(c.schedule_every_days ?? 1) || 1);
        setCountFrom(c.schedule_from === "date" ? "date" : "today");
        setStartDate(String(c.schedule_start_date ?? ""));
        setTimeOfDay(String(c.schedule_time ?? ""));
        const rule = status.triggers?.schedule;
        setPullNext(rule?.enabled ? (rule.next_run || rule.schedule || "") : "");
      }
      if (retrain && (force || !scheduleDirty)) {
        setTrigger(retrain.trigger ?? "pull");
        setTrainEvery(Number(retrain.settings.every_days ?? 7) || 7);
        setTrainAt(String(retrain.settings.at ?? "02:00"));
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
      setError(err instanceof ApiError ? err.detail : `${label} failed.`);
    } finally {
      setBusy("");
    }
  }

  const currentExists = files.some((f) => f.name === current);

  /** Save-as versus save-over, decided by whether a name has been typed.
   * The label says which, with the ending the server will add. */
  const typed = name.trim();
  const canSave = Boolean(typed || currentExists);
  const pullOn = mode === "schedule" || mode === "arrival";

  return (
    /* No sticky here: the page pins the whole band -- title and toolbar
       together -- and a second sticky context inside the first just
       fights it. This is a plain strip. */
    <div className="rounded-md border border-hairline bg-surface px-2.5 py-2">
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
          className="tabular w-36 rounded border border-black/15 bg-page px-2 py-1"
          placeholder={`name (saved as name${SUFFIX[scope]})`}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button
          type="button"
          disabled={!!busy || !canSave}
          onClick={() => void run("Save", async () => {
            const target = typed || (currentExists ? current : "");
            if (!target) {
              setError("Type a name -- there is no current file to save over.");
              return;
            }
            // A file is a snapshot of the settings IN FORCE, so a card
            // whose Apply has not been pressed is not in it. Said after
            // the write rather than refused over.
            const unapplied = pendingCards();
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
            setNote(unapplied.length
              ? `Saved ${saved}. ${unapplied.join(" and ")} has not had Apply `
                + "pressed, so the trigger is still set to what it was."
              : `Saved ${saved}.`);
            await refresh();
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
              className="tabular w-44 rounded border border-black/15 bg-page px-2 py-1"
              value={picked}
              onChange={(e) => { setPicked(e.target.value); setConfirmDelete(false); }}
            >
              {files.map((f) => (
                <option key={f.name} value={f.name}>
                  {f.scope ? f.name : `${f.name} (old, both tabs)`}
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
                const half = files.find((f) => f.name === picked)?.scope
                  ? "" : ` (only its ${TAB[scope]} settings)`;
                setNote(`Loaded ${out.name}${half}.${registry}${untouched} Refreshing...`);
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

      {/* ---- SECOND ROW: this tab's own schedule ---- */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-hairline pt-1.5 text-xs">
        {scope === "dp" ? (
          <>
            <span className="font-medium text-ink">PULL SCHEDULE</span>
            <div className="flex overflow-hidden rounded-md border border-black/15">
              {([
                ["off", "Off"],
                ["arrival", "When files arrive"],
                ["schedule", "On a schedule"],
              ] as const).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  disabled={!!busy}
                  onClick={() => { setMode(value); setScheduleDirty(true); }}
                  className={`px-2.5 py-1 font-medium disabled:opacity-40 ${
                    mode === value
                      ? "bg-accent text-white"
                      : "bg-page text-ink-secondary hover:text-ink"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {mode === "schedule" && (
              <>
                <label className="flex items-center gap-1 text-ink-muted">
                  every
                  <input
                    type="number"
                    min={1}
                    className="tabular w-12 rounded border border-black/15 bg-page px-1 py-1"
                    value={everyDays}
                    onChange={(e) => {
                      setEveryDays(Number(e.target.value) || 1);
                      setScheduleDirty(true);
                    }}
                  />
                  day(s) at
                  <input
                    className="tabular w-16 rounded border border-black/15 bg-page px-1 py-1"
                    placeholder="18:33"
                    value={timeOfDay}
                    onChange={(e) => { setTimeOfDay(e.target.value); setScheduleDirty(true); }}
                  />
                  SGT
                </label>
                {/* Where the N days count from. "Every N days" re-anchors on
                    each Apply; "from this date" keeps the grid that starts on
                    the chosen day, so the pull days do not move. */}
                <div className="flex items-center gap-2 text-ink-muted">
                  <label className="flex items-center gap-1">
                    <input type="radio" name="pull-count-from" checked={countFrom === "today"}
                      onChange={() => { setCountFrom("today"); setScheduleDirty(true); }} />
                    every {everyDays} day(s)
                  </label>
                  <label className="flex items-center gap-1">
                    <input type="radio" name="pull-count-from" checked={countFrom === "date"}
                      onChange={() => { setCountFrom("date"); setScheduleDirty(true); }} />
                    every {everyDays} day(s) from
                  </label>
                  <input
                    type="date"
                    aria-label="Count the schedule from this date"
                    className="tabular rounded border border-black/15 bg-page px-1 py-0.5 disabled:opacity-40"
                    value={startDate}
                    disabled={countFrom !== "date"}
                    onChange={(e) => { setStartDate(e.target.value); setScheduleDirty(true); }}
                  />
                </div>
              </>
            )}
            {scheduleDirty ? (
              <button
                type="button"
                disabled={!!busy}
                onClick={() => void run("Apply", async () => {
                  const res = await setTrack2PullMode(mode, {
                    everyDays, timeOfDay: mode === "schedule" ? timeOfDay : "",
                    from: countFrom, startDate,
                  });
                  setScheduleDirty(false);
                  setNote(res.problems?.length
                    ? res.problems.join(" ")
                    : `Pull is now ${mode === "schedule"
                        ? `every ${everyDays} day(s) at ${timeOfDay} SGT${
                            countFrom === "date" && startDate ? ` from ${startDate}` : ""}` : mode}.`);
                  await refresh(true);
                })}
                className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
              >
                {busy === "Apply" ? "Applying..." : "Apply"}
              </button>
            ) : pullNext ? (
              <span className="tabular text-ink-muted">
                next pull: {pullNext.includes("T") ? savedWhen(pullNext) : pullNext}
              </span>
            ) : null}
            {/* What the pull sets off is Multi-Auto Training's choice, so
                say which it is rather than leave it to be guessed. */}
            {!pullOn ? (
              <span className="text-ink-muted">nothing is pulled on its own</span>
            ) : trigger === "pull" ? (
              <span className="text-ink-secondary">
                each pull also builds, queues and trains &mdash; Multi-Auto
                Training follows the pull
              </span>
            ) : (
              <span className="text-ink-secondary">
                pulls data only &mdash; training runs on Multi-Auto
                Training&rsquo;s own schedule
              </span>
            )}
          </>
        ) : (
          <>
            <span className="font-medium text-ink">TRAINING SCHEDULE</span>
            <div className="flex overflow-hidden rounded-md border border-black/15">
              {([
                ["off", "Off"],
                ["pull", "After each data pull"],
                ["clock", "On its own schedule"],
              ] as const).map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  disabled={!!busy}
                  onClick={() => { setTrigger(value); setScheduleDirty(true); }}
                  className={`px-2.5 py-1 font-medium disabled:opacity-40 ${
                    trigger === value
                      ? "bg-accent text-white"
                      : "bg-page text-ink-secondary hover:text-ink"
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
            {trigger === "clock" && (
              <label className="flex items-center gap-1 text-ink-muted">
                every
                <input
                  type="number"
                  min={1}
                  className="tabular w-12 rounded border border-black/15 bg-page px-1 py-1"
                  value={trainEvery}
                  onChange={(e) => {
                    setTrainEvery(Number(e.target.value) || 1);
                    setScheduleDirty(true);
                  }}
                />
                day(s) at
                <input
                  className="tabular w-16 rounded border border-black/15 bg-page px-1 py-1"
                  placeholder="02:00"
                  value={trainAt}
                  onChange={(e) => { setTrainAt(e.target.value); setScheduleDirty(true); }}
                />
                SGT
              </label>
            )}
            {scheduleDirty ? (
              <button
                type="button"
                disabled={!!busy}
                onClick={() => void run("Apply", async () => {
                  await saveTrack2Retrain(trigger === "clock"
                    ? { trigger, every_days: trainEvery, at: trainAt }
                    : { trigger });
                  setScheduleDirty(false);
                  setNote(trigger === "clock"
                    ? `Training is now every ${trainEvery} day(s) at ${trainAt} SGT.`
                    : trigger === "pull"
                      ? "Training now follows each data pull."
                      : "Nothing trains on its own now.");
                  await refresh(true);
                })}
                className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
              >
                {busy === "Apply" ? "Applying..." : "Apply"}
              </button>
            ) : trigger === "clock" && trainNext ? (
              <span className="tabular text-ink-muted">
                next run: {trainNext.includes("T") ? savedWhen(trainNext) : trainNext}
              </span>
            ) : trigger === "pull" && pullOn && pullNext ? (
              <span className="tabular text-ink-muted">
                next pull: {pullNext.includes("T") ? savedWhen(pullNext) : pullNext}
              </span>
            ) : null}
            {trigger === "off" && (
              <span className="text-ink-muted">nothing trains on its own</span>
            )}
            {trigger === "clock" && (
              <span className="text-ink-secondary">
                builds, queues and trains on this clock, from the data as it stands
              </span>
            )}
            {trigger === "pull" && !pullOn && (
              <span className="text-critical">
                Data Preparation&rsquo;s pull is off, so nothing will train &mdash;
                turn the pull on there, or pick &ldquo;On its own schedule&rdquo;
              </span>
            )}
            {trigger === "pull" && mode === "schedule" && (
              <span className="text-ink-secondary">
                builds, queues and trains after each scheduled pull
              </span>
            )}
            {/* Arrival fires once per upload burst. Never offered without
                its two guards: on 6 October three arrival pulls fired in ten
                minutes, one moving a single file, and each would have been a
                GPU run. Saved on blur. */}
            {trigger === "pull" && mode === "arrival" && (
              <>
                <label className="flex items-center gap-1 text-ink-muted">
                  trains when a pull brings at least
                  <input
                    type="number"
                    min={1}
                    key={`min-${minFiles}`}
                    defaultValue={minFiles}
                    disabled={!!busy}
                    className="tabular w-16 rounded border border-black/15 bg-page px-1 py-1"
                    onBlur={(e) => {
                      const value = Math.max(1, Number(e.target.value) || 1);
                      void run("Saving", async () => {
                        await saveTrack2QueueAfterPull({ min_files: value });
                        setMinFiles(value);
                      });
                    }}
                  />
                  file(s), at most one run every
                  <input
                    type="number"
                    min={0}
                    key={`cool-${cooldown}`}
                    defaultValue={cooldown}
                    disabled={!!busy}
                    className="tabular w-14 rounded border border-black/15 bg-page px-1 py-1"
                    onBlur={(e) => {
                      const value = Math.max(0, Number(e.target.value) || 0);
                      void run("Saving", async () => {
                        await saveTrack2QueueAfterPull({ cooldown_minutes: value });
                        setCooldown(value);
                      });
                    }}
                  />
                  min
                </label>
                {minFiles <= 1 && cooldown === 0 && (
                  <span className="text-critical">
                    every upload will start a GPU run -- set a batch size and a wait
                  </span>
                )}
              </>
            )}
          </>
        )}
      </div>

      {/* ---- THIRD ROW: what the chain is doing right now ---- */}
      <Track2ChainStatusBar />
      {/* Training spend belongs with the training schedule. */}
      {scope === "at" && <Track2TrainingCostToday />}
    </div>
  );
}
