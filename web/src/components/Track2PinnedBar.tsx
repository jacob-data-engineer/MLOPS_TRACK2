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
  setTrack2PullMode,
  type Track2SettingsFile,
} from "../lib/api";
import { pendingCards } from "../lib/pendingChanges";

/** The two things you reach for from anywhere on the page, pinned.
 *
 * The settings file and the pull schedule were both buried: the file
 * controls sat at the bottom of a card five sections down, and the
 * schedule inside Pull Control's form. Saving meant scrolling past
 * everything you had just set, which is how a page teaches people not
 * to save.
 *
 * ONE file and ONE schedule, so there is nothing per-card to reconcile.
 * Every setting on the page lives in the pull config, so saving from
 * here saves all of it -- there was never a per-card scope, only a
 * per-card button, and five buttons for one action is what made it
 * look like there was.
 *
 * It saves what is IN EFFECT, not what is typed into a card and not yet
 * applied. A card with an Apply button has to be applied first.
 */
/** An ISO timestamp as "2026-10-02 10:30", in the reader's own zone.
 *
 * Local rather than UTC because this is read against the clock on the
 * wall next to you -- "did my save land" is a question about the last
 * minute, and an eight-hour offset makes a fresh save look like
 * yesterday's. Falls back to the raw string rather than rendering
 * "Invalid Date" if the server ever sends something unexpected. */
/** Both readings on this strip -- when the settings were saved, and when the
 * schedule next fires -- are instants the SERVER recorded or computed, so
 * they are shown in the zone the server's clock is set in, with the zone
 * named.
 *
 * This used to build the string from `getHours()`, the viewer's own clock,
 * and the next-run line then appended a literal " SGT" to it: correct on a
 * laptop already set to Singapore, and quietly hours out on any other. */
const savedWhen = formatDayTime;

export function Track2PinnedBar() {
  const [files, setFiles] = useState<Track2SettingsFile[]>([]);
  const [current, setCurrent] = useState("");
  const [name, setName] = useState("");
  const [picked, setPicked] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(false);
  /** New asks first. It rewrites the LIVE settings, which are what the
   * schedule and the sweep actually read -- not just what is on screen --
   * so an accidental click would disarm a running schedule and drop the
   * picked source folders. Delete asks for the same reason. */
  const [confirmNew, setConfirmNew] = useState(false);
  /** Shown only after the server has actually refused for a clash (409),
   * so the tick is an answer to something that was read rather than a
   * standing invitation to skip a check nobody has seen. */
  const [clashed, setClashed] = useState(false);
  const [allowClash, setAllowClash] = useState(false);

  // The pull schedule: the clock that starts everything downstream.
  const [mode, setMode] = useState<"off" | "arrival" | "schedule">("off");
  const [everyDays, setEveryDays] = useState(1);
  const [timeOfDay, setTimeOfDay] = useState("");
  // Where "every N days" counts from: today, or a chosen date whose grid
  // the pulls then keep. Saved server-side with the rest of the schedule.
  const [countFrom, setCountFrom] = useState<"today" | "date">("today");
  const [startDate, setStartDate] = useState("");
  const [nextRun, setNextRun] = useState("");
  const [scheduleDirty, setScheduleDirty] = useState(false);
  /** Whether a finished SCHEDULED pull goes on to rebuild, freeze, queue
   * and start a run. Read because the row's label names those steps, and
   * a label naming work that is switched off is worse than a short one:
   * the chain is off by default, and a manual pull or a file arrival
   * never sets it off whatever this says. */
  const [chains, setChains] = useState(false);
  // Which pulls set the chain off -- a setting since the triggers were
  // made selectable. Defaults to the old hardcoded answer.
  const [chainOn, setChainOn] = useState<string[]>(["schedule"]);
  const [minFiles, setMinFiles] = useState(1);

  const [busy, setBusy] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  /** The file the settings are in, and the moment it was last written.
   *
   * The time is the FILE's own mtime off the listing, not a clock read
   * when this tab happened to press Save: it is then still right after a
   * reload, and right for a save somebody else made. A stamp that only
   * knows about saves this tab performed is one that quietly resets to
   * nothing every refresh. */
  const [lastSave, setLastSave] = useState<{ name: string; at: string } | null>(null);

  /** `force` re-reads the schedule fields even while they are marked
   * dirty.
   *
   * The dirty guard stops a poll overwriting what somebody is typing.
   * Straight after Apply it is wrong: the edit has been committed and
   * the server is now the truth. Worse, `refresh` closes over
   * `scheduleDirty` from the render it was made in, so the
   * `setScheduleDirty(false)` on the line above has not taken effect
   * here -- the guard read `true`, skipped the whole block, and the
   * "next run" went on showing the previous schedule's date. */
  async function refresh(force = false) {
    try {
      const [out, status, after, retrain] = await Promise.all([
        getTrack2SettingsFiles(),
        getTrack2PullControl().catch(() => null),
        getTrack2QueueAfterPull().catch(() => null),
        getTrack2Retrain().catch(() => null),
      ]);
      setChains(Boolean(after?.settings.enabled));
      setChainOn(after?.settings.triggers ?? ["schedule"]);
      setMinFiles(Number(after?.settings.min_files ?? 1) || 1);
      setFiles(out.files);
      const live = status?.config.settings_file || "";
      setCurrent(live);

      // Derived here rather than in the Save handler, so the stamp is
      // right on a cold page load too -- Save writes `settings_file`, so
      // saving under a NEW name moves `current` and this follows without
      // the handler having to say anything.
      const liveFile = out.files.find((f) => f.name === live);
      setLastSave(liveFile ? { name: liveFile.name, at: liveFile.modified } : null);

      // The selection must always be a file that EXISTS. A <select>
      // whose value matches no <option> renders the FIRST option while
      // holding the other value -- the box said one name and Load sent
      // another.
      const names = out.files.map((f) => f.name);
      setPicked((p) =>
        names.includes(p) ? p : names.includes(live) ? live : names[0] ?? "");

      if (status && (force || !scheduleDirty)) {
        const c = status.config;
        setMode((c.mode as typeof mode) || "off");
        setEveryDays(Number(c.schedule_every_days ?? 1) || 1);
        setCountFrom(c.schedule_from === "date" ? "date" : "today");
        setStartDate(String(c.schedule_start_date ?? ""));
        setTimeOfDay(String(c.schedule_time ?? ""));
        // The DATE it next fires, computed server-side from the cadence
        // and the week cut-off. Falls back to the raw expression only if
        // the server has no next time to give -- "rate(1 day)" says what
        // it does forever, not when it does it next, which is the thing
        // somebody who just pressed Apply is asking about.
        const rule = status.triggers?.schedule;
        // The TRAINING clock first, because that is the thing this strip
        // now arms and the question being asked is "when does it train".
        // It is always computable from the cadence, where the EventBridge
        // rule's own `next_run` is often empty -- which is how this line
        // came to read "rate(1 day)", a sentence about what it does
        // forever rather than when it does it next.
        const trainingNext = retrain?.settings.enabled
          ? retrain.next_runs?.[0] || ""
          : "";
        setNextRun(trainingNext
          || (rule?.enabled ? (rule.next_run || rule.schedule || "") : ""));
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
   *
   * One button rather than two, because they are the same act with a
   * different destination -- but the LABEL has to say which, or "Save"
   * after loading somebody else's file reads as "overwrite it" to one
   * person and "keep a copy" to the next.
   *
   * Load a file, change things, type a new name, Save: the settings go
   * to the new file and that file becomes the one you are in, which is
   * what Save As does everywhere else. */
  const typed = name.trim();
  const canSave = Boolean(typed || currentExists);

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
              Reset everything to defaults? The settings in force now are
              replaced.
            </span>
            <button
              type="button"
              disabled={!!busy}
              onClick={() => void run("New", async () => {
                const out = await newTrack2Settings();
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
             leave it empty to write back over the file you are in. The
             button says which of the two it is about to do. */}
        <span className="font-medium text-ink">Save as</span>
        <input
          className="tabular w-36 rounded border border-black/15 bg-page px-2 py-1"
          placeholder="new file name"
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
            // whose Apply has not been pressed is not in it. That is
            // worth SAYING and not worth refusing over: settings get
            // built up over several sittings, and Pull Control's Apply
            // is itself disabled until a source folder is picked -- so
            // blocking the save left no way to record a half-finished
            // config at all. Reported after the write instead.
            const unapplied = pendingCards();
            try {
              await saveTrack2SettingsFile(target, allowClash);
            } catch (err) {
              if (err instanceof ApiError && err.status === 409) {
                // Offer the override rather than just reporting it: the
                // refusal names a real risk, and the only way past it was
                // to change the week prefix or the root.
                setClashed(true);
              }
              throw err;
            }
            setClashed(false);
            setAllowClash(false);
            setName("");
            // Said after the write, not instead of it. The file is real
            // either way; what is worth knowing is which card's typing
            // did not reach it, so the next sitting knows what to redo.
            setNote(unapplied.length
              ? `Saved ${target}. ${unapplied.join(" and ")} has not had Apply `
                + "pressed, so the trigger is still set to what it was."
              : `Saved ${target}.`);
            // The stamp below comes from the refreshed listing, so it is
            // the file's real write time rather than this tab's clock.
            await refresh();
          })}
          className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
        >
          {busy === "Save"
            ? "Saving..."
            : typed ? `Save as ${typed}` : "Save"}
        </button>
        {/* A greyed button with nothing beside it reads as broken rather
            than as waiting. This is the one state it cannot act in: the
            file you were in has been deleted, so there is nothing to
            write back over and no name to write to. */}
        {!canSave && (
          <span className="text-ink-muted">
            type a name &mdash; no file is loaded to save over
          </span>
        )}

        {files.length > 0 && (
          <>
            <select
              className="tabular w-36 rounded border border-black/15 bg-page px-2 py-1"
              value={picked}
              onChange={(e) => { setPicked(e.target.value); setConfirmDelete(false); }}
            >
              {files.map((f) => (
                <option key={f.name} value={f.name}>{f.name}</option>
              ))}
            </select>
            <button
              type="button"
              disabled={!!busy || !picked}
              onClick={() => void run("Load", async () => {
                const out = await loadTrack2SettingsFile(picked, true);
                // Every card reads the settings once, when it opens, so
                // without a refresh they go live on the server while the
                // page goes on showing what was there before.
                // Say what the registry did. A detector appearing on
                // this machine because a file was loaded is a change
                // worth naming, and one that was LEFT ALONE matters
                // just as much: the file may describe it differently.
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

      {/* ---- STATUS LINE: two of them, answering different questions ----
          Off the control row because they are read, not operated, and a
          file name growing by six characters was shunting the buttons
          about. Left: what was written, and when. Right: what is in
          force. They are not the same fact -- save under a new name and
          both move together, but load somebody else's file and only the
          right one does. */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-5 gap-y-1 text-xs">
        {lastSave ? (
          <span className="tabular text-ink-secondary">
            Saved <span className="font-medium text-ink">{lastSave.name}</span>
            {lastSave.at && <> &middot; {savedWhen(lastSave.at)}</>}
          </span>
        ) : (
          <span className="text-ink-muted">Not saved yet</span>
        )}

        {/* ALWAYS present, so "which settings am I looking at" never has
            to be inferred from the absence of something. */}
        <span className="tabular text-ink-secondary">
          Loaded:{" "}
          {/* A file that no longer exists is not loaded, whatever the
              config still points at -- naming it read as though it were.
              Both cases are the same fact: nothing is loaded. */}
          {currentExists ? (
            <span className="font-medium text-ink">{current}</span>
          ) : (
            <span className="text-ink-muted">no file loaded</span>
          )}
        </span>
      </div>

      {/* ---- SECOND ROW: the pull, and whether it runs on its own ----
          Its own row because it is a different question from "which
          settings file am I in", and crowding both onto one line made
          the schedule fields wrap unpredictably as the file names
          changed length. */}
      <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-2 border-t border-hairline pt-1.5 text-xs">
        {/* The row's NAME: the whole pull-to-train pipeline this row
            configures. Constant, because a heading that changes as you
            click the buttons under it reads as a status, and this is not
            one -- the buttons choose WHEN it fires and the line below
            says what will actually happen. Only a scheduled pull runs
            the chain; arrival and a manual Pull now deliberately do not,
            because they fire unpredictably often and each run holds the
            single GPU. */}
        <span className="font-medium text-ink">PULL, BUILD, QUEUE, TRAIN</span>

        {/* Run it automatically: off / on arrival / on a schedule. The
            same three states the "Or run it automatically" card offers,
            as buttons rather than a dropdown so the current one is
            readable without opening anything. That card stays. */}
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
              // force: the edit is committed, so the server's values are
              // now the right ones to show -- including the next run date.
              await refresh(true);
            })}
            className="rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
          >
            {busy === "Apply" ? "Applying..." : "Apply"}
          </button>
        ) : nextRun ? (
          <span className="tabular text-ink-muted">
            next run: {nextRun.includes("T")
              ? savedWhen(nextRun)
              : nextRun}
          </span>
        ) : null}
        {/* What the chosen mode ACTUALLY does. The heading above names
            the whole pipeline, so this is the only thing standing between
            that name and somebody believing a file arriving will train a
            model. Only one of the four states runs the chain. */}
        {mode === "off" && (
          <span className="text-ink-muted">nothing runs on its own</span>
        )}
        {/* Whether THIS mode's pulls chain is a setting now: the
            triggers on "Run after every pull" (step 4b). It used to be
            hardcoded here as "arrivals never train", which stopped being
            true when arrival became selectable. */}
        {(mode === "arrival" || mode === "schedule") && !(chains && chainOn.includes(mode)) && (
          <span className="text-ink-muted">
            pulls only &mdash; tick <em>{mode === "arrival" ? "Arrival" : "Schedule"}</em>{" "}
            under <em>Run after every pull</em> in step 4b to build, queue and train
          </span>
        )}
        {mode === "schedule" && chains && chainOn.includes("schedule") && (
          <span className="text-ink-secondary">
            builds, queues and trains after each scheduled pull
          </span>
        )}
        {mode === "arrival" && chains && chainOn.includes("arrival") && (
          <span className="text-ink-secondary">
            builds, queues and trains when an arrival pull moves at least{" "}
            {minFiles} file{minFiles === 1 ? "" : "s"}
          </span>
        )}

      </div>

      {/* ---- THIRD ROW: what the chain is doing right now ----
          Directly under the row that names the four steps, because it is
          the answer to the question that row raises. It was a strip
          across the top of every page; that put the GPU and the stages a
          long way from the control that starts them. */}
      <Track2ChainStatusBar />
      {/* Directly under the GPU line, because they answer one glance:
          what is the GPU doing, and what has it cost today. It renders
          nothing until it has an answer, so it costs no space on a cold
          load. */}
      <Track2TrainingCostToday />
    </div>
  );
}
