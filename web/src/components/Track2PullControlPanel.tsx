import { useCallback, useEffect, useRef, useState } from "react";
import { useCardOpen } from "../lib/cardState";
import {
  ApiError,
  checkTrack2Pairing,
  fillTrack2WeekGaps,
  getTrack2Buckets,
  getTrack2Forecast,
  getTrack2PullSchedule,
  getTrack2PullControl,
  previewTrack2WeekPurge,
  purgeTrack2Weeks,
  type Track2WeekPurgePreview,
  getTrack2SettingsFiles,
  type Track2SettingsClaim,
  listTrack2Folders,
  previewTrack2Pull,
  purgeTrack2Week,
  getTrack2CurrentRun,
  getTrack2Pending,
  getTrack2PullRun,
  startTrack2Pull,
  stopTrack2Pull,
  saveTrack2PullConfig,
  type Track2Bucket,
  type Track2ForecastWeek,
  type Track2SchedulePlan,
  type Track2ScheduleState,
  type Track2Pairing,
  type Track2PullConfig,
  type Track2PendingSummary,
  type Track2PullPlan,
  type Track2PullRunState,
  type Track2PullStatus,
  type Track2SettingsFile,
} from "../lib/api";
import { markPending } from "../lib/pendingChanges";
import { Card } from "./Card";
import { S3FolderField } from "./S3FolderField";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

type PickList = "train_images" | "train_labels" | "val_images" | "val_labels";
type TriggerMode = "off" | "arrival" | "schedule";

/** How far back the weeks table reaches, in days.
 *
 * Three years. It was eight weeks, which kept the table short but hid
 * the thing the table is for: a retraining programme is judged over
 * seasons, and "what did we collect last spring" is a question you ask
 * of it. The list scrolls; a row costs nothing until it is scrolled to.
 *
 * Still a window rather than everything: anything older stays on S3 and
 * stays counted, and `Show all` reaches it.
 */
const WEEK_HISTORY_DAYS = 3 * 365;

/** The four shapes a pull cadence is offered in. */
type PullKind = "day" | "days" | "week" | "weeks";

/** "HH:MM", 24-hour, or "" if it is not a time.
 *
 * The same rule the API applies, so the card can say so before Apply
 * rather than showing a server error afterwards. */
function timeOfDay(text: string): string {
  const m = text.trim().match(/^(\d{1,2}):(\d{2})$/);
  if (!m) return "";
  const hour = Number(m[1]);
  const minute = Number(m[2]);
  if (hour > 23 || minute > 59) return "";
  return `${String(hour).padStart(2, "0")}:${String(minute).padStart(2, "0")}`;
}

/** The smallest count a "every N ..." row may hold.
 *
 * One is not allowed, and that is not fussiness. "Every 1 weeks" is
 * seven days, which is indistinguishable from "Every week" once stored
 * -- so it came back selected on the OTHER row, and the choice looked
 * like it had not stuck. Same for "every 1 days" against "Every day".
 */
const MIN_COUNT = 2;

/** A cadence as a number of days -- the only thing that is stored. */
function everyDaysOf(kind: PullKind, n: number): number {
  const count = Math.max(Math.floor(n) || MIN_COUNT, MIN_COUNT);
  if (kind === "day") return 1;
  if (kind === "week") return 7;
  if (kind === "weeks") return count * 7;
  return count;
}

/** The inverse, for showing a stored cadence back as the choice that
 * says it most plainly: 14 days is "every 2 weeks", not "every 14 days".
 *
 * `n` is null for the two rows that carry no number. Returning 1 for
 * them used to overwrite whatever was typed in the counted rows, so
 * picking "Every day" and then "Every N weeks" silently offered 1 --
 * which is 7 days, which reads back as "Every week".
 */
function kindOfDays(days: number): { kind: PullKind; n: number | null } {
  const d = Math.max(Math.floor(days) || 1, 1);
  if (d === 1) return { kind: "day", n: null };
  if (d === 7) return { kind: "week", n: null };
  if (d % 7 === 0) return { kind: "weeks", n: d / 7 };
  return { kind: "days", n: d };
}

const LISTS: { key: PickList; label: string; hint: string; side: "train" | "val" }[] = [
  { key: "train_images", label: "Training images", hint: "jpg, png ...", side: "train" },
  { key: "train_labels", label: "Training labels", hint: "txt", side: "train" },
  { key: "val_images", label: "Validation images", hint: "jpg, png ...", side: "val" },
  { key: "val_labels", label: "Validation labels", hint: "txt", side: "val" },
];

/** Buckets the stack grants week-folder writes to. Mirrors
 * `dataset_write_buckets` in terraform/app; adding one there means adding
 * it here, and a bucket in neither simply cannot be chosen. */
export const WRITABLE = ["track2-s3", "safety-model-dataset-dev", "safety-model-dataset"];

const EMPTY_PICKS: Record<PickList, string[]> = {
  train_images: [], train_labels: [], val_images: [], val_labels: [],
};

/**
 * Pull control: pick the folders, and everything else follows.
 *
 * One card, deliberately. The folders, where they land, how a week is
 * decided, what fires a sweep and what happened last time are one
 * decision made at one moment. Split across cards they drift, and
 * whichever card wrote the configuration last silently wins.
 *
 * No `data.yaml` is read anywhere. Images and labels are told apart by
 * file EXTENSION, and which split a folder feeds is the list it was put
 * in -- so raw folders with no manifest work as well as a prepared
 * dataset, and a folder with `val` in its path still goes to training if
 * that is where it was put.
 */

/**
 * Two clocks on this card, and which is which is not arbitrary.
 *
 * **Dates, no clock** for the cut-off and everything derived from it --
 * the week boundaries, what each week covers, when it closes. They are
 * compared against the capture time in each filename
 * (`...__20260318T075125Z__...`), which is UTC by construction, but a
 * DATE needs no timezone label to be read correctly and carrying one
 * only invited the question of which clock the reader was in.
 *
 * A window's `ends_at` is the moment the next one begins, so the last
 * day it includes is one millisecond earlier -- see `lastDay`. Printing
 * the raw boundary showed a cut-off of 24 Mar as "closes 25 Mar", a date
 * nobody typed.
 *
 * **SGT, labelled** for the pull schedule, the run log and "last
 * looked". Those are times of day, so they do need a clock, and the EC2
 * schedule this system runs on is Asia/Singapore.
 *
 * A fixed offset rather than a timezone database: Singapore has been
 * UTC+8 with no daylight saving since 1982, so this is exact, not an
 * approximation that drifts twice a year.
 */
const SGT_OFFSET_MS = 8 * 60 * 60 * 1000;

/** A UTC ISO timestamp as `YYYY-MM-DD HH:MM` Singapore time. */
function sgt(iso: string): string {
  if (!iso) return "";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso;
  return new Date(at.getTime() + SGT_OFFSET_MS).toISOString().replace("T", " ").slice(0, 16);
}

const DAY_MS = 24 * 60 * 60 * 1000;

/** The last date a window includes.
 *
 * A week's `ends_at` is the moment the NEXT one begins, so it names a
 * day the week does not contain. Printing it raw made a cut-off of
 * 24 Mar show as "closes 2026-03-25" -- a date the person never typed,
 * one day after the one they did. One millisecond back lands on the day
 * the window actually ends.
 */
function lastDay(iso: string): string {
  if (!iso) return "";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "";
  return new Date(at.getTime() - 1).toISOString().slice(0, 10);
}

/** The date a moment falls on. */
function dayOf(iso: string): string {
  if (!iso) return "";
  const at = new Date(iso);
  return Number.isNaN(at.getTime()) ? "" : at.toISOString().slice(0, 10);
}

/** A picked date (`YYYY-MM-DD`) to the moment week 1 closes.
 *
 * The date is INCLUSIVE: picking 31 March means everything captured on
 * or before 31 March is the backlog, so the boundary is the midnight
 * that ends it -- 1 April 00:00. "Cut off after that date" is what a
 * date-only cut-off means to the person choosing it, and the alternative
 * reading silently excludes a whole day of captures.
 *
 * UTC, not Singapore time, and deliberately the one exception on a card
 * that is otherwise SGT: the cut-off is compared against the capture
 * timestamp carried in each filename -- `...__20260318T075125Z__...` --
 * which is UTC by construction. The pull SCHEDULE stays SGT, because
 * that says when a job should run, which is a local decision.
 */
function anchorToIso(text: string): string {
  const m = text.trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return "";
  const [, yyyy, mm, dd] = m;
  const at = new Date(Date.UTC(Number(yyyy), Number(mm) - 1, Number(dd)) + DAY_MS);
  return Number.isNaN(at.getTime()) ? "" : at.toISOString();
}

/** The inverse: the last date the backlog includes.
 *
 * One millisecond back before reading the date, so a boundary of
 * 1 April 00:00 reads as 31 March -- the day it closes, not the day
 * after it.
 */
function isoToAnchor(iso: string): string {
  if (!iso) return "";
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return "";
  return new Date(at.getTime() - 1).toISOString().slice(0, 10);
}

/** A picked date as `31 Mar 2026`, for prose rather than a table. */
function readableDate(text: string): string {
  const m = text.trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return "";
  const at = new Date(Date.UTC(Number(m[1]), Number(m[2]) - 1, Number(m[3])));
  return at.toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric", timeZone: "UTC",
  });
}

export function Track2PullControlPanel({
  only,
  onOnlyChange,
}: {
  /** Whether the run takes its data from the Assemble card's folders
   * instead, which is what switches this card off. Stored as
   * `dataset_assemble.only` and shown here INVERTED, because the
   * question this card asks is "use Pull Control".
   *
   * Optional, so the standard tab -- which has no Assemble card to
   * switch between -- renders the card with no tick at all rather
   * than one that turns nothing off. */
  only?: boolean;
  onOnlyChange?: (on: boolean) => void;
} = {}) {
  const [status, setStatus] = useState<Track2PullStatus | null>(null);
  const [open, setOpen] = useCardOpen("pull_control", false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  // The two-step delete: null until the count has been fetched and
  // shown, so the destructive button cannot exist before the number it
  // is confirming does.
  const [purge, setPurge] = useState<Track2WeekPurgePreview | null>(null);
  const [purgeTyped, setPurgeTyped] = useState("");

  // --- what is configured -------------------------------------------------
  const [bucket, setBucket] = useState("safety-model-dataset-dev");
  const [buckets, setBuckets] = useState<Track2Bucket[]>([]);
  const [picks, setPicks] = useState<Record<PickList, string[]>>({ ...EMPTY_PICKS });
  // One root holds everything. The train/val split lives inside each
  // week, under images/ and labels/ -- see the tree below.
  const [root, setRoot] = useState("retraining");
  /** Every saved settings file's root, and the Model Name it trains as.
   * Read from the FILES, not from live config -- the live one is
   * whatever was loaded last, and the question is what the others use. */
  const [claims, setClaims] = useState<Track2SettingsClaim[]>([]);
  /** Dismissed for this root. The clash is a warning worth reading, not
   * a rule: the same week folders CAN feed two Classes. Keyed on the
   * folder so changing to a different clashing one warns again. */
  const [rootClashOk, setRootClashOk] = useState("");
  /** The settings file these values came from, excluded from the clash
   * check -- a configuration cannot clash with itself. */
  const [loadedFile, setLoadedFile] = useState("");
  const [imagesSub, setImagesSub] = useState("images");
  const [labelsSub, setLabelsSub] = useState("labels");
  const [weekPrefix, setWeekPrefix] = useState("week_");
  const [weekDays, setWeekDays] = useState(7);
  // Which clock decides a file's week. capture_time reads the moment
  // written into the filename; file_time uses when it was uploaded.
  const [weekBy, setWeekBy] = useState<"capture_time" | "file_time" | "scan">("file_time");
  const [autoWeeks, setAutoWeeks] = useState(true);
  const [backlog, setBacklog] = useState(true);
  // Value unread since the trigger moved to the pinned bar -- the loader
  // still fills it so a future reader gets the live expression, but this
  // card no longer sends it anywhere.
  const [, setSchedule] = useState("rate(1 day)");
  // How often, as one of four shapes rather than a number of days. Only
  // `pullEveryDays` is stored -- the shape is re-derived on load, so
  // "every 2 weeks" comes back as 2 weeks and not as 14 days.
  const [pullKind, setPullKind] = useState<PullKind>("week");
  const [pullN, setPullN] = useState(2);
  // What time of day a pull lands on. "default" follows the cut-off,
  // which keeps the card down to one clock; "at" moves the clock only,
  // never the dates. Applies to all four cadences above it.
  const [pullTimeMode, setPullTimeMode] = useState<"default" | "at">("default");
  const [pullTimeText, setPullTimeText] = useState("02:00");
  const [destBucket, setDestBucket] = useState("");
  // Held as the text that was typed, so a half-finished date stays on
  // screen instead of being silently discarded on every keystroke.
  const [anchorText, setAnchorText] = useState("");
  // What the card wants the trigger to be. Separate from what is live,
  // because nothing on this card takes effect until Apply settings.
  const [wantedMode, setWantedMode] = useState<TriggerMode>("off");
  // Set by any edit, cleared by Apply -- so "what you are looking at is
  // not what is running" is visible rather than assumed.
  const [dirty, setDirty] = useState(false);
  /** True once the card has actually read the live config. Until then it
   * holds initial values, and saving those would overwrite the settings
   * it has not seen yet. */
  const [loadedConfig, setLoadedConfig] = useState(false);
  // Published so the toolbar's Save can refuse to write a file that would
  // not contain what is on this card. Nothing renders this; it is read at
  // the moment Save is clicked.
  useEffect(() => {
    markPending("Pull Control", dirty);
    return () => markPending("Pull Control", false);
  }, [dirty]);

  // --- browsing for folders ----------------------------------------------
  const [addingTo, setAddingTo] = useState<PickList | null>(null);
  const [browsePrefix, setBrowsePrefix] = useState("");
  const [browseFolders, setBrowseFolders] = useState<string[]>([]);

  // --- everything else the card shows ------------------------------------
  const [pairing, setPairing] = useState<Record<"train" | "val", Track2Pairing | null>>({
    train: null, val: null,
  });
  const [forecast, setForecast] = useState<Track2ForecastWeek[] | null>(null);
  const [currentWeek, setCurrentWeek] = useState("");
  // Where the next pull files, which is not the week the clock is in
  // while the backlog is still waiting -- see active_week on the API.
  const [activeWeek, setActiveWeek] = useState("");
  // The counts change only when a pull lands something. Held in refs
  // rather than state because nothing renders them -- they exist to
  // answer "did anything change since the last look", and putting them
  // in state would re-render the table on every poll that found nothing.
  const stampRef = useRef("");
  // Which request is the newest. Responses can arrive out of order --
  // an edit fires while a 20-second poll is still in flight -- and the
  // slower one would otherwise win and redraw the table with settings
  // nobody is looking at any more.
  const requestRef = useRef(0);
  const countsRef = useRef<Record<string, number>>({});
  const [grew, setGrew] = useState<Record<string, number>>({});
  const [watching, setWatching] = useState(true);
  const [checkedAt, setCheckedAt] = useState("");
  // Which configuration the weeks table is showing. "" is what is typed
  // in this card; a name is a saved file being LOOKED AT -- nothing is
  // applied by choosing one, so browsing another dataset cannot re-point
  // the next pull.
  const [viewing, setViewing] = useState("");
  const [showAllWeeks, setShowAllWeeks] = useState(false);
  // True while the table is dated by settings that have not been applied.
  const [previewingWeeks, setPreviewingWeeks] = useState(false);
  const [existingWeeks, setExistingWeeks] = useState(0);
  // Value unread since the schedule controls moved to the pinned bar;
  // the setter stays because the status fetch still calls it.
  const [, setSchedPlan] = useState<Track2SchedulePlan | null>(null);
  const [, setLiveSchedule] = useState<Track2ScheduleState | null>(null);
  const [confirmPurge, setConfirmPurge] = useState("");
  // How many objects THAT week holds, counted when the confirmation
  // opens. `null` means not counted yet, which disables the button --
  // the server now refuses unless the number sent back matches what
  // it re-counts, so a guess would only ever be a 409.
  const [purgeCount, setPurgeCount] = useState<number | null>(null);
  const [settingsFiles, setSettingsFiles] = useState<Track2SettingsFile[]>([]);
  const [plan, setPlan] = useState<Track2PullPlan | null>(null);
  // The sweep that is running, polled while it runs. Held rather than
  // awaited: a pull over a few thousand files is minutes, and a request
  // that blocks for minutes can neither show progress nor be stopped.
  const [run, setRun] = useState<Track2PullRunState | null>(null);
  const [stopping, setStopping] = useState(false);
  const [pending, setPending] = useState<Track2PendingSummary | null>(null);

  // ------------------------------------------------------------------ load
  /** Fills the controls from a config. Used both for what is live and for
   * a file being looked at -- neither changes anything by itself. */
  const fillFrom = useCallback((c: Partial<Track2PullConfig>) => {
    setPicks({ ...EMPTY_PICKS, ...(c.picked ?? {}) });
    if (c.source_bucket) setBucket(c.source_bucket);
    // `root` is the setting. A file saved before the two roots were
    // merged has only `training_root`, so it loads as the root rather
    // than as a blank field nobody would know to fill in.
    if (c.root) setRoot(c.root);
    else if (c.training_root) setRoot(c.training_root);
    // WHICH saved file is loaded, so the clash warning below does not
    // accuse this configuration of clashing with itself.
    setLoadedFile(String(c.settings_file || ""));
    if (c.images_subpath !== undefined) setImagesSub(c.images_subpath);
    if (c.labels_subpath !== undefined) setLabelsSub(c.labels_subpath);
    // The real value, always. The field is read-only now, and one that
    // hid what is actually configured would simply be lying -- the
    // whole point of showing it is that `week_3` on screen and
    // `week_3` in the bucket are the same string.
    if (c.week_prefix !== undefined) setWeekPrefix(c.week_prefix || "week_");
    if (c.week_length_days !== undefined) setWeekDays(c.week_length_days);
    if (c.week_by) setWeekBy(c.week_by);
    if (c.auto_create_weeks !== undefined) setAutoWeeks(c.auto_create_weeks);
    if (c.first_week_takes_all !== undefined) setBacklog(c.first_week_takes_all);
    if (c.schedule_expression !== undefined) setSchedule(c.schedule_expression);
    if (c.schedule_every_days !== undefined) {
      const { kind, n } = kindOfDays(c.schedule_every_days);
      setPullKind(kind);
      // Only when the row actually carries a number. "Every day" and
      // "Every week" carry none, and writing 1 here is what made the
      // next counted choice collapse onto a different row.
      if (n !== null) setPullN(Math.max(n, MIN_COUNT));
    }
    if (c.schedule_time !== undefined) {
      const saved = timeOfDay(c.schedule_time);
      setPullTimeMode(saved ? "at" : "default");
      // A blank leaves the box on its last value rather than emptying
      // it, so switching back to "At" does not start from nothing.
      if (saved) setPullTimeText(saved);
    }
    if (c.destination_bucket !== undefined) setDestBucket(c.destination_bucket);
    if (c.week_cutoff !== undefined) setAnchorText(isoToAnchor(c.week_cutoff));
    if (c.mode !== undefined) setWantedMode(c.mode as TriggerMode);
  }, []);

  const load = useCallback(async () => {
    try {
      const got = await getTrack2PullControl();
      setStatus(got);
      fillFrom(got.config);
      setDirty(false);
      setLoadedConfig(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not read pull control.");
    }
  }, [fillFrom]);

  const loadWeeks = useCallback(async (opts: {
    weekLengthDays?: number;
    weekCutoff?: string;
    weekPrefix?: string;
  } = {}) => {
    const ticket = ++requestRef.current;
    try {
      const got = await getTrack2Forecast(opts);
      // Someone asked a newer question while this was in flight.
      if (ticket !== requestRef.current) return;
      setPreviewingWeeks(got.previewing);
      setExistingWeeks(got.existing_weeks);
      setCheckedAt(new Date().toISOString());

      // Only the weeks whose counts actually moved are marked, so a pull
      // that filled week_6 does not light up week_2 as well. The first
      // look sets the baseline and marks nothing -- everything is "new"
      // the first time, which would be noise rather than news.
      const had = countsRef.current;
      const now: Record<string, number> = {};
      got.weeks.forEach((w) => { now[w.week] = w.images; });
      if (stampRef.current && stampRef.current !== got.stamp) {
        const added: Record<string, number> = {};
        got.weeks.forEach((w) => {
          const before = had[w.week];
          if (before !== undefined && w.images > before) {
            added[w.week] = w.images - before;
          }
        });
        if (Object.keys(added).length) setGrew(added);
      }
      stampRef.current = got.stamp;
      countsRef.current = now;

      setForecast(got.weeks);
      setCurrentWeek(got.current);
      setActiveWeek(got.active || got.current);
      setSchedPlan(got.schedule ?? null);
    } catch {
      if (ticket === requestRef.current) setForecast([]);
    }
  }, []);

  /** Reload the weeks keeping whatever is currently typed in view. */
  /** What the weeks table should be showing right now.
   *
   * A saved file wins outright: sending the typed values alongside it
   * would half-override the file being viewed and show a configuration
   * that exists nowhere. */
  const weekView = useCallback(() => (viewing
    ? { settings: viewing }
    : {
        weekLengthDays: weekDays,
        weekCutoff: anchorToIso(anchorText),
        weekPrefix: weekPrefix,
      }), [viewing, weekDays, anchorText, weekPrefix]);

  const refreshWeeks = useCallback(() => loadWeeks(weekView()),
                                   [loadWeeks, weekView]);

  /** What is still waiting to be pulled, and where it would land. */
  const loadPending = useCallback(async () => {
    try {
      setPending(await getTrack2Pending());
    } catch {
      setPending(null);
    }
  }, []);

  const loadSettings = useCallback(async () => {
    try {
      const got = await getTrack2SettingsFiles();
      // Only the names, for the weeks preview's "viewing" list. Saving,
      // loading and deleting moved to the Retention card, which writes
      // one file for every card rather than one for this.
      setSettingsFiles(got.files);
      // What every OTHER saved file already trains out of, so the root
      // picker can say so before a second one is pointed at it.
      setClaims(got.claims ?? []);
    } catch {
      setSettingsFiles([]);
      setClaims([]);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void load();
    void loadPending();
    // The weeks are NOT fetched here. The effect below already runs on
    // open and passes the settings as they are currently typed; fetching
    // here as well raced it, and the bare call -- which sends no
    // overrides and so gets the SAVED prefix back -- sometimes landed
    // last. That is why the table said week_1 while every other line in
    // the card said meow_1.
    void loadSettings();
    getTrack2Buckets().then((r) => setBuckets(r.buckets)).catch(() => undefined);
  }, [open, load, loadSettings, loadPending]);

  // Re-date the weeks as the length or the cut-off is edited, so the
  // table answers "what would this do" while it is being chosen rather
  // than only after Apply. Debounced: typing a date is several
  // keystrokes and only the last one is worth a listing.
  useEffect(() => {
    if (!open) return;
    const timer = window.setTimeout(() => { void loadWeeks(weekView()); }, 400);
    return () => window.clearTimeout(timer);
  }, [open, loadWeeks, weekView]);

  // Notice a sweep this page did not start: one already going when the
  // card was opened, and -- the case this exists for -- one the SCHEDULE
  // begins while somebody is watching.
  //
  // Checked on a timer, not only when the card opens. It used to run on
  // `open` alone, so a scheduled pull that started with the card already
  // in front of you never appeared: it ran to completion, moved every
  // file, and the card showed nothing the whole time. The one sweep
  // nobody is in a position to watch by hand is exactly the one that
  // needs announcing.
  //
  // Only while nothing is being tracked. The poll below takes over the
  // moment a run is in hand, and two pollers on one run would fight.
  useEffect(() => {
    if (!open || (run && run.status === "running")) return;
    const look = () => {
      getTrack2CurrentRun()
        .then((r) => {
          if (r.run && r.run.status === "running") setRun(r.run);
        })
        .catch(() => undefined);
    };
    look();
    const timer = window.setInterval(look, 10000);
    return () => window.clearInterval(timer);
  }, [open, run]);

  // Poll while it runs. Every 1.5s: the sweep publishes every 25 files,
  // so anything slower makes the bar jump and Stop feel unresponsive.
  useEffect(() => {
    if (!run || run.status !== "running") return;
    const timer = window.setInterval(() => {
      getTrack2PullRun(run.run_id)
        .then((got) => {
          setRun(got);
          if (got.status !== "running") {
            setStopping(false);
            void refreshWeeks();
            void loadPending();
            // Say what it did. A scheduled sweep has nobody watching the
            // button that started it, so the result is the only report
            // of it there is.
            setNote(
              `${got.trigger === "schedule" ? "Scheduled pull" : "Pull"} `
              + `${got.status === "stopped" ? "stopped" : "finished"}: `
              + `${got.moved.toLocaleString()} copied`
              + `${got.skipped ? `, ${got.skipped.toLocaleString()} skipped` : ""}`
              + `${got.errors ? `, ${got.errors} failed` : ""}.`
            );
          }
        })
        .catch(() => undefined);
    }, 1500);
    return () => window.clearInterval(timer);
  }, [run, refreshWeeks, loadPending]);

  // Watch for files landing. Two S3 listings per check, so twenty
  // seconds rather than two: a pull is minutes of work, and a count that
  // is twenty seconds stale has never been the thing that misled anyone
  // here. Stops when the card is shut, so a tab left open overnight is
  // not listing a bucket all night.
  useEffect(() => {
    if (!open || !watching) return;
    const timer = window.setInterval(() => {
      // The same view as the editing effect, or a poll would snap the
      // table back every twenty seconds while someone is still choosing
      // a week length or looking at another dataset.
      void loadWeeks(weekView());
      // What is waiting rides the same cycle. It is the more expensive of
      // the two -- every source folder plus the destination root -- so it
      // is skipped entirely while a sweep is running, when the number is
      // changing under it anyway and the sweep needs the bandwidth.
      if (!run || run.status !== "running") void loadPending();
    }, 20000);
    return () => window.clearInterval(timer);
  }, [open, watching, loadWeeks, weekView, loadPending, run]);

  // Clear the "just arrived" marks a little after they appear -- they
  // answer "did that pull do anything", which is a question with a
  // short life.
  useEffect(() => {
    if (!Object.keys(grew).length) return;
    const timer = window.setTimeout(() => setGrew({}), 12000);
    return () => window.clearTimeout(timer);
  }, [grew]);

  useEffect(() => {
    if (!open) return;
    getTrack2PullSchedule()
      .then((r) => { setLiveSchedule(r.schedule); setSchedPlan(r.plan); })
      .catch(() => undefined);
  }, [open]);

  useEffect(() => {
    if (!addingTo) return;
    let cancelled = false;
    listTrack2Folders(bucket, browsePrefix)
      .then((r) => !cancelled && setBrowseFolders(r.folders))
      .catch(() => !cancelled && setBrowseFolders([]));
    return () => { cancelled = true; };
  }, [addingTo, bucket, browsePrefix]);

  // Pairing, re-checked as either half of a pair changes. Debounced:
  // adding four folders in a row is four changes and only the last one
  // is worth a listing.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      (["train", "val"] as const).forEach((side) => {
        const images = picks[side === "train" ? "train_images" : "val_images"];
        const labels = picks[side === "train" ? "train_labels" : "val_labels"];
        if (!images.length && !labels.length) {
          setPairing((all) => ({ ...all, [side]: null }));
          return;
        }
        checkTrack2Pairing(bucket, images, labels)
          .then((r) => !cancelled && setPairing((all) => ({ ...all, [side]: r })))
          .catch(() => !cancelled && setPairing((all) => ({ ...all, [side]: null })));
      });
    }, 400);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [open, bucket, picks]);

  async function withBusy(label: string, work: () => Promise<void>) {
    setBusy(label);
    setError("");
    setNote("");
    try {
      await work();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "That did not work.");
    } finally {
      setBusy("");
    }
  }

  const liveMode = status?.config.mode ?? "off";

  // One number in days, whichever unit was used to type it.
  // The tree, drawn from what is actually picked. A root with no folders
  // behind it is left out entirely rather than shown empty, because
  // nothing will create it -- and drawing a folder that will not exist is
  // how the old version came to promise a shape it never built.
  const triggers = status?.triggers;
  // The prefix as SAVED, which is what the folders on S3 are named
  // after -- not the one being typed.
  const savedPrefix = status?.config?.week_prefix ?? "";
  const anyPicked = LISTS.some(({ key }) => picks[key].length);
  /** Sides with image folders and no label folder at all.
   *
   * The same rule the sweep refuses on, mirrored here so it is answered
   * while you are picking rather than at 18:33 by a schedule. Images with
   * no labels are taken as BACKGROUND examples: the trainer accepts them
   * and learns nothing from them, so nothing errors -- a week of them
   * just trains on nothing, and the first sign is a metric that will not
   * move. A FEW unmatched images are healthy and not counted here; this
   * is only the unambiguous case. */
  const unpairedSides = (["train", "val"] as const).filter((side) => {
    const images = picks[`${side}_images` as keyof typeof picks] ?? [];
    const labels = picks[`${side}_labels` as keyof typeof picks] ?? [];
    return images.some((p) => p.trim()) && !labels.some((p) => p.trim());
  });
  const unpaired = unpairedSides.length > 0;
  const firstWeek = `${weekPrefix}1`;

  // NOTE: everything this reads must be declared ABOVE it. It is invoked
  // immediately, so a `const` declared further down is still in its
  // temporal dead zone and throws -- which blanks the page, because a
  // throw during render unmounts the whole tree. TypeScript does not
  // catch it: the reference sits inside an arrow function, which it
  // treats as a closure that might be called later.
  const builtTree = (() => {
    const head = `s3://${destBucket || "<this app's bucket>"}/`;
    const splits: [string, number, number][] = [
      ["train", picks.train_images.length, picks.train_labels.length],
      ["val", picks.val_images.length, picks.val_labels.length],
    ];
    const live = splits.filter(([, images, labels]) => images || labels);
    if (!live.length) return `${head}\n  (pick some folders above)`;

    const lines = [head, `${root || "<root>"}/`, `  ${firstWeek}/`];
    // images/ then labels/, each with the splits under it -- the shape
    // Ultralytics reads, and the shape the pull actually writes.
    ([[imagesSub, 0], [labelsSub, 1]] as const).forEach(([folder, which]) => {
      const rows = live.filter((entry) => entry[1 + which]);
      if (!rows.length) return;
      lines.push(`    ${folder}/`);
      rows.forEach(([split, ...counts]) => {
        lines.push(`      ${split}/    <- ${counts[which]} folder(s)`);
      });
    });
    return lines.join("\n");
  })();

  const pullEveryDays = everyDaysOf(pullKind, pullN);
  // "" means "use the cut-off's time", which is exactly what the API
  // takes, so the two states need no translation between them.
  const pullTime = pullTimeMode === "at" ? timeOfDay(pullTimeText) : "";
  const pullTimeBroken = pullTimeMode === "at" && !pullTime;
  // The cut-off's own time, for labelling the default choice.

  // The rolling window. Past and current weeks are trimmed to the most
  // recent WEEK_WINDOW; the weeks still to come are always shown, since
  // they are the forecast and there are only ever a handful.
  //
  // Trimmed, not dropped: the folders and their files are untouched, and
  // the count of what is out of view is stated so nobody concludes week
  // 1 was deleted because it stopped being drawn.
  const weekRows = forecast ?? [];
  const settled = weekRows.filter((w) => w.state !== "future");   // incl. empty
  const upcoming = weekRows.filter((w) => w.state === "future");
  // Three years measured in weeks of whatever length is configured --
  // 157 at seven days, 79 at fourteen. Counting weeks rather than rows
  // keeps "three years" true when the week length changes.
  const weekWindow = Math.max(Math.ceil(WEEK_HISTORY_DAYS / Math.max(weekDays, 1)), 1);
  const weeksHidden = Math.max(settled.length - weekWindow, 0);
  const visibleWeeks = showAllWeeks
    ? weekRows
    : [...settled.slice(-weekWindow), ...upcoming];
  // A pull that reaches back further than a week fills more than one
  // week folder. That is fine -- but only because files are dated one
  // by one. Worth saying, because it is the thing people expect to go
  // wrong here.

  /** Everything the card holds, as the API takes it. */
  function currentConfig() {
    return {
      picked: picks,
      source_bucket: bucket,
      root: root,
      images_subpath: imagesSub,
      labels_subpath: labelsSub,
      week_prefix: weekPrefix,
      week_length_days: weekDays,
      week_by: weekBy,
      auto_create_weeks: autoWeeks,
      first_week_takes_all: backlog,
      destination_bucket: destBucket,
      week_cutoff: anchorToIso(anchorText),
      // The TRIGGER is deliberately absent: mode, cadence and time belong
      // to the toolbar, and this card only ever read them. Writing them
      // back from values read at page load silently undid the toolbar --
      // set a schedule there, and this card's next save put `mode` back
      // to the "off" it had been showing since the page opened. The
      // cadence survived, the mode did not, and the card then reported
      // "nothing fires on its own" quite correctly.
    };
  }


  /** This card's choices are SAVED as they are made, like every other
   * card's -- picking a folder here used to reach the server only when
   * Apply settings was pressed, which is at the bottom of a long card
   * while the Save button is pinned at the top. The folders looked
   * saved, the file did not have them, and `picked` stayed empty.
   *
   * This DOES take effect: `POST /config` rebuilds `sources` from the
   * picks itself, so a folder chosen here is one the next sweep collects
   * from. There is no staging step, and nothing in the UI should imply
   * one. The Apply settings button that used to sit at the bottom is
   * gone: it re-sent this same config, and the trigger is the toolbar's.
   *
   * `dirty` is the guard. It is false immediately after a load, so the
   * first render cannot write the values it has just read back over the
   * ones it read them from.
   */
  const configJson = JSON.stringify(currentConfig());
  const latestConfigRef = useRef(configJson);
  latestConfigRef.current = configJson;
  useEffect(() => {
    // BOTH guards. `dirty` alone was not enough: it says somebody typed,
    // not that the card has read what it is about to overwrite. Edit any
    // field before the first load lands and this wrote the card's blank
    // `week_cutoff` over a real date -- which is the shape of every
    // "my settings cleared themselves" report on this page.
    if (!loadedConfig || !dirty) return;
    const timer = window.setTimeout(() => {
      const sent = configJson;
      saveTrack2PullConfig(JSON.parse(sent))
        .then(() => {
          // Saved, so nothing on screen is pending any more -- unless it
          // was edited while this was in flight, in which case that edit
          // still needs its own save and must stay marked.
          if (latestConfigRef.current === sent) setDirty(false);
        })
        .catch(() => undefined);
    }, 800);
    return () => window.clearTimeout(timer);
  }, [loadedConfig, dirty, configJson]);

  return (
    // collapsible={false}: this card runs its own Show/Hide,
    // because it shows a summary of live state when shut and
    // the generic one in Card cannot know what to put there.
    <Card
      title="Pull control"
      step="3"
      help={
        <>
          <p>
            Copies images out of the folders somebody uploads to, into dated{" "}
            <span className="tabular">week_N</span> folders you can train from.
            Nothing else on this page works until there is data in those folders.
          </p>
          <p className="mt-2">
            <strong>Which week a file lands in</strong> is decided by the
            timestamp in its own filename, not by when it was copied. So a
            re-run never re-files anything, and a late upload still lands in the
            week it was taken.
          </p>
          <p className="mt-2">
            <strong>Manually:</strong> pick the source folders, set the cut-off
            date (the card saves as you go), then press <em>Pull now</em>.
            <br />
            <strong>Automatically:</strong> in the <em>Pull</em> row of the
            toolbar pinned at the top of this page, choose{" "}
            <em>When files arrive</em> or <em>On a schedule</em>. A scheduled
            pull runs whether or not anything changed, and the card below
            shows it happening.
          </p>
          <p className="mt-2">
            It moves real files. Both containers share one bucket, so a pull
            started here is not a rehearsal.
          </p>
        </>
      }
      next="Step 4, Auto Training -- point at the week folders and decide how much of each to use."
      collapsible={false}
      titleAside={onOnlyChange ? (
        // Beside the title, where the card's own name is -- the same
        // question the Assemble card's heading asks, asked the same way.
        //
        // It has to live OUTSIDE anything that disables the card, or
        // unticking it would be a one-way door: the control that turns
        // the card back on would be inside the part that stopped taking
        // clicks. That is why this card dims its own body below instead
        // of being wrapped in `SwitchedOff`.
        <label className="ml-2 inline-flex items-center gap-1.5 text-xs font-normal text-ink-secondary">
          <input
            type="checkbox"
            checked={!only}
            onChange={(e) => onOnlyChange(!e.target.checked)}
            className="h-3.5 w-3.5"
          />
          Use Pull Control
        </label>
      ) : undefined}
      caption="Pick the folders. Everything below follows from them -- where files land, how a week is decided, and what fires a sweep."
      action={
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
        >
          {open ? "Hide" : "Show"}
        </button>
      }
    >

      {/* Said here, where the tick is, rather than only on the card
          that turns it off. */}
      {only && onOnlyChange && (
        <p className="mb-2 rounded-md border border-accent/40 bg-accent-subtle px-3 py-1.5 text-xs text-ink-secondary">
          <span className="font-medium text-ink">Off.</span> The run is using
          the folders picked on the Assemble card. Tick{" "}
          <span className="font-medium text-ink">Use Pull Control</span> to
          collect data week by week again.
        </p>
      )}
      <div
        inert={only && onOnlyChange ? true : undefined}
        className={only && onOnlyChange
          ? "pointer-events-none select-none opacity-45 saturate-50"
          : undefined}
      >
      {!open ? (
        <p className="text-xs text-ink-muted">
          {status
            ? `${(status.config.sources ?? []).length} folder${(status.config.sources ?? []).length === 1 ? "" : "s"}, ${status.weeks.length} week${status.weeks.length === 1 ? "" : "s"}, trigger: ${liveMode}`
            : "Press Show to configure."}
        </p>
      ) : (
        <div className="flex flex-col gap-3">
          {error && <p className="text-xs text-critical">{error}</p>}
          {note && <p className="tabular text-xs text-ink-secondary">{note}</p>}

          {/* The router sweeps unattended. If it is running different
              routing code than this card, what it files will not match
              what the preview showed -- and nothing else would say so. */}
          {status?.router && status.router.state !== "current" && (
            <p className={`rounded-md border p-2 text-xs ${
              status.router.state === "stale"
                ? "border-critical/50 bg-critical-subtle text-critical"
                : "border-black/15 bg-page text-ink-secondary"
            }`}>
              {status.router.state === "stale" ? (
                <>
                  <span className="font-medium">
                    The unattended router is running older routing code.
                  </span>{" "}
                  Sweeps fired by the trigger will file using the previous rules,
                  which may not match what Preview shows.{" "}
                  <span className="tabular block">
                    app {status.router.app_sha} · router {status.router.router_sha}
                    {status.router.router_modified
                      ? ` · built ${sgt(status.router.router_modified)}`
                      : ""}
                  </span>
                  <span className="tabular mt-1 block">
                    cd terraform/app &amp;&amp; terraform apply
                    -target=aws_lambda_function.pull_control_router
                  </span>
                </>
              ) : (
                <>Could not check the router version: {status.router.detail}</>
              )}
            </p>
          )}

          {status && !status.ready && (
            <p className="rounded-md border border-critical/40 p-2 text-xs text-critical">
              Not deployed yet{status.error ? ` -- ${status.error}` : ""}. Run{" "}
              <span className="tabular">terraform apply</span> in{" "}
              <span className="tabular">terraform/app</span> first.
            </p>
          )}

          {/* ------------------------------------------------- 1. bucket */}
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Bucket</span>
            <select
              className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
              value={bucket}
              onChange={(e) => {
                setBucket(e.target.value);
                // The picks belong to the old bucket; keeping them would
                // read one bucket while showing folders from another.
                setPicks({ ...EMPTY_PICKS });
                setBrowsePrefix("");
                setDirty(true);
              }}
            >
              {!buckets.some((b) => b.name === bucket) && (
                <option value={bucket}>{bucket}</option>
              )}
              {buckets.map((b) => (
                <option key={b.name} value={b.name} disabled={!b.readable}>
                  {b.name}{b.readable ? "" : "  (no read access)"}
                </option>
              ))}
            </select>
          </label>

          {/* -------------------------------------------- 2. the folders */}
          {LISTS.map(({ key, label, hint, side }) => {
            const check = key.endsWith("labels") ? pairing[side] : null;
            const wrong = !!check && (check.labels_without_images > 0 || check.paired === 0);
            return (
              <div key={key} className="rounded-md border border-black/10 p-2">
                <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-medium text-ink">
                    {label} <span className="font-normal text-ink-muted/70">({hint})</span>
                  </span>
                  <button
                    type="button"
                    onClick={() => {
                      setAddingTo(addingTo === key ? null : key);
                      setBrowsePrefix("");
                    }}
                    className="rounded-md border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary"
                  >
                    {addingTo === key ? "Cancel" : "Add folder"}
                  </button>
                </div>

                {picks[key].length === 0 ? (
                  <p className="text-xs text-ink-muted">None yet.</p>
                ) : (
                  <div className="flex flex-col gap-0.5">
                    {picks[key].map((folder) => (
                      <div key={folder} className="flex items-center gap-2 text-xs">
                        <span className="tabular min-w-0 flex-1 truncate text-ink-secondary"
                              title={folder}>{folder}</span>
                        <button
                          type="button"
                          onClick={() => {
                            setPicks((all) => ({
                              ...all, [key]: all[key].filter((f) => f !== folder),
                            }));
                            setDirty(true);
                          }}
                          className="shrink-0 text-critical underline"
                        >
                          remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}

                {/* Sits under the LABELS list of each pair, directly below
                    the two folders it is comparing. */}
                {check?.checked && (
                  <div className={`mt-2 rounded-md border p-2 text-xs ${
                    wrong ? "border-critical/50 bg-critical-subtle" : "border-black/15 bg-page"
                  }`}>
                    <p className={wrong ? "font-medium text-critical" : "text-ink-secondary"}>
                      {check.paired.toLocaleString()} of {check.images.toLocaleString()} image
                      {check.images === 1 ? "" : "s"} matched a label
                    </p>
                    {check.paired === 0 && (check.images > 0 || check.labels > 0) && (
                      <p className="mt-1 text-critical">
                        Nothing pairs at all -- these folders do not describe each other.
                      </p>
                    )}
                    {check.labels_without_images > 0 && (
                      <p className="mt-1 text-critical">
                        {check.labels_without_images.toLocaleString()} label
                        {check.labels_without_images === 1 ? "" : "s"} with no image. A label
                        that matches nothing is never read, so whatever it annotated is
                        missing from the run.
                        <span className="tabular mt-0.5 block truncate text-ink-muted">
                          e.g. {check.sample_labels_without_images.slice(0, 2).join(", ")}
                        </span>
                      </p>
                    )}
                    {check.images_without_labels > 0 && (
                      <p className="mt-1 text-ink-secondary">
                        {check.images_without_labels.toLocaleString()} image
                        {check.images_without_labels === 1 ? "" : "s"} with no label.
                        Ultralytics treats those as background examples -- fine in small
                        numbers, but a whole folder means the labels were not picked.
                      </p>
                    )}
                    {check.labels_without_images === 0 && check.images_without_labels === 0 && (
                      <p className="mt-0.5 text-ink-muted">
                        Every image has a label and every label has an image.
                      </p>
                    )}
                  </div>
                )}

                {addingTo === key && (
                  <div className="mt-2 rounded-md border border-accent/40 p-3">
                    <p className="mb-1.5 text-xs text-ink-secondary">
                      Choose a folder in{" "}
                      <span className="tabular font-medium text-ink">{bucket}</span> to add to{" "}
                      <span className="font-medium text-ink">{label}</span>
                    </p>
                    <Track2S3FolderBrowser
                      prefix={browsePrefix}
                      folders={browseFolders}
                      onNavigate={setBrowsePrefix}
                      /* No `onSelect`, so no tick boxes: clicking a
                         folder name walks into it, the path at the top
                         builds up, sub-folders appear, and Done commits
                         whatever folder you walked to. One thing to click
                         per step, and the folder being added is the one
                         you are looking at. */
                    />
                    <button
                      type="button"
                      disabled={!browsePrefix.trim()}
                      onClick={() => {
                        const chosen = browsePrefix.trim();
                        if (chosen) {
                          setPicks((all) => all[key].includes(chosen)
                            ? all : { ...all, [key]: [...all[key], chosen] });
                          setDirty(true);
                        }
                        setAddingTo(null);
                      }}
                      className="mt-2 rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
                    >
                      {browsePrefix.trim()
                        ? `Done -- add ${browsePrefix.trim()}`
                        : "Open a folder to add it"}
                    </button>
                  </div>
                )}
              </div>
            );
          })}

          {/* ------------------------------------- 3 & 4. roots and shape */}
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">
              Destination bucket{" "}
              <span className="text-ink-muted/70">(where the week folders are written)</span>
            </span>
            <select
              className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
              value={destBucket}
              onChange={(e) => { setDestBucket(e.target.value); setDirty(true); }}
            >
              <option value="">(this app's bucket)</option>
              {/* Only the buckets the stack grants writes to. A bucket
                  that cannot be written is a choice that fails later, so
                  it is not offered. */}
              {WRITABLE.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </label>

          {/* Browsable. Everything downstream keys off this path, and it
              was the one field you had to know by heart. No <label>
              wrapper -- S3FolderField is one. */}
          <div className="block">
            <S3FolderField
              label="Root folder"
              hint="(holds every week; train and val sit inside each one)"
              value={root}
              onChange={(folder) => { setRoot(folder); setDirty(true); }}
              bucket={destBucket}
              allowCreate
            />
            {/* Said WHERE the folder was chosen, not as a banner at the
                top of the card. The pull writes into this root,
                retention prunes it and the chain trains from it -- so a
                second configuration pointed here inherits the first
                one's data, and the first one's pulls reshape the
                second one's corpus. */}
            {(() => {
              const here = (root || "").replace(/^\/+|\/+$/g, "");
              const clash = claims.filter(
                (c) => c.root && c.root === here && c.name !== loadedFile);
              if (!here || !clash.length || rootClashOk === here) return null;
              return (
                <div className="mt-1 rounded-md border border-critical/30 bg-critical-subtle px-2.5 py-2">
                  <p className="text-xs text-critical">
                    A directory already is in use for the following model name:{" "}
                    {clash.map((c, i) => (
                      <span key={c.name}>
                        {i > 0 && ", "}
                        <strong>{c.model || "no model chosen"}</strong>
                        <span className="text-ink-muted"> (saved as {c.name})</span>
                      </span>
                    ))}
                    .
                  </p>
                  <p className="mt-1 text-xs text-ink-secondary">
                    Pulls into <span className="tabular">{here}</span> will reshape
                    what that configuration trains on, and retention will prune
                    its weeks. Pick another root, or override if both are meant
                    to share these folders.
                  </p>
                  <button
                    type="button"
                    onClick={() => setRootClashOk(here)}
                    className="mt-1.5 rounded-md border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary"
                  >
                    Use it anyway
                  </button>
                </div>
              );
            })()}
            <p className="mt-1 text-xs text-ink-muted">
              One root means one S3 prefix, which is what a trainer downloads --
              so a <span className="tabular">data.yaml</span> naming both splits
              resolves from a single fetch.
            </p>
          </div>

          <div className="grid gap-2 sm:grid-cols-4">
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Images subfolder</span>
              <input className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                     value={imagesSub} onChange={(e) => { setImagesSub(e.target.value); setDirty(true); }} />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Labels subfolder</span>
              <input className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                     value={labelsSub} onChange={(e) => { setLabelsSub(e.target.value); setDirty(true); }} />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Week prefix</span>
              {/* FIXED, and shown rather than hidden.
                  `plan_routing` and `week_at` skip any folder that does
                  not start with this prefix, so changing it does not
                  rename anything -- it makes every week folder already
                  in the bucket invisible to the system at once. There
                  are seven of them.
                  `readOnly` rather than `disabled`: a disabled input is
                  skipped by the keyboard and reads to a screen reader as
                  unavailable, when the value is the thing worth
                  reading. */}
              <input className="tabular w-full cursor-default rounded-md border border-black/15 bg-ink/[0.04] px-3 py-2 text-xs text-ink-secondary"
                     value={weekPrefix} readOnly
                     aria-describedby="week-prefix-fixed" />
              <span id="week-prefix-fixed" className="mt-1 block text-xs text-ink-muted">
                Fixed. The folders already pulled are named with it, and the
                sweep only sees folders that start with it.
              </span>
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">New week every</span>
              <select className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                      value={weekDays}
                      onChange={(e) => { setWeekDays(Number(e.target.value)); setDirty(true); }}>
                {/* Labelled in both units where they differ, because
                    "14 days" and "2 weeks" are the same choice and
                    people arrive with either one in mind. */}
                <option value={1}>1 day</option>
                <option value={3}>3 days</option>
                <option value={5}>5 days</option>
                <option value={7}>7 days (1 week)</option>
                <option value={14}>14 days (2 weeks)</option>
                <option value={28}>28 days (4 weeks)</option>
                <option value={30}>30 days</option>
              </select>
            </label>
          </div>

          {/* Which clock the cut-off is compared against. It sits above
              the cut-off because it decides what the cut-off MEANS: the
              same date against upload time and against capture time can
              put the same file in different weeks. */}
          <div className="rounded-md border border-hairline p-3">
            <p className="mb-1.5 text-xs font-medium text-ink">Date each file by</p>
            <div className="flex flex-col gap-1.5">
              {([
                ["capture_time", "The time in the filename",
                 "When the frame was taken -- ...__20260318T075125Z__... . A week then means a week of operations."],
                ["file_time", "When it was uploaded to S3",
                 "S3 LastModified. A week means a week of copying, which is rarely what anyone wants."],
                ["scan", "All of it into the current week",
                 "One week for the whole sweep, whatever the files' dates. For a source that carries no dates at all."],
              ] as const).map(([value, label, why]) => (
                <label key={value} className="flex items-start gap-2 text-xs text-ink-secondary">
                  <input type="radio" name="week-by" className="mt-0.5"
                         checked={weekBy === value}
                         onChange={() => { setWeekBy(value); setDirty(true); }} />
                  <span>
                    <span className={weekBy === value ? "font-medium text-ink" : ""}>{label}</span>
                    <span className="block text-ink-muted">{why}</span>
                  </span>
                </label>
              ))}
            </div>
            {weekBy === "capture_time" && (
              <p className="mt-1.5 text-xs text-ink-muted">
                A file whose name carries no timestamp falls back to its upload
                time rather than being dropped. The count of those is shown under
                &quot;waiting to be pulled&quot;, so a source that does not use the
                convention is visible.
              </p>
            )}
          </div>

          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">
              Cut-off date{" "}
              <span className="text-ink-muted/70">
                (the date in the filename, not when it was uploaded)
              </span>
            </span>
            <input
              type="date"
              className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs sm:w-48"
              value={anchorText}
              onChange={(e) => { setAnchorText(e.target.value); setDirty(true); }}
            />
            {/* Two states only. A native date input gives either a valid
                YYYY-MM-DD or nothing, so there is no half-typed date to
                warn about any more. */}
            {!anchorToIso(anchorText) ? (
              <p className="mt-1 text-xs text-ink-muted">
                Without one there is only {firstWeek} -- nothing has said when the
                baseline stops, so nothing can be after it.
              </p>
            ) : (
              <p className="mt-1 text-xs text-ink-muted">
                Everything captured <span className="font-medium">on or before{" "}
                {readableDate(anchorText)}</span> goes into {firstWeek}, however old.{" "}
                {weekPrefix}2 starts the next day and runs{" "}
                {weekDays} day{weekDays === 1 ? "" : "s"}, and so on.
              </p>
            )}
          </label>

          <div className="flex flex-col gap-1">
            <label className="flex items-center gap-2 text-xs text-ink-secondary">
              <input type="checkbox" checked={autoWeeks}
                     onChange={(e) => { setAutoWeeks(e.target.checked); setDirty(true); }} />
              Start a new week automatically
            </label>
            <label className="flex items-center gap-2 text-xs text-ink-secondary">
              <input type="checkbox" checked={backlog}
                     onChange={(e) => { setBacklog(e.target.checked); setDirty(true); }} />
              {firstWeek} takes everything already there
            </label>
          </div>

          {/* -------------------------------------- 5. what will be built */}
          <div className="rounded-md border border-black/15 bg-page p-3">
            <p className="mb-1.5 text-xs font-medium text-ink">What gets built</p>
            <pre className="tabular overflow-x-auto text-xs leading-5 text-ink-secondary">
{builtTree}
            </pre>
            <p className="mt-1 text-xs text-ink-muted">
              No <span className="tabular">train/</span> or{" "}
              <span className="tabular">val/</span> level: the root says which one it
              is, so repeating it inside would be saying it twice. Only the folders
              that actually receive files are created -- a folder here means
              something is in it.
            </p>
            <p className="mt-1 text-xs text-ink-muted">
              {firstWeek} is created by the first pull and takes everything already in those
              folders. Anything appearing afterwards goes into {weekPrefix}2, and a new week
              opens every {weekDays} day{weekDays === 1 ? "" : "s"} from then on.
            </p>
          </div>

          {/* ------------------------------ 6. every week, past and ahead */}
          <div className="rounded-md border border-black/15 bg-page p-3">
            <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-medium text-ink">
                Weeks{" "}
                <span className="font-normal text-ink-muted">
                  (three years back, being filled, and the next four)
                </span>
              </p>
              <div className="flex items-center gap-2">
                {/* Viewing another dataset is read-only. The live pull
                    keeps whatever is applied, which is why this says
                    "viewing" and not "load". */}
                <label className="flex items-center gap-1 text-xs text-ink-muted">
                  viewing
                  <select
                    className="rounded-md border border-black/15 bg-page px-2 py-1 text-xs"
                    value={viewing}
                    onChange={(e) => setViewing(e.target.value)}>
                    <option value="">this card</option>
                    {settingsFiles.map((file) => (
                      <option key={file.name} value={file.name}>{file.name}</option>
                    ))}
                  </select>
                </label>
                <label className="flex items-center gap-1 text-xs text-ink-muted">
                  <input type="checkbox" checked={watching}
                         onChange={(e) => setWatching(e.target.checked)} />
                  {watching ? (
                    <span className="flex items-center gap-1">
                      <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-accent" />
                      watching
                    </span>
                  ) : "watching"}
                </label>
                <button type="button" onClick={() => void refreshWeeks()}
                        className="text-xs font-medium text-ink-secondary underline">
                  Refresh
                </button>
              </div>
            </div>
            <p className="mb-1.5 text-xs text-ink-muted">
              Counts come from the week folders themselves, rechecked every 20
              seconds while this is open -- so a pull landing files shows up
              here without reloading the page.
              {checkedAt && <> Last looked {sgt(checkedAt)} SGT.</>}
            </p>
            {viewing ? (
              <p className="mb-1.5 text-xs text-accent">
                Showing the folders of <span className="tabular">{viewing}</span>.
                Nothing here is applied -- the live pull still uses whatever is
                saved, and the fields above are unchanged.
              </p>
            ) : (
              <p className="mb-1.5 text-xs text-ink-muted">
                Showing this card&apos;s settings, prefix{" "}
                <span className="tabular text-ink-secondary">{weekPrefix}</span>
                {dirty && " (saving...)"}.
              </p>
            )}

            {forecast === null ? (
              <p className="text-xs text-ink-muted">Reading...</p>
            ) : forecast.length === 0 ? (
              <p className="text-xs text-ink-muted">Nothing yet.</p>
            ) : (
              <>
                <div className="tabular mb-1 flex gap-2 text-xs text-ink-muted">
                  <span className="w-20 shrink-0">week</span>
                  <span className="w-44 shrink-0">covers</span>
                  <span className="w-36 shrink-0">closes</span>
                  <span className="w-44 shrink-0">contents</span>
                  <span className="w-36 shrink-0">state</span>
                  <span className="flex-1">current active</span>
                </div>
                <div className="flex max-h-[28rem] flex-col gap-0.5 overflow-y-auto">
                  {visibleWeeks.map((w) => (
                    <div key={w.week}
                         className={`flex flex-wrap items-center gap-2 text-xs ${
                           w.week === activeWeek ? "font-medium text-ink" : "text-ink-secondary"
                         }`}>
                      <span className="tabular w-20 shrink-0">{w.week}</span>
                      <span className="tabular w-44 shrink-0 text-ink-muted">
                        {w.open_start
                          ? "everything up to the cut-off"
                          : w.starts_at
                            ? `${dayOf(w.starts_at)} -> ${lastDay(w.ends_at)}`
                            : "-"}
                      </span>
                      {/* Every week has a closing moment, and week 1's IS
                          the cut-off. Saying so on the row means the date
                          that decides the whole split is readable next to
                          the folder it decides, not only in the field
                          above. */}
                      <span className="tabular w-36 shrink-0">
                        {w.closes_at ? (
                          <span className={w.is_cutoff ? "text-accent" : "text-ink-muted"}>
                            {lastDay(w.closes_at)}
                            {w.is_cutoff && (
                              <span className="ml-1 font-medium">cut-off</span>
                            )}
                          </span>
                        ) : (
                          <span className="text-ink-muted">set a cut-off</span>
                        )}
                      </span>
                      <span className="tabular w-44 shrink-0 text-ink-muted">
                        {/* A week that has BEEN and received nothing is
                            empty, and 0 img / 0 lbl says so. Only a week
                            that has not happened yet gets "--" -- there
                            is nothing to count about the future. */}
                        {w.exists
                          ? `${w.images.toLocaleString()} img, ${w.labels.toLocaleString()} lbl`
                            + (w.bytes ? ` · ${(w.bytes / 1048576).toFixed(1)} MB` : "")
                          : "--"}
                        {grew[w.week] && (
                          <span className="ml-1 font-medium text-accent">
                            +{grew[w.week].toLocaleString()}
                          </span>
                        )}
                      </span>
                      {/* One question only: does the folder exist? It used
                          to also carry whether the week was being filled,
                          which made "filling now -- created on the next
                          pull" a single cell saying two things that sound
                          contradictory. Being the active week is its own
                          column now. */}
                      {/* One question, three answers, all about the
                          FOLDER: it holds files, it exists and holds
                          none, or it is not there. */}
                      <span className="w-36 shrink-0">
                        {w.exists
                          ? (w.state === "empty"
                              ? <span className="text-ink-muted">empty</span>
                              : <span className="text-ink-secondary">present</span>)
                          : <span className="text-ink-muted/70">waiting to be created</span>}
                      </span>
                      <span className="flex-1">
                        {w.week === activeWeek && (
                          <span className="font-medium text-accent">
                            Current folder where files are placed
                          </span>
                        )}
                      </span>
                      {w.exists && (
                        confirmPurge === w.week ? (
                          <>
                            {/* The COUNT, not just "delete its files?".
                                "Delete week_42" and "delete 1,496 files"
                                are not the same sentence, and the number
                                is also what the server now demands back
                                before it will act. */}
                            <span className="text-critical">
                              {purgeCount === null
                                ? "counting..."
                                : `delete ${purgeCount.toLocaleString()} object${purgeCount === 1 ? "" : "s"}?`}
                            </span>
                            <button type="button"
                              disabled={purgeCount === null || purgeCount === 0}
                              onClick={() => withBusy("Deleting...", async () => {
                                const out = await purgeTrack2Week(w.week, purgeCount ?? -1);
                                setConfirmPurge("");
                                setPurgeCount(null);
                                await refreshWeeks();
                                setNote(`Deleted ${out.deleted} objects from ${out.week}.`);
                              })}
                              className="rounded-md bg-red-600 px-2 py-0.5 text-xs font-medium text-white">
                              Yes, delete
                            </button>
                            <button type="button"
                                    onClick={() => { setConfirmPurge(""); setPurgeCount(null); }}
                                    className="text-xs text-ink-secondary underline">cancel</button>
                          </>
                        ) : (
                          <button type="button" onClick={() => {
                            setConfirmPurge(w.week);
                            setPurgeCount(null);
                            void previewTrack2WeekPurge(w.week)
                              .then((p) => setPurgeCount(p.objects))
                              .catch(() => setPurgeCount(null));
                          }}
                                  className="text-xs text-critical underline">delete</button>
                        )
                      )}
                    </div>
                  ))}
                </div>
                {/* Renaming the prefix does not rename what is already
                    on S3. The old folders stay, under their old names,
                    still holding their files -- and they keep appearing
                    here, because this table lists what is actually in
                    the bucket. Nothing is lost and nothing is re-copied
                    (dedup scans the whole root, not one week folder),
                    but the tree ends up with two naming schemes in it. */}
                {savedPrefix && weekPrefix !== savedPrefix && existingWeeks > 0 && (
                  <p className="mt-1.5 text-xs text-critical">
                    Folders already exist under{" "}
                    <span className="tabular">{savedPrefix}</span>. Renaming the
                    prefix to <span className="tabular">{weekPrefix}</span> does not
                    rename them -- they keep their files and their old names, and new
                    weeks are created as{" "}
                    <span className="tabular">{weekPrefix}N</span> beside them. Nothing
                    is re-copied; the tree just carries both names.
                  </p>
                )}

                {/* The two can differ, and when they do it looks like a
                    bug: week 1 marked as the active folder although the
                    clock is well past its cut-off. Saying why is cheaper
                    than letting someone work it out. */}
                {activeWeek && currentWeek && activeWeek !== currentWeek && (
                  <p className="mt-1.5 text-xs text-ink-muted">
                    Today falls in <span className="tabular">{currentWeek}</span>, but
                    the next pull fills <span className="tabular">{activeWeek}</span>{" "}
                    first: each file goes to the week its own arrival date falls in,
                    and everything waiting in the source folders predates the cut-off.
                    Once that backlog is in,{" "}
                    <span className="tabular">{currentWeek}</span> takes over.
                  </p>
                )}

                {/* Only the holes BETWEEN weeks that hold files. The
                    stretch after the last collection is the present, not
                    a gap, and giving each passing week a folder would
                    fill the bucket during any quiet period. */}
                <div className="mt-1.5 flex flex-wrap items-center gap-2">
                  <button type="button" disabled={!!busy}
                    onClick={() => withBusy("Creating...", async () => {
                      const got = await fillTrack2WeekGaps();
                      await refreshWeeks();
                      setNote(got.created.length
                        ? `Created ${got.created.join(", ")} (${got.keys} folders).`
                        : "No gaps to fill -- every week between the filled ones already exists.");
                    })}
                    className="rounded-md border border-black/15 px-3 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40">
                    Create empty week folders
                  </button>
                  <span className="text-xs text-ink-muted">
                    for weeks BETWEEN those that hold files -- they can then be
                    deleted like any other
                  </span>
                </div>

                {/* The only control in the app that destroys images.
                    Two-step on purpose: the count is fetched and shown
                    FIRST, because "delete the week folders" and "delete
                    2,401 files" are not the same sentence, and the
                    number has to be typed back -- not a checkbox and not
                    the word yes, because the number is the only thing
                    that proves the person read what they were about to
                    lose, and it changes as the corpus does so it cannot
                    become muscle memory. */}
                <div className="mt-3 rounded-md border border-critical/40 bg-critical-subtle p-2.5">
                  <p className="text-xs font-medium text-critical">
                    Delete the week folders and everything in them
                  </p>
                  <p className="mt-0.5 text-xs text-ink-secondary">
                    Permanent. There is no second copy &mdash; the pull{" "}
                    <strong>moves</strong> files out of the drop folders rather
                    than copying them, so these are the only ones. The week
                    definitions and the data.yaml are left alone.
                  </p>
                  {!purge ? (
                    <button
                      type="button"
                      disabled={!!busy}
                      onClick={() => withBusy("Counting...", async () => {
                        const got = await previewTrack2WeekPurge();
                        setPurge(got);
                        setPurgeTyped("");
                      })}
                      className="mt-1.5 rounded-md border border-critical/40 px-3 py-1 text-xs font-medium text-critical disabled:opacity-40">
                      Count what would be deleted
                    </button>
                  ) : (
                    <div className="mt-1.5">
                      <p className="tabular text-xs text-ink">
                        <span className="font-medium text-critical">
                          {purge.objects}
                        </span>{" "}
                        objects, {(purge.bytes / 1048576).toFixed(1)} MB, across{" "}
                        {Object.keys(purge.prefixes).length} folder(s):
                      </p>
                      <ul className="tabular mt-0.5 text-xs text-ink-muted">
                        {Object.entries(purge.prefixes).map(([prefix, n]) => (
                          <li key={prefix}>{prefix} &mdash; {n}</li>
                        ))}
                      </ul>
                      <label className="mt-1.5 block text-xs text-ink-secondary">
                        Type <span className="tabular font-medium">{purge.objects}</span>{" "}
                        to confirm
                        <input
                          className="tabular ml-2 w-28 rounded border border-black/15 bg-page px-2 py-1 text-xs"
                          value={purgeTyped}
                          onChange={(e) => setPurgeTyped(e.target.value)}
                        />
                      </label>
                      <div className="mt-1.5 flex gap-2">
                        <button
                          type="button"
                          disabled={!!busy || purgeTyped.trim() !== String(purge.objects)}
                          onClick={() => withBusy("Deleting...", async () => {
                            const got = await purgeTrack2Weeks(purge.objects);
                            setPurge(null);
                            setPurgeTyped("");
                            setNote(got.errors.length
                              ? `${got.message} ${got.errors.length} failed.`
                              : got.message);
                            await load();
                          })}
                          className="rounded-md bg-critical px-3 py-1 text-xs font-medium text-white disabled:opacity-40">
                          Delete {purge.objects} objects permanently
                        </button>
                        <button
                          type="button"
                          disabled={!!busy}
                          onClick={() => { setPurge(null); setPurgeTyped(""); }}
                          className="rounded-md border border-black/15 px-3 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40">
                          Cancel
                        </button>
                      </div>
                    </div>
                  )}
                </div>
                <p className="mt-1.5 text-xs text-ink-muted">
                  This also happens at the end of a pull that moved something. Weeks
                  after the last collection are left alone -- nothing has happened in
                  them yet, so they stay <span className="tabular">waiting to be
                  created</span>.
                </p>

                {weeksHidden > 0 && (
                  <p className="mt-1.5 text-xs text-ink-muted">
                    Showing the most recent {weekWindow} weeks (three years).{" "}
                    <span className="text-ink-muted/70">
                      {weeksHidden} earlier week{weeksHidden === 1 ? " is" : "s are"} out
                      of view -- still on S3, still counted, just not drawn.
                    </span>{" "}
                    <button type="button" onClick={() => setShowAllWeeks(true)}
                            className="font-medium text-ink-secondary underline">
                      Show all
                    </button>
                  </p>
                )}
                {showAllWeeks && settled.length > weekWindow && (
                  <p className="mt-1.5 text-xs text-ink-muted">
                    Showing every week.{" "}
                    <button type="button" onClick={() => setShowAllWeeks(false)}
                            className="font-medium text-ink-secondary underline">
                      Back to the last {weekWindow}
                    </button>
                  </p>
                )}

                {/* Changing the week length re-labels folders that
                    already hold files. The files do not move, so a
                    folder gathered under seven-day weeks keeps its
                    contents and gains a fourteen-day name. Worth saying
                    before Apply, not after. */}
                {previewingWeeks && dirty && existingWeeks > 0 && (
                  <p className="mt-1.5 text-xs text-critical">
                    These dates are what your unsaved settings would mean.{" "}
                    {existingWeeks} week folder{existingWeeks === 1 ? "" : "s"} already
                    exist{existingWeeks === 1 ? "s" : ""}: applying this re-dates{" "}
                    {existingWeeks === 1 ? "it" : "them"} without moving a single
                    file, so what is already in {firstWeek} keeps its contents under
                    the new boundaries.
                  </p>
                )}

                {!anchorToIso(anchorText) && (
                  <p className="mt-1.5 text-xs text-ink-muted">
                    Set a cut-off date above and the weeks after it appear here.
                  </p>
                )}
              </>
            )}
          </div>

          {/* ------------------------------------ 7. what fires a pull */}
          <div className="rounded-md border border-hairline p-3">
            <p className="mb-1 text-xs font-medium text-ink">Pull</p>
            <p className="mb-2 text-xs text-ink-muted">
              A pull walks every folder above and copies what is new into the week
              it belongs to --{" "}
              <span className="tabular">
                {root || "<root>"}/{activeWeek || `${weekPrefix}N`}/{imagesSub}/train/
              </span>{" "}
              for training,{" "}
              <span className="tabular">
                .../{imagesSub}/val/
              </span>{" "}
              for evaluation. Image or label comes from the file extension; the split
              comes from the list the folder is in. Anything already filed under the
              root is skipped, so a pull only ever copies what has appeared since the
              last one. Originals are never touched.
            </p>

            {/* The manual pull, first: it uses exactly the rules configured
                above, so it is also how you check them before arming
                anything. */}
            <div className="mb-3 flex flex-wrap items-center gap-2">
              <button type="button" disabled={!!busy || !anyPicked}
                onClick={() => withBusy("Looking...", async () => {
                  const got = await previewTrack2Pull();
                  setPlan(got);
                  setNote(`${got.planned.length} file${got.planned.length === 1 ? "" : "s"} would be copied.`);
                })}
                className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40">
                Preview
              </button>
              <button type="button"
                disabled={!!busy || !anyPicked || unpaired || run?.status === "running"}
                onClick={() => withBusy("Starting...", async () => {
                  const started = await startTrack2Pull(20000);
                  setPlan(null);
                  // Only the id comes back. Everything after this is the
                  // poll above, which is what makes progress and Stop
                  // possible at all.
                  setRun({
                    run_id: started.run_id, status: "running", trigger: "manual",
                    total: 0, done: 0, moved: 0, skipped: 0, errors: 0,
                  });
                  setNote("");
                })}
                className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40">
                {run?.status === "running" ? "Pulling..." : (busy || "Pull now")}
              </button>
              <span className="text-xs text-ink-muted">
                {anyPicked
                  ? <>runs the rules above once, filing each file by its own
                      date -- starting with{" "}
                      <span className="tabular">{activeWeek || firstWeek}</span></>
                  : "add folders above first"}
              </span>
            </div>

            {run && (
              <div className="mb-3 rounded-md border border-hairline bg-page p-2.5">
                {(() => {
                  const total = run.total || 0;
                  // No bar until the counting pass has a denominator. A
                  // percentage of an unknown total is a guess drawn as a
                  // fact.
                  const pct = total ? Math.min(Math.round((run.done / total) * 100), 100) : 0;
                  const live = run.status === "running";
                  return (
                    <>
                      <div className="mb-1.5 flex flex-wrap items-center justify-between gap-2">
                        <p className="text-xs font-medium text-ink">
                          {live && (run.stop_requested || stopping ? "Stopping..." : "Pulling")}
                          {run.status === "done" && "Pull finished"}
                          {run.status === "stopped" && "Pull stopped"}
                          {run.status === "failed" && "Pull failed"}
                          {run.status === "lost" && "Pull lost"}
                          <span className="ml-2 font-normal text-ink-muted">
                            {run.trigger === "manual" ? "started by hand" : `trigger: ${run.trigger}`}
                          </span>
                        </p>
                        <div className="flex items-center gap-2">
                          {live && (
                            <button type="button" disabled={stopping || !!run.stop_requested}
                              onClick={() => withBusy("Stopping...", async () => {
                                setStopping(true);
                                try {
                                  const got = await stopTrack2Pull(run.run_id);
                                  setRun(got);
                                } catch (err) {
                                  // Nothing else clears this. The poll
                                  // above only clears it when the run
                                  // LEAVES `running`, and a stop that
                                  // failed leaves it running -- so the
                                  // button would stay disabled for the
                                  // rest of the sweep and Stop could
                                  // never be pressed again.
                                  setStopping(false);
                                  throw err;
                                }
                              })}
                              className="rounded-md border border-critical/40 px-3 py-1 text-xs font-medium text-critical disabled:opacity-40">
                              {run.stop_requested || stopping ? "Stopping..." : "Stop"}
                            </button>
                          )}
                          {!live && (
                            <button type="button" onClick={() => setRun(null)}
                                    className="text-xs text-ink-secondary underline">
                              dismiss
                            </button>
                          )}
                        </div>
                      </div>

                      {/* The bar. Indeterminate while the denominator is
                          still being counted -- it is a LIST pass over
                          every source, which takes a moment on a large
                          one. */}
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/10">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            run.status === "stopped" || run.status === "failed"
                              ? "bg-critical" : "bg-accent"
                          } ${live && !total ? "w-1/4 animate-pulse" : ""}`}
                          style={total ? { width: `${pct}%` } : undefined}
                        />
                      </div>

                      <p className="tabular mt-1.5 text-xs text-ink-muted">
                        {total
                          ? <>{pct}% -- {run.done.toLocaleString()} of {total.toLocaleString()} looked at,{" "}
                              {run.moved.toLocaleString()} copied, {run.skipped.toLocaleString()} skipped
                              {run.errors ? `, ${run.errors} failed` : ""}</>
                          : "counting what is there..."}
                      </p>

                      {run.status === "lost" && (
                        <p className="mt-1 text-xs text-critical">
                          No progress for {Math.round((run.quiet_seconds ?? 0) / 60)} minutes.
                          Its container was probably replaced, or a scheduled run hit the
                          Lambda timeout. Whatever it copied is still there and the next
                          pull will skip it.
                        </p>
                      )}

                      {run.undone && (
                        run.undone.refused ? (
                          <p className="mt-1 text-xs text-critical">
                            Nothing undone. {run.undone.refused}
                          </p>
                        ) : (
                          <p className="mt-1 text-xs text-ink-secondary">
                            Undone: {run.undone.removed.toLocaleString()} of{" "}
                            {run.undone.created.toLocaleString()} files this run created
                            were removed.
                            {run.undone.failed.length > 0 && (
                              <span className="text-critical">
                                {" "}{run.undone.failed.length} could not be removed --
                                the destination is part-way, see the console.
                              </span>
                            )}
                          </p>
                        )
                      )}

                      {run.aborted && run.status !== "stopped" && (
                        <p className="mt-1 text-xs text-critical">{run.aborted}</p>
                      )}
                    </>
                  );
                })()}
              </div>
            )}

            {/* ------------------------------------ waiting to be pulled */}
            {pending && (
              <div className="mb-3 rounded-md border border-hairline bg-page p-2.5">
                <p className="mb-1 text-xs font-medium text-ink">
                  Waiting to be pulled{" "}
                  <span className="font-normal text-ink-muted">
                    (what the next pull, scheduled or by hand, would copy)
                  </span>
                </p>
                {pending.waiting === 0 ? (
                  <p className="text-xs text-ink-muted">
                    Nothing new. {pending.already_filed.toLocaleString()} file
                    {pending.already_filed === 1 ? " is" : "s are"} already filed and would
                    be skipped.
                  </p>
                ) : (
                  <>
                    <p className="tabular text-xs text-ink-secondary">
                      {pending.waiting.toLocaleString()}
                      {pending.capped && "+"} file
                      {pending.waiting === 1 ? "" : "s"} would be copied
                      {pending.already_filed > 0 && (
                        <span className="text-ink-muted">
                          {" "}({pending.already_filed.toLocaleString()} already filed, skipped)
                        </span>
                      )}
                    </p>
                    <div className="mt-1 flex flex-col gap-0.5">
                      {pending.weeks.map((w) => (
                        <p key={w.week} className="tabular text-xs text-ink-muted">
                          <span className="inline-block w-24">{w.week}</span>
                          {w.files.toLocaleString()} file{w.files === 1 ? "" : "s"}
                          {w.bytes > 0 && ` · ${(w.bytes / 1048576).toFixed(1)} MB`}
                          {!w.exists && (
                            <span className="ml-2 text-accent">folder created by the pull</span>
                          )}
                        </p>
                      ))}
                    </div>
                    {pending.capped && (
                      <p className="mt-1 text-xs text-ink-muted">
                        Stopped counting at {pending.scanned.toLocaleString()} -- there may
                        be more.
                      </p>
                    )}
                  </>
                )}
                {(pending.fell_back_to_upload_time ?? 0) > 0 && (
                  <p className="mt-1 text-xs text-critical">
                    {pending.fell_back_to_upload_time!.toLocaleString()} file(s) carry no
                    timestamp in the name and were dated by upload time instead. Their
                    week is when they were copied, not when they were taken.
                  </p>
                )}
                <p className="tabular mt-1 text-xs text-ink-muted/70">
                  checked {sgt(pending.checked_at)} SGT
                  {pending.dated_by && ` · dated by ${pending.dated_by.replace("_", " ")}`}
                </p>
              </div>
            )}

            {/* READ-ONLY here. These controls moved to the pinned bar
                at the top of the page -- the one place that is always
                reachable -- and leaving an editable copy behind would
                be two controls for one setting, which is the shape of
                every "I changed it and it did not take" bug on this
                page. This states what is in effect and where to change
                it. */}
            <p className="mb-1 text-xs font-medium text-ink">How it runs automatically</p>
            <p className="tabular text-xs text-ink-secondary">
              {wantedMode === "off" && "Off -- nothing fires on its own."}
              {wantedMode === "arrival" && "When files arrive."}
              {wantedMode === "schedule" && (
                <>
                  On a schedule
                  {pullEveryDays ? ` -- every ${pullEveryDays} day(s)` : ""}
                  {pullTime ? ` at ${pullTime} SGT` : ""}.
                </>
              )}
            </p>
            <p className="mt-0.5 text-xs text-ink-muted">
              Change it in the <strong>Pull</strong> row of the toolbar pinned
              at the top of the page.
            </p>

            <p className="mt-1.5 text-xs text-ink-muted">
              {wantedMode === "off" && "Nothing runs on its own -- only Pull now."}
              {wantedMode === "arrival" && "Every object written to the bucket wakes the router, which pulls only if that file was under one of these folders. Needs EventBridge notifications on the bucket."}
              {wantedMode === "schedule" && (
                anchorToIso(anchorText)
                  ? `A pull runs every ${pullEveryDays} day${pullEveryDays === 1 ? "" : "s"}, whether or not anything changed.`
                  : "Nothing is armed until a cut-off date is set."
              )}
            </p>
            {triggers && (
              <p className="tabular mt-1 text-xs text-ink-muted">
                arrival: {triggers.arrival.exists ? (triggers.arrival.enabled ? "enabled" : "disabled") : triggers.arrival.error}
                {"   "}schedule: {triggers.schedule.exists ? (triggers.schedule.enabled ? "enabled" : "disabled") : triggers.schedule.error}
              </p>
            )}

            {plan && (
              <div className="mt-2 rounded-md border border-black/15 bg-page p-2">
                {plan.would_create.length > 0 && (
                  <p className="mb-1 text-xs text-critical">
                    Will also create: {plan.would_create.join(", ")}
                  </p>
                )}
                <p className="mb-1 text-xs font-medium text-ink">
                  {plan.planned.length} to copy, {plan.skipped.length} skipped
                </p>
                <div className="flex max-h-40 flex-col gap-0.5 overflow-y-auto">
                  {plan.planned.slice(0, 200).map((e) => (
                    <p key={e.key} className="tabular truncate text-xs text-ink-secondary">
                      {e.key} {"->"} {e.destination}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Saved settings moved to the Retention card, above
              Repeatable runs. One file now holds every card's
              choices, so it belongs beside the other things that
              record what a run was given -- not inside the one card
              it used to be about. */}

          {/* ------------------------------- 9. saved as you go */}
          {/* No Apply settings button (removed 2026-10-26). Every change
              is saved 800ms after it is made, and `POST /config` rebuilds
              the sources from the picks itself -- so the button only
              re-sent what had already been sent. The trigger is the
              toolbar's. */}
          {pullTimeBroken && (
            <p className="text-xs text-critical">
              The pull time is not a valid HH:MM, so the pull keeps the
              time it had.
            </p>
          )}
          {!anyPicked && (
            <p className="text-xs text-ink-muted">Add at least one folder before saving.</p>
          )}
          {unpaired && (
            <p className="text-xs text-critical">
              The {unpairedSides.join(" and ")} side has image folders picked and no
              label folder. Those images would be filed as background examples
              &mdash; the trainer accepts them and learns nothing from them. Pick
              the {unpairedSides.join(" and ")} label folder, or remove the images.
              A scheduled pull refuses this for the same reason.
            </p>
          )}

          {/* ---------------------------------------- 10. recent sweeps */}
          <div className="rounded-md border border-black/15 bg-page p-3">
            <p className="mb-1 text-xs font-medium text-ink">
              Recent sweeps{" "}
              <span className="font-normal text-ink-muted">(times are SGT)</span>
            </p>
            {(status?.runs ?? []).length === 0 ? (
              <p className="text-xs text-ink-muted">None yet.</p>
            ) : (
              <div className="flex max-h-60 flex-col gap-0.5 overflow-y-auto">
                {(status?.runs ?? []).slice(0, 15).map((run) => (
                  <p key={run.at} className="tabular text-xs text-ink-secondary">
                    {sgt(run.at)} {run.trigger} -- copied {run.moved},
                    skipped {run.skipped}{run.errors ? `, ${run.errors} failed` : ""}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </Card>
  );
}
