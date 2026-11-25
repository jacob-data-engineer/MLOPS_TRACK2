import { CalendarDays, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Card } from "../components/Card";
import { useOtherSchedules } from "../components/Track2OtherSchedules";
import { formatDayTime } from "../lib/format";
import {
  ApiError,
  getTrack2Scheduled,
  type Track2Scheduled,
  type Track2ScheduledEvent,
} from "../lib/api";

/**
 * What the clocks will start, and when -- for the loaded settings file
 * and, side by side, for any saved file picked here.
 *
 * READ-ONLY. Every control stays on Multi-Auto Training; this page only
 * draws what the server says, computed by the same rules the queue
 * runner acts on (`scheduled_events`).
 *
 * Only ONE configuration is loaded at a time, and only its clocks fire.
 * A saved file that is not loaded is shown as what it WOULD run if it
 * were -- marked "not loaded" everywhere it appears, because a calendar
 * that drew both the same would promise runs that will not happen.
 *
 * Everything is in Singapore time, bucketed by SGT date.
 */

type CalendarEvent = Omit<Track2ScheduledEvent, "kind"> & {kind: "pull" | "train" | "sql" | "export"; details?: string};
const WINDOW_DAYS = 92;
const SGT_MS = 8 * 3600 * 1000;
const DAY_MS = 24 * 3600 * 1000;
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** One colour per file, in the order the files are listed. */
const PALETTE = ["#b45309", "#2563eb", "#059669", "#9333ea", "#dc2626", "#0891b2", "#ca8a04", "#db2777"];
/** Per-viewer convenience only: which saved files were last picked. */
const PICK_KEY = "track2-scheduled-files-v2";

/** What each source means, in the words the cards use. */
const SOURCE_NAMES: Record<string, string> = {
  "Pull": "Pull Control's week folders",
  "Golden set": "Golden set (its val side is the held-out evaluation)",
  "Assembled folders": "Assemble a dataset folders",
};

function sgtKey(ms: number): string {
  return new Date(ms + SGT_MS).toISOString().slice(0, 10);
}

function sgtClock(iso: string): string {
  return new Date(Date.parse(iso) + SGT_MS).toISOString().slice(11, 16);
}

function mondayOf(ms: number): number {
  const local = new Date(ms + SGT_MS);
  const dow = (local.getUTCDay() + 6) % 7;
  const midnight = Date.UTC(local.getUTCFullYear(), local.getUTCMonth(), local.getUTCDate());
  return midnight - dow * DAY_MS - SGT_MS;
}

function monthStart(ms: number, offset: number): number {
  const local = new Date(ms + SGT_MS);
  return Date.UTC(local.getUTCFullYear(), local.getUTCMonth() + offset, 1) - SGT_MS;
}

function dayLabel(ms: number): { day: number; month: string } {
  const local = new Date(ms + SGT_MS);
  return { day: local.getUTCDate(), month: MONTHS[local.getUTCMonth()] };
}

function readPicks(): string[] | "*" {
  try {
    const raw = window.localStorage.getItem(PICK_KEY);
    if (!raw) return "*";
    const got = JSON.parse(raw);
    return got === "*" ? "*" : Array.isArray(got) ? got.map(String) : [];
  } catch {
    return [];
  }
}

function writePicks(picks: string[] | "*") {
  try {
    window.localStorage.setItem(PICK_KEY, JSON.stringify(picks));
  } catch {
    // Private window or blocked storage: the picker still works, it just
    // will not be remembered.
  }
}

/** The pop-up on hover: which file, what kind of run, and from what. */
function EventPopup({ ev, color }: { ev: CalendarEvent; color: string }) {
  const train = ev.kind === "train";
  return (
    <div role="tooltip" className="pointer-events-none absolute left-0 top-full z-50 mt-1 hidden w-64 rounded-sm border border-hairline bg-surface p-2 text-xs text-ink shadow-lg group-hover:block group-focus-within:block">
      <p className="flex items-center gap-1.5 font-medium">
        <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
        <span className="tabular break-all">{ev.file}</span>
      </p>
      <p className={ev.live ? "text-accent" : "text-ink-muted"}>
        {ev.kind === "sql" || ev.kind === "export" ? "Enabled schedule" : ev.live ? "Loaded -- this runs" : "Not loaded -- would run if this file were loaded"}
      </p>
      <p className="mt-1 tabular">{formatDayTime(ev.at)}</p>
      <p>{ev.label}</p>
      {ev.details ? <p className="mt-1">{ev.details}</p> : train ? (
        <div className="mt-1">
          <p className="text-ink-muted">Trains on:</p>
          {ev.sources.length ? (
            <ul className="ml-4 list-disc">
              {(["Assembled folders", "Golden set", "Pull"] as const).map((s) => (
                <li key={s} className={ev.sources.includes(s) ? "" : "text-ink-muted line-through"}>
                  {SOURCE_NAMES[s]}
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-critical">nothing ticked -- the run would have no data</p>
          )}
        </div>
      ) : (
        <p className="mt-1 text-ink-muted">Pull Control files new images into week folders.</p>
      )}
    </div>
  );
}

function EventLine({ ev, color, compact }: { ev: CalendarEvent; color: string; compact?: boolean }) {
  const train = ev.kind === "train";
  return (
    <div className="group relative" tabIndex={0} aria-label={`${ev.label}: ${ev.file}, ${formatDayTime(ev.at)}`}>
      <div
        className={`flex items-center gap-1 rounded-sm border-l-[3px] px-1 py-0.5 text-[11px] leading-4 ${
          train ? "bg-accent-subtle text-ink" : "bg-black/[0.04] text-ink-secondary"} ${
          ev.live ? "" : "opacity-75"}`}
        style={{ borderLeftColor: color }}
      >
        <span className="tabular">{sgtClock(ev.at)}</span>
        <span className="shrink-0">{ev.kind === "sql" ? "SQL pull" : ev.kind === "export" ? "Export" : train ? "Train" : "Pull"}</span>
        <span className="truncate text-ink-muted">{compact ? "" : "· "}{ev.file}</span>
      </div>
      <EventPopup ev={ev} color={color} />
    </div>
  );
}

export function Track2ScheduledPage() {
  const [data, setData] = useState<Track2Scheduled | null>(null);
  const [error, setError] = useState("");
  const {jobs, errors: otherErrors} = useOtherSchedules();
  const [offset, setOffset] = useState(0);
  const [picks, setPicks] = useState<string[] | "*">(readPicks);
  const [loading, setLoading] = useState(false);
  const [showTraining, setShowTraining] = useState(true);
  const [showSql, setShowSql] = useState(true);
  const [showExports, setShowExports] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = () => {
      setLoading(true);
      getTrack2Scheduled(WINDOW_DAYS, picks)
        .then((d) => { if (!cancelled) { setData(d); setError(""); } })
        .catch((err) => !cancelled && setError(err instanceof ApiError ? err.detail : "Could not read the schedule."))
        .finally(() => !cancelled && setLoading(false));
    };
    load();
    const timer = window.setInterval(load, 60_000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, [picks]);

  function choose(next: string[] | "*") {
    setPicks(next);
    writePicks(next);
  }

  const colorOf = useMemo(() => {
    const map: Record<string, string> = {};
    (data?.files ?? []).forEach((f, i) => { map[f.name] = PALETTE[i % PALETTE.length]; });
    return (name: string) => map[name] ?? PALETTE[0];
  }, [data]);

  const byDay = useMemo(() => {
    const out: Record<string, CalendarEvent[]> = {};
    const add = (ev: CalendarEvent) => (out[sgtKey(Date.parse(ev.at))] ??= []).push(ev);
    if (showTraining) for (const ev of data?.events ?? []) add(ev);
    const start = data ? Date.parse(data.now) : Date.now();
    const end = data ? Date.parse(data.until) : start + WINDOW_DAYS * DAY_MS;
    for (const job of jobs) {
      if (!job.enabled || !job.next || !(job.kind === "sql" ? showSql : showExports)) continue;
      const interval = Math.max(1, job.everyDays || 1) * DAY_MS;
      let at = Date.parse(job.next);
      if (!Number.isFinite(at)) continue;
      if (at < start) at += Math.ceil((start - at) / interval) * interval;
      for (; at <= end; at += interval) add({at: new Date(at).toISOString(), kind: job.kind, file: job.file, label: job.label, sources: [], live: true, details: job.details});
    }
    for (const events of Object.values(out)) events.sort((a,b) => a.at.localeCompare(b.at));
    return out;
  }, [data, jobs, showTraining, showSql, showExports]);
  const undated = jobs.filter(job => (!job.enabled || !job.next) && (job.kind === "sql" ? showSql : showExports));

  const nowMs = data ? Date.parse(data.now) : Date.now();
  const todayKey = sgtKey(nowMs);
  const untilKey = data ? sgtKey(Date.parse(data.until)) : "";
  const liveName = data?.files.find((f) => f.live)?.name ?? "";
  const savedOthers = (data?.available ?? []).filter((n) => n !== liveName);

  const first = monthStart(nowMs, offset);
  const firstLocal = new Date(first + SGT_MS);
  const heading = `${MONTHS[firstLocal.getUTCMonth()]} ${firstLocal.getUTCFullYear()}`;
  const gridStart = mondayOf(first);
  const days = Array.from({ length: 42 }, (_, i) => gridStart + i * DAY_MS);
  const inMonth = (ms: number) => new Date(ms + SGT_MS).getUTCMonth() === firstLocal.getUTCMonth();
  const lastKey = sgtKey(days[days.length - 1]);
  const pastWindow = untilKey !== "" && lastKey > untilKey;

  const isPicked = (name: string) => picks === "*" || picks.includes(name);

  return (
    <div className="w-full min-w-0">
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <CalendarDays size={20} className="text-accent" strokeWidth={2} />
        Track2 Scheduled
        <span className="text-sm font-normal text-ink-muted">- prototype</span>
      </h1>
      <p className="mb-5 max-w-3xl text-sm text-ink-secondary">
        Scheduled jobs and their settings files, in Singapore time. Display toggles do not enable or disable jobs.
        For Multi-Auto Training, only the <strong>loaded</strong> settings file actually runs; any other
        saved Multi-Auto Training (_AT) file picked below is shown as what it would run if it were
        loaded. Hover over a calendar entry to see its purpose, settings file or record, and schedule details. Edit training settings on{" "}
        <Link to="/track2/multi-auto-training" className="underline decoration-dotted underline-offset-2 hover:text-ink">
          Multi-Auto Training
        </Link>. SQL and export schedules run independently and are managed on their respective pages.
      </p>
      <fieldset className="mb-5 flex flex-wrap gap-4 text-sm">
        <legend className="mb-2 font-medium">Display schedules</legend>
        <label><input type="checkbox" checked={showTraining} onChange={e => setShowTraining(e.target.checked)} /> Training and pulls</label>
        <label><input type="checkbox" checked={showSql} onChange={e => setShowSql(e.target.checked)} /> SQL pulls</label>
        <label><input type="checkbox" checked={showExports} onChange={e => setShowExports(e.target.checked)} /> Model exports</label>
      </fieldset>
      {otherErrors.map(message => <p key={message} role="alert" className="text-xs text-critical">{message}</p>)}

      {error && (
        <p className="mb-4 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">{error}</p>
      )}
      {!data && !error && (
        <p className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" /> Reading the schedule...
        </p>
      )}

      {data && (
        <>
          <div hidden={!showTraining}><Card title="Training files shown in calendar">
            <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
              <span className="text-ink-muted">Show beside the loaded file:</span>
              <label className="flex items-center gap-1 font-medium text-ink">
                <input type="checkbox" checked={picks === "*"}
                       onChange={(e) => choose(e.target.checked ? "*" : [])} />
                All saved training files
              </label>
              {savedOthers.map((name) => (
                <label key={name} className="flex items-center gap-1 text-ink">
                  <input type="checkbox" checked={isPicked(name)}
                         disabled={picks === "*"}
                         onChange={(e) => {
                           const cur = picks === "*" ? savedOthers : picks;
                           choose(e.target.checked ? [...cur, name] : cur.filter((n) => n !== name));
                         }} />
                  <span className="tabular">{name}</span>
                </label>
              ))}
              {savedOthers.length === 0 && <span className="text-ink-muted">no other saved files</span>}
              {loading && <Loader2 size={13} className="animate-spin text-ink-muted" />}
            </div>

          </Card></div>

          <div className="mt-5">
            <Card title="Calendar" collapsible={false}>
              {undated.length > 0 && <div className="mb-3 rounded border border-hairline p-2 text-xs">
                <p className="mb-2 font-medium">Without a scheduled date</p>
                <div className="flex flex-wrap gap-2">{undated.map(job => <div key={job.id} tabIndex={0} className="group relative rounded border border-hairline p-2">
                  {job.label} — {job.enabled ? "Event-driven / no next date" : "Off"}
                  <div role="tooltip" className="pointer-events-none absolute left-0 top-full z-50 hidden w-72 rounded border border-hairline bg-surface p-2 shadow-lg group-hover:block group-focus-within:block"><p>{job.file}</p><p>{job.details}</p></div>
                </div>)}</div>
              </div>}

              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <span className="text-sm font-medium">Monthly schedules (SGT)</span>
                <div className="flex items-center gap-1 text-sm">
                  <button type="button" aria-label="Previous" onClick={() => setOffset((o) => o - 1)}
                          disabled={offset <= 0}
                          className="rounded-sm p-1 text-ink-secondary hover:bg-black/[0.04] disabled:opacity-30">
                    <ChevronLeft size={16} />
                  </button>
                  <span className="min-w-[9rem] text-center font-medium text-ink">{heading}</span>
                  <button type="button" aria-label="Next" onClick={() => setOffset((o) => o + 1)}
                          disabled={pastWindow}
                          className="rounded-sm p-1 text-ink-secondary hover:bg-black/[0.04] disabled:opacity-30">
                    <ChevronRight size={16} />
                  </button>
                  {offset !== 0 && (
                    <button type="button" onClick={() => setOffset(0)}
                            className="ml-1 text-xs text-ink-secondary underline">
                      Today
                    </button>
                  )}
                </div>
                <div className="flex items-center gap-2 text-[11px] text-ink-secondary">
                  <span className="rounded-sm bg-accent-subtle px-1.5 py-0.5 text-ink">Training run</span>
                  <span className="rounded-sm bg-black/[0.04] px-1.5 py-0.5">Scheduled pull</span>
                  <span>SQL pulls</span><span>Model exports</span>
                </div>
              </div>

              <div className="grid grid-cols-7 gap-1">
                {WEEKDAYS.map((d) => (
                  <div key={d} className="px-1 text-[11px] font-medium text-ink-muted">{d}</div>
                ))}
                {days.map((ms) => {
                  const key = sgtKey(ms);
                  const { day, month } = dayLabel(ms);
                  const evs = byDay[key] ?? [];
                  const past = key < todayKey;
                  const beyond = untilKey !== "" && key > untilKey;
                  return (
                    <div key={key} data-date={key}
                         className={`flex min-w-0 flex-col gap-0.5 rounded-sm border p-1 ${
                           "min-h-[5.5rem]"} ${
                           key === todayKey ? "border-accent" : "border-hairline"} ${
                           past || beyond || !inMonth(ms) ? "bg-black/[0.02] opacity-60" : "bg-page"}`}>
                      <span className={`text-[11px] ${key === todayKey ? "font-semibold text-accent" : "text-ink-muted"}`}>
                        {day}{day === 1 ? ` ${month}` : ""}
                      </span>
                      {evs.map((ev, i) => (
                        <EventLine key={`${ev.file}-${ev.at}-${ev.kind}-${i}`} ev={ev}
                                   color={colorOf(ev.file)} compact />
                      ))}
                    </div>
                  );
                })}
              </div>
              <p className="mt-2 text-[11px] text-ink-muted">
                Shown up to {formatDayTime(data.until)}. Past days are greyed:
                this page shows what will run, and History shows what did.
                Slots from a file that is not loaded are faded.
              </p>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
