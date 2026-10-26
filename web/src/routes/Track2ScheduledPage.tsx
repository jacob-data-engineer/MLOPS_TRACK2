import { CalendarDays, ChevronLeft, ChevronRight, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Card } from "../components/Card";
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

const WINDOW_DAYS = 92;
const SGT_MS = 8 * 3600 * 1000;
const DAY_MS = 24 * 3600 * 1000;
const WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
/** One colour per file, in the order the files are listed. */
const PALETTE = ["#b45309", "#2563eb", "#059669", "#9333ea", "#dc2626", "#0891b2", "#ca8a04", "#db2777"];
/** Per-viewer convenience only: which saved files were last picked. */
const PICK_KEY = "track2-scheduled-files";

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
    if (!raw) return [];
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
function EventPopup({ ev, color }: { ev: Track2ScheduledEvent; color: string }) {
  const train = ev.kind === "train";
  return (
    <div className="pointer-events-none absolute left-0 top-full z-50 mt-1 hidden w-64 rounded-sm border border-hairline bg-surface p-2 text-xs text-ink shadow-lg group-hover:block">
      <p className="flex items-center gap-1.5 font-medium">
        <span className="inline-block h-2 w-2 rounded-full" style={{ background: color }} />
        <span className="tabular break-all">{ev.file}</span>
      </p>
      <p className={ev.live ? "text-accent" : "text-ink-muted"}>
        {ev.live ? "Loaded -- this runs" : "Not loaded -- would run if this file were loaded"}
      </p>
      <p className="mt-1 tabular">{formatDayTime(ev.at)}</p>
      <p>{ev.label}</p>
      {train ? (
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

function EventLine({ ev, color, compact }: { ev: Track2ScheduledEvent; color: string; compact?: boolean }) {
  const train = ev.kind === "train";
  return (
    <div className="group relative">
      <div
        className={`flex items-center gap-1 rounded-sm border-l-[3px] px-1 py-0.5 text-[11px] leading-4 ${
          train ? "bg-accent-subtle text-ink" : "bg-black/[0.04] text-ink-secondary"} ${
          ev.live ? "" : "opacity-75"}`}
        style={{ borderLeftColor: color }}
      >
        <span className="tabular">{sgtClock(ev.at)}</span>
        <span className="shrink-0">{train ? "Train" : "Pull"}</span>
        <span className="truncate text-ink-muted">{compact ? "" : "· "}{ev.file}</span>
      </div>
      <EventPopup ev={ev} color={color} />
    </div>
  );
}

export function Track2ScheduledPage() {
  const [data, setData] = useState<Track2Scheduled | null>(null);
  const [error, setError] = useState("");
  const [view, setView] = useState<"week" | "month">("week");
  const [offset, setOffset] = useState(0);
  const [picks, setPicks] = useState<string[] | "*">(readPicks);
  const [loading, setLoading] = useState(false);

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
    const out: Record<string, Track2ScheduledEvent[]> = {};
    for (const ev of data?.events ?? []) (out[sgtKey(Date.parse(ev.at))] ??= []).push(ev);
    return out;
  }, [data]);

  const nowMs = data ? Date.parse(data.now) : Date.now();
  const todayKey = sgtKey(nowMs);
  const untilKey = data ? sgtKey(Date.parse(data.until)) : "";
  const liveName = data?.files.find((f) => f.live)?.name ?? "";
  const savedOthers = (data?.available ?? []).filter((n) => n !== liveName);

  let days: number[] = [];
  let heading = "";
  let inMonth = (_ms: number) => true;
  if (view === "week") {
    const start = mondayOf(nowMs) + offset * 7 * DAY_MS;
    days = Array.from({ length: 7 }, (_, i) => start + i * DAY_MS);
    const a = dayLabel(days[0]);
    const b = dayLabel(days[6]);
    heading = `${a.day} ${a.month} – ${b.day} ${b.month}`;
  } else {
    const first = monthStart(nowMs, offset);
    const firstLocal = new Date(first + SGT_MS);
    heading = `${MONTHS[firstLocal.getUTCMonth()]} ${firstLocal.getUTCFullYear()}`;
    const gridStart = mondayOf(first);
    days = Array.from({ length: 42 }, (_, i) => gridStart + i * DAY_MS);
    const month = firstLocal.getUTCMonth();
    inMonth = (ms) => new Date(ms + SGT_MS).getUTCMonth() === month;
  }
  const lastKey = sgtKey(days[days.length - 1]);
  const pastWindow = untilKey !== "" && lastKey > untilKey;

  const isPicked = (name: string) => picks === "*" || picks.includes(name);

  return (
    <div className="mx-auto max-w-6xl">
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <CalendarDays size={20} className="text-accent" strokeWidth={2} />
        Track2 Scheduled
        <span className="text-sm font-normal text-ink-muted">- prototype</span>
      </h1>
      <p className="mb-5 max-w-3xl text-sm text-ink-secondary">
        Every pull and training run the clocks will start, in Singapore time.
        Only the <strong>loaded</strong> settings file actually runs; any other
        saved file picked below is shown as what it would run if it were
        loaded. Hover over a slot to see what that run trains on. Nothing
        here changes a setting; they are all on{" "}
        <Link to="/track2/multi-auto-training" className="underline decoration-dotted underline-offset-2 hover:text-ink">
          Multi-Auto Training
        </Link>.
      </p>

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
          <Card title="Settings files" collapsible={false}>
            <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs">
              <span className="text-ink-muted">Show beside the loaded file:</span>
              <label className="flex items-center gap-1 font-medium text-ink">
                <input type="checkbox" checked={picks === "*"}
                       onChange={(e) => choose(e.target.checked ? "*" : [])} />
                All saved files
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

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="text-ink-muted">
                  <tr>
                    <th className="py-1 pr-3 font-normal">File</th>
                    <th className="py-1 pr-3 font-normal">Next training run</th>
                    <th className="py-1 pr-3 font-normal">Kind of run (data it trains on)</th>
                  </tr>
                </thead>
                <tbody>
                  {data.files.map((f) => (
                    <tr key={f.name} className="border-t border-hairline align-top">
                      <td className="py-1.5 pr-3">
                        <span className="flex items-center gap-1.5">
                          <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
                                style={{ background: colorOf(f.name) }} />
                          <span className="tabular text-ink">{f.name}</span>
                          <span className={`rounded-sm px-1 text-[10px] font-medium ${
                            f.live ? "bg-accent text-white" : "border border-black/15 text-ink-muted"}`}>
                            {f.live ? "loaded" : "not loaded"}
                          </span>
                        </span>
                        {f.error && <p className="text-critical">{f.error}</p>}
                        {f.notes.map((n) => <p key={n} className="text-ink-muted">{n}</p>)}
                      </td>
                      <td className="tabular py-1.5 pr-3 text-ink">
                        {f.next ? formatDayTime(f.next.at) : <span className="text-ink-muted">none scheduled</span>}
                      </td>
                      <td className="py-1.5 pr-3 text-ink">
                        {f.error ? "" : f.sources.length ? f.sources.join(" + ") : (
                          <span className="text-critical">nothing ticked</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          <div className="mt-5">
            <Card title="Calendar" collapsible={false}>
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div className="flex rounded-md border border-black/15 text-xs">
                  {(["week", "month"] as const).map((v) => (
                    <button key={v} type="button"
                      onClick={() => { setView(v); setOffset(0); }}
                      className={`px-3 py-1.5 ${view === v ? "bg-accent text-white" : "text-ink-secondary hover:bg-black/[0.04]"}`}>
                      {v === "week" ? "Weekly" : "Monthly"}
                    </button>
                  ))}
                </div>
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
                  <span>colour = settings file</span>
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
                    <div key={key}
                         className={`flex min-w-0 flex-col gap-0.5 rounded-sm border p-1 ${
                           view === "week" ? "min-h-[9rem]" : "min-h-[5.5rem]"} ${
                           key === todayKey ? "border-accent" : "border-hairline"} ${
                           past || beyond || !inMonth(ms) ? "bg-black/[0.02] opacity-60" : "bg-page"}`}>
                      <span className={`text-[11px] ${key === todayKey ? "font-semibold text-accent" : "text-ink-muted"}`}>
                        {day}{view === "week" || day === 1 ? ` ${month}` : ""}
                      </span>
                      {evs.map((ev, i) => (
                        <EventLine key={`${ev.file}-${ev.at}-${ev.kind}-${i}`} ev={ev}
                                   color={colorOf(ev.file)} compact={view === "month"} />
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
