import { useEffect, useState } from "react";
import { apiGet, apiPost, ApiError } from "../lib/api";
import type { SqlPageSettings, SqlSchedule } from "./Track2SqlSettingsBar";

type Config = { enabled: boolean; every_days: number; at: string; from_date: string; settings: SqlPageSettings };
type Run = {
  id: string; status: string; started_at: string; finished_at?: string; error?: string;
  found?: number; eligible?: number; newly_eligible?: number; pending?: number; excluded?: number;
  exclusion_reasons?: { count: number; reason: string }[];
  summary?: { left_out_incomplete?: number; copied: number; skipped: number; failed: number; with_label: number; errors?: string[]; by_week?: boolean; weeks?: Record<string, number> };
  training?: { queued: boolean; reason: string };
  progress?: { completed: number; total: number; phase: string };
};
type Status = { config: Config; next_at: string | null; running: boolean; saved_at?: string; last_run?: Run; history: Run[] };
const stamp = (value?: string | null) => value ? new Date(value).toLocaleString("en-SG", { timeZone: "Asia/Singapore" }) : "—";


function PullTiming({ run }: { run: Run }) {
  const [now, setNow] = useState(Date.now());
  const active = run.status === "running";
  useEffect(() => {
    if (!active) return;
    setNow(Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active, run.id]);
  const start = Date.parse(run.started_at);
  const end = run.finished_at ? Date.parse(run.finished_at) : active ? now : NaN;
  const seconds = Math.max(0, Math.floor((end - start) / 1000));
  const duration = Number.isFinite(seconds) ? `${Math.floor(seconds / 60)} min ${seconds % 60} sec` : "Not recorded";
  return <p className="mt-1">Started: {stamp(run.started_at)} SGT
    {run.finished_at && <> / Finished: {stamp(run.finished_at)} SGT</>}
    {" / "}{active ? "Elapsed" : "Duration"}: {duration}
  </p>;
}

export function Track2SqlSensorPanel({ schedule, onChange }: {
  schedule: SqlSchedule; onChange: (value: SqlSchedule) => void;
}) {
  const [status, setStatus] = useState<Status | null>(null);
  const { enabled, every_days: everyDays, at, from_date: fromDate } = schedule;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      try {
        const got = await apiGet<Status>("/track2/sql/sensor");
        if (cancelled) return;
        setStatus(got);
        if (!got.running) setMessage("");

      } catch (err) {
        if (!cancelled) setError(err instanceof ApiError ? err.detail : "Could not read sensor status.");
      }
    }
    void refresh();
    const saved = () => void refresh();
    window.addEventListener("sql-settings-saved", saved);
    const timer = setInterval(() => void refresh(), 2000);
    return () => { cancelled = true; clearInterval(timer); window.removeEventListener("sql-settings-saved", saved); };
  }, []);
  const locked = busy || !status || status.running;
  async function action(endpoint: string, payload: unknown, text: string) {
    setBusy(true); setError(""); setMessage("");
    try {
      const got = await apiPost<Status>(endpoint, payload);
      setStatus(got); setMessage(got.running ? text : "");
    } catch (err) { setError(err instanceof ApiError ? err.detail : "Sensor operation failed."); }
    finally { setBusy(false); }
  }
  const last = status?.last_run;
  const finished = !!last && ["complete", "unchanged", "partial"].includes(last.status);
  const total = Math.max(0, last?.progress?.total ?? 0);
  const completed = Math.max(0, Math.min(total, last?.progress?.completed ?? 0));
  const percent = finished ? 100 : total > 0 ? Math.floor(completed / total * 100) : undefined;
  const progressLabel = finished
    ? (last?.status === "partial" ? "Finished with errors - 100% processed" : "Complete - 100%")
    : last?.status === "error" ? `Failed${percent !== undefined ? ` - ${percent}% processed` : ""}`
    : percent !== undefined ? `${last?.progress?.phase || "Processing"} - ${percent}%`
    : "Checking frames and destination - waiting for measured progress";
  const target = status?.config.settings;
  const button = "rounded border border-hairline px-3 py-1.5 text-xs disabled:opacity-40";
  return <section className="space-y-1" aria-label="New frames sensor">
    <h4 className="text-sm font-semibold">New frames sensor</h4>
    <p className="mt-1 text-xs text-ink-muted">Checks the database for eligible frames and automatically pulls missing images and labels using the saved sensor settings. No training job is started.</p>
    <fieldset disabled={locked} className="mt-3 flex flex-wrap items-end gap-4 text-xs">
      <label className="flex items-center gap-2"><input type="checkbox" checked={enabled} onChange={e => onChange({ ...schedule, enabled: e.target.checked })} />Enable scheduled pulls</label>
      <label>Every <input aria-label="Sensor interval in days" type="number" min={1} max={365} value={everyDays}
        onChange={e => onChange({ ...schedule, every_days: Number(e.target.value) })} className="mx-2 w-16 rounded border border-hairline bg-page px-2 py-1" />day(s)</label>
      <label>At (Singapore time)<input aria-label="Sensor time" type="time" value={at} onChange={e => onChange({ ...schedule, at: e.target.value })} className="ml-2 rounded border border-hairline bg-page px-2 py-1" /></label>
      <label>Start date (optional)<input type="date" aria-label="Sensor start date" value={fromDate} onChange={e => onChange({ ...schedule, from_date: e.target.value })} className="ml-2 rounded border border-hairline bg-page px-2 py-1" /></label>
    </fieldset>
    <p className="mt-2 text-xs text-ink-muted">Checks every {everyDays} day(s) at {at || "the selected time"} Singapore time. Saving after that time schedules the next occurrence. Use Save above to save the schedule and all page settings together. Load restores the file for editing; Save applies it to scheduled checks.</p>
    <div className="mt-3 flex flex-wrap gap-2">
      <button className={button} disabled={locked || !status?.saved_at}
        onClick={() => void action("/track2/sql/sensor/check", {}, "Check started using the saved sensor settings.")}>Check now</button>

    </div>
    <div className="mt-3 space-y-1 text-xs text-ink-secondary" aria-live="polite">
      <p>{status?.config.enabled ? "Scheduled" : "Schedule off"} · Next check: {stamp(status?.next_at)} SGT</p>
      <details><summary className="cursor-pointer">Saved destination</summary>
      {status?.saved_at && target && <>
        <p>Saved source: {target.database} / {target.preset_class} / {target.preset_scope}</p>
        <p>Destination: {target.by_week ? "Pull Control's week folders, by capture date" : `s3://${target.pull_bucket}/${target.pull_folder}/`} · {target.train_percent}% train · {target.with_labels ? "With labels" : "Images only"}{target.train_after && target.by_week ? " · trains after each scheduled pull" : ""}</p>
        {target.with_labels && <p>Class IDs: {Object.entries(target.class_ids).map(([name, id]) => `${name} = ${id}`).join(", ")}</p>}
      </>}
      </details>
      {status?.running && <p role="status" className="font-medium">Checking for new frames and missing labels... This can take a few minutes.</p>}
      {last && <div className="rounded border border-hairline bg-page p-3" role="status">
        <div className="my-2 space-y-1">
          <div className="flex items-center justify-between gap-3">
            <span>Check progress</span><span>{progressLabel}</span>
          </div>
          <div role="progressbar" aria-label="SQL check progress" aria-valuemin={0} aria-valuemax={100}
            aria-valuenow={percent} aria-valuetext={progressLabel}
            className="h-2 w-full overflow-hidden rounded bg-black/10">
            <div className="h-full rounded bg-green-600 transition-[width]" style={{ width: `${percent ?? 0}%` }} />
          </div>
          {!finished && total > 0 && <p>{completed.toLocaleString()} / {total.toLocaleString()} frames processed</p>}
        </div>
        {last.status === "unchanged" && <p className="font-semibold">Check complete  -  no new files to pull. All eligible images and labels are already in the saved destination.</p>}
        <p>Last check: {stamp(last.started_at)} SGT · {last.status}</p>
        <PullTiming run={last} />
        {last.found !== undefined && <p>{last.found} frames found · {last.eligible} eligible · {last.newly_eligible} newly seen eligible · {last.pending} needing images or labels · {last.excluded} excluded</p>}
        {last.summary && <p>{last.summary.copied} images copied · {last.summary.skipped} images already present · {last.summary.with_label} with labels · {last.summary.failed} failed</p>}
        {last.summary?.weeks && Object.keys(last.summary.weeks).length > 0 && <p>Filed into: {Object.entries(last.summary.weeks).map(([week, count]) => `${week} (${count})`).join(", ")}</p>}
        {last.training && <p className={last.training.queued ? "font-medium" : "text-ink-secondary"}>{last.training.queued ? "Training run queued." : `Training not started: ${last.training.reason}`}</p>}
        {last.exclusion_reasons?.map((item, index) => <p key={index} className="mt-1 text-ink-secondary">
          Not copied: {item.count} image(s) - {item.reason}
        </p>)}
        {!last.exclusion_reasons && (last.summary?.left_out_incomplete ?? 0) > 0 && <p className="mt-1 text-ink-secondary">
          Not copied: {last.summary?.left_out_incomplete} image(s) - no exportable label boxes. Add or recover object bounding boxes before exporting with labels.
        </p>}
        {!!last.summary?.errors?.length && <p className="text-critical">{last.summary.errors.join("; ")}</p>}
        {last.error && <p className="text-critical">{last.error}</p>}
      </div>}
      {status && status.history.length > 0 && <details><summary className="cursor-pointer">Recent checks ({status.history.length})</summary>
        <ul className="mt-1 space-y-1">{status.history.map(run => <li key={run.id}>{stamp(run.started_at)} · {run.status}{run.summary ? ` · ${run.summary.copied} copied` : ""}{run.error ? ` · ${run.error}` : ""}</li>)}</ul>
      </details>}
      {message && <p role="status">{message}</p>}
      {error && <p role="alert" className="text-critical">{error}</p>}
    </div>
  </section>;
}


export function SqlScheduledPullProgress() {
  const [status, setStatus] = useState<Status | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let disposed = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        const result = await apiGet<Status>("/track2/sql/sensor");
        if (!disposed) { setStatus(result); setError(false); }
      } catch { if (!disposed) setError(true); }
      finally { if (!disposed) timer = setTimeout(poll, 2000); }
    }
    void poll();
    return () => { disposed = true; clearTimeout(timer); };
  }, []);
  const run = status?.last_run;
  if (!run) return error ? <p role="alert">Could not load scheduled pull progress. Retrying...</p> : null;
  const finished = ["complete", "unchanged", "partial"].includes(run.status);
  const total = Math.max(0, run.progress?.total ?? 0);
  const done = Math.max(0, Math.min(total, run.progress?.completed ?? 0));
  const percent = finished ? 100 : total ? Math.floor(done / total * 100) : undefined;
  return <div className="mt-3 rounded border border-hairline p-3 text-xs" aria-live="polite">
    <div className="mb-2 flex justify-between gap-2"><strong>Scheduled / Check now pull progress</strong>
      <span>{run.status}{percent !== undefined ? ` - ${percent}% processed` : " - checking frames and destination"}</span></div>
    <div role="progressbar" aria-label="Scheduled SQL pull progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}
      className="h-2 overflow-hidden rounded bg-black/10"><div className="h-full rounded bg-green-600 transition-[width]" style={{width: `${percent ?? 0}%`}} /></div>
    <PullTiming run={run} />
    {total > 0 && <p>{finished ? total : done} of {total} frames processed</p>}
    {run.summary && <p>{run.summary.copied} copied / {run.summary.skipped} already present / {run.summary.failed} failed</p>}
    {run.error && <p className="text-critical">{run.error}</p>}
    {error && <p role="alert">Progress updates unavailable; retrying. Displayed progress may be stale.</p>}
  </div>;
}
