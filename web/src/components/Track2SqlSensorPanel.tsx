import { useEffect, useState } from "react";
import { apiGet, apiPost, ApiError } from "../lib/api";
import type { SqlPageSettings, SqlSchedule } from "./Track2SqlSettingsBar";

type Config = { enabled: boolean; every_days: number; at: string; from_date: string; settings: SqlPageSettings };
type Run = {
  id: string; status: string; started_at: string; finished_at?: string; error?: string;
  found?: number; eligible?: number; newly_eligible?: number; pending?: number; excluded?: number;
  exclusion_reasons?: { count: number; reason: string }[];
  summary?: { left_out_incomplete?: number; copied: number; skipped: number; failed: number; with_label: number; errors?: string[] };
  progress?: { completed: number; total: number; phase: string };
};
type Status = { config: Config; next_at: string | null; running: boolean; saved_at?: string; last_run?: Run; history: Run[] };
const stamp = (value?: string | null) => value ? new Date(value).toLocaleString("en-SG", { timeZone: "Asia/Singapore" }) : "—";

export function Track2SqlSensorPanel({ schedule, onChange }: { schedule: SqlSchedule; onChange: (value: SqlSchedule) => void }) {
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
    const timer = setInterval(() => void refresh(), 10000);
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
        <p>Destination: s3://{target.pull_bucket}/{target.pull_folder}/ · {target.train_percent}% train · {target.with_labels ? "With labels" : "Images only"}</p>
        {target.with_labels && <p>Class IDs: {Object.entries(target.class_ids).map(([name, id]) => `${name} = ${id}`).join(", ")}</p>}
      </>}
      </details>
      {status?.running && <p role="status" className="font-medium">Checking for new frames and missing labels... This can take a few minutes.</p>}
      {last && <div className="rounded border border-hairline bg-page p-3" role="status">
        {last.status === "unchanged" && <p className="font-semibold">Check complete  -  no new files to pull. All eligible images and labels are already in the saved destination.</p>}
        <p>Last check: {stamp(last.started_at)} SGT · {last.status}</p>
        {last.found !== undefined && <p>{last.found} frames found · {last.eligible} eligible · {last.newly_eligible} newly seen eligible · {last.pending} needing images or labels · {last.excluded} excluded</p>}
        {last.summary && <p>{last.summary.copied} images copied · {last.summary.skipped} images already present · {last.summary.with_label} with labels · {last.summary.failed} failed</p>}
        {last.exclusion_reasons?.map((item, index) => <p key={index} className="mt-1 text-ink-secondary">
          Not copied: {item.count} image(s) - {item.reason}
        </p>)}
        {!last.exclusion_reasons && (last.summary?.left_out_incomplete ?? 0) > 0 && <p className="mt-1 text-ink-secondary">
          Not copied: {last.summary?.left_out_incomplete} image(s) - no exportable label boxes. Add or recover object bounding boxes before exporting with labels.
        </p>}
        {!!last.summary?.errors?.length && <p className="text-critical">{last.summary.errors.join("; ")}</p>}
        {status?.running && last.progress && <progress aria-label="Sensor pull progress" className="w-full" value={last.progress.completed} max={Math.max(1, last.progress.total)} />}
        {last.error && <p className="text-critical">{last.error}</p>}
      </div>}
      <p className="text-ink-muted">Runs on EC2 with the browser closed while the app is running. After downtime, one catch-up check runs. The selected database must be running.</p>
      {status && status.history.length > 0 && <details><summary className="cursor-pointer">Recent checks ({status.history.length})</summary>
        <ul className="mt-1 space-y-1">{status.history.map(run => <li key={run.id}>{stamp(run.started_at)} · {run.status}{run.summary ? ` · ${run.summary.copied} copied` : ""}{run.error ? ` · ${run.error}` : ""}</li>)}</ul>
      </details>}
      {message && <p role="status">{message}</p>}
      {error && <p role="alert" className="text-critical">{error}</p>}
    </div>
  </section>;
}
