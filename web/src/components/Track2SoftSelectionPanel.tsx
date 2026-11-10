import { useEffect, useState } from "react";
import { post, listen } from "../lib/bus";
import { Card } from "./Card";
import { apiGet, apiPost, ApiError, getTrack2PullControl, saveTrack2PullConfig, saveTrack2Retention } from "../lib/api";

type Selection = { enabled: boolean; id: string; available: number; selected: number; train: number; val: number; excluded?: number; weeks: string[]; excluded_weeks?: string[]; retention?: { enabled?: boolean; weekly_rate?: number; full_weeks?: number; curve?: "exponential" | "linear"; use_floor?: boolean; floor?: number } };
export function Track2SoftSelectionPanel() {
  const [status, setStatus] = useState<Selection | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  async function read() { setStatus(await apiGet<Selection>("/track2/pull-control/soft-selection")); }
  useEffect(() => {
    void read().catch(e => setError(e instanceof ApiError ? e.detail : "Could not read database selection."));
    const unlisten = listen("data-sources-changed", () => { if (!busy) void read().catch(() => undefined); });
    const timer = setInterval(() => { if (!busy) void read().catch(() => undefined); }, 15000);
    return () => { clearInterval(timer); unlisten(); };
  }, [busy]);
  async function act(action: string) {
    setBusy(true); setError(""); setMessage("");
    try {
      setStatus(await apiPost<Selection>("/track2/pull-control/soft-selection", { action }));
      setMessage(action === "refresh" ? "Selection refreshed. Source files were not moved or deleted." : "Selection settings saved. Save the page file to keep this selection with it.");
      window.dispatchEvent(new Event("soft-selection-changed"));
      post("data-sources-changed", { from: "database selection" });
    } catch (e) { setError(e instanceof ApiError ? e.detail : "Selection operation failed."); }
    finally { setBusy(false); }
  }
  async function policy(retention?: Selection["retention"], excluded?: string[]) {
    setBusy(true); setError("");
    try {
      const current = (await getTrack2PullControl()).config;
      if (retention) await saveTrack2Retention(retention);
      else await saveTrack2PullConfig({ auto_training: { ...current.auto_training, excluded: excluded ?? [] } });
      await read(); post("data-sources-changed", { from: "database selection" });
    } catch (e) { setError(e instanceof ApiError ? e.detail : "Could not save the selection policy."); }
    finally { setBusy(false); }
  }
  const button = "rounded border border-hairline px-3 py-1.5 text-xs disabled:opacity-40";
  return <Card title="Database data selection">
    <p className="text-sm text-ink-secondary">Each saved configuration owns its selection in SQL. Images are preserved once in shared storage; label versions and membership are recorded in the database. Retention changes what trains, without deleting shared files.</p>
    <label className="mt-3 flex items-center gap-2 text-sm"><input type="checkbox" disabled={busy || !status}
      checked={status?.enabled ?? false} onChange={e => void act(e.target.checked ? "enable" : "disable")} />Use database selection instead of copied week folders</label>
    {status?.enabled && <>
      <p className="mt-2 text-xs text-ink-muted">Uses the sources selected on Assemble, Pull Control and Golden Set. Pair image and label folders in the same order. Pulls refresh membership; training freezes a snapshot and prepares YOLO files. Save as creates an independent selection.</p>
      <div className="mt-3 flex gap-2">
        <button className={button} disabled={busy} onClick={() => void act("refresh")}>{busy ? "Working..." : "Refresh selection"}</button>
        <button className={button} disabled={busy} onClick={() => void act("new")}>New independent selection</button>
      </div>
      <p className="mt-3 text-xs">Available: {status.available} · Selected: {status.selected} · Train: {status.train} · Validation: {status.val} · Retention / excluded weeks: {status.excluded ?? 0}</p>
      <p className="mt-1 text-xs text-ink-muted">Selection: {status.id}</p>
      <fieldset disabled={busy} className="mt-3 space-y-2 text-xs">
        <legend className="font-medium">Weeks included in training</legend>
        {status.weeks.filter(w => w !== "fixed").map(week => <label key={week} className="mr-3 inline-flex gap-2">
          <input type="checkbox" checked={!status.excluded_weeks?.includes(week)} onChange={e => void policy(undefined,
            e.target.checked ? (status.excluded_weeks ?? []).filter(w => w !== week) : [...(status.excluded_weeks ?? []), week])} />{week}
        </label>)}
        {status.weeks.includes("fixed") && <p>Manually selected and golden-set sources stay included.</p>}
        <label className="flex items-center gap-2"><input type="checkbox" checked={status.retention?.enabled ?? false}
          onChange={e => void policy({ enabled: e.target.checked })} />Apply retention to collected training images</label>
        {status.retention?.enabled && <div className="flex flex-wrap gap-3">
          <label>Reduction per older collection (%) <input className="w-16 rounded border border-hairline bg-page p-1" type="number" min="0" max="100" step="0.1"
            key={`rate-${status.retention.weekly_rate}`} defaultValue={(status.retention.weekly_rate ?? 0.01) * 100}
            onBlur={e => { const v=Number(e.target.value); if (Number.isFinite(v) && v>=0 && v<=100) void policy({weekly_rate:v/100}); }} /></label>
          <label>Newest collections kept in full <input className="w-16 rounded border border-hairline bg-page p-1" type="number" min="1"
            key={`full-${status.retention.full_weeks}`} defaultValue={status.retention.full_weeks ?? 1}
            onBlur={e => { const v=Number(e.target.value); if (Number.isInteger(v) && v>=1) void policy({full_weeks:v}); }} /></label>
          <label>Curve <select className="rounded border border-hairline bg-page p-1" value={status.retention.curve ?? "exponential"}
            onChange={e => void policy({curve:e.target.value as "exponential" | "linear"})}><option value="exponential">Exponential</option><option value="linear">Linear</option></select></label>
          <label className="flex items-center gap-2"><input type="checkbox" checked={status.retention.use_floor ?? true}
            onChange={e => void policy({use_floor:e.target.checked})} />Minimum retained share</label>
          {(status.retention.use_floor ?? true) && <label>Minimum (%) <input className="w-16 rounded border border-hairline bg-page p-1" type="number" min="0" max="100"
            key={`floor-${status.retention.floor}`} defaultValue={(status.retention.floor ?? 0.1) * 100}
            onBlur={e => { const v=Number(e.target.value); if (Number.isFinite(v) && v>=0 && v<=100) void policy({floor:v/100}); }} /></label>}
        </div>}
        <p className="text-ink-muted">Retention never reduces validation images. Unticking a whole week excludes both splits. Save the page file to keep these settings.</p>
      </fieldset>
    </>}
    {message && <p role="status" className="mt-2 text-xs">{message}</p>}
    {error && <p role="alert" className="mt-2 text-xs text-critical">{error}</p>}
  </Card>;
}
