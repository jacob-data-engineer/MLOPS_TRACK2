import { useEffect, useState } from "react";
import { apiGet, getModelExportSchedules, getTrack2AutoExport } from "../lib/api";
import { formatSgt } from "../lib/format";
import { Card } from "./Card";

type Job = { name: string; file: string; status: string; next: string; timing: string; output: string };
type Sensor = { config: { enabled: boolean; every_days: number; at: string }; next_at: string | null; running: boolean; settings_file?: string };
export function Track2OtherSchedules({ kind }: { kind: "sql" | "exports" }) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    let disposed = false;
    async function load() {
      const found: Job[] = [], errors: string[] = [];
      if (kind === "sql") {
        try {
          const s = await apiGet<Sensor>("/track2/sql/sensor");
          found.push({ name: "SQL image and label pull", file: s.settings_file || "sql-sensor/state.json (preset name not recorded)", status: s.running ? "Running" : s.config.enabled ? "Enabled" : "Off", next: s.config.enabled ? formatSgt(s.next_at, "Not scheduled") : "Not scheduled", timing: `Every ${s.config.every_days} day(s) at ${s.config.at} SGT`, output: "Images and labels; file names are determined by the SQL selection" });
        } catch { errors.push("Could not read SQL pull schedule."); }
      } else {
        const results = await Promise.allSettled([getModelExportSchedules(), getTrack2AutoExport()]);
        const plans = results[0];
        if (plans.status === "fulfilled") for (const s of plans.value.schedules) found.push({
          name: s.scope === "comparison" ? "Export best compared model" : `Export best model: ${s.model_name || "not selected"}`,
          file: `Database schedule: ${s.scope} (no settings file)`, status: s.enabled ? "Enabled" : "Off",
          next: s.enabled ? formatSgt(s.next_run_at, "Not scheduled") : "Not scheduled",
          timing: s.time_sgt ? `Every ${s.every_days || 1} day(s) at ${s.time_sgt} SGT` : "Not configured",
          output: s.last_result?.written_to || (s.bucket ? `s3://${s.bucket}/${s.prefix || ""} (filename assigned when exported)` : "No output recorded")
        }); else errors.push("Could not read model export schedules.");
        const watcher = results[1];
        if (watcher.status === "fulfilled") { const s = watcher.value.settings; found.push({ name: `New-registration export watcher: ${s.group_name || "not selected"}`, file: "Database record: metrics_auto_export (no settings file)", status: s.enabled ? "Enabled" : "Off", next: "On new model registration", timing: "Event-driven", output: s.last_result?.written_to || "No output recorded" }); }
        else errors.push("Could not read the registration export watcher.");
      }
      if (!disposed) { setJobs(found); setError(errors.join(" ")); setLoaded(true); }
    }
    void load(); const timer = window.setInterval(load, 60000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [kind]);
  return <div className="mb-5"><Card title={kind === "sql" ? "SQL pull schedules" : "Model export schedules"} collapsible={false}>
    {!loaded && <p className="text-xs">Reading schedules...</p>}
    {error && <p role="alert" className="text-xs text-critical">{error}</p>}
    <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr>{["Job", "Settings file / record", "Status", "Next run (SGT)", "Repeat", "Last output file / destination"].map(h => <th className="p-2" key={h}>{h}</th>)}</tr></thead>
      <tbody>{jobs.map(job => <tr key={job.name} className="border-t border-hairline align-top">{[job.name, job.file, job.status, job.next, job.timing, job.output].map((value, i) => <td className="p-2 break-words" key={i}>{value}</td>)}</tr>)}</tbody>
    </table></div>
    {loaded && !error && !jobs.length && <p className="text-xs">No schedules recorded.</p>}
  </Card></div>;
}
