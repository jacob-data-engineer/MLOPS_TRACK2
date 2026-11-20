import { useEffect, useState } from "react";
import { apiGet, getModelExportSchedules, getTrack2AutoExport } from "../lib/api";

export type CalendarJob = {
  id: string; kind: "sql" | "export"; label: string; file: string;
  enabled: boolean; next?: string | null; everyDays?: number; details: string;
};
type Sensor = {config: {enabled: boolean; every_days: number; at: string}; next_at: string | null; running: boolean; settings_file?: string};

export function useOtherSchedules() {
  const [jobs, setJobs] = useState<CalendarJob[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  useEffect(() => {
    let disposed = false;
    async function load() {
      const found: CalendarJob[] = [], errors: string[] = [];
      const results = await Promise.allSettled([apiGet<Sensor>("/track2/sql/sensor"), getModelExportSchedules(), getTrack2AutoExport()]);
      const sql = results[0];
      if (sql.status === "fulfilled") {
        const s = sql.value;
        found.push({id: "sql", kind: "sql", label: "SQL image and label pull", file: s.settings_file || "sql-sensor/state.json", enabled: s.config.enabled,
          next: s.next_at, everyDays: s.config.every_days,
          details: `Pulls images and labels selected from SQL. Every ${s.config.every_days} day(s) at ${s.config.at} SGT.${s.running ? " Currently running." : ""}`});
      } else errors.push("Could not read SQL pull schedules.");
      const plans = results[1];
      if (plans.status === "fulfilled") for (const s of plans.value.schedules) found.push({
        id: s.scope, kind: "export", label: s.scope === "comparison" ? "Export best compared model" : `Export best model: ${s.model_name || "not selected"}`,
        file: `Database schedule: ${s.scope}`, enabled: s.enabled, next: s.next_run_at, everyDays: s.every_days,
        details: `Selects the best model by ${s.criterion || "unconfigured criterion"}${s.scope === "comparison" ? ` from ${s.model_package_arns?.length ?? 0} saved comparison models` : ` for ${s.model_name || "an unselected model name"}`}. Every ${s.every_days || 1} day(s) at ${s.time_sgt || "unconfigured time"} SGT. Destination: ${s.bucket ? `s3://${s.bucket}/${s.prefix || ""}` : "not configured"}. Format: ${s.package_options?.format || "checkpoint .pt"}. Last output: ${s.last_result?.written_to || "none recorded"}.`
      }); else errors.push("Could not read model export schedules.");
      const watcher = results[2];
      if (watcher.status === "fulfilled") { const s = watcher.value.settings;
        found.push({id: "registration", kind: "export", label: `New-registration export watcher: ${s.group_name || "not selected"}`, file: "Database record: metrics_auto_export", enabled: s.enabled,
          details: `Runs on new model registration, not at a fixed date or time. Last output: ${s.last_result?.written_to || "none recorded"}.`});
      } else errors.push("Could not read the registration export watcher.");
      if (!disposed) {setJobs(found); setErrors(errors);}
    }
    void load(); const timer = window.setInterval(load, 60000);
    return () => {disposed = true; window.clearInterval(timer);};
  }, []);
  return {jobs, errors};
}
