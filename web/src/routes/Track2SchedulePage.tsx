import { AlertCircle, CalendarClock, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import {
  ApiError,
  getTrack2Schedule,
  setTrack2Schedule,
  trainTrack2Now,
  type Track2ScheduleResponse,
} from "../lib/api";

const WEEKDAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];

export function Track2SchedulePage() {
  const [data, setData] = useState<Track2ScheduleResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [mode, setMode] = useState<"manual" | "daily" | "weekly" | "monthly">("manual");
  const [hour, setHour] = useState(9);
  const [day, setDay] = useState("SUN");
  const [dom, setDom] = useState(1);
  const [saving, setSaving] = useState(false);
  const [triggering, setTriggering] = useState(false);

  function load() {
    getTrack2Schedule()
      .then((res) => {
        setData(res);
        setMode(res.schedule.enabled ? "weekly" : "manual");
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load schedule."));
  }

  useEffect(load, []);

  async function handleSave() {
    setSaving(true);
    setError(null);
    try {
      const res = await setTrack2Schedule({
        mode,
        daily_hour: mode === "daily" ? hour : undefined,
        weekly_day: mode === "weekly" ? day : undefined,
        weekly_hour: mode === "weekly" ? hour : undefined,
        monthly_day: mode === "monthly" ? dom : undefined,
        monthly_hour: mode === "monthly" ? hour : undefined,
      });
      setMessage(res.message);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Update failed.");
    } finally {
      setSaving(false);
    }
  }

  async function handleTrainNow() {
    setTriggering(true);
    setError(null);
    try {
      const res = await trainTrack2Now();
      setMessage(res.message);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Trigger failed.");
    } finally {
      setTriggering(false);
    }
  }


  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <CalendarClock size={20} className="text-accent" strokeWidth={2} />
        Track2 Schedule
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        The single global retrain schedule, plus a manual trigger.
      </p>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {message && (
        <div className="mb-6 rounded-sm border border-hairline bg-accent-subtle px-4 py-3 text-sm text-ink">
          {message}
        </div>
      )}

      {!data && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading...
        </div>
      )}

      {data && (
        <div className="flex flex-col gap-5">
          <Card title="Latest training run">
            {data.training_progress ? (
              <div className="flex items-center gap-2 text-sm">
                <Badge
                  tone={data.training_progress.status === "Executing" ? "pending" : data.training_progress.status === "Succeeded" ? "confirmed" : "neutral"}
                  label={data.training_progress.status}
                />
                <span className="tabular text-xs text-ink-muted">
                  {data.training_progress.start_time ?? ""}
                </span>
              </div>
            ) : (
              <p className="text-sm text-ink-muted">No pipeline execution has ever run.</p>
            )}
            <button
              type="button"
              onClick={handleTrainNow}
              disabled={triggering}
              className="mt-3 flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              {triggering && <Loader2 size={14} className="animate-spin" />}
              Train Now
            </button>
          </Card>

          <Card title="Global retrain schedule" caption={data.schedule.enabled ? data.schedule.expression : "Manual trigger only"}>
            <div className="flex flex-col gap-3">
              <div className="flex gap-2">
                {(["manual", "daily", "weekly", "monthly"] as const).map((m) => (
                  <button
                    key={m}
                    type="button"
                    onClick={() => setMode(m)}
                    className={`rounded-sm border px-3 py-1.5 text-xs font-medium capitalize ${
                      mode === m ? "border-accent bg-accent-subtle text-ink" : "border-black/15 text-ink-secondary"
                    }`}
                  >
                    {m}
                  </button>
                ))}
              </div>

              {mode !== "manual" && (
                <div className="flex flex-wrap items-end gap-3">
                  {mode === "weekly" && (
                    <label className="block">
                      <span className="mb-1 block text-xs text-ink-muted">Day</span>
                      <select className="rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={day} onChange={(e) => setDay(e.target.value)}>
                        {WEEKDAYS.map((d) => (
                          <option key={d} value={d}>{d}</option>
                        ))}
                      </select>
                    </label>
                  )}
                  {mode === "monthly" && (
                    <label className="block">
                      <span className="mb-1 block text-xs text-ink-muted">Day of month</span>
                      <input
                        type="number"
                        min={1}
                        max={28}
                        className="tabular w-20 rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                        value={dom}
                        onChange={(e) => setDom(Number(e.target.value))}
                      />
                    </label>
                  )}
                  <label className="block">
                    <span className="mb-1 block text-xs text-ink-muted">Hour (SGT)</span>
                    <input
                      type="number"
                      min={0}
                      max={23}
                      className="tabular w-20 rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                      value={hour}
                      onChange={(e) => setHour(Number(e.target.value))}
                    />
                  </label>
                </div>
              )}

              <button
                type="button"
                onClick={handleSave}
                disabled={saving}
                className="flex w-fit items-center gap-1.5 rounded-md border border-black/15 px-3 py-2 text-sm font-medium text-ink-secondary disabled:opacity-40"
              >
                {saving && <Loader2 size={14} className="animate-spin" />}
                Save schedule
              </button>
            </div>
          </Card>

        </div>
      )}
    </div>
  );
}
