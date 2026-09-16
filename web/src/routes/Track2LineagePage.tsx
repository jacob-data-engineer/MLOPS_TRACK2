import { AlertCircle, GitBranch, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "../components/Card";
import { ApiError, getTrack2Lineage, type Track2Lineage } from "../lib/api";
import { formatTimestamp } from "../lib/format";

export function Track2LineagePage() {
  const [records, setRecords] = useState<Track2Lineage[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getTrack2Lineage()
      .then((res) => {
        if (!cancelled) setRecords(res.records);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof ApiError ? err.detail : "Couldn't load lineage.");
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <GitBranch size={20} className="text-accent" strokeWidth={2} />
        Track2 Lineage
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Every training run and baseline registration, newest first -- what
        dataset, base model, and hyperparameters produced each version.
      </p>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {!records && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading lineage...
        </div>
      )}

      {records && records.length === 0 && (
        <p className="text-sm text-ink-muted">No lineage records yet.</p>
      )}

      {records && records.length > 0 && (
        <div className="flex flex-col gap-3">
          {records.map((r) => (
            <Card key={r.id} title={r.record_type === "baseline" ? "Baseline registration" : "Training run"}>
              <dl className="grid grid-cols-1 gap-1.5 text-sm sm:grid-cols-2">
                <div className="flex gap-2">
                  <dt className="text-ink-secondary">Timestamp:</dt>
                  <dd className="tabular text-ink">
                    {r.timestamp_utc ? formatTimestamp(String(r.timestamp_utc)) : "--"}
                  </dd>
                </div>
                {typeof r.dataset === "object" && r.dataset !== null && (
                  <div className="flex gap-2 sm:col-span-2">
                    <dt className="shrink-0 text-ink-secondary">Dataset:</dt>
                    <dd className="tabular truncate text-ink">
                      {String((r.dataset as Record<string, unknown>).s3_uri ?? "")}
                    </dd>
                  </div>
                )}
                {typeof r.base_model === "object" && r.base_model !== null && (
                  <div className="flex gap-2 sm:col-span-2">
                    <dt className="shrink-0 text-ink-secondary">Base model:</dt>
                    <dd className="tabular truncate text-ink">
                      {String(
                        (r.base_model as Record<string, unknown>).model_package_arn ??
                          (r.base_model as Record<string, unknown>).s3_uri ??
                          "",
                      )}
                    </dd>
                  </div>
                )}
                {typeof r.hyperparameters === "object" && r.hyperparameters !== null && (
                  <div className="flex gap-2 sm:col-span-2">
                    <dt className="shrink-0 text-ink-secondary">Hyperparameters:</dt>
                    <dd className="tabular text-ink">
                      {Object.entries(r.hyperparameters as Record<string, unknown>)
                        .map(([k, v]) => `${k}=${v}`)
                        .join(", ")}
                    </dd>
                  </div>
                )}
                {typeof r.notes === "string" && r.notes && (
                  <div className="flex gap-2 sm:col-span-2">
                    <dt className="shrink-0 text-ink-secondary">Notes:</dt>
                    <dd className="text-ink">{r.notes}</dd>
                  </div>
                )}
              </dl>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
