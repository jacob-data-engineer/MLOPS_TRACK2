import { AlertCircle, BarChart3, Loader2, Trash2 } from "lucide-react";
import { useState, useEffect } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { StatTile } from "../components/StatTile";
import {
  approveTrack2Model,
  ApiError,
  deleteTrack2Model,
  getTrack2Metrics,
  getTrack2Review,
  rejectTrack2Model,
  TRACK2_COMPARE_COLORS,
  type Track2MetricsRow,
  type Track2ReviewResponse,
} from "../lib/api";

const MAX_COMPARE = 5;

type MetricKey = "mAP50" | "mAP50-95" | "precision" | "recall" | "accuracy";
const METRICS: { key: MetricKey; label: string }[] = [
  { key: "mAP50", label: "mAP50" },
  { key: "mAP50-95", label: "mAP50-95" },
  { key: "precision", label: "Precision" },
  { key: "recall", label: "Recall" },
  { key: "accuracy", label: "Accuracy" },
];
const ALL_METRIC_KEYS = new Set(METRICS.map((m) => m.key));

function rowKey(row: Track2MetricsRow): string {
  return `${row.group_name}/${row.version}`;
}

function approvalTone(status: string): "confirmed" | "rejected" | "pending" {
  if (status === "Approved") return "confirmed";
  if (status === "Rejected") return "rejected";
  return "pending";
}

/** One cluster of side-by-side vertical bars per metric -- each bar in a
 * cluster is one compared version, so each metric's own cluster lets you
 * read every version's value for it at a glance. Only the metrics chosen
 * via the checkboxes above get a cluster; only rebuilds when "Compare" is
 * clicked, not on every checkbox toggle. Comparison is never restricted by
 * approval status -- an Approved version stays comparable against
 * whatever's Rejected or still Pending. */
const GRID_LINES = [0, 25, 50, 75, 100];

function Track2MetricsBarChart({ rows, metrics }: { rows: Track2MetricsRow[]; metrics: MetricKey[] }) {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {rows.map((row, i) => (
          <span key={rowKey(row)} className="flex items-center gap-1.5 text-xs text-ink-secondary">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ background: TRACK2_COMPARE_COLORS[i % TRACK2_COMPARE_COLORS.length] }}
            />
            YOLO11-{row.size.toUpperCase()} v{row.version} ({row.status})
          </span>
        ))}
      </div>
      {metrics.map((key) => {
        const label = METRICS.find((m) => m.key === key)?.label ?? key;
        const values = rows.map((r) => r.metrics?.[key] ?? null);
        const max = Math.max(0.01, ...values.map((v) => v ?? 0));
        return (
          <div key={key} className="rounded-sm border border-hairline bg-page p-4">
            <p className="mb-3 text-sm font-semibold text-ink">{label}</p>
            <div className="relative h-44">
              {GRID_LINES.map((pct) => (
                <div
                  key={pct}
                  className="absolute inset-x-0 border-t border-hairline/70"
                  style={{ bottom: `${pct}%` }}
                />
              ))}
              <div className="relative flex h-full items-end gap-4 px-1">
                {rows.map((row, i) => {
                  const value = values[i];
                  const heightPct = value != null ? Math.max(2, (value / max) * 100) : 0;
                  return (
                    <div key={rowKey(row)} className="flex h-full flex-1 flex-col items-center justify-end gap-1">
                      <span className="tabular text-xs font-medium text-ink">
                        {value != null ? value.toFixed(3) : "--"}
                      </span>
                      <div className="flex w-full flex-1 items-end">
                        <div
                          className="w-full rounded-t-md transition-[height] duration-300"
                          style={{
                            height: `${heightPct}%`,
                            background: value != null ? TRACK2_COMPARE_COLORS[i % TRACK2_COMPARE_COLORS.length] : "transparent",
                          }}
                          title={`YOLO11-${row.size.toUpperCase()} v${row.version}: ${value != null ? value.toFixed(3) : "no data"}`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            <div className="mt-2 flex gap-4 px-1">
              {rows.map((row) => (
                <span key={rowKey(row)} className="tabular flex-1 truncate text-center text-[11px] text-ink-muted">
                  YOLO11-{row.size.toUpperCase()} v{row.version}
                </span>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function Track2MetricsPage() {
  const [rows, setRows] = useState<Track2MetricsRow[] | null>(null);
  const [reviewCounts, setReviewCounts] = useState<Track2ReviewResponse["counts"] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [busyArn, setBusyArn] = useState<string | null>(null);
  const [selectedVersions, setSelectedVersions] = useState<Set<string>>(new Set());
  const [selectedMetrics, setSelectedMetrics] = useState<Set<MetricKey>>(new Set(ALL_METRIC_KEYS));

  // Only set by "Compare" -- checking boxes builds up a pending selection
  // without immediately redrawing the chart underneath.
  const [compareRows, setCompareRows] = useState<Track2MetricsRow[]>([]);
  const [compareMetrics, setCompareMetrics] = useState<MetricKey[]>([]);

  function load() {
    getTrack2Metrics()
      .then((res) => setRows(res.rows))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load metrics."));
    getTrack2Review()
      .then((res) => setReviewCounts(res.counts))
      .catch(() => {});
  }

  useEffect(load, []);

  function toggleVersion(key: string) {
    setSelectedVersions((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else if (next.size < MAX_COMPARE) next.add(key);
      return next;
    });
  }

  function toggleMetric(key: MetricKey) {
    setSelectedMetrics((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function handleCompare() {
    setCompareRows((rows ?? []).filter((r) => selectedVersions.has(rowKey(r))));
    setCompareMetrics(METRICS.map((m) => m.key).filter((k) => selectedMetrics.has(k)));
  }

  function handleClear() {
    setSelectedVersions(new Set());
    setSelectedMetrics(new Set(ALL_METRIC_KEYS));
    setCompareRows([]);
    setCompareMetrics([]);
  }

  async function handleDecision(arn: string, decision: "approve" | "reject" | "delete") {
    setBusyArn(arn);
    setError(null);
    try {
      if (decision === "approve") await approveTrack2Model(arn);
      else if (decision === "reject") await rejectTrack2Model(arn);
      else {
        if (!window.confirm("Permanently delete this model package?")) return;
        await deleteTrack2Model(arn);
      }
      setMessage(`Model v${arn.split("/").pop()} ${decision === "delete" ? "deleted" : `marked ${decision}d`}`);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : `${decision} failed.`);
    } finally {
      setBusyArn(null);
    }
  }

  // Groups the "All versions" table with a header row between each model
  // size -- rows already arrive sorted size-first from the API, this just
  // makes that grouping visible instead of implicit in the sort order.
  const rowsBySize: { size: string; rows: Track2MetricsRow[] }[] = [];
  for (const row of rows ?? []) {
    const group = rowsBySize[rowsBySize.length - 1];
    if (group && group.size === row.size) group.rows.push(row);
    else rowsBySize.push({ size: row.size, rows: [row] });
  }

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <BarChart3 size={20} className="text-accent" strokeWidth={2} />
        Track2 Review &amp; Metrics
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Model generation in use: <span className="font-semibold text-ink">YOLO11</span>. Real
        registry metrics per version -- no fabricated numbers. Approve, reject,
        or delete a version, and compare metrics across any versions
        regardless of status -- Approved, Rejected, or still Pending.
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

      {!rows && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading metrics...
        </div>
      )}

      {reviewCounts && (
        <div className="mb-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <StatTile label="Review queue: pending" value={reviewCounts.pending} />
          <StatTile label="Review queue: approved" value={reviewCounts.approved} deltaTone="good" />
          <StatTile label="Review queue: rejected" value={reviewCounts.rejected} />
        </div>
      )}

      {rows && (
        <div className="mb-5">
          <Card title={`Comparison (${selectedVersions.size} of ${MAX_COMPARE} versions selected)`}>
            <div className="mb-4 flex flex-wrap gap-x-4 gap-y-1.5 border-b border-hairline pb-4">
              {METRICS.map(({ key, label }) => (
                <label key={key} className="flex items-center gap-1.5 text-xs text-ink-secondary">
                  <input type="checkbox" checked={selectedMetrics.has(key)} onChange={() => toggleMetric(key)} />
                  {label}
                </label>
              ))}
            </div>
            <div className="mb-4 flex gap-2">
              <button
                type="button"
                onClick={handleCompare}
                disabled={selectedVersions.size === 0 || selectedMetrics.size === 0}
                className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
              >
                Compare
              </button>
              <button
                type="button"
                onClick={handleClear}
                className="rounded-md border border-black/15 px-3 py-2 text-sm font-medium text-ink-secondary"
              >
                Clear
              </button>
            </div>
            {compareRows.length > 0 ? (
              <Track2MetricsBarChart rows={compareRows} metrics={compareMetrics} />
            ) : (
              <p className="text-sm text-ink-muted">
                Check versions below and metrics above, then click Compare.
              </p>
            )}
          </Card>
        </div>
      )}

      {rows && (
        <Card title="All versions">
          {rows.length === 0 ? (
            <p className="text-sm text-ink-muted">No registered versions yet.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-hairline text-xs text-ink-muted">
                    <th className="py-1.5 pr-2 font-medium"></th>
                    <th className="py-1.5 pr-4 font-medium">Version</th>
                    <th className="py-1.5 pr-4 font-medium">Status</th>
                    <th className="tabular py-1.5 pr-4 font-medium">mAP50</th>
                    <th className="tabular py-1.5 pr-4 font-medium">mAP50-95</th>
                    <th className="tabular py-1.5 pr-4 font-medium">Precision</th>
                    <th className="tabular py-1.5 pr-4 font-medium">Recall</th>
                    <th className="py-1.5 pr-4 font-medium">Notes</th>
                    <th className="py-1.5 font-medium">Actions</th>
                  </tr>
                </thead>
                {rowsBySize.map(({ size, rows: sizeRows }) => (
                  <tbody key={size}>
                    <tr>
                      <td colSpan={9} className="bg-page px-2 py-1.5 text-xs font-semibold uppercase tracking-wide text-ink-secondary">
                        YOLO11 &middot; Size {size}
                      </td>
                    </tr>
                    {sizeRows.map((row) => {
                      const key = rowKey(row);
                      const busy = busyArn === row.model_package_arn;
                      return (
                        <tr key={key} className="border-b border-hairline last:border-0">
                          <td className="py-1.5 pr-2">
                            <input
                              type="checkbox"
                              checked={selectedVersions.has(key)}
                              onChange={() => toggleVersion(key)}
                              disabled={!selectedVersions.has(key) && selectedVersions.size >= MAX_COMPARE}
                            />
                          </td>
                          <td className="tabular py-1.5 pr-4">{row.version}</td>
                          <td className="py-1.5 pr-4">
                            <Badge tone={approvalTone(row.status)} label={row.status} />
                          </td>
                          <td className="tabular py-1.5 pr-4">{row.metrics?.mAP50?.toFixed(3) ?? "--"}</td>
                          <td className="tabular py-1.5 pr-4">{row.metrics?.["mAP50-95"]?.toFixed(3) ?? "--"}</td>
                          <td className="tabular py-1.5 pr-4">{row.metrics?.precision?.toFixed(3) ?? "--"}</td>
                          <td className="tabular py-1.5 pr-4">{row.metrics?.recall?.toFixed(3) ?? "--"}</td>
                          <td className="max-w-[16rem] truncate py-1.5 pr-4 text-xs text-ink-secondary">{row.description || "--"}</td>
                          <td className="py-1.5">
                            <div className="flex gap-1.5">
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleDecision(row.model_package_arn, "approve")}
                                className="rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
                              >
                                Approve
                              </button>
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleDecision(row.model_package_arn, "reject")}
                                className="rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
                              >
                                Reject
                              </button>
                              <button
                                type="button"
                                disabled={busy}
                                onClick={() => handleDecision(row.model_package_arn, "delete")}
                                className="flex items-center gap-1 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
                              >
                                <Trash2 size={12} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                ))}
              </table>
            </div>
          )}
        </Card>
      )}
    </div>
  );
}
