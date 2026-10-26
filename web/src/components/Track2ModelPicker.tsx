import { useEffect, useMemo, useState } from "react";
import { isSeedVersion, statusLabel } from "../lib/slug";
import { Badge } from "./Badge";
import { Card } from "./Card";
import type { Track2MetricsRow, Track2TrainingAudit } from "../lib/api";
import { AuditRow } from "./Track2TrainingAuditPanel";

/** Choosing which trained models to compare.
 *
 * Two dropdowns and a button rather than a long table of tick boxes. The
 * table was honest but it scaled badly: fourteen versions of one Class
 * filled a screen, and picking four of them meant hunting four rows apart
 * in a list where thirteen of the numbers were identical. A group, then a
 * version, then the numbers for that one version, is the question
 * somebody actually asks.
 *
 * Versions are addressed by REGISTRY GROUP first, because the group is
 * what a version means: `yolo11s-helmet-detector v7` is the seventh model
 * of the helmet Class at size s, and v7 of another group is an unrelated
 * model that happens to share a number.
 *
 * The detail panel leads with the metrics and ends with the validation
 * count, which is there for a reason. On this estate today:
 *
 *   v2  mAP50 0.995  measured on 5 images
 *   v9  mAP50 0.000  measured on 149 images
 *
 * Ranked on mAP50 alone, v2 wins by a mile. It was measured on five
 * pictures. The count sits beside the score so that comparison is never
 * made by accident.
 */

export type MetricKey = "mAP50" | "mAP50-95" | "precision" | "recall" | "f1" | "accuracy";

export function rowKey(row: Track2MetricsRow): string {
  return `${row.group_name}/${row.version}`;
}

/** `YOLO26-S v3` -- the generation the row states, never a literal. */
export function modelLabel(row: Track2MetricsRow): string {
  if (!row.generation) return `${row.group_name} v${row.version}`;
  const size = row.size ? `-${row.size.toUpperCase()}` : "";
  return `YOLO${row.generation}${size} v${row.version}`;
}

/** "--" and "0.000" are different answers. A version that was never
 * evaluated is not one that scored zero -- and the usual reason a row has
 * no metrics, an imported checkpoint, is exactly what somebody would
 * otherwise read as a failed model. */
function metric(row: Track2MetricsRow | undefined, key: MetricKey): string {
  const value = row?.metrics?.[key];
  return value == null ? "--" : value.toFixed(3);
}

function validationCount(row: Track2MetricsRow | undefined): number | null {
  const n = row?.metrics?.dataset?.validation_count;
  return typeof n === "number" ? n : null;
}

function statusTone(status: string): "confirmed" | "rejected" | "pending" {
  if (status === "Approved") return "confirmed";
  if (status === "Rejected") return "rejected";
  return "pending";
}

/** A seed is Pending BECAUSE it is a seed -- one fact, and this is the
 * half that explains the other. Keyed on the description the server
 * wrote, never on the version number: a v1 produced by a training run is
 * a real model. */
function versionLabel(row: Track2MetricsRow): string {
  return isSeedVersion(row.description) ? "Seed" : statusLabel(row.status);
}

/** Below about fifty validation images a detection metric is noise: one
 * picture moves mAP by whole percentage points. */
const TOO_FEW = 50;

const SELECT_CLASS =
  "rounded-sm border border-black/15 bg-page px-2 py-1 text-xs text-ink";

export function Track2ModelPicker({
  rows,
  selected,
  onToggle,
  onClear,
  max,
  colors,
  audits,
  onAuditDeleted,
}: {
  rows: Track2MetricsRow[];
  selected: string[];
  onToggle: (key: string) => void;
  onClear: () => void;
  max: number;
  /** In selection order, so the swatch beside a chosen version here is
   * the colour it carries in the chart. Identity has to survive the trip
   * between the two cards or the legend is a second thing to memorise. */
  colors: string[];
  /** The run that produced each version, keyed on package ARN. Shown
   * under the version it belongs to: the metrics answer "how good is
   * it", the run answers "what made it", and separating those two put
   * the epochs, the dataset and the snapshot on a different card from
   * the number they explain. */
  audits?: Record<string, Track2TrainingAudit>;
  onAuditDeleted?: () => void;
}) {
  const [classFilter, setClassFilter] = useState("all");
  const [generationFilter, setGenerationFilter] = useState("all");
  const [sizeFilter, setSizeFilter] = useState("all");
  const [measuredOnly, setMeasuredOnly] = useState(false);
  const [group, setGroup] = useState("");
  const [version, setVersion] = useState("");

  const classes = useMemo(
    () => Array.from(new Set(rows.map((r) => r.registry))).sort(), [rows]);
  const generations = useMemo(
    () => Array.from(new Set(rows.map((r) => r.generation).filter(Boolean))).sort(), [rows]);
  const sizes = useMemo(
    () => Array.from(new Set(rows.map((r) => r.size).filter(Boolean))).sort(), [rows]);

  const visible = useMemo(() => rows.filter((r) =>
    (classFilter === "all" || r.registry === classFilter)
    && (generationFilter === "all" || r.generation === generationFilter)
    && (sizeFilter === "all" || r.size === sizeFilter)
    && (!measuredOnly || r.metrics != null)),
    [rows, classFilter, generationFilter, sizeFilter, measuredOnly]);

  const groups = useMemo(
    () => Array.from(new Set(visible.map((r) => r.group_name))).sort(), [visible]);

  // Newest first: the newest version is what somebody is usually
  // deciding about.
  const versions = useMemo(
    () => visible.filter((r) => r.group_name === group)
      .sort((a, b) => Number(b.version) - Number(a.version)),
    [visible, group]);

  // Keep the two dropdowns on something that exists. Narrowing a filter
  // can remove whatever was chosen, and a select holding a value absent
  // from its own options renders blank with no way to tell why.
  useEffect(() => {
    if (!groups.includes(group)) setGroup(groups[0] ?? "");
  }, [groups, group]);
  useEffect(() => {
    if (!versions.some((r) => r.version === version)) setVersion(versions[0]?.version ?? "");
  }, [versions, version]);

  const chosen = versions.find((r) => r.version === version);
  const chosenKey = chosen ? rowKey(chosen) : "";
  const already = chosenKey !== "" && selected.includes(chosenKey);
  const full = selected.length >= max;
  const count = validationCount(chosen);
  const audit = chosen ? audits?.[chosen.model_package_arn] : undefined;

  const byKey = useMemo(() => new Map(rows.map((r) => [rowKey(r), r])), [rows]);

  return (
    <Card title={`Models (${selected.length} of ${max} selected to compare)`}>
      {/* Filters in one row, above everything they narrow. */}
      <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-hairline pb-3 text-xs">
        <label className="flex items-center gap-1.5 text-ink-secondary">
          Model Name
          <select value={classFilter} onChange={(e) => setClassFilter(e.target.value)} className={SELECT_CLASS}>
            <option value="all">All</option>
            {classes.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-ink-secondary">
          Model type
          <select value={generationFilter} onChange={(e) => setGenerationFilter(e.target.value)} className={SELECT_CLASS}>
            <option value="all">All</option>
            {generations.map((g) => <option key={g} value={g}>YOLO{g}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-ink-secondary">
          Size
          <select value={sizeFilter} onChange={(e) => setSizeFilter(e.target.value)} className={SELECT_CLASS}>
            <option value="all">All</option>
            {sizes.map((s) => <option key={s} value={s}>{s.toUpperCase()}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-1.5 text-ink-secondary">
          <input type="checkbox" checked={measuredOnly} onChange={(e) => setMeasuredOnly(e.target.checked)} />
          Only versions with metrics
        </label>
      </div>

      {groups.length === 0 ? (
        <p className="text-sm text-ink-muted">No versions match these filters.</p>
      ) : (
        <>
          {/* Group, then version, then that version's numbers. */}
          <div className="flex flex-wrap items-end gap-x-4 gap-y-2">
            <label className="flex flex-col gap-1 text-xs text-ink-secondary">
              Registry group
              <select
                value={group}
                onChange={(e) => setGroup(e.target.value)}
                className={`${SELECT_CLASS} tabular min-w-[18rem] py-1.5`}
              >
                {groups.map((g) => {
                  const n = visible.filter((r) => r.group_name === g).length;
                  return (
                    <option key={g} value={g}>
                      {g} ({n} version{n === 1 ? "" : "s"})
                    </option>
                  );
                })}
              </select>
            </label>

            <label className="flex flex-col gap-1 text-xs text-ink-secondary">
              Version
              <select
                value={version}
                onChange={(e) => setVersion(e.target.value)}
                className={`${SELECT_CLASS} tabular min-w-[20rem] py-1.5`}
              >
                {versions.map((r) => {
                  const n = validationCount(r);
                  return (
                    <option key={r.version} value={r.version}>
                      {/* Enough to choose by without opening the panel:
                          the score, and what it was measured on. */}
                      v{r.version} &middot; {versionLabel(r)} &middot; mAP50 {metric(r, "mAP50")}
                      {n != null ? ` · ${n} val images` : " · not evaluated"}
                      {selected.includes(rowKey(r)) ? " · selected" : ""}
                    </option>
                  );
                })}
              </select>
            </label>

            {/* Adds only. Removing is the x on the chip in "Comparing"
                below, which is where the selection is listed -- one
                button that changed its own meaning depending on the
                dropdown above it was a second way to do a thing that
                already had an obvious one, and it read as the primary
                action while doing the opposite. */}
            <button
              type="button"
              onClick={() => chosenKey && !already && onToggle(chosenKey)}
              disabled={!chosenKey || already || full}
              title={already
                ? "Already being compared -- remove it below."
                : full ? `Already comparing ${max}. Remove one below first.` : undefined}
              className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
            >
              {already ? "Already selected" : "Select this"}
            </button>
          </div>

          {/* The chosen version's numbers, in full. */}
          {chosen && (
            <div className="mt-3 rounded-sm border border-hairline bg-page p-3">
              <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="tabular text-sm font-semibold text-ink">{modelLabel(chosen)}</span>
                <Badge tone={statusTone(chosen.status)} label={versionLabel(chosen)} />
                <span className="tabular text-xs text-ink-muted">{chosen.group_name}</span>
              </div>

              {chosen.metrics ? (
                <>
                  {/* F1 was missing here and `accuracy` was in its place.
                      Every measured version on this estate carries an F1
                      and not one carries an accuracy -- the current
                      evaluator reports F1, and `accuracy` is the legacy
                      evaluator's equivalent, kept only so older versions
                      still read. So this panel showed a column that was
                      "--" on everything and hid the one with data. */}
                  <div className="tabular grid grid-cols-2 gap-x-4 gap-y-2 sm:grid-cols-5">
                    {([
                      ["mAP50", "mAP50"],
                      ["mAP50-95", "mAP50-95"],
                      ["precision", "Precision"],
                      ["recall", "Recall"],
                      ["f1", "F1"],
                    ] as [MetricKey, string][]).map(([key, label]) => (
                      <span key={key} className="flex flex-col">
                        <span className="text-[11px] uppercase tracking-wide text-ink-muted">{label}</span>
                        <span className="text-sm font-medium text-ink">{metric(chosen, key)}</span>
                      </span>
                    ))}
                  </div>

                  <div className="mt-3 flex flex-wrap gap-x-6 gap-y-1 border-t border-hairline pt-2 text-xs">
                    <span className="text-ink-secondary">
                      Validation images{" "}
                      <span className={count != null && count < TOO_FEW
                        ? "font-semibold text-critical" : "font-medium text-ink"}>
                        {count ?? "--"}
                      </span>
                      {/* Never colour alone: the words carry it for
                          anyone who cannot see the red. */}
                      {count != null && count < TOO_FEW && (
                        <span className="ml-1 text-critical">
                          -- too few to compare against anything
                        </span>
                      )}
                    </span>
                    {/* Only when it is there: a blank column on every
                        modern version is noise, but an older version's
                        accuracy is a real number somebody may be
                        comparing against. */}
                    {chosen.metrics.accuracy != null && (
                      <span className="text-ink-secondary">
                        Accuracy (legacy){" "}
                        <span className="font-medium text-ink">{metric(chosen, "accuracy")}</span>
                      </span>
                    )}
                    <span className="text-ink-secondary">
                      Classes{" "}
                      <span className="text-ink">
                        {(chosen.metrics.classes ?? []).join(", ") || "--"}
                      </span>
                    </span>
                    {chosen.metrics.dataset?.dataset_version && (
                      <span className="tabular truncate text-ink-muted">
                        {chosen.metrics.dataset.dataset_version}
                      </span>
                    )}
                  </div>
                </>
              ) : (
                <p className="text-xs text-ink-muted">
                  No metrics recorded for this version. Usually an imported
                  checkpoint, or a run that failed before it evaluated -- not a
                  model that scored zero.
                </p>
              )}

              {/* The run behind the numbers: epochs, seed, the images it
                  trained on, the frozen snapshot, the base model it
                  continued from, and the exports. The History panel's own
                  card, reused rather than restated -- one description of
                  a run, so the two cannot drift. */}
              {audit ? (
                <div className="mt-3 border-t border-hairline pt-3">
                  <AuditRow audit={audit} onDeleted={() => onAuditDeleted?.()} />
                </div>
              ) : (
                <p className="mt-3 border-t border-hairline pt-3 text-xs text-ink-muted">
                  No training run recorded against this version. Versions
                  registered before run auditing, and imported checkpoints,
                  have metrics but no run to show.
                </p>
              )}
            </div>
          )}

          {/* What is being compared, and the way back out of it. With the
              tick boxes gone this is the only place a selection shows. */}
          {selected.length > 0 && (
            <div className="mt-3 flex flex-wrap items-center gap-2 border-t border-hairline pt-3">
              <span className="text-xs text-ink-muted">Comparing</span>
              {selected.map((key, i) => {
                const row = byKey.get(key);
                return (
                  <span
                    key={key}
                    className="flex items-center gap-1.5 rounded-sm border border-hairline bg-page px-2 py-1 text-xs text-ink"
                  >
                    <span
                      className="h-2.5 w-2.5 shrink-0 rounded-sm"
                      style={{ background: colors[i % colors.length] }}
                    />
                    <span className="tabular">{row ? modelLabel(row) : key}</span>
                    <button
                      type="button"
                      onClick={() => onToggle(key)}
                      aria-label={`Remove ${row ? modelLabel(row) : key}`}
                      className="text-ink-muted hover:text-critical"
                    >
                      &times;
                    </button>
                  </span>
                );
              })}
              <button
                type="button"
                onClick={onClear}
                className="rounded-md border border-black/15 px-2.5 py-1 text-xs font-medium text-ink-secondary"
              >
                Clear
              </button>
            </div>
          )}
        </>
      )}
    </Card>
  );
}
