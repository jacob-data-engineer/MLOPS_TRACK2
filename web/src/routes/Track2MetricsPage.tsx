import { AlertCircle, BarChart3, Loader2 } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Card } from "../components/Card";
import { Track2ExportSettingsBar } from "../components/Track2ExportSettingsBar";
import { Track2MetricsExportCard } from "../components/Track2MetricsExportCard";
import { Track2ModelPicker, modelLabel, rowKey } from "../components/Track2ModelPicker";
import {
  ApiError,
  getTrack2Metrics,
  getTrack2TrainingAudit,
  TRACK2_COMPARE_COLORS,
  type Track2ExportSettings,
  type Track2MetricsRow,
  type Track2TrainingAudit,
} from "../lib/api";

/** Four, because four is the most the palette can separate. See
 * TRACK2_COMPARE_COLORS: a fifth slot failed the normal-vision floor
 * however it was stepped, and the skill's rule for that is to cut series
 * rather than invent a hue. Four is also the point past which direct
 * labels stop fitting under the bars. */
const MAX_COMPARE = 4;

/** What the export settings start as, and what New goes back to. Held
 * here so "blank" means one thing wherever it is pressed.
 *
 * Empty for the three dropdowns rather than a guess: the card fills them
 * from whatever is actually registered on its first render, and a default
 * of "helmet" would be a Class that may not exist. `local` because a
 * download writes nothing anybody else can see -- the safe reading of
 * silence. */
const EXPORT_DEFAULTS: Track2ExportSettings = {
  registry: "",
  generation: "",
  size: "",
  group_name: "",
  criterion: "mAP50",
  destination: "local",
  bucket: "",
  prefix: "",
  // The policy you get by not choosing one, and the only one that
  // removes nothing. A default that deleted would be a retention
  // decision taken on somebody's behalf.
  retention: "keep_all",
};

type MetricKey = "mAP50" | "mAP50-95" | "precision" | "recall" | "f1" | "accuracy";

/** `on` is whether the checkbox starts ticked.
 *
 * **F1 was not on this list at all**, and `accuracy` was -- ticked. On
 * this estate every one of the fourteen measured versions carries an F1
 * and NOT ONE carries an accuracy: the current evaluator reports F1, and
 * `accuracy` is the legacy evaluator's equivalent, kept only so older
 * versions still read. So the page opened on an empty Accuracy cluster
 * for every version, with its own comparability note firing to explain
 * the blank bars, while the metric that had data on all fourteen could
 * not be charted anywhere -- and Export Control would happily rank by it.
 *
 * `accuracy` stays selectable rather than being removed, because a
 * version old enough to have one is still comparable. It just does not
 * start ticked. */
const METRICS: { key: MetricKey; label: string; on: boolean; hint?: string }[] = [
  { key: "mAP50", label: "mAP50", on: true },
  { key: "mAP50-95", label: "mAP50-95", on: true },
  { key: "precision", label: "Precision", on: true },
  { key: "recall", label: "Recall", on: true },
  { key: "f1", label: "F1", on: true },
  { key: "accuracy", label: "Accuracy", on: false,
    hint: "the legacy evaluator's number -- blank on anything trained recently" },
];
const DEFAULT_METRIC_KEYS = new Set(METRICS.filter((m) => m.on).map((m) => m.key));



/** One cluster of side-by-side vertical bars per metric -- each bar in a
 * cluster is one compared version, so each metric's own cluster lets you
 * read every version's value for it at a glance. Only the metrics ticked
 * above get a cluster, and it redraws as soon as one is ticked: the
 * "Compare" button this used to wait for went when the table of tick
 * boxes became a dropdown and a Select this, which is already deliberate
 * enough not to need confirming. Comparison is never restricted by
 * approval status -- an Approved version stays comparable against
 * whatever is Rejected or still Pending. */
/** A FIXED 0-1 axis, not one scaled to the tallest selected bar.
 *
 * mAP50, mAP50-95, precision, recall and accuracy are all bounded [0,1],
 * so 1.0 is the real top of the scale. Normalising each cluster to its
 * own maximum -- which this did -- drew 0.60 and 0.55 as a full bar and
 * a nine-tenths bar, a 9% visual gap for a 5-point difference, while
 * 0.995 against 0.000 looked exactly the same way. Every cluster now
 * shares one scale, so bar height means the same thing in all of them
 * and across every comparison anybody runs.
 */
const GRID_LINES = [0, 25, 50, 75, 100];


function Track2MetricsBarChart({ rows, metrics }: { rows: Track2MetricsRow[]; metrics: MetricKey[] }) {
  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {rows.map((row, i) => (
          <span key={rowKey(row)} className="flex items-center gap-1.5 text-xs text-ink-secondary">
            <span
              className="h-2.5 w-2.5 shrink-0 rounded-sm"
              style={{ background: TRACK2_COMPARE_COLORS[i] }}
            />
            {modelLabel(row)} ({row.status})
          </span>
        ))}
      </div>
      {metrics.map((key) => {
        const label = METRICS.find((m) => m.key === key)?.label ?? key;
        const values = rows.map((r) => r.metrics?.[key] ?? null);
        return (
          <div key={key} className="rounded-sm border border-hairline bg-page p-4">
            <p className="mb-3 text-sm font-semibold text-ink">{label}</p>
            {/* A GUTTER for the scale, rather than labels hung off the
                left of the plot. `-left-7` put them outside the element
                they were positioned against, which is outside the card
                -- so every number sat on the border, half cut off.
                The gridlines start at 2.5rem and the labels live in the
                space before it: `right-full` ends a label exactly where
                its line begins, so nothing can overlap whatever the
                number's width turns out to be. */}
            <div className="relative h-44">
              {GRID_LINES.map((pct) => (
                <div
                  key={pct}
                  className="absolute left-10 right-0 border-t border-hairline/70"
                  style={{ bottom: `${pct}%` }}
                >
                  {/* The scale, stated. A fixed axis nobody can see is
                      the same as no axis at all. */}
                  <span className="tabular absolute right-full top-[-0.55rem] mr-1.5 text-[10px] text-ink-muted">
                    {(pct / 100).toFixed(2)}
                  </span>
                </div>
              ))}
              <div className="relative ml-10 flex h-full items-end gap-4 px-1">
                {rows.map((row, i) => {
                  const value = values[i];
                  // Against 1.0, the metric's own ceiling.
                  const heightPct = value != null ? Math.max(1, Math.min(1, value) * 100) : 0;
                  return (
                    <div key={rowKey(row)} className="flex h-full flex-1 flex-col items-center justify-end gap-1">
                      <span className="tabular text-xs font-medium text-ink">
                        {value != null ? value.toFixed(3) : "--"}
                      </span>
                      <div className="flex w-full flex-1 items-end px-[1px]">
                        <div
                          className="w-full rounded-t-md transition-[height] duration-300"
                          style={{
                            height: `${heightPct}%`,
                            background: value != null ? TRACK2_COMPARE_COLORS[i] : "transparent",
                          }}
                          title={`${modelLabel(row)}: ${value != null ? value.toFixed(3) : "no data"}`}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
            {/* The same 2.5rem gutter as the plot above, or the names
                stop sitting under the bars they name. */}
            <div className="mt-2 ml-10 flex gap-4 px-1">
              {rows.map((row) => (
                <span key={rowKey(row)} className="tabular flex-1 truncate text-center text-[11px] text-ink-muted">
                  {modelLabel(row)}
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
  const [error, setError] = useState<string | null>(null);
  // An ARRAY, not a Set: the colour a version carries is its position in
  // the selection, and a Set has no position. Picking v2 then v9 must
  // keep v2 blue when v9 is unticked -- colour follows the entity, never
  // its rank.
  // The training audit for each registered version, keyed on its package
  // ARN. Fetched here rather than inside the picker because `load()`
  // already refreshes this page and one lookup serves every version the
  // picker can show.
  const [auditsByArn, setAuditsByArn] = useState<Record<string, Track2TrainingAudit>>({});
  // The export settings live HERE, not in the card, because the toolbar
  // pinned at the top of the page edits the same values. Two copies
  // would need keeping in step, and the first time they drifted Save
  // would write something nobody was looking at.
  const [exportSettings, setExportSettings] =
    useState<Track2ExportSettings>(EXPORT_DEFAULTS);
  const [selectedVersions, setSelectedVersions] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<Set<MetricKey>>(new Set(DEFAULT_METRIC_KEYS));
  // When the rows were read. A training run that finishes while this page
  // is open changes nothing on it, and without a stated "as of" there is
  // no way to tell a page from five minutes ago from one from five hours
  // ago -- which on a page about which model is best is the difference
  // between a decision and a guess.
  const [loadedAt, setLoadedAt] = useState<Date | null>(null);
  const [reloading, setReloading] = useState(false);

  // No staging state. A "Compare" button earned its place when versions
  // were ticked in a table of fourteen rows -- the chart would have
  // redrawn on every stray tick. Choosing a version is now a dropdown
  // plus "Select this", which is already deliberate, so a second
  // confirmation was one press between a decision and its answer.
  const compareRows = useMemo(() => {
    const byKey = new Map((rows ?? []).map((r) => [rowKey(r), r]));
    // In SELECTION order, so the swatch beside a version in the picker
    // is the colour it wears here.
    return selectedVersions.map((k) => byKey.get(k)).filter(Boolean) as Track2MetricsRow[];
  }, [rows, selectedVersions]);
  const compareMetrics = useMemo(
    () => METRICS.map((m) => m.key).filter((k) => selectedMetrics.has(k)),
    [selectedMetrics],
  );

  function load() {
    // Which run produced each version. The picker shows it under the
    // version it belongs to, so "what were the numbers" and "what made
    // them" are one answer rather than two cards apart.
    getTrack2TrainingAudit(200)
      .then((res) => {
        const byArn: Record<string, Track2TrainingAudit> = {};
        for (const audit of res.audits) {
          if (audit.model?.model_package_arn) byArn[audit.model.model_package_arn] = audit;
        }
        setAuditsByArn(byArn);
      })
      .catch(() => undefined);
    setReloading(true);
    setError(null);
    getTrack2Metrics()
      .then((res) => { setRows(res.rows); setLoadedAt(new Date()); })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load metrics."))
      .finally(() => setReloading(false));
  }

  useEffect(load, []);


  function toggleMetric(key: MetricKey) {
    setSelectedMetrics((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  function toggleVersion(key: string) {
    setSelectedVersions((current) => current.includes(key)
      ? current.filter((k) => k !== key)
      : current.length >= MAX_COMPARE ? current : [...current, key]);
  }

  function handleClear() {
    // The comparison only. This also reset the metric checkboxes, which
    // is not what the button next to "Comparing" says it does -- having
    // unticked three metrics to look at one, Clear put them all back.
    setSelectedVersions([]);
  }

  /** Why the selected versions may not be comparable.
   *
   * Two numbers on one axis imply one question answered twice. These are
   * the cases where they are not: a different class list means a
   * different task, and validation counts an order of magnitude apart
   * mean one number is an estimate and the other is noise. Both are
   * invisible in the bars themselves, which is exactly why they belong
   * above them. */
  const comparability = (() => {
    const byKey = new Map((rows ?? []).map((r) => [rowKey(r), r]));
    const picked = selectedVersions.map((k) => byKey.get(k)).filter(Boolean) as Track2MetricsRow[];
    const notes: string[] = [];

    // Said at ONE version, not two: a chart of empty bars with nothing
    // explaining them is the worst answer here, and it is a real
    // selection -- v1 of both helmet groups has no metrics at all.
    const unmeasured = picked.filter((r) => r.metrics == null);
    if (unmeasured.length === picked.length && picked.length > 0) {
      return [
        picked.length === 1
          ? "This version has no recorded metrics, so there is nothing to "
            + "chart. Usually an imported checkpoint, or a run that failed "
            + "before it evaluated -- not a model that scored zero."
          : "None of these versions has recorded metrics, so there is nothing "
            + "to chart. Usually imported checkpoints, or runs that failed "
            + "before they evaluated -- not models that scored zero.",
      ];
    }
    if (unmeasured.length) {
      notes.push(
        `${unmeasured.map((r) => `v${r.version}`).join(", ")} `
        + `${unmeasured.length === 1 ? "has" : "have"} no recorded metrics and `
        + "will draw as empty bars -- not as zero.",
      );
    }

    if (picked.length < 2) return notes;
    const taxonomies = new Set(picked.map((r) => (r.metrics?.classes ?? []).join(" | ")));
    if (taxonomies.size > 1) {
      notes.push(
        "These versions do not detect the same classes, so their metrics "
        + "answer different questions. mAP over one class and mAP over four "
        + "are not on the same scale.",
      );
    }
    const counts = picked
      .map((r) => r.metrics?.dataset?.validation_count)
      .filter((n): n is number => typeof n === "number");
    if (counts.length) {
      const low = Math.min(...counts), high = Math.max(...counts);
      if (low < 50) {
        notes.push(
          `One version was measured on ${low} validation image${low === 1 ? "" : "s"}. `
          + "Below about fifty, a single picture moves mAP by whole percentage "
          + "points -- that number is noise, not a score.",
        );
      } else if (high >= low * 3) {
        notes.push(
          `Validation sets differ widely (${low} to ${high} images), so these `
          + "metrics carry very different confidence.",
        );
      }
    }
    return notes;
  })();


  // Groups the "All versions" table with a header row between each model
  // size -- rows already arrive sorted size-first from the API, this just
  // makes that grouping visible instead of implicit in the sort order.
  // What is ACTUALLY registered, rather than the one generation this page
  // used to assert. Sorted so the order does not shuffle as rows arrive.
  const generationsInUse =
    Array.from(new Set((rows ?? []).map((r) => r.generation).filter(Boolean)))
      .sort()
      .map((g) => `YOLO${g}`)
      .join(", ") || "none registered yet";

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <BarChart3 size={20} className="text-accent" strokeWidth={2} />
        Track2 Review &amp; Metrics
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Model generations in use:{" "}
        <span className="font-semibold text-ink">{generationsInUse}</span>. Pick
        up to {MAX_COMPARE} registered versions and compare their real registry
        metrics side by side -- no fabricated numbers, and no filtering by
        approval state, so an Approved version stays comparable against
        whatever is Rejected or still Pending. The validation count beside each
        version is part of the comparison: a score measured on five images is
        not a better score than one measured on two hundred.
      </p>

      {/* When this was read, and the way to read it again. A training run
          that finishes while the page is open changes nothing on it, and
          there was no refresh anywhere -- the only way to pick up a new
          version was a browser reload, which also loses the comparison
          you had built. */}
      <div className="mb-4 flex flex-wrap items-center gap-3 text-xs">
        <button
          type="button"
          onClick={load}
          disabled={reloading}
          className="rounded-md border border-black/15 px-2.5 py-1 font-medium text-ink-secondary disabled:opacity-40"
        >
          {reloading ? "Reading the registry..." : "Refresh"}
        </button>
        <span className="tabular text-ink-muted">
          {loadedAt
            ? `As of ${loadedAt.toLocaleTimeString()} -- ${rows?.length ?? 0} version${rows?.length === 1 ? "" : "s"} across every registry`
            : "Not loaded yet"}
        </span>
      </div>

      {/* Pinned under the title, the same place Multi-Auto Training keeps
          its own: a settings file is about the page, so it belongs where
          the page's name is.

          STICKY, so it stays reachable on a long page -- "which file am I
          in, and save it" is the question you ask at the bottom, after
          changing something, and it was three screens away. <main> is the
          real scroll container (Layout gives it `overflow-y-auto` and a
          fixed `md:h-svh`), so `top-0` is the top of the content area
          rather than of a page that is itself scrolling away.

          `-mx-4 px-4` pulls the background out to the full content width,
          so what scrolls underneath is covered rather than appearing
          beside it. z-20 keeps it under the stale-build banner, which is
          z-40 and has to stay the thing you see first. */}
      <div className="sticky top-0 z-20 -mx-4 mb-5 border-b border-hairline bg-surface px-4 py-2 shadow-sm md:-mx-8 md:px-8">
        <Track2ExportSettingsBar
          settings={exportSettings}
          onChange={setExportSettings}
          defaults={EXPORT_DEFAULTS}
        />
      </div>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {!rows && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading metrics...
        </div>
      )}

      {rows && (
        <Track2ModelPicker
          rows={rows}
          selected={selectedVersions}
          onToggle={toggleVersion}
          onClear={handleClear}
          max={MAX_COMPARE}
          colors={TRACK2_COMPARE_COLORS}
          audits={auditsByArn}
          onAuditDeleted={load}
        />
      )}

      {rows && (
        <div className="mb-5">
          <Card title={"Comparison"}>
            <div className="mb-4 flex flex-wrap gap-x-4 gap-y-1.5 border-b border-hairline pb-4">
              {METRICS.map(({ key, label, hint }) => (
                <label
                  key={key}
                  title={hint}
                  className="flex items-center gap-1.5 text-xs text-ink-secondary"
                >
                  <input type="checkbox" checked={selectedMetrics.has(key)} onChange={() => toggleMetric(key)} />
                  {label}
                  {hint && <span className="text-ink-muted">(legacy)</span>}
                </label>
              ))}
            </div>
            {comparability.length > 0 && (
              <div className="mb-4 flex flex-col gap-1.5">
                {comparability.map((note) => (
                  <p
                    key={note}
                    className="flex items-start gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical"
                  >
                    <AlertCircle size={14} className="mt-0.5 shrink-0" />
                    <span>{note}</span>
                  </p>
                ))}
              </div>
            )}
            {compareRows.length === 0 ? (
              <p className="text-sm text-ink-muted">
                Pick a version above and press <em>Select this</em>. Up to{" "}
                {MAX_COMPARE} can be compared at once.
              </p>
            ) : compareMetrics.length === 0 ? (
              <p className="text-sm text-ink-muted">Choose at least one metric.</p>
            ) : (
              <Track2MetricsBarChart rows={compareRows} metrics={compareMetrics} />
            )}
          </Card>
        </div>
      )}


      {rows && (
        <Track2MetricsExportCard
          rows={rows}
          settings={exportSettings}
          onChange={setExportSettings}
        />
      )}

      {/* The "All versions" card was here: the per-registry table,
          with approve / reject / delete, the CSV and .pt exports, the
          audit expansion, and the tick boxes that fed the Comparison
          card above. Removed on request while this page is rebuilt.

          NOTE: nothing else selects versions, so the Comparison card
          can no longer be populated. It is left standing rather than
          quietly removed too -- that is a second decision, not a
          consequence to take silently. */}

    </div>
  );
}
