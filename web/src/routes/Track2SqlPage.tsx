import { Track2SqlSettingsBar, SQL_DEFAULTS, type SqlPageSettings, type SqlSavedPreset } from "../components/Track2SqlSettingsBar";
import { useCardOpen, ensureCardState } from "../lib/cardState";
import {
  AlertCircle, Copy, Database, Download, Loader2, Play,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Card } from "../components/Card";
import { S3FolderField } from "../components/S3FolderField";
import { Track2ReadinessFlowPanel } from "../components/Track2ReadinessFlowPanel";
import {
  ApiError,
  apiGet,
  apiPost,
  getTrack2Buckets,
  getTrack2PullControl,
  getTrack2SqlClasses,
  getTrack2SqlDatabases,
  getTrack2SqlPreset,
  getTrack2SqlPreviews,
  pullTrack2Sql,
  runTrack2Sql,
  saveTrack2PullConfig,
  setTrack2SqlDatabaseRunning,
  type Track2SqlClass,
  type Track2SqlDatabase,
  type Track2SqlFrameLabels,
  type Track2SqlPresetScope,
  type Track2SqlPullResult,
  type Track2SqlPullProgress,
  type Track2SqlResult,
} from "../lib/api";

/** Thumbnails per screenful; "Show more" asks for the next one. */
const PREVIEW_PAGE = 60;

const SCOPE_LABEL: Record<Track2SqlPresetScope, string> = {
  resolved: "All human-resolved",
  ready: "Ready for training (helmet box present)",
  test: "Test-grade only (box drawn by a person)",
};

/** TRACK2 SQL DATABASE QUERY (2026-10-28).
 *
 * POINT AND SELECT (2026-10-29). Choose a database, a class and which of
 * its frames, and the page shows the frames as thumbnails -- no SQL to
 * type. Behind the drop-downs the server writes the review database's own
 * rule (every person in the frame answered by a human; VLM-only and
 * pre-filled answers left out) and runs it read-only.
 *
 * Generated SQL stays internal to frame selection and pulling.
 *
 * Every choice is saved like every other card's settings.
 */
export function Track2SqlPage() {
  const [explorationHazard, setExplorationHazard] = useState("all");
  const [framesOpen, setFramesOpen] = useCardOpen("sql frames", true);
  const [explorationOpen, setExplorationOpen] = useCardOpen("data exploration", true);
  const [dbs, setDbs] = useState<Track2SqlDatabase[]>([]);
  const [database, setDatabase] = useState("");
  const [buckets, setBuckets] = useState<string[]>([]);
  const bucket = "";
  const folder = "";
  const [sql, setSql] = useState("");
  const [result, setResult] = useState<Track2SqlResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  // Starting or stopping the chosen database, and what the server said.
  const [power, setPower] = useState("");
  const [powerError, setPowerError] = useState("");
  // The point-and-select choice: the classes the database has, the one
  // chosen, and which of its frames.
  const [classes, setClasses] = useState<Track2SqlClass[] | null>(null);
  const [presetClass, setPresetClass] = useState("");
  const [presetScope, setPresetScope] = useState<Track2SqlPresetScope>("resolved");
  // Thumbnails: signed links for the images shown so far.
  const [previews, setPreviews] = useState<{ uri: string; url: string }[]>([]);
  const [framePage, setFramePage] = useState(1);
  const [previewError, setPreviewError] = useState("");
  const previewRequest = useRef(0);
  const [previewBusy, setPreviewBusy] = useState(false);
  // Draw each frame's labels over its thumbnail.
  const [showLabels, setShowLabels] = useState(true);
  // Write labels (and a data.yaml) into the training folder with the frames.
  const [withLabels, setWithLabels] = useState(true);
  const [classIds, setClassIds] = useState(SQL_DEFAULTS.class_ids);
  const availableLabels = result?.available_labels ?? [];
  const selectedClassIds = Object.fromEntries(availableLabels.map(label => [label.name, classIds[label.name] ?? label.class_id]));
  const selectedIds = Object.values(selectedClassIds);
  const invalidClassIds = new Set(selectedIds).size !== selectedIds.length || selectedIds.some(id => !Number.isInteger(id) || id < 0 || id > 99);
  // Where a pull copies to. Saved with the rest, so the next pull of the
  // same kind lands in the same place.
  const [pullBucket, setPullBucket] = useState("track2-s3");
  const [pullFolder, setPullFolder] = useState("");
  const [trainPercent, setTrainPercent] = useState(80);
  const [pullProgress, setPullProgress] = useState<Track2SqlPullProgress | null>(null);
  const [pulling, setPulling] = useState(false);
  const [pulled, setPulled] = useState<Track2SqlPullResult | null>(null);
  const [pullError, setPullError] = useState("");
  // The writer below must not run before the saved settings are read, or
  // it would save the empty defaults over them.
  const [settingsRevision, setSettingsRevision] = useState(0);
  const [initialPreset, setInitialPreset] = useState<SqlSavedPreset | null>(null);
  const [settingsRead, setSettingsRead] = useState(false);
  // Each run gets a number, so a slow answer to an earlier choice cannot
  // overwrite the answer to the latest one.
  const runId = useRef(0);

  useEffect(() => {
    getTrack2SqlDatabases()
      .then((res) => setDbs(res.databases))
      .catch(() => setDbs([]));
    getTrack2Buckets()
      .then((res) => setBuckets(res.buckets.filter((b) => b.readable).map((b) => b.name)))
      .catch(() => undefined);
    getTrack2PullControl()
      .then(async (status) => {
        setExplorationHazard(status.config.readiness_flow?.hazard || "all");
        const saved = status.config.sql_explorer || {};
        if (saved.database) setDatabase(saved.database);
        if (saved.sql) setSql(saved.sql);
        if (saved.preset_class) setPresetClass(saved.preset_class);
        if (saved.preset_scope === "ready" || saved.preset_scope === "test"
            || saved.preset_scope === "resolved") setPresetScope(saved.preset_scope);
        if (saved.pull_bucket) setPullBucket(saved.pull_bucket);
        if (saved.pull_folder) setPullFolder(saved.pull_folder);
        if (typeof saved.train_percent === "number") setTrainPercent(saved.train_percent);
        if (typeof saved.show_labels === "boolean") setShowLabels(saved.show_labels);
        if (typeof saved.with_labels === "boolean") setWithLabels(saved.with_labels);
        // Named EC2 files take precedence over the last auto-saved working state.
        // Finish restoring them before enabling queries or auto-save.
        const listed = await apiGet<{files: {name: string}[]}>("/track2/sql/settings");
        const newest = listed.files[0];
        if (newest) {
          const preset = await apiPost<SqlSavedPreset>("/track2/sql/settings/load", { name: newest.name });
          await ensureCardState();
          loadPageSettings(preset.settings);
          setInitialPreset(preset);
        }
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not load the latest saved settings. Please use Load to retry."))
      .finally(() => setSettingsRead(true));
  }, []);

  // First database by default, once the list is in and nothing was saved.
  useEffect(() => {
    if (settingsRead && !database && dbs.length) setDatabase(dbs[0].key);
  }, [settingsRead, database, dbs]);

  useEffect(() => {
    if (!settingsRead) return;
    const timer = window.setTimeout(() => {
      void saveTrack2PullConfig({ readiness_flow: { hazard: explorationHazard }, sql_explorer: {
        database, bucket, folder, sql,
        preset_class: presetClass, preset_scope: presetScope,
        pull_bucket: pullBucket, pull_folder: pullFolder, train_percent: trainPercent,
        show_labels: showLabels, with_labels: withLabels,
      } }).catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [settingsRead, explorationHazard, database, bucket, folder, sql, presetClass, presetScope,
      pullBucket, pullFolder, trainPercent, showLabels, withLabels]);

  // ...and different classes. Read live, so a new class shows up here
  // without a change to the app. Only once the database is running: a
  // stopped one would just time out.
  const chosen = dbs.find((d) => d.key === database);

  // While a database is starting or stopping, re-read the statuses every
  // 20 s, so the page notices by itself when it is ready -- no reload.
  const changing = chosen && ["starting", "stopping", "rebooting"].includes(chosen.status);
  useEffect(() => {
    if (!changing) return;
    const timer = window.setInterval(() => {
      getTrack2SqlDatabases().then((res) => setDbs(res.databases)).catch(() => undefined);
    }, 20000);
    return () => window.clearInterval(timer);
  }, [changing]);

  function switchPower(run: boolean) {
    if (!chosen || power) return;
    setPower(run ? "Starting" : "Stopping");
    setPowerError("");
    setTrack2SqlDatabaseRunning(chosen.key, run)
      .then((res) => setDbs((all) => all.map((d) => (d.key === res.key
        ? { ...d, status: res.status, available: res.status === "available" } : d))))
      .catch((err) => setPowerError(err instanceof ApiError ? err.detail : "It did not respond."))
      .finally(() => setPower(""));
  }

  // The class list belongs to ONE database. Without this, the render in
  // which the database changes still holds the old database's classes, and
  // the effect below would run a query against the new -- possibly stopped
  // -- database (UAT-02, 2026-11-04).
  const [classesFor, setClassesFor] = useState("");
  useEffect(() => {
    setClasses(null);
    setClassesFor("");
    if (!database || !chosen?.available) return;
    let current = true;
    getTrack2SqlClasses(database)
      .then((res) => { if (current) { setClasses(res.classes); setClassesFor(database); } })
      .catch(() => { if (current) { setClasses([]); setClassesFor(database); } });
    // A late answer for a database no longer chosen is dropped.
    return () => { current = false; };
  }, [database, chosen?.available]);

  const pickedClass = classesFor === database && chosen?.available
    ? classes?.find((c) => c.name === presetClass)
    : undefined;
  const scopeAllowed = (scope: Track2SqlPresetScope) =>
    scope === "resolved" || !pickedClass || pickedClass.has_boxes;

  // POINT AND SELECT: choosing a class, or which frames, shows them. No
  // button -- the drop-downs are the whole control. Waits for the class
  // list so a saved class is only run once the database has confirmed it.
  useEffect(() => {
    // Never query a database that is not running: it can only time out.
    if (!settingsRead || !database || !chosen?.available || !pickedClass) return;
    const scope = scopeAllowed(presetScope) ? presetScope : "resolved";
    if (scope !== presetScope) {
      setPresetScope(scope);
      return;
    }
    void showClass(pickedClass.name, scope);
  }, [settingsRead, database, chosen?.available, pickedClass?.name, presetScope, settingsRevision]);

  async function showClass(name: string, scope: Track2SqlPresetScope) {
    const mine = ++runId.current;
    setResult(null); setPreviews([]); setPulled(null); setPullProgress(null);
    setBusy(true);
    setError("");
    try {
      const { sql: text } = await getTrack2SqlPreset(name, scope, database);
      if (mine !== runId.current) return;
      setSql(text);
      await runQuery(text, mine);
    } catch (err) {
      if (mine === runId.current) {
        setError(err instanceof ApiError ? err.detail : "Could not build the query.");
        setBusy(false);
      }
    }
  }

  async function runQuery(text: string, mine: number) {
    setError("");
    setCopied(false);
    setPulled(null);
    setPreviews([]);
    setFramePage(1);
    previewRequest.current += 1;
    try {
      const res = await runTrack2Sql(database, text, bucket, folder);
      if (mine !== runId.current) return;
      setResult(res);
      await loadPreviews(res.images.map((i) => i.uri), 0, mine);
    } catch (err) {
      if (mine !== runId.current) return;
      setResult(null);
      setError(err instanceof ApiError ? err.detail : "The query failed.");
    } finally {
      if (mine === runId.current) setBusy(false);
    }
  }

  /** Replace the gallery with one page; late requests cannot replace newer pages. */
  async function loadPreviews(uris: string[], from: number, mine = runId.current) {
    const request = ++previewRequest.current;
    const page = uris.slice(from, from + PREVIEW_PAGE).filter((u) => u.startsWith("s3://"));
    setFramePage(Math.floor(from / PREVIEW_PAGE) + 1);
    setPreviews([]); setPreviewError(""); setPreviewBusy(true);
    try {
      const res = page.length ? await getTrack2SqlPreviews(page) : { previews: [] };
      if (mine !== runId.current || request !== previewRequest.current) return;
      setPreviews(res.previews);
    } catch {
      if (mine === runId.current && request === previewRequest.current)
        setPreviewError("Could not load this page. Please retry.");
    } finally {
      if (mine === runId.current && request === previewRequest.current) setPreviewBusy(false);
    }
  }

  /** Runs the query again server-side and copies what it names, so the
   * folder records the exact query that chose its images. */
  function pullImages() {
    if (!database || !sql.trim() || pulling) return;
    setPullProgress({ phase: "querying", completed: 0, total: 0 });
    setPulling(true);
    setPullError("");
    setPulled(null);
    pullTrack2Sql({ database, sql, bucket, folder,
                    destBucket: pullBucket, destFolder: pullFolder, trainPercent,
                    withLabels, classIds: availableLabels.length ? selectedClassIds : undefined }, setPullProgress)
      .then((summary) => { setPulled(summary); setPullProgress(p => p ? { ...p, phase: "complete" } : null); })
      .catch((err) => { setPullError(err instanceof ApiError ? err.detail : "The pull connection failed. Check the destination before retrying."); setPullProgress(p => p ? { ...p, phase: "error" } : null); })
      .finally(() => setPulling(false));
  }

  function copyImages() {
    if (!result) return;
    void navigator.clipboard
      .writeText(result.images.map((image) => image.uri).join("\n"))
      .then(() => setCopied(true))
      .catch(() => undefined);
  }

  function loadPageSettings(next: SqlPageSettings) {
    runId.current += 1;
    setBusy(false); setResult(null); setPreviews([]); setPulled(null);
    setError(""); setPullError(""); setSql("");
    setDatabase(next.database); setPresetClass(next.preset_class); setPresetScope(next.preset_scope);
    setPullBucket(next.pull_bucket); setPullFolder(next.pull_folder); setTrainPercent(next.train_percent);
    setShowLabels(next.show_labels); setWithLabels(next.with_labels);
    setClassIds(next.class_ids ?? SQL_DEFAULTS.class_ids);
    setExplorationHazard(next.exploration_hazard); setExplorationOpen(next.exploration_open); setFramesOpen(next.frames_open);
    setSettingsRevision(v => v + 1);
  }

  const selectClass = "mt-1 w-full rounded border border-black/15 bg-page px-2 py-1.5 text-sm text-ink";
  const labelsByUri = new Map(
    (result?.images ?? []).filter((i) => i.labels).map((i) => [i.uri, i.labels!]));
  const totalPages = Math.max(1, Math.ceil((result?.images.length ?? 0) / PREVIEW_PAGE));
  const firstPage = Math.max(1, Math.min(framePage - 4, totalPages - 9));
  const pageNumbers = Array.from({ length: Math.min(10, totalPages) }, (_, i) => firstPage + i);
  function renderFrameNavigation(position: "top" | "bottom") {
    if (!result) return null;
    return (
                <div className="my-3 flex flex-wrap items-center gap-3">
                  <nav aria-label={`Frame pages (${position})`} className="flex flex-wrap items-center gap-1">
                    <button type="button" disabled={previewBusy || framePage <= 1}
                      onClick={() => void loadPreviews(result.images.map(i => i.uri), (framePage - 2) * PREVIEW_PAGE)}
                      className="rounded px-3 py-1 text-xs text-accent disabled:opacity-40">Previous</button>
                    {pageNumbers.map(page => <button key={page} type="button"
                      aria-label={`Page ${page}`} aria-current={framePage === page ? "page" : undefined}
                      disabled={previewBusy}
                      onClick={() => void loadPreviews(result.images.map(i => i.uri), (page - 1) * PREVIEW_PAGE)}
                      className={`rounded px-2.5 py-1 text-xs disabled:opacity-40 ${framePage === page ? "bg-accent font-semibold text-white" : "text-accent hover:bg-black/5"}`}>
                      {page}
                    </button>)}
                    <button type="button" disabled={previewBusy || framePage >= totalPages}
                      onClick={() => void loadPreviews(result.images.map(i => i.uri), framePage * PREVIEW_PAGE)}
                      className="rounded px-3 py-1 text-xs text-accent disabled:opacity-40">Next</button>
                    <span className="ml-2 text-xs text-ink-muted">Page {framePage} of {totalPages}</span>
                  </nav>
                  {previewBusy && <span role="status" className="text-xs">Loading frames…</span>}
                  {previewError && <span role="alert" className="text-xs text-critical">{previewError}
                    <button type="button" className="ml-2 underline" onClick={() => void loadPreviews(result.images.map(i => i.uri), (framePage - 1) * PREVIEW_PAGE)}>Retry</button>
                  </span>}
                  <button
                    type="button"
                    onClick={copyImages}
                    className="inline-flex items-center gap-1.5 rounded-md border border-black/15 bg-page px-2.5 py-1 text-xs font-medium text-ink"
                  >
                    <Copy size={13} />
                    Copy the list
                  </button>
                  {copied && <span className="text-xs text-ink-muted">Copied.</span>}
                </div>
    );
  }

  const hasLabels = labelsByUri.size > 0;
  const incompleteCount = [...labelsByUri.values()].filter((l) => !l.complete).length;

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Database size={20} className="text-accent" strokeWidth={2} />
        Track2 SQL Database Query
      </h1>
      <p className="mb-4 max-w-3xl text-sm text-ink-secondary">
        Pick a class and see its frames. Only frames people have fully answered are shown &mdash;
        every person in the frame has a human violation / no violation; VLM-only and
        pre-filled answers are left out. Read only: nothing here changes the database.
      </p>

      <Track2SqlSettingsBar initialPreset={initialPreset} disabled={!settingsRead || pulling} onLoad={loadPageSettings}
        settings={{ database, preset_class: presetClass, preset_scope: presetScope,
          pull_bucket: pullBucket, pull_folder: pullFolder, train_percent: trainPercent,
          show_labels: showLabels, with_labels: withLabels,
          exploration_hazard: explorationHazard, exploration_open: explorationOpen, frames_open: framesOpen,
          class_ids: { ...classIds, ...selectedClassIds }, export_mode: "detection", detector_names: SQL_DEFAULTS.detector_names, classifier_names: SQL_DEFAULTS.classifier_names }} />
      <div className="flex flex-col gap-5">
        <Card title="Choose the frames" collapsible={false}>
          <div className="grid gap-3 md:grid-cols-3">
            <label className="block text-xs text-ink-muted">
              Database
              <select
                value={database}
                onChange={(e) => {
                  runId.current += 1;      // abandon a search against the old one
                  setBusy(false);
                  setError("");
                  setDatabase(e.target.value);
                  setResult(null);
                  setPreviews([]);
                }}
                className={selectClass}
              >
                {dbs.map((d) => (
                  <option key={d.key} value={d.key}>
                    {d.label}{d.available ? "" : ` -- ${d.status}`}
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-ink-muted">
              Class
              <select
                value={presetClass}
                disabled={!classes?.length}
                onChange={(e) => setPresetClass(e.target.value)}
                className={`${selectClass} disabled:opacity-50`}
              >
                <option value="">
                  {!chosen?.available ? "database not running"
                    : classes === null ? "reading the classes..."
                    : classes.length ? "Choose a class" : "no classes found"}
                </option>
                {classes?.map((c) => (
                  <option key={c.name} value={c.name}>
                    {c.name.replace(/_/g, " ")} ({c.frames.toLocaleString()} judged)
                  </option>
                ))}
              </select>
            </label>
            <label className="block text-xs text-ink-muted">
              Which frames
              <select
                value={presetScope}
                disabled={!pickedClass}
                onChange={(e) => setPresetScope(e.target.value as Track2SqlPresetScope)}
                className={`${selectClass} disabled:opacity-50`}
              >
                {(Object.keys(SCOPE_LABEL) as Track2SqlPresetScope[]).map((scope) => (
                  <option key={scope} value={scope} disabled={!scopeAllowed(scope)}>
                    {SCOPE_LABEL[scope]}
                  </option>
                ))}
              </select>
            </label>
          </div>
          {/* ---- the database's power ---- Here, under the drop-down that
               said it is stopped, so the fix is where the problem is read. */}
          {chosen && !chosen.available && (
            <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
              {changing ? (
                <span className="flex items-center gap-1.5 text-ink-secondary">
                  <Loader2 size={13} className="animate-spin" />
                  {chosen.status === "starting"
                    ? "Starting -- this takes about 5 to 10 minutes. The classes appear by themselves when it is ready."
                    : `The database is ${chosen.status}.`}
                </span>
              ) : (
                <>
                  <span className="text-critical">
                    This database is {chosen.status}. Its frames cannot be read until it is running.
                  </span>
                  {chosen.status === "stopped" && (
                    <button
                      type="button"
                      disabled={!!power}
                      onClick={() => switchPower(true)}
                      className="inline-flex items-center gap-1.5 rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40"
                    >
                      {power ? <Loader2 size={13} className="animate-spin" /> : <Play size={13} />}
                      {power || "Start the database"}
                    </button>
                  )}
                </>
              )}
            </div>
          )}
          {chosen?.available && (
            <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-ink-muted">
              <span>Running.</span>
              <button
                type="button"
                disabled={!!power || busy}
                onClick={() => switchPower(false)}
                className="underline disabled:opacity-40"
                title="A stopped database costs nothing but storage. Start it again here when needed."
              >
                {power || "Stop it when you are done"}
              </button>
            </div>
          )}
          {powerError && <p className="mt-1 text-xs text-critical">{powerError}</p>}
          {pickedClass && !pickedClass.has_boxes && (
            <p className="mt-2 text-xs text-ink-muted">
              {pickedClass.name.replace(/_/g, " ")} has no helmet-box step yet, so every
              human-resolved frame is shown.
            </p>
          )}
          {busy && (
            <p className="mt-3 flex items-center gap-1.5 text-sm text-ink-secondary">
              <Loader2 size={14} className="animate-spin" /> Finding the frames...
            </p>
          )}
          {error && (
            <p className="mt-3 flex items-start gap-1.5 text-sm text-critical">
              <AlertCircle size={15} className="mt-0.5 shrink-0" />
              <span className="whitespace-pre-wrap">{error}</span>
            </p>
          )}
        </Card>

        {result && (
          <Card
            title={`Frames (${result.images.length.toLocaleString()})`}
            stateKey="sql frames"
            caption={result.images.length
              ? `Showing ${previews.length.toLocaleString()} of ${result.images.length.toLocaleString()}. Click a frame to open it full size.`
              : undefined}
          >
            {result.images.length === 0 ? (
              <p className="text-sm text-ink-muted">No frames match this choice.</p>
            ) : (
              <>
                {/* ---- the preview ---- */}
                {hasLabels && (
                  <div className="mb-2 flex flex-wrap items-center gap-4 text-xs">
                    <label className="flex items-center gap-1.5 text-ink">
                      <input type="checkbox" checked={showLabels}
                             onChange={(e) => setShowLabels(e.target.checked)} />
                      Show labels
                    </label>
                    {showLabels && (
                      <span className="flex items-center gap-3 text-ink-muted">
                        {availableLabels.map(label => <span key={label.name} className="flex items-center gap-1"
                          title={`${label.boxes.toLocaleString()} boxes in ${label.frames.toLocaleString()} selected frames`}>
                          <i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: boxColour(label.name) }} />
                          {label.name} ({label.boxes.toLocaleString()})
                        </span>)}
                        {availableLabels.length === 0 && <span>No box labels available in these frames.</span>}
                        {incompleteCount > 0 && (
                          <span className="text-critical">
                            {incompleteCount.toLocaleString()} frame(s) marked: a helmet box is missing
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                )}
                {renderFrameNavigation("top")}
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
                  {previews.map((p) => {
                    const labels = labelsByUri.get(p.uri);
                    return (
                      <a key={p.uri} href={p.url} target="_blank" rel="noreferrer"
                         title={labels && !labels.complete ? `${p.uri} -- ${labels.reason}` : p.uri}
                         className={`relative block aspect-video overflow-hidden rounded border bg-black/80 ${
                           labels && !labels.complete && showLabels ? "border-critical" : "border-hairline"}`}>
                        <img src={p.url} alt="" loading="lazy"
                             className="absolute inset-0 h-full w-full object-contain" />
                        {showLabels && labels && <LabelOverlay labels={labels} />}
                      </a>
                    );
                  })}
                </div>
                {renderFrameNavigation("bottom")}

                {/* ---- pull them into a training folder ---- Here, under
                     what it acts on, so the answer appears where the button
                     was pressed. */}
                <div className="mt-4 rounded border border-hairline bg-page p-3">
                  <p className="mb-2 text-sm font-medium text-ink">Pull these frames into a training folder</p>
                  <p className="mb-3 text-xs text-ink-secondary">
                    Copies frames and their labels into the selected training folder. Source images and reviews stay unchanged.

                  </p>
                  <div className="grid gap-3 md:grid-cols-[14rem_1fr_9rem]">
                    <label className="block text-xs text-ink-muted">
                      Into bucket
                      <select
                        value={pullBucket}
                        onChange={(e) => { setPullBucket(e.target.value); setPullFolder(""); }}
                        className="mt-1 w-full rounded border border-black/15 bg-surface px-2 py-1.5 text-sm text-ink"
                      >
                        {[pullBucket, ...buckets.filter((b) => b !== pullBucket)].map((name) => (
                          <option key={name} value={name}>{name}</option>
                        ))}
                      </select>
                    </label>
                    <S3FolderField
                      // Reload the saved destination even if the user only browsed
                      // elsewhere without changing the selected folder.
                      key={`${settingsRevision}:${pullBucket}`}
                      label="Folder"
                      hint="new or existing; images/ and labels/ go inside it"
                      value={pullFolder}
                      onChange={setPullFolder}
                      bucket={pullBucket}
                      allowCreate
                    />
                    <label className="block text-xs text-ink-muted">
                      % to train
                      <input
                        type="number"
                        min={0}
                        max={100}
                        value={trainPercent}
                        onChange={(e) => setTrainPercent(Math.max(0, Math.min(100, Number(e.target.value) || 0)))}
                        className="tabular mt-1 w-full rounded border border-black/15 bg-surface px-2 py-1.5 text-sm text-ink"
                      />
                    </label>
                  </div>
                  <p className="mt-1 text-xs text-ink-muted">
                    The rest go to val. Each frame&rsquo;s side is fixed by its address, so pulling
                    again never moves one from val into train &mdash; which also means the split is
                    approximate on a small set (80% of 72 frames came out 54 / 18). Keep your test
                    set out of this folder.
                    {hasLabels && incompleteCount > 0 && withLabels && (
                      <span className="mt-1 block text-ink-secondary">
                        With labels included, the {incompleteCount.toLocaleString()} frame
                        {incompleteCount === 1 ? "" : "s"} marked &ldquo;helmet box missing&rdquo;
                        {" "}{incompleteCount === 1 ? "is" : "are"} left out, so fewer frames are pulled
                        than shown. Test-grade means a person drew the boxes that exist, not that
                        every helmet has one.
                      </span>
                    )}
                  </p>
                  <label className="mt-3 flex items-start gap-1.5 text-xs text-ink">
                    <input type="checkbox" className="mt-0.5" checked={withLabels}
                           onChange={(e) => setWithLabels(e.target.checked)} />
                    <span>
                      Include labels
                      <span className="block text-ink-muted">
                        {hasLabels
                          ? "Exports the available frame-aligned boxes shown above, plus data.yaml. Missing required boxes can exclude a frame. Check the overlay for unlabelled objects before training."
                          : "Copies the label that sits beside each image at the source, where there is one."}
                      </span>
                    </span>
                  </label>
                  {withLabels && <fieldset disabled={pulling} className="mt-3 rounded border border-hairline p-3">
                    <legend className="px-1 text-sm font-medium">Available labels and class IDs</legend>
                    <div className="flex flex-wrap gap-6">
                      {availableLabels.map(({name}) => <label key={name} className="flex items-center gap-2 text-xs">
                        {name}<input aria-label={`${name} class ID`} type="number" min={0} max={99}
                          value={selectedClassIds[name]} onChange={e => setClassIds(previous => ({...previous, [name]: Number(e.target.value)}))}
                          className="w-20 rounded border border-hairline bg-page px-2 py-1" />
                      </label>)}
                    </div>
                    {availableLabels.length === 0 && <p className="text-xs text-ink-muted">No box labels are available to map.</p>}
                    <p className="mt-2 text-xs text-ink-muted">Labels come from all selected frames, not just this page. Hazard names and detections from other video moments are not frame labels.</p>
                    <p className="mt-2 text-xs text-ink-muted">These numbers are used in the label files and data.yaml. IDs must be different. Gaps are reserved as unused IDs in data.yaml.</p>
                    {invalidClassIds && <p role="alert" className="mt-1 text-xs text-critical">Use a different whole number from 0 to 99 for each available label.</p>}
                  </fieldset>}
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={pullImages}
                      disabled={pulling || !pullBucket || !pullFolder.trim() || (withLabels && invalidClassIds)}
                      className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
                    >
                      {pulling ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                      {pulling ? "Pulling..." : `Pull ${result.images.length.toLocaleString()} frame${result.images.length === 1 ? "" : "s"}`}
                    </button>
                    {!pullFolder.trim() && <span className="text-xs text-ink-muted">choose a folder first</span>}
                  </div>
                  {pullError && (
                    <p className="mt-2 flex items-start gap-1.5 text-sm text-critical">
                      <AlertCircle size={15} className="mt-0.5 shrink-0" />
                      <span className="whitespace-pre-wrap">{pullError}</span>
                    </p>
                  )}
                  {pullProgress && (
                    <div className="mt-3 rounded border border-hairline p-3" aria-label="Frame pull progress">
                      <div className="mb-2 flex justify-between text-xs" role="status">
                        <span>{pullProgress.phase === "querying" ? "Reading frames and checking labels…"
                          : pullProgress.phase === "finalizing" ? "Saving dataset metadata…"
                          : pullProgress.phase === "error" ? "Pull interrupted"
                          : pullProgress.phase === "complete" ? ((pulled?.failed ?? 0) > 0 ? "Finished with errors" : "Pull complete")
                          : "Copying frames and labels…"}</span>
                        <span>{pullProgress.completed.toLocaleString()} / {pullProgress.total.toLocaleString()} eligible samples</span>
                      </div>
                      <progress aria-label="Frame pull completion" max={100}
                        value={pullProgress.phase === "querying" ? undefined : pullProgress.phase === "complete" ? 100
                          : Math.min(99, Math.round(100 * pullProgress.completed / Math.max(1, pullProgress.total)))}
                        className="h-3 w-full accent-accent" />
                      <p className="mt-1 text-xs text-ink-muted">
                        {pullProgress.copied ?? 0} copied · {pullProgress.skipped ?? 0} already present · {pullProgress.failed ?? 0} failed
                        {(pullProgress.left_out_incomplete ?? 0) > 0 && ` · ${pullProgress.left_out_incomplete} frames with no eligible samples excluded`}
                      </p>
                    </div>
                  )}
                  {pulled && (
                    <div className="tabular mt-2 text-xs text-ink-secondary">
                      <p>
                        Into <span className="font-medium text-ink">s3://{pulled.bucket}/{pulled.folder}/</span>:{" "}
                        {pulled.copied.toLocaleString()} copied, {pulled.skipped.toLocaleString()} already there
                        {" "}({pulled.train.toLocaleString()} train, {pulled.val.toLocaleString()} val).{" "}
                        {pulled.with_label.toLocaleString()} with a label
                        {pulled.export_mode === "classification" ? " (from explicit human person verdicts)" : pulled.labels_from === "database" ? " (from the reviewed boxes, with a data.yaml)" : ""}.
                      </p>
                      {pulled.left_out_incomplete > 0 && (
                        <p className="mt-1">
                          {pulled.left_out_incomplete.toLocaleString()} frame(s) had no eligible samples.
                        </p>
                      )}
                      {pulled.labels_from !== "none" && pulled.with_label < pulled.copied + pulled.skipped && (
                        <p className="mt-1 text-critical">
                          {(pulled.copied + pulled.skipped - pulled.with_label).toLocaleString()} frame(s) have no
                          label beside them at the source. Training reads an unlabelled image as
                          &ldquo;nothing here&rdquo;, so label them before training on this folder.
                        </p>
                      )}
                      {pulled.bare_paths_left_out > 0 && (
                        <p className="mt-1">
                          {pulled.bare_paths_left_out} path(s) named no bucket and were left out.
                        </p>
                      )}
                      {pulled.failed > 0 && (
                        <p className="mt-1 text-critical">
                          {pulled.failed} failed: {pulled.errors.join("; ")}
                        </p>
                      )}
                      <p className="mt-1 text-ink-muted">
                        {pulled.export_mode ? "The exports/ receipts record source images, labels, exclusions and the query. export-contract.json records the class mapping and split."
                          : "sources.json and query.json record source images and the query."}
                      </p>
                      {pulled.excluded_reasons && Object.entries(pulled.excluded_reasons).map(([reason, count]) =>
                        <p key={reason}>{reason.replaceAll("_", " ")}: {count}</p>)}
                    </div>
                  )}
                </div>
              </>
            )}
          </Card>
        )}

        <Track2ReadinessFlowPanel
          hazard={explorationHazard}
          onHazardChange={setExplorationHazard}
          database={database}
          available={!!chosen?.available}
          databaseLabel={chosen?.label ?? ""}
        />


      </div>
    </div>
  );
}

/** Stable per-label colours shared by the legend and overlays. */
function boxColour(name: string): string {
  let hash = 0;
  for (const char of name) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
  return `hsl(${hash % 360} 75% 43%)`;
}

/** A frame's boxes drawn over its thumbnail. The SVG is the frame's own
 * size and letterboxed the same way as the image (`meet` / object-contain),
 * so the boxes land where they belong whatever the frame's shape. */
function LabelOverlay({ labels }: { labels: Track2SqlFrameLabels }) {
  const w = labels.width || 16;
  const h = labels.height || 9;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="xMidYMid meet"
         className="pointer-events-none absolute inset-0 h-full w-full">
      {labels.boxes.map((b, i) => (
        <rect key={i} x={b.x1 * w} y={b.y1 * h} width={(b.x2 - b.x1) * w} height={(b.y2 - b.y1) * h}
              fill="none" stroke={boxColour(b.name)} strokeWidth={2}
              vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}
