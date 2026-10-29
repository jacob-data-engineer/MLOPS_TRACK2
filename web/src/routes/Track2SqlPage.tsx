import {
  AlertCircle, ChevronDown, ChevronRight, Copy, Database, Download, Loader2, Play, Table2,
} from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import { Card } from "../components/Card";
import { S3FolderField } from "../components/S3FolderField";
import {
  ApiError,
  getTrack2Buckets,
  getTrack2PullControl,
  getTrack2SqlClasses,
  getTrack2SqlDatabases,
  getTrack2SqlPreset,
  getTrack2SqlPreviews,
  getTrack2SqlSchema,
  pullTrack2Sql,
  runTrack2Sql,
  saveTrack2PullConfig,
  setTrack2SqlDatabaseRunning,
  type Track2SqlClass,
  type Track2SqlDatabase,
  type Track2SqlFrameLabels,
  type Track2SqlPresetScope,
  type Track2SqlPullResult,
  type Track2SqlResult,
  type Track2SqlTable,
} from "../lib/api";

/** Rows drawn in the table. The images list below it is never cut. */
const ROWS_SHOWN = 500;
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
 * The SQL editor is still here, folded under "Advanced", for anything the
 * drop-downs do not cover. A chosen class fills it, so what is shown is
 * always exactly what "Pull these images" will copy.
 *
 * Every choice is saved like every other card's settings.
 */
export function Track2SqlPage() {
  const [dbs, setDbs] = useState<Track2SqlDatabase[]>([]);
  const [database, setDatabase] = useState("");
  const [buckets, setBuckets] = useState<string[]>([]);
  const [bucket, setBucket] = useState("");
  const [folder, setFolder] = useState("");
  const [sql, setSql] = useState("");
  const [tables, setTables] = useState<Track2SqlTable[] | null>(null);
  const [openTable, setOpenTable] = useState("");
  const [result, setResult] = useState<Track2SqlResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [schemaBusy, setSchemaBusy] = useState(false);
  const [error, setError] = useState("");
  const [copied, setCopied] = useState(false);
  const [advanced, setAdvanced] = useState(false);
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
  const [previewBusy, setPreviewBusy] = useState(false);
  // Draw each frame's labels over its thumbnail.
  const [showLabels, setShowLabels] = useState(true);
  // Write labels (and a data.yaml) into the training folder with the frames.
  const [withLabels, setWithLabels] = useState(true);
  // Where a pull copies to. Saved with the rest, so the next pull of the
  // same kind lands in the same place.
  const [pullBucket, setPullBucket] = useState("track2-s3");
  const [pullFolder, setPullFolder] = useState("");
  const [trainPercent, setTrainPercent] = useState(80);
  const [pulling, setPulling] = useState(false);
  const [pulled, setPulled] = useState<Track2SqlPullResult | null>(null);
  const [pullError, setPullError] = useState("");
  // The writer below must not run before the saved settings are read, or
  // it would save the empty defaults over them.
  const [settingsRead, setSettingsRead] = useState(false);
  const editor = useRef<HTMLTextAreaElement | null>(null);
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
      .then((status) => {
        const saved = status.config.sql_explorer || {};
        if (saved.database) setDatabase(saved.database);
        if (saved.bucket) setBucket(saved.bucket);
        if (saved.folder) setFolder(saved.folder);
        if (saved.sql) setSql(saved.sql);
        if (saved.preset_class) setPresetClass(saved.preset_class);
        if (saved.preset_scope === "ready" || saved.preset_scope === "test"
            || saved.preset_scope === "resolved") setPresetScope(saved.preset_scope);
        if (saved.pull_bucket) setPullBucket(saved.pull_bucket);
        if (saved.pull_folder) setPullFolder(saved.pull_folder);
        if (typeof saved.train_percent === "number") setTrainPercent(saved.train_percent);
        if (typeof saved.show_labels === "boolean") setShowLabels(saved.show_labels);
        if (typeof saved.with_labels === "boolean") setWithLabels(saved.with_labels);
      })
      .catch(() => undefined)
      .finally(() => setSettingsRead(true));
  }, []);

  // First database by default, once the list is in and nothing was saved.
  useEffect(() => {
    if (settingsRead && !database && dbs.length) setDatabase(dbs[0].key);
  }, [settingsRead, database, dbs]);

  useEffect(() => {
    if (!settingsRead) return;
    const timer = window.setTimeout(() => {
      void saveTrack2PullConfig({ sql_explorer: {
        database, bucket, folder, sql,
        preset_class: presetClass, preset_scope: presetScope,
        pull_bucket: pullBucket, pull_folder: pullFolder, train_percent: trainPercent,
        show_labels: showLabels, with_labels: withLabels,
      } }).catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [settingsRead, database, bucket, folder, sql, presetClass, presetScope,
      pullBucket, pullFolder, trainPercent, showLabels, withLabels]);

  // A different database has different tables.
  useEffect(() => {
    setTables(null);
    setOpenTable("");
  }, [database]);

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

  useEffect(() => {
    setClasses(null);
    if (!database || !chosen?.available) return;
    getTrack2SqlClasses(database)
      .then((res) => setClasses(res.classes))
      .catch(() => setClasses([]));
  }, [database, chosen?.available]);

  const pickedClass = classes?.find((c) => c.name === presetClass);
  const scopeAllowed = (scope: Track2SqlPresetScope) =>
    scope === "resolved" || !pickedClass || pickedClass.has_boxes;

  // POINT AND SELECT: choosing a class, or which frames, shows them. No
  // button -- the drop-downs are the whole control. Waits for the class
  // list so a saved class is only run once the database has confirmed it.
  useEffect(() => {
    if (!settingsRead || !database || !pickedClass) return;
    const scope = scopeAllowed(presetScope) ? presetScope : "resolved";
    if (scope !== presetScope) {
      setPresetScope(scope);
      return;
    }
    void showClass(pickedClass.name, scope);
  }, [settingsRead, database, pickedClass?.name, presetScope]);

  async function showClass(name: string, scope: Track2SqlPresetScope) {
    const mine = ++runId.current;
    setBusy(true);
    setError("");
    try {
      const { sql: text } = await getTrack2SqlPreset(name, scope);
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

  function loadTables() {
    if (!database) return;
    setSchemaBusy(true);
    setError("");
    getTrack2SqlSchema(database)
      .then((res) => setTables(res.tables))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not read the tables."))
      .finally(() => setSchemaBusy(false));
  }

  function run() {
    if (!database || !sql.trim() || busy) return;
    setBusy(true);
    void runQuery(sql, ++runId.current);
  }

  async function runQuery(text: string, mine: number) {
    setError("");
    setCopied(false);
    setPulled(null);
    setPreviews([]);
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

  /** Signs the next screenful of thumbnails and adds it to the grid. */
  async function loadPreviews(uris: string[], from: number, mine = runId.current) {
    const page = uris.slice(from, from + PREVIEW_PAGE).filter((u) => u.startsWith("s3://"));
    if (!page.length) return;
    setPreviewBusy(true);
    try {
      const res = await getTrack2SqlPreviews(page);
      if (mine !== runId.current) return;
      setPreviews((current) => (from === 0 ? res.previews : [...current, ...res.previews]));
    } catch {
      // A grid that cannot load is not worth hiding the list for.
    } finally {
      setPreviewBusy(false);
    }
  }

  function onEditorKey(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      run();
    }
  }

  /** Puts text at the cursor, so clicking a column name writes it there. */
  function insert(text: string) {
    const box = editor.current;
    if (!box) {
      setSql((current) => current + text);
      return;
    }
    const start = box.selectionStart ?? sql.length;
    const end = box.selectionEnd ?? sql.length;
    const next = sql.slice(0, start) + text + sql.slice(end);
    setSql(next);
    window.requestAnimationFrame(() => {
      box.focus();
      box.selectionStart = box.selectionEnd = start + text.length;
    });
  }

  /** Runs the query again server-side and copies what it names, so the
   * folder records the exact query that chose its images. */
  function pullImages() {
    if (!database || !sql.trim() || pulling) return;
    setPulling(true);
    setPullError("");
    setPulled(null);
    pullTrack2Sql({ database, sql, bucket, folder,
                    destBucket: pullBucket, destFolder: pullFolder, trainPercent,
                    withLabels })
      .then(setPulled)
      .catch((err) => setPullError(err instanceof ApiError ? err.detail : "The pull failed."))
      .finally(() => setPulling(false));
  }

  function copyImages() {
    if (!result) return;
    void navigator.clipboard
      .writeText(result.images.map((image) => image.uri).join("\n"))
      .then(() => setCopied(true))
      .catch(() => undefined);
  }

  const selectClass = "mt-1 w-full rounded border border-black/15 bg-page px-2 py-1.5 text-sm text-ink";
  const labelsByUri = new Map(
    (result?.images ?? []).filter((i) => i.labels).map((i) => [i.uri, i.labels!]));
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
            collapsible={false}
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
                        <span className="flex items-center gap-1">
                          <i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: BOX_COLOUR[2] }} />person
                        </span>
                        <span className="flex items-center gap-1">
                          <i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: BOX_COLOUR[0] }} />helmet
                        </span>
                        {incompleteCount > 0 && (
                          <span className="text-critical">
                            {incompleteCount.toLocaleString()} frame(s) marked: a helmet box is missing
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                )}
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
                <div className="mt-3 flex flex-wrap items-center gap-3">
                  {previews.length < result.images.length && (
                    <button
                      type="button"
                      disabled={previewBusy}
                      onClick={() => void loadPreviews(result.images.map((i) => i.uri), previews.length)}
                      className="rounded-md border border-black/15 bg-page px-2.5 py-1 text-xs font-medium text-ink disabled:opacity-50"
                    >
                      {previewBusy ? "Loading..." : `Show ${Math.min(PREVIEW_PAGE, result.images.length - previews.length)} more`}
                    </button>
                  )}
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

                {/* ---- pull them into a training folder ---- Here, under
                     what it acts on, so the answer appears where the button
                     was pressed. */}
                <div className="mt-4 rounded border border-hairline bg-page p-3">
                  <p className="mb-2 text-sm font-medium text-ink">Pull these frames into a training folder</p>
                  <p className="mb-3 text-xs text-ink-secondary">
                    Copies each frame to <span className="tabular">images/train</span> or{" "}
                    <span className="tabular">images/val</span>, and its label to{" "}
                    <span className="tabular">labels/</span> when one sits beside it at the source.
                    Copies only &mdash; nothing at the source is changed, and a name already in the
                    folder is skipped. Point Assemble on Data Preparation at the folder to train on it.
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
                    again never moves one from val into train. Keep your test set out of this folder.
                  </p>
                  <label className="mt-3 flex items-start gap-1.5 text-xs text-ink">
                    <input type="checkbox" className="mt-0.5" checked={withLabels}
                           onChange={(e) => setWithLabels(e.target.checked)} />
                    <span>
                      Include labels
                      <span className="block text-ink-muted">
                        {hasLabels
                          ? "A YOLO label per frame from the reviewed boxes (person, and helmet where worn), plus a data.yaml naming the classes. Frames with a helmet box missing are left out, so the helmet is never taught as background. The boxes cover only the people the review app judged: anyone else in the frame is unlabelled -- check the overlay, and complete those frames before training on them."
                          : "Copies the label that sits beside each image at the source, where there is one."}
                      </span>
                    </span>
                  </label>
                  <div className="mt-3 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={pullImages}
                      disabled={pulling || !pullBucket || !pullFolder.trim()}
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
                  {pulled && (
                    <div className="tabular mt-2 text-xs text-ink-secondary">
                      <p>
                        Into <span className="font-medium text-ink">s3://{pulled.bucket}/{pulled.folder}/</span>:{" "}
                        {pulled.copied.toLocaleString()} copied, {pulled.skipped.toLocaleString()} already there
                        {" "}({pulled.train.toLocaleString()} train, {pulled.val.toLocaleString()} val).{" "}
                        {pulled.with_label.toLocaleString()} with a label
                        {pulled.labels_from === "database" ? " (from the reviewed boxes, with a data.yaml)" : ""}.
                      </p>
                      {pulled.left_out_incomplete > 0 && (
                        <p className="mt-1">
                          {pulled.left_out_incomplete.toLocaleString()} frame(s) left out: a helmet box is
                          missing. Draw or accept it in the box queue, then pull again.
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
                        sources.json and query.json in the folder record where each file came from
                        and the query that chose them.
                      </p>
                    </div>
                  )}
                </div>
              </>
            )}
          </Card>
        )}

        {/* ---- ADVANCED: write your own SQL ---- Folded away, because the
             drop-downs are the normal way in; kept, for anything they do
             not cover. A chosen class fills the editor. */}
        <section className="rounded-sm border border-hairline bg-surface p-5">
          <button
            type="button"
            onClick={() => setAdvanced((v) => !v)}
            className="flex items-center gap-1.5 text-sm font-medium text-ink"
          >
            {advanced ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
            Advanced: write your own SQL
          </button>
          {advanced && (
            <div className="mt-4 flex flex-col gap-4">
              <div className="grid gap-3 md:grid-cols-2">
                <label className="block text-xs text-ink-muted">
                  Only images in bucket
                  <select
                    value={bucket}
                    onChange={(e) => { setBucket(e.target.value); setFolder(""); }}
                    className={selectClass}
                  >
                    <option value="">Any bucket</option>
                    {buckets.map((name) => (
                      <option key={name} value={name}>{name}</option>
                    ))}
                  </select>
                </label>
                {bucket ? (
                  <S3FolderField
                    label="Folder"
                    hint="only images under this folder; blank for the whole bucket"
                    value={folder}
                    onChange={setFolder}
                    bucket={bucket}
                  />
                ) : (
                  <p className="self-end text-xs text-ink-muted">Pick a bucket to narrow the images to one of its folders.</p>
                )}
              </div>

              <div className="grid gap-4 md:grid-cols-[16rem_1fr]">
                <div className="max-h-96 overflow-y-auto rounded border border-hairline bg-page p-2 text-xs">
                  <div className="mb-2 flex items-center justify-between gap-2">
                    <span className="font-medium text-ink">Tables</span>
                    <button
                      type="button"
                      onClick={loadTables}
                      disabled={!database || schemaBusy}
                      className="rounded border border-black/15 px-2 py-0.5 text-ink disabled:opacity-50"
                    >
                      {schemaBusy ? "Reading..." : tables ? "Reload" : "Load tables"}
                    </button>
                  </div>
                  {tables === null && !schemaBusy && (
                    <p className="text-ink-muted">Not loaded yet.</p>
                  )}
                  {tables?.length === 0 && <p className="text-ink-muted">No tables this user can read.</p>}
                  {tables?.map((table) => (
                    <div key={table.name} className="mb-1">
                      <div className="flex items-center gap-1">
                        <button
                          type="button"
                          onClick={() => setOpenTable(openTable === table.name ? "" : table.name)}
                          className="text-ink-muted"
                          aria-label={openTable === table.name ? "Hide columns" : "Show columns"}
                        >
                          {openTable === table.name ? "▾" : "▸"}
                        </button>
                        <button
                          type="button"
                          onClick={() => setSql(`SELECT *\nFROM ${table.name}\nLIMIT 100`)}
                          className="tabular truncate text-left text-ink hover:underline"
                          title={`Start a query on ${table.name}`}
                        >
                          <Table2 size={12} className="mr-1 inline" />
                          {table.name}
                        </button>
                      </div>
                      {openTable === table.name && (
                        <ul className="ml-5 mt-0.5">
                          {table.columns.map((column) => (
                            <li key={column.name}>
                              <button
                                type="button"
                                onClick={() => insert(column.name)}
                                className="tabular text-left text-ink-secondary hover:underline"
                              >
                                {column.name} <span className="text-ink-muted">{column.type}</span>
                              </button>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                  ))}
                </div>

                <div>
                  <textarea
                    ref={editor}
                    value={sql}
                    onChange={(e) => setSql(e.target.value)}
                    onKeyDown={onEditorKey}
                    spellCheck={false}
                    rows={12}
                    placeholder={"SELECT image_key\nFROM some_table\nWHERE ...\nLIMIT 100"}
                    className="tabular w-full rounded border border-black/15 bg-page px-3 py-2 font-mono text-sm text-ink"
                  />
                  <div className="mt-2 flex flex-wrap items-center gap-3">
                    <button
                      type="button"
                      onClick={run}
                      disabled={busy || !database || !sql.trim()}
                      className="inline-flex items-center gap-1.5 rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white hover:bg-accent-hover disabled:opacity-40"
                    >
                      {busy ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                      {busy ? "Running..." : "Run query"}
                    </button>
                    <span className="text-xs text-ink-muted">Ctrl+Enter runs it.</span>
                    {result && (
                      <span className="tabular text-xs text-ink-muted">
                        {result.rows.length.toLocaleString()} row{result.rows.length === 1 ? "" : "s"}
                        {result.truncated && " (cut short: the answer was too large -- add a LIMIT or select fewer columns)"}
                      </span>
                    )}
                  </div>
                </div>
              </div>

              {result && result.columns.length > 0 && (
                <div>
                  <p className="mb-1 text-xs text-ink-muted">
                    {result.rows.length > ROWS_SHOWN
                      ? `The first ${ROWS_SHOWN.toLocaleString()} of ${result.rows.length.toLocaleString()} rows.`
                      : "Rows"}
                  </p>
                  <div className="max-h-[32rem] overflow-auto rounded border border-hairline">
                    <table className="tabular min-w-full text-xs">
                      <thead className="sticky top-0 bg-surface">
                        <tr>
                          {result.columns.map((column, i) => (
                            <th key={`${column}-${i}`} className="border-b border-hairline px-2 py-1 text-left font-medium text-ink">
                              {column}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.rows.slice(0, ROWS_SHOWN).map((row, r) => (
                          <tr key={r} className="odd:bg-black/[0.02]">
                            {row.map((cell, c) => (
                              <td key={c} className="max-w-xs truncate border-b border-hairline/50 px-2 py-0.5 text-ink-secondary"
                                  title={cellText(cell)}>
                                {cellText(cell)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}

/** person blue, helmet green -- the report's own colours for human boxes
 * and ready frames. */
const BOX_COLOUR: Record<number, string> = { 0: "#0c8f3c", 2: "#2a78d6" };

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
              fill="none" stroke={BOX_COLOUR[b.cls] ?? "#d48a00"} strokeWidth={2}
              vectorEffect="non-scaling-stroke" />
      ))}
    </svg>
  );
}

function cellText(value: unknown): string {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}
