import { AlertCircle, Library } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Card } from "../components/Card";
import { Track2CatalogMetadata, type CatalogSelection } from "../components/Track2CatalogMetadata";
import { Track2S3FolderBrowser } from "../components/Track2S3FolderBrowser";
import {
  ApiError,
  getTrack2Buckets,
  listTrack2Folders,
  type Track2Bucket,
  getTrack2CatalogRuns,
  getTrack2CatalogStatus,
  getTrack2CatalogVersions,
  getTrack2PullControl,
  saveTrack2PullConfig,
  type Track2CatalogFilter,
  type Track2CatalogRun,
  type Track2CatalogStatus,
  type Track2CatalogVersion,
  type CatalogExportDestination,
} from "../lib/api";
import { formatSgt } from "../lib/format";

/** TRACK2 DATA CATALOG (2026-10-29; reworked 2026-11-18).
 *
 * Every training run (by run number) and every dataset version and
 * snapshot (by version number), read from the catalog database. Search and
 * filters are applied by the SERVER across the whole catalog, and Load more
 * reaches older records. Pick one to review its metadata, check its dataset
 * can be rebuilt, or export it.
 *
 * Read only. Nothing here changes a run, a version or a file.
 */
type Tab = "runs" | "versions";
type Picked = { kind: "run"; run: Track2CatalogRun } | { kind: "version"; version: Track2CatalogVersion };

/** How many rows a page asks for; Load more asks for the next this many. */
const PAGE = 50;

const when = (iso: string | null | undefined) => formatSgt(iso, "--");

/** A run's model in the four words the Models page uses -- Model name,
 * Model type, Size, Model version -- so a row here can be matched to a
 * version there (and its .pt exported) without decoding "yolo11s-helmet-
 * detector v32". Type and size come from the run's architecture, or from
 * the registry group it registered into when that was not recorded. */
function runModel(r: { detector: string | null; architecture: string | null; model_package_arn: string | null }) {
  const [group, version] = r.model_package_arn ? r.model_package_arn.split("/").slice(-2) : ["", ""];
  const arch = /^yolo(\d+)([nslmx])/i.exec(r.architecture || group);
  return {
    name: r.detector || "--",
    type: arch ? `YOLO${arch[1]}` : "--",
    size: arch ? arch[2].toUpperCase() : "--",
    version: version ? `v${version}` : "--",
    group: group || "--",
  };
}

/** One table's rows and how the last read of them went. A failed read
 * keeps whatever was already on screen and says it failed -- never an
 * empty list standing in for an error. */
type Table<T> = { status: "loading" | "ready" | "failed"; rows: T[]; total: number; error: string; at: string; more: boolean };
function emptyTable<T>(): Table<T> {
  return { status: "loading", rows: [], total: 0, error: "", at: "", more: false };
}

const RUN_STATUSES = ["Completed", "Failed", "InProgress", "Stopped", "Stopping", "Queued"];
const VERSION_STATES = ["frozen", "preserved", "deleted"];

export function Track2CatalogPage({ refreshKey = 0 }: { refreshKey?: number }) {
  const [status, setStatus] = useState<Track2CatalogStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [tab, setTab] = useState<Tab>("runs");
  const [runs, setRuns] = useState<Table<Track2CatalogRun>>(emptyTable);
  const [versions, setVersions] = useState<Table<Track2CatalogVersion>>(emptyTable);
  const [picked, setPicked] = useState<Picked | null>(null);
  // The dataset version a picked RUN trained on, for its train/val counts.
  const [runDataset, setRunDataset] = useState<{ runId: string; version: Track2CatalogVersion | null; failed: boolean } | null>(null);
  const [destination, setDestination] = useState<CatalogExportDestination>({ destination: "local", bucket: "", prefix: "", format: "zip" });
  const [error, setError] = useState("");
  // Filters as typed, and as last applied. Only applied ones reach the
  // server; both survive a refresh.
  const [draft, setDraft] = useState<Track2CatalogFilter>({});
  const [filter, setFilter] = useState<Track2CatalogFilter>({});
  const destinationSaves = useRef<Promise<unknown>>(Promise.resolve());
  const destinationChanged = useRef(false);
  // Request counters: an answer to an older request (an earlier filter, a
  // page before a refresh) is dropped rather than drawn over a newer one.
  const runsRequest = useRef(0);
  const versionsRequest = useRef(0);
  function changeDestination(value: CatalogExportDestination) {
    destinationChanged.current = true;
    setDestination(value);
    destinationSaves.current = destinationSaves.current.catch(() => undefined)
      .then(() => saveTrack2PullConfig({ catalog_export_destination: value }))
      .catch(() => setError("Could not save the export destination. Select it again to retry."));
  }

  /** `append`: Load more. Otherwise the first page again -- as many rows as
   * were on screen, so a refresh does not drop records that were loaded. */
  function loadRuns(append = false, keep = 0) {
    const mine = ++runsRequest.current;
    const offset = append ? runs.rows.length : 0;
    const limit = append ? PAGE : Math.min(500, Math.max(PAGE, keep));
    setRuns((t) => ({ ...t, status: "loading", error: "" }));
    getTrack2CatalogRuns(offset, filter, limit)
      .then((r) => {
        if (mine !== runsRequest.current) return;
        setRuns((t) => {
          const rows = append ? [...t.rows, ...r.runs] : r.runs;
          return { status: "ready", rows, total: r.total, error: "", at: new Date().toISOString(), more: rows.length < r.total };
        });
      })
      .catch((err) => {
        if (mine !== runsRequest.current) return;
        setRuns((t) => ({ ...t, status: "failed", error: err instanceof ApiError ? err.detail : "Could not read the training runs." }));
      });
  }

  function loadVersions(append = false, keep = 0) {
    const mine = ++versionsRequest.current;
    const offset = append ? versions.rows.length : 0;
    const limit = append ? PAGE : Math.min(500, Math.max(PAGE, keep));
    setVersions((t) => ({ ...t, status: "loading", error: "" }));
    getTrack2CatalogVersions(offset, filter, limit)
      .then((r) => {
        if (mine !== versionsRequest.current) return;
        setVersions((t) => {
          const rows = append ? [...t.rows, ...r.versions] : r.versions;
          return { status: "ready", rows, total: r.total, error: "", at: new Date().toISOString(), more: rows.length < r.total };
        });
      })
      .catch((err) => {
        if (mine !== versionsRequest.current) return;
        setVersions((t) => ({ ...t, status: "failed", error: err instanceof ApiError ? err.detail : "Could not read the dataset versions." }));
      });
  }

  function loadStatus() {
    setStatusError("");
    getTrack2CatalogStatus().then(setStatus)
      .catch((err) => setStatusError(err instanceof ApiError ? err.detail : "Could not read the catalog status."));
  }

  useEffect(() => {
    getTrack2PullControl().then((value) => {
      const saved = value.config.catalog_export_destination;
      if (!destinationChanged.current && saved && ["local", "s3"].includes(saved.destination))
        setDestination({ destination: saved.destination, bucket: saved.bucket || "", prefix: saved.prefix || "", format: saved.destination === "local" ? "zip" : saved.format === "folder" ? "folder" : "zip" });
    }).catch(() => undefined);
  }, []);

  // A new filter: both tables from their first page.
  useEffect(() => {
    loadRuns(); loadVersions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter]);

  // Refresh all (from the page): status and both tables, keeping the
  // filter, the rows already loaded and the selection.
  const firstRefresh = useRef(true);
  useEffect(() => {
    if (firstRefresh.current) { firstRefresh.current = false; loadStatus(); return; }
    loadStatus(); loadRuns(false, runs.rows.length); loadVersions(false, versions.rows.length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [refreshKey]);

  // The selection follows its own row through a refresh, so the details
  // below reflect what was just read; it is never dropped.
  useEffect(() => {
    if (picked?.kind !== "run") return;
    const fresh = runs.rows.find((r) => r.run_id === picked.run.run_id);
    if (fresh && fresh !== picked.run) setPicked({ kind: "run", run: fresh });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runs.rows]);
  useEffect(() => {
    if (picked?.kind !== "version") return;
    const fresh = versions.rows.find((v) => v.id === picked.version.id);
    if (fresh && fresh !== picked.version) setPicked({ kind: "version", version: fresh });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [versions.rows]);

  // A picked run's dataset, for the export target's train/val counts.
  const pickedRunId = picked?.kind === "run" ? picked.run.run_id : "";
  const pickedRunVersion = picked?.kind === "run" ? picked.run.version_id : null;
  useEffect(() => {
    if (!pickedRunId) return;
    if (!pickedRunVersion) { setRunDataset({ runId: pickedRunId, version: null, failed: false }); return; }
    let cancelled = false;
    setRunDataset(null);
    getTrack2CatalogVersions(0, { search: pickedRunVersion }, 5)
      .then((r) => { if (!cancelled) setRunDataset({ runId: pickedRunId, version: r.versions.find((v) => v.id === pickedRunVersion) ?? null, failed: false }); })
      .catch(() => { if (!cancelled) setRunDataset({ runId: pickedRunId, version: null, failed: true }); });
    return () => { cancelled = true; };
  }, [pickedRunId, pickedRunVersion, refreshKey]);

  let selection: CatalogSelection | null = null;
  if (picked?.kind === "run") {
    const r = picked.run;
    const m = runModel(r);
    const ds = runDataset?.runId === r.run_id ? runDataset.version : null;
    selection = { kind: "run", id: r.run_id, label: `Run ${r.run_number}`, target: {
      runNumber: r.run_number, modelName: m.name, modelVersion: m.version,
      datasetLabel: r.version_number ? `${r.version_kind === "snapshot" ? "Snapshot" : "Version"} ${r.version_number}` : null,
      datasetId: r.version_id, train: ds?.train_images ?? null, val: ds?.val_images ?? null, images: r.images } };
  } else if (picked?.kind === "version") {
    const v = picked.version;
    selection = { kind: "version", id: v.id, label: `Version ${v.version_number}`, target: {
      datasetLabel: `${v.kind === "snapshot" ? "Snapshot" : "Version"} ${v.version_number}`,
      datasetId: v.id, train: v.train_images, val: v.val_images } };
  }

  const field = "rounded border border-black/15 bg-page px-2 py-1 text-xs text-ink";
  const applied = Object.values(filter).some((v) => v && v.trim());
  const tabButton = (value: Tab, label: string, table: Table<unknown>) => (
    <button
      type="button"
      onClick={() => setTab(value)}
      className={`px-3 py-1.5 text-sm font-medium ${
        tab === value ? "bg-accent text-white" : "bg-page text-ink-secondary hover:text-ink"}`}
    >
      {label}{table.at ? ` (${table.total.toLocaleString()})` : ""}
    </button>
  );

  /** Loading / Empty / Failed / Retry, said the same way for both tables. */
  function tableState(table: Table<unknown>, noun: string, retry: () => void) {
    if (table.status === "loading" && !table.rows.length) return <p className="text-sm text-ink-muted" role="status">Loading {noun}...</p>;
    if (table.status === "failed" && !table.rows.length) return (
      <div className="text-sm text-critical" role="alert">
        Could not load {noun}: {table.error}
        <button type="button" className="ml-3 rounded border border-hairline px-3 py-1 text-ink" onClick={retry}>Retry</button>
      </div>
    );
    if (table.status === "ready" && !table.rows.length) return (
      <p className="text-sm text-ink-muted">{applied ? `No ${noun} match these filters.` : `No ${noun} are recorded yet.`}</p>
    );
    return null;
  }

  function tableFooter(table: Table<unknown>, noun: string, retry: () => void, more: () => void) {
    return (
      <div className="mt-2 flex flex-wrap items-center gap-3 text-xs">
        <span className="tabular text-ink-muted">
          Showing {table.rows.length.toLocaleString()} of {table.total.toLocaleString()} {noun}{applied ? " matching the filters" : ""}
          {table.at && ` · read ${formatSgt(table.at)}`}
        </span>
        {table.more && <button type="button" className="rounded border border-hairline px-2.5 py-1" disabled={table.status === "loading"} onClick={more}>
          {table.status === "loading" ? "Loading..." : `Load ${Math.min(PAGE, table.total - table.rows.length)} more`}</button>}
        {table.status === "loading" && table.rows.length > 0 && <span className="text-ink-muted" role="status">Refreshing...</span>}
        {table.status === "failed" && table.rows.length > 0 && (
          <span className="text-critical" role="alert">
            The last read failed ({table.error}); the rows shown are from {formatSgt(table.at)}.{" "}
            <button type="button" className="underline" onClick={retry}>Retry</button>
          </span>
        )}
      </div>
    );
  }

  return (
    <div>
      <h2 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Library size={20} className="text-accent" strokeWidth={2} />
        Track2 Data Catalog
      </h2>
      <p className="mb-2 max-w-3xl text-sm text-ink-secondary">
        Every training run and every dataset version, with the files each used. Pick one to review
        its metadata, check its dataset can be rebuilt, or export it. Selecting a row here is
        independent of the versions compared or exported above.
      </p>
      {status?.record_backend === "postgres" && (
        <p className="mb-4 text-xs text-ink-muted">Records stored in PostgreSQL. Changes are recorded directly.</p>
      )}
      {status?.record_backend !== "postgres" && status?.last_sync?.finished_at && (
        <p className="mb-4 text-xs text-ink-muted">
          Records last copied from DynamoDB {when(status.last_sync.finished_at)}. They copy across
          every 5 minutes.
        </p>
      )}
      {statusError && (
        <p className="mb-3 text-xs text-critical" role="alert">{statusError} <button type="button" className="underline" onClick={loadStatus}>Retry</button></p>
      )}
      {error && (
        <p className="mb-3 flex items-start gap-1.5 text-sm text-critical">
          <AlertCircle size={15} className="mt-0.5 shrink-0" /> {error}
        </p>
      )}

      {/* Search: applied by the server to the whole catalog. Dates are
          calendar days in Singapore time, both ends included. */}
      <form className="mb-4 flex flex-wrap items-end gap-3 rounded border border-hairline bg-page p-3 text-xs"
            onSubmit={(e) => { e.preventDefault(); setFilter({ ...draft }); }} aria-label="Search the catalog">
        <label className="flex flex-col gap-1">Search
          <input className={`${field} w-64`} value={draft.search ?? ""} placeholder="run number or ID, model, dataset ID"
                 onChange={(e) => setDraft({ ...draft, search: e.target.value })} />
        </label>
        <label className="flex flex-col gap-1">{tab === "runs" ? "Run status" : "Version state"}
          <input className={`${field} w-32`} list={`catalog-status-${tab}`} value={draft.status ?? ""}
                 onChange={(e) => setDraft({ ...draft, status: e.target.value })} placeholder="any" />
          <datalist id={`catalog-status-${tab}`}>{(tab === "runs" ? RUN_STATUSES : VERSION_STATES).map((v) => <option key={v} value={v} />)}</datalist>
        </label>
        <label className="flex flex-col gap-1">From (SGT)
          <input type="date" className={field} value={draft.from_date ?? ""} onChange={(e) => setDraft({ ...draft, from_date: e.target.value })} />
        </label>
        <label className="flex flex-col gap-1">To (SGT, inclusive)
          <input type="date" className={field} value={draft.to_date ?? ""} onChange={(e) => setDraft({ ...draft, to_date: e.target.value })} />
        </label>
        <button type="submit" className="rounded bg-accent px-3 py-1.5 font-medium text-white">Apply filters</button>
        <button type="button" className="rounded border border-hairline px-3 py-1.5" disabled={!applied && !Object.values(draft).some(Boolean)}
                onClick={() => { setDraft({}); setFilter({}); }}>Clear</button>
        {applied && <span className="text-ink-muted">Filters apply to both tables.</span>}
      </form>

      <div className="mb-4 inline-flex overflow-hidden rounded-md border border-black/15">
        {tabButton("runs", "Training runs", runs)}
        {tabButton("versions", "Dataset versions", versions)}
      </div>

      <div className="flex flex-col gap-5">
        {tab === "runs" && (
          <Card title="Training runs" collapsible={false}
                caption="Newest first. Click a run to see its metadata and export options below.">
            {tableState(runs, "training runs", () => loadRuns(false, runs.rows.length)) ?? (
              <>
              <div className="max-h-[calc(22rem+2px)] overflow-auto rounded border border-hairline"
                   tabIndex={0} role="region" aria-label="Training runs; scroll for older runs">
                <table className="tabular min-w-full whitespace-nowrap text-xs [&_tr]:h-8">
                  <thead className="sticky top-0 bg-surface text-left text-ink">
                    <tr>
                      {["Run", "Submitted", "Status", "Model name", "Model type", "Size", "Model version",
                        "Dataset version", "Images", "Registered as"].map((h) => (
                        <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {runs.rows.map((r) => { const m = runModel(r); return (
                      <tr key={r.run_id} onClick={() => setPicked({ kind: "run", run: r })}
                          className={`cursor-pointer border-b border-hairline/50 hover:bg-black/[0.03] ${
                            picked?.kind === "run" && picked.run.run_id === r.run_id ? "bg-accent-subtle" : ""}`}>
                        <td className="px-2 py-1 font-medium text-ink"><button type="button" aria-pressed={picked?.kind === "run" && picked.run.run_id === r.run_id}
                          className="rounded px-1 underline focus-visible:outline focus-visible:outline-2" onClick={() => setPicked({ kind: "run", run: r })}>Run {r.run_number}{picked?.kind === "run" && picked.run.run_id === r.run_id ? " ✓" : ""}</button></td>
                        <td className="px-2 py-1">{when(r.submitted_at)}</td>
                        <td className="px-2 py-1">{r.status ?? "--"}</td>
                        <td className="px-2 py-1">{m.name}</td>
                        <td className="px-2 py-1">{m.type}</td>
                        <td className="px-2 py-1">{m.size}</td>
                        <td className="px-2 py-1 font-medium text-ink">{m.version}</td>
                        <td className="px-2 py-1">
                          {r.version_number ? `${r.version_kind === "snapshot" ? "Snapshot" : "Version"} ${r.version_number}` : "--"}
                        </td>
                        <td className="px-2 py-1">{r.images.toLocaleString()}</td>
                        <td className="px-2 py-1">{m.group}</td>
                      </tr>
                    ); })}
                  </tbody>
                </table>
              </div>
              {tableFooter(runs, "runs", () => loadRuns(false, runs.rows.length), () => loadRuns(true))}
              </>
            )}
          </Card>
        )}

        {tab === "versions" && (
          <Card title="Dataset versions and snapshots" collapsible={false}
                caption="Newest first. A version is frozen when a run is queued; a snapshot when a dataset is frozen by hand. Click one to see its metadata and export options.">
            {tableState(versions, "dataset versions", () => loadVersions(false, versions.rows.length)) ?? (
              <>
              <div className="max-h-[28rem] overflow-auto rounded border border-hairline">
                <table className="tabular min-w-full text-xs">
                  <thead className="sticky top-0 bg-surface text-left text-ink">
                    <tr>
                      {["Version", "Kind", "Created", "State", "Train", "Val", "Files", "Used by"].map((h) => (
                        <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {versions.rows.map((v) => (
                      <tr key={v.id} onClick={() => setPicked({ kind: "version", version: v })}
                          className={`cursor-pointer border-b border-hairline/50 hover:bg-black/[0.03] ${
                            picked?.kind === "version" && picked.version.id === v.id ? "bg-accent-subtle" : ""}`}>
                        <td className="px-2 py-1 font-medium text-ink" title={v.id}><button type="button" className="rounded px-1 underline focus-visible:outline focus-visible:outline-2"
                          aria-pressed={picked?.kind === "version" && picked.version.id === v.id} onClick={() => setPicked({ kind: "version", version: v })}>Version {v.version_number}{picked?.kind === "version" && picked.version.id === v.id ? " ✓" : ""}</button></td>
                        <td className="px-2 py-1">{v.kind}</td>
                        <td className="px-2 py-1">{when(v.created_at)}</td>
                        <td className={`px-2 py-1 ${v.state === "deleted" ? "text-critical" : ""}`}>{v.state ?? "--"}</td>
                        <td className="px-2 py-1">{v.kind === "version" ? (v.train_images ?? 0).toLocaleString() : "--"}</td>
                        <td className="px-2 py-1">{v.kind === "version" ? (v.val_images ?? 0).toLocaleString() : "--"}</td>
                        <td className="px-2 py-1">{v.members.toLocaleString()}</td>
                        <td className="px-2 py-1">{v.runs?.length ? v.runs.map((n) => `Run ${n}`).join(", ") : "--"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {tableFooter(versions, "versions", () => loadVersions(false, versions.rows.length), () => loadVersions(true))}
              </>
            )}
          </Card>
        )}

        {/* What is selected -- in the flow, NOT sticky: the page's pinned
            toolbar already holds the top edge, and two sticky bars there
            covered each other. */}
        <div className="rounded border border-hairline bg-surface p-3 text-sm" role="status">
          {picked?.kind === "run" ? <><strong>Selected: Run {picked.run.run_number}</strong> · {picked.run.architecture || "Model not recorded"} · Dataset {picked.run.version_number ?? "—"} · {picked.run.images.toLocaleString()} images</>
            : picked ? <><strong>Selected: Version {picked.version.version_number}</strong> · {picked.version.kind} · {picked.version.members.toLocaleString()} files</>
            : "No run or dataset version selected. Select a row above to begin."}
          {runDataset?.failed && picked?.kind === "run" && <span className="ml-2 text-xs text-critical">(its dataset&apos;s train/validation counts could not be read)</span>}
        </div>
        <Track2CatalogMetadata selection={selection} destination={destination} refreshKey={refreshKey}
          renderDestination={(showFormat) => <ExportDestination destination={destination} onDestination={changeDestination} showFormat={showFormat} />} />
      </div>
    </div>
  );
}

type DestinationProps = {
  destination: CatalogExportDestination;
  onDestination: (value: CatalogExportDestination) => void;
};

function ExportDestination({ destination, onDestination, showFormat = true }: DestinationProps & { showFormat?: boolean }) {
  const field = "rounded border border-black/15 bg-page px-2 py-1 text-xs text-ink";
  const [buckets, setBuckets] = useState<Track2Bucket[]>([]);
  const [prefix, setPrefix] = useState(destination.prefix);
  const [folders, setFolders] = useState<string[]>([]);
  const [bucketError, setBucketError] = useState("");
  const [folderError, setFolderError] = useState("");
  const [loading, setLoading] = useState(false);
  const [reload, setReload] = useState(0);
  useEffect(() => {
    if (destination.destination !== "s3") return;
    let cancelled = false;
    setLoading(true); setBucketError("");
    getTrack2Buckets().then((r) => { if (!cancelled) setBuckets(r.buckets); })
      .catch((err) => { if (!cancelled) setBucketError(err instanceof ApiError ? err.detail : "Could not list S3 buckets."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [destination.destination, reload]);
  useEffect(() => {
    setFolders([]); setFolderError("");
    if (destination.destination !== "s3" || !destination.bucket) return;
    let cancelled = false;
    listTrack2Folders(destination.bucket, prefix)
      .then((r) => { if (!cancelled) setFolders(r.folders); })
      .catch((err) => { if (!cancelled) setFolderError(err instanceof ApiError ? err.detail : "Could not list folders."); });
    return () => { cancelled = true; };
  }, [destination.destination, destination.bucket, prefix, reload]);
  return <div className="mb-3 rounded border border-hairline p-3 text-xs">
    <label className="flex items-center gap-2">Export actual files to
      <select aria-label="Export destination" className={field} value={destination.destination}
              onChange={(e) => onDestination({ ...destination, destination: e.target.value as "local" | "s3", format: e.target.value === "local" ? "zip" : destination.format })}>
        <option value="local">Local PC</option>
        <option value="s3">S3 bucket</option>
      </select>
    </label>
    {showFormat && <label className="mt-2 flex items-center gap-2">Export format
      <select aria-label="Export format" className={field} value={destination.format}
        onChange={(e) => onDestination({ ...destination, format: e.target.value as "zip" | "folder" })}>
        <option value="zip">ZIP archive</option>
        {destination.destination === "s3" && <option value="folder">Folder (individual files)</option>}
      </select>
    </label>}
    {destination.destination === "s3" && <div className="mt-2 space-y-2">
      <label>Destination bucket <select className={field} value={destination.bucket}
        onChange={(e) => { setPrefix(""); setFolders([]); onDestination({ ...destination, bucket: e.target.value, prefix: "" }); }}>
        <option value="">{loading ? "Loading buckets..." : "Choose a bucket"}</option>
        {buckets.map((b) => <option key={b.name} value={b.name}>{b.name}{b.readable ? "" : " (listing restricted)"}</option>)}
      </select></label>
      <button type="button" className="ml-2 underline" onClick={() => setReload((n) => n + 1)}>Refresh buckets and folders</button>
      {bucketError && <p className="text-critical">{bucketError}</p>}
      {destination.bucket && <>
        <p>Choose an existing folder, or enter a new folder name and press Create &amp; use.</p>
        <Track2S3FolderBrowser key={destination.bucket} prefix={prefix} folders={folders}
          onNavigate={setPrefix} selected={destination.prefix} allowCreate
          onSelect={(value) => onDestination({ ...destination, prefix: value })} />
        {folderError && <p className="text-critical">{folderError}</p>}
        <p className="break-all">Destination: s3://{destination.bucket}/{destination.prefix ? `${destination.prefix}/` : ""}</p>
      </>}
    </div>}
    <p className="mt-2 text-ink-muted">{destination.destination === "local" ? "Your browser will download the selected output." : "Files are written to a new subfolder in the selected S3 destination."}</p>
  </div>;
}
