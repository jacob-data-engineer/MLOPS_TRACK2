import { AlertCircle, Download, FileText, Library, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Card } from "../components/Card";
import { Track2CatalogMetadata } from "../components/Track2CatalogMetadata";
import { Track2S3FolderBrowser } from "../components/Track2S3FolderBrowser";
import {
  ApiError,
  getTrack2Buckets,
  listTrack2Folders,
  type Track2Bucket,
  downloadTrack2CatalogCsv,
  getTrack2CatalogExports,
  getTrack2CatalogExportUrl,
  getTrack2CatalogRunFiles,
  getTrack2CatalogRuns,
  getTrack2CatalogStatus,
  getTrack2CatalogVersionFiles,
  getTrack2CatalogVersions,
  getTrack2SqlPreviews,
  stopTrack2CatalogExport,
  getTrack2PullControl,
  saveTrack2PullConfig,
  type Track2CatalogExport,
  type Track2CatalogFile,
  type Track2CatalogRun,
  type Track2CatalogStatus,
  type Track2CatalogVersion,
  type CatalogExportDestination,
} from "../lib/api";
import { formatDayTime } from "../lib/format";

/** TRACK2 DATA CATALOG (2026-10-29).
 *
 * Every training run (by run number) and every dataset version and
 * snapshot (by version number), read from the catalog database -- the
 * files each one used, and exports of either: the file list as CSV, or the
 * files themselves as a zip in the training layout, built inside AWS.
 *
 * Read only. Nothing here changes a run, a version or a file.
 */
type Tab = "runs" | "versions" | "exports";
type Picked = { kind: "run"; run: Track2CatalogRun } | { kind: "version"; version: Track2CatalogVersion };

const when = (iso: string | null | undefined) => (iso ? formatDayTime(iso) : "--");

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
const shortName = (uri: string) => uri.split("/").slice(-1)[0];

function bytes(n: number | null | undefined): string {
  if (!n) return "--";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = n;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function Track2CatalogPage() {
  const [status, setStatus] = useState<Track2CatalogStatus | null>(null);
  const [tab, setTab] = useState<Tab>("runs");
  const [runs, setRuns] = useState<Track2CatalogRun[] | null>(null);
  const [versions, setVersions] = useState<Track2CatalogVersion[] | null>(null);
  const [exportsList, setExportsList] = useState<Track2CatalogExport[] | null>(null);
  const [picked, setPicked] = useState<Picked | null>(null);
  const [destination, setDestination] = useState<CatalogExportDestination>({ destination: "local", bucket: "", prefix: "", format: "zip" });
  const [error, setError] = useState("");
  const [runsLoading, setRunsLoading] = useState(true);
  const destinationSaves = useRef<Promise<unknown>>(Promise.resolve());
  const destinationChanged = useRef(false);
  function changeDestination(value: CatalogExportDestination) {
    destinationChanged.current = true;
    setDestination(value);
    destinationSaves.current = destinationSaves.current.catch(() => undefined)
      .then(() => saveTrack2PullConfig({ catalog_export_destination: value }))
      .catch(() => setError("Could not save the export destination. Select it again to retry."));
  }

  function loadRuns() {
    setRunsLoading(true);
    setError("");
    getTrack2CatalogRuns().then((r) => setRuns(r.runs))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not read the runs."))
      .finally(() => setRunsLoading(false));
  }

  useEffect(() => {
    getTrack2PullControl().then((value) => {
      const saved = value.config.catalog_export_destination;
      if (!destinationChanged.current && saved && ["local", "s3"].includes(saved.destination))
        setDestination({ destination: saved.destination, bucket: saved.bucket || "", prefix: saved.prefix || "", format: saved.destination === "local" ? "zip" : saved.format === "folder" ? "folder" : "zip" });
    }).catch(() => undefined);
    getTrack2CatalogStatus().then(setStatus).catch(() => setStatus(null));
    loadRuns();
    getTrack2CatalogVersions().then((r) => setVersions(r.versions)).catch(() => setVersions([]));
    getTrack2CatalogExports().then((r) => setExportsList(r.exports)).catch(() => setExportsList([]));
  }, []);

  // While an export is building, look again every 10 s.
  const building = exportsList?.some((e) => e.state === "building" || e.state === "stopping");
  useEffect(() => {
    if (!building) return;
    const timer = window.setInterval(() => {
      getTrack2CatalogExports().then((r) => setExportsList(r.exports)).catch(() => undefined);
    }, 10000);
    return () => window.clearInterval(timer);
  }, [building]);

  function refreshExports() {
    getTrack2CatalogExports().then((r) => setExportsList(r.exports)).catch(() => undefined);
  }

  const tabButton = (value: Tab, label: string, count?: number) => (
    <button
      type="button"
      onClick={() => { setTab(value); if (value === "exports") setTimeout(() => document.getElementById("catalog-export-history")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0); }}
      className={`px-3 py-1.5 text-sm font-medium ${
        tab === value ? "bg-accent text-white" : "bg-page text-ink-secondary hover:text-ink"}`}
    >
      {label}{count !== undefined ? ` (${count.toLocaleString()})` : ""}
    </button>
  );

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Library size={20} className="text-accent" strokeWidth={2} />
        Track2 Data Catalog
      </h1>
      <p className="mb-2 max-w-3xl text-sm text-ink-secondary">
        Every training run and every dataset version, with the files each used. Pick one to see
        its files, download the list, or export the files themselves as a zip.
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
      {error && (
        <p className="mb-3 flex items-start gap-1.5 text-sm text-critical">
          <AlertCircle size={15} className="mt-0.5 shrink-0" /> {error}
        </p>
      )}

      <div className="mb-4 inline-flex overflow-hidden rounded-md border border-black/15">
        {tabButton("runs", "Training runs", runs?.length)}
        {tabButton("versions", "Dataset versions", versions?.length)}
        {tabButton("exports", "Exports", exportsList?.length)}
      </div>

      <div className="flex flex-col gap-5">
        {tab === "runs" && (
          <Card title="Training runs" collapsible={false}
                caption="Latest 10 runs shown first. Scroll inside the table for older runs. Click a run to see its files and export options below.">
            {runsLoading ? <p className="text-sm text-ink-muted">Reading...</p> : runs === null ? (
              <div className="text-sm text-ink-secondary">
                Could not load training runs.
                <button type="button" className="ml-3 rounded border border-hairline px-3 py-1"
                        onClick={loadRuns}>Retry</button>
              </div>
            ) : (
              <div className="max-h-[calc(22rem+2px)] overflow-auto rounded border border-hairline"
                   tabIndex={0} role="region" aria-label="Training runs; scroll for older runs">
                {/* One 2rem header plus ten 2rem rows; older runs stay in this scroll area. */}
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
                    {runs.map((r) => { const m = runModel(r); return (
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
            )}
          </Card>
        )}

        {tab === "versions" && (
          <Card title="Dataset versions and snapshots" collapsible={false}
                caption="Newest first. A version is frozen when a run is queued; a snapshot when a dataset is frozen by hand. Click one to see its files.">
            {versions === null ? <p className="text-sm text-ink-muted">Reading...</p> : (
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
                    {versions.map((v) => (
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
            )}
          </Card>
        )}

        <div className="sticky top-0 z-10 rounded border border-hairline bg-surface p-3 shadow-sm" role="status">
          {picked?.kind === "run" ? <><strong>Run {picked.run.run_number}</strong> · {picked.run.architecture || "Model not recorded"} · Dataset {picked.run.version_number ?? "—"} · {picked.run.images.toLocaleString()} images</>
            : picked ? <><strong>Version {picked.version.version_number}</strong> · {picked.version.kind} · {picked.version.members.toLocaleString()} files</>
            : "No run or dataset version selected. Select a row above to begin."}
        </div>
        <Track2CatalogMetadata
          selection={picked ? picked.kind === "run" ? { kind: "run", id: picked.run.run_id, label: `Run ${picked.run.run_number}` }
            : { kind: "version", id: picked.version.id, label: `Version ${picked.version.version_number}` } : null}
          destination={destination} onExported={refreshExports}
          renderDestination={(showFormat) => <ExportDestination destination={destination} onDestination={changeDestination} showFormat={showFormat} />}>
          <FilesCard key={picked?.kind === "run" ? picked.run.run_id : picked?.version.id || "empty"}
            picked={picked} emptyMessage="No run or dataset version selected" />
          <ExportsCard exportsList={exportsList} onExported={refreshExports} />
        </Track2CatalogMetadata>
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

function FilesCard({ picked, emptyMessage }: { picked: Picked | null; emptyMessage: string }) {
  const [split, setSplit] = useState("");
  const [role, setRole] = useState("");
  const [offset, setOffset] = useState(0);
  const [files, setFiles] = useState<Track2CatalogFile[] | null>(null);
  const [total, setTotal] = useState(0);
  const [thumbs, setThumbs] = useState(false);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const id = picked?.kind === "run" ? picked.run.run_id : picked?.version.id ?? "";
  const title = picked?.kind === "run"
    ? `Run ${picked.run.run_number}`
    : picked ? `Version ${picked.version.version_number} (${picked.version.kind})` : emptyMessage;

  useEffect(() => {
    if (!picked) { setFiles([]); return; }
    setFiles(null);
    const load = picked.kind === "run"
      ? getTrack2CatalogRunFiles(id, { split, role, offset })
      : getTrack2CatalogVersionFiles(id, { split, offset });
    load.then((r) => { setFiles(r.files); setTotal(r.total); })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not read the files."));
  }, [id, split, role, offset]);

  // Thumbnails for the page on screen, signed for 15 minutes.
  useEffect(() => {
    if (!thumbs || !files?.length) return;
    getTrack2SqlPreviews(files.map((f) => f.uri))
      .then((r) => setPreviews(Object.fromEntries(r.previews.map((p) => [p.uri, p.url]))))
      .catch(() => undefined);
  }, [thumbs, files]);

  async function downloadCsv() {
    if (!picked) return;
    setBusy("csv");
    setError("");
    try {
      const blob = await downloadTrack2CatalogCsv(picked.kind, id);
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${title.replace(/\W+/g, "-").toLowerCase()}-files.csv`;
      link.click();
      URL.revokeObjectURL(link.href);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Download failed.");
    } finally {
      setBusy("");
    }
  }

  const select = "rounded border border-black/15 bg-page px-2 py-1 text-xs text-ink";

  return (
    <Card title={picked ? `Files in ${title}` : "Files and exports"} collapsible={false}
          caption={!picked ? `${emptyMessage}. Select a row above to view and export its files.` : picked.kind === "run"
            ? "The images this run trained and evaluated on, as the training audit recorded them."
            : "The images in this version, with the split each was given."}>
      <fieldset disabled={!picked} className={!picked ? "opacity-50" : undefined}>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
        <select className={select} value={split} onChange={(e) => { setSplit(e.target.value); setOffset(0); }}>
          <option value="">All splits</option>
          <option value="train">train</option>
          <option value="val">val</option>
        </select>
        {picked?.kind === "run" && (
          <select className={select} value={role} onChange={(e) => { setRole(e.target.value); setOffset(0); }}>
            <option value="">Training and evaluation</option>
            <option value="train">Training only</option>
            <option value="eval">Evaluation only</option>
          </select>
        )}
        <label className="flex items-center gap-1.5 text-ink">
          <input type="checkbox" checked={thumbs} onChange={(e) => setThumbs(e.target.checked)} />
          Show thumbnails
        </label>
        <span className="text-ink-muted">{total.toLocaleString()} image(s)</span>
        <span className="flex-1" />
        <button type="button" onClick={() => void downloadCsv()} disabled={!!busy}
                className="inline-flex items-center gap-1.5 rounded-md border border-black/15 bg-page px-2.5 py-1 font-medium text-ink disabled:opacity-40">
          {busy === "csv" ? <Loader2 size={13} className="animate-spin" /> : <FileText size={13} />}
          File paths only (CSV)
        </button>

      </div>
      </fieldset>
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}

      {!picked ? <p role="status" className="rounded border border-hairline p-6 text-center text-sm text-ink-muted">{emptyMessage}</p>
      : files === null ? <p className="text-sm text-ink-muted">Reading...</p> : thumbs ? (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-6">
          {files.map((f) => (
            <a key={f.uri} href={previews[f.uri]} target="_blank" rel="noreferrer" title={f.uri}
               className="block overflow-hidden rounded border border-hairline bg-black/80">
              {previews[f.uri]
                ? <img src={previews[f.uri]} alt="" loading="lazy" className="aspect-video w-full object-contain" />
                : <div className="aspect-video" />}
              <p className="truncate bg-surface px-1 text-[10px] text-ink-muted">{f.split} &middot; {shortName(f.uri)}</p>
            </a>
          ))}
        </div>
      ) : (
        <div className="max-h-[28rem] overflow-auto rounded border border-hairline">
          <table className="tabular min-w-full text-xs">
            <thead className="sticky top-0 bg-surface text-left text-ink">
              <tr>
                <th className="border-b border-hairline px-2 py-1.5 font-medium">File</th>
                <th className="border-b border-hairline px-2 py-1.5 font-medium">Split</th>
                {picked.kind === "run"
                  ? <th className="border-b border-hairline px-2 py-1.5 font-medium">Used for</th>
                  : <th className="border-b border-hairline px-2 py-1.5 font-medium">Week</th>}
                <th className="border-b border-hairline px-2 py-1.5 font-medium">Where it is</th>
              </tr>
            </thead>
            <tbody>
              {files.map((f) => (
                <tr key={f.uri} className="border-b border-hairline/50">
                  <td className="px-2 py-1 text-ink">{shortName(f.uri)}</td>
                  <td className="px-2 py-1">{f.split ?? "--"}</td>
                  <td className="px-2 py-1">{picked.kind === "run" ? (f.role === "eval" ? "evaluation" : "training") : (f.week ?? "--")}</td>
                  <td className="max-w-md truncate px-2 py-1 text-ink-muted" title={f.uri}>{f.uri}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {total > 100 && (
        <div className="mt-2 flex items-center gap-3 text-xs">
          <button type="button" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - 100))}
                  className="underline disabled:opacity-40">Previous</button>
          <span className="text-ink-muted">{(offset + 1).toLocaleString()}&ndash;{Math.min(offset + 100, total).toLocaleString()} of {total.toLocaleString()}</span>
          <button type="button" disabled={offset + 100 >= total} onClick={() => setOffset(offset + 100)}
                  className="underline disabled:opacity-40">Next</button>
        </div>
      )}
    </Card>
  );
}

function ExportsCard({ exportsList, onExported }: {
  exportsList: Track2CatalogExport[] | null; onExported: () => void;
}) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function stop(id: string) {
    setError(""); setBusy(true);
    try {
      await stopTrack2CatalogExport(id);
      onExported();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not stop the export.");
    } finally { setBusy(false); }
  }

  function download(id: string) {
    setError("");
    getTrack2CatalogExportUrl(id)
      .then((r) => { window.location.href = r.url; })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not get the link."));
  }

  return (
    <div id="catalog-export-history" className="scroll-mt-20"><Card title="Export history" collapsible={false}
          caption="Export actual images and labels to your PC as a ZIP, or copy them to S3. Missing source files are reported as incomplete exports. This list refreshes while an export is building.">
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {exportsList === null ? <p className="text-sm text-ink-muted">Reading...</p>
        : exportsList.length === 0 ? (
          <p className="text-sm text-ink-muted">No exports yet. Pick a run or a version, choose a destination, and export its files.</p>
        ) : (
          <table className="tabular min-w-full text-xs">
            <thead className="text-left text-ink">
              <tr>
                {["Export", "State", "Started", "Finished", "Files", "Size", "Format", "Destination", ""].map((h) => (
                  <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {exportsList.map((e) => (
                <tr key={e.id} className="border-b border-hairline/50">
                  <td className="px-2 py-1 font-medium text-ink">{e.label}</td>
                  <td className={`px-2 py-1 ${e.state === "failed" || e.missing > 0 ? "text-critical" : ""}`} title={e.error ?? ""}>
                    {e.state === "building" || e.state === "stopping"
                      ? <span className="inline-flex items-center gap-1"><Loader2 size={12} className="animate-spin" />{e.state === "stopping" ? "Stopping…" : "building"}</span>
                      : e.state === "cancelled" ? "Stopped"
                      : e.state === "partial" || (e.state === "ready" && e.missing > 0) ? "Incomplete" : e.state}
                    {e.state === "failed" && e.error ? ` -- ${e.error}` : ""}
                  </td>
                  <td className="px-2 py-1">{when(e.created_at)}</td>
                  <td className="px-2 py-1">{when(e.finished_at)}</td>
                  <td className="px-2 py-1">
                    {(e.files ?? 0).toLocaleString()}
                    {e.missing ? <span className="text-critical"> ({e.missing} missing)</span> : ""}
                  </td>
                  <td className="px-2 py-1">{bytes(e.bytes)}</td>
                  <td className="px-2 py-1">{e.format === "folder" ? "Folder" : "ZIP"}</td>
                  <td className="max-w-xs break-all px-2 py-1">
                    {e.destination === "s3" ? e.destination_uri : "Local PC (ZIP)"}
                    {e.destination === "s3" && e.destination_uri && <button type="button" className="ml-2 underline"
                      onClick={() => {
                        if (!navigator.clipboard) { setError("Select and copy the S3 address shown in the table."); return; }
                        void navigator.clipboard.writeText(e.destination_uri!).catch(() => setError("Select and copy the S3 address shown in the table."));
                      }}>Copy address</button>}
                  </td>
                  <td className="px-2 py-1">
                    {e.format !== "folder" && (e.state === "ready" || e.state === "partial") && (
                      <button type="button" onClick={() => download(e.id)}
                              className="inline-flex items-center gap-1 rounded-md border border-black/15 bg-page px-2 py-0.5 font-medium text-ink">
                        <Download size={12} /> {e.missing ? "Download incomplete ZIP" : "Download ZIP"}
                      </button>
                    )}
                    {(e.state === "building" || e.state === "stopping") && <button type="button"
                      className="rounded-md border border-black/15 bg-page px-2 py-1 font-medium disabled:opacity-40"
                      disabled={busy || e.state === "stopping"} onClick={() => void stop(e.id)}>
                      {e.state === "stopping" ? "Stopping…" : "Stop export"}
                    </button>}

                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
    </Card></div>
  );
}
