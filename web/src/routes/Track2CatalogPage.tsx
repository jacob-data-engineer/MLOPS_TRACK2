import { AlertCircle, Archive, Download, FileText, Library, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
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
  startTrack2CatalogExport,
  stopTrack2CatalogExport,
  exportTrack2CatalogWeights,
  getTrack2CatalogMetadata,
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

  function loadRuns() {
    setRunsLoading(true);
    setError("");
    getTrack2CatalogRuns().then((r) => setRuns(r.runs))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not read the runs."))
      .finally(() => setRunsLoading(false));
  }

  useEffect(() => {
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
      onClick={() => setTab(value)}
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
      {status?.last_sync?.finished_at && (
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
                      {["Run", "Submitted", "Status", "Model", "Dataset version", "Images", "Registered as"].map((h) => (
                        <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {runs.map((r) => (
                      <tr key={r.run_id} onClick={() => setPicked({ kind: "run", run: r })}
                          className={`cursor-pointer border-b border-hairline/50 hover:bg-black/[0.03] ${
                            picked?.kind === "run" && picked.run.run_id === r.run_id ? "bg-accent-subtle" : ""}`}>
                        <td className="px-2 py-1 font-medium text-ink">Run {r.run_number}</td>
                        <td className="px-2 py-1">{when(r.submitted_at)}</td>
                        <td className="px-2 py-1">{r.status ?? "--"}</td>
                        <td className="px-2 py-1">{[r.detector, r.architecture].filter(Boolean).join(" / ") || "--"}</td>
                        <td className="px-2 py-1">
                          {r.version_number ? `${r.version_kind === "snapshot" ? "Snapshot" : "Version"} ${r.version_number}` : "--"}
                        </td>
                        <td className="px-2 py-1">{r.images.toLocaleString()}</td>
                        <td className="px-2 py-1">
                          {r.model_package_arn ? r.model_package_arn.split("/").slice(-2).join(" v") : "--"}
                        </td>
                      </tr>
                    ))}
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
                        <td className="px-2 py-1 font-medium text-ink" title={v.id}>Version {v.version_number}</td>
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

        {tab === "exports" && (
          <ExportsCard exportsList={exportsList} destination={destination} onDestination={setDestination} onExported={refreshExports} />
        )}

        {tab !== "exports" && (
          <FilesCard key={picked?.kind === "run" && tab === "runs" ? picked.run.run_id
            : picked?.kind === "version" && tab === "versions" ? picked.version.id : `empty-${tab}`}
                     picked={picked?.kind === (tab === "runs" ? "run" : "version") ? picked : null}
                     emptyMessage={tab === "runs" ? "No run selected" : "No dataset version selected"}
                     destination={destination} onDestination={setDestination}
                     onExported={() => { refreshExports(); setTab("exports"); }} />
        )}
        <Track2CatalogMetadata
          selection={picked && (tab === "exports" || picked.kind === (tab === "runs" ? "run" : "version"))
            ? picked.kind === "run" ? { kind: "run", id: picked.run.run_id, label: `Run ${picked.run.run_number}` }
              : { kind: "version", id: picked.version.id, label: `Version ${picked.version.version_number}` } : null}
          destination={destination} onExported={() => { refreshExports(); setTab("exports"); }} />
      </div>
    </div>
  );
}

type DestinationProps = {
  destination: CatalogExportDestination;
  onDestination: (value: CatalogExportDestination) => void;
};

function ExportDestination({ destination, onDestination }: DestinationProps) {
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
        <option value="local">Local PC (download ZIP)</option>
        <option value="s3">S3 bucket</option>
      </select>
    </label>
    <label className="mt-2 flex items-center gap-2">Export format
      <select aria-label="Export format" className={field} value={destination.format}
        onChange={(e) => onDestination({ ...destination, format: e.target.value as "zip" | "folder" })}>
        <option value="zip">ZIP archive (images and labels)</option>
        {destination.destination === "s3" && <option value="folder">Folder (individual images and labels)</option>}
      </select>
    </label>
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
    <p className="mt-2 text-ink-muted">{destination.destination === "local"
      ? "Builds a ZIP containing the images, labels and metadata. When ready, use Download ZIP; your browser chooses where to save it."
      : destination.format === "zip"
        ? "Writes a ZIP containing the actual images, labels and metadata to the selected S3 folder."
        : "Copies images, labels and metadata as individual objects into a new export subfolder. New folders are created when files are written. The app needs write access to this bucket."}</p>
  </div>;
}

function FilesCard({ picked, emptyMessage, onExported, destination, onDestination }: {
  picked: Picked | null; emptyMessage: string; onExported: () => void;
} & DestinationProps) {
  const [split, setSplit] = useState("");
  const [role, setRole] = useState("");
  const [offset, setOffset] = useState(0);
  const [files, setFiles] = useState<Track2CatalogFile[] | null>(null);
  const [total, setTotal] = useState(0);
  const [thumbs, setThumbs] = useState(false);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [weightsMessage, setWeightsMessage] = useState("");
  const [metadataOptions, setMetadataOptions] = useState<Record<string, unknown> | null>(null);
  const [metadataSelected, setMetadataSelected] = useState<string[]>([]);
  const [metadataReady, setMetadataReady] = useState(false);
  useEffect(() => {
    let active = true;
    setMetadataReady(false);
    if (picked?.kind !== "run") return;
    Promise.all([getTrack2CatalogMetadata("run", picked.run.run_id), getTrack2PullControl()])
      .then(([info, settings]) => {
        if (!active) return;
        setMetadataOptions(info.groups);
        const available = Object.keys(info.groups).filter((name) => !info.unavailable.includes(name));
        setMetadataSelected((settings.config.catalog_pt_metadata?.selected_groups ?? available).filter((name) => available.includes(name)));
        setMetadataReady(true);
      }).catch((err) => { if (active) setError(err instanceof ApiError ? err.detail : "Could not load metadata choices. Refresh to retry."); });
    return () => { active = false; };
  }, [picked?.kind === "run" ? picked.run.run_id : ""]);

  async function selectMetadata(names: string[]) {
    setBusy("metadata"); setError("");
    try {
      await saveTrack2PullConfig({ catalog_pt_metadata: { selected_groups: names } });
      setMetadataSelected(names);
    } catch (err) { setError(err instanceof ApiError ? err.detail : "Could not save metadata choices."); }
    finally { setBusy(""); }
  }

  async function exportWeights(target: "local" | "s3") {
    if (picked?.kind !== "run") return;
    setBusy("pt"); setError(""); setWeightsMessage("");
    try {
      const result = await exportTrack2CatalogWeights(picked.run.run_id, {
        destination: target, bucket: destination.bucket, prefix: destination.prefix,
        metadata_groups: metadataSelected,
      });
      if (result.url) {
        const link = document.createElement("a");
        link.href = result.url;
        document.body.appendChild(link); link.click(); link.remove();
      }
      if (result.uri) setWeightsMessage(`Model exported to ${result.uri}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not export the .pt file.");
    } finally { setBusy(""); }
  }

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

  function exportZip() {
    if (!picked) return;
    setBusy("zip");
    setError("");
    startTrack2CatalogExport(picked.kind, id, destination)
      .then(onExported)
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not start the export."))
      .finally(() => setBusy(""));
  }

  const select = "rounded border border-black/15 bg-page px-2 py-1 text-xs text-ink";

  return (
    <Card title={picked ? `Files in ${title}` : "Files and exports"} collapsible={false}
          caption={!picked ? `${emptyMessage}. Select a row above to view and export its files.` : picked.kind === "run"
            ? "The images this run trained and evaluated on, as the training audit recorded them."
            : "The images in this version, with the split each was given."}>
      <fieldset disabled={!picked} className={!picked ? "opacity-50" : undefined}>
      <ExportDestination destination={destination} onDestination={onDestination} />
      {picked?.kind === "run" && <div className="mb-3 rounded border border-hairline p-3 text-xs">
        <p className="mb-2 font-medium">Trained model (.pt) — Run {picked.run.run_number}</p>
        <p className="mb-2 text-ink-muted">Export this run’s published model checkpoint. Images and labels use the ZIP or folder export below.</p>
        <p className="mb-2 font-medium">Choose metadata to add to the .pt file</p>
        <div className="mb-2 flex gap-3">
          <button type="button" className="underline disabled:opacity-40" disabled={!metadataReady || !!busy}
            onClick={() => void selectMetadata(Object.entries(metadataOptions || {}).filter(([, value]) => value && typeof value === "object" && Object.keys(value).length > 0).map(([name]) => name))}>Select all available</button>
          <button type="button" className="underline disabled:opacity-40" disabled={!metadataReady || !!busy} onClick={() => void selectMetadata([])}>Clear all</button>
          <span>{metadataSelected.length} selected</span>
        </div>
        <div className="mb-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(metadataOptions || {}).map(([name, value]) => {
            const available = !!value && typeof value === "object" && Object.keys(value).length > 0;
            return <label key={name} className={`flex items-start gap-2 ${available ? "" : "text-ink-muted"}`}>
              <input type="checkbox" checked={metadataSelected.includes(name)} disabled={!available || !!busy || !metadataReady}
                onChange={(e) => void selectMetadata(e.target.checked ? [...metadataSelected, name] : metadataSelected.filter((key) => key !== name))} />
              <span>{name}{!available ? " — Not recorded" : ""}</span>
            </label>;
          })}
        </div>
        {!metadataReady && <p className="mb-2">Loading metadata choices…</p>}
        <p className="mb-3 text-ink-muted">Selections are saved. Checked categories are embedded in a new model copy. Clear all exports the original checkpoint; existing model contents are preserved. These choices apply to the .pt buttons here.</p>
        <div className="flex flex-wrap items-center gap-3">
          <button type="button" className="rounded border border-black/15 bg-page px-3 py-1 font-medium disabled:opacity-40"
            disabled={!!busy || !metadataReady} onClick={() => void exportWeights("local")}>
            {busy === "pt" ? "Exporting…" : "Download .pt"}
          </button>
          {destination.destination === "s3" ? <button type="button"
            className="rounded border border-black/15 bg-page px-3 py-1 font-medium disabled:opacity-40"
            disabled={!!busy || !metadataReady || !destination.bucket.trim()} onClick={() => void exportWeights("s3")}>
            Export .pt to selected S3 folder
          </button> : <span className="text-ink-muted">Choose S3 bucket above to copy the model to S3.</span>}
        </div>
        {weightsMessage && <p role="status" className="mt-2 break-all">{weightsMessage}</p>}
      </div>}
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
        <button type="button" onClick={exportZip} disabled={!!busy || (destination.destination === "s3" && !destination.bucket.trim())}
                className="inline-flex items-center gap-1.5 rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40">
          {busy === "zip" ? <Loader2 size={13} className="animate-spin" /> : <Archive size={13} />}
          {destination.destination === "s3" ? (destination.format === "zip" ? "Export ZIP to S3" : "Copy files to S3 folder") : "Prepare ZIP for PC"}
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

function ExportsCard({ exportsList, destination, onDestination, onExported }: {
  exportsList: Track2CatalogExport[] | null; onExported: () => void;
} & DestinationProps) {
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function exportAgain(e: Track2CatalogExport) {
    const id = e.run_id || e.version_id;
    if (!id) return;
    setError(""); setBusy(true);
    try {
      await startTrack2CatalogExport(e.run_id ? "run" : "version", id, destination);
      onExported();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not start the export.");
    } finally { setBusy(false); }
  }

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
    <Card title="Exports" collapsible={false}
          caption="Export actual images and labels to your PC as a ZIP, or copy them to S3. Missing source files are reported as incomplete exports. This list refreshes while an export is building.">
      <ExportDestination destination={destination} onDestination={onDestination} />
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
                    {e.state !== "building" && e.state !== "stopping" && (e.run_id || e.version_id) && <button type="button" className="mt-1 block underline disabled:opacity-40"
                      disabled={busy || (destination.destination === "s3" && !destination.bucket.trim())}
                      onClick={() => void exportAgain(e)}>
                      {destination.destination === "s3" ? (destination.format === "zip" ? "Export ZIP to S3" : "Copy files to S3 folder") : "Prepare new ZIP"}
                    </button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
    </Card>
  );
}
