import { AlertCircle, Archive, Download, FileText, Library, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "../components/Card";
import {
  ApiError,
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
  type Track2CatalogExport,
  type Track2CatalogFile,
  type Track2CatalogRun,
  type Track2CatalogStatus,
  type Track2CatalogVersion,
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
  const [error, setError] = useState("");

  useEffect(() => {
    getTrack2CatalogStatus().then(setStatus).catch(() => setStatus(null));
    getTrack2CatalogRuns().then((r) => setRuns(r.runs))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not read the runs."));
    getTrack2CatalogVersions().then((r) => setVersions(r.versions)).catch(() => setVersions([]));
    getTrack2CatalogExports().then((r) => setExportsList(r.exports)).catch(() => setExportsList([]));
  }, []);

  // While an export is building, look again every 10 s.
  const building = exportsList?.some((e) => e.state === "building");
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
                caption="Newest first. Click a run to see the files it trained and evaluated on.">
            {runs === null ? <p className="text-sm text-ink-muted">Reading...</p> : (
              <div className="max-h-[28rem] overflow-auto rounded border border-hairline">
                <table className="tabular min-w-full text-xs">
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
          <ExportsCard exportsList={exportsList} />
        )}

        {picked && tab !== "exports" && (
          <FilesCard key={picked.kind === "run" ? picked.run.run_id : picked.version.id}
                     picked={picked}
                     onExported={() => { refreshExports(); setTab("exports"); }} />
        )}
      </div>
    </div>
  );
}

function FilesCard({ picked, onExported }: { picked: Picked; onExported: () => void }) {
  const [split, setSplit] = useState("");
  const [role, setRole] = useState("");
  const [offset, setOffset] = useState(0);
  const [files, setFiles] = useState<Track2CatalogFile[] | null>(null);
  const [total, setTotal] = useState(0);
  const [thumbs, setThumbs] = useState(false);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");

  const id = picked.kind === "run" ? picked.run.run_id : picked.version.id;
  const title = picked.kind === "run"
    ? `Run ${picked.run.run_number}`
    : `Version ${picked.version.version_number} (${picked.version.kind})`;

  useEffect(() => {
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
    setBusy("zip");
    setError("");
    startTrack2CatalogExport(picked.kind, id)
      .then(onExported)
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not start the export."))
      .finally(() => setBusy(""));
  }

  const select = "rounded border border-black/15 bg-page px-2 py-1 text-xs text-ink";

  return (
    <Card title={`Files in ${title}`} collapsible={false}
          caption={picked.kind === "run"
            ? "The images this run trained and evaluated on, as the training audit recorded them."
            : "The images in this version, with the split each was given."}>
      <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
        <select className={select} value={split} onChange={(e) => { setSplit(e.target.value); setOffset(0); }}>
          <option value="">All splits</option>
          <option value="train">train</option>
          <option value="val">val</option>
        </select>
        {picked.kind === "run" && (
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
          Download file list (CSV)
        </button>
        <button type="button" onClick={exportZip} disabled={!!busy}
                className="inline-flex items-center gap-1.5 rounded-md bg-accent px-2.5 py-1 font-medium text-white disabled:opacity-40">
          {busy === "zip" ? <Loader2 size={13} className="animate-spin" /> : <Archive size={13} />}
          Export as zip
        </button>
      </div>
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}

      {files === null ? <p className="text-sm text-ink-muted">Reading...</p> : thumbs ? (
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

function ExportsCard({ exportsList }: { exportsList: Track2CatalogExport[] | null }) {
  const [error, setError] = useState("");

  function download(id: string) {
    setError("");
    getTrack2CatalogExportUrl(id)
      .then((r) => { window.location.href = r.url; })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not get the link."));
  }

  return (
    <Card title="Exports" collapsible={false}
          caption="Zips of a run or a dataset version, in the training layout with a data.yaml and a manifest.csv naming where every file came from. Built in the background; this list refreshes while one is building.">
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {exportsList === null ? <p className="text-sm text-ink-muted">Reading...</p>
        : exportsList.length === 0 ? (
          <p className="text-sm text-ink-muted">No exports yet. Pick a run or a version and press Export as zip.</p>
        ) : (
          <table className="tabular min-w-full text-xs">
            <thead className="text-left text-ink">
              <tr>
                {["Export", "State", "Started", "Finished", "Files", "Size", ""].map((h) => (
                  <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {exportsList.map((e) => (
                <tr key={e.id} className="border-b border-hairline/50">
                  <td className="px-2 py-1 font-medium text-ink">{e.label}</td>
                  <td className={`px-2 py-1 ${e.state === "failed" ? "text-critical" : ""}`} title={e.error ?? ""}>
                    {e.state === "building"
                      ? <span className="inline-flex items-center gap-1"><Loader2 size={12} className="animate-spin" />building</span>
                      : e.state}
                    {e.state === "failed" && e.error ? ` -- ${e.error}` : ""}
                  </td>
                  <td className="px-2 py-1">{when(e.created_at)}</td>
                  <td className="px-2 py-1">{when(e.finished_at)}</td>
                  <td className="px-2 py-1">
                    {(e.files ?? 0).toLocaleString()}
                    {e.missing ? <span className="text-critical"> ({e.missing} missing)</span> : ""}
                  </td>
                  <td className="px-2 py-1">{bytes(e.bytes)}</td>
                  <td className="px-2 py-1">
                    {e.state === "ready" && (
                      <button type="button" onClick={() => download(e.id)}
                              className="inline-flex items-center gap-1 rounded-md border border-black/15 bg-page px-2 py-0.5 font-medium text-ink">
                        <Download size={12} /> Download
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
    </Card>
  );
}
