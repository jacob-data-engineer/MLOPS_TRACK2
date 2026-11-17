import type { Track2CatalogExport } from "../lib/api";

function archiveSize(bytes: number): string {
  const units = ["bytes", "KiB", "MiB", "GiB", "TiB"];
  const unit = bytes > 0 ? Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1) : 0;
  return `${(bytes / 1024 ** unit).toLocaleString(undefined, { maximumFractionDigits: unit ? 2 : 0 })} ${units[unit]}`;
}

export function Track2ExportProgress({ state, error, bytes, format = "zip" }: Pick<Track2CatalogExport, "state"> & {
  error?: string | null; bytes?: number | null; format?: "zip" | "folder";
}) {
  const active = state === "building" || state === "stopping";
  const complete = state === "ready" || state === "partial";
  const label = state === "building" ? "Preparing export files. Progress updates automatically."
    : state === "stopping" ? "Stopping export..."
    : state === "ready" ? "Export ready."
    : state === "partial" ? "Export incomplete: some source files are missing."
    : state === "cancelled" ? "Export stopped." : `Export failed${error ? `: ${error}` : "."}`;
  return <div className="mt-3 space-y-2 text-xs" role="status">
    <p>{label}</p>
    {(active || complete) && <div role="progressbar" aria-label="Export preparation" aria-valuemin={0} aria-valuemax={100}
      aria-valuenow={complete ? 100 : undefined} aria-valuetext={label}
      className="h-2 overflow-hidden rounded-sm border border-hairline bg-page">
      <div className={`h-full bg-ink ${active ? "w-full animate-pulse motion-reduce:animate-none opacity-40" : "w-full"}`} />
    </div>}
    {active && <p className="text-ink-muted">Preparing on the server; percentage is not available yet.</p>}
    {format === "zip" && active && <p className="text-ink-muted">ZIP size will appear when compression finishes.</p>}
    {format === "zip" && complete && <p className="tabular" title={bytes != null ? `${bytes.toLocaleString()} bytes` : undefined}>
      {bytes != null && Number.isFinite(bytes) && bytes >= 0
        ? `${state === "partial" ? "Incomplete ZIP size" : "ZIP download size"}: ${archiveSize(bytes)} (actual compressed size)`
        : "ZIP size was not recorded for this export."}
    </p>}
  </div>;
}
