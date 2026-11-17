import type { Track2CatalogExport } from "../lib/api";

export function Track2ExportProgress({ state, error }: Pick<Track2CatalogExport, "state"> & { error?: string | null }) {
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
  </div>;
}
