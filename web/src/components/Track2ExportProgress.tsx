import { useState } from "react";
import { ApiError, stopTrack2CatalogExport, type Track2CatalogExport } from "../lib/api";
import { formatSgt } from "../lib/format";

export function archiveSize(bytes: number): string {
  const units = ["bytes", "KiB", "MiB", "GiB", "TiB"];
  const unit = bytes > 0 ? Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1) : 0;
  return `${(bytes / 1024 ** unit).toLocaleString(undefined, { maximumFractionDigits: unit ? 2 : 0 })} ${units[unit]}`;
}

/** WHERE AN EXPORT HAS GOT TO, from what the server measured (2026-11-18).
 *
 * The stage it reports, files processed of files planned, and when that was
 * last written. Nothing is estimated: no percentage beyond processed/total,
 * no speed, no ETA. Packing files and finalizing the archive are told apart,
 * because the count is complete long before the upload is -- and "ready" is
 * shown only once the server has recorded the output as finished.
 *
 * `starting`: the request is out and no row is back yet. */
export function Track2ExportProgress({ exp, starting = false, onChanged }: {
  exp: Track2CatalogExport | null; starting?: boolean; onChanged?: () => void;
}) {
  const [stopping, setStopping] = useState(false);
  const [stopError, setStopError] = useState("");
  if (!exp) {
    return starting ? <p className="mt-3 text-xs" role="status">Starting the export on the server...</p> : null;
  }
  const active = exp.state === "building" || exp.state === "stopping";
  const total = exp.total_files ?? exp.files ?? null;
  const processed = exp.processed_files ?? null;
  const finalizing = active && /final/i.test(exp.stage ?? "");
  const counted = active && !finalizing && total != null && total > 0 && processed != null;
  const fraction = counted ? Math.min(1, processed! / total!) : 0;
  const stage = exp.state === "stopping" ? "Stopping"
    : exp.stage || (active ? "Started -- the first progress report has not arrived yet" : "");

  async function stop() {
    if (!exp) return;
    setStopping(true); setStopError("");
    try { await stopTrack2CatalogExport(exp.id); onChanged?.(); }
    catch (err) { setStopError(err instanceof ApiError ? err.detail : "Could not stop the export."); }
    finally { setStopping(false); }
  }

  return (
    <div className="mt-3 space-y-1.5 text-xs" role="status">
      {active && (
        <>
          <p>
            <span className="font-medium text-ink">{stage}</span>
            {counted && <> &middot; <span className="tabular">{processed!.toLocaleString()} / {total!.toLocaleString()} files</span></>}
            {finalizing && " -- every file is packed; the archive is being written to S3"}
          </p>
          <div role="progressbar" aria-label="Export progress" aria-valuemin={0}
               aria-valuemax={counted ? total! : undefined} aria-valuenow={counted ? processed! : undefined}
               aria-valuetext={counted ? `${processed} of ${total} files` : stage}
               className="h-2 overflow-hidden rounded-sm border border-hairline bg-page">
            <div className={`h-full bg-ink ${counted ? "" : "w-full animate-pulse opacity-40 motion-reduce:animate-none"}`}
                 style={counted ? { width: `${fraction * 100}%` } : undefined} />
          </div>
          <p className="text-ink-muted">
            Last progress report {formatSgt(exp.updated_at, "not received yet")}. The ZIP size is known only once it is finalized.
          </p>
          {exp.state === "building" && (
            <button type="button" onClick={() => void stop()} disabled={stopping}
                    className="rounded border border-critical px-2.5 py-1 font-medium text-critical disabled:opacity-40">
              {stopping ? "Stopping..." : "Stop export"}
            </button>
          )}
          {stopError && <p className="text-critical" role="alert">{stopError}</p>}
        </>
      )}
      {exp.state === "ready" && (
        <p className="font-medium text-ink">
          Ready &middot; {(exp.files ?? 0).toLocaleString()} file(s) packed
          {exp.format === "zip" && <> &middot; final ZIP size <span className="tabular" title={exp.bytes != null ? `${exp.bytes.toLocaleString()} bytes` : undefined}>{exp.bytes != null ? archiveSize(exp.bytes) : "not recorded"}</span></>}
        </p>
      )}
      {exp.state === "partial" && (
        <p className="font-medium text-critical">
          Incomplete &middot; {exp.missing.toLocaleString()} source file(s) were missing and are not in it
          ({(exp.files ?? 0).toLocaleString()} packed; the manifest lists the missing ones)
          {exp.format === "zip" && <> &middot; incomplete ZIP size <span className="tabular">{exp.bytes != null ? archiveSize(exp.bytes) : "not recorded"}</span></>}
        </p>
      )}
      {exp.state === "cancelled" && <p>Stopped before it finished. Nothing was kept.</p>}
      {exp.state === "failed" && <p className="text-critical" role="alert">Failed{exp.error ? `: ${exp.error}` : "."}</p>}
    </div>
  );
}

/** The options an export was STARTED with, in one comparable shape. */
export type ExportChoice = {
  include_images: boolean; include_model: boolean; include_metadata: boolean;
  metadata_groups: string[] | null; destination: string; format: string;
};

/** true / false, or null when the export predates recorded options. */
export function sameChoice(exp: Track2CatalogExport, now: ExportChoice): boolean | null {
  if (!exp.options) return null;                                   // started before options were recorded
  const groups = (g: string[] | null | undefined) => [...(g ?? [])].sort().join("|");
  return exp.options.include_images === now.include_images
    && exp.options.include_model === now.include_model
    && exp.options.include_metadata === now.include_metadata
    && (!now.include_metadata || groups(exp.options.metadata_groups) === groups(now.metadata_groups))
    && exp.destination === now.destination && exp.format === now.format;
}

/** WHICH EXPORT THIS IS (2026-11-18): source record, contents, destination,
 * creation time and export ID -- and whether it matches the options chosen
 * now. An export made with other options is labelled the PREVIOUS export,
 * so an old ZIP is never presented as the answer to a new choice. */
export function Track2ExportResult({ exp, source, current }: {
  exp: Track2CatalogExport; source: string; current: ExportChoice | null;
}) {
  const match = current ? sameChoice(exp, current) : null;
  const o = exp.options;
  const contents = o
    ? [o.include_images && "images and labels", o.include_model && "model (.pt)",
       o.include_metadata && `metadata (${(o.metadata_groups ?? []).length} categories)`]
        .filter(Boolean).join(", ") || "file manifest only"
    : "not recorded (started before contents were recorded)";
  return (
    <div className="mt-3 rounded border border-hairline bg-surface p-2 text-xs">
      <p className="font-medium text-ink">
        {match === false ? "Previous export -- made with different options from those selected now"
          : match === null ? "Latest export of this record -- its options were not recorded"
          : "Export of the current selection"}
      </p>
      {match === false && (
        <p className="text-ink-secondary">Prepare a new export to get the current choices. This one is shown so it stays reachable.</p>
      )}
      <dl className="tabular mt-1 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
        <dt className="text-ink-muted">Source record</dt><dd>{source}</dd>
        <dt className="text-ink-muted">Contents</dt><dd>{contents} &middot; {exp.format === "folder" ? "folder" : "ZIP"}</dd>
        <dt className="text-ink-muted">Destination</dt><dd className="break-all">{exp.destination === "local" ? "Download to this PC" : exp.destination_uri || "S3"}</dd>
        <dt className="text-ink-muted">Created</dt><dd>{formatSgt(exp.created_at)}</dd>
        {exp.finished_at && <><dt className="text-ink-muted">Finished</dt><dd>{formatSgt(exp.finished_at)}</dd></>}
        <dt className="text-ink-muted">Export ID</dt><dd className="break-all">{exp.id}</dd>
      </dl>
    </div>
  );
}
