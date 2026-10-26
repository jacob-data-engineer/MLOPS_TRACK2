import { AlertCircle, Download, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { Track2S3FolderBrowser } from "../components/Track2S3FolderBrowser";
import {
  ApiError,
  exportTrack2Weights,
  getTrack2Export,
  type Track2ExportResponse,
} from "../lib/api";

const SIZES = ["s", "m", "l", "x"];

/** `yolo26s-open-detector` -> {generation: "26", size: "s", rest: "-open"}.
 *  Mirrors parse_group_name server-side. null for a name not in that
 *  shape -- the legacy custom group -- which then shows as itself. */
function parseGroup(group: string): { generation: string; size: string; rest: string } | null {
  const m = /^yolo(\d+)([smlx])(-.*)?$/.exec(group.replace(/-detector$/, ""));
  return m ? { generation: m[1], size: m[2], rest: m[3] ?? "" } : null;
}

/** Every group the server sent, in a stable, readable order.
 *
 * This page used to render four hardcoded `yolo11{size}-detector` keys.
 * The route has always passed `include_open=True` -- with a comment
 * saying why export, unlike deploy, is safe for open-taxonomy models --
 * and the page then threw all of them away. Nothing said so: the groups
 * were fetched, counted, and silently not drawn, so a custom or YOLO26
 * model was simply absent with no empty state to explain it.
 */
function groupsToShow(registries: Track2ExportResponse["registries"]): string[] {
  const sizeRank = (size: string) => (SIZES.indexOf(size) + 1 || 99);
  return Object.keys(registries)
    .filter((g) => (registries[g]?.packages.length ?? 0) > 0)
    .sort((a, b) => {
      const pa = parseGroup(a);
      const pb = parseGroup(b);
      // Unparseable names last rather than interleaved by accident.
      if (!pa || !pb) return (pa ? 0 : 1) - (pb ? 0 : 1) || a.localeCompare(b);
      // Production (no suffix) before open/named, then generation, then size.
      return (pa.rest ? 1 : 0) - (pb.rest ? 1 : 0)
        || pa.generation.localeCompare(pb.generation)
        || sizeRank(pa.size) - sizeRank(pb.size)
        || a.localeCompare(b);
    });
}

/** "YOLO26 · Size S · open" -- what this block actually holds. */
function groupHeading(group: string): string {
  const parsed = parseGroup(group);
  if (!parsed) return group;
  const suffix = parsed.rest ? ` · ${parsed.rest.replace(/^-/, "")}` : "";
  return `YOLO${parsed.generation} · Size ${parsed.size.toUpperCase()}${suffix}`;
}

function approvalTone(status: string): "confirmed" | "rejected" | "pending" {
  if (status === "Approved") return "confirmed";
  if (status === "Rejected") return "rejected";
  return "pending";
}

export function Track2ExportPage() {
  const [prefix, setPrefix] = useState("");
  const [data, setData] = useState<Track2ExportResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [selectedArn, setSelectedArn] = useState("");
  const [destination, setDestination] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getTrack2Export(prefix)
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load export data."));
  }, [prefix]);

  async function handleExport() {
    if (!selectedArn || !destination.trim()) return;
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const res = await exportTrack2Weights(selectedArn, destination.trim());
      setMessage(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Export failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Download size={20} className="text-accent" strokeWidth={2} />
        Track2 Export
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Pull an Approved registry package's <code className="tabular">model.pt</code> out to
        any S3 folder, written as{" "}
        <code className="tabular">yolo&lt;generation&gt;-&lt;size&gt;-v&lt;version&gt;-&lt;date&gt;.pt</code>{" "}
        so a folder full of exports stays legible on its own. An existing file
        at that exact name is archived first, never silently overwritten.
      </p>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {message && (
        <div className="mb-6 rounded-sm border border-hairline bg-accent-subtle px-4 py-3 text-sm text-ink">
          {message}
        </div>
      )}

      {!data && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading...
        </div>
      )}

      {data && (
        <div className="flex flex-col gap-5">
          <Card title="Select a model version to export">
            {groupsToShow(data.registries).length === 0 && (
              <p className="text-sm text-ink-muted">
                No registered versions yet.
              </p>
            )}
            {groupsToShow(data.registries).map((group) => {
              const summary = data.registries[group];
              if (!summary || summary.packages.length === 0) return null;
              return (
                <div key={group} className="mb-4 last:mb-0">
                  <p className="mb-1.5 bg-page px-2 py-1.5 text-xs font-semibold uppercase tracking-wide text-ink-secondary">
                    {groupHeading(group)}
                  </p>
                  <div className="flex flex-col gap-1.5">
                    {summary.packages.map((p) => {
                      const disabled = p.approval_status !== "Approved";
                      return (
                        <label
                          key={p.model_package_arn}
                          className={`flex items-center gap-3 rounded-sm border border-hairline bg-page px-3 py-2 ${disabled ? "opacity-50" : "cursor-pointer hover:bg-black/[0.04]"}`}
                        >
                          <input
                            type="checkbox"
                            checked={selectedArn === p.model_package_arn}
                            disabled={disabled}
                            onChange={(e) => setSelectedArn(e.target.checked ? p.model_package_arn : "")}
                          />
                          <span className="tabular text-sm text-ink">v{p.version}</span>
                          <Badge tone={approvalTone(p.approval_status)} label={p.approval_status} />
                        </label>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </Card>

          <Card title="Destination folder">
            <Track2S3FolderBrowser
              prefix={prefix}
              folders={data.folders}
              onNavigate={setPrefix}
              onSelect={setDestination}
              selected={destination}
              allowCreate
            />
            <input
              className="tabular mt-2 w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm text-ink"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
              placeholder="weights/s/manual-export"
            />
          </Card>

          <button
            type="button"
            onClick={handleExport}
            disabled={busy || !selectedArn || !destination.trim()}
            className="flex w-fit items-center justify-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
          >
            {busy && <Loader2 size={14} className="animate-spin" />}
            Export
          </button>
        </div>
      )}
    </div>
  );
}
