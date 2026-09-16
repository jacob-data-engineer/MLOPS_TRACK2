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
        any S3 folder, written as <code className="tabular">yolo11-&lt;size&gt;-v&lt;version&gt;-&lt;date&gt;.pt</code>{" "}
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
          <Card title="Select a YOLO11 model version to export">
            {SIZES.map((size) => {
              const group = `yolo11${size}-detector`;
              const summary = data.registries[group];
              if (!summary || summary.packages.length === 0) return null;
              return (
                <div key={group} className="mb-4 last:mb-0">
                  <p className="mb-1.5 bg-page px-2 py-1.5 text-xs font-semibold uppercase tracking-wide text-ink-secondary">
                    YOLO11 &middot; Size {size.toUpperCase()}
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
