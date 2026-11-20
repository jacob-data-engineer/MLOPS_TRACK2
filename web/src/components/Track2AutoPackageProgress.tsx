import { useEffect, useState } from "react";
import { getTrack2CatalogExports, getTrack2CatalogExportUrl, type Track2CatalogExport } from "../lib/api";
import { Track2ExportProgress } from "./Track2ExportProgress";

export function Track2AutoPackageProgress({id, runId}: {id: string; runId: string}) {
  const [job, setJob] = useState<Track2CatalogExport | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    let disposed = false;
    async function refresh() {
      try {
        const result = await getTrack2CatalogExports({kind: "run", id: runId});
        if (!disposed) { setJob(result.exports.find(j => j.id === id) ?? null); setError(""); }
      } catch { if (!disposed) setError("Could not refresh package progress."); }
    }
    void refresh(); const timer = window.setInterval(refresh, 3000);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [id, runId]);
  return <div className="mt-2 rounded border border-hairline p-3 text-xs">
    <p>Package {id}</p><Track2ExportProgress exp={job} starting={!job}/>
    {job?.state === "ready" && job.format === "zip" && <button className="mt-2 rounded border p-2" onClick={() => void getTrack2CatalogExportUrl(id).then(r => window.location.assign(r.url)).catch(() => setError("Could not get the package download link."))}>Download package ZIP</button>}
    {job?.destination_uri && <p className="break-all">{job.destination_uri}</p>}
    {error && <p role="alert" className="text-critical">{error}</p>}
  </div>;
}
