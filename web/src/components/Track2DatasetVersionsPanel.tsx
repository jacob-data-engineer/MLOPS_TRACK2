import { useCallback, useEffect, useState } from "react";
import { Card } from "./Card";
import { formatDayTime } from "../lib/format";
import {
  ApiError,
  getTrack2DatasetVersion,
  listTrack2DatasetVersions,
  queueTrack2DatasetVersion,
  type Track2DatasetVersion,
} from "../lib/api";

/**
 * Dataset versions: the frozen dataset every queued run trains on.
 *
 * When a run is queued, the yaml it would train on is resolved to the
 * exact images and written once under `_datasets/<id>/` -- so nothing
 * built while it waits can change it (see dataset_versions.py). This card
 * lists them, shows each one's yaml, and queues a run on one again: the
 * same images and the settings it was first queued with.
 *
 * Nothing here is a setting: the list is re-read live, never saved.
 */
export function Track2DatasetVersionsPanel() {
  const [versions, setVersions] = useState<Track2DatasetVersion[] | null>(null);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [shown, setShown] = useState<{ id: string; data: string; eval: string } | null>(null);
  const [busy, setBusy] = useState("");

  const load = useCallback(() => {
    listTrack2DatasetVersions(30)
      .then((r) => { setVersions(r.versions); setError(""); })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not list dataset versions."));
  }, []);

  useEffect(() => {
    load();
    const timer = window.setInterval(load, 30_000);
    return () => window.clearInterval(timer);
  }, [load]);

  async function view(id: string) {
    if (shown?.id === id) { setShown(null); return; }
    setBusy(id);
    try {
      const v = await getTrack2DatasetVersion(id);
      setShown({ id, data: v.data_yaml, eval: v.eval_yaml });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not read that version.");
    } finally {
      setBusy("");
    }
  }

  async function again(id: string) {
    setBusy(id);
    setNote("");
    try {
      const r = await queueTrack2DatasetVersion(id);
      setNote(`Queued ${r.queued} on ${id} -- the same images and settings. It appears in the training queue above.`);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not queue it.");
    } finally {
      setBusy("");
    }
  }

  return (
    <Card
      title="Dataset versions"
      caption="The frozen dataset each queued run trains on -- written once, never changed."
    >
      <p className="mb-2 text-xs text-ink-muted">
        A version is made when a run is queued: the exact training and
        validation images, the classes and the filter, frozen under{" "}
        <span className="tabular">_datasets/</span>. Builds made afterwards
        cannot change it, and images that arrive later are not in it.
        <em> Queue again</em> retries on identical data.
      </p>
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {note && <p className="mb-2 text-xs text-ink-secondary">{note}</p>}
      {!versions && !error && <p className="text-xs text-ink-muted">Reading...</p>}
      {versions && versions.length === 0 && (
        <p className="text-xs text-ink-muted">None yet -- the next queued run makes the first.</p>
      )}
      {versions && versions.length > 0 && (
        <div className="flex max-h-96 flex-col gap-1 overflow-y-auto pr-1">
          {versions.map((v) => (
            <div key={v.version_id} className="rounded-sm border border-hairline bg-page p-2 text-xs">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="tabular font-medium text-ink">{v.version_id}</span>
                <span className="tabular text-ink-muted">{v.created_at ? formatDayTime(v.created_at) : ""}</span>
                <span className="text-ink-secondary">
                  {(v.train_images ?? 0).toLocaleString()} train / {(v.val_images ?? 0).toLocaleString()} val
                </span>
                {v.classes && v.classes.length > 0 && (
                  <span className="text-ink-muted">{v.classes.join(", ")}</span>
                )}
                {v.drop_undeclared && <span className="text-ink-muted">filter on</span>}
                <span className={`rounded-sm px-1 text-[10px] font-medium ${
                  v.state === "started" ? "bg-accent text-white" : "border border-black/15 text-ink-muted"}`}>
                  {v.state || "created"}
                </span>
                {v.trigger && <span className="text-ink-muted">{v.trigger}</span>}
              </div>
              {v.sources && v.sources.length > 0 && (
                <p className="mt-0.5 text-ink-muted">{v.sources.join("; ")}</p>
              )}
              <div className="mt-1 flex gap-3">
                <button type="button" disabled={!!busy} onClick={() => void view(v.version_id)}
                        className="font-medium text-ink-secondary underline disabled:opacity-40">
                  {shown?.id === v.version_id ? "Hide yaml" : "View yaml"}
                </button>
                <button type="button" disabled={!!busy || !v.entry_template}
                        title={v.entry_template ? "" : "Started directly, so its settings were not kept"}
                        onClick={() => void again(v.version_id)}
                        className="font-medium text-accent underline disabled:opacity-40">
                  {busy === v.version_id ? "Working..." : "Queue again"}
                </button>
              </div>
              {shown?.id === v.version_id && (
                <div className="mt-1.5 grid gap-2 md:grid-cols-2">
                  <div>
                    <p className="text-[11px] text-ink-muted">Training yaml</p>
                    <pre className="tabular max-h-56 overflow-auto rounded-sm bg-black/[0.04] p-1.5 text-[11px]">{shown.data}</pre>
                  </div>
                  <div>
                    <p className="text-[11px] text-ink-muted">Evaluation yaml</p>
                    <pre className="tabular max-h-56 overflow-auto rounded-sm bg-black/[0.04] p-1.5 text-[11px]">{shown.eval}</pre>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
