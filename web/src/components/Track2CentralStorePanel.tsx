import { useEffect, useState } from "react";
import {
  ApiError,
  getTrack2Combine,
  getTrack2PullControl,
  startTrack2Combine,
  stopTrack2Combine,
  type Track2CombineJob,
  saveTrack2PullConfig,
} from "../lib/api";
import { formatDayTime } from "../lib/format";
import { Card } from "./Card";
import { S3FolderField } from "./S3FolderField";
import { WRITABLE } from "./Track2PullControlPanel";

/** THE CENTRAL DATA STORE (2026-10-29).
 *
 * The one bucket and folder where all the data and images are meant to be
 * kept. Picked the way Pull control picks its destination: a bucket from
 * the ones the stack grants writes to -- a bucket that cannot be written
 * is a choice that fails later, so it is not offered -- and a folder that
 * can be browsed or created.
 *
 * Saved in `central_store` like every other card's choice, so it goes
 * into the Data Preparation settings file and comes back on load.
 */
export function Track2CentralStorePanel() {
  const [bucket, setBucket] = useState("");
  const [folder, setFolder] = useState("");
  /** Train from the decayed copy of the combined store, rebuilt before
   * every run. Saved with the bucket and folder. */
  const [trainFromCombined, setTrainFromCombined] = useState(false);
  const [combine, setCombine] = useState<Track2CombineJob | null>(null);
  const [combineError, setCombineError] = useState("");
  // The writer must not run before the saved choice is read, or it would
  // save the empty defaults over it.
  const [settingsRead, setSettingsRead] = useState(false);
  const [saved, setSaved] = useState(false);
  useEffect(() => {
    getTrack2PullControl()
      .then((status) => {
        const store = status.config.central_store || {};
        if (store.bucket) setBucket(store.bucket);
        if (store.folder) setFolder(store.folder);
        setTrainFromCombined(Boolean(store.train_from_combined));
      })
      .catch(() => undefined)
      .finally(() => setSettingsRead(true));
  }, []);

  useEffect(() => {
    if (!settingsRead) return;
    setSaved(false);
    const timer = window.setTimeout(() => {
      void saveTrack2PullConfig({ central_store: { bucket, folder, train_from_combined: trainFromCombined } })
        .then(() => setSaved(true))
        .catch(() => undefined);
    }, 600);
    return () => window.clearTimeout(timer);
  }, [settingsRead, bucket, folder, trainFromCombined]);

  // The last combine, and while one runs, every 5 s until it ends.
  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const poll = () => {
      getTrack2Combine()
        .then((job) => {
          if (cancelled) return;
          setCombine(job);
          if (job.running) timer = window.setTimeout(poll, 5000);
        })
        .catch(() => undefined);
    };
    if (!combine || combine.running) poll();
    return () => { cancelled = true; window.clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [combine?.running]);

  function combineNow() {
    setCombineError("");
    startTrack2Combine()
      .then(setCombine)
      .catch((err) => setCombineError(err instanceof ApiError ? err.detail : "Could not start it."));
  }

  function stopCombine() {
    setCombineError("");
    stopTrack2Combine()
      .then(setCombine)
      .catch((err) => setCombineError(err instanceof ApiError ? err.detail : "Could not stop it."));
  }


  const path = bucket
    ? `s3://${bucket}/${folder.replace(/^\/+|\/+$/g, "")}${folder.trim() ? "/" : ""}`
    : "";
  const decayedPath = path ? `${path.replace(/\/$/, "")}_decay/` : "";

  return (
    <Card
      title="Central data store"
      caption="The one place all the data and images are kept. Pick the bucket, then the folder inside it."
    >
      <div className="grid gap-3 md:grid-cols-2">
        <label className="block">
          <span className="mb-1 block text-xs text-ink-muted">
            Bucket{" "}
            <span className="text-ink-muted/70">(where all the data and images are stored)</span>
          </span>
          <select
            className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
            value={bucket}
            onChange={(e) => { setBucket(e.target.value); setFolder(""); }}
          >
            <option value="">Choose a bucket</option>
            {WRITABLE.map((b) => <option key={b} value={b}>{b}</option>)}
          </select>
        </label>
        <div className="block">
          {bucket ? (
            <S3FolderField
              label="Folder"
              hint="(new or existing; everything is kept under it)"
              value={folder}
              onChange={setFolder}
              bucket={bucket}
              allowCreate
            />
          ) : (
            <p className="mt-6 text-xs text-ink-muted">Pick a bucket to choose a folder in it.</p>
          )}
        </div>
      </div>
      {path && (
        <p className="tabular mt-3 text-xs text-ink-secondary">
          Central store: <span className="font-medium text-ink">{path}</span>
          {saved && <span className="ml-2 text-ink-muted">saved</span>}
        </p>
      )}

      {/* ---- combine (2026-11-13) ---- Every ticked source into the
           store, which is only ever added to; then a decayed copy of it
           beside it, which is what training reads. */}
      <div className="mt-4 border-t border-hairline pt-3 text-xs">
        <label className="flex items-start gap-1.5 text-ink">
          <input type="checkbox" className="mt-0.5" checked={trainFromCombined}
                 onChange={(e) => setTrainFromCombined(e.target.checked)} />
          <span>
            Train from the combined store
            <span className="block text-ink-muted">
              Before every run, each ticked source (Assemble&apos;s folders, the golden set,
              Pull Control&apos;s weeks) is copied into the store, then{" "}
              <span className="tabular">{decayedPath || "<store>_decay/"}</span> is rebuilt from it
              through the retention policy and the run trains on that alone. It is deleted once
              the run has finished. The store itself is never pruned or overwritten.
            </span>
          </span>
        </label>
        <div className="mt-3 flex flex-wrap items-center gap-3">
          <button type="button" onClick={combineNow}
                  disabled={!path || !!combine?.running}
                  className="rounded-md bg-accent px-3 py-1.5 text-sm font-medium text-white disabled:opacity-40">
            {combine?.running ? "Combining..." : "Combine data sources"}
          </button>
          {combine?.running && (
            <button type="button" onClick={stopCombine} disabled={!!combine.stopping}
                    className="rounded-md border border-critical px-3 py-1.5 text-sm font-medium text-critical disabled:opacity-40">
              {combine.stopping ? "Stopping..." : "Stop"}
            </button>
          )}
          <span className="text-ink-muted">
            Does the copy and the decay now, without training. The decayed copy is kept until the next run rebuilds it.
          </span>
        </div>
        {combine?.running && (
          <p className="mt-2 text-ink-secondary" role="status">
            {combine.stopping
              ? "Stopping: copies already in flight finish, the rest are skipped."
              : <>Copying since {formatDayTime(combine.started_at ?? "")}. Large sources take a few minutes.</>}
          </p>
        )}
        {combine && !combine.running && combine.finished_at && (
          combine.ok ? (
            <p className="tabular mt-2 text-ink-secondary" role="status">
              {formatDayTime(combine.finished_at)}: {(combine.new_in_store ?? 0).toLocaleString()} new file(s) into{" "}
              <span className="text-ink">{combine.store}</span>
              {" "}({(combine.already_in_store ?? 0).toLocaleString()} already there);{" "}
              {(combine.in_decayed ?? 0).toLocaleString()} into <span className="text-ink">{combine.decayed}</span> after decay.
            </p>
          ) : combine.stopped ? (
            <p className="mt-2 text-ink-secondary" role="status">
              {formatDayTime(combine.finished_at)}: {combine.error} Files already copied into the store stay
              there, and the next combine carries on from them. The half-built decayed copy was deleted.
            </p>
          ) : (
            <p className="mt-2 text-critical" role="alert">Combine failed: {combine.error}</p>
          )
        )}
        {combineError && <p className="mt-2 text-critical" role="alert">{combineError}</p>}
      </div>
    </Card>
  );
}
