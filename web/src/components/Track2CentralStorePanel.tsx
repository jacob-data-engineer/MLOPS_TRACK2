import { useEffect, useState } from "react";
import { getTrack2PullControl, saveTrack2PullConfig } from "../lib/api";
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
      })
      .catch(() => undefined)
      .finally(() => setSettingsRead(true));
  }, []);

  useEffect(() => {
    if (!settingsRead) return;
    setSaved(false);
    const timer = window.setTimeout(() => {
      void saveTrack2PullConfig({ central_store: { bucket, folder } })
        .then(() => setSaved(true))
        .catch(() => undefined);
    }, 600);
    return () => window.clearTimeout(timer);
  }, [settingsRead, bucket, folder]);

  const path = bucket
    ? `s3://${bucket}/${folder.replace(/^\/+|\/+$/g, "")}${folder.trim() ? "/" : ""}`
    : "";

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
    </Card>
  );
}
