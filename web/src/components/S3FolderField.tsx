import { useEffect, useState } from "react";
import {
  ApiError,
  checkTrack2FolderDelete,
  deleteTrack2Folder,
  listTrack2Folders,
  type Track2FolderDeleteCheck,
} from "../lib/api";
import { ConfirmDialog } from "./ConfirmDialog";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

function sizeText(bytes: number): string {
  if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
  if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
  if (bytes >= 1e3) return `${(bytes / 1e3).toFixed(0)} KB`;
  return `${bytes} bytes`;
}

/**
 * One way to pick an S3 folder, everywhere: the import card's way.
 *
 * That card shows the browser OPEN -- breadcrumb, search box, the folders
 * in view -- and prints what was picked underneath it. No text box and no
 * Browse button, because the browser is the field.
 *
 * The first version of this put the same browser behind a "Browse"
 * toggle beside a text input, which is not the same thing at all: what
 * you saw on arriving was still a box wanting a path typed into it, and
 * the folders were a click away. The point of copying the import card is
 * that the folders are simply there.
 *
 * What this costs, and it is worth saying: the breadcrumb is buttons,
 * not an editable address bar, so a long path can no longer be pasted --
 * it has to be clicked to. Every caller here is picking a folder a few
 * levels down, so that is a fair trade, but it is a trade.
 */
export function S3FolderField({
  label,
  hint,
  value,
  onChange,
  bucket = "",
  allowCreate = false,
  disabled = false,
}: {
  label: string;
  /** The parenthetical after the label -- what this folder is for. */
  hint?: string;
  value: string;
  onChange: (folder: string) => void;
  /** Blank means the app's own bucket, which is what the API defaults to. */
  bucket?: string;
  /** For a destination that is allowed not to exist yet. */
  allowCreate?: boolean;
  disabled?: boolean;
}) {
  // Opens where the field already points, so a card that restored a saved
  // path shows that folder rather than sending you back to the root to
  // walk down to it again.
  const [prefix, setPrefix] = useState(value.trim().replace(/\/+$/, ""));
  const [folders, setFolders] = useState<string[]>([]);
  const [error, setError] = useState("");
  // Bumped after a delete so the listing is fetched again: the prefix has
  // not changed, so nothing else would re-run the effect.
  const [reload, setReload] = useState(0);

  // The delete dialog. One state for the whole exchange -- the count, the
  // server's refusal, and the result -- so it all appears in the dialog
  // that was opened from the row, never somewhere else on the card.
  const [toDelete, setToDelete] = useState<string | null>(null);
  const [check, setCheck] = useState<Track2FolderDeleteCheck | null>(null);
  const [deleteError, setDeleteError] = useState("");
  const [deleting, setDeleting] = useState(false);

  function openDelete(path: string) {
    setToDelete(path);
    setCheck(null);
    setDeleteError("");
    checkTrack2FolderDelete(bucket, path)
      .then(setCheck)
      .catch((err) => setDeleteError(err instanceof ApiError ? err.detail : "Could not read that folder."));
  }

  function closeDelete() {
    if (deleting) return;
    setToDelete(null);
    setCheck(null);
    setDeleteError("");
  }

  async function confirmDelete() {
    if (!toDelete || !check) return;
    setDeleting(true);
    setDeleteError("");
    try {
      const out = await deleteTrack2Folder(bucket, toDelete, check.in_use.length > 0);
      if (out.errors.length) {
        setDeleteError(`Deleted ${out.deleted.toLocaleString()}, but ${out.errors.length} `
          + `could not be: ${out.errors.slice(0, 3).join("; ")}`);
        return;
      }
      // A folder that no longer exists cannot stay chosen.
      const gone = toDelete.replace(/\/+$/, "");
      if (value === gone || value.startsWith(gone + "/")) onChange("");
      setToDelete(null);
      setCheck(null);
      setReload((n) => n + 1);
    } catch (err) {
      setDeleteError(err instanceof ApiError ? err.detail : "Could not delete that folder.");
    } finally {
      setDeleting(false);
    }
  }

  useEffect(() => {
    if (disabled) return;
    let cancelled = false;
    // Cleared before the fetch, not after it lands. The browser draws its
    // rows against the prefix it is given, so leaving the PREVIOUS
    // folder's rows under a new prefix invites a click that builds a path
    // which has never existed.
    setFolders([]);
    setError("");
    listTrack2Folders(bucket, prefix)
      .then((res) => !cancelled && setFolders(res.folders))
      .catch(() => !cancelled && setError("Could not read that folder."));
    return () => { cancelled = true; };
  }, [prefix, bucket, disabled, reload]);

  return (
    <div className={disabled ? "opacity-40" : undefined}>
      <span className="mb-1 block text-xs text-ink-muted">
        {label}{" "}
        {hint && <span className="text-ink-muted/70">{hint}</span>}
      </span>
      <div inert={disabled || undefined}>
        <Track2S3FolderBrowser
          prefix={prefix}
          folders={folders}
          onNavigate={setPrefix}
          selected={value}
          allowCreate={allowCreate}
          onSelect={onChange}
          onDelete={openDelete}
        />
      </div>
      {/* What is actually chosen, under the browser -- the import card
          prints the key it picked the same way. Without it the only
          record of the choice is a tick several rows up, which scrolls
          out of view the moment you navigate anywhere. */}
      {value && (
        <p className="tabular mt-1 text-xs text-ink-secondary">{value}</p>
      )}
      {error && <p className="mt-1 text-xs text-critical">{error}</p>}

      <ConfirmDialog
        open={toDelete !== null}
        title={`Delete ${toDelete ?? ""}?`}
        destructive
        busy={deleting}
        confirmLabel={check?.in_use.length ? "Delete it anyway" : "Delete it"}
        cancelLabel="Cancel"
        onCancel={closeDelete}
        onConfirm={() => {
          // Nothing to confirm until the count is in -- or when the
          // server refused outright. Cancel is the only way out then.
          if (check) void confirmDelete();
        }}
        body={
          <>
            {!check && !deleteError && <p>Counting what is in it...</p>}
            {check && (
              <p>
                Every file under{" "}
                <span className="tabular">{check.bucket}/{check.prefix}/</span> is
                permanently deleted: <strong>{check.objects.toLocaleString()} files,{" "}
                {sizeText(check.bytes)}</strong>, including every sub-folder. S3 keeps
                no copy, so <strong>this cannot be undone</strong>. Other folders,
                trained models and registered versions are untouched.
              </p>
            )}
            {check && check.in_use.length > 0 && (
              <div className="mt-2 rounded-sm border border-critical/40 bg-critical-subtle px-2 py-1.5 text-critical">
                <p className="font-medium">Still in use by:</p>
                <ul className="ml-4 list-disc">
                  {check.in_use.map((u) => <li key={u}>{u}</li>)}
                </ul>
                <p className="mt-1">
                  Deleting it empties a folder those still read; the next pull or
                  run finds nothing there.
                </p>
              </div>
            )}
            {deleteError && <p className="mt-2 text-critical">{deleteError}</p>}
          </>
        }
      />
    </div>
  );
}
