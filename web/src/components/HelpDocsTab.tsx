import { useEffect, useState } from "react";
import { Card } from "./Card";
import { ConfirmDialog } from "./ConfirmDialog";
import {
  ApiError, deleteTrack2AssistantDoc, getTrack2AssistantStatus, listTrack2AssistantDocs,
  uploadTrack2AssistantDoc, type Track2AssistantDoc, type Track2AssistantStatus,
} from "../lib/api";
import { formatSgt } from "../lib/format";

/** "LLM documentation" -- the last tab of Track2 Help (2026-11-20).
 *
 * Extra documentation for the help assistant: procedures, site rules,
 * explanations the screens do not carry. Uploaded as .md or .txt, kept in
 * the app's bucket, and read by the assistant as REFERENCE MATERIAL --
 * never as instructions. The assistant's own guide to the pages and cards
 * is generated from the application's code and needs no upload. */
export function HelpDocsTab() {
  const [docs, setDocs] = useState<Track2AssistantDoc[] | null>(null);
  const [limits, setLimits] = useState<{ max_bytes: number; allowed: string[] } | null>(null);
  const [status, setStatus] = useState<Track2AssistantStatus | null>(null);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState(false);
  const [removing, setRemoving] = useState<string | null>(null);

  function load() {
    setError("");
    listTrack2AssistantDocs()
      .then((r) => { setDocs(r.documents); setLimits({ max_bytes: r.max_bytes, allowed: r.allowed }); })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not list the documentation."));
    getTrack2AssistantStatus().then(setStatus).catch(() => undefined);
  }
  useEffect(load, []);

  async function upload(file: File) {
    setBusy(true); setError(""); setMessage("");
    try {
      const r = await uploadTrack2AssistantDoc(file);
      setMessage(`${r.replaced ? "Replaced" : "Uploaded"} ${r.name} (${r.bytes.toLocaleString()} bytes).`);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "The upload failed.");
    } finally {
      setBusy(false);
    }
  }

  async function remove(name: string) {
    setBusy(true); setError("");
    try {
      await deleteTrack2AssistantDoc(name);
      setMessage(`Deleted ${name}.`);
      setRemoving(null);
      load();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not delete it.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Documentation for the help assistant">
      <p className="mb-3 text-sm text-ink-secondary">
        The help assistant already knows every page and card from the application&apos;s own code. Upload
        extra documentation here &mdash; procedures, site rules, explanations the screens do not carry &mdash; and
        it will use it when answering. Documents are reference material for the assistant, never instructions to it.
      </p>
      {status && (
        <p className="mb-3 text-xs text-ink-muted">
          Assistant: {status.enabled ? "on" : "not switched on yet (waiting for Bedrock approval)"} &middot; model{" "}
          <span className="tabular">{status.model}</span> in {status.region} &middot; {status.guide_sections} sections generated from the code
        </p>
      )}

      <label className="inline-flex cursor-pointer items-center gap-2 rounded-md border border-black/15 bg-page px-3 py-1.5 text-sm font-medium">
        {busy ? "Working..." : "Upload a document (.md or .txt)"}
        <input type="file" accept=".md,.txt,text/markdown,text/plain" className="sr-only" disabled={busy}
               aria-label="Upload documentation for the help assistant"
               onChange={(e) => { const f = e.target.files?.[0]; if (f) void upload(f); e.target.value = ""; }} />
      </label>
      {limits && (
        <span className="ml-3 text-xs text-ink-muted">
          Up to {(limits.max_bytes / 1000).toLocaleString()} KB each. A file with the same name replaces the old one.
        </span>
      )}
      {message && <p className="mt-2 text-sm" role="status">{message}</p>}
      {error && <p className="mt-2 text-sm text-critical" role="alert">{error} <button type="button" className="underline" onClick={load}>Retry</button></p>}

      <div className="mt-4">
        {docs === null && !error && <p className="text-sm text-ink-muted" role="status">Loading...</p>}
        {docs && docs.length === 0 && <p className="text-sm text-ink-muted">No documentation uploaded yet.</p>}
        {docs && docs.length > 0 && (
          <table className="tabular w-full text-left text-xs">
            <thead><tr className="border-b border-hairline">
              <th className="py-1.5 pr-3 font-medium">Document</th><th className="py-1.5 pr-3 font-medium">Size</th>
              <th className="py-1.5 pr-3 font-medium">Uploaded</th><th />
            </tr></thead>
            <tbody>
              {docs.map((d) => (
                <tr key={d.name} className="border-b border-hairline/50">
                  <td className="py-1.5 pr-3 text-ink">{d.name}</td>
                  <td className="py-1.5 pr-3">{d.bytes.toLocaleString()} bytes</td>
                  <td className="py-1.5 pr-3">{formatSgt(d.updated_at)}</td>
                  <td className="py-1.5 text-right">
                    <button type="button" className="text-critical underline" disabled={busy} onClick={() => setRemoving(d.name)}>Delete</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <ConfirmDialog
        open={removing !== null}
        title={`Delete ${removing ?? ""}?`}
        body={<>The help assistant stops using this document straight away. Its guide to the pages and cards,
          generated from the code, is untouched, and so are all other documents. This cannot be undone; upload the
          file again to bring it back.</>}
        confirmLabel="Delete document"
        destructive
        busy={busy}
        onConfirm={() => removing && void remove(removing)}
        onCancel={() => setRemoving(null)}
      />
    </Card>
  );
}
