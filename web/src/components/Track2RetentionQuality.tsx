import { useEffect, useRef, useState } from "react";
import { ApiError, inspectTrack2Retention, type Track2RetentionConfig, type Track2RetentionQuality as Quality } from "../lib/api";

export function Track2RetentionQuality({ root, config }: { root: string; config: Partial<Track2RetentionConfig> }) {
  const [result, setResult] = useState<Quality | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const generation = useRef(0);
  const settings = JSON.stringify(config);
  useEffect(() => {
    generation.current += 1;
    setResult(null); setError(""); setBusy(false);
    return () => { generation.current += 1; };
  }, [root, settings]);

  async function inspect() {
    const request = ++generation.current;
    setBusy(true); setError(""); setResult(null);
    try {
      const got = await inspectTrack2Retention(root, config);
      if (request === generation.current) setResult(got);
    } catch (err) {
      if (request === generation.current) setError(err instanceof ApiError ? err.detail : "Coverage inspection failed.");
    } finally {
      if (request === generation.current) setBusy(false);
    }
  }

  return <div className="rounded-md border border-hairline p-3 text-xs">
    <div className="flex flex-wrap items-center justify-between gap-2">
      <strong>Class coverage before and after decay</strong>
      <button type="button" disabled={busy || !root} onClick={() => void inspect()}
        className="rounded-md border border-black/15 px-3 py-1.5 disabled:opacity-40">
        {busy ? "Reading labels..." : "Check label coverage"}
      </button>
    </div>
    <p className="mt-2 text-ink-muted">Reads the weekly and golden training labels. Reports warnings without changing your selection.
      Class IDs are local to each source; picked folders and final class remapping are not included.</p>
    {error && <p role="alert" className="mt-2 text-critical">{error}</p>}
    {result && <>
      <p className="mt-2 font-medium">{result.complete ? "All inspected labels were readable and valid." : "Coverage incomplete — resolve label issues before relying on these counts."}</p>
      {result.issues.map((issue) => <p key={issue} className="mt-1 text-critical">{issue}</p>)}
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-left tabular">
          <thead><tr><th>Source</th><th>Class ID</th><th>Before</th><th>Selected</th><th>Coverage</th></tr></thead>
          <tbody>{result.sources.flatMap((source) => source.classes.map((row) =>
            <tr key={`${source.source}-${row.class_id}`} className="border-t border-hairline">
              <td className="py-1">{source.source}</td><td>{row.class_id}</td><td>{row.before}</td><td>{row.after}</td>
              <td className={row.low ? "text-critical" : ""}>{row.lost ? "Lost" : row.low ? `Below ${result.min_class_images} images` : "Meets threshold"}</td>
            </tr>))}</tbody>
        </table>
      </div>
      {result.sources.map((source) => <p key={source.source} className="mt-1 text-ink-muted">
        {source.source}: {source.selected}/{source.available} selected; {source.empty_labels} explicit empty labels;
        {" "}{source.missing_labels} missing, {source.unreadable_labels} unreadable, {source.invalid_labels} invalid.
      </p>)}
      <p className="mt-2 text-ink-muted">{result.note}</p>
    </>}
  </div>;
}
