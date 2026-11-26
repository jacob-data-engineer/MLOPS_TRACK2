import { useEffect, useState } from "react";
import { getTrack2PullControl, readTrack2YamlClasses, inspectTrack2LabelClasses } from "../lib/api";
import { listen } from "../lib/bus";

type Row = { source: string; path: string; classes: Record<string, string>; differences: string[]; verified: boolean; labelNote?: string };
export function Track2CentralClasses({ bucket, folder }: { bucket: string; folder: string }) {
  const [rows, setRows] = useState<Row[]>([]);
  const [expected, setExpected] = useState<Record<string, string>>({});
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => listen("data-sources-changed", () => setRevision(n => n + 1)), []);
  useEffect(() => {
    let cancelled = false;
    setBusy(true); setRows([]); setError("");
    const run = async () => {
      const { config: c } = await getTrack2PullControl();
      const mapping = c.dataset_assemble?.label_classes ?? {};
      if (cancelled) return;
      setExpected(mapping);
      const sources: { source: string; path: string; bucket: string; labels?: string }[] = [];
      const add = (source: string, path: string, home: string, labels?: string) => {
        if (path) sources.push({ source, path, bucket: home, labels });
      };
      const a = c.dataset_assemble;
      if (a?.use !== false) {
        const paths = [...(a?.train_sources ?? []), ...(a?.eval_sources ?? [])].flatMap(s => [s.images, s.labels]);
        if (!paths.length && a?.folders) paths.push(...Object.values(a.folders));
        paths.forEach(p => add("Selected folders", p, c.source_bucket || "track2-s3"));
      }
      if (c.retention?.use_golden) {
        add("Golden Set images", c.golden_set?.source || "", c.golden_set?.bucket || "track2-s3", c.golden_set?.labels_source);
        add("Golden Set labels", c.golden_set?.labels_source || "", c.golden_set?.bucket || "track2-s3");
      }
      if (!a?.only) Object.values(c.picked ?? {}).flat().forEach(p => add("Pull source", p, c.source_bucket || "track2-s3"));
      add("Central store", folder, bucket);
      const seen = new Set<string>();
      const results: Row[] = [];
      for (const source of sources) {
        if (cancelled) return;
        const match = source.path.match(/^s3:\/\/([^/]+)\/(.*)$/);
        const home = match ? match[1] : source.bucket;
        const prefix = (match ? match[2] : source.path).replace(/^\/+|\/+$/g, "");
        const root = prefix.replace(/\/(images|labels)(\/.*)?$/, "");
        const key = `${home}/${root}`;
        if (seen.has(key)) continue;
        seen.add(key);
        let classes: Record<string, string> = {};
        let yaml = "";
        // Prefer original label taxonomy over generated/filtered training YAML.
        for (const name of ["data.source.yaml", "data.yaml"]) {
          try {
            const result = await readTrack2YamlClasses(home, `${root}/${name}`);
            if (Object.keys(result.classes).length) { classes = result.classes; yaml = name; break; }
          } catch { /* Missing or unreadable metadata is explicitly unverified. */ }
        }
        let labelNote = "";
        let labelsAvailable = false;
        if (!Object.keys(classes).length) {
          const labelPath = source.labels || (prefix.includes("/labels") ? prefix :
            prefix.includes("/images") ? prefix.replace(/\/images(?=\/|$)/, "/labels") : `${prefix}/labels`);
          const labelUri = labelPath.match(/^s3:\/\/([^/]+)\/(.*)$/);
          try {
            const scan = await inspectTrack2LabelClasses(labelUri ? labelUri[1] : home, labelUri ? labelUri[2] : labelPath);
            labelsAvailable = Object.keys(scan.counts).length > 0;
            classes = Object.fromEntries(Object.keys(scan.counts).map(id => [id, mapping[id] ?? "Unknown class"]));
            labelNote = `${scan.scanned}/${scan.total} label files checked; ${scan.unreadable} unreadable; ${scan.invalid_lines} invalid lines. ` +
              (scan.complete ? "All files in this labels folder checked. " : "Partial check. ") +
              "IDs read from labels; names taken from saved settings, not verified from labels.";
            if (!scan.total) labelNote = "No label files found in the labels folder.";
          } catch { labelNote = "Could not read label files."; }
        }
        const ids = [...new Set([...Object.keys(mapping), ...Object.keys(classes)])].sort((a,b) => Number(a)-Number(b));
        const differences = ids.filter(id => (yaml || id in classes) && mapping[id] !== classes[id]).map(id =>
          `${id}: source ${classes[id] ?? "missing"}; expected ${mapping[id] ?? "not configured"}`);
        results.push({ source: source.source, path: `s3://${key}/${yaml}`, classes, differences,
          verified: !!yaml && !!Object.keys(mapping).length, labelNote: labelsAvailable || labelNote ? labelNote : undefined });
      }
      if (!cancelled) setRows(results);
    };
    run().catch(() => { if (!cancelled) setError("Could not check source classes. Retry the check."); })
      .finally(() => { if (!cancelled) setBusy(false); });
    return () => { cancelled = true; };
  }, [bucket, folder, revision]);
  const mismatches = rows.some(r => Object.keys(r.classes).length && r.differences.length);
  const unknown = !rows.length || rows.some(r => !r.verified);
  return <div className="mt-3 rounded-md border border-hairline p-3 text-xs">
    <div className="flex items-center justify-between gap-2"><strong>Dataset classes and source checks</strong>
      <button type="button" disabled={busy} onClick={() => setRevision(n => n+1)} className="rounded border px-2 py-1 disabled:opacity-40">{busy ? "Checking classes..." : "Check classes"}</button></div>
    <p className="mt-2">Expected: {Object.entries(expected).sort(([a],[b])=>Number(a)-Number(b)).map(([id,name])=>`${id} ${name}`).join(", ") || "No dataset classes configured"}</p>
    <p className="mt-1 text-ink-muted">Checks YAML mappings, or reads up to 200 label files when YAML is unavailable. Label IDs alone cannot verify class names. Save changed settings before checking.</p>
    {error && <p role="alert" className="mt-2 text-critical">{error}</p>}
    {!busy && !error && <p role="status" className={`mt-2 rounded border p-2 ${mismatches || unknown ? "border-amber-500 bg-amber-50 text-amber-900" : "border-green-600 text-green-800"}`}>
      {mismatches ? "Warning: class mappings differ. Review the highlighted differences before combining." : unknown ? "Not fully verified: one or more class mappings are missing or unreadable." : "All checked YAML class IDs and names match the configured dataset classes."}</p>}
    {!!rows.length && <div className="mt-2 overflow-x-auto"><table className="w-full text-left"><thead><tr><th>Source</th><th>Classes</th><th>Check</th></tr></thead><tbody>{rows.map(r=><tr key={r.path} className="border-t border-hairline"><td className="p-2">{r.source}<div className="break-all text-ink-muted">{r.path}</div></td><td className="p-2">{Object.entries(r.classes).map(([id,name])=><span key={id} className={`mr-2 inline-block ${expected[id] !== name ? "bg-amber-100 text-amber-900" : ""}`}>{id} {name}</span>)}</td><td className="p-2">{r.labelNote ? <span>{r.differences.length > 0 && <strong className="text-critical">{r.differences.join("; ")} — </strong>}{r.labelNote}</span> : !r.verified ? "Not verified: mapping unavailable" : r.differences.length ? <span className="text-amber-900">{r.differences.join("; ")}</span> : "Match"}</td></tr>)}</tbody></table></div>}
  </div>;
}
