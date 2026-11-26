import { resolveClassTarget, editedTargetClasses } from "../lib/classMappingInput";
import { useEffect, useRef, useState } from "react";
import { getTrack2PullControl, readTrack2YamlClasses, inspectTrack2LabelClasses, saveTrack2PullConfig, type SourceClassMapping } from "../lib/api";
import { usePageSave } from "../lib/pageSave";
import { listen, post } from "../lib/bus";

type Row = { source: string; path: string; classes: Record<string, string>; differences: string[]; verified: boolean; labelNote?: string; mappingKey: string; labelBucket: string; labelPrefix: string };
export function Track2CentralClasses({ bucket, folder }: { bucket: string; folder: string }) {
  const [drafts, setDrafts] = useState<Record<string, SourceClassMapping>>({});
  const [targetText, setTargetText] = useState<Record<string, Record<string, string>>>({});
  const loaded = useRef(false);
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);
  const [scanNotes, setScanNotes] = useState<Record<string, string>>({});
  const [observed, setObserved] = useState<Record<string, string[]>>({});
  async function saveMappings() {
    if (!loaded.current) throw new Error("Class mappings are still loading.");
    const classes = editedTargetClasses(targetText, drafts, expected);
    const renamed = Object.keys(classes).some(id => classes[id] !== expected[id]);
    const all = {...drafts};
    // Sources that already matched also need explicit ID maps after a shared rename.
    if (renamed) for (const row of rows) {
      if (row.source === "Central store" || all[row.mappingKey] || !Object.keys(row.classes).length) continue;
      all[row.mappingKey] = {target_classes: classes, ids: Object.fromEntries(Object.entries(row.classes).map(([id,name]) =>
        [id, Object.entries(expected).find(([,old]) => old === name)?.[0] ?? null]))};
    }
    const resolved = Object.fromEntries(Object.entries(all).map(([source, record]) => {
      const ids = {...record.ids};
      for (const [id, text] of Object.entries(targetText[source] ?? {})) {
        try { ids[id] = resolveClassTarget(text, classes); }
        catch {
          if (text.trim() === classes[record.ids[id] ?? ""]) ids[id] = record.ids[id];
          else ids[id] = resolveClassTarget(text, expected);
        }
      }
      return [source, {...record, target_classes: classes, ids}];
    }));
    const {config} = await getTrack2PullControl();
    await saveTrack2PullConfig({ source_class_mappings: resolved,
      ...(renamed ? {dataset_assemble:{...config.dataset_assemble,label_classes:classes}} : {}) });
    setDrafts(resolved);
    setExpected(classes);
    setTargetText({});
    if (renamed) post("dataset-classes-saved", classes);
    post("data-sources-changed", {from:"class-mapping-editor"});
    setMessage("Class mappings saved. They apply to the next combined-store preparation.");
  }
  usePageSave("Source class mappings", saveMappings);
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
      if (!loaded.current) { setDrafts(c.source_class_mappings ?? {}); loaded.current = true; }
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
        const labelPath = source.labels || (prefix.includes("/labels") ? prefix :
          prefix.includes("/images") ? prefix.replace(/\/images(?=\/|$)/, "/labels") : `${prefix}/labels`);
        const labelUri = labelPath.match(/^s3:\/\/([^/]+)\/(.*)$/);
        const labelBucket = labelUri ? labelUri[1] : home;
        const labelPrefix = labelUri ? labelUri[2] : labelPath;
        let labelNote = "";
        let labelsAvailable = false;
        if (!Object.keys(classes).length) {
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
          mappingKey: `s3://${key}`, labelBucket, labelPrefix, verified: !!yaml && !!Object.keys(mapping).length, labelNote: labelsAvailable || labelNote ? labelNote : undefined });
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
    <details className="mt-3 rounded border border-hairline p-3">
      <summary className="cursor-pointer font-medium">Edit source class IDs</summary>
      <p className="my-2 text-ink-muted">Type a target class ID, name, or ID and name (for example: 0 helmets). A new name renames that target ID for all sources when you save. Leave blank only for an unused ID. Changes are applied only to rebuilt combined-store training copies, never to originals. An ID marked unused will stop preparation if any processed label uses it. Save explicitly; editing does not start processing.</p>
      {rows.filter(r => r.source !== "Central store").map(row => {
        const draft = drafts[row.mappingKey];
        const ids = [...new Set([...Object.keys(row.classes), ...Object.keys(draft?.ids ?? {}), ...(observed[row.mappingKey] ?? [])])].sort((a,b)=>Number(a)-Number(b));
        return <div key={row.mappingKey} className="my-2 rounded border border-hairline p-2">
          <div className="flex flex-wrap items-center gap-2"><strong>{row.source}</strong><span className="break-all">{row.mappingKey}</span>
            <button type="button" className="rounded border px-2 py-1" onClick={() => {
              setScanNotes(n=>({...n,[row.mappingKey]:"Scanning labels..."}));
              inspectTrack2LabelClasses(row.labelBucket,row.labelPrefix).then(scan=>{
                setObserved(n=>({...n,[row.mappingKey]:Object.keys(scan.counts)}));
                setScanNotes(n=>({...n,[row.mappingKey]:`${scan.scanned}/${scan.total} files checked${scan.complete ? "" : " (partial)"}; ${scan.unreadable} unreadable; ${scan.invalid_lines} invalid lines. Boxes by ID: ${Object.entries(scan.counts).map(([id,count])=>`${id}: ${count}`).join(", ") || "none"}`}));
              }).catch(()=>setScanNotes(n=>({...n,[row.mappingKey]:"Could not scan labels."})));
            }}>Scan label IDs</button>
            <button type="button" className="rounded border px-2 py-1" onClick={()=>{
              setTargetText(all=>{const next={...all};delete next[row.mappingKey];return next;});
              if(draft) setDrafts(all=>{const next={...all};delete next[row.mappingKey];return next;});
              else setDrafts(all=>({...all,[row.mappingKey]:{target_classes:{...expected},ids:Object.fromEntries(ids.map(id=>[id,Object.entries(expected).find(([,name])=>name===row.classes[id])?.[0] ?? null]))}}));
              setMessage("Unsaved mapping changes.");
            }}>{draft ? "Remove mapping" : "Edit mapping"}</button>
          </div>
          {scanNotes[row.mappingKey] && <p className="mt-1">{scanNotes[row.mappingKey]}</p>}
          {draft && <><table className="mt-2 w-full text-left"><thead><tr><th>Source ID / name</th><th>Target class</th></tr></thead><tbody>{ids.map(id=><tr key={id}><td>{id} {row.classes[id] ?? "Name not recorded"}</td><td><input type="text" aria-label={`Target class for ${row.mappingKey} ID ${id}`}
            value={targetText[row.mappingKey]?.[id] ?? (draft.ids[id] != null ? `${draft.ids[id]} ${expected[draft.ids[id]!] ?? ""}`.trim() : "")}
            placeholder="e.g. 0 helmet; blank = unused"
            className="my-1 w-full rounded border bg-page p-1" onChange={e=>{
              const value=e.target.value;
              setTargetText(all=>({...all,[row.mappingKey]:{...all[row.mappingKey],[id]:value}}));
              setMessage("Unsaved mapping changes.");
            }} /></td></tr>)}</tbody></table>
          {!ids.length && <p className="text-critical">No class IDs found. Scan labels before saving this mapping.</p>}</>}
        </div>;
      })}
      <button type="button" disabled={saving || busy} className="mt-2 rounded bg-accent px-3 py-2 text-white disabled:opacity-40" onClick={()=>{setSaving(true);saveMappings().catch(e=>setMessage(e instanceof Error?e.message:"Could not save mappings.")).finally(()=>setSaving(false));}}>{saving?"Saving...":"Save class mappings"}</button>
      {message && <p role="status" className="mt-2">{message}</p>}
    </details>
  </div>;
}
