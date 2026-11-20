import { useEffect, useRef, useState } from "react";
import { ApiError, checkCatalogDataset, getTrack2CatalogMetadata, getTrack2CatalogExports,
  getTrack2PullControl, saveTrack2PullConfig, exportTrack2BestWeights,
  getTrack2CatalogExportUrl, startTrack2CatalogExport, type Track2MetricsRow,
  type Track2RunForModel, type Track2CatalogExport, type CatalogExportDestination } from "../lib/api";
import { modelLabel } from "./Track2ModelPicker";
import { METADATA_CATEGORIES } from "./Track2CatalogMetadata";
import { Track2ExportProgress } from "./Track2ExportProgress";
import { isSeedVersion, statusLabel } from "../lib/slug";

import { formatSgt } from "../lib/format";

const METRICS = ["mAP50-95", "mAP50", "precision", "recall", "f1", "accuracy"] as const;
type Metric = typeof METRICS[number];
type Target = { row: Track2MetricsRow; run?: Track2RunForModel };
const message = (e: unknown) => e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "Export failed.";

export function Track2ComparisonExport({ rows, runs, browseAll = false, onOpenRun }: {
  rows: Track2MetricsRow[]; runs: Record<string, Track2RunForModel>; browseAll?: boolean;
  onOpenRun?: (id:string)=>void;
}) {
  const [metric, setMetric] = useState<Metric>("mAP50-95");
  const [chosen, setChosen] = useState<string[]>([]);
  const [images, setImages] = useState(true);
  const [model, setModel] = useState(true);
  const [metadata, setMetadata] = useState(true);
  const [groups, setGroups] = useState(Object.keys(METADATA_CATEGORIES));
  const [destination, setDestination] = useState<CatalogExportDestination>({ destination: "local", bucket: "", prefix: "", format: "zip" });
  const [preview, setPreview] = useState<Target[]>([]);
  const [notes, setNotes] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [settingsReady, setSettingsReady] = useState(false);
  const [saved, setSaved] = useState("");
  const [jobs, setJobs] = useState<Track2CatalogExport[]>([]);
  const [search, setSearch] = useState("");
  const [direct, setDirect] = useState<{label: string; url: string; uri: string}[]>([]);
  const cancel = useRef(false);
  useEffect(() => () => { cancel.current = true; }, []);
  useEffect(() => {
    let disposed = false;
    getTrack2PullControl().then(result => {
      if (disposed) return;
      const config = result.config.comparison_export;
      if (config) {
        if (METRICS.includes(config.metric as Metric)) setMetric(config.metric as Metric);
        setImages(config.images ?? true); setModel(config.model ?? true); setMetadata(config.metadata ?? true);
        if (config.groups) setGroups(config.groups.filter(g => g in METADATA_CATEGORIES));
        if (config.destination) setDestination(config.destination);
      }
    }).catch(e => { if (!disposed) setError(`Could not load saved package options: ${message(e)}`); })
      .finally(() => { if (!disposed) setSettingsReady(true); });
    return () => { disposed = true; };
  }, []);
  async function saveOptions() {
    setBusy(true); setError(""); setSaved("");
    try {
      await saveTrack2PullConfig({ comparison_export: { metric, images, model, metadata, groups, destination } });
      setSaved("Package options saved. Model selections and pending exports are not resumed on reload.");
    } catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  }
  const ranked = [...rows].sort((a, b) => (b.metrics?.[metric] ?? -1) - (a.metrics?.[metric] ?? -1)
    || a.model_package_arn.localeCompare(b.model_package_arn));
  const selectionKey = rows.map(r => r.model_package_arn).join("|");
  useEffect(() => { setPreview([]); }, [selectionKey, chosen, images, model, metadata, groups, destination, metric]);

  async function review(arns: string[]) {
    setBusy(true); setPreview([]); setError(""); setNotes([]);
    try {
      const targets: Target[] = []; const warnings: string[] = [];
      for (const row of ranked.filter(r => arns.includes(r.model_package_arn))) {
        const run = runs[row.model_package_arn];
        if (!run) {
          if (images || metadata || !model) throw new Error(`${modelLabel(row)} has no recorded training run. Select only Model (.pt) to export its registered checkpoint.`);
          targets.push({row}); continue;
        }
        const info = await getTrack2CatalogMetadata("run", run.run_id);
        if (model && !info.checkpoint_available) throw new Error(`Run ${run.run_number} has no published checkpoint.`);
        if (images) {
          const availability = await checkCatalogDataset("run", run.run_id);
          if (availability.state === "unavailable" || availability.state === "check_failed") throw new Error(`Run ${run.run_number}: ${availability.detail}`);
          warnings.push(`Run ${run.run_number}: ${availability.detail}`);
        }
        if (metadata && info.unavailable.length) warnings.push(`Run ${run.run_number}: metadata not recorded for ${info.unavailable.join(", ")}.`);
        targets.push({ row, run });
      }
      if (!targets.length) throw new Error("Choose at least one model to export.");
      setNotes(warnings); setPreview(targets);
    } catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  }

  async function exportPackages() {
    setRunning(true); setError(""); cancel.current = false;
    const targets = [...preview]; setPreview([]);
    const options = { ...destination, format: destination.destination === "local" ? "zip" as const : destination.format, include_images: images, include_model: model,
      include_metadata: metadata, metadata_groups: groups };
    try {
      for (const target of targets) {
        if (cancel.current) break;
        if (!target.run) {
          const result = await exportTrack2BestWeights({group_name:target.row.group_name,criterion:"newest",
            model_package_arn:target.row.model_package_arn,destination:options.destination,bucket:options.bucket,prefix:options.prefix,retention:"keep_all"});
          setDirect(old=>[...old,{label:`${target.row.registry} · ${modelLabel(target.row)}`,url:result.url,uri:result.written_to}]);
          continue;
        }
        const started = await startTrack2CatalogExport("run", target.run.run_id,
          { ...options, model_package_arn: target.row.model_package_arn });
        let finished = false;
        while (!finished && !cancel.current) {
          const result = await getTrack2CatalogExports({ kind: "run", id: target.run.run_id });
          const job = result.exports.find(e => e.id === started.id);
          if (job) {
            setJobs(previous => [...previous.filter(e => e.id !== job.id), job]);
            finished = !["building", "stopping"].includes(job.state);
            if (finished && job.state !== "ready") throw new Error(`Batch paused: ${job.label} is ${job.state}. Review its result before exporting the remaining models.`);
          }
          if (!finished) await new Promise(resolve => window.setTimeout(resolve, 3000));
        }
      }
    } catch (e) { setError(message(e)); }
    finally { setRunning(false); }
  }
  const invalid = (!images && !model && !metadata) || (metadata && !groups.length)
    || (destination.destination === "s3" && !destination.bucket.trim());
  const button = "rounded border border-hairline px-2 py-1 text-xs disabled:opacity-40";
  return <section className="mb-5 space-y-3 border-b border-hairline pb-4" aria-label="Rank and export compared models">
    <fieldset disabled={busy || running || !settingsReady} className="space-y-3 disabled:opacity-60">
      {browseAll&&<label className="block text-xs">Find a model <input aria-label="Find manual model" className="ml-2 rounded border p-2" placeholder="Model name, version, run or status" value={search} onChange={e=>setSearch(e.target.value)}/></label>}
      <label className="text-sm font-medium">{browseAll?"Sort models by":"Rank selected models by"} <select aria-label="Ranking metric" value={metric} onChange={e => setMetric(e.target.value as Metric)} className="ml-2 rounded border p-1">
        {METRICS.map(key => <option key={key}>{key}</option>)}
      </select></label>
      <p className="text-xs text-ink-muted">Highest recorded score first; missing scores are unranked. Ties share a score. Choose any model to export.</p>
      <p className="text-xs">{rows.length} available models · {chosen.filter(a=>rows.some(r=>r.model_package_arn===a)).length} selected for export</p>
      <div className="max-h-[calc(22rem+2px)] overflow-auto rounded border border-hairline" tabIndex={0} role="region" aria-label="Models available for manual export">
        <table className="tabular min-w-full whitespace-nowrap text-xs [&_tr]:h-8">
          <thead className="sticky top-0 bg-surface text-left text-ink"><tr>
            {["Export", "Run", "Submitted", "Status", "Model name", "Model type", "Size", "Model version", "Dataset version", "Images", "Registered as", metric, "Validation images"].map(h =>
              <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{ranked.filter(row => `${row.registry} ${row.group_name} ${modelLabel(row)} ${row.status} ${runs[row.model_package_arn]?.status ?? ""} Run ${runs[row.model_package_arn]?.run_number ?? ""}`.toLowerCase().includes(search.toLowerCase())).map(row => {
            const run = runs[row.model_package_arn];
            const selected = chosen.includes(row.model_package_arn);
            return <tr key={row.model_package_arn} className={`border-b border-hairline/50 hover:bg-black/[0.03] [&_td]:px-2 [&_td]:py-1 ${selected ? "bg-accent-subtle" : ""}`}>
              <td><input aria-label={`Export ${modelLabel(row)}`} type="checkbox" checked={selected} onChange={e => setChosen(old => e.target.checked ? [...old, row.model_package_arn] : old.filter(a => a !== row.model_package_arn))} /></td>
              <td className="font-medium text-ink">{run ? onOpenRun ? <button type="button" className="rounded px-1 underline focus-visible:outline focus-visible:outline-2" aria-label={`Open Run ${run.run_number} in Run Central`} onClick={() => onOpenRun(run.run_id)}>Run {run.run_number}</button> : `Run ${run.run_number}` : "No run recorded"}</td>
              <td>{formatSgt(run?.submitted_at, "--")}</td>
              <td>{run?.status || (isSeedVersion(row.description) ? "Seed" : statusLabel(row.status))}</td>
              <td>{run?.detector || row.registry}</td>
              <td>{row.generation ? `YOLO${row.generation}` : "--"}</td>
              <td>{row.size?.toUpperCase() || "--"}</td>
              <td className="font-medium text-ink">v{row.version}</td>
              <td title={run?.version_id || ""}>{run?.version_number ? `${run.version_kind === "snapshot" ? "Snapshot" : "Version"} ${run.version_number}` : "--"}</td>
              <td>{run?.images?.toLocaleString() ?? "--"}</td>
              <td>{row.group_name}</td>
              <td>{row.metrics?.[metric]?.toFixed(4) ?? "Not measured"}</td>
              <td>{row.metrics?.dataset?.validation_count?.toLocaleString() ?? "--"}</td>
            </tr>;
          })}</tbody>
        </table>
      </div>
      <p className="text-xs">Recorded dataset versions and counts do not prove identical evaluation images or settings. <a className="underline text-accent" href="/track2/eval">Open evaluation</a> to explicitly evaluate models on a common dataset.</p>
      <div className="flex flex-wrap gap-4 text-xs">
        <label><input type="checkbox" checked={model} onChange={e => setModel(e.target.checked)} /> Model (.pt)</label>
        <label><input type="checkbox" checked={images} onChange={e => setImages(e.target.checked)} /> Actual images and labels</label>
        <label><input type="checkbox" checked={metadata} onChange={e => setMetadata(e.target.checked)} /> Metadata</label>
      </div>
      <p className="text-xs text-ink-muted">Each model gets its own package with data.yaml and a file manifest. Unselected dataset files are referenced only. Metadata is also embedded in the .pt when both are selected.</p>
      {metadata && <details><summary className="cursor-pointer text-xs">Metadata categories ({groups.length})</summary><div className="grid gap-1 text-xs md:grid-cols-2">{Object.keys(METADATA_CATEGORIES).map(g => <label key={g}><input type="checkbox" checked={groups.includes(g)} onChange={e => setGroups(old => e.target.checked ? [...old, g] : old.filter(x => x !== g))} /> {g}</label>)}</div></details>}
      <div className="flex flex-wrap gap-2 text-xs"><select aria-label="Package destination" value={destination.destination} onChange={e => setDestination({ ...destination, destination: e.target.value as "local" | "s3" })}><option value="local">Download ZIP</option><option value="s3">S3</option></select>
        {destination.destination === "s3" && <><input aria-label="Export bucket" placeholder="Bucket" value={destination.bucket} onChange={e => setDestination({ ...destination, bucket: e.target.value })} /><input aria-label="Export prefix" placeholder="Folder prefix" value={destination.prefix} onChange={e => setDestination({ ...destination, prefix: e.target.value })} /><select aria-label="Package format" value={destination.format} onChange={e => setDestination({ ...destination, format: e.target.value as "zip" | "folder" })}><option value="zip">ZIP</option><option value="folder">Folder</option></select></>}
      </div>
      <button className={button} disabled={invalid || !ranked.some(r => chosen.includes(r.model_package_arn))} onClick={() => void review(chosen)}>{busy ? "Checking availability..." : "Review selected packages"}</button>
      <button className={`${button} ml-2`} onClick={() => void saveOptions()}>Save package options</button>
    </fieldset>
    {saved && <p className="text-xs" role="status">{saved}</p>}
    {notes.map((note, i) => <p className="text-xs" key={i}>{note}</p>)}
    {preview.length > 0 && <div className="space-y-2 rounded border border-hairline p-3 text-xs">
      <p className="font-medium">Ready to export these exact versions:</p>
      {preview.map(t => <p className="break-all" key={t.row.model_package_arn}>{modelLabel(t.row)} · {t.run ? `Run ${t.run.run_number}` : "Registered checkpoint only"} · {t.row.model_package_arn}</p>)}
      <p>{model ? ".pt; " : ""}{images ? "images and labels; " : "dataset references only; "}{metadata ? `${groups.length} metadata categories; ` : ""}{destination.destination === "local" ? "downloadable ZIP" : `s3://${destination.bucket}/${destination.prefix}`}</p>
      <button className={button} disabled={running} onClick={() => void exportPackages()}>Export {preview.length} package(s)</button>
    </div>}
    {running && <p className="text-xs" role="status">Exporting packages sequentially. Keep this page open for remaining packages; completed exports also appear in the catalog.</p>}
    {error && <p className="text-xs text-critical" role="alert">{error}</p>}
    {direct.map((result,i)=><p key={i} className="text-xs">{result.label}: {result.url?<a className="underline" href={result.url}>Download .pt</a>:result.uri}</p>)}
    {jobs.map(job => <div key={job.id} className="rounded border border-hairline p-2"><p className="text-xs">{job.label} · {job.id}</p><Track2ExportProgress exp={job} />
      {(job.state === "ready" || job.state === "partial") && job.format === "zip" && <button className={button} onClick={() => void getTrack2CatalogExportUrl(job.id).then(r => window.location.assign(r.url)).catch(e => setError(message(e)))}>Download {job.state === "partial" ? "incomplete " : ""}ZIP</button>}
      {job.destination_uri && <p className="break-all text-xs">{job.destination_uri}</p>}
    </div>)}
  </section>;
}
