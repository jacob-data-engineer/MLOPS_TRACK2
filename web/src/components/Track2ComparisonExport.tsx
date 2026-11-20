import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, checkCatalogDataset, getTrack2CatalogMetadata, getTrack2CatalogExports,
  estimateTrack2CatalogExport, type PackageSizeEstimate, type Track2ExportSettings, type ModelPackageOptions, exportTrack2BestWeights, getTrack2Detectors, type Track2Detector, getTrack2Buckets, listTrack2Folders,
  getTrack2CatalogExportUrl, startTrack2CatalogExport, type Track2MetricsRow,
  type Track2RunForModel, type Track2CatalogExport, type CatalogExportDestination } from "../lib/api";
import { modelLabel } from "./Track2ModelPicker";
import { METADATA_CATEGORIES } from "./Track2CatalogMetadata";
import { Track2ExportProgress } from "./Track2ExportProgress";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";
import { isSeedVersion, statusLabel } from "../lib/slug";

import { formatSgt } from "../lib/format";

const METRICS = ["mAP50-95", "mAP50", "precision", "recall", "f1", "accuracy"] as const;
type Metric = typeof METRICS[number];
type Target = { row: Track2MetricsRow; run?: Track2RunForModel; estimate?: PackageSizeEstimate };
const message = (e: unknown) => e instanceof ApiError ? e.detail : e instanceof Error ? e.message : "Export failed.";

export function Track2ComparisonExport({ rows, runs, browseAll = false, onOpenRun, settings, onChange }: {
  rows: Track2MetricsRow[]; runs: Record<string, Track2RunForModel>; browseAll?: boolean;
  onOpenRun?: (id:string)=>void;
  settings: Track2ExportSettings; onChange: (value: Track2ExportSettings) => void;
}) {
  const [metric, setMetric] = useState<Metric>("mAP50-95");
  const [chosen, setChosen] = useState<string[]>([]);
  const options = useMemo(() => settings.package_options ?? {include_model: true, include_images: true, include_metadata: true, metadata_groups: Object.keys(METADATA_CATEGORIES), format: "zip" as const}, [settings.package_options]);
  const images = options.include_images, model = options.include_model, metadata = options.include_metadata;
  const groups = useMemo(() => options.metadata_groups ?? Object.keys(METADATA_CATEGORIES), [options.metadata_groups]);
  const setOptions = (patch: Partial<ModelPackageOptions>) => onChange({...settings, package_options: {...options, ...patch}});
  const setImages = (value: boolean) => setOptions({include_images: value});
  const setModel = (value: boolean) => setOptions({include_model: value});
  const setMetadata = (value: boolean) => setOptions({include_metadata: value});
  const setGroups = (update: (old: string[]) => string[]) => setOptions({metadata_groups: update(groups)});
  const destination = useMemo<CatalogExportDestination>(() => ({destination: settings.destination, bucket: settings.bucket, prefix: settings.prefix, format: options.format}), [settings.destination, settings.bucket, settings.prefix, options.format]);
  const setDestination = (value: CatalogExportDestination) => onChange({...settings, destination: value.destination, bucket: value.bucket, prefix: value.prefix, package_options: {...options, format: value.format}});
  const [preview, setPreview] = useState<Target[]>([]);
  const [notes, setNotes] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [buckets, setBuckets] = useState<string[]>([]);
  const [folders, setFolders] = useState<string[]>([]);
  const [folderError, setFolderError] = useState("");
  useEffect(() => {
    if (destination.destination !== "s3") return;
    let disposed = false;
    getTrack2Buckets().then(result => { if (!disposed) setBuckets((result.buckets ?? []).map(b => b.name)); }).catch(() => {});
    return () => { disposed = true; };
  }, [destination.destination]);
  useEffect(() => {
    let disposed = false;
    setFolders([]); setFolderError("");
    if (destination.destination === "s3" && destination.bucket) listTrack2Folders(destination.bucket, destination.prefix)
      .then(result => { if (!disposed) setFolders(result.folders ?? []); })
      .catch(() => { if (!disposed) setFolderError("Could not list folders. You can still enter the destination folder above."); });
    return () => { disposed = true; };
  }, [destination.destination, destination.bucket, destination.prefix]);
  const [busy, setBusy] = useState(false);
  const [running, setRunning] = useState(false);
  const [jobs, setJobs] = useState<Track2CatalogExport[]>([]);
  const [search, setSearch] = useState("");
  const [modelFilter, setModelFilter] = useState("");
  const [detectors, setDetectors] = useState<Track2Detector[]>([]);
  const [modelNamesError, setModelNamesError] = useState("");
  const [generationFilter, setGenerationFilter] = useState("");
  const [sizeFilter, setSizeFilter] = useState("");
  const [showBy, setShowBy] = useState<"model" | "run">("model");
  const [runOrder, setRunOrder] = useState<"desc" | "asc">("desc");
  const [direct, setDirect] = useState<{label: string; url: string; uri: string}[]>([]);
  const cancel = useRef(false);
  useEffect(() => () => { cancel.current = true; }, []);
  useEffect(() => {
    if (!browseAll) return;
    let disposed = false;
    getTrack2Detectors().then(result => { if (!disposed) setDetectors(result.detectors); })
      .catch(() => { if (!disposed) setModelNamesError("Model display names could not be loaded. Showing recorded model names."); });
    return () => { disposed = true; };
  }, [browseAll]);
  const ranked = [...rows].sort((a, b) => {
    if (browseAll && showBy === "run") {
      const left = runs[a.model_package_arn]?.run_number;
      const right = runs[b.model_package_arn]?.run_number;
      if (left == null && right != null) return 1;
      if (right == null && left != null) return -1;
      return (left != null && right != null ? (left - right) * (runOrder === "asc" ? 1 : -1) : 0)
        || a.model_package_arn.localeCompare(b.model_package_arn);
    }
    return (b.metrics?.[metric] ?? -1) - (a.metrics?.[metric] ?? -1)
      || a.model_package_arn.localeCompare(b.model_package_arn);
  });
  const nameOf = (row: Track2MetricsRow) => runs[row.model_package_arn]?.detector || row.registry;
  const displayName = (name: string) => {
    const definition = detectors.find(d => d.slug === name);
    if (!definition) return name;
    const count = Object.keys(definition.classes).length;
    return `${definition.label} (${count} ${count === 1 ? "class" : "classes"})`;
  };
  const modelNames = [...new Set([...detectors.map(d => d.slug), ...rows.map(nameOf)])].sort((a, b) => displayName(a).localeCompare(displayName(b)));
  const namedModels = ranked.filter(row => !modelFilter || nameOf(row) === modelFilter);
  const generations = [...new Set(namedModels.map(row => row.generation).filter(Boolean))].sort((a, b) => Number(a) - Number(b));
  const typedModels = namedModels.filter(row => !generationFilter || row.generation === generationFilter);
  const sizes = [...new Set(typedModels.map(row => row.size).filter(Boolean))].sort((a, b) => "nslmx".indexOf(a) - "nslmx".indexOf(b));
  const matchingModels = showBy === "model" ? typedModels.filter(row => !sizeFilter || row.size === sizeFilter) : ranked;
  const visibleRows = matchingModels.filter(row => `${nameOf(row)} ${displayName(nameOf(row))} ${row.group_name} ${modelLabel(row)} ${row.status} ${runs[row.model_package_arn]?.status ?? ""} Run ${runs[row.model_package_arn]?.run_number ?? ""}`.toLowerCase().includes(search.toLowerCase()));
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
        let estimate: PackageSizeEstimate | undefined;
        try {
          estimate = await estimateTrack2CatalogExport("run", run.run_id, {...destination, format: destination.destination === "local" ? "zip" : destination.format,
            include_images: images, include_model: model, include_metadata: metadata, metadata_groups: groups, model_package_arn: row.model_package_arn});
        } catch { warnings.push(`Run ${run.run_number}: storage estimate unavailable. Required space is unknown.`); }
        targets.push({ row, run, estimate });
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
  const bytes = (value: number) => value >= 1024 ** 3 ? `${(value / 1024 ** 3).toFixed(2)} GiB` : value >= 1024 ** 2 ? `${(value / 1024 ** 2).toFixed(2)} MiB` : `${(value / 1024).toFixed(1)} KiB`;
  const sizeComplete = preview.every(t => t.estimate?.recommended_bytes != null);
  const totalSpace = preview.reduce((sum, t) => sum + (t.estimate?.recommended_bytes ?? 0), 0);
  const invalid = (!images && !model && !metadata) || (metadata && !groups.length)
    || (destination.destination === "s3" && !destination.bucket.trim());
  const button = "rounded border border-hairline px-2 py-1 text-xs disabled:opacity-40";
  return <section className="mb-5 space-y-3 border-b border-hairline pb-4" aria-label="Rank and export compared models">
    <fieldset disabled={busy || running} className="space-y-3 disabled:opacity-60">
      {browseAll && <fieldset className="flex gap-4 text-xs" aria-label="Manual browsing mode">
        <label><input type="radio" name="manual-show-by" checked={showBy === "model"} onChange={() => { setShowBy("model"); setSearch(""); }} /> Show by model name</label>
        <label><input type="radio" name="manual-show-by" checked={showBy === "run"} onChange={() => { setShowBy("run"); setSearch(""); }} /> Show by runs</label>
      </fieldset>}
      {browseAll && <div className="flex flex-wrap items-center gap-3 text-xs">
        {showBy === "model" && <div className="flex w-full flex-wrap items-end gap-3 rounded border border-hairline p-2"><label className="min-w-48 flex-1">Model Name <select aria-label="Filter manual models by name" className="mt-1 block w-full rounded border p-2" value={modelFilter} onChange={e => { setModelFilter(e.target.value); setGenerationFilter(""); setSizeFilter(""); }}>
          <option value="">All model names</option>{modelNames.map(name => <option key={name} value={name}>{displayName(name)}</option>)}
        </select></label>
        <label>Model type <select aria-label="Filter manual models by YOLO type" className="mt-1 block rounded border p-2" value={generationFilter} onChange={e => { setGenerationFilter(e.target.value); setSizeFilter(""); }}>
          <option value="">All YOLO types</option>{generations.map(generation => <option key={generation} value={generation}>YOLO{generation}</option>)}
        </select></label>
        <label>Size <select aria-label="Filter manual models by size" className="mt-1 block rounded border p-2" value={sizeFilter} onChange={e => setSizeFilter(e.target.value)}>
          <option value="">All sizes</option>{sizes.map(size => <option key={size} value={size}>{size.toUpperCase()}</option>)}
        </select></label></div>}
        <label>Keyword search <input aria-label="Keyword search" className="ml-2 rounded border p-2" placeholder="Model name, run number, version or status" value={search} onChange={e => setSearch(e.target.value)}/></label>
        <button type="button" className={button} onClick={() => { setModelFilter(""); setGenerationFilter(""); setSizeFilter(""); setSearch(""); }}>Clear filters</button>
        <span>{visibleRows.length} shown</span>
      </div>}
      {browseAll && modelNamesError && <p role="status" className="text-xs text-critical">{modelNamesError}</p>}
      {browseAll && showBy === "run" ? <>
        <label className="text-sm font-medium">Sort by run number <select aria-label="Run number sort order" value={runOrder} onChange={e => setRunOrder(e.target.value as "asc" | "desc")} className="ml-2 rounded border p-1">
          <option value="desc">Descending (highest first)</option><option value="asc">Ascending (lowest first)</option>
        </select></label>
        <p className="text-xs text-ink-muted">Sorted by run number only. Models without a recorded run appear last.</p>
      </> : <><label className="text-sm font-medium">{browseAll?"Sort models by":"Rank selected models by"} <select aria-label="Ranking metric" value={metric} onChange={e => setMetric(e.target.value as Metric)} className="ml-2 rounded border p-1">
        {METRICS.map(key => <option key={key}>{key}</option>)}
      </select></label>
      <p className="text-xs text-ink-muted">Highest recorded score first; missing scores are unranked. Ties share a score. Choose any model to export.</p></>}
      <p className="text-xs">{rows.length} available models · {chosen.filter(a=>rows.some(r=>r.model_package_arn===a)).length} selected for export</p>
      <div className="max-h-[calc(22rem+2px)] overflow-auto rounded border border-hairline" tabIndex={0} role="region" aria-label="Models available for manual export">
        <table className="tabular min-w-full whitespace-nowrap text-xs [&_tr]:h-8">
          <thead className="sticky top-0 bg-surface text-left text-ink"><tr>
            {["Export", "Run", "Submitted", "Status", "Model name", "Model type", "Size", "Model version", "Dataset version", "Images", "Registered as", metric, "Validation images"].map(h =>
              <th key={h} className="border-b border-hairline px-2 py-1.5 font-medium">{h}</th>)}
          </tr></thead>
          <tbody>{visibleRows.map(row => {
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
        {!visibleRows.length && <p className="p-3 text-xs">No models match these filters.</p>}
      </div>
      <p className="text-xs">Recorded dataset versions and counts do not prove identical evaluation images or settings. <a className="underline text-accent" href="/track2/eval">Open evaluation</a> to explicitly evaluate models on a common dataset.</p>
      <div className="flex flex-wrap gap-4 text-xs">
        <label><input type="checkbox" checked={model} onChange={e => setModel(e.target.checked)} /> Model (.pt)</label>
        <label><input type="checkbox" checked={images} onChange={e => setImages(e.target.checked)} /> Actual images and labels</label>
        <label><input type="checkbox" checked={metadata} onChange={e => setMetadata(e.target.checked)} /> Metadata</label>
      </div>
      <p className="text-xs text-ink-muted">Each model gets its own package with data.yaml and a file manifest. Unselected dataset files are referenced only. Metadata is also embedded in the .pt when both are selected.</p>
      {metadata && <details><summary className="cursor-pointer text-xs">Metadata categories ({groups.length})</summary><div className="grid gap-1 text-xs md:grid-cols-2">{Object.keys(METADATA_CATEGORIES).map(g => <label key={g}><input type="checkbox" checked={groups.includes(g)} onChange={e => setGroups(old => e.target.checked ? [...old, g] : old.filter(x => x !== g))} /> {g}</label>)}</div></details>}
      <div className="space-y-3 rounded border border-hairline p-3 text-xs">
        <label>Export destination <select className="rounded border p-1" aria-label="Package destination" value={destination.destination} onChange={e => setDestination({ ...destination, destination: e.target.value as "local" | "s3" })}><option value="local">Download ZIP to this computer</option><option value="s3">S3 bucket and folder</option></select></label>
        {destination.destination === "s3" ? <>
          <div className="grid gap-3 md:grid-cols-2">
            <label>Bucket <input className="mt-1 block w-full rounded border p-2" list="manual-export-buckets" aria-label="Export bucket" placeholder="Choose or enter a bucket" value={destination.bucket} onChange={e => setDestination({ ...destination, bucket: e.target.value, prefix: "" })} /><datalist id="manual-export-buckets">{buckets.map(bucket => <option key={bucket} value={bucket}/>)}</datalist></label>
            <label>Folder (new or existing) <input className="mt-1 block w-full rounded border p-2" aria-label="Export prefix" placeholder="Folder path, or leave blank for bucket root" value={destination.prefix} onChange={e => setDestination({ ...destination, prefix: e.target.value.replace(/^\/+/, "") })} /></label>
          </div>
          {destination.bucket && <Track2S3FolderBrowser key={destination.bucket} folders={folders} prefix={destination.prefix} selected={destination.prefix} onNavigate={prefix => setDestination({...destination, prefix})} onSelect={prefix => setDestination({...destination, prefix})} allowCreate />}
          {folderError && <p role="status">{folderError}</p>}
          <label>Export as <select className="rounded border p-1" aria-label="Package format" value={destination.format} onChange={e => setDestination({ ...destination, format: e.target.value as "zip" | "folder" })}><option value="zip">ZIP file</option><option value="folder">Files directly into a folder</option></select></label>
          <p className="break-all">Destination: s3://{destination.bucket || "[choose bucket]"}/{destination.prefix.replace(/\/+$/, "")}{destination.prefix ? "/" : ""}</p>
          <p>Each model gets its own uniquely named {destination.format === "zip" ? "ZIP file" : "subfolder"} inside this folder.</p>
        </> : <p>Your browser controls the download location. Choose S3 bucket and folder to export ZIP files or unpacked files into a specific folder.</p>}
      </div>
      <button className={button} disabled={invalid || !ranked.some(r => chosen.includes(r.model_package_arn))} onClick={() => void review(chosen)}>{busy ? "Checking availability..." : "Review selected packages"}</button>
      <p className="text-xs text-muted">Package options are saved with the named file using Save at the top of this page.</p>
    </fieldset>
    {notes.map((note, i) => <p className="text-xs" key={i}>{note}</p>)}
    {preview.length > 0 && <div className="space-y-2 rounded border border-hairline p-3 text-xs">
      <p className="font-medium">Ready to export these exact versions:</p>
      <table className="w-full text-left" aria-label="Package storage estimate"><thead><tr><th>Model / run</th><th>Files</th><th>Estimated package size</th><th>Suggested space</th></tr></thead>
        <tbody>{preview.map(t => <tr key={t.row.model_package_arn} className="border-t border-hairline"><td>{modelLabel(t.row)} / {t.run ? `Run ${t.run.run_number}` : "Registered checkpoint"}</td><td>{t.estimate?.files ?? "Unknown"}</td><td>{t.estimate?.estimated_bytes != null ? bytes(t.estimate.estimated_bytes) : `Unknown${t.estimate ? ` (${t.estimate.unknown_files} file sizes unavailable; ${bytes(t.estimate.source_bytes)} known)` : ""}`}</td><td>{t.estimate?.recommended_bytes != null ? bytes(t.estimate.recommended_bytes) : "Unknown"}</td></tr>)}</tbody>
      </table>
      <p className="font-medium">Total suggested space: {sizeComplete ? bytes(totalSpace) : "Unknown — one or more packages could not be fully measured"}</p>
      <p>Includes estimated generated files and 10% headroom. ZIP exports are not compressed. Each model has its own package, so shared data is counted for each package. This estimates required storage; it does not check free space.</p>
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
