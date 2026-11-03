import { useEffect, useState, type ReactNode } from "react";
import { Card } from "./Card";
import { ApiError, getTrack2CatalogMetadata, inspectTrack2CatalogMetadata, startTrack2CatalogExport,
  exportTrack2CatalogWeights, exportTrack2CatalogMetadata,
  getTrack2PullControl, saveTrack2PullConfig,
  type CatalogMetadata, type CatalogExportDestination } from "../lib/api";

function saveFile(name: string, body: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function MetadataTable({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="text-ink-muted">Not recorded</span>;
  if (typeof value !== "object") return <span className="break-all">{typeof value === "boolean" ? (value ? "Yes" : "No") : String(value)}</span>;
  const entries = Object.entries(value);
  if (!entries.length) return <p className="mt-2 text-ink-muted">Not recorded</p>;
  return <div className="mt-2 overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th className="p-2">Field</th><th className="p-2">Value</th></tr></thead>
    <tbody>{entries.map(([key, item]) => <tr key={key} className="border-t border-hairline"><th className="w-1/3 p-2 align-top font-medium">{key.replaceAll("_", " ")}</th>
      <td className="p-2 align-top">{item && typeof item === "object" ? <details><summary className="cursor-pointer">View {Array.isArray(item) ? `${item.length} items` : "details"}</summary><MetadataTable value={item} /></details> : <MetadataTable value={item} />}</td></tr>)}</tbody></table></div>;
}

type Output = "model" | "dataset" | "metadata" | "complete";
const descriptions: Record<string, string> = {
  Identity: "Run, model and project identifiers", Timestamps: "Submission, start and completion times",
  "Model and classes": "Architecture, model version and class mapping", "Starting checkpoint": "Base model and source checkpoint",
  "Dataset and preparation": "Dataset versions, sources and preparation settings", "Training settings": "Epochs, batch size, seed and parameters",
  "Training history": "Recorded epoch history and best epoch", Evaluation: "Precision, recall, mAP and class results",
  "Inference settings": "Recorded thresholds and prediction settings", "Software and code": "Software versions and build identifiers",
  Infrastructure: "Execution and compute information", "Registry and deployment": "Model package and recorded approval status",
  "Governance and notes": "Ownership, intended use and notes", "File provenance": "Source addresses, splits and file inventory",
};

export function Track2CatalogMetadata({ selection, destination, onExported, renderDestination, children }: {
  selection: { kind: "run" | "version"; id: string; label: string } | null;
  destination: CatalogExportDestination; onExported: () => void;
  renderDestination: (showFormat: boolean) => ReactNode; children?: ReactNode;
}) {
  const [metadata, setMetadata] = useState<CatalogMetadata | null>(null);
  const [opened, setOpened] = useState<CatalogMetadata | null>(null);
  const [openedName, setOpenedName] = useState("");
  const [error, setError] = useState("");
  const [readError, setReadError] = useState("");
  const [message, setMessage] = useState("");
  const [busy, setBusy] = useState("");
  const [output, setOutput] = useState<Output>("complete");
  const [includeImages, setIncludeImages] = useState(true);
  const [includeModel, setIncludeModel] = useState(true);
  const [selected, setSelected] = useState<string[] | null>(null);
  const [search, setSearch] = useState("");
  const [reload, setReload] = useState(0);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  useEffect(() => {
    let active = true;
    getTrack2PullControl().then((value) => {
      if (!active) return;
      setSelected(value.config.catalog_pt_metadata?.selected_groups ?? null);
      setIncludeImages(value.config.catalog_export?.include_images ?? true);
      setIncludeModel(value.config.catalog_export?.include_model ?? true);
      const saved = value.config.catalog_export?.output_type;
      if (saved && ["model", "dataset", "metadata", "complete"].includes(saved)) setOutput(saved as Output);
    }).catch(() => { if (active) setError("Could not load export preferences. Default choices are shown."); })
      .finally(() => { if (active) setSettingsLoaded(true); });
    return () => { active = false; };
  }, []);
  useEffect(() => {
    let active = true; setMetadata(null); setError(""); setMessage("");
    if (selection) getTrack2CatalogMetadata(selection.kind, selection.id)
      .then((value) => { if (active) setMetadata(value); })
      .catch((err) => { if (active) setError(err instanceof ApiError ? err.detail : "Could not read metadata."); });
    return () => { active = false; };
  }, [selection?.kind, selection?.id, reload]);
  const available = metadata ? Object.keys(metadata.groups).filter((name) => !metadata.unavailable.includes(name)) : [];
  const chosen = (selected ?? available).filter((name) => available.includes(name));
  const mode = output === "model" && selection?.kind === "version" ? "complete" : output;
  async function choose(names: string[]) {
    setBusy("settings"); setError("");
    try { await saveTrack2PullConfig({ catalog_pt_metadata: { selected_groups: names } }); setSelected(names); }
    catch (err) { setError(err instanceof ApiError ? err.detail : "Could not save metadata choices."); }
    finally { setBusy(""); }
  }
  async function chooseOutput(value: Output, images = includeImages, model = includeModel) {
    setBusy("settings"); setError("");
    try { await saveTrack2PullConfig({ catalog_export: { output_type: value, include_images: images, include_model: model } }); setOutput(value); setIncludeImages(images); setIncludeModel(model); }
    catch (err) { setError(err instanceof ApiError ? err.detail : "Could not save output choice."); }
    finally { setBusy(""); }
  }
  async function prepare() {
    if (!selection || !metadata) return;
    setBusy("export"); setError(""); setMessage("");
    try {
      if (mode === "model") {
        const result = await exportTrack2CatalogWeights(selection.id, { ...destination, metadata_groups: chosen });
        if (result.url) window.location.assign(result.url);
        setMessage(result.uri ? `Exported to ${result.uri}` : "Model download started.");
      } else if (mode === "metadata") {
        const result = await exportTrack2CatalogMetadata(selection.kind, selection.id, { ...destination, metadata_groups: chosen });
        if (result.url) window.location.assign(result.url);
        setMessage(result.uri ? `Exported to ${result.uri}` : "Metadata download started.");
      } else {
        const result = await startTrack2CatalogExport(selection.kind, selection.id, { ...destination,
          include_metadata: mode === "complete", include_images: mode === "dataset" || includeImages,
          include_model: mode === "complete" && includeModel && selection.kind === "run" && metadata.checkpoint_available,
          metadata_groups: chosen });
        onExported(); setMessage(`Export ${result.label} started. Follow progress in Export history below.`);
      }
    } catch (err) { setError(err instanceof ApiError ? err.detail : "Export failed. Please retry."); }
    finally { setBusy(""); }
  }
  async function open(file: File) {
    setBusy("read"); setReadError("");
    try {
      if (file.size > 512 * 1024 * 1024) throw new Error("Choose a file up to 512 MB, or extract and open metadata.json.");
      const result = await inspectTrack2CatalogMetadata(file); setOpened(result); setOpenedName(file.name);
    } catch (err) { setReadError(err instanceof ApiError ? err.detail : err instanceof Error ? err.message : "Could not read metadata."); }
    finally { setBusy(""); }
  }
  const view = opened || metadata;
  const button = "rounded border border-black/15 bg-page px-3 py-1.5 text-xs font-medium disabled:opacity-40";
  const format = mode === "model" ? ".pt" : mode === "metadata" ? "JSON" : destination.format === "folder" && destination.destination === "s3" ? "Folder" : "ZIP";
  return <>
    <Card title="Export" collapsible={false} caption="Choose the contents, metadata and destination, then review your export.">
      <fieldset disabled={!!busy || !settingsLoaded || !selection} className="space-y-4">
        <div><p className="mb-2 font-medium">1. What to export?</p><div className="flex flex-wrap gap-3">
          {([["model", "Model (.pt)"], ["dataset", "Dataset files"], ["metadata", "Metadata (JSON)"], ["complete", "Complete package"]] as [Output,string][]).map(([value,label]) =>
            <label key={value} className="rounded border border-hairline p-3 text-sm"><input type="radio" name="catalog-output" value={value} checked={mode === value}
              disabled={value === "model" && (selection?.kind !== "run" || !metadata?.checkpoint_available)} onChange={() => void chooseOutput(value)} /> {label}</label>)}
        </div></div>
        {mode === "complete" && <div className="flex flex-wrap gap-4 text-sm">
          <label><input type="checkbox" checked={includeImages} onChange={(e) => void chooseOutput(output, e.target.checked, includeModel)} /> Include images and labels</label>
          <label><input type="checkbox" checked={includeModel} disabled={selection?.kind !== "run" || !metadata?.checkpoint_available} onChange={(e) => void chooseOutput(output, includeImages, e.target.checked)} /> Include model (.pt)</label>
        </div>}
        {mode !== "dataset" && <div><p className="mb-2 font-medium">2. Metadata to include ({chosen.length} selected)</p>
          <div className="mb-3 flex gap-3 text-xs"><button type="button" className="underline" onClick={() => void choose(available)}>Select all available</button>
            <button type="button" className="underline" onClick={() => void choose([])}>Clear all</button></div>
          <div className="grid gap-2 md:grid-cols-2 lg:grid-cols-3">{Object.entries(metadata?.groups || {}).map(([name, value]) =>
            <div key={name} className="rounded border border-hairline p-2 text-xs"><label className="flex gap-2"><input type="checkbox" checked={chosen.includes(name)} disabled={!available.includes(name)}
              onChange={(e) => void choose(e.target.checked ? [...chosen, name] : chosen.filter((n) => n !== name))} /><span className="font-medium">{name}{!available.includes(name) ? " — Not recorded" : ""}</span></label>
              <p className="mt-1 text-ink-muted">{descriptions[name]}</p><details className="mt-1"><summary className="cursor-pointer">Preview fields</summary><MetadataTable value={value} /></details></div>)}</div>
          <p className="mt-2 text-xs text-ink-muted">The same selection applies to model, JSON and complete-package metadata. Existing checkpoint contents are preserved. Dataset packages always retain their layout and file manifest.</p>
        </div>}
        <div><p className="mb-2 font-medium">{mode === "dataset" ? "2" : "3"}. Destination</p>{renderDestination(mode === "dataset" || mode === "complete")}</div>
      </fieldset>
      {error && <div role="alert" className="my-3 text-sm text-critical">{error} <button type="button" className="underline" onClick={() => setReload((n) => n + 1)}>Reload information</button></div>}
      <div className="mt-4 rounded border border-hairline bg-page p-3">
        <p className="mb-2 text-sm"><strong>{selection?.label || "No run selected"}</strong> · {mode} · {format} · {mode === "dataset" ? "Dataset files" : `${chosen.length} metadata categories`}</p>
        <p className="mb-3 break-all text-xs">{destination.destination === "local" ? "Download to local PC" : `Destination: s3://${destination.bucket || "(choose bucket)"}/${destination.prefix}`}</p>
        {mode === "complete" && metadata && !metadata.checkpoint_available && <p className="mb-2 text-xs">No published model is recorded; this package contains dataset files and selected metadata.</p>}
        <button type="button" className="rounded bg-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
          disabled={!metadata || !settingsLoaded || !!busy || (destination.destination === "s3" && !destination.bucket) || (mode === "model" && !metadata.checkpoint_available)} onClick={() => void prepare()}>
          {busy === "export" ? "Preparing…" : destination.destination === "local" && (mode === "model" || mode === "metadata") ? `Download ${format}` : `Prepare ${format} export`}</button>
        {message && <p role="status" className="mt-2 break-all text-sm">{message}</p>}
      </div>
    </Card>
    {children}
    <Card title="Metadata information viewer" collapsible={false}
      caption="Read the selected catalog record, or open metadata.json, an enriched .pt or a complete export ZIP. The viewer reads metadata without loading the model.">
      <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
        <label className={button}>Open exported file <input aria-label="Open metadata JSON, PT or ZIP" className="ml-2" type="file" accept=".json,.pt,.zip"
          disabled={!!busy} onChange={(e) => { const file = e.target.files?.[0]; if (file) void open(file); e.target.value = ""; }} /></label>
        <button type="button" className={button} disabled={!!busy} onClick={() => { setOpened(null); setOpenedName(""); setReload((n) => n + 1); setReadError(""); }}>Show selected catalog record</button>
        {view && <button type="button" className={button} onClick={() => saveFile(`${view.label}-metadata.json`, JSON.stringify(view, null, 2))}>Save displayed information</button>}
      </div>
      {readError && <p role="alert" className="mb-3 text-sm text-critical">{readError}</p>}
      {busy === "read" && <p role="status">Reading metadata…</p>}
      {!view ? <p className="text-sm text-ink-muted">{selection ? "Loading recorded information…" : "No run selected. Select a run or open an exported file."}</p> : <>
        <p className="mb-2 text-sm">{opened ? `File: ${openedName}` : `Catalog: ${selection?.label}`} — captured {view.captured_at}</p>
        <input aria-label="Search metadata" placeholder="Search metadata fields or values" className="mb-3 w-full rounded border border-hairline bg-page p-2 text-sm"
          value={search} onChange={(e) => setSearch(e.target.value)} />
        {Object.entries(view.groups).filter(([name, value]) => `${name} ${JSON.stringify(value)}`.toLowerCase().includes(search.toLowerCase())).map(([name, value]) => (
          <details key={name} className="mb-2 rounded border border-hairline p-3">
            <summary className="cursor-pointer text-sm font-medium">{name}{value && typeof value === "object" && Object.keys(value).length === 0 ? " — Not recorded" : ""}</summary>
            <MetadataTable value={value} />
          </details>
        ))}
        <details className="mb-2 rounded border border-hairline p-3"><summary className="cursor-pointer text-sm font-medium">View raw JSON (advanced)</summary>
          <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs">{JSON.stringify({ ...view, file_manifest: view.file_manifest ? "See file manifest below" : undefined }, null, 2)}</pre>
        </details>
        {view.file_manifest && <details className="rounded border border-hairline p-3"><summary className="cursor-pointer text-sm font-medium">File manifest</summary>
          <button type="button" className={button} onClick={() => saveFile(`${view.label}-manifest.csv`, view.file_manifest!, "text/csv")}>Download full manifest CSV</button>
          <p className="mt-2 text-xs">Preview of the first 200 lines. Download the CSV to read all files.</p>
          <pre className="mt-2 max-h-96 overflow-auto text-xs">{view.file_manifest.split("\n").slice(0, 200).join("\n")}</pre>
        </details>}
      </>}
    </Card>
  </>;
}
