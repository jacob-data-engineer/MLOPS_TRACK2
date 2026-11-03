import { useEffect, useState } from "react";
import { Card } from "./Card";
import { ApiError, getTrack2CatalogMetadata, inspectTrack2CatalogMetadata, startTrack2CatalogExport,
  getTrack2PullControl, saveTrack2PullConfig,
  type CatalogMetadata, type CatalogExportDestination } from "../lib/api";

function saveFile(name: string, body: string, type = "application/json") {
  const url = URL.createObjectURL(new Blob([body], { type }));
  const link = document.createElement("a"); link.href = url; link.download = name;
  document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function Track2CatalogMetadata({ selection, destination, onExported }: {
  selection: { kind: "run" | "version"; id: string; label: string } | null;
  destination: CatalogExportDestination; onExported: () => void;
}) {
  const [metadata, setMetadata] = useState<CatalogMetadata | null>(null);
  const [opened, setOpened] = useState<CatalogMetadata | null>(null);
  const [openedName, setOpenedName] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [images, setImages] = useState(true);
  const [model, setModel] = useState(true);
  const [search, setSearch] = useState("");
  const [reload, setReload] = useState(0);
  const [settingsLoaded, setSettingsLoaded] = useState(false);
  useEffect(() => {
    let active = true;
    getTrack2PullControl().then((value) => {
      if (!active) return;
      setImages(value.config.catalog_export?.include_images ?? true);
      setModel(value.config.catalog_export?.include_model ?? true);
    }).catch(() => { if (active) setError("Could not load export preferences. Default choices are shown."); })
      .finally(() => { if (active) setSettingsLoaded(true); });
    return () => { active = false; };
  }, []);

  async function saveOptions(nextImages: boolean, nextModel: boolean) {
    setBusy("settings"); setError("");
    try {
      await saveTrack2PullConfig({ catalog_export: { include_images: nextImages, include_model: nextModel } });
      setImages(nextImages); setModel(nextModel);
    } catch (err) { setError(err instanceof ApiError ? err.detail : "Could not save export preferences."); }
    finally { setBusy(""); }
  }
  useEffect(() => {
    let active = true; setMetadata(null); setError("");
    if (selection) getTrack2CatalogMetadata(selection.kind, selection.id)
      .then((value) => { if (active) setMetadata(value); })
      .catch((err) => { if (active) setError(err instanceof ApiError ? err.detail : "Could not read metadata."); });
    return () => { active = false; };
  }, [selection?.kind, selection?.id, reload]);

  async function prepare() {
    if (!selection) return;
    setBusy("export"); setError("");
    try {
      await startTrack2CatalogExport(selection.kind, selection.id, { ...destination,
        include_metadata: true, include_images: images, include_model: model && selection.kind === "run" });
      onExported();
    } catch (err) { setError(err instanceof ApiError ? err.detail : "Could not prepare the metadata package."); }
    finally { setBusy(""); }
  }

  async function open(file: File) {
    setBusy("read"); setError("");
    try {
      if (file.size > 512 * 1024 * 1024) throw new Error("Choose a file up to 512 MB, or extract and open metadata.json.");
      const result = await inspectTrack2CatalogMetadata(file);
      setOpened(result); setOpenedName(file.name);
    } catch (err) { setError(err instanceof ApiError ? err.detail : err instanceof Error ? err.message : "Could not read metadata."); }
    finally { setBusy(""); }
  }

  const view = opened || metadata;
  const button = "rounded border border-black/15 bg-page px-3 py-1.5 text-xs font-medium disabled:opacity-40";
  return <>
    <Card title="Metadata and complete export" collapsible={false}
      caption="Export recorded model, dataset, training, evaluation and provenance information. Unrecorded fields are marked unavailable.">
      <p className="mb-3 text-sm">{selection ? `Selected: ${selection.label}` : "No run or dataset version selected. Select a row above."}</p>
      <p className="mb-3 text-xs text-ink-muted">Includes metadata.json, metrics.json, training-history.json, data.yaml and manifest.csv.
        The manifest records missing files and SHA-256 checksums for exported files. A published model is copied into a new .pt with embedded metadata and dataset manifest.</p>
      <div className="mb-3 flex flex-wrap gap-4 text-xs">
        <label><input type="checkbox" checked={images} disabled={!!busy || !settingsLoaded} onChange={(e) => void saveOptions(e.target.checked, model)} /> Include actual images and labels</label>
        <label><input type="checkbox" checked={model} disabled={selection?.kind !== "run" || !!busy || !settingsLoaded}
          onChange={(e) => void saveOptions(images, e.target.checked)} /> Include model with embedded metadata</label>
      </div>
      {selection?.kind === "run" && metadata && !metadata.checkpoint_available && <p className="mb-3 text-xs text-ink-muted">No published model is recorded for this run. The package will contain the available metadata and selected dataset files.</p>}
      <p className="mb-3 break-all text-xs">Destination: {destination.destination === "local" ? "Local PC — ZIP download" : `s3://${destination.bucket || "(choose bucket above)"}/${destination.prefix} (${destination.format})`}</p>
      <div className="flex flex-wrap gap-3">
        <button type="button" className={button} disabled={!metadata || !!busy}
          onClick={() => metadata && saveFile(`${metadata.label}-metadata.json`, JSON.stringify(metadata, null, 2))}>Download metadata JSON</button>
        <button type="button" className={button} disabled={!metadata || !settingsLoaded || !!busy || (destination.destination === "s3" && !destination.bucket)}
          onClick={() => void prepare()}>{busy === "export" ? "Preparing…" : "Prepare complete export"}</button>
      </div>
      <p className="mt-2 text-xs text-ink-muted">Uses the destination selected above. Track progress and stop a running build in Exports. Existing model files are preserved.</p>
    </Card>

    <Card title="Metadata information viewer" collapsible={false}
      caption="Read the selected catalog record, or open metadata.json, an enriched .pt or a complete export ZIP. The viewer reads metadata without loading the model.">
      <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
        <label className={button}>Open exported file <input aria-label="Open metadata JSON, PT or ZIP" className="ml-2" type="file" accept=".json,.pt,.zip"
          disabled={!!busy} onChange={(e) => { const file = e.target.files?.[0]; if (file) void open(file); e.target.value = ""; }} /></label>
        <button type="button" className={button} disabled={!!busy} onClick={() => { setOpened(null); setOpenedName(""); setReload((n) => n + 1); }}>Show selected catalog record</button>
        {view && <button type="button" className={button} onClick={() => saveFile(`${view.label}-metadata.json`, JSON.stringify(view, null, 2))}>Save displayed information</button>}
      </div>
      {error && <p role="alert" className="mb-3 text-sm text-critical">{error}</p>}
      {busy === "read" && <p role="status">Reading metadata…</p>}
      {!view ? <p className="text-sm text-ink-muted">{selection ? "Loading recorded information…" : "No run selected. Select a run or open an exported file."}</p> : <>
        <p className="mb-2 text-sm">{opened ? `File: ${openedName}` : `Catalog: ${selection?.label}`} — captured {view.captured_at}</p>
        <input aria-label="Search metadata" placeholder="Search metadata fields or values" className="mb-3 w-full rounded border border-hairline bg-page p-2 text-sm"
          value={search} onChange={(e) => setSearch(e.target.value)} />
        {Object.entries(view.groups).filter(([name, value]) => `${name} ${JSON.stringify(value)}`.toLowerCase().includes(search.toLowerCase())).map(([name, value]) => (
          <details key={name} className="mb-2 rounded border border-hairline p-3">
            <summary className="cursor-pointer text-sm font-medium">{name}{value && typeof value === "object" && Object.keys(value).length === 0 ? " — Not recorded" : ""}</summary>
            <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words text-xs">{JSON.stringify(value, null, 2)}</pre>
          </details>
        ))}
        <details className="mb-2 rounded border border-hairline p-3"><summary className="cursor-pointer text-sm font-medium">Complete metadata document</summary>
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
