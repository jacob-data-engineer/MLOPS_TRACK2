import { useEffect, useRef, useState, type ReactNode } from "react";
import { Card } from "./Card";
import { Track2ExportProgress, Track2ExportResult, sameChoice, type ExportChoice } from "./Track2ExportProgress";
import { ApiError, getTrack2CatalogMetadata, inspectTrack2CatalogMetadata, startTrack2CatalogExport,
  exportTrack2CatalogWeights, exportTrack2CatalogMetadata, getTrack2CatalogExportUrl,
  getTrack2CatalogExports, checkCatalogDataset, getTrack2PullControl, saveTrack2PullConfig,
  type CatalogMetadata, type CatalogExportDestination, type DatasetAvailability,
  type Track2CatalogExport } from "../lib/api";
import { formatSgt } from "../lib/format";

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
/** The metadata categories, and what each holds. Exported so Model export
 * offers the same ones Catalog export does. */
export const METADATA_CATEGORIES: Record<string, string> = {
  Identity: "Run, model and project identifiers", Timestamps: "Submission, start and completion times",
  "Model and classes": "Architecture, model version and class mapping", "Starting checkpoint": "Base model and source checkpoint",
  "Dataset and preparation": "Dataset versions, sources and preparation settings", "Training settings": "Epochs, batch size, seed and parameters",
  "Training history": "Recorded epoch history and best epoch", Evaluation: "Precision, recall, mAP and class results",
  "Inference settings": "Recorded thresholds and prediction settings", "Software and code": "Software versions and build identifiers",
  Infrastructure: "Execution and compute information", "Registry and deployment": "Model package and recorded approval status",
  "Governance and notes": "Ownership, intended use and notes", "File provenance": "Source addresses, splits and file inventory",
};

/** Exactly what the catalog export acts on, as recorded. Any field may be
 * missing on an older record; it then says so rather than guessing. */
export type CatalogTarget = {
  runNumber?: number | null; modelName?: string | null; modelVersion?: string | null;
  datasetLabel?: string | null; datasetId?: string | null;
  train?: number | null; val?: number | null; images?: number | null;
};

export type CatalogSelection = { kind: "run" | "version"; id: string; label: string; target?: CatalogTarget };

type Load<T> = { status: "idle" | "loading" | "ready" | "failed"; value: T; error: string; at: string };

const AVAILABILITY_LABEL: Record<DatasetAvailability["state"], string> = {
  ready: "Ready to rebuild", unavailable: "Unavailable", legacy: "Legacy record", check_failed: "Check failed",
};

/** CATALOG EXPORT AND METADATA for ONE selected run or dataset version.
 *
 * A separate workflow from the model export above it on the page: nothing
 * chosen there selects anything here, and nothing selected here changes it.
 * `refreshKey` is bumped by the page's Refresh all: metadata, this record's
 * exports and any availability result are read again, the selection and
 * the export choices stay as they are. */
export function Track2CatalogMetadata({ selection, destination, renderDestination, children, refreshKey = 0 }: {
  selection: CatalogSelection | null;
  destination: CatalogExportDestination;
  renderDestination: (showFormat: boolean) => ReactNode; children?: ReactNode;
  refreshKey?: number;
}) {
  const key = selection ? `${selection.kind}:${selection.id}` : "";
  const [meta, setMeta] = useState<Load<CatalogMetadata | null>>({ status: "idle", value: null, error: "", at: "" });
  const [exps, setExps] = useState<Load<Track2CatalogExport[] | null>>({ status: "idle", value: null, error: "", at: "" });
  const [opened, setOpened] = useState<CatalogMetadata | null>(null);
  const [openedName, setOpenedName] = useState("");
  const [closedNotice, setClosedNotice] = useState("");
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
  const [startedExport, setStartedExport] = useState<{ id: string; key: string } | null>(null);
  const [avail, setAvail] = useState<Load<DatasetAvailability | null> & { forKey: string }>(
    { status: "idle", value: null, error: "", at: "", forKey: "" });
  // The selection a response belongs to. A slow answer for Run 3 must not
  // land after Run 7 was picked.
  const current = useRef(key);
  current.current = key;

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

  // A new record: the external file and the last availability answer
  // belonged to the old one. Say so rather than keep showing them.
  const previousKey = useRef(key);
  useEffect(() => {
    if (previousKey.current === key) return;
    previousKey.current = key;
    if (opened) {
      setClosedNotice(`The external file ${openedName} was closed because ${selection?.label ?? "another record"} was selected.`);
      setOpened(null); setOpenedName("");
    }
    setMessage(""); setError("");
    setMeta({ status: "idle", value: null, error: "", at: "" });
    setExps({ status: "idle", value: null, error: "", at: "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  // This record's metadata. Kept on screen through a refresh; a failure is
  // a failure, never an empty record.
  useEffect(() => {
    if (!selection) return;
    const mine = key;
    setMeta((m) => ({ ...m, status: "loading", error: "" }));
    getTrack2CatalogMetadata(selection.kind, selection.id)
      .then((value) => { if (current.current === mine) setMeta({ status: "ready", value, error: "", at: new Date().toISOString() }); })
      .catch((err) => { if (current.current === mine) setMeta((m) => ({ ...m, status: "failed", error: err instanceof ApiError ? err.detail : "Could not read the metadata." })); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, reload, refreshKey]);

  // This record's own exports, newest first -- not the newest 50 overall,
  // so its latest export stays reachable however many others came since.
  function loadExports() {
    if (!selection) return;
    const mine = key;
    setExps((e) => ({ ...e, status: "loading", error: "" }));
    getTrack2CatalogExports({ kind: selection.kind, id: selection.id }, 5)
      .then((r) => { if (current.current === mine) setExps({ status: "ready", value: r.exports, error: "", at: new Date().toISOString() }); })
      .catch((err) => { if (current.current === mine) setExps((e) => ({ ...e, status: "failed", error: err instanceof ApiError ? err.detail : "Could not read this record's exports." })); });
  }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(loadExports, [key, refreshKey]);

  const list = exps.value ?? [];
  const tracked = (startedExport?.key === key && list.find((e) => e.id === startedExport.id)) || list[0] || null;
  const starting = startedExport?.key === key && !list.some((e) => e.id === startedExport.id);
  const building = !!tracked && (tracked.state === "building" || tracked.state === "stopping");
  // Every 5 s while this record's export is running, or until a just-started
  // one appears.
  useEffect(() => {
    if (!building && !starting) return;
    const timer = window.setInterval(loadExports, 5000);
    return () => window.clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [building, starting, key]);

  function checkAvailability() {
    if (!selection) return;
    const mine = key;
    setAvail({ status: "loading", value: null, error: "", at: "", forKey: mine });
    checkCatalogDataset(selection.kind, selection.id)
      .then((value) => { if (current.current === mine) setAvail({ status: "ready", value, error: "", at: value.checked_at, forKey: mine }); })
      .catch((err) => { if (current.current === mine) setAvail({ status: "failed", value: null, error: err instanceof ApiError ? err.detail : "The check did not answer.", at: new Date().toISOString(), forKey: mine }); });
  }
  const availability = avail.forKey === key ? avail : null;

  const metadata = meta.value;
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
  // What a package export started NOW would be made of, to compare with the
  // last one. A model or JSON download is not a package export at all, so
  // any package shown then is a previous one.
  const choice: ExportChoice = {
    include_metadata: mode === "complete", include_images: mode === "dataset" || includeImages,
    include_model: mode === "complete" && includeModel && selection?.kind === "run" && !!metadata?.checkpoint_available,
    metadata_groups: chosen, destination: destination.destination,
    format: destination.destination === "local" ? "zip" : destination.format,
  };
  const packageMode = mode === "dataset" || mode === "complete";
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
          include_metadata: choice.include_metadata, include_images: choice.include_images,
          include_model: choice.include_model, metadata_groups: chosen });
        setStartedExport({ id: result.id, key });
        setMessage(`Export ${result.id} started for ${selection.label}.`);
        loadExports();
      }
    } catch (err) { setError(err instanceof ApiError ? err.detail : "Export failed. Please retry."); }
    finally { setBusy(""); }
  }
  async function open(file: File) {
    setBusy("read"); setReadError(""); setClosedNotice("");
    try {
      if (file.size > 512 * 1024 * 1024) throw new Error("Choose a file up to 512 MB, or extract and open metadata.json.");
      const result = await inspectTrack2CatalogMetadata(file); setOpened(result); setOpenedName(file.name);
    } catch (err) { setReadError(err instanceof ApiError ? err.detail : err instanceof Error ? err.message : "Could not read metadata."); }
    finally { setBusy(""); }
  }
  const view = opened || metadata;
  const button = "rounded border border-black/15 bg-page px-3 py-1.5 text-xs font-medium disabled:opacity-40";
  const format = mode === "model" ? ".pt" : mode === "metadata" ? "JSON" : destination.format === "folder" && destination.destination === "s3" ? "Folder" : "ZIP";
  const t: CatalogTarget = selection?.target ?? {};
  const count = (n: number | null | undefined) => (n == null ? "not recorded" : n.toLocaleString());
  // Only an export made with exactly the choices above is "the" export of
  // them; anything else is offered as the previous one.
  const isCurrent = !!tracked && packageMode && sameChoice(tracked, choice) === true;

  return <>
    <section id="catalog-export" className="scroll-mt-48">
    <Card title="Catalog export: the selected run or dataset version" collapsible={false}
      caption="A separate workflow from Model export above. Versions chosen in Comparison or Model export do not select anything here, and selecting a row in the catalog does not change them.">
      {/* What this export acts on, as recorded -- nothing inferred. */}
      <div className="mb-4 rounded border border-hairline bg-page p-3 text-xs">
        {!selection ? <p className="text-ink-muted">No run or dataset version selected. Select a row in the Data Catalog above.</p> : (
          <dl className="tabular grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5">
            <dt className="text-ink-muted">Export target</dt><dd className="font-medium text-ink">{selection.label}</dd>
            {selection.kind === "run" && <><dt className="text-ink-muted">Run number</dt><dd>{t.runNumber ?? "not recorded"}</dd>
              <dt className="text-ink-muted">Model version</dt><dd>{[t.modelName, t.modelVersion].filter((v) => v && v !== "--").join(" ") || "not registered"}</dd></>}
            <dt className="text-ink-muted">Dataset</dt><dd className="break-all">{t.datasetLabel || "not recorded"}{t.datasetId ? ` (ID ${t.datasetId})` : ""}</dd>
            <dt className="text-ink-muted">Train / validation images</dt><dd>{count(t.train)} / {count(t.val)}</dd>
          </dl>
        )}
      </div>

      {/* 0. Could it be rebuilt? Read-only. */}
      <div className="mb-4 rounded border border-hairline p-3 text-xs">
        <div className="flex flex-wrap items-center gap-3">
          <button type="button" className={button} disabled={!selection || availability?.status === "loading"} onClick={checkAvailability}>
            {availability?.status === "loading" ? "Checking..." : "Check dataset availability"}
          </button>
          <span className="font-medium text-ink">
            {!selection ? "Not checked" : !availability ? "Not checked"
              : availability.status === "loading" ? "Checking"
              : availability.status === "failed" ? "Check failed"
              : availability.value ? AVAILABILITY_LABEL[availability.value.state] : "Not checked"}
          </span>
          {availability?.at && availability.status !== "loading" && <span className="text-ink-muted">checked {formatSgt(availability.at)}</span>}
        </div>
        {availability?.status === "ready" && availability.value && (
          <p className={`mt-1 ${availability.value.state === "ready" ? "text-ink-secondary" : availability.value.state === "legacy" ? "text-ink-secondary" : "text-critical"}`}>
            {availability.value.detail}
            {availability.value.version_number != null && ` Dataset version ${availability.value.version_number}: ${count(availability.value.train_images)} train / ${count(availability.value.val_images)} validation images.`}
          </p>
        )}
        {availability?.status === "failed" && <p className="mt-1 text-critical" role="alert">{availability.error} This says nothing about whether the files exist. <button type="button" className="underline" onClick={checkAvailability}>Retry</button></p>}
        <p className="mt-1 text-ink-muted">
          Reads only: verifies the SQL file list and every preserved central-store image and label against its recorded ETag.
          Nothing is rebuilt, queued, changed or deleted. Model-class compatibility is checked separately, when a run is queued.
        </p>
      </div>

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
              <p className="mt-1 text-ink-muted">{METADATA_CATEGORIES[name]}</p><details className="mt-1"><summary className="cursor-pointer">Preview fields</summary><MetadataTable value={value} /></details></div>)}</div>
          <p className="mt-2 text-xs text-ink-muted">The same selection applies to model, JSON and complete-package metadata. Existing checkpoint contents are preserved. Dataset packages always retain their layout and file manifest.</p>
        </div>}
        <div><p className="mb-2 font-medium">{mode === "dataset" ? "2" : "3"}. Destination</p>{renderDestination(mode === "dataset" || mode === "complete")}</div>
      </fieldset>
      {error && <div role="alert" className="my-3 text-sm text-critical">{error} <button type="button" className="underline" onClick={() => setReload((n) => n + 1)}>Reload information</button></div>}
      {selection && meta.status === "loading" && !metadata && <p className="my-2 text-xs text-ink-muted" role="status">Loading this record&apos;s metadata...</p>}
      {selection && meta.status === "failed" && <p className="my-2 text-xs text-critical" role="alert">{meta.error} <button type="button" className="underline" onClick={() => setReload((n) => n + 1)}>Retry</button></p>}
      <div className="mt-4 rounded border border-hairline bg-page p-3">
        <p className="mb-2 text-sm"><strong>{selection?.label || "No run selected"}</strong> · {mode} · {format} · {mode === "dataset" ? "Dataset files" : `${chosen.length} metadata categories`}</p>
        <p className="mb-3 break-all text-xs">{destination.destination === "local" ? "Download to local PC" : `Destination: s3://${destination.bucket || "(choose bucket)"}/${destination.prefix}`}</p>
        {mode === "complete" && metadata && !metadata.checkpoint_available && <p className="mb-2 text-xs">No published model is recorded; this package contains dataset files and selected metadata.</p>}
        <button type="button" className="rounded bg-accent px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
          disabled={!metadata || !settingsLoaded || !!busy || (destination.destination === "s3" && !destination.bucket) || (mode === "model" && !metadata.checkpoint_available)} onClick={() => void prepare()}>
          {busy === "export" ? "Preparing…" : destination.destination === "local" && (mode === "model" || mode === "metadata") ? `Download ${format}` : `Prepare ${format} export`}</button>
        {message && <p role="status" className="mt-2 break-all text-sm">{message}</p>}

        {/* This record's latest package export: which one it is, how far it
            got, and whether it matches the choices above. */}
        {selection && exps.status === "loading" && !exps.value && <p className="mt-3 text-xs text-ink-muted" role="status">Reading this record&apos;s exports...</p>}
        {selection && exps.status === "failed" && <p className="mt-3 text-xs text-critical" role="alert">{exps.error} <button type="button" className="underline" onClick={loadExports}>Retry</button></p>}
        {selection && exps.status === "ready" && list.length === 0 && !starting && <p className="mt-3 text-xs text-ink-muted">No package export of {selection.label} yet.</p>}
        {tracked && selection && <Track2ExportResult exp={tracked} source={selection.label} current={packageMode ? choice : { ...choice, destination: "--" }} />}
        {(tracked || starting) && <Track2ExportProgress exp={tracked} starting={starting} onChanged={loadExports} />}
        {tracked && tracked.format !== "folder" && ["ready", "partial"].includes(tracked.state) &&
          <button type="button" className={`${button} mt-3`} onClick={() => {
            void getTrack2CatalogExportUrl(tracked.id).then((result) => window.location.assign(result.url))
              .catch((err) => setError(err instanceof ApiError ? err.detail : "Could not get the download link."));
          }}>{!isCurrent ? `Download previous ZIP (${tracked.id})` : tracked.state === "partial" ? "Download incomplete ZIP" : "Download ZIP"}</button>}
        {exps.at && <p className="mt-2 text-[11px] text-ink-muted">Export status read {formatSgt(exps.at)}.</p>}
      </div>
    </Card>
    </section>
    {children}
    <section id="catalog-metadata" className="scroll-mt-48">
    <Card title="Metadata information viewer" collapsible={false}
      caption="Shows either the selected catalog record or an external file you open -- labelled which. Reads metadata without loading the model.">
      <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
        <label className={button}>Open external file <input aria-label="Open metadata JSON, PT or ZIP" className="ml-2" type="file" accept=".json,.pt,.zip"
          disabled={!!busy} onChange={(e) => { const file = e.target.files?.[0]; if (file) void open(file); e.target.value = ""; }} /></label>
        <button type="button" className={button} disabled={!!busy || (!opened && !selection)} onClick={() => { setOpened(null); setOpenedName(""); setReload((n) => n + 1); setReadError(""); setClosedNotice(""); }}>Show selected catalog record</button>
        {view && <button type="button" className={button} onClick={() => saveFile(`${view.label}-metadata.json`, JSON.stringify(view, null, 2))}>Save displayed information</button>}
      </div>
      {closedNotice && <p className="mb-3 text-xs text-ink-secondary" role="status">{closedNotice}</p>}
      {readError && <p role="alert" className="mb-3 text-sm text-critical">{readError}</p>}
      {busy === "read" && <p role="status">Reading metadata…</p>}
      {!view ? <p className="text-sm text-ink-muted">
          {!selection ? "Nothing selected. Select a run or dataset version in the catalog, or open an external file."
            : meta.status === "failed" ? "The selected record's metadata could not be read (see above)." : "Loading the selected record..."}
        </p> : <>
        <p className="mb-2 rounded border border-hairline bg-page px-2 py-1 text-sm">
          {opened
            ? <><strong>Opened external file:</strong> {openedName} -- not the catalog selection{selection ? ` (${selection.label} is selected)` : ""}</>
            : <><strong>Selected catalog record:</strong> {selection?.label}</>}
          <span className="text-ink-muted"> &middot; captured {formatSgt(view.captured_at)}</span>
        </p>
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
        {meta.at && !opened && <p className="mt-2 text-[11px] text-ink-muted">Record read {formatSgt(meta.at)}.</p>}
      </>}
    </Card>
    </section>
  </>;
}
