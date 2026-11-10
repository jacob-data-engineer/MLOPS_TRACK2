import { useEffect, useRef, useState } from "react";
import { Database } from "lucide-react";
import { Card } from "./Card";
import { Track2PinnedBar } from "./Track2PinnedBar";
import { Track2CentralStorePanel } from "./Track2CentralStorePanel";
import { apiGet, apiPost, ApiError, saveTrack2Retention, saveTrack2PullConfig, getTrack2PullControl, type Track2RetentionConfig } from "../lib/api";
import { post } from "../lib/bus";

type Names = Record<string, string>;
type Source = { id: string; name: string; enabled: boolean; bucket: string; images: string; labels: string; kind: "fixed" | "collected" | "golden"; split: "train" | "val" | "split"; train_percent: number; classes: Names };
type Definition = { version: 2; sources: Source[]; class_ids: Names; filter_to_model: boolean; detector_slug: string; model_classes: Names; schedule: { mode: "off" | "schedule" | "arrival"; every_days: number; at: string; from_date: string } };
type Scan = { sources?: { id: string; name: string; classes: Names; images: number; eligible: number; excluded: number }[]; blockers?: string[]; excluded?: { source_uri: string; reason: string }[]; images?: number; eligible?: number; selected?: number; moved?: number; error?: string; ready?: boolean };
type Job = { id: string; kind: string; status: string; done: number; total: number; error: string; result: Scan; created_at: string };
type Status = { definition: Definition; migration_required: boolean; migration_notes: string[]; selection_id: string; retention: Track2RetentionConfig; excluded_weeks: string[]; weeks: string[]; counts: { available: number; selected: number; train: number; val: number; excluded: Record<string, number> }; sources: { id: string; available: number; selected: number; train: number; val: number; classes: Names }[]; collections: { week: string; available: number; selected: number }[]; coverage: { source_id: string; class_name: string; available: number; selected: number; warning: boolean }[]; last_refresh: Scan; jobs: Job[]; versions: { version_id: string; version_number: number; created_at: string; train_images: number; val_images: number; state: string }[]; next_at: string | null; saved_name: string; schedule_saved: boolean; sources_stale: boolean };
type Member = { source_uri: string; split: string; included: boolean; reason: string; url: string; boxes: { name: string; id: number; x: number; y: number; w: number; h: number }[] };
type Members = { items: Member[]; page: number; pages: number; total: number };
const base = "/track2/preparation";
const input = "rounded border border-hairline bg-page px-2 py-1 text-xs";
const button = "rounded border border-hairline px-3 py-1.5 text-xs disabled:opacity-40";
const when = (v: string) => new Date(v).toLocaleString("en-SG", { timeZone: "Asia/Singapore" });

/** A single definition replaces folder ownership throughout Data Preparation. */
export function Track2DataPreparationWorkflow() {
  const [data, setData] = useState<Status | null>(null);
  const [draft, setDraft] = useState<Definition | null>(null);
  const [policy, setPolicy] = useState<Track2RetentionConfig | null>(null);
  const [excluded, setExcluded] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [dirty, setDirty] = useState(false);
  const [gallery, setGallery] = useState<Members | null>(null);
  const initialized = useRef(false);
  const [sourceMappings, setSourceMappings] = useState<Record<string, string>>({});
  async function refresh(reset = false) {
    const next = await apiGet<Status>(base);
    setData(next);
    if (!initialized.current || reset) {
      setDraft(next.definition); setSourceMappings({}); setPolicy(next.retention); setExcluded(next.excluded_weeks);
      initialized.current = true; setDirty(false);
    }
  }
  useEffect(() => {
    void refresh().catch(e => setError(e instanceof ApiError ? e.detail : String(e)));
    const timer = setInterval(() => void refresh().catch(() => undefined), 5000);
    return () => clearInterval(timer);
  }, []);
  function edit(patch: Partial<Definition>) { setDraft(d => d ? { ...d, ...patch } : d); setDirty(true); }
  function source(id: string, patch: Partial<Source>) { if (draft) edit({ sources: draft.sources.map(s => s.id === id ? { ...s, ...patch } : s) }); }
  function retention(patch: Partial<Track2RetentionConfig>) { setPolicy(p => p ? { ...p, ...patch } : p); setDirty(true); }
  async function persist() {
    if (!draft || !policy) throw new Error("Wait for the saved definition to load.");
    const sources = draft.sources.map(s => {
      if (!(s.id in sourceMappings)) return s;
      const names: Names = {};
      for (const part of sourceMappings[s.id].split(",").filter(v => v.trim())) {
        const [k, ...rest] = part.split("="); const name = rest.join("=").trim();
        if (!/^\d+$/.test(k.trim()) || !name || String(Number(k)) in names) throw new Error("Source mapping must use unique ID=name entries, for example 0=helmet, 2=person.");
        names[String(Number(k))] = name;
      }
      return { ...s, classes: names };
    });
    await apiPost(base + "/definition", { ...draft, sources });
    await saveTrack2Retention(policy);
    const live = (await getTrack2PullControl()).config;
    await saveTrack2PullConfig({ auto_training: { ...live.auto_training, excluded } });
    post("data-sources-changed", { from: "Data Preparation definition" });
  }
  async function run(kind: "scan" | "refresh" | "freeze") {
    setBusy(true); setError(""); setNote("");
    try {
      await persist(); await apiPost(base + "/jobs", { kind });
      setNote("The job is running on the server. You can close this page; progress and results are saved.");
      await refresh();
    } catch (e) { setError(e instanceof ApiError ? e.detail : String(e)); }
    finally { setBusy(false); }
  }
  async function images(page = 1) {
    try { setGallery(await apiGet<Members>(`${base}/members?page=${page}`)); }
    catch (e) { setError(e instanceof ApiError ? e.detail : String(e)); }
  }
  function discoverNames() {
    if (!draft || !data) return;
    const names = new Set<string>(data.sources.flatMap(s => Object.values(s.classes)));
    for (const j of data.jobs) for (const s of j.result.sources ?? []) for (const n of Object.values(s.classes)) names.add(n);
    const mapped = { ...draft.class_ids };
    for (const name of [...names].sort()) if (!Object.values(mapped).includes(name)) {
      let i = 0; while (String(i) in mapped) i++; mapped[String(i)] = name;
    }
    edit({ class_ids: mapped });
  }
  function classId(old: string, next: number) {
    if (!draft || !Number.isInteger(next) || next < 0) return;
    const names = { ...draft.class_ids }; const name = names[old]; const occupant = names[String(next)];
    delete names[old]; names[String(next)] = name;
    if (occupant && String(next) !== old) names[old] = occupant;
    edit({ class_ids: names });
  }
  if (!data || !draft || !policy) return <div><h1 className="text-xl">Data Preparation</h1><p role="status">{error || "Loading the dataset definition..."}</p></div>;
  const active = data.jobs.find(j => j.status === "queued" || j.status === "running");
  const latest = data.jobs[0];
  const sourceName = (id: string) => draft.sources.find(s => s.id === id)?.name || id;
  return <div className="space-y-5">
    <div className="sticky top-0 z-30 -mx-4 -mt-6 border-b border-hairline bg-page px-4 pb-3 pt-6 md:-mx-8 md:px-8">
      <h1 className="mb-2 flex items-center gap-2 text-xl font-semibold"><Database size={20} />Track2 Data Preparation</h1>
      <Card title="Settings and schedule" stateKey="dp settings and schedule" keepMounted summary={data.saved_name || "Unsaved dataset definition"}>
        <Track2PinnedBar scope="dp" beforeSave={persist} onSaved={async () => { await refresh(true); setNote("Definition and refresh schedule saved together."); }} />
        <div className="mt-3 flex flex-wrap items-center gap-3 text-xs">
          <label>Refresh <select className={input} value={draft.schedule.mode} onChange={e => edit({ schedule: { ...draft.schedule, mode: e.target.value as Definition["schedule"]["mode"] } })}>
            <option value="off">Manual only</option><option value="schedule">On a schedule</option><option value="arrival">Check for arrivals every minute</option>
          </select></label>
          {draft.schedule.mode === "schedule" && <>
            <label>Every <input aria-label="Refresh interval in days" className={`${input} w-16`} type="number" min={1} max={365} value={draft.schedule.every_days} onChange={e => edit({ schedule: { ...draft.schedule, every_days: Number(e.target.value) } })} /> days</label>
            <label>At (Singapore time) <input className={input} type="time" value={draft.schedule.at} onChange={e => edit({ schedule: { ...draft.schedule, at: e.target.value } })} /></label>
            <label>Start date <input className={input} type="date" value={draft.schedule.from_date} onChange={e => edit({ schedule: { ...draft.schedule, from_date: e.target.value } })} /></label>
          </>}
        </div>
        <p className="mt-2 text-xs text-ink-secondary">Save captures every source, split, class ID, retention setting and this refresh schedule in the same file. Scheduled refreshes run on EC2 with the page closed. Training is controlled on Multi-Auto Training.</p>
        <p className="mt-1 text-xs">{dirty ? "Unsaved edits — use Save above." : data.schedule_saved ? "Schedule matches the saved definition." : "Use Save to apply this definition’s schedule."} {data.next_at && `Next saved check: ${when(data.next_at)} SGT.`}</p>
      </Card>
    </div>
    {error && <p role="alert" className="rounded border border-critical p-3 text-sm text-critical">{error}</p>}
    {note && <p role="status" className="text-sm">{note}</p>}
    {data.migration_required && <Card title="Review your existing settings">
      <p className="text-sm">Your existing configuration has been translated into the source rows below. Scan sources to check it; Save adopts this definition without moving or deleting the source files.</p>
      <ul className="mt-2 list-disc pl-5 text-xs">{data.migration_notes.map(n => <li key={n}>{n}</li>)}</ul>
    </Card>}
    <Card title="Data sources" step="1" keepMounted caption="Select the data this definition can use. No destination folders are required.">
      <div className="space-y-3">{draft.sources.map(s => <fieldset key={s.id} className="rounded border border-hairline p-3">
        <div className="flex flex-wrap items-center gap-2">
          <input aria-label={`Include ${s.name}`} type="checkbox" checked={s.enabled} onChange={e => source(s.id, { enabled: e.target.checked })} />
          <input aria-label="Source name" className={`${input} flex-1`} value={s.name} onChange={e => source(s.id, { name: e.target.value })} />
          <select aria-label="Source role" className={input} value={s.kind} onChange={e => source(s.id, { kind: e.target.value as Source["kind"] })}><option value="fixed">Existing dataset</option><option value="collected">New reviewed data</option><option value="golden">Golden / curated examples</option></select>
          <button className={button} onClick={() => edit({ sources: draft.sources.filter(r => r.id !== s.id) })}>Remove from definition</button>
        </div>
        <div className="mt-2 grid gap-2 md:grid-cols-3">
          <label className="text-xs">Bucket<input className={`${input} block w-full`} value={s.bucket} placeholder="Default bucket" onChange={e => source(s.id, { bucket: e.target.value })} /></label>
          <label className="text-xs">Images location<input className={`${input} block w-full`} value={s.images} placeholder="folder/images/train or s3://bucket/path" onChange={e => source(s.id, { images: e.target.value })} /></label>
          <label className="text-xs">Labels location<input className={`${input} block w-full`} value={s.labels} placeholder="folder/labels/train or s3://bucket/path" onChange={e => source(s.id, { labels: e.target.value })} /></label>
        </div>
        <details className="mt-2 text-xs"><summary>Source class numbering (only if its original YAML is unavailable)</summary>
          <input aria-label="Explicit source class mapping" className={`${input} mt-1 w-full`} placeholder="0=helmet, 2=person" value={sourceMappings[s.id] ?? Object.entries(s.classes).map(([k,v]) => `${k}=${v}`).join(", ")}
            onChange={e => { setSourceMappings(m => ({ ...m, [s.id]: e.target.value })); setDirty(true); }} />
          <p className="mt-1 text-ink-muted">This describes the existing label files. Output numbering is edited in Class mapping below.</p>
        </details>
      </fieldset>)}</div>
      <button className={`${button} mt-3`} onClick={() => edit({ sources: [...draft.sources, { id: ("source-" + Date.now().toString(36) + Math.random().toString(36).slice(2)), name: "New source", enabled: true, bucket: "", images: "", labels: "", kind: "collected", split: "train", train_percent: 80, classes: {} }] })}>Add source</button>
    </Card>
    <Card title="Split and golden-set policy" step="2" keepMounted>
      <p className="mb-3 text-xs text-ink-secondary">Existing train and validation sources keep their roles. A source split is frozen on first indexing; later arrivals go to training. Golden examples never decay. Identical image content cannot appear in both splits.</p>
      <div className="space-y-2">{draft.sources.filter(s => s.enabled).map(s => <div key={s.id} className="flex flex-wrap items-center gap-3 text-xs">
        <span className="min-w-48 flex-1">{s.name}</span><select aria-label={`Split for ${s.name}`} className={input} value={s.split} onChange={e => source(s.id, { split: e.target.value as Source["split"] })}>
          <option value="train">Training</option><option value="val">Validation</option><option value="split">Freeze a train/validation split</option></select>
        {s.split === "split" && <label>Train % <input className={`${input} w-16`} type="number" min={1} max={99} value={s.train_percent} onChange={e => source(s.id, { train_percent: Number(e.target.value) })} /></label>}
      </div>)}</div>
      <p className="mt-2 text-xs text-ink-muted">To change an already frozen split, create a new selection. Existing versions remain reproducible.</p>
      <button className={`${button} mt-2`} disabled={busy || !!active} onClick={async () => { setError(""); try { await persist(); await apiPost(base + "/new-selection", {}); await refresh(); setNote("New selection created. Index it, then Save as a new definition to preserve the previous one."); } catch (e) { setError(e instanceof ApiError ? e.detail : String(e)); } }}>Create new selection for a different split</button>
    </Card>
    <Card title="Class mapping" step="3" keepMounted>
      <p className="text-xs text-ink-secondary">Source YAML files describe the original IDs. These output IDs are used in the prepared YOLO labels and data.yaml; source labels stay unchanged.</p>
      <button className={`${button} my-2`} onClick={discoverNames}>Add classes discovered by scan</button>
      <div className="flex flex-wrap gap-4">{Object.entries(draft.class_ids).sort((a,b) => Number(a[0])-Number(b[0])).map(([id,name]) => <label key={name} className="flex items-center gap-2 text-xs">{name}<input aria-label={`Output ID for ${name}`} className={`${input} w-20`} type="number" min={0} value={id} onChange={e => classId(id,Number(e.target.value))} /></label>)}</div>
      <div className="mt-3 flex flex-wrap gap-3 text-xs"><label><input type="checkbox" checked={draft.filter_to_model} onChange={e => edit({ filter_to_model: e.target.checked })} /> Keep only the chosen model’s classes</label>
        <label>Model name <input className={input} value={draft.detector_slug} onChange={e => edit({ detector_slug: e.target.value })} /></label></div>
      <p className="mt-1 text-xs text-ink-muted">{Object.values(draft.model_classes).length > 0 && `Model classes: ${Object.values(draft.model_classes).join(", ")}. `}Changing an ID swaps it with the class already using that number.</p>
    </Card>
    <Card title="Retention" step="4" keepMounted>
      <label className="text-sm"><input type="checkbox" checked={policy.enabled} onChange={e => retention({ enabled: e.target.checked })} /> Apply retention to collected training data</label>
      <p className="mt-1 text-xs text-ink-muted">Age is measured in nonempty collections. Validation and fixed/golden sources do not decay. Files are never deleted by this policy.</p>
      <div className="mt-3 flex flex-wrap gap-3 text-xs">
        <label>Reduction per older collection (%) <input className={`${input} w-20`} type="number" min={0} max={100} step="0.1" value={policy.weekly_rate*100} onChange={e => retention({ weekly_rate:Number(e.target.value)/100 })} /></label>
        <label>Newest collections kept in full <input className={`${input} w-16`} type="number" min={1} value={policy.full_weeks} onChange={e => retention({ full_weeks:Number(e.target.value) })} /></label>
        <label>Curve <select className={input} value={policy.curve} onChange={e => retention({ curve:e.target.value as Track2RetentionConfig["curve"] })}><option value="exponential">Exponential</option><option value="linear">Linear</option></select></label>
        <label><input type="checkbox" checked={policy.use_floor} onChange={e => retention({ use_floor:e.target.checked })} /> Use minimums</label>
        <label>Minimum retained (%) <input className={`${input} w-20`} type="number" min={0} max={100} value={policy.floor*100} onChange={e => retention({ floor:Number(e.target.value)/100 })} /></label>
        <label>Minimum images per collection <input className={`${input} w-20`} type="number" min={0} value={policy.min_images} onChange={e => retention({ min_images:Number(e.target.value) })} /></label>
        <label>Warn below selected images per class/source <input className={`${input} w-20`} type="number" min={1} value={policy.min_class_images ?? 5} onChange={e => retention({ min_class_images:Number(e.target.value) })} /></label>
      </div>
      <div className="mt-3 flex flex-wrap gap-4 text-xs">{data.collections.map(w => <label key={w.week}>
        {w.week !== "fixed" && <input type="checkbox" checked={!excluded.includes(w.week)} onChange={e => { setExcluded(e.target.checked ? excluded.filter(v => v !== w.week) : [...excluded,w.week]); setDirty(true); }} />} {w.week}: {w.selected} / {w.available} selected
      </label>)}</div>
      <p className="mt-2 text-xs text-ink-muted">Unticking a whole collection excludes both splits. Counts below reflect the last applied settings; Scan or Save applies edits.</p>
    </Card>
    <Card title="Selection preview and readiness" step="5">
      <div className="flex flex-wrap gap-2"><button className={button} disabled={busy || !!active} onClick={() => void run("scan")}>Scan sources</button><button className={button} disabled={busy || !!active} onClick={() => void run("refresh")}>Index / refresh selection</button><button className={button} onClick={() => void images()}>Show images and labels</button></div>
      <p className="mt-3 text-sm">Available {data.counts.available.toLocaleString()} · Selected {data.counts.selected.toLocaleString()} · Train {data.counts.train.toLocaleString()} · Validation {data.counts.val.toLocaleString()}</p>
      {data.sources_stale && <p className="mt-2 text-xs text-critical">Sources have not been indexed, or changed since the last index. Refresh before freezing a version.</p>}
      {latest && <div className="mt-3 rounded border border-hairline p-3 text-xs" role="status"><strong>{latest.kind}: {latest.status}</strong>
        {active && <progress className="mt-2 block w-full" max={active.total || undefined} value={active.total ? active.done : undefined} />}
        {latest.total > 0 && <p>{latest.done.toLocaleString()} / {latest.total.toLocaleString()}</p>}
        {latest.error && <p className="mt-1 text-critical">{latest.error}</p>}
        {latest.result.images != null && <p>{latest.result.images.toLocaleString()} images found · {latest.result.eligible?.toLocaleString() ?? latest.result.selected ?? 0} eligible</p>}
        {latest.result.blockers?.map(b => <p key={b} className="mt-1 text-critical">{b}</p>)}
        {(latest.result.excluded?.length ?? 0) > 0 && <details><summary>{latest.result.excluded!.length} images excluded — reasons</summary>{latest.result.excluded!.slice(0,100).map(r => <p key={r.source_uri} className="mt-1 break-all">{r.source_uri}: {r.reason}</p>)}{latest.result.excluded!.length > 100 && <p>Showing the first 100; the complete report is retained on the server.</p>}</details>}
      </div>}
      {Object.entries(data.counts.excluded).map(([reason,count]) => <p key={reason} className="mt-1 text-xs">{count} excluded: {reason}</p>)}
      <div className="mt-3 overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Source</th><th>Class</th><th>Available</th><th>Selected</th><th>Coverage</th></tr></thead><tbody>{data.coverage.map(r => <tr key={r.source_id+r.class_name}><td className="py-1">{sourceName(r.source_id)}</td><td>{r.class_name}</td><td>{r.available}</td><td>{r.selected}</td><td>{r.warning ? "Below threshold" : "OK"}</td></tr>)}</tbody></table></div>
      {gallery && <div className="mt-3"><div className="grid grid-cols-2 gap-3 lg:grid-cols-4">{gallery.items.map(m => <div key={m.source_uri} className="rounded border border-hairline p-1">
        <div className="relative"><img src={m.url} alt={m.source_uri} className="block w-full" /><svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none">{m.boxes.map((b,i) => <rect key={i} x={(b.x-b.w/2)*100} y={(b.y-b.h/2)*100} width={b.w*100} height={b.h*100} fill="none" stroke="#2685ef" strokeWidth="0.4" />)}</svg></div>
        <p className="mt-1 text-xs">{m.split} · {m.included ? "Included" : m.reason}</p><p className="text-xs">{[...new Set(m.boxes.map(b => `${b.id}: ${b.name}`))].join(", ") || "Explicit empty label (background)"}</p><p className="truncate text-xs" title={m.source_uri}>{m.source_uri}</p>
      </div>)}</div><div className="mt-2 flex gap-3 text-xs"><button className={button} disabled={gallery.page<=1} onClick={() => void images(gallery.page-1)}>Previous</button><span>Page {gallery.page} of {gallery.pages} · {gallery.total} images</span><button className={button} disabled={gallery.page>=gallery.pages} onClick={() => void images(gallery.page+1)}>Next</button></div></div>}
    </Card>
    <Card title="Dataset versions" step="6">
      <p className="text-xs text-ink-secondary">Freeze the exact membership, label revisions and settings. Training and Data Catalog use this same version. Repeated freezes reuse an unchanged selection.</p>
      <button className={`${button} mt-2`} disabled={busy || !!active || data.sources_stale || !data.counts.train || !data.counts.val} onClick={() => void run("freeze")}>Freeze dataset version</button>
      <div className="mt-3 overflow-auto"><table className="w-full text-left text-xs"><thead><tr><th>Version</th><th>Created (SGT)</th><th>Train</th><th>Validation</th><th>State</th></tr></thead><tbody>{data.versions.map(v => <tr key={v.version_id}><td className="py-1" title={v.version_id}>{v.version_number} · {v.version_id}</td><td>{when(v.created_at)}</td><td>{v.train_images}</td><td>{v.val_images}</td><td>{v.state}</td></tr>)}</tbody></table></div>
      <details className="mt-3 text-xs"><summary>Recent preparation jobs</summary>{data.jobs.map(j => <p key={j.id}>{when(j.created_at)} SGT · {j.kind} · {j.status}{j.error ? ` · ${j.error}` : ""}</p>)}</details>
    </Card>
    <Track2CentralStorePanel />
  </div>;
}
