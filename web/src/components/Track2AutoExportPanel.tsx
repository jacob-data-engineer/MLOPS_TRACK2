import { useEffect, useRef, useState } from "react";
import { ApiError, getModelExportSchedules, saveTrack2ModelExportSchedule, previewModelExport, runModelExport,
  getTrack2ExportGroups, listTrack2Folders, type Track2RunForModel, type ModelExportPlan, type ModelExportSchedule, type Track2ExportPick,
  type Track2ExportSettings, type Track2MetricsRow } from "../lib/api";
import { formatSgt } from "../lib/format";
import { modelLabel } from "./Track2ModelPicker";
import { isSeedVersion, statusLabel } from "../lib/slug";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

export function Track2AutoExportPanel({rows, selectedRows, settings, onChange, runs = {}, onOpenRun}: {
  rows: Track2MetricsRow[]; selectedRows: Track2MetricsRow[]; settings: Track2ExportSettings;
  onChange: (value: Track2ExportSettings) => void;
  runs?: Record<string, Track2RunForModel>; onOpenRun?: (id: string)=>void;
}) {
  const scope = settings.auto_scope ?? "comparison";
  const [names, setNames] = useState<string[]>([]);
  const [schedules, setSchedules] = useState<ModelExportSchedule[]>([]);
  const [pick, setPick] = useState<Track2ExportPick | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [folders, setFolders] = useState<string[]>([]);
  const set = (patch: Partial<Track2ExportSettings>) => onChange({...settings, ...patch});
  const plan: ModelExportPlan = {scope, model_package_arns: scope === "comparison" ? selectedRows.map(r => r.model_package_arn) : [],
    model_name: scope === "model_name" ? settings.auto_model_name ?? "" : "", criterion: settings.criterion,
    destination: settings.destination, bucket: settings.bucket, prefix: settings.prefix,
    time_sgt: settings.auto_time_sgt ?? "09:00", every_days: settings.auto_every_days ?? 1, start_date: settings.auto_start_date ?? ""};
  const key = JSON.stringify(plan);
  const currentKey = useRef(key); currentKey.current = key;
  useEffect(()=>{let disposed=false;setFolders([]);if(settings.bucket)listTrack2Folders(settings.bucket,settings.prefix)
    .then(r=>{if(!disposed)setFolders(r.folders ?? []);}).catch(()=>{if(!disposed)setFolders([]);});
    return()=>{disposed=true;};},[settings.bucket,settings.prefix]);
  useEffect(() => {setPick(null); setNote("");}, [key]);
  useEffect(() => {
    let disposed=false;
    getTrack2ExportGroups().then(r => {if(!disposed)setNames([...new Set(r.groups.map(g=>g.registry))].sort());})
      .catch(e=>{if(!disposed)setError(e instanceof ApiError ? e.detail : "Could not load Model Names.");});
    async function refresh(){try{const r=await getModelExportSchedules();if(!disposed)setSchedules(r.schedules);}
      catch(e){if(!disposed)setError(e instanceof ApiError ? e.detail : "Could not load schedules.");}}
    void refresh(); const timer=window.setInterval(refresh,30000);
    return()=>{disposed=true;window.clearInterval(timer);};
  },[]);
  async function action(label: string, task:()=>Promise<void>){setBusy(label);setError("");setNote("");try{await task();}catch(e){setError(e instanceof ApiError ? e.detail : "The action failed. Please retry.");}finally{setBusy("");}}
  const choices = [...new Set([...names,...rows.map(r=>r.registry)])].sort();
  const missingComparison = (settings.comparison_versions ?? []).some(id=>!selectedRows.some(r=>`${r.group_name}/${r.version}`===id));
  const validScope = scope === "comparison" ? selectedRows.length > 0 && !missingComparison : !!plan.model_name;
  const button="rounded border border-hairline px-3 py-1.5 text-xs disabled:opacity-40";
  return <div className="space-y-4">
    <fieldset disabled={!!busy} className="space-y-3">
      <legend className="mb-2 text-sm font-semibold">Auto selection type</legend>
      <label className="mr-5 inline-flex gap-2 text-sm"><input type="radio" name="auto-selection-scope" checked={scope==="comparison"} onChange={()=>set({auto_scope:"comparison"})}/>Best of compared models</label>
      <label className="inline-flex gap-2 text-sm"><input type="radio" name="auto-selection-scope" checked={scope==="model_name"} onChange={()=>set({auto_scope:"model_name"})}/>Best for a Model Name</label>
      {scope==="comparison" ? <div className="text-xs"><p>Only these exact versions from Comparison compete. A schedule keeps this list until you explicitly update it.</p>
        {selectedRows.length ? <ul className="mt-2 list-inside list-disc">{selectedRows.map(r=><li key={r.model_package_arn}>{r.registry} &middot; {modelLabel(r)} &middot; {isSeedVersion(r.description)?"Seed":statusLabel(r.status)}</li>)}</ul> : <p className="mt-2">Select models in Comparison first.</p>}</div>
        : <div><label className="text-xs">Model Name <select aria-label="Auto Model Name" className="ml-2 rounded border p-1" value={plan.model_name} onChange={e=>set({auto_model_name:e.target.value})}><option value="">Choose Model Name</option>{choices.map(n=><option key={n}>{n}</option>)}</select></label>
          <p className="mt-2 text-xs">Ranks all registered versions, model types and sizes belonging to this Model Name, including new versions at the scheduled time.</p></div>}
      <label className="block text-xs">Best model decided by <select aria-label="Auto ranking criterion" className="ml-2 rounded border p-1" value={plan.criterion} onChange={e=>set({criterion:e.target.value as Track2ExportSettings["criterion"]})}>{["mAP50","mAP50-95","precision","recall","f1","accuracy","newest"].map(c=><option key={c}>{c}</option>)}</select></label>
      <p className="text-xs text-ink-muted">Highest recorded score wins; missing scores are excluded. Ties prefer the higher version number. Check evaluation datasets and classes before comparing scores.</p>
      <div className="flex flex-wrap gap-3 text-xs"><label>Destination <select aria-label="Auto export destination" value={plan.destination} onChange={e=>set({destination:e.target.value as "local"|"s3"})}><option value="local">Download .pt now</option><option value="s3">S3</option></select></label>
        <label>S3 bucket <input className="rounded border p-1" aria-label="Auto S3 bucket" value={plan.bucket} onChange={e=>set({bucket:e.target.value})}/></label>
        <label>Folder prefix <input className="rounded border p-1" aria-label="Auto S3 prefix" value={plan.prefix} onChange={e=>set({prefix:e.target.value})}/></label></div>
      <p className="text-xs">Auto exports one winning .pt. Earlier exports are kept. Use Manual for packages with dataset files and metadata.</p>
      {plan.bucket&&<details><summary className="cursor-pointer text-xs">Browse S3 folders</summary><Track2S3FolderBrowser folders={folders} prefix={plan.prefix} onNavigate={(next) => set({ prefix: next })} allowCreate /></details>}
      {missingComparison&&scope==="comparison"&&<p className="text-xs text-critical">Some saved comparison models are unavailable. Refresh or update the comparison before exporting or scheduling.</p>}
      <button className={button} disabled={!validScope} onClick={()=>void action("Previewing...",async()=>{const r=await previewModelExport(plan);if(currentKey.current!==key)return;setPick(r.best);setNote(r.best ? `Ranked ${r.considered} model(s).` : r.reason);})}>Preview best model</button>
      {pick && <div className="rounded border border-hairline p-3 text-xs"><p className="font-medium">Winner: {pick.group_name} &middot; v{pick.version} &middot; {pick.criterion}: {pick.value ?? "not measured"}</p><p className="break-all">{pick.model_package_arn}</p>
        <p>{pick.is_seed?"Seed":statusLabel(pick.status)} &middot; Validation images: {pick.validation_images ?? "not recorded"}</p>
        {pick.is_seed&&<p className="text-critical">This is a seed checkpoint; it has not yet been trained for this Model Name.</p>}
        {pick.thin_validation&&<p className="text-critical">Few validation images: treat this score cautiously.</p>}
        {runs[pick.model_package_arn]&&<button className="underline" onClick={()=>onOpenRun?.(runs[pick.model_package_arn].run_id)}>Run {runs[pick.model_package_arn].run_number} · Open run in catalog</button>}
        <button className={`${button} mt-2`} disabled={plan.destination==="s3"&&!plan.bucket} onClick={()=>void action("Exporting...",async()=>{const r=await runModelExport({...plan,expected_arn:pick.model_package_arn});if(r.url){const a=document.createElement("a");a.href=r.url;a.download=r.filename;a.click();}setNote(r.written_to || `Downloading ${r.filename}`);})}>Export this winner now</button></div>}
      <div className="space-y-3 rounded border border-hairline p-3"><p className="text-sm font-semibold">Schedule this Auto type</p>
        <div className="flex flex-wrap gap-3 text-xs"><label>Start date (SGT) <input type="date" aria-label="Schedule start date" value={plan.start_date} onChange={e=>set({auto_start_date:e.target.value})}/></label>
          <label>Time (Singapore) <input type="time" aria-label="Schedule Singapore time" value={plan.time_sgt} onChange={e=>set({auto_time_sgt:e.target.value})}/></label>
          <label>Every <input type="number" min="1" max="365" className="w-16 rounded border p-1" aria-label="Schedule interval days" value={plan.every_days} onChange={e=>set({auto_every_days:Number(e.target.value)})}/> day(s)</label></div>
        <p className="text-xs">Schedules export to S3 even when the page is closed, in a separate subfolder for each scheduled run. Each Auto type has its own schedule. A blank start date means today; past times advance to the next interval. Missed intervals produce one export when the worker resumes.</p>
        <button className={button} disabled={!validScope||!plan.bucket||!plan.time_sgt||!plan.every_days} onClick={()=>void action("Saving schedule...",async()=>{const r=await saveTrack2ModelExportSchedule(scope,{...plan,destination:"s3",enabled:true});setSchedules(old=>[...old.filter(p=>p.scope!==scope),r.schedule]);setNote("Schedule saved and enabled. The first export runs at the next scheduled time.");})}>Save and enable schedule</button>
      </div>
    </fieldset>
    {busy&&<p role="status" className="text-xs">{busy}</p>}{error&&<p role="alert" className="text-xs text-critical">{error}</p>}{note&&<p role="status" className="text-xs">{note}</p>}
    <div className="space-y-2"><p className="text-sm font-semibold">Saved schedules</p>{schedules.map(s=><div key={s.scope} className="rounded border border-hairline p-3 text-xs">
      <p className="font-medium">{s.scope==="comparison"?"Best of compared models":"Best for a Model Name"}: {s.enabled?"Enabled":"Off"}</p>
      {s.criterion&&<p>{s.scope==="comparison"?`${s.model_package_arns?.length ?? 0} saved model(s)`:s.model_name} &middot; {s.criterion} &middot; every {s.every_days} day(s) at {s.time_sgt} SGT &middot; s3://{s.bucket}/{s.prefix}</p>}
      {s.scope==="comparison"&&!!s.model_package_arns?.length&&<details><summary className="cursor-pointer">Saved models</summary>{s.model_package_arns.map(a=><p key={a} className="break-all">{a}</p>)}</details>}
      {s.enabled&&<p>Next: {formatSgt(s.next_run_at)}</p>}{s.last_result&&<p>Last result: {s.last_result.state} &middot; {formatSgt(s.last_result.at)} &middot; {s.last_result.detail || s.last_result.written_to}</p>}
      {s.enabled&&<button className={`${button} mt-2`} disabled={!!busy} onClick={()=>void action("Stopping schedule...",async()=>{const r=await saveTrack2ModelExportSchedule(s.scope,{enabled:false});setSchedules(old=>old.map(p=>p.scope===s.scope?r.schedule:p));setNote("Schedule stopped. An export already in progress may finish.");})}>Stop {s.scope==="comparison"?"comparison":"Model Name"} schedule</button>}
    </div>)}</div>
  </div>;
}
