import { useEffect, useState } from "react";
import { ApiError, getTrack2AutoExport, disarmTrack2AutoExport, type Track2AutoExport,
  type Track2MetricsRow, type Track2ExportSettings, type Track2RunForModel } from "../lib/api";
import { Card } from "./Card";
import { Track2ComparisonExport } from "./Track2ComparisonExport";
import { Track2AutoExportPanel } from "./Track2AutoExportPanel";
export function Track2MetricsExportCard(props: {
  rows: Track2MetricsRow[]; selectedRows?: Track2MetricsRow[];
  settings: Track2ExportSettings; onChange: (value: Track2ExportSettings) => void;
  runsByArn?: Record<string, Track2RunForModel>; onOpenRun?: (id: string) => void;
}) {
  const {rows, selectedRows = [], settings, onChange, runsByArn = {}} = props;
  const [legacy, setLegacy] = useState<Track2AutoExport | null>(null);
  const [legacyError, setLegacyError] = useState("");
  useEffect(() => {getTrack2AutoExport().then(r=>setLegacy(r.settings)).catch(()=>setLegacyError("Could not check the existing registration watcher."));}, []);
  const auto = settings.export_mode === "auto";
  return <Card title="Model export">
    <fieldset className="mb-4 flex gap-5 border-b border-hairline pb-3"><legend className="mb-2 text-xs font-medium">Export mode</legend>
      <label className="flex gap-2 text-sm"><input type="radio" name="model-export-mode" checked={auto} onChange={()=>onChange({...settings,export_mode:"auto"})}/>Auto</label>
      <label className="flex gap-2 text-sm"><input type="radio" name="model-export-mode" checked={!auto} onChange={()=>onChange({...settings,export_mode:"selected"})}/>Manual</label>
    </fieldset>
    {legacy?.enabled&&<p className="mb-3 rounded border border-hairline p-3 text-xs">The previous new-registration watcher is still active for {legacy.group_name}. It is separate from the schedules below. <button className="underline" onClick={()=>void disarmTrack2AutoExport().then(r=>setLegacy(r.settings)).catch(e=>setLegacyError(e instanceof ApiError?e.detail:"Could not stop the watcher."))}>Stop previous watcher</button></p>}
    {legacyError&&<p className="text-xs text-critical">{legacyError}</p>}
    {auto ? <Track2AutoExportPanel rows={rows} selectedRows={selectedRows} settings={settings} onChange={onChange} runs={runsByArn} onOpenRun={props.onOpenRun}/> : <>
      <p className="mb-3 text-xs">Manual: browse all registered models, scroll through the list, and tick any versions to export. Comparison does not restrict this list.</p>
      <Track2ComparisonExport rows={rows} runs={runsByArn} settings={settings} onChange={onChange} browseAll onOpenRun={props.onOpenRun} />
    </>}
  </Card>;
}
