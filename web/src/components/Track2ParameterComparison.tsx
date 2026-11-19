import type { Track2MetricsRow, Track2TrainingAudit } from "../lib/api";
import { GROUPS } from "./Track2TrainingAdvanced";

// Sorted object keys avoid false differences caused by JSON property order.
export function parameterText(value: unknown): string {
  if (value === undefined || value === null) return "Not recorded";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return `[${value.map(parameterText).join(", ")}]`;
  if (typeof value === "object") return Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([k,v]) => `${k}: ${parameterText(v)}`).join("; ");
  return String(value);
}

export function comparedParameters({ rows, audits }: {
  rows: Track2MetricsRow[]; audits: Record<string, Track2TrainingAudit>;
}) {
  const records = rows.map(r => audits[r.model_package_arn]);
  const definitions = [
    ...["epochs", "img_size", "batch", "seed", "patience"].map(key => ({key, group:"Training", label:({epochs:"Epochs",img_size:"Image size",batch:"Batch",seed:"Seed",patience:"Patience"} as Record<string,string>)[key], kind:"basic"})),
    ...GROUPS.flatMap(g => g.params.map(p => ({key:p.key, group:g.title, label:p.label, kind:"hyper"}))),
  ];
  const known = new Set(definitions.filter(d=>d.kind==="hyper").map(d=>d.key));
  for (const key of [...new Set(records.flatMap(a=>Object.keys(a?.hyperparams ?? {})))].sort()) {
    if (!known.has(key)) definitions.push({key,group:"Other recorded parameters",label:key,kind:"hyper"});
  }
  for (const key of [...new Set(records.flatMap(a=>Object.keys(a?.training_options ?? {})))].sort()) definitions.push({key,group:"Class options",label:key,kind:"option"});
  const entries = definitions.map(d => {
    const values = records.map(a => d.kind === "hyper" ? a?.hyperparams?.[d.key] : d.kind === "option" ? a?.training_options?.[d.key] : a?.parameters?.[d.key as keyof Track2TrainingAudit["parameters"]]);
    const recorded = values.filter(v=>v!==undefined && v!==null);
    return {...d,values, recorded:recorded.length, different:new Set(recorded.map(parameterText)).size>1};
  }).filter(d => d.kind==="basic" || d.recorded>0);
  return entries;
}

export function TrainingSettingsCell({ entries, index }: {
  entries: ReturnType<typeof comparedParameters>; index: number;
}) {
  return <dl className="space-y-1" aria-label="Recorded training settings">
    {entries.map((e, position) => {
      const value = e.values[index];
      const different = e.different && value !== undefined && value !== null;
      return <div key={`${e.kind}/${e.key}`}>
        {(position === 0 || entries[position - 1].group !== e.group) && <p className="mt-2 font-semibold text-ink-secondary">{e.group}</p>}
        <div className={`rounded px-1 py-0.5 ${different ? "bg-amber-100/80 font-medium" : ""}`} data-parameter={e.key}>
          <dt className="inline" title={e.key}>{e.label}: </dt>
          <dd className="inline break-words tabular">{parameterText(value)}</dd>
          {different && <span className="ml-1 text-xs font-semibold">(Different)</span>}
        </div>
      </div>;
    })}
  </dl>;
}
