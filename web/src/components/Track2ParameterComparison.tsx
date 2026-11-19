import { useState } from "react";
import type { Track2MetricsRow, Track2TrainingAudit } from "../lib/api";
import { GROUPS } from "./Track2TrainingAdvanced";
import { modelLabel } from "./Track2ModelPicker";

// Sorted object keys avoid false differences caused by JSON property order.
export function parameterText(value: unknown): string {
  if (value === undefined || value === null) return "Not recorded";
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return `[${value.map(parameterText).join(", ")}]`;
  if (typeof value === "object") return Object.entries(value).sort(([a], [b]) => a.localeCompare(b)).map(([k,v]) => `${k}: ${parameterText(v)}`).join("; ");
  return String(value);
}

export function Track2ParameterComparison({ rows, audits }: {
  rows: Track2MetricsRow[]; audits: Record<string, Track2TrainingAudit>;
}) {
  const [differencesOnly, setDifferencesOnly] = useState(false);
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
  const differences = entries.filter(e=>e.different).length;
  if (!rows.length) return null;
  return <section className="mb-5" aria-label="Training parameter comparison">
    <div className="mb-2 flex flex-wrap items-center justify-between gap-2"><h3 className="text-sm font-semibold">Training parameters side by side</h3><label className="text-xs"><input type="checkbox" checked={differencesOnly} onChange={e=>setDifferencesOnly(e.target.checked)}/> Show differences only ({differences})</label></div>
    <p className="mb-2 text-xs text-ink-muted">Recorded run parameters, including saved defaults. Highlighted rows have different recorded values. Missing values are not treated as differences; current defaults are never substituted. These are saved settings, not a reconstruction of unrecorded trainer defaults.</p>
    <div className="overflow-x-auto rounded-sm border border-hairline"><table className="w-full min-w-[600px] text-left text-xs">
      <thead className="border-b border-hairline bg-page"><tr><th className="px-3 py-2">Parameter</th>{rows.map(r=><th key={r.model_package_arn} className="px-3 py-2">{r.registry} · {modelLabel(r)}<p className="font-normal">Run {audits[r.model_package_arn]?.run_number ?? "not recorded"}</p></th>)}</tr></thead>
      <tbody>{entries.filter(e=>!differencesOnly || e.different).map(e=><tr key={`${e.kind}/${e.key}`} className={`border-b border-hairline last:border-0 ${e.different ? "bg-amber-100/70" : ""}`}>
        <th scope="row" className="px-3 py-2 align-top font-medium"><span className="block text-ink-muted">{e.group}</span>{e.label} <span className="font-normal text-ink-muted">({e.key})</span>{e.different && <span className="ml-2 font-semibold">Different</span>}{e.recorded<rows.length && <span className="block font-normal text-ink-muted">Incomplete record</span>}</th>
        {e.values.map((v,i)=><td key={rows[i].model_package_arn} className="max-w-xs break-words px-3 py-2 align-top tabular">{parameterText(v)}</td>)}
      </tr>)}</tbody>
    </table></div>
    {differencesOnly && !differences && <p className="mt-2 text-xs">No differences among recorded values.</p>}
    {records.some(a=>!a || !Object.keys(a.hyperparams ?? {}).length) && <p className="mt-2 text-xs text-ink-muted">Some runs have no advanced parameters in their saved record. Their trainer defaults cannot be inferred from an empty record.</p>}
  </section>;
}
