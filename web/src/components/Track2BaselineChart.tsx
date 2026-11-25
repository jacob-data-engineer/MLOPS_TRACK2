import type {ComparisonMetricRow} from '../lib/api';
const scores = ['mAP50-95','mAP50','precision','recall','f1'] as const;
const colors = ['#2563eb','#d97706','#059669','#7c3aed','#db2777'];
export function Track2BaselineChart({rows}: {rows:ComparisonMetricRow[]}) {
  if (!rows.length) return null;
  const ordered=[...rows].sort((a,b)=>Number(b.role==='baseline')-Number(a.role==='baseline'));
  const label=(row:ComparisonMetricRow)=>`${row.role==='baseline'?'Baseline':row.role==='selected'?'Selected run':'Model'}: ${row.label}`;
  return <div className="my-4 space-y-4" aria-label="Baseline comparison bar charts">
    <div className="flex flex-wrap gap-4 text-xs">{ordered.map((r,i)=><span key={r.role+r.id} className="flex items-center gap-2 break-all"><span className="h-3 w-3 shrink-0 rounded-sm" style={{backgroundColor:colors[i%colors.length]}}/>{label(r)}</span>)}</div>
    <p className="text-xs text-ink-muted">Scores: higher is better. All score bars use the same 0–100% scale.</p>
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">{scores.map(metric=><section key={metric} className="rounded border border-hairline p-3" aria-label={`${metric} comparison`}>
      <h3 className="mb-3 text-sm font-semibold">{metric}</h3>
      {ordered.map((r,i)=><div key={r.role+r.id} className="mb-2 flex items-center gap-2" title={label(r)}>
        <div className="h-5 flex-1 rounded bg-black/5" role="img" aria-label={`${label(r)}, ${metric}: ${(r[metric]*100).toFixed(2)}%`}><div className="h-full rounded" style={{width:`${Math.max(0,Math.min(100,r[metric]*100))}%`,backgroundColor:colors[i%colors.length]}}/></div>
        <span className="w-16 text-right text-xs tabular-nums">{(r[metric]*100).toFixed(2)}%</span>
      </div>)}
      <div className="mr-[4.5rem] flex justify-between text-[10px] text-ink-muted"><span>0%</span><span>50%</span><span>100%</span></div>
    </section>)}</div>
    <p className="text-xs text-ink-muted">Error counts: lower is better. Bars share a scale within each chart.</p>
    <div className="grid gap-4 md:grid-cols-2">{(['fp','fn'] as const).map(key=>{
      const maximum=Math.max(1,...ordered.map(r=>r[key]));const title=key==='fp'?'False detections':'Missed objects';
      return <section key={key} className="rounded border border-hairline p-3" aria-label={`${title} comparison`}><h3 className="mb-3 text-sm font-semibold">{title}</h3>
        {ordered.map((r,i)=><div key={r.role+r.id} className="mb-2 flex items-center gap-2" title={label(r)}><div className="h-5 flex-1 rounded bg-black/5" role="img" aria-label={`${label(r)}, ${title}: ${r[key]}`}><div className="h-full rounded" style={{width:`${Math.max(0,r[key])/maximum*100}%`,backgroundColor:colors[i%colors.length]}}/></div><span className="w-16 text-right text-xs tabular-nums">{r[key].toLocaleString()}</span></div>)}
      </section>;
    })}</div>
  </div>;
}
