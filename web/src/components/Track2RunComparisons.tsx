import {useEffect, useState} from 'react';
import {ApiError, listRunComparisons, runBaselineComparison, type BaselineSelection,
  type RunComparisonSummary} from '../lib/api';
import {Track2BaselineComparison} from './Track2BaselineComparison';
import {formatDayTime} from '../lib/format';
import {Track2BaselineChart} from './Track2BaselineChart';

const metrics = ['mAP50-95', 'mAP50', 'precision', 'recall', 'f1'] as const;
export function Track2RunComparisons({initialRun = '', onRunChange}: {initialRun?: string; onRunChange?: (id: string) => void} = {}) {
  const [runs, setRuns] = useState<RunComparisonSummary[]>([]);
  const [selected, setSelected] = useState(initialRun);
  const [selection, setSelection] = useState<BaselineSelection>({enabled: false});
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  useEffect(() => {
    let active = true;
    async function load() {
      try {const result = await listRunComparisons(); if (active) {setRuns(result.runs); setError('');}}
      catch (e) {if (active) setError(e instanceof ApiError ? e.detail : 'Could not load run comparisons.');}
    }
    void load(); const timer = window.setInterval(load, 10000);
    return () => {active = false; window.clearInterval(timer);};
  }, []);
  const run = runs.find(r => r.id === selected);
  const result = run?.comparison?.result;
  const waiting = run?.status === 'Running' || run?.status === 'Pending';
  async function start(retry: boolean) {
    setBusy(true); setError('');
    try {
      await runBaselineComparison(selected, retry ? undefined : selection);
      setRuns((await listRunComparisons()).runs);
    } catch (e) {setError(e instanceof ApiError ? e.detail : 'Could not queue the comparison.');}
    finally {setBusy(false);}
  }
  return <section className="mb-5 rounded-md border border-hairline bg-page p-4" aria-label="Saved-run baseline comparisons">
    <h2 className="font-semibold">Saved-run baseline comparisons</h2>
    <p className="my-2 text-xs text-ink-muted">Every result here evaluates all selected models on one saved validation snapshot. Original training scores below remain unchanged.</p>
    <label className="block text-sm">Training run<select className="my-2 block w-full rounded border border-hairline bg-page p-2" value={selected}
      onChange={e => {setSelected(e.target.value); onRunChange?.(e.target.value); setSelection({enabled:false});}}>
      <option value="">Choose a saved training run</option>
      {runs.map(r => <option key={r.id} value={r.id}>{r.run_number != null ? `Run ${r.run_number}` : r.id} · {r.label} · {formatDayTime(r.submitted_at)} · {r.status}</option>)}
    </select></label>
    {run && <>
      <p role="status" className="my-2 text-sm font-medium">{run.status}</p>
      {run.comparison?.error && <p className="text-xs text-critical">{run.comparison.error}</p>}
      {run.can_compare && !waiting && <details>
        <summary className="cursor-pointer text-sm">Compare this completed run</summary>
        <Track2BaselineComparison value={selection} onChange={setSelection} disabled={busy} completed/>
        <button type="button" disabled={busy || !selection.enabled || !selection.baseline_arn} onClick={() => start(false)} className="rounded border border-hairline px-3 py-2 text-sm disabled:opacity-40">Run comparison</button>
        {['Failed','Unavailable'].includes(run.status) && <button type="button" disabled={busy} onClick={() => start(true)} className="ml-2 rounded border border-hairline px-3 py-2 text-sm">Retry saved comparison</button>}
      </details>}
      {result?.status === 'Completed' && <>
        <Track2BaselineChart rows={result.rows ?? []} />
        <p className="my-2 break-all text-xs">{result.validation_images} validation images · Dataset {result.dataset_version_id}<br/>Snapshot: {result.snapshot_id}</p>
        <p className="my-2 text-xs">Overall precision, recall and F1 are calculated from total detections across classes, at confidence {result.settings?.confidence}. AP uses confidence {result.settings?.ap_conf}; image size {result.settings?.imgsz}. Evaluator: Ultralytics {result.ultralytics_version}. Differences are percentage points from baseline.</p>
        <div className="overflow-x-auto"><table className="w-full text-left text-xs"><thead><tr>
          <th className="p-2">Model</th>{metrics.map(k => <th className="p-2" key={k}>{k}</th>)}<th className="p-2">False detections</th><th className="p-2">Missed objects</th>
        </tr></thead><tbody>{result.rows?.map(r => <tr key={r.role + r.id} className="border-t border-hairline">
          <td className="p-2">{r.role === 'baseline' ? 'Baseline: ' : ''}{r.label}</td>
          {metrics.map(k => <td className="p-2" key={k}>{(r[k] * 100).toFixed(2)}%<br/>{r.role !== 'baseline' && `${r.delta_pp[k] >= 0 ? '+' : ''}${r.delta_pp[k].toFixed(2)} pp`}</td>)}
          <td className="p-2">{r.fp}</td><td className="p-2">{r.fn}</td>
        </tr>)}</tbody></table></div>
        <details className="mt-3 text-xs"><summary className="cursor-pointer">Per-class results</summary>
          {result.rows?.map(r => <div key={r.role + r.id} className="my-3"><strong>{r.label}</strong><div className="overflow-x-auto"><table className="w-full text-left"><thead><tr><th>Class</th><th>mAP50–95</th><th>Precision</th><th>Recall</th><th>False detections</th><th>Missed objects</th></tr></thead><tbody>
            {r.per_class.map(p => <tr key={p.class}><td>{p.class}</td><td>{p['mAP50-95'] == null ? 'Not evaluated' : `${(p['mAP50-95'] * 100).toFixed(2)}%`}</td><td>{(p.precision*100).toFixed(2)}%</td><td>{(p.recall*100).toFixed(2)}%</td><td>{p.fp}</td><td>{p.fn}</td></tr>)}
          </tbody></table></div></div>)}
        </details>
        <p className="mt-3 text-xs text-ink-muted">These are validation results, not an independent test benchmark. A comparison model may have trained on some of these images; training overlap is not verified.</p>
      </>}
    </>}
    {error && <p role="alert" className="mt-2 text-xs text-critical">{error}</p>}
  </section>;
}
