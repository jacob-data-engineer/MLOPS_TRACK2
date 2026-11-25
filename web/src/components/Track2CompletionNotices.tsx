import {useEffect, useState} from 'react';
import {getTrack2CatalogExports, listRunComparisons} from '../lib/api';
import {detectCompletions, s3ConsoleUrl, type CompletionSnapshot} from '../lib/completionEvents';
import {Track2RunComparisons} from './Track2RunComparisons';

type Notice = {id: string; kind: string; label: string; state: string; run?: string; uri?: string};
const empty = (): CompletionSnapshot => ({initialized: false, states: {}});
const storageKey = 'track2-completion-observer-v1';

/** Observe while signed in. Open results in-app, without popup permissions or
 * navigating away from unsaved work. Initial historical results stay quiet. */
export function Track2CompletionNotices() {
  const [notices, setNotices] = useState<Notice[]>([]);
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    let snapshots: Record<string, CompletionSnapshot> = {training: empty(), comparison: empty(), export: empty()};
    try {const saved = JSON.parse(sessionStorage.getItem(storageKey) || 'null');
      if (saved && ['training', 'comparison', 'export'].every(k => typeof saved[k]?.initialized === 'boolean' && saved[k]?.states)) snapshots = saved;
    } catch { /* Storage unavailable: observe in memory. */ }
    let timer: number;
    async function poll() {
      const results = await Promise.allSettled([listRunComparisons(), getTrack2CatalogExports(undefined, 200)]);
      if (!active) return;
      const next: Notice[] = [];
      const runs = results[0];
      if (runs.status === 'fulfilled') {
        for (const kind of ['training', 'comparison'] as const) {
          const detected = detectCompletions(snapshots[kind], runs.value.runs.map(r => ({id: r.id, label: r.label,
            attempt: kind === 'comparison' ? r.comparison?.id : undefined,
            state: kind === 'training' ? r.training_status : r.status})),
            kind === 'training' ? ['Succeeded', 'Failed', 'Stopped'] : ['Completed', 'Failed', 'Unavailable']);
          snapshots[kind] = detected.snapshot;
          next.push(...detected.events.map(e => ({...e, id: `${kind}:${e.id}:${e.state}`, kind, run: e.id})));
        }
      }
      const exports = results[1];
      if (exports.status === 'fulfilled') {
        const detected = detectCompletions(snapshots.export, exports.value.exports.map(e => ({id:e.id,label:e.label,state:e.state})), ['ready', 'partial', 'failed', 'cancelled']);
        snapshots.export = detected.snapshot;
        next.push(...detected.events.map(e => ({...e, id:`export:${e.id}:${e.state}`, kind:'export',
          uri: exports.value.exports.find(job => job.id === e.id)?.destination_uri || undefined})));
      }
      setError(results.some(r => r.status === 'rejected') ? 'Completion updates temporarily unavailable. Retrying automatically.' : '');
      if (next.length) setNotices(old => [...old.filter(n => !next.some(item => item.id === n.id)), ...next].slice(-20));
      try {sessionStorage.setItem(storageKey, JSON.stringify(snapshots));} catch { /* Keep observing. */ }
      timer = window.setTimeout(poll, 10000);
    }
    void poll();
    return () => {active = false; window.clearTimeout(timer);};
  }, []);
  return <aside aria-label="Completion notifications" className="fixed right-4 top-16 z-40 max-h-[75vh] w-[calc(100%-2rem)] max-w-4xl overflow-y-auto shadow-lg">
    {error && <p role="status" className="text-xs text-ink-muted">{error}</p>}
    {notices.map(n => <section key={n.id} className="my-3 rounded border border-hairline bg-page p-4">
      <div className="flex justify-between gap-3"><p role="status" className="font-semibold">{n.kind === 'training' ? 'Training' : n.kind === 'comparison' ? 'Comparison' : 'Export'} {n.state}: {n.label}</p>
        <button type="button" className="underline" onClick={() => setNotices(old => old.filter(item => item.id !== n.id))}>Dismiss</button></div>
      {n.kind === 'training' && <p className="text-sm">{n.state === 'Succeeded' ? 'Training finished. Selected baseline comparisons continue separately; their results will open here when ready.' : 'Open the training history to inspect this run.'}</p>}
      {n.kind === 'comparison' && n.state === 'Completed' && <Track2RunComparisons initialRun={n.run} />}
      {n.kind === 'export' && <><p className="break-all text-sm">{n.state === 'ready' ? 'Export saved in S3.' : 'Export did not finish as a complete package. Review its status before using it.'} {n.uri}</p>
        {n.uri && s3ConsoleUrl(n.uri) && <a className="underline" target="_blank" rel="noreferrer" href={s3ConsoleUrl(n.uri)}>Open export in S3</a>}</>}
    </section>)}
  </aside>;
}
