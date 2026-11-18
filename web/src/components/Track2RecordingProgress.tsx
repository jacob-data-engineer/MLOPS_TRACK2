import type { Track2QueueSnapshot } from '../lib/api';
import { ProgressBar } from './ProgressBar';

export function Track2RecordingProgress({progress}: {progress: Track2QueueSnapshot['submission_progress']}) {
  if (!progress) return null;
  return <div role="status" aria-live="polite" className="mb-3 rounded-md border border-black/20 bg-page p-3 text-sm">
    <p className="font-semibold">{progress.stage}{progress.total > 0 ? `: ${progress.completed.toLocaleString()} / ${progress.total.toLocaleString()}` : '…'}</p>
    {progress.total > 0 ? <div role="progressbar" aria-label="Recording training files" aria-valuemin={0}
      aria-valuemax={progress.total} aria-valuenow={progress.completed} className="mt-2">
      <ProgressBar processed={progress.completed} total={progress.total}/>
    </div> : <p className="mt-1 animate-pulse text-xs motion-reduce:animate-none">Preparing the training submission…</p>}
    <p className="mt-2 text-xs text-ink-muted">Queue entry {progress.entry_id}. Training starts after its file records are saved. You can close this page; the worker continues.</p>
  </div>;
}
