import { useCallback, useEffect, useState } from "react";
import {
  ApiError,
  advanceTrack2PullQueue,
  clearTrack2PullQueue,
  getTrack2PullQueue,
  queueTrack2Pull,
  removeTrack2PullQueueEntry,
  type Track2PullQueueEntry,
  type Track2PullRunState,
} from "../lib/api";
import { Card } from "./Card";
import { ConfirmDialog } from "./ConfirmDialog";

/** Sweeps lined up, so a second one waits instead of running beside the
 * first.
 *
 * The sibling of the training queue, and deliberately shaped like it: a
 * list in the order it will run, the thing currently running called out
 * above it, and a way to remove an entry whose turn has not come.
 *
 * What differs is what an entry MEANS. A training entry is a model to
 * build. A pull entry is a set of folders to sweep, snapshotted when it
 * was queued -- so the settings that run are the ones that were on screen
 * at the time, not whatever the card holds when its turn comes round.
 */
export function Track2PullQueuePanel() {
  const [entries, setEntries] = useState<Track2PullQueueEntry[]>([]);
  const [running, setRunning] = useState<Track2PullRunState | null>(null);
  const [busy, setBusy] = useState("");
  // Clearing the queue throws away work somebody lined up. Asked in a
  // dialog that says how many and what survives, rather than a browser
  // box that can only manage one sentence.
  const [confirmClear, setConfirmClear] = useState(false);
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  const refresh = useCallback(async () => {
    try {
      const got = await getTrack2PullQueue();
      setEntries(got.entries);
      setRunning(got.running);
      setError("");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not read the pull queue.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    // Same cadence as the app's own queue tick, so what is on screen and
    // what the server is doing never drift by more than one tick.
    const timer = window.setInterval(() => void refresh(), 25000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  async function withBusy(label: string, work: () => Promise<void>) {
    setBusy(label);
    setError("");
    try {
      await work();
      await refresh();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "That did not work.");
    } finally {
      setBusy("");
    }
  }

  const live = running && running.status === "running";

  return (
    <Card
      title="Pull queue"
      caption="Sweeps waiting their turn. One runs at a time -- a second trigger joins the queue instead of sweeping alongside the first."
      summary={
        <p className="tabular text-xs text-ink-muted">
          {live ? "1 sweeping" : "nothing sweeping"}
          {entries.length ? `, ${entries.length} queued` : ", queue empty"}
        </p>
      }
      help={
        <>
          <p>
            A sweep <strong>moves</strong> files out of the drop folders and
            there is no second copy, so two running at once duplicate every
            copy and race each other&rsquo;s deletes. This lines them up.
          </p>
          <p className="mt-2">
            Each entry carries the settings it was queued with. Change the
            folders on the Pull Control card afterwards and the queued entry
            still sweeps what it was queued to sweep &mdash; which is what
            lets you line up several different pulls and walk away.
          </p>
          <p className="mt-2">
            It drains whether or not this page is open: the app ticks it
            every 25 seconds, and the router Lambda ticks it on its own
            schedule, so a queued sweep still runs while the app is down.
          </p>
        </>
      }
    >
      {error && (
        <p className="mb-2 rounded-md border border-critical/40 bg-critical-subtle px-3 py-2 text-xs text-critical">
          {error}
        </p>
      )}

      {/* What is sweeping NOW, above the queue -- the queue is what is
          waiting, and conflating the two makes "is anything happening?"
          a question you have to work out rather than read. */}
      <div className="mb-3 rounded-md border border-hairline bg-page p-2.5">
        {live ? (
          <>
            <p className="text-xs font-medium text-ink">Sweeping now</p>
            <p className="tabular mt-0.5 text-xs text-ink-secondary">
              {running?.run_id}
              {typeof running?.moved === "number"
                ? ` -- ${running.moved.toLocaleString()} moved`
                : ""}
            </p>
          </>
        ) : (
          <p className="text-xs text-ink-muted">
            Nothing sweeping. The next queued entry starts within 25 seconds.
          </p>
        )}
      </div>

      <div className="mb-3 flex flex-wrap items-end gap-2">
        <label className="flex min-w-0 flex-1 flex-col gap-1 text-xs text-ink-secondary">
          Note <span className="text-ink-muted/70">(optional, so you can tell entries apart)</span>
          <input
            className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
            value={note}
            placeholder="site A backlog"
            onChange={(e) => setNote(e.target.value)}
          />
        </label>
        <button
          type="button"
          disabled={!!busy}
          onClick={() => withBusy("Queueing...", async () => {
            await queueTrack2Pull(note);
            setNote("");
          })}
          className="rounded-md bg-accent px-3 py-2 text-xs font-medium text-white disabled:opacity-40"
        >
          {busy === "Queueing..." ? "Queueing..." : "Queue a sweep"}
        </button>
      </div>
      <p className="mb-3 text-xs text-ink-muted">
        Queues the Pull Control settings <em>as they stand now</em>.
      </p>

      {entries.length === 0 ? (
        <p className="text-xs text-ink-muted">Nothing queued.</p>
      ) : (
        <div className="flex flex-col gap-1">
          {entries.map((entry, index) => (
            <div
              key={entry.id}
              className="flex items-center gap-2 rounded-sm border border-hairline bg-page px-2.5 py-2"
            >
              <span className="tabular w-6 shrink-0 text-xs text-ink-muted">
                {index + 1}
              </span>
              <div className="min-w-0 flex-1">
                <p className="tabular truncate text-xs text-ink">{entry.summary}</p>
                <p className="tabular truncate text-xs text-ink-muted">
                  {new Date(entry.added_at).toLocaleString()}
                  {entry.note ? ` -- ${entry.note}` : ""}
                  {entry.added_by ? ` (${entry.added_by})` : ""}
                </p>
              </div>
              <button
                type="button"
                disabled={!!busy}
                onClick={() => withBusy("Removing...", async () => {
                  await removeTrack2PullQueueEntry(entry.id);
                })}
                className="shrink-0 text-xs text-ink-secondary underline disabled:opacity-40"
              >
                remove
              </button>
            </div>
          ))}
        </div>
      )}

      {entries.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            disabled={!!busy || !!live}
            title={live ? "A sweep is already running." : undefined}
            onClick={() => withBusy("Starting...", async () => {
              const out = await advanceTrack2PullQueue();
              if (!out.started && out.reason) setError(out.reason);
            })}
            className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40"
          >
            Start the next one now
          </button>
          <button
            type="button"
            disabled={!!busy}
            onClick={() => setConfirmClear(true)}
            className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40"
          >
            Clear the queue
          </button>
        </div>
      )}
      <ConfirmDialog
        open={confirmClear}
        title={`Clear ${entries.length} queued sweep${entries.length === 1 ? "" : "s"}?`}
        destructive
        busy={busy === "Clearing..."}
        confirmLabel="Clear the queue"
        cancelLabel="Keep them"
        onCancel={() => setConfirmClear(false)}
        onConfirm={() => void withBusy("Clearing...", async () => {
          await clearTrack2PullQueue();
          setConfirmClear(false);
        })}
        body={
          <>
            <p>
              Removes everything waiting, with the settings each one was
              queued with. Nothing is collected and nothing already in the
              week folders is touched.
            </p>
            <p className="mt-2">
              <strong>A sweep already running keeps going.</strong> This
              only empties the queue behind it.
            </p>
          </>
        }
      />

    </Card>
  );
}
