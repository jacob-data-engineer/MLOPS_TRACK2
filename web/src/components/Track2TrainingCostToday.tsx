import { useCallback, useEffect, useState } from "react";
import { useCardOpen } from "../lib/cardState";
import {
  getTrack2Cost,
  getTrack2CostToday,
  track2TrainingTotal,
  type Track2CostToday,
} from "../lib/api";

/** What SageMaker training has cost today -- the GPU training jobs plus the
 * evaluation Processing jobs that run with them.
 *
 * TWO SOURCES, and the order matters.
 *
 * Cost Explorer is the authority, and it lags roughly a day. On the
 * afternoon of a day spent training it answers $0.00 -- which reads as
 * "this is free", not as "ask again tomorrow". That is the one thing a
 * cost figure must never say, and it is exactly what this card said
 * after seven runs on 6 October.
 *
 * So the measured figure leads: SageMaker reports `BillableTimeInSeconds`
 * per training job, which is the quantity it bills on, and the published
 * rate turns it into money. It agrees with the bill for the jobs it can
 * see, and it can see them the moment they finish. Cost Explorer takes
 * over once it has caught up, because it also catches what this cannot --
 * Studio notebooks and storage, which dominated this account's bill on
 * 4 October.
 */
/** "14:02-14:34", in the reader's own timezone.
 *
 * `started_at` is null until an instance is allocated -- billing starts
 * there too -- so a job still waiting on capacity says so rather than
 * being given a time it never had. */
function clockRange(job: { started_at?: string | null; ended_at?: string | null }): string {
  if (!job.started_at) return "waiting";
  const clock = (iso: string) => new Date(iso).toLocaleTimeString(undefined,
    { hour: "2-digit", minute: "2-digit" });
  const from = clock(job.started_at);
  return job.ended_at ? `${from}-${clock(job.ended_at)}` : `${from}-now`;
}

export function Track2TrainingCostToday() {
  const [explorer, setExplorer] = useState<{ amount: number; currency: string; date: string } | null>(null);
  const [measured, setMeasured] = useState<Track2CostToday | null>(null);
  const [open, setOpen] = useCardOpen("cost_today", false);
  const [busy, setBusy] = useState(false);
  const [at, setAt] = useState<Date | null>(null);

  /** The MEASURED side only. Cost Explorer is not re-read on a tick:
   * it charges $0.01 per request and lags about a day, so polling it
   * every five seconds would bill real money to re-read a figure that
   * cannot have changed. `/track2/cost/today` reads the jobs
   * themselves -- two list calls and a describe each, no CE charge. */
  const reload = useCallback(async () => {
    setBusy(true);
    try {
      setMeasured(await getTrack2CostToday());
      setAt(new Date());
    } catch {
      // A cost read failing is not worth blocking the page over, and a
      // tick that fails must not stop the next one.
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void reload();
    // Cost Explorer once, on mount. It is the bill and it lags a day.
    getTrack2Cost(1)
      .then((res) => {
        const latest = res.days[0];
        if (!latest) return;
        setExplorer({
          amount: track2TrainingTotal(latest.items),
          currency: res.currency,
          date: latest.date,
        });
      })
      .catch(() => undefined);
  }, [reload]);

  useEffect(() => {
    // Five seconds, and PAUSED when the tab is hidden. A background tab
    // polling an AWS API forever is throttling somebody else's call for
    // numbers nobody is looking at.
    let timer: number | undefined;
    const tick = () => {
      if (!document.hidden) void reload();
    };
    const start = () => {
      window.clearInterval(timer);
      timer = window.setInterval(tick, 5000);
    };
    start();
    document.addEventListener("visibilitychange", tick);
    return () => {
      window.clearInterval(timer);
      document.removeEventListener("visibilitychange", tick);
    };
  }, [reload]);

  if (!explorer && !measured) return null;

  // Cost Explorer wins once it has anything to say -- it is the bill.
  // Until then the measured figure is the only honest answer.
  const billed = explorer && explorer.amount > 0;
  const amount = billed ? explorer.amount : (measured?.estimated ?? 0);
  const currency = explorer?.currency ?? measured?.currency ?? "USD";
  const date = explorer?.date ?? measured?.date ?? "";
  const value = currency === "USD" ? `$${amount.toFixed(2)}` : `${amount.toFixed(2)} ${currency}`;

  const minutes = measured
    ? Math.round((measured.training_seconds + measured.processing_seconds) / 60)
    : 0;
  const runs = measured?.jobs.length ?? 0;

  return (
    <div className="mb-4 rounded-sm border border-hairline bg-page px-3 py-2">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
          SageMaker training cost today &middot; {date}
          {!billed && <span className="ml-2 normal-case text-ink-muted/80">(measured)</span>}
        </p>
        <div className="flex items-baseline gap-3">
          {/* Updated every five seconds, so the time it was last read is
              the thing that says whether the number is live or stuck --
              a figure that quietly stopped refreshing looks exactly like
              one that has not changed. */}
          {at && (
            <span className="tabular text-xs text-ink-muted">
              {busy ? "refreshing..." : `read ${at.toLocaleTimeString()}`}
            </span>
          )}
          <button
            type="button"
            onClick={() => void reload()}
            disabled={busy}
            className="rounded-md border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40"
          >
            {busy ? "Refreshing..." : "Refresh"}
          </button>
          <p className="tabular text-lg font-bold text-ink">{value}</p>
        </div>
      </div>

      <p className="text-xs text-ink-secondary">
        {billed ? (
          <>
            From Cost Explorer. Training and evaluation jobs only -- the
            inference endpoint and Studio are billed separately.
          </>
        ) : (
          <>
            Measured from this account&rsquo;s own jobs: {runs} job{runs === 1 ? "" : "s"},{" "}
            {minutes} minute{minutes === 1 ? "" : "s"} of billable machine time at
            the published rate. Cost Explorer reports about a day behind, so it
            still shows nothing for today; this is what those jobs will cost.
            Training and evaluation only -- Studio and storage are not counted.
            {measured && measured.estimated_other != null
              && measured.estimated_other > 0 && (
              <>
                {" "}Of that, <span className="tabular font-medium text-critical">
                  ${measured.estimated_other.toFixed(2)}
                </span>{" "}
                is from jobs this app did not start.
              </>
            )}
            {measured && !measured.complete && (
              <> A rate or a job could not be read, so treat this as a floor.</>
            )}
          </>
        )}
      </p>

      {measured && measured.jobs.length > 0 && (
        <>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            className="mt-1 text-xs font-medium text-ink-secondary underline"
          >
            {open ? "Hide the jobs" : `Show the ${measured.jobs.length} jobs`}
          </button>
          {open && (
            // Three rows tall, then it scrolls: a busy day lists a dozen
            // jobs and pushed everything below the strip off the screen.
            // 1rem line-height x 3 + two 0.125rem gaps = 3.25rem.
            <div className="tabular mt-1 flex max-h-[3.25rem] flex-col gap-0.5 overflow-y-auto pr-1 text-xs text-ink-muted">
              {/* By start time, earliest first, so the list reads as the
                  day happened. The server orders by cost; that answers a
                  different question. A job still waiting for an instance
                  has no start yet and goes last. */}
              {[...measured.jobs]
                .sort((a, b) => (a.started_at ?? "￿").localeCompare(b.started_at ?? "￿"))
                .map((job) => (
                <span key={job.name} className="shrink-0 truncate" title={job.name}>
                  ${job.cost.toFixed(3)} &middot; {job.instance_type} &middot;{" "}
                  {/* WHEN, before how long. The panel listed a duration
                      and a cost and no clock time, so "what was the GPU
                      doing at three o'clock" could not be answered from
                      the one card that lists every job. */}
                  <span className="text-ink-secondary">{clockRange(job)}</span>
                  {" "}&middot; {Math.round(job.seconds / 60)}m &middot;{" "}
                  {job.status} &middot;{" "}
                  {/* Whose job it is, before the name rather than after:
                      the names are long and truncate, and "is this ours"
                      is the first question somebody reads this list to
                      answer. */}
                  <span className={job.own_pipeline ? "text-accent" : "text-critical"}>
                    {job.own_pipeline ? "this app" : "another system"}
                  </span>{" "}
                  &middot; {job.name}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
