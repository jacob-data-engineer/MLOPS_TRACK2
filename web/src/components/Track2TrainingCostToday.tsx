import { useEffect, useState } from "react";
import { getTrack2Cost, track2TrainingTotal } from "../lib/api";

/** What SageMaker training has cost today -- the GPU training jobs plus the
 * evaluation Processing jobs that run with them.
 *
 * Read once on mount, never on a poll: Cost Explorer charges per request,
 * and the figure only moves hourly anyway. The backend caches it too, so
 * coming back to this page is free.
 */
export function Track2TrainingCostToday() {
  const [cost, setCost] = useState<{ amount: number; currency: string; date: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getTrack2Cost(1)
      .then((res) => {
        const latest = res.days[0];
        if (!latest || cancelled) return;
        setCost({
          amount: track2TrainingTotal(latest.items),
          currency: res.currency,
          date: latest.date,
        });
      })
      .catch(() => {
        // A cost read failing is not worth blocking the page over -- the
        // tile just stays hidden.
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (!cost) return null;
  const value =
    cost.currency === "USD" ? `$${cost.amount.toFixed(2)}` : `${cost.amount.toFixed(2)} ${cost.currency}`;
  return (
    <div className="mb-4 flex flex-wrap items-baseline justify-between gap-2 rounded-sm border border-hairline bg-page px-3 py-2">
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
        SageMaker training cost today &middot; {cost.date}
      </p>
      <p className="tabular text-lg font-bold text-ink">{value}</p>
      <p className="w-full text-xs text-ink-secondary">
        Training and evaluation jobs only -- the inference endpoint and Studio
        are billed separately. Cost Explorer reports several hours behind, so a
        run you started just now will not appear here yet.
      </p>
    </div>
  );
}
