import { AlertCircle, ChevronDown, ChevronRight, Loader2, RefreshCw, Wallet } from "lucide-react";
import { useEffect, useState } from "react";
import { Card } from "../components/Card";
import {
  ApiError,
  getTrack2Cost,
  track2HostingTotal,
  type Track2CostDay,
  type Track2CostItem,
  type Track2CostResponse,
} from "../lib/api";
import { formatTimestamp } from "../lib/format";

const RANGES = [1, 7, 14, 30] as const;


function money(amount: number, currency: string): string {
  const value = amount.toFixed(2);
  return currency === "USD" ? `$${value}` : `${value} ${currency}`;
}

/** Single series (spend per day), so there's no legend and no categorical
 * palette to validate -- one accent hue, with the number shown beside every
 * bar rather than encoded by color alone. The bar is proportional to the
 * largest day in view, which is why it's a relative read, not an axis. */
function CostBar({ amount, max }: { amount: number; max: number }) {
  const pct = max > 0 ? Math.max((amount / max) * 100, 1.5) : 0;
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-black/[0.06]">
      <div className="h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
    </div>
  );
}

function UsageTypeRows({ items, currency, max }: { items: Track2CostItem[]; currency: string; max: number }) {
  if (items.length === 0) {
    return <p className="px-3 py-2 text-xs text-ink-muted">No charges this day.</p>;
  }
  return (
    <div className="flex flex-col gap-1.5 border-t border-hairline bg-surface px-3 py-2">
      {items.map((item) => (
        <div key={item.usage_type} className="flex items-center gap-3">
          <p className="tabular w-64 shrink-0 truncate text-xs text-ink-secondary" title={item.usage_type}>
            {item.usage_type}
          </p>
          <div className="min-w-0 flex-1">
            <CostBar amount={item.amount} max={max} />
          </div>
          <p className="tabular w-20 shrink-0 text-right text-xs text-ink">{money(item.amount, currency)}</p>
        </div>
      ))}
    </div>
  );
}

function DayRow({ day, currency, max }: { day: Track2CostDay; currency: string; max: number }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-b border-hairline last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-3 px-3 py-2 text-left hover:bg-black/[0.03]"
        title={`${day.items.length} usage type${day.items.length === 1 ? "" : "s"} on ${day.date}`}
      >
        {open ? (
          <ChevronDown size={13} className="shrink-0 text-ink-muted" />
        ) : (
          <ChevronRight size={13} className="shrink-0 text-ink-muted" />
        )}
        <p className="tabular w-24 shrink-0 text-sm text-ink">{day.date}</p>
        <div className="min-w-0 flex-1">
          <CostBar amount={day.total} max={max} />
        </div>
        <p className="tabular w-24 shrink-0 text-right text-sm font-semibold text-ink">
          {money(day.total, currency)}
        </p>
      </button>
      {open && <UsageTypeRows items={day.items} currency={currency} max={max} />}
    </div>
  );
}

export function Track2CostPage() {
  const [data, setData] = useState<Track2CostResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [days, setDays] = useState<number>(14);

  async function load(nextDays: number, refresh = false) {
    setLoading(true);
    setError(null);
    try {
      setData(await getTrack2Cost(nextDays, refresh));
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not load cost data.");
    } finally {
      setLoading(false);
    }
  }

  // Fetched once on open and then only on an explicit Refresh / range
  // change -- Cost Explorer bills per request, so this must never poll.
  useEffect(() => {
    load(days);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [days]);

  const maxDay = data ? Math.max(...data.days.map((d) => d.total), 0) : 0;
  const maxUsageType = data ? Math.max(...data.by_usage_type.map((i) => i.amount), 0) : 0;

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Wallet size={20} className="text-accent" strokeWidth={2} />
        Track2 SageMaker Cost
      </h1>
      <p className="mb-4 max-w-3xl text-sm text-ink-secondary">
        SageMaker spend per 24-hour period, split by usage type. Hosting
        (&quot;Host:&quot;) is the standing cost of the inference endpoint and
        accrues every hour it exists; training, processing and Studio lines are
        one-off work.
      </p>

      <div className="mb-6 flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={() => load(days, true)}
          disabled={loading}
          className="inline-flex items-center gap-1.5 rounded-sm border border-black/15 px-3 py-1.5 text-sm font-medium text-ink-secondary disabled:opacity-40"
        >
          <RefreshCw size={13} className={loading ? "animate-spin" : ""} />
          {loading ? "Refreshing..." : "Refresh"}
        </button>
        <div className="flex items-center gap-1">
          {RANGES.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => setDays(r)}
              disabled={loading}
              className={`rounded-sm border px-2.5 py-1.5 text-xs font-medium disabled:opacity-40 ${
                days === r ? "border-accent bg-accent-subtle text-ink" : "border-black/15 text-ink-secondary"
              }`}
            >
              {r}d
            </button>
          ))}
        </div>
        {data && (
          <p className="tabular text-xs text-ink-muted">Fetched {formatTimestamp(data.fetched_at)}</p>
        )}
      </div>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {!data && loading && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading...
        </div>
      )}

      {data && (
        <div className="flex flex-col gap-6">
          {data.days.length > 0 && (
            <div className="rounded-sm border border-hairline bg-page p-4">
              <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
                Endpoint hosting &middot; {data.days[0].date}
                {data.days[0].date === new Date().toISOString().slice(0, 10) && " (today, still accruing)"}
              </p>
              <p className="mt-1 text-3xl font-bold text-ink">
                {money(track2HostingTotal(data.days[0].items), data.currency)}
              </p>
              <p className="mt-1 text-xs text-ink-secondary">
                One day of the inference endpoint alone, separate from training
                and Studio. This is the line that accrues while nobody is running
                anything &mdash; it should read close to zero on a day the
                endpoint was only up for the jobs that needed it. Today&apos;s
                figure is always incomplete: Cost Explorer reports hours late, so
                compare whole days rather than reading this one as final.
              </p>
            </div>
          )}

          <div className="rounded-sm border border-hairline bg-page p-4">
            <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
              All SageMaker, last {data.days.length} days
            </p>
            <p className="mt-1 text-3xl font-bold text-ink">{money(data.total, data.currency)}</p>
            <p className="mt-1 text-xs text-ink-secondary">
              {data.start} to {data.end} (end exclusive). The most recent day is
              usually partial &mdash; Cost Explorer lags a few hours, so a low final
              bar is normal rather than a real drop.
            </p>
          </div>

          <Card title="Per 24 hours">
            <div className="rounded-sm border border-hairline bg-page">
              {data.days.length === 0 ? (
                <p className="px-3 py-3 text-sm text-ink-muted">No SageMaker charges in this period.</p>
              ) : (
                data.days.map((day) => (
                  <DayRow key={day.date} day={day} currency={data.currency} max={maxDay} />
                ))
              )}
            </div>
            <p className="mt-2 text-xs text-ink-muted">
              Click a day to see which usage types made it up.
            </p>
          </Card>

          <Card title="By usage type, whole period">
            <div className="rounded-sm border border-hairline bg-page px-3 py-2">
              <UsageTypeRows items={data.by_usage_type} currency={data.currency} max={maxUsageType} />
            </div>
          </Card>
        </div>
      )}
    </div>
  );
}
