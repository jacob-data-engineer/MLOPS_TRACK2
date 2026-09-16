import type { ReactNode } from "react";

/**
 * Value + optional delta, the "figure contract" from the dataviz skill's
 * stat-tile spec. Numeric value always renders with tabular figures (now
 * monospace, per the case-file direction -- see `.tabular` in index.css)
 * so a row of tiles lines up.
 *
 * `deltaTone` no longer carries a color -- "critical" still reads as a
 * miss because it's the same alarm-red this system reserves for
 * verdicts/errors; "good" is plain bold ink, not green, since nothing
 * outside a stamp gets a second hue.
 */
export function StatTile({
  label,
  value,
  delta,
  deltaTone = "neutral",
  help,
}: {
  label: string;
  value: ReactNode;
  delta?: string;
  deltaTone?: "good" | "critical" | "neutral";
  help?: string;
}) {
  const deltaColor =
    deltaTone === "good"
      ? "font-semibold text-ink"
      : deltaTone === "critical"
        ? "font-semibold text-critical"
        : "text-ink-muted";

  return (
    <div className="flex flex-col gap-1.5 rounded-sm border border-hairline bg-surface p-4">
      <span className="text-sm text-ink-secondary">{label}</span>
      <span className="tabular text-2xl font-semibold leading-none text-ink">{value}</span>
      {delta && <span className={`tabular text-xs ${deltaColor}`}>{delta}</span>}
      {help && <span className="text-xs text-ink-muted">{help}</span>}
    </div>
  );
}
