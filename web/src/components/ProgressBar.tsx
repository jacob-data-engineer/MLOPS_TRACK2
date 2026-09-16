/** A filled ink bar on a hairline-bordered track -- no new accent color,
 * per DESIGN.md's "One Red Rule" (the system's only saturated hue is
 * reserved for verdicts and urgent errors, not a progress indicator).
 *
 * Lifted out of `BulkRunPanel` when Track1's batch run needed the same
 * readout: DESIGN.md calls this "the one progress-bar treatment in the
 * system", so a second background job shares this component rather than
 * growing a second bar that drifts from it. */
export function ProgressBar({ processed, total }: { processed: number; total: number }) {
  const pct = total > 0 ? Math.min(100, Math.round((processed / total) * 100)) : 0;
  return (
    <div className="h-2 w-full overflow-hidden rounded-sm border border-hairline bg-page">
      <div className="h-full bg-ink transition-[width] duration-300" style={{ width: `${pct}%` }} />
    </div>
  );
}
