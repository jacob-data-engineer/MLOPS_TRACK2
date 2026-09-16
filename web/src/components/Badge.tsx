import {
  AlertTriangle,
  CheckCircle2,
  CircleDashed,
  Info,
  XCircle,
  type LucideIcon,
} from "lucide-react";

/**
 * Non-verdict status pill -- deployment status, approval status, and
 * similar lifecycle states that the direction contract explicitly allows
 * to stay simpler than a full stamp (verdicts -- TP/FP/uncertain,
 * CONFIRM_TP/CONFIRM_FP/ESCALATE -- render as a `VerdictStamp` instead,
 * never as a `Badge`).
 *
 * This system spends its one saturated color on stamps, so these pills
 * carry no color-coded hue triad either: "confirmed" is a solid ink fill,
 * "pending" is an ink outline, "rejected" fades the ink and swaps the
 * icon -- weight and icon carry the meaning, never color alone (several
 * of the old hues sat under 3:1 contrast for exactly this reason).
 */
export type BadgeTone = "confirmed" | "pending" | "rejected" | "info" | "neutral";

const TONE_STYLES: Record<BadgeTone, string> = {
  confirmed: "bg-ink text-surface",
  pending: "border border-ink/40 text-ink-secondary",
  rejected: "text-ink-muted line-through decoration-1",
  info: "bg-accent-subtle text-ink",
  neutral: "bg-black/[0.05] text-ink-secondary",
};

const TONE_ICON: Record<BadgeTone, LucideIcon> = {
  confirmed: CheckCircle2,
  pending: CircleDashed,
  rejected: XCircle,
  info: Info,
  neutral: AlertTriangle,
};

export function Badge({ tone, label }: { tone: BadgeTone; label: string }) {
  const Icon = TONE_ICON[tone];
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${TONE_STYLES[tone]}`}
    >
      <Icon size={13} strokeWidth={2.25} />
      {label}
    </span>
  );
}
