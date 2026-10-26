import type { ReactNode } from "react";

/**
 * Wraps a card that is out of play, saying so rather than removing it.
 *
 * When the dataset is picked by hand, Pull Control, Auto Training and
 * Retention are not just unnecessary -- they are a second, contradicting
 * answer to "where does the training data come from". Leaving them live
 * means two cards claiming to decide the same thing, and the one that
 * wins is whichever wrote `datasetFolder` last.
 *
 * Dimmed and left where it was, not hidden. Hiding them would shorten
 * the page and cost the operator the one thing they need to know: that
 * these exist, that they are off, and what turns them back on. A card
 * that vanishes teaches nobody anything.
 *
 * `inert` is what actually makes it unusable -- it takes the subtree out
 * of the tab order and swallows clicks. `pointer-events-none` alone
 * leaves every control still reachable by keyboard, which is the
 * accessibility version of looking disabled without being disabled.
 */
export function SwitchedOff({
  off,
  reason,
  children,
}: {
  off: boolean;
  /** One line, naming what turned this off and how to turn it back on. */
  reason: ReactNode;
  children: ReactNode;
}) {
  if (!off) return <>{children}</>;
  return (
    <div className="relative">
      <p className="mb-1 rounded-md border border-accent/40 bg-accent-subtle px-3 py-1.5 text-xs text-ink-secondary">
        <span className="font-medium text-ink">Off.</span> {reason}
      </p>
      <div inert className="pointer-events-none select-none opacity-45 saturate-50">
        {children}
      </div>
    </div>
  );
}
