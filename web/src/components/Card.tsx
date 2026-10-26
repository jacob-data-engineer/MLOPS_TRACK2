import { useState, type ReactNode } from "react";
import { useCardOpen } from "../lib/cardState";
import { CardBoundary } from "./CardBoundary";

/** The one card container used everywhere: a manifest-page panel -- ruled
 * hairline border, cream surface fill, square-ish corners like a case-file
 * sheet rather than a rounded SaaS tile -- so every panel in the app reads
 * as the same system rather than one-off containers per screen.
 *
 * Every card collapses. The button lives here rather than in each panel
 * because there are thirty-odd cards across nineteen files, and a
 * per-card copy would drift: four of them already had one, each written
 * slightly differently, and the rest had none. Here it is one
 * implementation and every card gets it.
 *
 * Open by default, so adding this changed nothing anybody had to undo.
 * A card that wants to start shut says `defaultOpen={false}`; the four
 * that run their own toggle -- because they show a summary of live state
 * when collapsed -- say `collapsible={false}` and keep it.
 *
 * WHETHER A CARD IS OPEN IS SAVED, with the rest of the settings, so a
 * loaded file puts the page back as it was left rather than collapsed.
 * Remembered against the card's TITLE, so every one of the thirty-odd
 * cards gets this without thirty call sites being edited. A card with no
 * title, or one that should keep its memory across a retitling, passes
 * `stateKey`.
 *
 * Renaming a title forgets how that card was left, once, and it falls
 * back to `defaultOpen` -- which is what it did before any of this. That
 * is the right trade for not maintaining a key per card.
 */
export function Card({
  title,
  titleAside,
  caption,
  action,
  children,
  collapsible = true,
  defaultOpen = true,
  stateKey,
  summary,
  step,
  help,
  next: whatNext,
}: {
  title?: string;
  /** Drawn immediately after the title text, inside the heading.
   *
   * `action` is the far right of the header, next to Hide -- right for a
   * button, wrong for a tick that says whether the card is in play: at
   * that distance it reads as belonging to Hide rather than to the
   * title. This sits against the words it qualifies. */
  titleAside?: ReactNode;
  caption?: string;
  action?: ReactNode;
  children: ReactNode;
  /** False for a card that renders its own Show/Hide. */
  collapsible?: boolean;
  defaultOpen?: boolean;
  /** What this card's open/closed state is remembered as. Defaults to
   * the title. Pass one for a card with no title. */
  stateKey?: string;
  /** Drawn in place of the body while collapsed. Without one a shut card
   * is a title and nothing else, which says less than it could. */
  summary?: ReactNode;
  /** Where this card sits in the run of them, e.g. "2". Six cards in a
   * fixed order with nothing saying so is a sequence you have to have
   * been told. */
  step?: string;
  /** What the card is for, in the reader's terms. Folded away behind a
   * "?" -- shown by default it is noise to whoever already knows, and
   * absent it is a card you have to be shown by somebody. */
  help?: ReactNode;
  /** The handover. Stated at the bottom because the question "what now"
   * arrives when you have finished, not when you arrive. */
  next?: ReactNode;
}) {
  // Keyed on the title when nothing better is given. Lower-cased and
  // squeezed so a stray capital or double space cannot make a card
  // forget how it was left.
  const rememberAs = (stateKey ?? title ?? "")
    .toLowerCase().replace(/\s+/g, " ").trim();
  const [open, setOpen] = useCardOpen(rememberAs, defaultOpen);
  const [helpOpen, setHelpOpen] = useState(false);
  const shut = collapsible && !open;

  return (
    <section className="rounded-sm border border-hairline bg-surface p-5 shadow-[0_1px_0_rgba(36,29,22,0.06)]">
      {(title || action || collapsible) && (
        <div className="mb-3 flex items-start justify-between gap-4 border-b border-hairline pb-2.5">
          {title && (
            <h3 className="flex items-baseline gap-2 text-base font-semibold tracking-tight text-ink">
              {step && (
                <span className="tabular rounded-sm bg-ink/10 px-1.5 text-xs font-medium text-ink-secondary">
                  {step}
                </span>
              )}
              {title}
              {titleAside}
            </h3>
          )}
          <div className="ml-auto flex shrink-0 items-center gap-2">
            {help && (
              <button
                type="button"
                onClick={() => setHelpOpen((v) => !v)}
                aria-expanded={helpOpen}
                title="What is this card for?"
                className="rounded-md border border-black/15 px-2.5 py-1.5 text-xs font-medium text-ink-secondary"
              >
                {helpOpen ? "Close help" : "?"}
              </button>
            )}
            {action}
            {collapsible && (
              <button
                type="button"
                onClick={() => setOpen((v) => !v)}
                aria-expanded={open}
                className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
              >
                {open ? "Hide" : "Show"}
              </button>
            )}
          </div>
        </div>
      )}
      {/* Readable while the card is SHUT as well. "What is this for" is
          most often asked by somebody who has not opened it yet. */}
      {helpOpen && help && (
        <div className="mb-3 rounded-md border border-accent/40 bg-page p-3 text-sm text-ink-secondary">
          {help}
        </div>
      )}
      {caption && !shut && (
        <p className="mb-3 text-sm text-ink-secondary">{caption}</p>
      )}
      {shut ? (
        // The summary renders when the card is SHUT, and it reads the
        // same state the body does -- so it can throw for the same
        // reasons. Outside the boundary it would take the page with it.
        summary ? <CardBoundary title={title}>{summary}</CardBoundary> : null
      ) : (
        <>
          <CardBoundary title={title}>{children}</CardBoundary>
          {whatNext && (
            <p className="mt-3 border-t border-hairline pt-2 text-xs text-ink-muted">
              <span className="font-medium text-ink-secondary">Next:</span>{" "}
              {whatNext}
            </p>
          )}
        </>
      )}
    </section>
  );
}
