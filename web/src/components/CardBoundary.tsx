import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * Keeps one card's failure inside that card.
 *
 * Without this, a throw during render unmounts the entire React tree and
 * the page goes white with nothing on it -- no message, no other cards,
 * nothing saying which of a dozen panels was at fault. That happened:
 * `builtTree` in pull control was an IIFE reading a `const` declared
 * below it, so the moment a saved config filled the folder lists the
 * card threw and took the whole page with it.
 *
 * The error is shown where the card was, so the page still works and the
 * failure names itself. `Try again` re-renders rather than reloads: the
 * usual cause is state that only some values break on, and a reload
 * would throw the state away before it could be looked at.
 *
 * A class, because that is the only way React offers to catch a render
 * error -- there is no hook for it.
 */
export class CardBoundary extends Component<
  { children: ReactNode; title?: string },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Straight to the console, with the component stack: the message
    // alone rarely says which card, and the stack always does.
    console.error(`Card failed${this.props.title ? `: ${this.props.title}` : ""}`,
                  error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="rounded-md border border-critical/40 bg-critical-subtle p-3">
        <p className="text-xs font-medium text-critical">
          This card failed to render.
        </p>
        <p className="tabular mt-1 break-words text-xs text-ink-secondary">
          {error.message || String(error)}
        </p>
        <p className="mt-1 text-xs text-ink-muted">
          The rest of the page is unaffected. The browser console has the
          component stack.
        </p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-2 rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
        >
          Try again
        </button>
      </div>
    );
  }
}
