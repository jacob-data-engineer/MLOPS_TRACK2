import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * What to actually DO about it, where the message alone does not say.
 *
 * An error card that prints `Cannot read properties of undefined` tells
 * somebody that something is broken, which they already knew from the
 * blank card. The useful half is whether to wait, re-sign-in, or report
 * it -- and for the handful of failures this app really produces, that
 * is knowable.
 */
function adviceFor(error: Error): string {
  const text = `${error.message || error}`.toLowerCase();
  if (text.includes("rate exceeded") || text.includes("throttl")) {
    return "The AWS registry is rate-limiting us. Wait a few seconds and try "
      + "again -- this clears on its own and nothing is wrong with the data.";
  }
  if (text.includes("401") || text.includes("session") || text.includes("token")) {
    return "The session probably ended -- every deploy ends every session. "
      + "Sign out and back in.";
  }
  if (text.includes("networkerror") || text.includes("failed to fetch")) {
    return "The browser could not reach the server. Check the container is "
      + "still up on this port before looking any further.";
  }
  if (text.includes("before initialization") || text.includes("is not a function")
      || text.includes("undefined")) {
    return "This is a bug in the page itself, not in your data or your "
      + "session -- reloading will not fix it. Worth reporting with the "
      + "message above.";
  }
  return "If it happens again on the same page, the message above is the "
    + "part worth reporting.";
}

/**
 * Keeps one PAGE's failure inside that page, with the nav still working.
 *
 * `CardBoundary` only wraps a card's children, so it catches a throw in
 * what a card renders. It cannot catch a throw in the panel component
 * that renders the card -- that code runs above the boundary. That is
 * not theoretical: a `const` in a panel body read a `useState` declared
 * below it, every render threw `Cannot access 'chosenSlug' before
 * initialization`, and because nothing above caught it React unmounted
 * the whole tree. The page was white. No message, no sidebar, no way to
 * navigate to a page that still worked.
 *
 * This sits around the route outlet, so the worst a panel can now do is
 * replace its own page. The sidebar, the build badge and every other
 * page stay usable, which is the difference between "one page is broken"
 * and "the app is down".
 *
 * Remount it with `key={pathname}` and navigating away clears the error
 * by itself -- a boundary that latches would make every later page look
 * broken too.
 */
export class RouteBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Page failed to render", error, info.componentStack);
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;
    return (
      <div className="rounded-md border border-critical/40 bg-critical-subtle p-4">
        <p className="text-sm font-semibold text-critical">
          This page failed to render.
        </p>
        <p className="tabular mt-2 break-words text-xs text-ink-secondary">
          {error.message || String(error)}
        </p>
        <p className="mt-2 text-xs text-ink-secondary">{adviceFor(error)}</p>
        <p className="mt-2 text-xs text-ink-muted">
          Every other page still works &mdash; pick one from the menu. Nothing
          was changed by this, and no run was started or stopped. The browser
          console has the full component stack.
        </p>
        <div className="mt-3 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => this.setState({ error: null })}
            className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
          >
            Try again
          </button>
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
          >
            Reload the page
          </button>
        </div>
      </div>
    );
  }
}
