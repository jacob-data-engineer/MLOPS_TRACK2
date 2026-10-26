/** One switch, two cards, no disagreement.
 *
 * `retention.use_golden` is shown in two places now: the Golden Set card,
 * which owns it, and the three-source row on the Assemble card, where it
 * is chosen alongside the other two sources. Both read it on mount and
 * both write it with a PATCH, so they agree after any reload -- but not
 * while the page is open, and a page showing one answer in two places is
 * exactly the trap the Golden Set card's own comment warned about.
 *
 * A browser event rather than a store: there is one field, it changes
 * rarely, and the alternative is a context provider threaded through two
 * cards that are otherwise independent.
 */
const EVENT = "track2:use-golden";

export function announceGolden(value: boolean): void {
  window.dispatchEvent(new CustomEvent<boolean>(EVENT, { detail: value }));
}

/** Calls `onChange` when the OTHER card changes it. Returns the cleanup,
 * so it drops straight into a `useEffect`. */
export function onGoldenChanged(handler: (value: boolean) => void): () => void {
  const listen = (e: Event) => handler(Boolean((e as CustomEvent<boolean>).detail));
  window.addEventListener(EVENT, listen);
  return () => window.removeEventListener(EVENT, listen);
}
