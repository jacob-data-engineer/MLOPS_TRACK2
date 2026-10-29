/** One switch, two cards, no disagreement.
 *
 * `retention.use_golden` is shown in two places now: the Golden Set card,
 * which owns it, and the three-source row on the Assemble card, where it
 * is chosen alongside the other two sources. Both read it on mount and
 * both write it with a PATCH, so they agree after any reload -- but not
 * while the page is open, and a page showing one answer in two places is
 * exactly the trap the Golden Set card's own comment warned about.
 *
 * Carried by the notice board (lib/bus.ts) since the data/training split,
 * so it also reaches the Training tab open in another window. A change to
 * the golden set is also a change to what a run trains on, so it posts
 * that too.
 */
import { listen, post } from "./bus";

export function announceGolden(value: boolean): void {
  post("golden-changed", value);
  post("data-sources-changed", { from: "golden-set" });
}

/** Calls `onChange` when the OTHER card changes it. Returns the cleanup,
 * so it drops straight into a `useEffect`. */
export function onGoldenChanged(handler: (value: boolean) => void): () => void {
  return listen("golden-changed", (value) => handler(Boolean(value)));
}
