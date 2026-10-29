/** The notice board: how cards tell each other something changed.
 *
 * Added with the data/training split (2026-10-28). Before it, cards spoke
 * through the training page -- it held the state and passed callbacks
 * down -- which only works while every card is on that page. With the
 * data cards on their own tab there is no shared parent to pass through,
 * so a card POSTS what happened here and any card LISTENS for it, without
 * either knowing the other exists.
 *
 * Messages are hints to re-read, never the data itself: the settings
 * record on the server stays the one source of truth, so a listener that
 * missed a message is corrected by its next load rather than left wrong.
 *
 * Two carriers. A window event reaches cards on the same page; a
 * BroadcastChannel reaches the app open in another browser tab or window
 * -- the Data Preparation tab in one, Training in the other -- which is
 * exactly the split's common case.
 */

/** Every message a card may post, and what it carries. Adding one here is
 * the whole contract: the type checker then holds both ends to it. */
export interface Track2Messages {
  /** The golden set was ticked on or off (`retention.use_golden`). */
  "golden-changed": boolean;
  /** A data card saved something that changes what a run trains on:
   *  Assemble's folders or classes, the golden set, the week ticks,
   *  retention. Training re-reads its summary of the sources. */
  "data-sources-changed": { from: string };
}

type Name = keyof Track2Messages;

const PREFIX = "track2:";
const CHANNEL = "track2-notice-board";

let channel: BroadcastChannel | null = null;
function theChannel(): BroadcastChannel | null {
  if (channel || typeof BroadcastChannel === "undefined") return channel;
  try {
    channel = new BroadcastChannel(CHANNEL);
    // Re-raised as a window event, so listeners only ever subscribe one way.
    channel.onmessage = (e: MessageEvent<{ name: Name; detail: unknown }>) => {
      if (e.data && typeof e.data.name === "string") {
        window.dispatchEvent(new CustomEvent(PREFIX + e.data.name, { detail: e.data.detail }));
      }
    };
  } catch {
    channel = null;
  }
  return channel;
}

/** Tell every listening card, on this page and in the app's other open windows. */
export function post<K extends Name>(name: K, detail: Track2Messages[K]): void {
  window.dispatchEvent(new CustomEvent(PREFIX + name, { detail }));
  try {
    theChannel()?.postMessage({ name, detail });
  } catch {
    // A closed channel is a missed hint, not an error: the next load re-reads.
  }
}

/** Calls `handler` for each post of `name`. Returns the cleanup, so it
 * drops straight into a `useEffect`. */
export function listen<K extends Name>(name: K, handler: (detail: Track2Messages[K]) => void): () => void {
  theChannel();
  const relay = (e: Event) => handler((e as CustomEvent<Track2Messages[K]>).detail);
  window.addEventListener(PREFIX + name, relay);
  return () => window.removeEventListener(PREFIX + name, relay);
}
