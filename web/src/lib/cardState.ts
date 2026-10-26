import { useEffect, useState } from "react";
import { getTrack2PullControl, saveTrack2PullConfig } from "./api";

/**
 * Which cards are open, saved with the rest of the settings.
 *
 * ONE SHARED RECORD, not a save per card. `write_config` merges at the
 * TOP level, so `card_state` is replaced wholesale on every write --
 * seven cards each writing their own key would mean six of them being
 * dropped by whichever wrote last. So every card reads and writes this
 * one object, and it is sent once, debounced.
 *
 * This is not a setting in the sense the others are: it changes nothing
 * about what a run does. It is here because restoring a settings file
 * should put the page back the way it was left, and a page that reopens
 * with every card collapsed has to be rebuilt by hand before it can be
 * read.
 */
let state: Record<string, boolean> = {};
let ready = false;
let inFlight: Promise<void> | null = null;
let timer: ReturnType<typeof setTimeout> | undefined;
const listeners = new Set<() => void>();

function notify() {
  listeners.forEach((listener) => listener());
}

/** Reads the saved record once, however many cards ask for it. */
export function ensureCardState(): Promise<void> {
  if (!inFlight) {
    inFlight = getTrack2PullControl()
      .then((status) => {
        state = { ...(status.config.card_state || {}) };
      })
      .catch(() => undefined)
      .finally(() => {
        ready = true;
        notify();
      });
  }
  return inFlight;
}

function scheduleSave() {
  if (timer) clearTimeout(timer);
  timer = setTimeout(() => {
    void saveTrack2PullConfig({ card_state: state }).catch(() => undefined);
  }, 600);
}

/**
 * A card's open/closed state, restored from the settings and saved back.
 *
 * A real drop-in for `useState(fallback)`, which means the setter has to
 * take BOTH forms: `setOpen(false)` and `setOpen((v) => !v)`. Every
 * toggle in this codebase is written the second way, so a setter that
 * only accepted a boolean would store the function itself -- and every
 * card that uses the updater form fails to compile, which is how this
 * was caught.
 *
 * The fallback is used until the saved record has been read, and for a
 * card the record says nothing about, so a new card opens the way its
 * author intended rather than closed.
 */
export function useCardOpen(
  key: string,
  fallback: boolean,
): [boolean, (open: boolean | ((previous: boolean) => boolean)) => void] {
  // Always called, both of them, because hooks cannot be conditional.
  // `local` is only USED when there is no key -- a card with no title and
  // no explicit one, which has nothing stable to be remembered by.
  const [local, setLocal] = useState(fallback);
  const [, bump] = useState(0);

  useEffect(() => {
    const listener = () => bump((n) => n + 1);
    listeners.add(listener);
    void ensureCardState();
    return () => {
      listeners.delete(listener);
    };
  }, []);

  if (!key) return [local, setLocal];

  const open = ready && key in state ? state[key] : fallback;

  function setOpen(next: boolean | ((previous: boolean) => boolean)) {
    // Resolved against what is on screen now, which is the saved value
    // when there is one and the fallback before the read lands.
    const current = ready && key in state ? state[key] : fallback;
    const value = typeof next === "function" ? next(current) : next;
    state = { ...state, [key]: value };
    notify();
    scheduleSave();
  }

  return [open, setOpen];
}
