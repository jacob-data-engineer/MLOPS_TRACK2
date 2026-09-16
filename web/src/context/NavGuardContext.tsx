import { createContext, useCallback, useContext, useRef, type ReactNode } from "react";

interface NavGuardContextValue {
  /** Registers a function returning `true` if the calling page currently
   * has unsaved changes -- `Layout`'s sidebar `NavLink`s consult this
   * before letting a click through. Only one page's guard is ever active
   * at a time in practice (there's one route mounted at a time), so a
   * later `setGuard` call simply replaces whatever's registered. */
  setGuard: (check: () => boolean) => void;
  /** Unregisters the active guard -- call on unmount, and whenever the
   * page's own draft becomes clean (e.g. right after a save), so a stale
   * "dirty" check can't outlive the state it was reading. */
  clearGuard: () => void;
  /** `Layout`'s own check -- `false` (never blocks) if no page has
   * registered a guard. */
  isDirty: () => boolean;
}

const NavGuardContext = createContext<NavGuardContextValue | undefined>(undefined);

/**
 * A tiny escape hatch so a page's own "you have unsaved changes" dirty
 * check -- today only `GoldsetPage`'s `guarded()`/`pendingNav`, which only
 * ever intercepted *in-page* navigation (filters, the case stepper) -- can
 * also be consulted by `Layout`'s sidebar `NavLink`s, which live outside
 * the routed page and previously had no way to ask "is it safe to leave?"
 * before swapping routes out from under an unsaved draft.
 */
export function NavGuardProvider({ children }: { children: ReactNode }) {
  const checkRef = useRef<(() => boolean) | null>(null);

  const setGuard = useCallback((check: () => boolean) => {
    checkRef.current = check;
  }, []);

  const clearGuard = useCallback(() => {
    checkRef.current = null;
  }, []);

  const isDirty = useCallback(() => checkRef.current?.() ?? false, []);

  return (
    <NavGuardContext.Provider value={{ setGuard, clearGuard, isDirty }}>{children}</NavGuardContext.Provider>
  );
}

export function useNavGuard(): NavGuardContextValue {
  const ctx = useContext(NavGuardContext);
  if (!ctx) throw new Error("useNavGuard must be used within a NavGuardProvider");
  return ctx;
}
