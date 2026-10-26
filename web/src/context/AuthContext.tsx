import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import {
  ApiError,
  login as loginRequest,
  setAuthToken,
  setUnauthorizedHandler,
} from "../lib/api";

const STORAGE_KEY = "ppe.session_token";

interface AuthContextValue {
  token: string | null;
  isAuthenticated: boolean;
  login: (password: string) => Promise<void>;
  logout: () => void;
  loading: boolean;
  error: string | null;
  /** The last sign-out was the SERVER dropping us, not a deliberate logout --
   * which is what every deploy does, since sessions live in server memory. */
  sessionExpired: boolean;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

/** Storage can throw outright in a private window or with site data
 * blocked, which would take the whole app down on first render. A
 * missing token only means signing in again. */
function readStoredToken(): string | null {
  try {
    // Migrates anyone still holding a token from when this was per-tab,
    // so the change does not itself sign everyone out once.
    const carriedOver = sessionStorage.getItem(STORAGE_KEY);
    if (carriedOver && !localStorage.getItem(STORAGE_KEY)) {
      localStorage.setItem(STORAGE_KEY, carriedOver);
      sessionStorage.removeItem(STORAGE_KEY);
    }
    return localStorage.getItem(STORAGE_KEY);
  } catch {
    return null;
  }
}

function writeStoredToken(value: string | null): void {
  try {
    if (value === null) localStorage.removeItem(STORAGE_KEY);
    else localStorage.setItem(STORAGE_KEY, value);
  } catch {
    // Unavailable storage: the session still works for this page, it
    // just will not survive a reload.
  }
}

export function AuthProvider({ children }: { children: ReactNode }) {
  // localStorage, not sessionStorage: sessionStorage is per TAB, so
  // opening the app in a second tab landed on the sign-in screen, and
  // closing a tab threw the session away. The token is shared across the
  // browser and survives a tab being reopened.
  const [token, setToken] = useState<string | null>(() => {
    const stored = readStoredToken();
    if (stored) setAuthToken(stored);
    return stored;
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  //: Whether the last sign-out was the server dropping us, so the login
  //: screen can say why rather than looking like a plain logout.
  const [expired, setExpired] = useState(false);

  const login = useCallback(async (password: string) => {
    setLoading(true);
    setError(null);
    try {
      const { token: newToken } = await loginRequest(password);
      setExpired(false);
      writeStoredToken(newToken);
      setAuthToken(newToken);
      setToken(newToken);
    } catch (err) {
      if (err instanceof ApiError) {
        if (err.status === 429) {
          setError("Too many incorrect attempts. Try again later.");
        } else if (err.status === 503) {
          setError("Server isn't configured with a password yet.");
        } else {
          setError("Incorrect password.");
        }
      } else {
        setError("Couldn't reach the API. Is it running?");
      }
      throw err;
    } finally {
      setLoading(false);
    }
  }, []);

  const logout = useCallback(() => {
    writeStoredToken(null);
    setAuthToken(null);
    setToken(null);
  }, []);

  // A dead session drops to sign-in instead of painting "Invalid or expired
  // session token." across every panel. Sessions are held in the server's
  // memory, so this happens on every deploy, which makes it the most common
  // way a user meets the app in a broken-looking state.
  useEffect(() => {
    setUnauthorizedHandler(() => {
      setExpired(true);
      logout();
    });
    return () => setUnauthorizedHandler(null);
  }, [logout]);

  // Now that the token is shared across tabs, the other tabs have to
  // follow it. Without this, signing out in one tab leaves the others
  // holding a token that has been removed from storage -- working until
  // their next reload, then silently signed out.
  useEffect(() => {
    function onStorage(event: StorageEvent) {
      if (event.key !== STORAGE_KEY && event.key !== null) return;
      const current = readStoredToken();
      setAuthToken(current);
      setToken(current);
      if (current) setExpired(false);
    }
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      token,
      isAuthenticated: token !== null,
      login,
      logout,
      loading,
      error,
      sessionExpired: expired,
    }),
    [token, login, logout, loading, error, expired],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
