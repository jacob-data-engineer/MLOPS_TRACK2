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

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => {
    const stored = sessionStorage.getItem(STORAGE_KEY);
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
      sessionStorage.setItem(STORAGE_KEY, newToken);
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
    sessionStorage.removeItem(STORAGE_KEY);
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
