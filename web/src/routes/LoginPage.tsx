import { Factory, Loader2 } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export function LoginPage() {
  const { login, loading, error, isAuthenticated, sessionExpired } = useAuth();
  const [password, setPassword] = useState("");
  const location = useLocation();

  if (isAuthenticated) {
    const from = (location.state as { from?: string } | null)?.from ?? "/metrics";
    return <Navigate to={from} replace />;
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    try {
      await login(password);
    } catch {
      // error surfaced via context
    }
  }

  return (
    <div className="flex min-h-svh items-center justify-center bg-page px-4">
      <div className="w-full max-w-sm rounded-sm border border-hairline bg-surface p-8">
        <div className="mb-6 flex flex-col items-center gap-2">
          <Factory size={28} className="text-accent" strokeWidth={1.75} />
          <h1 className="text-lg font-semibold text-ink">Training and Development</h1>
          <p className="text-center text-sm text-ink-secondary">
            {sessionExpired
              ? "The server restarted, so your session ended. Sign in again. Nothing you saved was lost."
              : "Sign in with the shared reviewer password to continue."}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-ink-secondary">Password</span>
            <input
              type="password"
              autoFocus
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-md border border-black/15 bg-page px-3 py-2 text-sm text-ink outline-none focus:border-accent focus:ring-1 focus:ring-accent"
              placeholder="••••••••"
            />
          </label>

          {error && (
            <p className="rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-sm text-critical">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={loading || !password}
            className="flex items-center justify-center gap-2 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-accent-hover disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading && <Loader2 size={15} className="animate-spin" />}
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}
