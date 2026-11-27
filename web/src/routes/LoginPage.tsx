import { Loader2 } from "lucide-react";
import { useState, type FormEvent } from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

import jurongPortLogo from "../assets/jurong-port-logo.png";

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
    <div className="flex min-h-svh items-center justify-center bg-[#f1fbfa] px-4 py-10">
      <div className="w-full max-w-sm rounded-2xl border border-[#cce8e6] bg-white p-8 shadow-lg shadow-[#178d87]/10">
        <div className="mb-6 flex flex-col items-center gap-2">
          <div className="mb-4 bg-white px-6 py-4">
            <img src={jurongPortLogo} alt="Jurong Port" width={128} height={54} className="h-auto w-48 max-w-full object-contain" />
          </div>
          <h1 className="text-lg font-semibold text-slate-900">Training and Development</h1>
          <p className="text-center text-sm text-slate-600">
            {sessionExpired
              ? "The server restarted, so your session ended. Sign in again. Nothing you saved was lost."
              : "Sign in with the shared reviewer password to continue."}
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <label className="flex flex-col gap-1.5">
            <span className="text-sm font-medium text-slate-600">Password</span>
            <input
              type="password"
              autoFocus
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="rounded-md border border-[#cce8e6] bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-[#1dbbb5] focus:ring-2 focus:ring-[#1dbbb5]/20"
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
            className="flex items-center justify-center gap-2 rounded-md bg-[#087f7a] px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-[#06645f] disabled:cursor-not-allowed disabled:opacity-50"
          >
            {loading && <Loader2 size={15} className="animate-spin" />}
            Sign in
          </button>
        </form>
      </div>
    </div>
  );
}
