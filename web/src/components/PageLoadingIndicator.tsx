import { useEffect, useState, useSyncExternalStore } from "react";
import { subscribeLoading, loadingCount } from "../lib/loadingActivity";

export function PageLoadingIndicator() {
  const count = useSyncExternalStore(subscribeLoading, loadingCount, () => 0);
  const [visible, setVisible] = useState(false);
  const busy = count > 0;
  useEffect(() => {
    if (!busy) { setVisible(false); return; }
    const timer = window.setTimeout(() => setVisible(true), 300);
    return () => window.clearTimeout(timer);
  }, [busy]);
  if (!busy || !visible) return null;
  return <div role="status" aria-live="polite" className="fixed right-4 top-4 z-[100] flex items-center gap-2 rounded-full border border-hairline bg-page px-3 py-2 text-xs text-ink shadow-sm" title={`${count} data request${count === 1 ? "" : "s"} pending`}>
    <span aria-hidden="true" className="h-4 w-4 animate-spin rounded-full border-2 border-green-600/25 border-t-green-600 motion-reduce:animate-none" />
    <span>Loading data…</span>
  </div>;
}
