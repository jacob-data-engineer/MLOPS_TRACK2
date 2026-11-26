let pending = 0;
const listeners = new Set<() => void>();
export const loadingCount = () => pending;
export function subscribeLoading(listener: () => void) {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}
export function beginLoading() {
  pending += 1;
  listeners.forEach(listener => listener());
  let finished = false;
  return () => {
    if (finished) return;
    finished = true;
    pending -= 1;
    listeners.forEach(listener => listener());
  };
}

// Repeated reads refresh cards silently once their initial request has started.
// Scope these keys to a page visit so navigating back loads visibly again.
let page = "";
const initialReads = new Set<string>();
export function beginPageRequest(path: string, method = "GET", pageKey = window.location.pathname) {
  if (pageKey !== page) {
    page = pageKey;
    initialReads.clear();
  }
  if (method.toUpperCase() === "GET") {
    if (initialReads.has(path)) return () => {};
    initialReads.add(path);
  }
  return beginLoading();
}
