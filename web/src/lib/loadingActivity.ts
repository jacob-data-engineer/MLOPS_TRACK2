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
