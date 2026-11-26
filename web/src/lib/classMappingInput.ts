/** Resolve a typed target without silently inventing a class or ignoring a mismatched name. */
export function resolveClassTarget(text: string, classes: Record<string, string>): string | null {
  const value = text.trim();
  if (!value) return null;
  if (Object.hasOwn(classes, value)) return value;
  const matches = Object.entries(classes).filter(([, name]) => name === value);
  if (matches.length === 1) return matches[0][0];
  const pair = value.match(/^(\d+)\s*[:=]?\s+(.+)$/);
  if (pair && classes[pair[1]] === pair[2].trim()) return pair[1];
  throw new Error(`Unknown or mismatched target "${value}". Enter a configured class ID, name, or ID and name.`);
}
