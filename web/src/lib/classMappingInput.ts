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

/** Collect explicit renames before resolving IDs so every source shares one target list. */
export function editedTargetClasses(
  edits: Record<string, Record<string, string>>,
  mappings: Record<string, { ids: Record<string, string | null> }>,
  current: Record<string, string>,
): Record<string, string> {
  const next = {...current};
  const renamed: Record<string, string> = {};
  for (const [source, fields] of Object.entries(edits)) {
    if (!mappings[source]) continue;
    for (const [sourceId, raw] of Object.entries(fields)) {
      const value = raw.trim();
      if (!value || Object.hasOwn(current, value) || Object.values(current).includes(value)) continue;
      const pair = value.match(/^(\d+)\s*[:=]?\s+(.+)$/);
      const id = pair ? pair[1] : mappings[source].ids[sourceId];
      const name = pair ? pair[2].trim() : value;
      if (id == null || !Object.hasOwn(current, id) || /^\d+$/.test(name))
        throw new Error(`Enter an existing target ID with its new name, for example: 0 helmets.`);
      if (name === current[id]) continue;
      if (renamed[id] && renamed[id] !== name)
        throw new Error(`Class ${id} has conflicting names: ${renamed[id]} and ${name}. Use one shared name.`);
      renamed[id] = name;
      next[id] = name;
    }
  }
  if (new Set(Object.values(next)).size !== Object.keys(next).length)
    throw new Error('Each target class must have a unique name.');
  return next;
}
