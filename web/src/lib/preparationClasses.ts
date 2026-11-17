/** Model classes from the loaded preparation recipe, in dataset index order.
 * Model IDs are contiguous; training maps dataset IDs by class name. */
export function preparationClasses(config: Record<string, unknown> | undefined): Record<string, string> {
  const assemble = config?.dataset_assemble as { label_classes?: unknown } | undefined;
  const raw = assemble?.label_classes;
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return {};
  const entries = Object.entries(raw);
  if (entries.some(([id, name]) => !/^\d+$/.test(id) || typeof name !== "string" || !name.trim())) return {};
  entries.sort(([a], [b]) => Number(a) - Number(b));
  const names = entries.map(([, name]) => (name as string).trim());
  if (new Set(names).size !== names.length) return {};
  return Object.fromEntries(names.map((name, index) => [String(index), name]));
}
