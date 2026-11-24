export interface CompletionItem { id: string; state: string; label: string }
export interface CompletionSnapshot { initialized: boolean; states: Record<string, string> }
export function detectCompletions(previous: CompletionSnapshot, items: CompletionItem[], terminal: string[]) {
  const events = previous.initialized ? items.filter(item => terminal.includes(item.state)
    && previous.states[item.id] !== item.state) : [];
  return {events, snapshot: {initialized: true, states: Object.fromEntries(items.map(item => [item.id, item.state]))}};
}

export function s3ConsoleUrl(uri: string): string | undefined {
  const match = /^s3:\/\/([a-z0-9.-]+)\/(.*)$/.exec(uri);
  if (!match) return undefined;
  return `https://s3.console.aws.amazon.com/s3/buckets/${match[1]}?prefix=${encodeURIComponent(match[2])}&showversions=false`;
}
