import { HelpCircle, Send, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { ApiError, askTrack2Assistant, getTrack2AssistantStatus, type Track2AssistantStatus } from "../lib/api";
import { ASK_CARD_EVENT, readCard, readPage, type AskCardDetail } from "../lib/helpContext";

type Turn = { role: "user" | "assistant"; content: string; sources?: { title: string; uploaded: boolean }[]; error?: boolean };

/** THE HELP GUIDE (2026-11-20): a chat on every page that explains the
 * application -- what a card does, what its fields mean, what a button does
 * and what to do next -- from documentation generated from the code.
 *
 * Read-only: it sends text and shows text. Nothing it answers is run.
 *
 * Context is read when a question is SENT, not when the panel opens: the
 * page you are on, the cards in view, and -- after "Ask about this card" --
 * that card's fields as they are at that moment. Moving to another page
 * clears the asked-about card, so an answer never describes a card that is
 * no longer on screen. */
export function HelpChat() {
  const location = useLocation();
  const [open, setOpen] = useState(false);
  const [status, setStatus] = useState<Track2AssistantStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  // The card asked about: its title, and the element to read it from.
  const [card, setCard] = useState<AskCardDetail | null>(null);
  const [movedNote, setMovedNote] = useState("");
  const listRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    if (!open || status) return;
    getTrack2AssistantStatus().then(setStatus)
      .catch((err) => setStatusError(err instanceof ApiError ? err.detail : "Could not reach the help assistant."));
  }, [open, status]);

  // "Ask about this card" on any card.
  useEffect(() => {
    const onAsk = (e: Event) => {
      const detail = (e as CustomEvent<AskCardDetail>).detail;
      setCard(detail);
      setMovedNote("");
      setOpen(true);
      if (!draft.trim()) setDraft(`What does the "${detail.title}" card do, and what should I do next?`);
      window.setTimeout(() => inputRef.current?.focus(), 50);
    };
    window.addEventListener(ASK_CARD_EVENT, onAsk);
    return () => window.removeEventListener(ASK_CARD_EVENT, onAsk);
  }, [draft]);

  // A new page: the asked-about card belonged to the old one.
  const lastPath = useRef(location.pathname);
  useEffect(() => {
    if (lastPath.current === location.pathname) return;
    lastPath.current = location.pathname;
    if (card) setMovedNote(`You moved to another page, so "${card.title}" is no longer the card being asked about.`);
    setCard(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  useEffect(() => { listRef.current?.scrollTo({ top: listRef.current.scrollHeight }); }, [turns, busy]);

  async function send() {
    const question = draft.trim();
    if (!question || busy) return;
    const context = { ...readPage(location.pathname),
      ...(card ? { card: readCard(card.title, card.element) } : {}) };
    const history = [...turns.filter((t) => !t.error), { role: "user" as const, content: question }];
    setTurns((t) => [...t, { role: "user", content: question }]);
    setDraft("");
    setBusy(true);
    try {
      const res = await askTrack2Assistant(history.map(({ role, content }) => ({ role, content })), context);
      setTurns((t) => [...t, { role: "assistant", content: res.answer + (res.truncated ? "\n\n(The answer was cut short.)" : ""),
                                sources: res.sources }]);
    } catch (err) {
      setTurns((t) => [...t, { role: "assistant", error: true,
        content: err instanceof ApiError ? err.detail : "The help assistant did not answer. Try again." }]);
    } finally {
      setBusy(false);
    }
  }

  const page = typeof document !== "undefined" ? readPage(location.pathname).page : "";

  return (
    <>
      {!open && (
        <button type="button" onClick={() => setOpen(true)} aria-label="Open the help guide"
                className="fixed bottom-4 right-4 z-50 flex items-center gap-1.5 rounded-full bg-accent px-4 py-2.5 text-sm font-semibold text-white shadow-lg">
          <HelpCircle size={16} /> Help
        </button>
      )}
      {open && (
        <div role="dialog" aria-label="Help guide"
             className="fixed bottom-4 right-4 z-50 flex max-h-[80vh] w-[min(26rem,calc(100vw-2rem))] flex-col rounded-md border border-hairline bg-surface shadow-2xl">
          <div className="flex items-start justify-between gap-2 border-b border-hairline px-3 py-2">
            <div className="min-w-0">
              <p className="text-sm font-semibold text-ink">Help guide</p>
              <p className="truncate text-[11px] text-ink-muted">
                On: {page}{card ? <> &middot; asking about <strong className="text-ink">{card.title}</strong></> : null}
              </p>
            </div>
            <div className="flex shrink-0 items-center gap-1">
              {card && <button type="button" className="rounded border border-hairline px-1.5 py-0.5 text-[11px]" onClick={() => setCard(null)}>Whole page</button>}
              {turns.length > 0 && <button type="button" className="rounded border border-hairline px-1.5 py-0.5 text-[11px]" onClick={() => setTurns([])}>New</button>}
              <button type="button" aria-label="Close the help guide" onClick={() => setOpen(false)} className="rounded p-1 text-ink-secondary"><X size={15} /></button>
            </div>
          </div>

          <div ref={listRef} className="flex-1 space-y-2 overflow-y-auto px-3 py-2 text-sm">
            {statusError && <p className="text-xs text-critical" role="alert">{statusError}</p>}
            {status && !status.enabled && (
              <p className="rounded border border-hairline bg-page p-2 text-xs text-ink-secondary" role="status">
                The help assistant is installed but not switched on yet: it is waiting for approval to use
                Amazon Bedrock. Questions will be answered once it is enabled.
              </p>
            )}
            {movedNote && <p className="text-xs text-ink-muted" role="status">{movedNote}</p>}
            {turns.length === 0 && (
              <p className="text-xs text-ink-muted">
                Ask what a card does, what a field means, or what to do next. Use <strong>Ask about this card</strong> on
                any card to ask about that card with its current settings. This guide explains the application; it
                cannot change, start or delete anything.
              </p>
            )}
            {turns.map((t, i) => (
              <div key={i} className={t.role === "user" ? "ml-6 rounded bg-accent-subtle px-2 py-1.5" : "mr-2"}>
                <p className={`whitespace-pre-wrap ${t.error ? "text-critical" : "text-ink"}`}>{t.content}</p>
                {t.sources && t.sources.length > 0 && (
                  <p className="mt-1 text-[11px] text-ink-muted">
                    From: {t.sources.map((s) => s.title + (s.uploaded ? " (uploaded)" : "")).join("; ")}
                  </p>
                )}
              </div>
            ))}
            {busy && <p className="text-xs text-ink-muted" role="status">Looking it up...</p>}
          </div>

          <form className="flex items-end gap-2 border-t border-hairline p-2"
                onSubmit={(e) => { e.preventDefault(); void send(); }}>
            <textarea ref={inputRef} value={draft} rows={2} maxLength={2000}
                      onChange={(e) => setDraft(e.target.value)}
                      onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void send(); } }}
                      placeholder="Ask about this page..." aria-label="Question for the help guide"
                      className="flex-1 resize-none rounded border border-black/15 bg-page px-2 py-1.5 text-sm" />
            <button type="submit" disabled={busy || !draft.trim()} aria-label="Send"
                    className="rounded bg-accent p-2 text-white disabled:opacity-40"><Send size={15} /></button>
          </form>
        </div>
      )}
    </>
  );
}
