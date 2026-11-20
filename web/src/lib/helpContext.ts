/** What the help assistant is told about where the user is (2026-11-20).
 *
 * Read from the page AT THE MOMENT a question is sent -- never cached -- so
 * navigating, collapsing a card or changing a field cannot leave the
 * assistant describing a screen that is no longer there.
 *
 * Only what is on screen and needed to explain it: the page, the cards in
 * view, and for an asked-about card its field labels and values plus any
 * status line it shows. Password and file inputs are skipped, and a field
 * whose label says secret/token/key is dropped here as well as on the
 * server. */

export type HelpField = { label: string; value: string };
export type HelpCard = { title: string; fields: HelpField[]; status: string[] };
export type HelpContext = { path: string; page: string; visible_cards: string[]; card?: HelpCard };

/** Raised by a card's "Ask about this card" button; the chat listens. */
export const ASK_CARD_EVENT = "track2:ask-card";
export type AskCardDetail = { title: string; element: HTMLElement | null };

const SECRET_LABEL = /pass(word)?|secret|token|api[ _-]?key|access[ _-]?key|credential|private[ _-]?key|session/i;
const MAX_FIELDS = 40;
const MAX_VALUE = 200;

function text(el: Element | null | undefined): string {
  return (el?.textContent ?? "").replace(/\s+/g, " ").trim();
}

function labelOf(el: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement): string {
  const aria = el.getAttribute("aria-label");
  if (aria) return aria.trim();
  if (el.id) {
    const forLabel = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
    if (forLabel) return text(forLabel);
  }
  const wrapping = el.closest("label");
  if (wrapping) {
    // The label's own words, without the option text of a select inside it.
    const clone = wrapping.cloneNode(true) as HTMLElement;
    clone.querySelectorAll("select, option, input, textarea").forEach((n) => n.remove());
    const own = text(clone);
    if (own) return own.slice(0, 120);
  }
  return el.getAttribute("placeholder") || el.getAttribute("name") || "";
}

function valueOf(el: HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement): string {
  if (el instanceof HTMLInputElement && (el.type === "checkbox" || el.type === "radio")) {
    return el.checked ? "ticked" : "not ticked";
  }
  if (el instanceof HTMLSelectElement) {
    return el.selectedOptions[0]?.textContent?.trim() ?? el.value;
  }
  return el.value;
}

/** The fields and status lines of one card, as they are right now. */
export function readCard(title: string, element: HTMLElement | null): HelpCard {
  const fields: HelpField[] = [];
  if (element && element.isConnected) {
    const inputs = element.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>(
      "input, select, textarea");
    for (const el of Array.from(inputs)) {
      if (fields.length >= MAX_FIELDS) break;
      if (el instanceof HTMLInputElement && ["password", "hidden", "file"].includes(el.type)) continue;
      const label = labelOf(el);
      if (!label || SECRET_LABEL.test(label)) continue;
      if (el instanceof HTMLInputElement && el.type === "radio" && !el.checked) continue;
      fields.push({ label, value: valueOf(el).slice(0, MAX_VALUE) });
    }
  }
  const status = element && element.isConnected
    ? Array.from(element.querySelectorAll('[role="status"], [role="alert"]'))
        .map((n) => text(n).slice(0, 300)).filter(Boolean).slice(0, 10)
    : [];
  return { title, fields, status };
}

/** The page and the cards currently in view. */
export function readPage(path: string): Omit<HelpContext, "card"> {
  const main = document.querySelector("main");
  const page = text(main?.querySelector("h1")) || text(main?.querySelector("h2")) || path;
  const visible: string[] = [];
  main?.querySelectorAll<HTMLElement>("section[data-card-title]").forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.bottom > 0 && r.top < window.innerHeight) visible.push(el.dataset.cardTitle || "");
  });
  return { path, page, visible_cards: visible.filter(Boolean).slice(0, 40) };
}
