import { verdictWord } from "../components/VerdictStamp";

/** The ink each verdict is stamped in. CEO decision, 1 Sep: violation red, no
 * violation green, escalation blue.
 *
 * ONE table, in a module of its own so both the stamp and any aggregate view
 * read the same mapping — a screen cannot invent a fourth colour or pair the
 * wrong hue with the wrong word.
 *
 * Keyed by the WORD rather than the raw verdict, so all three vocabularies
 * (TP/FP, CONFIRM_TP/CONFIRM_FP/ESCALATE, violation/no_violation/
 * cannot_determine) land on one ink without a third table.
 *
 * It replaces DESIGN.md's One Red Rule and No Triad Rule for verdicts. What
 * survives of both is the safeguard they were really protecting: these hues
 * are for VERDICTS and errors, never decoration or a hover tint, and the
 * colour is never the only carrier — every verdict is also stated in words.
 */
const VERDICT_INK: Record<string, string> = {
  VIOLATION: "var(--color-verdict-violation)",
  "NO VIOLATION": "var(--color-verdict-clear)",
  ESCALATE: "var(--color-verdict-escalate)",
};

/** The ink for a stamp WORD (`VIOLATION`, `NO VIOLATION`, `ESCALATE`). */
export function inkForWord(word: string): string {
  // An unknown verdict keeps the ink the app has always used rather than
  // falling back to a body-text colour, which would not read as a stamp.
  return VERDICT_INK[word] ?? "var(--color-verdict-violation)";
}

/** The ink for a raw verdict in any of the three vocabularies. */
export function verdictInk(verdict: string): string {
  return inkForWord(verdictWord(verdict));
}
