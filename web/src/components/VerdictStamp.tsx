import { useId } from "react";
import { Badge } from "./Badge";
import { inkForWord } from "../lib/verdictColors";

/**
 * The one place a verdict is rendered anywhere in this app: TP / FP /
 * uncertain, and Track1's CONFIRM_TP / CONFIRM_FP / ESCALATE (the same
 * triad in Track1's own vocabulary -- see OutcomeBarChart's comment). A
 * real stamped word in the single alarm-red ink this system spends its
 * one saturated color on, never a color-coded pill -- a customs officer's
 * NO VIOLATION stamp and VIOLATION stamp are the same red ink; only the
 * word differs, and this component holds to that.
 *
 * `landed`: play the stamp-landing motion (index.css's `stamp-land`
 * keyframes) once, for the moment a verdict is *confirmed* right now --
 * a fresh Track1 run, a saved few-shot verdict. Historical rows in a
 * ledger table (already-judged alarms, already-saved examples) render
 * `landed={false}`: at rest, no replay on every re-render.
 *
 * Texture: this used to be a clean CSS box (a solid border + rotation).
 * It now reads as an actually-inked rubber stamp: the border/fill layer
 * runs through an SVG feTurbulence+feDisplacementMap filter so the
 * rectangle's edge wobbles like worn rubber rather than a crisp
 * rounded-sm corner, and the word itself carries two low-opacity offset
 * ghost copies underneath the real text -- a typewriter-era double-strike
 * read as slightly uneven ink density. The one real accessible name and
 * the one full-opacity, undistorted glyph layer are unchanged from
 * before, so this is a texture pass, not a legibility or contrast
 * regression -- see DESIGN.md's VerdictStamp entry for the numbers.
 */
/** The words the CEO asked for, and the only place they are defined: what
 * read HOLD now reads VIOLATION, and what read CLEARED now reads NO
 * VIOLATION, because a port safety officer meeting this stamp should not have
 * to learn customs vocabulary to know whether a violation was found.
 * ESCALATE stays as it is -- it already says what it means.
 *
 * The API's own keys did NOT change: an audit or eval distribution is still
 * keyed `hold` / `cleared` / `escalate` server-side (see api.ts's
 * `VerdictDistribution`). This is a display mapping, which is the whole point
 * of it living in one component. */
const VERDICT_WORD: Record<string, string> = {
  TP: "VIOLATION",
  CONFIRM_TP: "VIOLATION",
  FP: "NO VIOLATION",
  CONFIRM_FP: "NO VIOLATION",
  uncertain: "ESCALATE",
  ESCALATE: "ESCALATE",
  // The REVIEW vocabulary too -- the same triad a third time, the words a
  // human answers in. Without these a caller handing over a settled verdict
  // got the raw key stamped on the sheet: the live feed's frame column read
  // "cannot_determine", which looks like the screen withholding an answer
  // rather than the judge declining to give one.
  violation: "VIOLATION",
  no_violation: "NO VIOLATION",
  cannot_determine: "ESCALATE",
};

/** Above this many characters a mark takes the tighter of two letter-spacings.
 * A rubber stamp is one physical die: the two-word NO VIOLATION cannot set its
 * letters as loosely as VIOLATION and still read as the same object pressed
 * onto the same sheet, and at 0.14em it also widens the ledger column the
 * stamp sits in on five screens. Two steps only, keyed on the word -- not a
 * continuous scale, which would give every verdict its own spacing and no
 * verdict a recognizable shape. */
const LONG_WORD_CHARS = 9;

export function verdictWord(verdict: string): string {
  return VERDICT_WORD[verdict] ?? verdict.toUpperCase();
}

export function VerdictStamp({
  verdict,
  landed = false,
  size = "md",
}: {
  verdict: string;
  landed?: boolean;
  size?: "sm" | "md";
}) {
  const word = verdictWord(verdict);
  const dims = size === "sm" ? "px-2 py-0.5 text-xs" : "px-3 py-1 text-sm";
  // `whitespace-nowrap` is load-bearing now that a mark can be two words: a
  // wrapped stamp inside a rotated box reads as a rendering fault rather than
  // as a stamp, and every one of these sits in a table cell that would
  // happily break it.
  const tracking = word.length > LONG_WORD_CHARS ? "tracking-[0.09em]" : "tracking-[0.14em]";
  const filterId = `stamp-roughen-${useId()}`;
  const ink = inkForWord(word);

  return (
    <span
      className={`relative inline-flex items-center whitespace-nowrap font-semibold ${tracking} ${dims} ${
        landed ? "stamp-landing" : "-rotate-[7deg]"
      }`}
      style={{ fontFamily: "var(--font-sans)", color: ink }}
      role="status"
      aria-label={`Verdict: ${word} (${verdict})`}
      title={verdict !== word ? `Backend verdict: ${verdict}` : undefined}
    >
      {/* Filter def only -- zero footprint, never painted directly. A
          fixed seed so every stamp in the app shares one worn shape, the
          same way a real rubber stamp is one physical object reused
          case after case, not a fresh random edge each press. */}
      <svg aria-hidden="true" className="absolute h-0 w-0" focusable="false">
        <filter id={filterId} x="-30%" y="-30%" width="160%" height="160%">
          <feTurbulence
            type="fractalNoise"
            baseFrequency="0.045 0.09"
            numOctaves={2}
            seed={3}
            result="noise"
          />
          <feDisplacementMap
            in="SourceGraphic"
            in2="noise"
            scale={3.2}
            xChannelSelector="R"
            yChannelSelector="G"
          />
        </filter>
      </svg>

      {/* Rough-edged ink layer: the border + fill, distorted by the
          filter above so the rectangle reads as worn rubber, not a
          crisp CSS box. */}
      <span
        aria-hidden="true"
        className="absolute inset-0 rounded-sm border-2"
        style={{
          filter: `url(#${filterId})`,
          borderColor: ink,
          background: `color-mix(in srgb, ${ink} 6%, transparent)`,
        }}
      />

      {/* Ghost strikes: two faint, offset copies of the word underneath
          the real text, simulating uneven ink density from a hand-pressed
          stamp rather than a second inking pass on the real glyph. */}
      <span
        aria-hidden="true"
        className="absolute inset-0 flex items-center justify-center"
        style={{ transform: "translate(0.7px, -0.5px)", opacity: 0.3 }}
      >
        {word}
      </span>
      <span
        aria-hidden="true"
        className="absolute inset-0 flex items-center justify-center"
        style={{ transform: "translate(-0.6px, 0.6px)", opacity: 0.24 }}
      >
        {word}
      </span>

      {/* The one real, undistorted, full-opacity glyph layer -- legibility
          and contrast are unchanged from the pre-texture version.
          aria-hidden because the aria-label above already carries the
          full accessible name; this avoids any risk of a duplicate
          announcement. */}
      <span aria-hidden="true" className="relative">
        {word}
      </span>
    </span>
  );
}

/**
 * A HUMAN'S ANSWER, IN THE REVIEW VOCABULARY -- `violation` /
 * `no_violation` / `cannot_determine`, the words a reviewer answers in and
 * the same set `AuditSummary.corrected_verdicts` uses.
 *
 * Beside `VerdictStamp` rather than inside a screen because three surfaces
 * now render one of these: the escalation sheet's "already recorded" line,
 * the crop strip's per-crop answer, and (in its own local form, kept for its
 * own reasons) Gold Set. Two of those sit on ONE screen, and a mapping that
 * drifted between them would show one answer as two different objects.
 *
 * `cannot_determine` is deliberately NOT a stamp. Per DESIGN.md a verdict is
 * always a `VerdictStamp` and a lifecycle state is always a `Badge`, and "the
 * evidence could not settle this" decides the case without asserting anything
 * about the worker -- there is no stamped word for it and it yields no label.
 */
export function RecordedVerdict({
  verdict,
  landed = false,
  size = "md",
}: {
  verdict: string;
  landed?: boolean;
  size?: "sm" | "md";
}) {
  if (verdict === "violation") return <VerdictStamp verdict="TP" landed={landed} size={size} />;
  if (verdict === "no_violation") return <VerdictStamp verdict="FP" landed={landed} size={size} />;
  if (verdict === "cannot_determine") return <Badge tone="neutral" label="Cannot determine" />;
  return <Badge tone="pending" label="Not yet decided" />;
}
