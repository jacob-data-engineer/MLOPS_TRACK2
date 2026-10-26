/**
 * Shared reader-facing formatting for values the API hands back raw.
 *
 * Its own module rather than a home inside a route file: the batch panel and
 * the single-alarm run had each grown a copy of `formatConfidence`, and a
 * figure that reads two ways in two places is a figure a reviewer has to
 * decode twice.
 */

/** "0.90 (high)" -- both figures the judge reports, together.
 *
 * `score` is nullable and a null is NOT a zero: only the categorical `level`
 * is guaranteed across providers, and the backend escalates rather than
 * guessing a number. `level` is nullable on the batch row shape alone, which
 * is why it is handled here instead of at that one call site. */
export function formatConfidence(score: number | null, level: string | null): string {
  if (level === null) return score === null ? "not recorded" : score.toFixed(2);
  return score === null ? level : `${score.toFixed(2)} (${level})`;
}

/** Every stored timestamp in this system is UTC, and the raw ISO string is
 * not a thing to put in front of a reader: `2026-08-28T05:20:52.657201+00:00`
 * is six microsecond digits and an offset nobody at Jurong Port works in.
 *
 * Storage stays UTC and must: the backend orders "newest row wins" by
 * comparing timestamp STRINGS lexicographically, so a mixed-offset log would
 * silently reorder rows. The zone lives here, at the last possible moment,
 * and it is pinned to `Asia/Singapore` rather than the viewer's browser --
 * this is one port's operations log, and a reader in another zone must not be
 * shown a different "judged at" for the same row than the person beside them.
 *
 * Assembled from `formatToParts` rather than a locale format string so the
 * shape is ours and not the runtime's: sortable `YYYY-MM-DD HH:MM:SS`, the
 * same left-to-right descending order every case id and coordinate in this
 * app already reads in, with the zone named so the reading is checkable.
 * `hourCycle: "h23"` because `hour12: false` renders midnight as `24` on some
 * engines. */
const SGT_FORMAT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Singapore",
  year: "numeric",
  month: "2-digit",
  day: "2-digit",
  hour: "2-digit",
  minute: "2-digit",
  second: "2-digit",
  hourCycle: "h23",
});

function sgtParts(date: Date): Record<string, string> {
  const parts: Record<string, string> = {};
  for (const part of SGT_FORMAT.formatToParts(date)) parts[part.type] = part.value;
  return parts;
}

/** "2026-08-28 13:20:52 SGT". Takes the stored ISO string or a live `Date`.
 *
 * An unparseable value falls back to the raw string rather than to a dash or
 * a thrown error: a timestamp we cannot read is still a fact about the row,
 * and hiding it would make a malformed log entry look like an absent one. */
export function formatTimestamp(value: string | Date): string {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return typeof value === "string" ? value : "unrecorded";
  const p = sgtParts(date);
  return `${p.year}-${p.month}-${p.day} ${p.hour}:${p.minute}:${p.second} SGT`;
}

/** "16 Oct 15:34 SGT" -- a wall-clock instant, short enough to sit inside a
 * line of running text.
 *
 * Same zone and the same reasoning as `formatTimestamp`: storage is UTC, the
 * zone is applied at the last possible moment, and it is pinned to
 * `Asia/Singapore` rather than the viewer's browser.
 *
 * THE ZONE IS NAMED, always. Two of these were rendered with the browser's
 * own clock and then labelled "SGT" by hand, which is right only on a machine
 * already set to Singapore and silently wrong everywhere else. A reading that
 * carries its own zone cannot be mislabelled by the line it sits in.
 *
 * Seconds are dropped: a schedule is set to the minute and a snapshot is
 * named for the day it was taken, so the third figure is noise in both. */
const SGT_SHORT = new Intl.DateTimeFormat("en-GB", {
  timeZone: "Asia/Singapore",
  day: "2-digit",
  month: "short",
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
});

export function formatDayTime(value: string | Date): string {
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) {
    return typeof value === "string" ? value : "unrecorded";
  }
  const p: Record<string, string> = {};
  for (const part of SGT_SHORT.formatToParts(date)) p[part.type] = part.value;
  return `${p.day} ${p.month} ${p.hour}:${p.minute} SGT`;
}

/** How long ago `from` was: `8s`, `2m 14s`, `1h 6m`.
 *
 * NOT `formatTimestamp`'s job, and not a competitor to it. That renders a
 * wall-clock instant, which is the right thing for a row in a ledger and the
 * wrong thing for a progress indicator: `started 2026-09-04 13:20:52 SGT`
 * makes a reader subtract, and the answer they wanted was `for 2m 14s`. A
 * reading that changes on its own is also the proof of life the wall clock
 * cannot give -- a frozen absolute timestamp and a working one look identical.
 *
 * Seconds are zero-padded inside the minute form so the line does not shift
 * every time the count crosses ten; the leading unit never is, because a
 * padded `08s` reads as a column width rather than a measurement. Past an
 * hour the seconds are dropped: at that scale they are noise, and a
 * three-unit reading invites arithmetic again.
 *
 * A value in the future is clamped to zero rather than reported. The realistic
 * cause is skew between the server's clock and the browser's, of a second or
 * two, and a clause that blinks out of the sentence whenever the skew is the
 * wrong way is worse than a reading that starts a beat late.
 *
 * `null` only for a value that will not parse. Unlike `formatTimestamp`, which
 * falls back to the raw string because an unreadable timestamp is still a fact
 * about the row, there is nothing honest to show here: the failure is in the
 * elapsed reading itself, and a caller is expected to drop the clause. */
export function formatElapsed(from: string | Date, now: Date = new Date()): string | null {
  const seconds = elapsedSeconds(from, now);
  if (seconds === null) return null;
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ${String(seconds % 60).padStart(2, "0")}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** The same measurement as a number, for a caller that has to JUDGE a duration
 * rather than print one.
 *
 * Split out rather than re-parsed at the call site so one definition owns the
 * future-clamp and the unparseable answer: a screen deciding "is this reading
 * still plausible" and the reading it is deciding about must not disagree
 * about what the elapsed time is. */
export function elapsedSeconds(from: string | Date, now: Date = new Date()): number | null {
  const start = from instanceof Date ? from : new Date(from);
  if (Number.isNaN(start.getTime())) return null;
  return Math.max(0, Math.floor((now.getTime() - start.getTime()) / 1000));
}

/** A count with thousands separators, pinned to one locale: `75,242`.
 *
 * THE APP'S ONE EXCEPTION, and deliberately narrow. Every other figure in
 * this system renders as a bare number, and changing that wholesale is a
 * decision about every StatTile and every Rolling metrics column, not
 * something to arrive at by copying this line. The live feed's
 * keys-examined reading earns it on its own: the corpus listing is already
 * past 75,000 and climbing, and a five-figure run of digits with no grouping
 * is read wrong at a glance, which defeats the point of showing it.
 *
 * The locale is pinned for the same reason `formatTimestamp`'s zone is: two
 * people reading one operations panel side by side must not see `75,242` and
 * `75.242`, and the browser's own locale is not a fact about this port. */
const COUNT_FORMAT = new Intl.NumberFormat("en-SG");

export function formatCount(value: number): string {
  return Number.isFinite(value) ? COUNT_FORMAT.format(value) : String(value);
}

/** Just the clock, same zone -- for "as of" on something loaded moments ago,
 * where the date is today by construction and spelling it out is noise. */
export function formatClockTime(value: Date): string {
  const p = sgtParts(value);
  return `${p.hour}:${p.minute}:${p.second} SGT`;
}

/** ### THE "unk" RULE. NEVER PRINT `unk` TO SCREEN.
 *
 * The helmet frame extractor writes `site` and `camera` as the literal string
 * `"unk"` for this entire corpus, because the clip keys carry neither. `unk`
 * on screen reads as a corrupted record; the truth is plainer than that, so
 * it gets said in words.
 *
 * Lives here beside `formatTimestamp` rather than inside the one screen that
 * shows it today, so a second screen meeting the same field cannot render it
 * a second way. The parameters exist for the day a source DOES carry a site
 * or a camera -- today every caller passes nothing, and that is the answer.
 */
const UNRECORDED_FIELDS = new Set(["", "unk", "unknown", "none", "null"]);

export const LOCATION_HELP = "Clip keys carry no camera or site.";

export function formatLocation(site?: string | null, camera?: string | null): string {
  const known = [site, camera]
    .map((v) => (typeof v === "string" ? v.trim() : ""))
    .filter((v) => !UNRECORDED_FIELDS.has(v.toLowerCase()));
  return known.length === 0 ? "Location not recorded" : known.join(" / ");
}

/** Milliseconds into a clip, as `M:SS.mmm` -- where in the clip a frame sits.
 *
 * Not `formatTimestamp`'s job: that renders a wall-clock instant in SGT, and
 * this is an elapsed measurement inside one clip, which has no zone and must
 * not grow one. Zero-padded so a column of them aligns in the `.tabular`
 * treatment every other measurement in this app reads in. */
export function formatClipOffset(ms: number): string {
  if (!Number.isFinite(ms)) return "unrecorded";
  const total = Math.max(0, Math.round(ms));
  const minutes = Math.floor(total / 60000);
  const seconds = Math.floor((total % 60000) / 1000);
  const millis = total % 1000;
  return `${minutes}:${String(seconds).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}
