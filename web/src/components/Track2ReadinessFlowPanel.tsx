import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  getTrack2SqlReadiness,
  type Track2Readiness,
  type Track2ReadinessRole,
} from "../lib/api";
import { formatDayTime } from "../lib/format";
import { Card } from "./Card";

/** THE READINESS FLOW (2026-11-05).
 *
 * Every frame the review database has judged, followed until it is ready
 * for retraining or stops -- and, for the ones that stopped somewhere a
 * reviewer can fix, what to do and where. Read live from the database on
 * the tab's own query function (`sql_explorer.readiness`), with the same
 * human-only rule as the class drop-down above it: "Fully resolved" here
 * is exactly the frames that drop-down lists.
 *
 * Saved: the class shown (`readiness_flow.hazard`) and whether the card
 * is open. Not saved: the numbers -- a result, re-read on every visit, so
 * nobody acts on a count the database has moved past.
 */

const ROLE_COLOUR: Record<Track2ReadinessRole, string> = {
  flow: "var(--color-baseline)",
  human: "var(--color-info)",
  fix: "var(--color-status-prototype)",
  ready: "var(--color-status-ready)",
  out: "var(--color-hairline)",
};
const ROLE_LABEL: Record<Track2ReadinessRole, string> = {
  flow: "Pipeline step",
  human: "Human decision recorded",
  fix: "Reviewers can recover these",
  ready: "Ready for retraining",
  out: "Out, no reviewer action",
};

/** What each step means, what put a frame there, and what a reviewer can
 * do about it. In the reviewer's words; the record names are there for
 * whoever has to check the query. */
const ABOUT: Record<string, { what: string; records?: string; action?: string; where?: string }> = {
  "Frames received": { what: "One alarm frame from CAD-IT, judged by the app.",
    records: "audit_rows, one per person per judge run." },
  "Inference found people": { what: "The detector found at least one person and the VLM judged each person crop." },
  "No person / skipped": { what: "The detector found no person, or the judge skipped the frame. A frame a person audited anyway carries on to that step.",
    records: "audit_rows outcome SKIP, or reason no_person_detected / no_person_present." },
  "Escalated by VLM": { what: "The VLM was unsure about at least one person, so the frame went to the Escalations queue.",
    records: "audit_rows outcome ESCALATE or AUDIT_REQUIRED." },
  "Settled by VLM": { what: "The VLM decided every person itself. No human was asked.",
    records: "audit_rows outcome CONFIRM_TP / CONFIRM_FP." },
  "Answered in Crop Similarity": { what: "Reviewers answered people in this frame on the Crop Similarity page.",
    records: "corrections.similarity_human_crop_ids, or a review with review_origin human_crop_pool." },
  "Audited (independent)": { what: "A person audited the frame on the Audit page, without a similar-frame suggestion.",
    records: "corrections row, decision_origin human or absent." },
  "Audited (with suggestion)": { what: "A person audited the frame with a similar-frame suggestion on screen.",
    records: "corrections row, decision_origin human_assisted." },
  "Reviewed on Escalations": { what: "A person answered the escalation on the Escalations page.",
    records: "reviews row with a verdict or crop verdicts (not pre-filled, not auto-settled)." },
  "Box fixed, verdict still VLM": { what: "A person fixed the helmet box but gave no verdict, so the verdict is still the VLM's.",
    records: "reviews row with a helmet box and no verdict.",
    action: "Give the frame a human verdict.", where: "Audit page / Crop Similarity" },
  "Pending escalation": { what: "Escalated by the VLM and still waiting for a person.",
    records: "No reviews or corrections row yet.",
    action: "Answer the escalation.", where: "Escalations page" },
  "Not audited (VLM only)": { what: "Settled by the VLM and never checked by a person. Left out of retraining until the VLM reaches 85% shadow agreement.",
    records: "audit_rows only.",
    action: "Audit the frame, or answer its people.", where: "Audit page / Crop Similarity" },
  "Fully resolved": { what: "Every kept person has a human violation / no violation answer. These are the frames the class drop-down above lists." },
  "Violation, but who? (multi-person)": { what: "A person called the frame a violation, but it has several people and nobody was marked as the one in violation, so the label cannot be put on a box.",
    action: "Answer each person, so the violation lands on the right worker.", where: "Audit crop strip / Crop Similarity" },
  "Person unanswered": { what: "At least one kept person has no human answer yet.",
    action: "Answer the remaining people.", where: "Crop Similarity / Audit crop strip" },
  "Cannot determine": { what: "A person was marked cannot determine. An unlabelled worker would be trained as background, so the frame stays out.",
    action: "Take a second look. Strike out anything that is not a person; if it is still unclear, the frame stays out.", where: "Audit page / Crop Similarity" },
  "All people struck out": { what: "Every detected box was struck out as not a person, so nothing is left to label." },
  "Box: human": { what: "Ready for retraining and test-grade: the helmet box was drawn or accepted by a person." },
  "Box: detector >= 0.5": { what: "Ready for retraining (training only): every helmet-wearer has a detector helmet box at confidence 0.5 or more. Accept or draw the box to make it test-grade too." },
  "Box missing": { what: "Every verdict is in, but a helmet-wearer has no human box and no confident detector box.",
    action: "Draw or accept the helmet box.", where: "Escalations box queue" },
};

const HUMAN_STEPS = ["Answered in Crop Similarity", "Audited (independent)",
  "Audited (with suggestion)", "Reviewed on Escalations"];

const shown = (name: string) => name.replace(">=", "≥");
const pretty = (hazard: string) =>
  hazard === "all" ? "All types" : hazard.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

type Link = { source: string; target: string; value: number };
type Focus = { node?: string; link?: Link } | null;

export function Track2ReadinessFlowPanel({ database, available, databaseLabel, hazard, onHazardChange }: {
  hazard: string;
  onHazardChange: (value: string) => void;
  database: string;
  available: boolean;
  databaseLabel: string;
}) {
  const [flow, setFlow] = useState<Track2Readiness | null>(null);
  const [flowFor, setFlowFor] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [focus, setFocus] = useState<Focus>(null);
  const [pinned, setPinned] = useState<Focus>(null);
  // A slow answer for a database no longer chosen must not land.
  const askId = useRef(0);

  function read() {
    if (!database || !available) return;
    const id = ++askId.current;
    setBusy(true);
    setError("");
    getTrack2SqlReadiness(database)
      .then((res) => {
        if (id !== askId.current) return;
        setFlow(res);
        setFlowFor(database);
      })
      .catch((err) => {
        if (id === askId.current) setError(err instanceof ApiError ? err.detail : "Could not read the flow.");
      })
      .finally(() => { if (id === askId.current) setBusy(false); });
  }

  // Read once the database is up, and again when a different one is chosen.
  useEffect(() => {
    askId.current += 1;
    setFlow(null);
    setFlowFor("");
    setBusy(false);
    setError("");
    setFocus(null);
    setPinned(null);
    if (available) read();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [database, available]);

  const current = flow && flowFor === database ? flow : null;
  const choice = current && current.links[hazard] ? hazard : "all";
  const links = current?.links[choice] ?? [];
  const total = (name: string) => Math.max(
    links.filter((l) => l.target === name).reduce((a, l) => a + l.value, 0),
    links.filter((l) => l.source === name).reduce((a, l) => a + l.value, 0));
  const frames = current?.frames[choice] ?? 0;
  const role = (name: string): Track2ReadinessRole => current?.steps[name] ?? "flow";
  const boxed = !!current && (choice === "all" || choice === current.box_class);
  const ready = total("Box: human") + total("Box: detector >= 0.5");
  const human = HUMAN_STEPS.reduce((a, n) => a + total(n), 0);
  const todo = Object.keys(ABOUT)
    .filter((n) => role(n) === "fix" && total(n) > 0)
    .map((n) => ({ name: n, frames: total(n) }))
    .sort((a, b) => b.frames - a.frames);
  const recoverable = todo.reduce((a, r) => a + r.frames, 0);
  const shownFocus = focus ?? pinned;

  const classes = current
    ? ["all", ...Object.keys(current.frames).filter((k) => k !== "all").sort(
        (a, b) => (current.frames[b] ?? 0) - (current.frames[a] ?? 0))]
    : [];

  return (
    <Card
      title="Data Exploration statistics"
      stateKey="data exploration"
      caption="Where every judged frame stops before retraining, read live. Only human verdicts count: VLM-only and pre-filled answers stay out. Amber steps are where reviewers can recover frames."
    >
      {!available ? (
        <p className="text-xs text-ink-muted">
          {databaseLabel || "The database"} is not running. Start it above to read the flow.
        </p>
      ) : (
        <div className="flex flex-col gap-4">
          <div className="flex flex-wrap items-center gap-2">
            {classes.map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={key === choice}
                onClick={() => { onHazardChange(key); setFocus(null); setPinned(null); }}
                className={`rounded-full border px-3 py-1 text-xs ${key === choice
                  ? "border-ink bg-ink text-page" : "border-hairline bg-page text-ink"}`}
              >
                {pretty(key)}
                <span className="tabular ml-1.5 opacity-70">{(current?.frames[key] ?? 0).toLocaleString()}</span>
              </button>
            ))}
            <button
              type="button"
              onClick={read}
              disabled={busy}
              className="ml-auto rounded-md border border-hairline px-2.5 py-1 text-xs text-ink disabled:opacity-40"
            >
              {busy ? "Reading..." : "Refresh"}
            </button>
          </div>

          {error && <p className="text-xs text-critical">{error}</p>}
          {!current && busy && <p className="text-xs text-ink-muted">Reading every judged frame (about ten seconds)...</p>}

          {current && (
            <>
              <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
                <Stat label="Frames received" value={frames} note="judged by the app" />
                <Stat label="Human decision" value={human}
                      note={frames ? `${Math.round((human / frames) * 100)}% of frames` : ""} />
                <Stat label="Ready for retraining" value={boxed ? ready : null} tone="ready"
                      note={boxed ? `helmet frames · ${total("Box: human").toLocaleString()} test-grade`
                                  : "no retraining pipeline for this class yet"} />
                <Stat label="Recoverable by reviewers" value={recoverable} tone="fix" note="frames on amber steps" />
              </div>

              <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-ink-muted">
                {(Object.keys(ROLE_LABEL) as Track2ReadinessRole[]).map((r) => (
                  <span key={r} className="inline-flex items-center gap-1.5">
                    <i className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: ROLE_COLOUR[r] }} />
                    {ROLE_LABEL[r]}
                  </span>
                ))}
              </div>

              <FlowChart
                links={links}
                role={role}
                total={total}
                focus={shownFocus}
                onHover={setFocus}
                onPick={(f) => setPinned((p) => (sameFocus(p, f) ? null : f))}
              />

              <Detail focus={shownFocus} total={total} frames={frames} role={role} pinned={!!pinned && !focus} />

              <p className="text-xs text-ink-muted">
                {boxed
                  ? "Helmet frames go on to a helmet-box check. The other classes stop at “Fully resolved”: only the helmet / person detectors have a retraining pipeline so far."
                  : "This class stops at “Fully resolved”: it has no retraining pipeline yet, so there is no box check."}
              </p>

              <div>
                <p className="mb-1 text-sm font-medium text-ink">What reviewers can do next</p>
                <p className="mb-2 text-xs text-ink-muted">
                  Ranked by frames. A frame can need more than one fix: one recovered at the verdict step may then need a helmet box.
                </p>
                {todo.length ? (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b border-hairline text-left text-ink-muted">
                          <th className="py-1.5 pr-3 font-medium">Reviewer action</th>
                          <th className="py-1.5 pr-3 font-medium">Where</th>
                          <th className="py-1.5 pr-3 text-right font-medium">Frames</th>
                          <th className="w-1/5 py-1.5" />
                        </tr>
                      </thead>
                      <tbody>
                        {todo.map((r) => (
                          <tr key={r.name} className="border-b border-hairline/60 align-top">
                            <td className="py-1.5 pr-3 text-ink">
                              {ABOUT[r.name]?.action}
                              <span className="ml-1.5 rounded-full bg-accent-subtle px-1.5 text-[10px] text-ink-secondary">{shown(r.name)}</span>
                            </td>
                            <td className="py-1.5 pr-3 text-ink-secondary">{ABOUT[r.name]?.where}</td>
                            <td className="tabular py-1.5 pr-3 text-right text-ink">{r.frames.toLocaleString()}</td>
                            <td className="py-2">
                              <div className="h-1.5 rounded-sm" style={{
                                width: `${(r.frames / todo[0].frames) * 100}%`, minWidth: 2,
                                background: ROLE_COLOUR.fix }} />
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <p className="text-xs text-ink-muted">Nothing a reviewer can recover for this class.</p>
                )}
              </div>

              <p className="text-[11px] text-ink-muted">
                Read {formatDayTime(current.measured_at)} from {databaseLabel}. Read only: one query, nothing is
                written. Not applied yet: de-duplication, the train / val split, and whether each image is still in S3.
              </p>
            </>
          )}
        </div>
      )}
    </Card>
  );
}

function sameFocus(a: Focus, b: Focus) {
  if (!a || !b) return false;
  if (a.node || b.node) return a.node === b.node;
  return a.link?.source === b.link?.source && a.link?.target === b.link?.target;
}

function Stat({ label, value, note, tone }: {
  label: string; value: number | null; note: string; tone?: "ready" | "fix";
}) {
  const colour = tone === "ready" ? ROLE_COLOUR.ready : tone === "fix" ? ROLE_COLOUR.fix : undefined;
  return (
    <div className="min-w-0 rounded-sm border border-hairline bg-page px-3 py-2">
      <p className="text-[10px] uppercase tracking-wide text-ink-muted">{label}</p>
      <p className="tabular text-xl font-semibold" style={{ color: colour }}>
        {value === null ? "—" : value.toLocaleString()}
      </p>
      <p className="text-[11px] text-ink-muted">{note}</p>
    </div>
  );
}

function Detail({ focus, total, frames, role, pinned }: {
  focus: Focus;
  total: (name: string) => number;
  frames: number;
  role: (name: string) => Track2ReadinessRole;
  pinned: boolean;
}) {
  if (!focus) {
    return <p className="text-xs text-ink-muted">Point at a step or a band for what it means and what a reviewer can do. Click to keep it here.</p>;
  }
  const name = focus.node ?? focus.link!.target;
  const about = ABOUT[name] ?? { what: "" };
  const count = focus.node ? total(name) : focus.link!.value;
  const share = focus.node
    ? `${frames ? ((count / frames) * 100).toFixed(1) : 0}% of ${frames.toLocaleString()} frames`
    : `${((count / Math.max(1, total(focus.link!.source))) * 100).toFixed(1)}% of “${shown(focus.link!.source)}”`;
  return (
    <div className="rounded-sm border border-hairline bg-page px-3 py-2 text-xs"
         style={{ borderLeft: `3px solid ${ROLE_COLOUR[role(name)]}` }}>
      <p className="font-medium text-ink">
        {focus.node ? shown(name) : `${shown(focus.link!.source)} → ${shown(name)}`}
        <span className="tabular ml-2 font-normal text-ink-muted">{count.toLocaleString()} frames · {share}</span>
        {pinned && <span className="ml-2 font-normal text-ink-muted">(click again to let go)</span>}
      </p>
      <p className="mt-1 text-ink-secondary">{about.what}</p>
      {about.records && <p className="mt-1 text-ink-muted">Records: {about.records}</p>}
      {about.action && (
        <p className="mt-1 text-ink">
          <span className="font-medium">Reviewer action:</span> {about.action}
          {about.where && <span className="text-ink-muted"> ({about.where})</span>}
        </p>
      )}
    </div>
  );
}

/** A Sankey laid out by hand: the steps are a fixed, shallow sequence, so
 * a column per depth and a stack per column is the whole layout -- no
 * chart library to ship for one card. */
function FlowChart({ links, role, total, focus, onHover, onPick }: {
  links: Link[];
  role: (name: string) => Track2ReadinessRole;
  total: (name: string) => number;
  focus: Focus;
  onHover: (f: Focus) => void;
  onPick: (f: Focus) => void;
}) {
  const W = 1000, H = 440, NODE = 12, GAP = 12, LABEL = 190;
  const layout = useMemo(() => {
    const names = [...new Set(links.flatMap((l) => [l.source, l.target]))];
    // Depth = longest path from the start; the steps form a DAG.
    const depth: Record<string, number> = {};
    const visit = (n: string, d: number) => {
      if ((depth[n] ?? -1) >= d) return;
      depth[n] = d;
      links.filter((l) => l.source === n).forEach((l) => visit(l.target, d + 1));
    };
    visit("Frames received", 0);
    names.forEach((n) => { if (depth[n] === undefined) depth[n] = 0; });
    const columns = Math.max(1, ...Object.values(depth)) + 1;
    const order = Object.keys(ABOUT);
    const byColumn: string[][] = Array.from({ length: columns }, () => []);
    names.forEach((n) => byColumn[depth[n]].push(n));
    byColumn.forEach((c) => c.sort((a, b) => order.indexOf(a) - order.indexOf(b)));
    const k = Math.min(...byColumn.map((c) =>
      (H - GAP * Math.max(0, c.length - 1)) / Math.max(1, c.reduce((a, n) => a + total(n), 0))));
    const step = (W - LABEL - NODE) / Math.max(1, columns - 1);
    const pos: Record<string, { x: number; y: number; h: number }> = {};
    byColumn.forEach((c, i) => {
      let y = 0;
      c.forEach((n) => {
        const h = Math.max(2, total(n) * k);
        pos[n] = { x: i * step, y, h };
        y += h + GAP;
      });
    });
    // Each band leaves its source and enters its target stacked in step order.
    const out: Record<string, number> = {}, inn: Record<string, number> = {};
    const sorted = [...links].sort((a, b) =>
      order.indexOf(a.target) - order.indexOf(b.target) || order.indexOf(a.source) - order.indexOf(b.source));
    const bands = sorted.map((l) => {
      const w = Math.max(1, l.value * k);
      const s = pos[l.source], t = pos[l.target];
      const y0 = s.y + (out[l.source] ?? 0) + w / 2;
      const y1 = t.y + (inn[l.target] ?? 0) + w / 2;
      out[l.source] = (out[l.source] ?? 0) + w;
      inn[l.target] = (inn[l.target] ?? 0) + w;
      const x0 = s.x + NODE, x1 = t.x, mid = (x0 + x1) / 2;
      return { l, w, d: `M${x0},${y0} C${mid},${y0} ${mid},${y1} ${x1},${y1}` };
    });
    return { pos, bands };
  }, [links, total]);

  const lit = (l: Link) => !focus ? null
    : focus.node ? (l.source === focus.node || l.target === focus.node)
    : (l.source === focus.link!.source && l.target === focus.link!.target);

  return (
    <div className="overflow-x-auto rounded-sm border border-hairline bg-page p-2">
      <svg viewBox={`-2 -6 ${W + 4} ${H + 12}`} className="block h-auto w-full min-w-[760px]"
           role="img" aria-label="Flow of frames from receipt to retraining readiness"
           onMouseLeave={() => onHover(null)}>
        <g fill="none">
          {layout.bands.map(({ l, w, d }) => {
            const on = lit(l);
            return (
              <path key={`${l.source}>${l.target}`} d={d} strokeWidth={w}
                    stroke={ROLE_COLOUR[role(l.target)]}
                    strokeOpacity={on === null ? 0.3 : on ? 0.6 : 0.08}
                    className="cursor-pointer"
                    onMouseEnter={() => onHover({ link: l })}
                    onClick={() => onPick({ link: l })}>
                <title>{`${shown(l.source)} → ${shown(l.target)}: ${l.value.toLocaleString()} frames`}</title>
              </path>
            );
          })}
        </g>
        {Object.entries(layout.pos).map(([name, p]) => (
          <g key={name} className="cursor-pointer" tabIndex={0} role="button"
             aria-label={`${shown(name)}: ${total(name).toLocaleString()} frames`}
             onMouseEnter={() => onHover({ node: name })}
             onFocus={() => onHover({ node: name })}
             onBlur={() => onHover(null)}
             onClick={() => onPick({ node: name })}>
            <rect x={p.x} y={p.y} width={NODE} height={p.h} rx={2} fill={ROLE_COLOUR[role(name)]} />
            <text x={p.x + NODE + 5} y={p.y + p.h / 2} dy="0.35em" fontSize={11}
                  fill="var(--color-ink)" stroke="var(--color-page)" strokeWidth={3}
                  paintOrder="stroke" style={{ pointerEvents: "none" }}>
              {shown(name)}
              <tspan dx={5} fill="var(--color-ink-muted)">{total(name).toLocaleString()}</tspan>
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}
