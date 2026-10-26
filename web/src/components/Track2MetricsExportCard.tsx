import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  armTrack2AutoExport,
  disarmTrack2AutoExport,
  exportTrack2BestWeights,
  getTrack2AutoExport,
  getTrack2Buckets,
  getTrack2ExportFolder,
  getTrack2ExportGroups,
  listTrack2Folders,
  previewTrack2BestVersion,
  type Track2AutoExport,
  type Track2Bucket,
  type Track2ExportCriterion,
  type Track2ExportGroup,
  type Track2ExportPick,
  type Track2ExportRetention,
  type Track2ExportSettings,
  type Track2MetricsRow,
} from "../lib/api";
import { statusLabel } from "../lib/slug";
import { Badge } from "./Badge";
import { Card } from "./Card";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

/** Export control -- the best checkpoint of one Class, by hand or on a watch.
 *
 * What leaves here is the **.pt itself**. "The best version of the file"
 * is something you deploy or hand to somebody; a spreadsheet describing
 * it is not, and this card exported the spreadsheet until now.
 *
 * The flow is the brief, in order:
 *
 *   Class -> model -> size      which registry group
 *   criterion                   which version in it is "best"
 *   this computer | S3 bucket   where it goes, one or the other
 *   Export now                  by hand
 *   Auto export                 or whenever a new version registers
 *
 * **Nothing is excluded on approval status or validation count.** That
 * was asked and answered, and it has teeth on this estate:
 *
 *   v4  mAP50 0.995  measured on   5 images
 *   v9  mAP50 0.000  measured on 149 images
 *
 * Ranked on mAP50 with no floor, v4 wins by a mile on five pictures. So
 * both numbers are stated next to whatever gets picked, every time. Not
 * gating is a decision; hiding what it costs would be a defect.
 *
 * The pick is the SERVER's, not this card's, although the page already
 * holds every row and could rank them itself. Two implementations of
 * "best" -- one deciding what the card says, one deciding what the button
 * does -- is how a card comes to promise v4 and ship v9.
 */

const CRITERIA: { key: Track2ExportCriterion; label: string; hint: string }[] = [
  { key: "mAP50", label: "Highest mAP50", hint: "the usual headline detection score" },
  { key: "mAP50-95", label: "Highest mAP50-95", hint: "stricter -- averaged over ten IoU thresholds" },
  { key: "precision", label: "Highest precision", hint: "fewest false alarms" },
  { key: "recall", label: "Highest recall", hint: "fewest misses" },
  { key: "f1", label: "Highest F1", hint: "precision and recall balanced" },
  { key: "accuracy", label: "Highest accuracy", hint: "the legacy evaluator's number" },
  { key: "newest", label: "Newest version", hint: "whatever trained last, whatever it scored" },
];

/** What happens to the earlier exports once a new one lands.
 *
 * Every mode writes the new file first. They differ only in what they
 * then do about the ones already there, which is a retention decision
 * and belongs to whoever owns the folder. */
const RETENTION: { key: Track2ExportRetention; label: string; detail: string }[] = [
  { key: "keep_all", label: "Keep every export",
    detail: "-- each one under its own date and time. The folder is the history." },
  { key: "archive_oldest", label: "Keep the latest, archive the rest",
    detail: "-- earlier ones move into archive/ under the same folder. Nothing is lost." },
  { key: "keep_latest", label: "Keep only the latest, delete the rest",
    detail: "-- earlier ones are removed. This cannot be undone." },
];

const SELECT_CLASS =
  "rounded-sm border border-black/15 bg-page px-2 py-1 text-xs text-ink";

function statusTone(status: string): "confirmed" | "rejected" | "pending" {
  if (status === "Approved") return "confirmed";
  if (status === "Rejected") return "rejected";
  return "pending";
}

/** The chosen version, with the two things nothing filters on. */
function Pick({ pick }: { pick: Track2ExportPick }) {
  return (
    <div className="rounded-sm border border-hairline bg-page p-3">
      <div className="mb-2 flex flex-wrap items-center gap-x-3 gap-y-1">
        <span className="tabular text-sm font-semibold text-ink">{pick.label}</span>
        <Badge
          tone={statusTone(pick.status)}
          label={pick.is_seed ? "Seed" : statusLabel(pick.status)}
        />
        <span className="tabular text-xs text-ink-muted">{pick.group_name}</span>
      </div>
      <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs">
        <span className="text-ink-secondary">
          {pick.criterion === "newest" ? "Version" : pick.criterion}{" "}
          <span className="tabular font-medium text-ink">
            {pick.value == null ? "--" : pick.criterion === "newest" ? pick.value : pick.value.toFixed(3)}
          </span>
        </span>
        <span className="text-ink-secondary">
          Validation images{" "}
          <span className={`tabular ${pick.thin_validation ? "font-semibold text-critical" : "font-medium text-ink"}`}>
            {pick.validation_images ?? "--"}
          </span>
          {/* Never colour alone -- the words carry it for anyone who
              cannot see the red. */}
          {pick.thin_validation && (
            <span className="ml-1 text-critical">
              -- too few for that score to mean much
            </span>
          )}
        </span>
        {pick.classes.length > 0 && (
          <span className="text-ink-secondary">
            Model Names <span className="text-ink">{pick.classes.join(", ")}</span>
          </span>
        )}
      </div>
      {pick.is_seed && (
        <p className="mt-2 text-xs text-critical">
          This version is a <strong>seed</strong> &mdash; its weights detect
          something other than this Model Name, and it is waiting for the training
          run that rebuilds the head. Exporting it hands over a checkpoint
          that does not do what this Model Name says.
        </p>
      )}
      {!pick.is_seed && pick.status !== "Approved" && (
        <p className="mt-2 text-xs text-critical">
          This version is {pick.status === "PendingManualApproval" ? "still pending" : pick.status.toLowerCase()}.
          Nothing here filters on approval, so it will be exported as it stands.
        </p>
      )}
    </div>
  );
}

export function Track2MetricsExportCard({
  rows,
  settings,
  onChange,
}: {
  rows: Track2MetricsRow[];
  /** Owned by the page, so the preset toolbar at the top and this card
   * edit one set of values rather than two that can disagree. */
  settings: Track2ExportSettings;
  onChange: (next: Track2ExportSettings) => void;
}) {
  const { registry, generation, size, group_name: group, criterion, destination, bucket, prefix, retention } = settings;

  // Patches are built from the LATEST settings, not from whatever this
  // render closed over. Several things here write back -- the three
  // dropdowns normalising themselves, the group they resolve to, and the
  // bucket list arriving from an async call made at mount -- and the
  // async one would otherwise spread a `settings` captured before the
  // other two had run, silently reverting the Class that had just been
  // filled in.
  const latest = useRef(settings);
  latest.current = settings;
  const set = (patch: Partial<Track2ExportSettings>) =>
    onChange({ ...latest.current, ...patch });

  const [buckets, setBuckets] = useState<Track2Bucket[]>([]);
  const [bucketsError, setBucketsError] = useState("");
  const [folders, setFolders] = useState<string[]>([]);
  const [pick, setPick] = useState<Track2ExportPick | null>(null);
  const [pickReason, setPickReason] = useState("");
  const [considered, setConsidered] = useState(0);
  const [picking, setPicking] = useState(false);
  const [busy, setBusy] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState("");
  const [auto, setAuto] = useState<Track2AutoExport | null>(null);
  const [autoPending, setAutoPending] = useState("");

  // --- the three dropdowns, and what they resolve to --------------------
  //
  // Built from EVERY GROUP THE PLATFORM CAN PRODUCE, not from the
  // versions that happen to exist. Built from what exists, a Class that
  // has only ever trained at YOLO11-S offers YOLO11 and S and nothing
  // else -- so the group a YOLO26-X run will land in next week is
  // unreachable, which is exactly when you would want it set up: armed in
  // advance, the first version to appear there goes out the moment it
  // registers.
  //
  // The names are the SERVER's. Composing "yolo{gen}{size}-{slug}-detector"
  // here is how the Models card once told somebody a YOLO26 run would
  // register into yolo11s-helmet-detector while the run used the right
  // name -- a card and a run disagreeing about where a model was going.
  const [groups, setGroups] = useState<Track2ExportGroup[]>([]);
  // Said, not swallowed. A failed fetch left `groups` empty and the Class
  // dropdown reading "loading..." for ever, which is indistinguishable
  // from a slow one -- and the whole card is dead behind it.
  const [groupsError, setGroupsError] = useState("");
  useEffect(() => {
    getTrack2ExportGroups()
      .then((r) => { setGroups(r.groups); setGroupsError(""); })
      .catch((err) => setGroupsError(
        err instanceof ApiError ? err.detail : "Could not read the registry groups."));
  }, []);

  /** How many versions are registered in a group, from the rows the page
   * already loaded. Zero is a real and useful answer here -- it is what
   * "nothing has trained into this yet" looks like -- so it annotates the
   * options rather than removing them. */
  const versionCount = useMemo(() => {
    const counts = new Map<string, number>();
    for (const r of rows) counts.set(r.group_name, (counts.get(r.group_name) ?? 0) + 1);
    return counts;
  }, [rows]);

  const classes = useMemo(
    () => Array.from(new Set(groups.map((g) => g.registry))).sort(), [groups]);
  const generations = useMemo(
    () => Array.from(new Set(groups.filter((g) => !registry || g.registry === registry)
      .map((g) => g.generation))).sort(),
    [groups, registry]);
  const sizes = useMemo(
    () => Array.from(new Set(groups
      .filter((g) => (!registry || g.registry === registry) && g.generation === generation)
      .map((g) => g.size))).sort(),
    [groups, registry, generation]);

  // ONE group, or none. Three dropdowns that narrowed to several would
  // leave "which one is it exporting" unanswered, so the resolved name is
  // shown rather than implied.
  const matching = useMemo(() => Array.from(new Set(groups
    .filter((g) => (!registry || g.registry === registry)
      && g.generation === generation && g.size === size)
    .map((g) => g.group_name))).sort(),
    [groups, registry, generation, size]);

  // Keep the dropdowns on something that exists: narrowing one can remove
  // what another was set to, and a select holding a value absent from its
  // own options renders blank with no way to tell why.
  //
  // "" is a legitimate value for generation and size -- the legacy custom
  // group states neither -- so these test MEMBERSHIP rather than
  // truthiness. Falsy-testing them put that group permanently out of
  // reach while looking like it was being defaulted.
  useEffect(() => {
    if (!groups.length) return;
    const patch: Partial<Track2ExportSettings> = {};
    const nextRegistry = classes.includes(registry) ? registry : (classes[0] ?? "");
    if (nextRegistry !== registry) {
      patch.registry = nextRegistry;
      // Only when something was already chosen. Filling an empty card on
      // first render is not a retarget.
      if (registry) setRetargeted(registry);
    }
    if (!generations.includes(generation)) patch.generation = generations[0] ?? "";
    if (!sizes.includes(size)) patch.size = sizes[0] ?? "";
    if (Object.keys(patch).length) set(patch);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [groups, classes, generations, sizes]);

  // The resolved group IS the address the server acts on, so it is stored
  // rather than recomputed at export time -- a preset saved today has to
  // mean the same group when it is loaded next month.
  const resolved = matching.length === 1 ? matching[0] : "";
  useEffect(() => {
    if (resolved && resolved !== group) set({ group_name: resolved });
    if (!resolved && group && matching.length !== 1) set({ group_name: "" });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resolved, matching.length]);

  const registered = group ? (versionCount.get(group) ?? 0) : 0;
  // A preset saved against a Class this card no longer offers -- the open
  // registry, or a Class since deleted. The normalising effect quietly
  // moves the selection to the first one that does exist, which is the
  // right recovery and the wrong silence: an export would go out under a
  // Class nobody chose.
  const [retargeted, setRetargeted] = useState("");
  // The stem the file will really carry, from the server. Composed here
  // it came out as `yolo11s-helmet` while the file lands as
  // `yolo11-s-helmet`, so the line promising where it would be written
  // named a key that never appears in the bucket.
  const exportStem = groups.find((g) => g.group_name === group)?.export_stem ?? "";

  // --- buckets and folders ----------------------------------------------
  useEffect(() => {
    getTrack2Buckets()
      .then((r) => {
        setBuckets(r.buckets);
        // Only when nothing is chosen, and read at RESOLUTION time: a
        // preset loaded while this call was in flight names its own
        // bucket, and defaulting over it would quietly redirect an
        // export somebody had already decided on.
        if (!latest.current.bucket) set({ bucket: r.default || r.buckets[0]?.name || "" });
      })
      .catch((err) => setBucketsError(
        err instanceof ApiError ? err.detail : "Could not list the buckets."));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    // Only once S3 is the destination: listing a bucket's prefixes is a
    // real call, and nobody downloading to their own machine needs it.
    if (!bucket || destination !== "s3") return;
    let cancelled = false;
    listTrack2Folders(bucket, prefix)
      .then((r) => !cancelled && setFolders(r.folders))
      .catch(() => !cancelled && setFolders([]));
    return () => { cancelled = true; };
  }, [bucket, prefix, destination]);

  // What the two housekeeping modes would act on, from the SERVER's own
  // `export_siblings` -- the same function they call, not a second
  // approximation of it drawn in the browser. A preview that scoped the
  // deletion differently from the deletion would be worse than none.
  const [siblings, setSiblings] = useState<string[] | null>(null);
  useEffect(() => {
    if (!bucket || !group || destination !== "s3" || retention === "keep_all") {
      setSiblings(null);
      return;
    }
    let cancelled = false;
    setSiblings(null);
    getTrack2ExportFolder({ group_name: group, bucket, prefix })
      .then((r) => !cancelled && setSiblings(r.files))
      .catch(() => !cancelled && setSiblings([]));
    return () => { cancelled = true; };
  }, [bucket, prefix, group, destination, retention, note]);

  // --- which version wins ------------------------------------------------
  useEffect(() => {
    if (!group) { setPick(null); setPickReason(""); setConsidered(0); return; }
    let cancelled = false;
    setPicking(true);
    previewTrack2BestVersion({ group_name: group, criterion })
      .then((r) => {
        if (cancelled) return;
        setPick(r.best);
        setPickReason(r.reason);
        setConsidered(r.considered);
      })
      .catch((err) => {
        if (cancelled) return;
        setPick(null);
        setPickReason(err instanceof ApiError ? err.detail : "Could not read that group.");
      })
      .finally(() => !cancelled && setPicking(false));
    return () => { cancelled = true; };
  }, [group, criterion]);

  // --- the watcher -------------------------------------------------------
  const refreshAuto = useCallback(() => {
    getTrack2AutoExport()
      .then((r) => { setAuto(r.settings); setAutoPending(r.pending.reason); })
      .catch(() => undefined);
  }, []);
  useEffect(refreshAuto, [refreshAuto]);

  // Polled while the card is open. The watcher fires on the server's own
  // 25s tick, so a card that only read it on mount would show "waiting"
  // for an export that had already happened.
  const armed = auto?.enabled ?? false;
  const pollRef = useRef<number | undefined>(undefined);
  useEffect(() => {
    if (!armed) return;
    pollRef.current = window.setInterval(refreshAuto, 30000);
    return () => window.clearInterval(pollRef.current);
  }, [armed, refreshAuto]);

  async function exportNow() {
    // The one irreversible path on this card. The list above already
    // says what would go, but a delete that happens on the same single
    // press as an export is a delete nobody separately agreed to -- and
    // this is somebody else's bucket as often as not.
    if (destination === "s3" && retention === "keep_latest" && siblings?.length) {
      // Line by line, joined -- a multi-line template literal here would
      // carry this file's own indentation into the dialog.
      const ok = window.confirm([
        `Delete ${siblings.length} earlier export${siblings.length === 1 ? "" : "s"}`
        + ` of ${group} from s3://${bucket}/${prefix ? `${prefix}/` : ""}?`,
        "",
        ...siblings,
        "",
        "The new export is written first. This cannot be undone.",
      ].join("\n"));
      if (!ok) return;
    }
    setBusy(destination === "s3" ? "Copying..." : "Preparing...");
    setNote("");
    setError("");
    try {
      const res = await exportTrack2BestWeights({
        group_name: group, criterion, destination,
        ...(destination === "s3" ? { bucket, prefix, retention } : {}),
      });
      if (res.url) {
        // Straight to S3 from the browser. A 19MB-110MB checkpoint
        // streamed through the API container would stall every request
        // in that single-worker process for its duration.
        const link = document.createElement("a");
        link.href = res.url;
        link.download = res.filename;
        link.click();
        setNote(`v${res.picked.version} downloading as ${res.filename}`);
      } else {
        const tidied = res.retention.deleted.length
          ? ` -- ${res.retention.deleted.length} earlier export${res.retention.deleted.length === 1 ? "" : "s"} deleted`
          : res.retention.archived.length
            ? ` -- ${res.retention.archived.length} earlier export${res.retention.archived.length === 1 ? "" : "s"} moved to archive/`
            : "";
        setNote(`v${res.picked.version} written to ${res.written_to}${tidied}`);
      }
      setPick(res.picked);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "The export failed.");
    } finally {
      setBusy("");
    }
  }

  async function toggleAuto() {
    // Armed, this deletes without anybody present. Agreed to once, here,
    // rather than silently inherited from the radio above.
    if (!armed && retention === "keep_latest") {
      const ok = window.confirm([
        `Every automatic export of ${group} will delete the earlier ones in `
        + `s3://${bucket}/${prefix ? `${prefix}/` : ""} once the new file has `
        + `landed, with nobody watching. Only this Model Name's own .pt exports, `
        + `and only directly in that folder.`,
        "",
        "Switch it on?",
      ].join("\n"));
      if (!ok) return;
    }
    setBusy(armed ? "Switching off..." : "Arming...");
    setNote("");
    setError("");
    try {
      const res = armed
        ? await disarmTrack2AutoExport()
        : await armTrack2AutoExport({ group_name: group, criterion, bucket, prefix, retention });
      setAuto(res.settings);
      setNote(res.settings.enabled
        ? `Watching ${res.settings.group_name} from v${res.settings.last_version}.`
        : "Automatic export switched off.");
      refreshAuto();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "That did not work.");
    } finally {
      setBusy("");
    }
  }

  const noGroup = !group;
  const nothingToExport = !pick;
  const needsBucket = destination === "s3" && !bucket;
  const result = auto?.last_result;

  // What the watcher is running on. ARMED, this is the server's copy --
  // taken when On was pressed, not read live off the controls above --
  // so a radio changed on screen cannot silently change what is already
  // running against somebody's folder. OFF, it is the controls above,
  // because the panel is then a preview of what On would do.
  const watched = armed && auto
    ? { group: auto.group_name, criterion: auto.criterion, bucket: auto.bucket,
        prefix: auto.prefix, retention: auto.retention }
    : { group, criterion, bucket, prefix, retention };
  const watchedCriterionLabel =
    CRITERIA.find((c) => c.key === watched.criterion)?.label ?? watched.criterion;
  const watchedRetentionLabel =
    RETENTION.find((r) => r.key === watched.retention)?.label ?? watched.retention;
  // Armed, but against settings the controls above no longer agree with.
  // Worth saying plainly: the gap between what the page shows and what is
  // actually running is the whole failure mode of a background job.
  const drifted = Boolean(armed && auto && (
    auto.group_name !== group || auto.criterion !== criterion
    || auto.bucket !== bucket || auto.prefix !== prefix
    || auto.retention !== retention));

  return (
    <Card title="Export control">
      <p className="mb-3 text-xs text-ink-secondary">
        Sends the <strong>checkpoint itself</strong> &mdash; the{" "}
        <span className="tabular">.pt</span> you would deploy or hand to
        somebody. Pick a Model Name, pick what decides which of its versions is
        best, and send it. Every export is stamped to the second and never
        replaces an earlier one.
      </p>

      {retargeted && (
        <p className="mb-3 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical">
          This was set to the Model Name <span className="tabular">{retargeted}</span>,
          which is not offered here any more, so it has moved to{" "}
          <span className="tabular">{registry}</span>. Check the selection
          below before exporting.{" "}
          <button type="button" onClick={() => setRetargeted("")} className="underline">
            Dismiss
          </button>
        </p>
      )}

      {groupsError && (
        <p className="mb-3 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical">
          {groupsError} Nothing below can be chosen until that succeeds.
        </p>
      )}

      {/* 1. Which Model Name. Every combination the platform can produce, not
          only the ones something has already trained into. */}
      <div className="flex flex-wrap items-end gap-x-4 gap-y-2 border-b border-hairline pb-3">
        <label className="flex flex-col gap-1 text-xs text-ink-secondary">
          Model Name
          <select value={registry} onChange={(e) => set({ registry: e.target.value })} className={`${SELECT_CLASS} py-1.5`}>
            {classes.length === 0 && (
              <option value="">{groupsError ? "unavailable" : "loading..."}</option>
            )}
            {classes.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-ink-secondary">
          Model type
          <select value={generation} onChange={(e) => set({ generation: e.target.value })} className={`${SELECT_CLASS} py-1.5`}>
            {generations.length === 0 && <option value="">--</option>}
            {/* "" is a real value, not a missing one: the legacy custom
                group's name states no generation, and labelling it YOLO
                would give it one it never had. */}
            {generations.map((g) => (
              <option key={g || "legacy"} value={g}>{g ? `YOLO${g}` : "Legacy"}</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-ink-secondary">
          Size
          <select value={size} onChange={(e) => set({ size: e.target.value })} className={`${SELECT_CLASS} py-1.5`}>
            {sizes.length === 0 && <option value="">--</option>}
            {sizes.map((sz) => (
              <option key={sz || "none"} value={sz}>{sz ? sz.toUpperCase() : "--"}</option>
            ))}
          </select>
        </label>
        {/* Stated, not implied. Three dropdowns that resolve to a registry
            group leave "which group is this actually" unanswered unless
            the answer is on screen. */}
        <span className="tabular pb-1.5 text-xs text-ink-muted">
          {group
            ? <>Exports from <span className="text-ink">{group}</span>{" "}
                {registered > 0
                  ? `-- ${registered} version${registered === 1 ? "" : "s"}`
                  : "-- nothing registered yet"}</>
            : matching.length > 1
              ? `${matching.length} groups match -- narrow it with Model and Size.`
              : "No registry group matches."}
        </span>
      </div>

      {/* An empty group is a normal thing to set up, not a mistake: it is
          what "arm this before the first run lands" looks like. Said here
          so the empty pick below reads as waiting rather than broken. */}
      {group && registered === 0 && (
        <p className="mt-3 text-xs text-ink-secondary">
          Nothing has trained into <span className="tabular text-ink">{group}</span> yet.
          There is nothing to export by hand, but you can still switch on the
          automatic export &mdash; the first version to register here goes out
          the moment it does.
        </p>
      )}

      {/* 2. Which version of it. */}
      <div className="flex flex-wrap items-end gap-x-4 gap-y-2 border-b border-hairline py-3">
        <label className="flex flex-col gap-1 text-xs text-ink-secondary">
          Best version decided by
          <select
            value={criterion}
            onChange={(e) => set({ criterion: e.target.value as Track2ExportCriterion })}
            className={`${SELECT_CLASS} min-w-[16rem] py-1.5`}
          >
            {CRITERIA.map((c) => (
              <option key={c.key} value={c.key}>{c.label} -- {c.hint}</option>
            ))}
          </select>
        </label>
        <span className="pb-1.5 text-xs text-ink-muted">
          Ties go to the higher version number.
        </span>
      </div>

      <div className="py-3">
        {picking ? (
          <p className="text-xs text-ink-muted">Reading the registry...</p>
        ) : pick ? (
          <>
            <Pick pick={pick} />
            {/* The server's own count, not the page's: the pick was made
                against what the registry held a second ago, and saying so
                is how a stale page becomes visible rather than silent. */}
            <p className="mt-1.5 text-xs text-ink-muted">
              Best of {considered} version{considered === 1 ? "" : "s"} in this group.
            </p>
          </>
        ) : (
          <p className="text-xs text-ink-muted">
            {pickReason || (noGroup ? "Pick a Model Name, model and size above." : "Nothing to export.")}
          </p>
        )}
      </div>

      {/* 3. Where it goes -- one or the other, by construction. */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-t border-hairline pt-3 text-xs">
        <span className="text-ink-muted">Send it to</span>
        {([["local", "This computer"], ["s3", "An S3 bucket"]] as const).map(([value, label]) => (
          <label key={value} className="flex items-center gap-1.5 text-ink-secondary">
            <input
              type="radio"
              name="track2-export-destination"
              checked={destination === value}
              onChange={() => set({ destination: value })}
            />
            {label}
          </label>
        ))}
      </div>

      {destination === "s3" && (
        <div className="mt-2">
          {bucketsError && (
            <p className="mb-2 text-xs text-critical">{bucketsError}</p>
          )}
          <label className="flex flex-col gap-1 text-xs text-ink-secondary">
            S3 bucket
            <select
              value={bucket}
              onChange={(e) => set({ bucket: e.target.value, prefix: "" })}
              className={`${SELECT_CLASS} tabular w-full max-w-lg py-1.5`}
            >
              {buckets.map((b) => (
                <option key={b.name} value={b.name} disabled={!b.readable}>
                  {b.name}{b.readable ? "" : "  (no read access)"}
                </option>
              ))}
            </select>
          </label>

          {/* Always open on this side. There is nothing else to look at
              once S3 is chosen, and a browser behind a toggle made the
              destination something you had to go and check. Wherever you
              have walked to IS the destination, which the line below
              states. */}
          <div className="mt-2">
            <Track2S3FolderBrowser
              prefix={prefix}
              folders={folders}
              onNavigate={(next) => set({ prefix: next })}
            />
          </div>

          <p className="tabular mt-2 break-all text-xs text-ink-muted">
            Writes to{" "}
            <span className="text-ink">
              s3://{bucket || "(pick a bucket)"}/{prefix ? `${prefix}/` : ""}
              {exportStem || "<class>"}-v&lt;n&gt;-&lt;timestamp&gt;.pt
            </span>
          </p>

        </div>
      )}

      {/* What happens to what is ALREADY there.

          ALWAYS on screen, whichever destination is chosen. It only acts
          on an S3 folder -- a download adds no file to anybody's folder,
          so there is nothing there to tidy -- but hiding it behind the
          radio made a standing retention policy look like a property of
          one destination. It is a decision whether or not you make it:
          "keep everything" is the policy you get by not choosing, and
          saying so beats leaving it to be discovered when the folder has
          ninety files in it. The line below states where it applies
          rather than the card quietly disappearing. */}
      <div className="mt-4 border-t border-hairline pt-3">
        <p className="mb-1 text-xs font-semibold text-ink">
          Once the new one lands
        </p>
        <p className="mb-2 text-xs text-ink-secondary">
          Every export is written first, under its own date and time. This
          is what then happens to the <em>earlier</em> exports of this
          Model Name in that folder &mdash; by hand and automatically, the same
          way both times.
        </p>
        <div className="flex flex-col gap-1.5">
          {RETENTION.map(({ key, label, detail }) => (
            <label key={key} className="flex items-start gap-2 text-xs text-ink-secondary">
              <input
                type="radio"
                name="track2-export-retention"
                className="mt-0.5"
                checked={retention === key}
                onChange={() => set({ retention: key })}
              />
              <span>
                <span className="font-medium text-ink">{label}</span>
                <span className="ml-1.5 text-ink-muted">{detail}</span>
              </span>
            </label>
          ))}
        </div>

        {/* Where it applies, said rather than implied by the card
            vanishing. A download leaves nothing behind to tidy. */}
        {destination !== "s3" && (
          <p className="mt-2 text-xs text-ink-muted">
            Applies to exports written to an S3 bucket, and to the
            automatic export. A download leaves nothing in a folder, so
            there is nothing for this to act on.
          </p>
        )}

        {/* The blast radius, listed rather than described. A button whose
            reach you have to infer from its label is one nobody should
            press -- and this one deletes objects in a bucket somebody
            else may also be using. */}
        {destination === "s3" && retention !== "keep_all" && (
          <div className={`mt-2 rounded-sm border px-3 py-2 text-xs ${
            retention === "keep_latest"
              ? "border-critical/30 bg-critical-subtle text-critical"
              : "border-hairline bg-page text-ink-secondary"}`}>
            {siblings === null ? (
              <span>Reading the folder...</span>
            ) : siblings.length === 0 ? (
              <span>
                No earlier export of{" "}
                <span className="tabular">{group || "this Model Name"}</span>{" "}
                is in that folder, so nothing would be{" "}
                {retention === "keep_latest" ? "deleted" : "archived"} this time.
              </span>
            ) : (
              <>
                <span>
                  {retention === "keep_latest" ? "Would delete" : "Would move to archive/"}{" "}
                  {siblings.length} earlier export{siblings.length === 1 ? "" : "s"}:
                </span>
                <span className="tabular mt-1 block break-all">
                  {siblings.slice(0, 4).join(", ")}
                  {siblings.length > 4 && ` and ${siblings.length - 4} more`}
                </span>
              </>
            )}
            {/* Said every time, not only when something would go. The
                scope is the reassurance, and it has to be readable when
                the list is empty too. */}
            <span className="mt-1 block text-ink-muted">
              Only this Model Name&rsquo;s own{" "}
              <span className="tabular">.pt</span> exports, directly in
              that folder. Anything else there is left alone.
            </span>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={exportNow}
          disabled={!!busy || noGroup || nothingToExport || needsBucket}
          className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy || (destination === "s3" ? "Export to S3 now" : "Download to this computer")}
        </button>
        {needsBucket && <span className="text-xs text-ink-muted">Pick a bucket first.</span>}
      </div>

      {note && <p className="tabular mt-2 break-all text-xs text-ink-secondary">{note}</p>}
      {error && <p className="mt-2 text-xs text-critical">{error}</p>}

      {/* 4. The watcher.

          On and Off are two buttons, not one that changes its own label.
          A single toggle reading "Watch this group" never says what the
          current state IS -- you have to know that a button offering to
          start something means it is stopped -- and the same button
          turning into "Switch it off" is the control you press by
          muscle memory to do the opposite of what you meant.

          Everything below them is the SETTINGS the watcher runs on, and
          is greyed while it is off: off, they are a preview of what On
          would do, which is worth showing and must not read as something
          currently running. */}
      <div className="mt-5 rounded-sm border border-hairline bg-page p-3">
        <div className="mb-3 flex flex-wrap items-center gap-x-3 gap-y-2">
          <span className="text-sm font-semibold text-ink">Automatic export</span>
          <span className="inline-flex overflow-hidden rounded-md border border-black/15">
            <button
              type="button"
              onClick={() => { if (!armed) void toggleAuto(); }}
              disabled={!!busy || (!armed && (noGroup || destination !== "s3" || !bucket))}
              aria-pressed={armed}
              className={`px-4 py-1.5 text-xs font-semibold disabled:opacity-40 ${
                armed ? "bg-accent text-white" : "bg-surface text-ink-secondary"}`}
            >
              On
            </button>
            <button
              type="button"
              onClick={() => { if (armed) void toggleAuto(); }}
              disabled={!!busy || !armed}
              aria-pressed={!armed}
              className={`border-l border-black/15 px-4 py-1.5 text-xs font-semibold disabled:opacity-40 ${
                !armed ? "bg-ink text-white" : "bg-surface text-ink-secondary"}`}
            >
              Off
            </button>
          </span>
          <span className={`text-xs ${armed ? "font-medium text-ink" : "text-ink-muted"}`}>
            {busy && busy.startsWith("Arm") ? "Arming..."
              : busy && busy.startsWith("Switch") ? "Switching off..."
              : armed ? "Watching -- this will export without you." : "Nothing is being watched."}
          </span>
        </div>

        {/* S3 only, and said rather than silently refused. Above the
            panel, because it is the reason On is greyed out. */}
        {destination !== "s3" && !armed && (
          <p className="mb-3 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical">
            An automatic export has to go to an S3 bucket. Nothing is running
            in a browser when it fires, so there is nowhere on your computer
            for the file to land. Choose <em>An S3 bucket</em> above to switch
            this on.
          </p>
        )}
        {destination === "s3" && !bucket && !armed && (
          <p className="mb-3 text-xs text-ink-muted">Pick a bucket above to switch this on.</p>
        )}
        {noGroup && !armed && (
          <p className="mb-3 text-xs text-ink-muted">
            Pick a Model Name, model and size above to switch this on.
          </p>
        )}

        <div className={armed ? "" : "pointer-events-none select-none opacity-50"}>
          <p className="mb-2 text-xs font-semibold text-ink">
            {armed ? "What it is doing" : "What switching it on would do"}
          </p>

          {/* Spelled out as a sequence rather than described. "Watches
              for new versions" is a sentence somebody has to translate
              into "so when my 3am run finishes, what lands where?" --
              which is the only question being asked. */}
          <ol className="mb-3 flex list-decimal flex-col gap-1 pl-4 text-xs text-ink-secondary">
            <li>
              A training run finishes and registers a new version in{" "}
              <span className="tabular text-ink">{watched.group || "the chosen group"}</span>.
              That is the only thing it waits for &mdash; no clock, no schedule.
            </li>
            <li>
              Within about 25 seconds, every version in that group is ranked by{" "}
              <span className="text-ink">{watchedCriterionLabel}</span>.
            </li>
            <li>
              If that changed which version wins, its{" "}
              <span className="tabular">.pt</span> is written to{" "}
              <span className="tabular break-all text-ink">
                s3://{watched.bucket || "(no bucket)"}/{watched.prefix ? `${watched.prefix}/` : ""}
              </span>{" "}
              under a fresh date and time. If the new version did not beat the
              one already there, <strong>nothing is written</strong>.
            </li>
            <li>
              The earlier exports of this Model Name in that folder are then handled
              as{" "}
              <span className={watched.retention === "keep_latest" ? "font-medium text-critical" : "text-ink"}>
                {watchedRetentionLabel}
              </span>.
            </li>
          </ol>

          {/* The settings themselves, as a table. Armed, these are the
              SERVER's copy -- taken when you pressed On, not read live
              off the radios -- so a control changed on screen cannot
              silently change what is already running against a folder. */}
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-xs">
            <dt className="text-ink-muted">Watching</dt>
            <dd className="tabular break-all text-ink">{watched.group || "--"}</dd>
            <dt className="text-ink-muted">Best version by</dt>
            <dd className="text-ink">{watchedCriterionLabel}</dd>
            <dt className="text-ink-muted">Writes to</dt>
            <dd className="tabular break-all text-ink">
              s3://{watched.bucket || "--"}/{watched.prefix ? `${watched.prefix}/` : ""}
            </dd>
            <dt className="text-ink-muted">Earlier exports</dt>
            <dd className={watched.retention === "keep_latest" ? "font-medium text-critical" : "text-ink"}>
              {watchedRetentionLabel}
            </dd>
            {armed && auto && (
              <>
                <dt className="text-ink-muted">Seen up to</dt>
                <dd className="tabular text-ink">
                  v{auto.last_version}
                  {auto.last_exported_version && ` -- exported v${auto.last_exported_version}`}
                </dd>
                <dt className="text-ink-muted">Switched on</dt>
                <dd className="tabular text-ink">
                  {auto.armed_at ? new Date(auto.armed_at).toLocaleString() : "--"}
                </dd>
              </>
            )}
          </dl>

          {armed && (
            <div className="mt-2 flex flex-col gap-1 text-xs">
              {autoPending && <span className="text-ink-muted">{autoPending}</span>}
              {result?.at && (
                <span className={`tabular break-all ${result.ok ? "text-ink-secondary" : "text-critical"}`}>
                  Last check {new Date(result.at).toLocaleString()}: {result.detail}
                  {result.ok && result.thin_validation
                    && ` -- measured on ${result.validation_images} validation images`}
                </span>
              )}
              {/* Both containers share this host volume's parent but not
                  this folder, so dev and stable are armed independently.
                  Said here because the two are a second window onto the
                  same production data, and a file written by the other
                  one is otherwise unexplainable. */}
              <span className="text-ink-muted">
                Switched on per container. Doing it here does not do it on the
                other one.
              </span>
            </div>
          )}
        </div>

        {/* Outside the greyed panel: it is about the controls above, not
            about the watcher, and it has to be readable. */}
        {drifted && (
          <p className="mt-3 rounded-sm border border-hairline bg-surface px-3 py-2 text-xs text-ink-secondary">
            The settings above the buttons have changed since this was switched
            on, and <strong>it is still doing what it was switched on to do</strong>.
            Switch it off and on again to pick up the new ones.
          </p>
        )}
      </div>
    </Card>
  );
}
