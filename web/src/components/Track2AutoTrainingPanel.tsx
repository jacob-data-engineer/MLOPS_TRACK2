import { post } from "../lib/bus";
import { formatDayTime } from "../lib/format";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError,
  discoverTrack2Dataset,
  getTrack2Models,
  getTrack2LastPull,
  getTrack2PullControl,
  saveTrack2PullConfig,
  type Track2DiscoveredChild,
} from "../lib/api";
import { Card } from "./Card";
import { Track2RetentionPanel } from "./Track2RetentionPanel";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

/** How a week folder is shaped underneath. Detected, not asked for --
 * the operator picked a root, and which of these it is is a fact about
 * that root rather than a decision anyone should have to restate.
 *
 *   split   week_1/images/train  labels/train
 *           week_1/images/val    labels/val
 *   flat    week_1/images        labels
 */
type Layout = "split" | "flat";

/** One week folder, both splits together. */
interface WeekRow {
  name: string;
  train: Track2DiscoveredChild | null;
  val: Track2DiscoveredChild | null;
}

/** week_9 before week_10. A plain sort puts week_10 second, which reads
 * as a missing week to anybody scanning the column. */
function naturally(a: string, b: string): number {
  const na = /\d+$/.exec(a);
  const nb = /\d+$/.exec(b);
  if (na && nb && a.slice(0, a.length - na[0].length) === b.slice(0, b.length - nb[0].length)) {
    return Number(na[0]) - Number(nb[0]);
  }
  return a.localeCompare(b);
}

/**
 * Assembles a dataset out of the week folders the pull control fills.
 *
 *     retraining/                 <- the one root
 *       week_1/  images/train  images/val  labels/train  labels/val
 *       week_2/  images/train  images/val  labels/train  labels/val
 *
 * One root, because that is now the only shape the pull control writes:
 * train and val live inside each week, not in two parallel trees. This
 * card used to ask for a training root and an evaluation root separately,
 * which meant two pickers that had to be kept pointing at the same place
 * and nothing checking that they were -- pointing the evaluation side at
 * the training root is train/val leakage, and it said nothing.
 *
 * Every week is included unless ticked out, so the week created next
 * month is picked up with no action at all. That is the failure this
 * removes: a retrain that quietly misses the newest data because somebody
 * forgot to add a folder.
 */
export function Track2AutoTrainingPanel() {
  // Blank means "not set yet, use the name below it". Seeded from the
  // pull control so the two cards agree without anybody typing twice.
  const [root, setRoot] = useState("");
  // Read from the pull control rather than typed here. They are the same
  // two names the pull writes with, so a second place to set them was a
  // second place for them to be wrong -- and a mismatch shows up as a
  // root that appears to hold nothing.
  const [imagesSub, setImagesSub] = useState("");
  const [labelsSub, setLabelsSub] = useState("");
  const imagesName = imagesSub || "images";
  const labelsName = labelsSub || "labels";

  const [layout, setLayout] = useState<Layout | null>(null);
  const [trainChildren, setTrainChildren] = useState<Track2DiscoveredChild[] | null>(null);
  const [valChildren, setValChildren] = useState<Track2DiscoveredChild[] | null>(null);
  const [looking, setLooking] = useState(false);

  const [excluded, setExcluded] = useState<string[]>([]);
  // False until the saved selection has been read. Writing before then
  // would persist this card's empty starting state over whatever was
  // saved, which is a silent way to lose it.
  const [loadedSelection, setLoadedSelection] = useState(false);
  /** Flat roots only: which side each folder feeds. Nothing defaults to
   * val -- a val set chosen for you is a val set nobody checked. */
  const [roles, setRoles] = useState<Record<string, "train" | "val">>({});
  const [expanded, setExpanded] = useState<string | null>(null);

  const [picking, setPicking] = useState(false);
  const [pickPrefix, setPickPrefix] = useState("");
  const [pickFolders, setPickFolders] = useState<string[]>([]);

  const [error, setError] = useState("");
  /** The pull control's own root -- the folder it writes week folders
   * into, and the one the unattended chain reads. Held so the card can
   * follow it, and can say what it would snap back to. */
  const [pullRoot, setPullRoot] = useState("");
  /** True while the root is the pull's. Cleared the moment somebody
   * picks a different folder, so an override is a decision rather than
   * a drift. */
  const [followPull, setFollowPull] = useState(true);

  // Start, and STAY, where the pull control is writing.
  //
  // This used to seed the root once and then let the card's own saved
  // value win forever. Change the pull's root afterwards and this card
  // went on showing the old one -- while the unattended chain, which
  // reads the pull's `root` directly, used the new one. Two folders, one
  // of them on screen, nothing saying they differed.
  useEffect(() => {
    getTrack2PullControl()
      .then((s) => {
        const c = s.config;
        const saved = c.auto_training || {};
        const live = c.root || c.training_root || "";
        setPullRoot(live);
        // Absent means follow: a config written before this existed was
        // following in spirit, and defaulting the other way would strand
        // every one of them on a root nobody chose.
        const follows = saved.follow_pull !== false;
        setFollowPull(follows);
        setRoot((cur) => cur || (follows ? live : saved.root || live));
        setExcluded((cur) => (cur.length ? cur : saved.excluded || []));
        setImagesSub((cur) => cur || c.images_subpath || "");
        setLabelsSub((cur) => cur || c.labels_subpath || "");
        setLoadedSelection(true);
      })
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!picking) return;
    let cancelled = false;
    getTrack2Models(pickPrefix, undefined, undefined)
      .then((res) => !cancelled && setPickFolders(res.dataset_browser.folders))
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [picking, pickPrefix]);

  // Kept server-side so it rides along in the settings file. Debounced,
  // because ticking three weeks out is three changes and the first two
  // are not worth a write.
  useEffect(() => {
    if (!loadedSelection) return;
    const timer = window.setTimeout(() => {
      saveTrack2PullConfig({
        auto_training: { root, excluded, follow_pull: followPull },
      })
        // Which weeks are ticked is what a run trains on.
        .then(() => post("data-sources-changed", { from: "weeks" }))
        .catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [loadedSelection, root, excluded, followPull]);

  // Out-of-order responses: changing the root twice quickly used to leave
  // the first root's weeks on screen under the second root's name.
  const ticketRef = useRef(0);

  const rediscover = useCallback(async () => {
    const ticket = ticketRef.current + 1;
    ticketRef.current = ticket;
    if (!root) {
      setLayout(null);
      setTrainChildren(null);
      setValChildren(null);
      return;
    }
    setLooking(true);
    setError("");
    try {
      // Ask for the split shape first. It is what the pull control
      // writes, so on this platform it is the answer almost every time.
      const [t, v] = await Promise.all([
        discoverTrack2Dataset(root, `${imagesName}/train`, `${labelsName}/train`)
          .catch(() => null),
        discoverTrack2Dataset(root, `${imagesName}/val`, `${labelsName}/val`)
          .catch(() => null),
      ]);
      if (ticketRef.current !== ticket) return;
      const anySplit = (t?.children ?? []).some((c) => c.image_count > 0)
        || (v?.children ?? []).some((c) => c.image_count > 0);
      if (anySplit) {
        setLayout("split");
        setTrainChildren(t?.children ?? []);
        setValChildren(v?.children ?? []);
        return;
      }
      const flat = await discoverTrack2Dataset(root, imagesName, labelsName)
        .catch((exc) => {
          // Said rather than swallowed. All three probes returning
          // nothing looks identical to an empty root, and "this folder
          // is empty" sends you somewhere quite different from "that
          // read was refused".
          if (ticketRef.current === ticket) {
            setError(exc instanceof ApiError
              ? exc.detail
              : `Could not read ${root}.`);
          }
          return null;
        });
      if (ticketRef.current !== ticket) return;
      setLayout("flat");
      setTrainChildren(flat?.children ?? []);
      setValChildren(null);
    } finally {
      if (ticketRef.current === ticket) setLooking(false);
    }
  }, [root, imagesName, labelsName]);

  useEffect(() => { void rediscover(); }, [rediscover]);

  /** The last finished sweep this card has already accounted for.
   *
   * A pull is what CREATES the week folders this card lists, so until
   * now the card showed whatever was there when it mounted and said
   * nothing when that changed: run a pull, come back, and it still read
   * "No folders directly under retraining" until you pressed Reload.
   * Worse for a scheduled pull, where nobody is on the page to press it.
   *
   * Polled rather than pushed because there is nothing to push with, and
   * the endpoint is one DynamoDB query by design -- the full status
   * response lists S3 and is far too heavy for this.
   */
  const seenPullRef = useRef<string | null>(null);
  const [lastPull, setLastPull] = useState<{ at: string; moved: number; status: string } | null>(null);
  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const tick = async () => {
      try {
        const got = await getTrack2LastPull();
        if (cancelled) return;
        if (got.finished_at) {
          setLastPull({ at: got.finished_at, moved: got.moved, status: got.status });
          // First poll only RECORDS where we are. Re-discovering on it
          // would double every mount's work for a pull that finished
          // days ago.
          if (seenPullRef.current === null) {
            seenPullRef.current = got.finished_at;
          } else if (seenPullRef.current !== got.finished_at) {
            seenPullRef.current = got.finished_at;
            void rediscover();
          }
        }
      } catch {
        // A poll that fails is not worth an error banner; the next one
        // is twenty seconds away and Reload still works.
      }
      if (!cancelled) timer = window.setTimeout(tick, 20000);
    };
    void tick();
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [rediscover]);

  /** Both splits lined up by folder name, so one row is one week. */
  const rows = useMemo<WeekRow[]>(() => {
    const byName = new Map<string, WeekRow>();
    for (const c of trainChildren ?? []) {
      byName.set(c.name, { name: c.name, train: c, val: null });
    }
    for (const c of valChildren ?? []) {
      const row = byName.get(c.name);
      if (row) row.val = c;
      else byName.set(c.name, { name: c.name, train: null, val: c });
    }
    return [...byName.values()].sort((a, b) => naturally(a.name, b.name));
  }, [trainChildren, valChildren]);

  const totals = useMemo(() => {
    let train = 0;
    let val = 0;
    for (const r of rows) {
      train += r.train?.image_count ?? 0;
      val += r.val?.image_count ?? 0;
    }
    return { train, val };
  }, [rows]);

  function usable(row: WeekRow): boolean {
    return (row.train?.image_count ?? 0) + (row.val?.image_count ?? 0) > 0;
  }

  function included(row: WeekRow): boolean {
    return usable(row) && !excluded.includes(row.name);
  }

  /** The four (or two) prefixes a week actually holds, with what is in
   * them. Folded away by default -- it is the answer to "why is this week
   * not being used", which is a question asked about one week at a time,
   * not about all of them at once. */
  function renderDetail(row: WeekRow) {
    const parts: { label: string; child: Track2DiscoveredChild | null }[] =
      layout === "split"
        ? [
          { label: `${imagesName}/train + ${labelsName}/train`, child: row.train },
          { label: `${imagesName}/val + ${labelsName}/val`, child: row.val },
        ]
        : [{ label: `${imagesName} + ${labelsName}`, child: row.train }];
    return (
      <div className="border-t border-hairline bg-page/60 px-2 py-1.5">
        {parts.map((part) => (
          <div key={part.label} className="tabular text-[11px] text-ink-secondary">
            <span className="text-ink-muted">{part.label}</span>
            {part.child ? (
              <>
                {" -- "}
                {part.child.image_count.toLocaleString()} images,{" "}
                {part.child.label_count.toLocaleString()} labels,{" "}
                {part.child.paired.toLocaleString()} paired
                <div className="truncate text-ink-muted" title={part.child.images_prefix}>
                  {part.child.images_prefix}
                </div>
                <div className="truncate text-ink-muted" title={part.child.labels_prefix}>
                  {part.child.labels_prefix}
                </div>
              </>
            ) : (
              <span className="text-ink-muted"> -- not present</span>
            )}
          </div>
        ))}
      </div>
    );
  }

  return (
    <Card
      title="Auto Training"
      step="4"
      help={
        <>
          <p>
            Shows what is actually in the week folders the pull filled: how many
            images each week holds on the train and val sides, and which ones
            are empty. Tick a week out to leave it behind.
          </p>
          <p className="mt-2">
            Every week is included unless you tick it out, so the week created
            next month is picked up with no action at all. That is the point of
            the card -- a retrain that quietly misses the newest data because
            somebody forgot to add a folder is the failure it removes.
          </p>
          <p className="mt-2">
            Nothing is copied here. The dataset itself is described by the{" "}
            <span className="tabular">data.yaml</span> in the Retention section
            below, which names these folders where they already are.
          </p>
        </>
      }
      next="Retention, just below -- decide how much of each older week to use, then build the data.yaml."
      caption="One root of week folders, and what each one holds. The dataset itself is built from the data.yaml below."
    >

      {error && <p className="mb-2 text-xs text-critical">{error}</p>}

      {/* The root, and the two names that shape a week inside it. */}
      <div>
        <div className="min-w-0">
          <span className="mb-1 block text-xs text-ink-muted">
            Root folder{" "}
            <span className="text-ink-muted/70">
              (holds the week folders; images/ and labels/ come from the pull
              control)
            </span>
          </span>
          {/* Which folder this is, and whether it is the one the automated
              run will use. Silence here is what let the card and the chain
              drift apart: the chain reads the pull's root whatever this
              card shows. */}
          {followPull ? (
            <p className="mb-1 text-xs text-ink-muted">
              Following the pull control
              {pullRoot && (
                <> &mdash; <span className="tabular font-medium text-ink">{pullRoot}</span></>
              )}
              . Change it there and this follows.
            </p>
          ) : (
            <p className="mb-1 text-xs text-critical">
              Overridden. The automated run still uses the pull control&rsquo;s
              root{pullRoot ? " " : ""}
              {pullRoot && <span className="tabular font-medium">{pullRoot}</span>}
              , so what you see here is not what it will train on.{" "}
              <button
                type="button"
                onClick={() => {
                  setRoot(pullRoot);
                  setFollowPull(true);
                  setExcluded([]);
                  setRoles({});
                  setExpanded(null);
                }}
                className="font-medium underline"
              >
                Follow the pull control again
              </button>
            </p>
          )}
          <div className="flex gap-2">
            <button
              type="button"
              onClick={() => { setPicking((v) => !v); setPickPrefix(""); }}
              className="tabular min-w-0 flex-1 truncate rounded-md border border-black/15 px-3 py-2 text-left text-xs"
              title={root || undefined}
            >
              {root || "Choose the root folder..."}
            </button>
            <button
              type="button"
              onClick={() => void rediscover()}
              disabled={!root || looking}
              className="shrink-0 rounded-md border border-black/15 px-3 py-2 text-xs font-medium text-ink-secondary disabled:opacity-40"
            >
              {looking ? "Looking..." : "Reload"}
            </button>
          </div>
          {/* What the list below is as of. Without it, "no folders" reads
              the same whether the pull has never run or ran and moved
              nothing -- two different problems with two different fixes. */}
          {lastPull && (
            <p className="tabular mt-1 text-xs text-ink-muted">
              Last pull {formatDayTime(lastPull.at)} &mdash;{" "}
              {lastPull.status === "done"
                ? `${lastPull.moved} file${lastPull.moved === 1 ? "" : "s"} moved`
                : lastPull.status}
              . This list refreshes itself when one finishes.
            </p>
          )}
        </div>
      </div>

      {picking && (
        <div className="mt-2 rounded-md border border-accent/40 p-3">
          <p className="mb-1.5 text-xs text-ink-secondary">
            Choose the folder that CONTAINS the week folders
          </p>
          <Track2S3FolderBrowser
            prefix={pickPrefix}
            folders={pickFolders}
            onNavigate={setPickPrefix}
            onSelect={(p) => {
              setRoot(p);
              // Choosing by hand is an override unless it happens to land
              // back on the pull's own root, in which case there is
              // nothing to override and the card goes back to following.
              setFollowPull(p === pullRoot);
              setExcluded([]);
              setRoles({});
              setExpanded(null);
              setPicking(false);
            }}
            selected={root}
          />
          <button
            type="button"
            onClick={() => setPicking(false)}
            className="mt-2 rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
          >
            Cancel
          </button>
        </div>
      )}

      {/* What is inside the root. One row per week folder, every split it
          holds -- the same shape as the pull control's weeks table, since
          the two cards are looking at the same folders and should not
          need translating between. */}
      {root && (
        <div className="mt-3">
          {layout === null ? (
            <p className="text-xs text-ink-muted">Looking inside {root}...</p>
          ) : rows.length === 0 ? (
            <p className="text-xs text-critical">
              No folders directly under {root}.
            </p>
          ) : (
            <>
              <p className="mb-1.5 text-xs text-ink-muted">
                {rows.length} folder{rows.length === 1 ? "" : "s"} under{" "}
                <span className="tabular text-ink">{root}</span>
                {layout === "split" ? (
                  <>
                    {" "}-- each holds{" "}
                    <span className="tabular">{imagesName}/train</span>,{" "}
                    <span className="tabular">{imagesName}/val</span>,{" "}
                    <span className="tabular">{labelsName}/train</span>{" and "}
                    <span className="tabular">{labelsName}/val</span>
                  </>
                ) : (
                  <>
                    {" "}-- each holds{" "}
                    <span className="tabular">{imagesName}</span>{" and "}
                    <span className="tabular">{labelsName}</span>, with no
                    train/val split inside, so say which side each one feeds
                  </>
                )}
              </p>

              <div className="overflow-hidden rounded-md border border-hairline">
                <div className="flex items-center gap-2 border-b border-hairline bg-page px-2 py-1.5 text-[11px] font-medium text-ink-muted">
                  <span className="w-6 shrink-0" />
                  <span className="min-w-0 flex-1">folder</span>
                  {layout === "split" ? (
                    <>
                      <span className="w-16 shrink-0 text-right">train</span>
                      <span className="w-16 shrink-0 text-right">val</span>
                    </>
                  ) : (
                    <span className="w-16 shrink-0 text-right">images</span>
                  )}
                  <span className="w-16 shrink-0 text-right">paired</span>
                  <span className="w-24 shrink-0 pl-2">
                    {layout === "split" ? "state" : "feeds"}
                  </span>
                </div>

                {rows.map((row) => {
                  const empty = !usable(row);
                  const on = included(row);
                  const paired = (row.train?.paired ?? 0) + (row.val?.paired ?? 0);
                  return (
                    <div key={row.name} className="border-b border-hairline last:border-b-0">
                      <div
                        className={`flex items-center gap-2 px-2 py-1.5 text-xs ${
                          empty ? "text-ink-muted" : "text-ink-secondary"
                        }`}
                      >
                        <input
                          type="checkbox"
                          className="w-6 shrink-0"
                          checked={on}
                          disabled={empty}
                          aria-label={`Use ${row.name}`}
                          onChange={() =>
                            setExcluded((current) => (current.includes(row.name)
                              ? current.filter((n) => n !== row.name)
                              : [...current, row.name]))
                          }
                        />
                        <button
                          type="button"
                          onClick={() => setExpanded((c) => (c === row.name ? null : row.name))}
                          className="tabular min-w-0 flex-1 truncate text-left font-medium text-ink"
                          title="Show the images and labels folders inside"
                        >
                          {expanded === row.name ? "▾" : "▸"} {row.name}
                        </button>
                        {layout === "split" ? (
                          <>
                            <span className="tabular w-16 shrink-0 text-right">
                              {(row.train?.image_count ?? 0).toLocaleString()}
                            </span>
                            <span className="tabular w-16 shrink-0 text-right">
                              {(row.val?.image_count ?? 0).toLocaleString()}
                            </span>
                          </>
                        ) : (
                          <span className="tabular w-16 shrink-0 text-right">
                            {(row.train?.image_count ?? 0).toLocaleString()}
                          </span>
                        )}
                        <span className="tabular w-16 shrink-0 text-right">
                          {paired.toLocaleString()}
                        </span>
                        <span className="w-24 shrink-0 pl-2">
                          {layout === "split" ? (
                            empty ? "empty" : on ? "in use" : "left out"
                          ) : (
                            <select
                              className="w-full rounded border border-black/15 bg-page px-1 py-0.5 text-[11px]"
                              value={roles[row.name] ?? "train"}
                              disabled={empty || !on}
                              onChange={(e) => setRoles((c) => ({
                                ...c, [row.name]: e.target.value as "train" | "val",
                              }))}
                            >
                              <option value="train">train</option>
                              <option value="val">val</option>
                            </select>
                          )}
                        </span>
                      </div>
                      {expanded === row.name && renderDetail(row)}
                    </div>
                  );
                })}

                <div className="flex items-center gap-2 border-t border-hairline bg-page px-2 py-1.5 text-[11px] text-ink-muted">
                  <span className="w-6 shrink-0" />
                  <span className="min-w-0 flex-1">everything in the root</span>
                  {layout === "split" ? (
                    <>
                      <span className="tabular w-16 shrink-0 text-right">
                        {totals.train.toLocaleString()}
                      </span>
                      <span className="tabular w-16 shrink-0 text-right">
                        {totals.val.toLocaleString()}
                      </span>
                    </>
                  ) : (
                    <span className="tabular w-16 shrink-0 text-right">
                      {totals.train.toLocaleString()}
                    </span>
                  )}
                  <span className="w-16 shrink-0" />
                  <span className="w-24 shrink-0 pl-2" />
                </div>
              </div>
            </>
          )}
        </div>
      )}

      {/* Retention, in the card that uses it. It decides how much of each
          older week the assembly above takes, so reading it anywhere else
          meant scrolling away from the folder list it is describing. */}
      <div className="mt-4 border-t border-hairline pt-4">
        <Track2RetentionPanel />
      </div>
    </Card>
  );
}
