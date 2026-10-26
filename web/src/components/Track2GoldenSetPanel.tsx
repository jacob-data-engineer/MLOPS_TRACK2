import { useCallback, useEffect, useState } from "react";
import { useCardOpen } from "../lib/cardState";
import { announceGolden, onGoldenChanged } from "../lib/useGoldenSync";
import {
  ApiError,
  buildTrack2Split,
  getTrack2Buckets,
  getTrack2PullControl,
  getTrack2Retention,
  previewTrack2Split,
  saveTrack2PullConfig,
  checkTrack2Pairing,
  listTrack2Folders,
  saveTrack2Retention,
  type Track2Bucket,
  type Track2SplitPlan,
  type Track2SplitResult,
} from "../lib/api";
import { Card } from "./Card";
// Both, deliberately. The SOURCE pair is the Assemble card's shape --
// collapsed fields with one browser opened under whichever was clicked.
// The TARGET folder keeps the always-open field: it is one folder, not a
// pair, and it is the one picker here that may create a folder that does
// not exist yet.
import { S3FolderField } from "./S3FolderField";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

/** Buckets the stack grants writes to. Mirrors `dataset_write_buckets` in
 * terraform/app; a bucket in neither simply cannot be chosen. */
const WRITABLE = ["track2-s3", "safety-model-dataset-dev", "safety-model-dataset"];

/**
 * Split a folder you curated into train and val.
 *
 * This card decides nothing about WHAT belongs in the set -- that was
 * your judgement, made before you got here. All it does is divide the
 * folder, copy it where you ask, and be honest about what it did.
 *
 * Two properties it guarantees, both of which fail silently if broken:
 * an image and its label always land on the same side, and the same
 * folder always splits the same way.
 */
export function Track2GoldenSetPanel() {
  const [open, setOpen] = useCardOpen("golden_set", false);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");

  const [bucket, setBucket] = useState("track2-s3");
  const [buckets, setBuckets] = useState<Track2Bucket[]>([]);
  const [source, setSource] = useState("");
  /** Where the labels are, when they are not under `source`.
   *
   * Empty is the normal case and keeps the two layouts this card always
   * read -- a folder holding images/ and labels/, and a flat folder with
   * both in it. Filled, the two are listed separately. */
  const [labelsSource, setLabelsSource] = useState("");
  /** Which of the two fields a folder click applies to. ONE browser on
   * screen rather than one per field, which is the Assemble card's
   * shape -- two always-open lists side by side were two things to read
   * before either had been answered. */
  const [picking, setPicking] = useState<"images" | "labels" | null>(null);
  const [pickPrefix, setPickPrefix] = useState("");
  const [pickFolders, setPickFolders] = useState<string[]>([]);
  /** Whether the two folders describe each other. A label is matched to
   * its image by STEM, so independently chosen folders can look right
   * and pair with nothing. */
  const [pairing, setPairing] = useState<Awaited<
    ReturnType<typeof checkTrack2Pairing>> | null>(null);
  const [pairingBusy, setPairingBusy] = useState(false);

  const [targetBucket, setTargetBucket] = useState("track2-s3");
  const [targetFolder, setTargetFolder] = useState("golden_set");
  const [split, setSplit] = useState(true);
  const [trainPercent, setTrainPercent] = useState(80);

  // A preview or a build result -- the latter is the former plus what it
  // did, so one slot holds whichever came back last.
  const [plan, setPlan] = useState<Track2SplitPlan | Track2SplitResult | null>(null);

  /** Whether a golden set is used at all.
   *
   * ONE field, `use_golden`, which Retention already owned and still
   * stores -- this card is just where it is now set, because it is the
   * card named after the thing. A second checkbox writing the same field
   * from another card is how two panels end up disagreeing about one
   * setting, which this page has been bitten by more than once.
   *
   * `null` while it is still being read, so the card does not flash
   * "not in use" at somebody who is using one.
   */
  const [inUse, setInUse] = useState<boolean | null>(null);
  const [switching, setSwitching] = useState(false);

  useEffect(() => {
    getTrack2Retention()
      .then((r) => setInUse(Boolean(r.config.use_golden)))
      .catch(() => setInUse(false));
  }, []);

  // The same switch is in the three-source row on the Assemble
  // card. Without this the two sit on one page showing different
  // answers until a reload.
  useEffect(() => onGoldenChanged(setInUse), []);

  /** This card's own choices live server-side, like every other card's.
   *
   * They used to be browser-tab state and nothing else, so saving a
   * settings file recorded the golden set's on/off switch and none of
   * the folders it pointed at -- the file looked complete and restored a
   * card with every field blank.
   *
   * `loaded` gates the writer below: without it the first render would
   * save the initial defaults straight over whatever was stored, before
   * the read had come back. */
  const [loaded, setLoaded] = useState(false);
  useEffect(() => {
    getTrack2PullControl()
      .then((s) => {
        const g = s.config.golden_set || {};
        if (g.bucket) setBucket(g.bucket);
        if (g.source) setSource(g.source);
        if (g.labels_source) setLabelsSource(g.labels_source);
        if (g.target_bucket) setTargetBucket(g.target_bucket);
        if (g.target_folder) setTargetFolder(g.target_folder);
        if (typeof g.split === "boolean") setSplit(g.split);
        if (typeof g.train_percent === "number") setTrainPercent(g.train_percent);
      })
      .catch(() => undefined)
      .finally(() => setLoaded(true));
  }, []);

  // The folders under whatever the open picker is looking at. One
  // listing, because only one picker is open at a time.
  useEffect(() => {
    if (!picking) return;
    let cancelled = false;
    setPickFolders([]);
    listTrack2Folders(bucket, pickPrefix)
      .then((res) => !cancelled && setPickFolders(res.folders))
      .catch(() => !cancelled && setPickFolders([]));
    return () => { cancelled = true; };
  }, [picking, bucket, pickPrefix]);

  // Do the two folders describe each other? The same check the Assemble
  // card runs, against the same endpoint -- a second implementation
  // would be a second answer.
  useEffect(() => {
    const images = source.trim();
    const labels = labelsSource.trim();
    if (!images || !labels) { setPairing(null); return; }
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setPairingBusy(true);
      checkTrack2Pairing(bucket, [images], [labels])
        .then((res) => !cancelled && setPairing(res))
        .catch(() => !cancelled && setPairing(null))
        .finally(() => !cancelled && setPairingBusy(false));
    }, 700);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [bucket, source, labelsSource]);

  // Debounced: dragging the split slider is one decision, not forty.
  useEffect(() => {
    if (!loaded) return;
    const timer = window.setTimeout(() => {
      saveTrack2PullConfig({
        golden_set: {
          bucket, source, labels_source: labelsSource,
          target_bucket: targetBucket,
          target_folder: targetFolder, split, train_percent: trainPercent,
        },
      }).catch(() => undefined);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [loaded, bucket, source, labelsSource, targetBucket, targetFolder,
      split, trainPercent]);

  async function setUse(next: boolean) {
    setSwitching(true);
    setError("");
    try {
      // A PATCH, so the folder and bucket survive being switched off --
      // turning it back on must not mean finding the path again.
      const out = await saveTrack2Retention({ use_golden: next });
      setInUse(Boolean(out.config.use_golden));
      announceGolden(Boolean(out.config.use_golden));
      // Switching it ON opens the card, so the folder picker it needs is
      // on screen. The card remembers being shut, and saying "use a
      // golden set" to a collapsed card left the one thing it then wants
      // -- which folder -- behind a second click on Show.
      //
      // Only on the way ON. Turning it off and having the card fold away
      // would hide the settings the patch above deliberately keeps.
      if (next) setOpen(true);
      setNote(next
        ? "In use -- the val side is the fixed evaluation set, and the train "
          + "side is trained on in full and never decays."
        : "Not in use -- nothing downstream expects a golden set. The folder "
          + "and split settings are kept.");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not change that.");
    } finally {
      setSwitching(false);
    }
  }

  useEffect(() => {
    if (!open) return;
    getTrack2Buckets().then((r) => setBuckets(r.buckets)).catch(() => undefined);
  }, [open]);

  const request = useCallback(() => ({
    source, labels_source: labelsSource,
    bucket, target_bucket: targetBucket, target_folder: targetFolder,
    split, train_percent: trainPercent,
  }), [source, labelsSource, bucket, targetBucket, targetFolder, split,
       trainPercent]);

  async function withBusy(label: string, work: () => Promise<void>) {
    setBusy(label);
    setError("");
    setNote("");
    try {
      await work();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "That did not work.");
    } finally {
      setBusy("");
    }
  }

  // The slider is about PAIRS, so both numbers are shown -- "80%" alone
  // leaves you working out what the other side gets.
  const valPercent = 100 - trainPercent;

  return (
    // collapsible={false}: this card runs its own Show/Hide,
    // because it shows a summary of live state when shut and
    // the generic one in Card cannot know what to put there.
    <Card
      title="Pick Golden Set / Edge Cases"
      step="2"
      titleAside={
        // Beside the title, not under the caption. This says whether the
        // card is in play at all, so it belongs with the card's name --
        // and it is the same question, asked the same way, as the three
        // ticks in the Assemble card's heading.
        <label className="ml-2 inline-flex items-center gap-1.5 text-xs font-normal text-ink-secondary">
          <input
            type="checkbox"
            checked={!!inUse}
            disabled={switching || inUse === null}
            onChange={(e) => void setUse(e.target.checked)}
            className="h-3.5 w-3.5"
          />
          Use Pick Golden Set / Edge Cases
        </label>
      }
      help={
        <>
          <p>
            <strong>Optional.</strong> A golden set is a folder of images you
            chose by hand -- the hard cases, the ones a model must not get
            wrong. This card splits that folder into a train and a val side.
          </p>
          <p className="mt-2">
            The val side becomes a fixed evaluation set, which is what makes
            one run&apos;s numbers comparable with the last. The train side can be
            switched on in Retention, where it is trained on in full and never
            decays.
          </p>
          <p className="mt-2">
            Skip this if you have not curated anything yet. Nothing below
            depends on it.
          </p>
        </>
      }
      next="Step 3, Pull Control -- collect the images into dated week folders."
      collapsible={false}
      caption="Split a folder you have already curated into train and val. Images and their labels always stay together."
      action={
        <button type="button" onClick={() => setOpen((v) => !v)}
                className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary">
          {open ? "Hide" : "Show"}
        </button>
      }
    >

      {/* The tick itself is up in the heading. What stays here is what
          the tick MEANS, which is too long for a title row and is only
          worth saying when the answer is no. */}
      {inUse === false && (
        <p className="mb-2 text-xs text-ink-muted">
          Step skipped. Training uses whichever other sources are ticked.
        </p>
      )}

      {!open ? (
        <p className="tabular text-xs text-ink-muted">
          {inUse === false
            ? "Not in use."
            : source ? `${source} -> ${targetFolder}` : "Nothing chosen yet."}
        </p>
      ) : (
        <div
          className={`flex flex-col gap-3 ${
            inUse === false ? "pointer-events-none opacity-40" : ""
          }`}
        >
          {error && <p className="text-xs text-critical">{error}</p>}
          {note && <p className="text-xs text-ink-secondary">{note}</p>}

          {/* ------------------------------------------- the source ----

              The bucket spans the full width above both pickers, and the
              two pickers sit side by side under it. That order is the
              order of the decisions: one bucket holds both folders, so
              asking for it once at the top says so, where a bucket in a
              half-width column beside a folder list read as belonging to
              that one picker.

              Side by side is what makes the pair legible as a pair --
              the same shape the Assemble card uses for its Images and
              Labels. They stack below the `sm` breakpoint, where two
              scrolling lists would each be too narrow to read a folder
              name in. */}
          <div className="grid gap-2">
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Source bucket</span>
              <select className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                      value={bucket} onChange={(e) => { setBucket(e.target.value); setSource(""); }}>
                {(buckets.length ? buckets.map((b) => b.name) : [bucket]).map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
            {/* The Assemble card's pair row, the same shape and the
                same behaviour: a collapsed field per folder, ONE
                browser opened underneath whichever was clicked, and a
                pairing check under both.

                Two always-open browsers side by side were two scrolling
                lists to read before either question had been answered,
                and neither said whether the folders described each
                other. */}
            <div className="rounded-md border border-hairline bg-page p-2.5">
              <div className="grid gap-2 sm:grid-cols-2">
                {([
                  ["images", "Images", source, setSource] as const,
                  ["labels", "Labels", labelsSource, setLabelsSource] as const,
                ]).map(([field, label, value, set]) => {
                  const active = picking === field;
                  return (
                    <div key={field} className="min-w-0">
                      <span className="mb-1 block text-xs text-ink-muted">
                        {label}
                        {field === "labels" && (
                          <span className="text-ink-muted/70">
                            {" "}(optional -- only when the labels are not under
                            the images folder)
                          </span>
                        )}
                      </span>
                      <button
                        type="button"
                        onClick={() => {
                          setPicking(active ? null : field);
                          setPickPrefix(value.trim().replace(/\/+$/, ""));
                        }}
                        className={`tabular w-full truncate rounded-md border px-3 py-2 text-left text-xs ${
                          active ? "border-accent bg-accent-subtle" : "border-black/15"
                        }`}
                        title={value || undefined}
                      >
                        {value || "Choose a folder..."}
                      </button>
                      {/* Always shown, like the Assemble card's: a
                          control that appears and disappears is one you
                          have to hunt for, and on an empty field it
                          simply does nothing. */}
                      <button
                        type="button"
                        onClick={() => { set(""); if (active) setPicking(null); }}
                        className="mt-1 text-xs font-medium text-critical underline"
                      >
                        Remove this folder
                      </button>
                    </div>
                  );
                })}
              </div>

              {picking && (
                <div className="mt-2">
                  <Track2S3FolderBrowser
                    prefix={pickPrefix}
                    folders={pickFolders}
                    onNavigate={setPickPrefix}
                    onSelect={(chosen) => {
                      if (picking === "images") setSource(chosen);
                      else setLabelsSource(chosen);
                      setPicking(null);
                    }}
                    selected={picking === "images" ? source : labelsSource}
                  />
                </div>
              )}

              {/* The same four answers the Assemble card gives, from the
                  same endpoint. An image with no label is a background
                  example and fine; a label with no image never is. */}
              {pairingBusy ? (
                <p className="mt-1.5 text-xs text-ink-muted">Checking the pairing...</p>
              ) : pairing?.checked ? (
                pairing.paired === 0 ? (
                  <p className="mt-1.5 text-xs text-critical">
                    <strong>Nothing pairs.</strong>{" "}
                    {pairing.images.toLocaleString()} images and{" "}
                    {pairing.labels.toLocaleString()} labels, and not one filename
                    matches. The usual cause is one folder pointing at train and
                    the other at val.
                  </p>
                ) : pairing.labels_without_images > 0 ? (
                  <p className="mt-1.5 text-xs text-critical">
                    <strong>
                      {pairing.labels_without_images.toLocaleString()} labels have
                      no image.
                    </strong>{" "}
                    They describe nothing and never reach the trainer.
                  </p>
                ) : pairing.images_without_labels === pairing.images
                  && pairing.images > 0 ? (
                  <p className="mt-1.5 text-xs text-critical">
                    <strong>No image has a label.</strong> The labels folder is
                    almost certainly the wrong one.
                  </p>
                ) : (
                  <p className="mt-1.5 text-xs text-ink-secondary">
                    {pairing.paired.toLocaleString()} of{" "}
                    {pairing.images.toLocaleString()} images are labelled.
                  </p>
                )
              ) : null}
            </div>
          </div>

          {/* --------------------------------------------- the split ---- */}
          <div className="rounded-md border border-hairline p-3">
            <label className="mb-2 flex items-center gap-2 text-xs font-medium text-ink">
              <input type="checkbox" checked={split}
                     onChange={(e) => setSplit(e.target.checked)} />
              Split into train and val
            </label>

            {split ? (
              <>
                <input
                  type="range" min={5} max={95} step={5}
                  value={trainPercent}
                  onChange={(e) => setTrainPercent(Number(e.target.value))}
                  className="w-full accent-[var(--color-accent)]"
                />
                <div className="tabular mt-1 flex justify-between text-xs">
                  <span className="font-medium text-ink">train {trainPercent}%</span>
                  <span className="text-ink-muted">val {valPercent}%</span>
                </div>
                {/* Said out loud because it is the thing that goes wrong:
                    a percentage of FILES would put an image in train and
                    its label in val. */}
                <p className="mt-1.5 text-xs text-ink-muted">
                  The percentage is of image/label PAIRS, not of files -- each
                  image goes wherever its label goes. The division comes from a
                  hash of the filename, so the same folder always splits the same
                  way and a build can be repeated exactly.
                </p>
              </>
            ) : (
              <p className="text-xs text-ink-muted">
                Off: the folder is copied whole into{" "}
                <span className="tabular">{targetFolder || "<target>"}/images/</span> and{" "}
                <span className="tabular">labels/</span>, with no train/val level.
              </p>
            )}
          </div>

          {/* -------------------------------------------- the target ---- */}
          <div className="grid gap-2 sm:grid-cols-2">
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Target bucket</span>
              <select className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                      value={targetBucket} onChange={(e) => setTargetBucket(e.target.value)}>
                {WRITABLE.map((name) => (
                  <option key={name} value={name}>{name}</option>
                ))}
              </select>
            </label>
            <S3FolderField
              label="Target folder"
              hint="(where the split is written, e.g. golden_set)"
              value={targetFolder}
              onChange={setTargetFolder}
              bucket={targetBucket}
              allowCreate
            />
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button type="button" disabled={!!busy || !source}
              onClick={() => withBusy("Looking...", async () => {
                const got = await previewTrack2Split(request());
                setPlan(got);
                setNote(`${got.pairs.toLocaleString()} pair(s) found.`);
              })}
              className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40">
              Preview
            </button>
            <button type="button" disabled={!!busy || !plan?.ready}
              onClick={() => withBusy("Copying...", async () => {
                const got = await buildTrack2Split(request());
                setPlan(got);
                setNote(`Copied ${got.copied.toLocaleString()} files to ${got.uri}`
                  + `${got.errors.length ? ` -- ${got.errors.length} failed` : ""}.`);
              })}
              className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40">
              {busy || "Build the split"}
            </button>
          </div>

          {plan && (
            <div className="rounded-md border border-black/15 bg-page p-3">
              {plan.problems.map((problem) => (
                <p key={problem.text}
                   className={`mb-1 text-xs ${
                     problem.level === "blocking" ? "text-critical" : "text-ink-secondary"
                   }`}>
                  {problem.level === "blocking" ? "Blocking: " : "Note: "}{problem.text}
                </p>
              ))}

              <p className="tabular text-xs text-ink-secondary">
                {plan.pairs.toLocaleString()} pair(s) -- {plan.train.toLocaleString()} train,{" "}
                {plan.val.toLocaleString()} val -- {plan.files_to_copy.toLocaleString()} files
                to copy. Layout read as <span className="text-ink">{plan.layout}</span>.
              </p>

              {plan.sample.length > 0 && (
                <div className="mt-2 flex flex-col gap-1">
                  <p className="text-xs font-medium text-ink">Where they go</p>
                  {plan.sample.map((row) => (
                    <div key={`${row.side}-${row.stem}`} className="flex flex-col">
                      <p className="tabular truncate text-xs text-ink-muted">
                        <span className="mr-2 inline-block w-10 text-ink-secondary">
                          {row.side}
                        </span>
                        {row.image}
                      </p>
                      <p className="tabular truncate pl-12 text-xs text-ink-muted/70">
                        {row.label}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {"manifest" in plan && plan.manifest && (
                <p className="tabular mt-2 text-xs text-ink-muted">
                  Which stems went where: {plan.manifest}
                </p>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}
