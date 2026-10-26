import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { useCardOpen } from "../lib/cardState";
import { onGoldenChanged } from "../lib/useGoldenSync";
import {
  ApiError,
  previewTrack2DatasetYaml,
  writeTrack2DatasetYaml,
  checkTrack2Pairing,
  getTrack2DatasetLayout,
  getTrack2Detectors,
  getTrack2Models,
  getTrack2Buckets,
  listTrack2Folders,
  getTrack2PullControl,
  saveTrack2PullConfig,
  getTrack2Retention,
  type Track2DatasetFolders,
  type Track2DatasetLayout,
  type Track2Pairing,
  type Track2DatasetRequest,
  type Track2DatasetYamlInPlace,
  type Track2Detector,
} from "../lib/api";
import { Card } from "./Card";
import { ConfirmDialog } from "./ConfirmDialog";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

/** "0 helmet, 1 gloves, 2 vest" -- or one per line, or "0: helmet" --
 * into `{ "0": "helmet", ... }`. Forgiving on punctuation because it is
 * typed once, by hand, from a labelling tool's class list. */
export function parseLabelClasses(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of text.split(/[,\n;]+/)) {
    const m = raw.trim().match(/^(\d+)\s*[:=\-]?\s*(.+)$/);
    if (m && m[2].trim()) out[String(Number(m[1]))] = m[2].trim();
  }
  return out;
}

/** The inverse, for restoring the field from a saved file. */
export function formatLabelClasses(classes: Record<string, string>): string {
  return Object.entries(classes)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([i, n]) => `${i} ${n}`)
    .join(", ");
}

/** Draws the layout that will be built, as a tree.
 *
 * The shape is the whole point of this panel, and a shape is far easier to
 * check by eye than four paths listed one per line.
 */
function LayoutTree({
  layout,
  destination,
  classes,
}: {
  layout: Track2DatasetLayout;
  destination: string;
  classes: string;
}) {
  // Totalled PER SPLIT, not per source. `layout.splits` has one row per
  // source folder now, so drawing it directly rendered "train/" once for
  // every folder -- as if the dataset had several training directories,
  // which it will not: they are merged into one.
  const totals = new Map<string, number>();
  for (const source of layout.splits) {
    totals.set(source.split, (totals.get(source.split) ?? 0) + source.image_count);
  }
  const splits = [...totals.entries()].filter(([, count]) => count > 0);
  if (splits.length === 0) return null;

  const lines: string[] = [`${destination || "<the folder the run downloads>"}/`];
  lines.push(`├── data.yaml${classes ? `        names: ${classes}` : ""}`);
  (["images", "labels"] as const).forEach((kind, kindIndex) => {
    const last = kindIndex === 1;
    lines.push(`${last ? "└──" : "├──"} ${kind}/`);
    splits.forEach(([split, count], index) => {
      const stem = last ? "    " : "│   ";
      const branch = index === splits.length - 1 ? "└──" : "├──";
      lines.push(
        `${stem}${branch} ${split}/`.padEnd(20)
        + `${count.toLocaleString()} ${kind === "images" ? "images" : "label files"}`,
      );
    });
  });
  return (
    <pre className="tabular overflow-x-auto rounded-md border border-black/15 bg-page p-3 text-xs leading-relaxed text-ink-secondary">
      {lines.join("\n")}
    </pre>
  );
}

/** One image folder and the label folder that goes with it. */
interface AssembleSource {
  images: string;
  labels: string;
}

/** Builds an Ultralytics-shaped dataset out of folders laid out any way.
 *
 * Ultralytics finds labels by swapping `/images/` for `/labels/` in each
 * image path. Labels anywhere else are invisible to it: the run trains on
 * unannotated images and reports a confident zero. So rather than teach
 * the trainer about other layouts -- its image is digest-pinned anyway --
 * the chosen folders are copied into the one shape it accepts.
 *
 * In `multiSource` mode a split can be SEVERAL folder pairs, merged into
 * one dataset. Stems that collide between them are suffixed rather than
 * overwritten -- see assemble_dataset.
 */
function mb(bytes: number): string {
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GB`;
  return `${Math.max(1, Math.round(bytes / 1024 ** 2))} MB`;
}

/** The yaml as it would be written, and what the run would pay to use it.
 *
 * The second half is the one that is easy to miss. The common parent of
 * the chosen folders becomes the download prefix, and the trainer takes
 * EVERYTHING under it -- so two folders sharing nothing but a top-level
 * name make a run download all of it. That is invisible until a GPU has
 * been sitting idle for twenty minutes, so it is counted and shown here.
 */
function YamlPreview({
  preview,
  dropUndeclared,
  onNameTheRest,
}: {
  preview: Track2DatasetYamlInPlace;
  /** Read only, to word the warning. The checkbox itself lives under
   * Label classes, where it is visible whether or not a preview has
   * found anything. */
  dropUndeclared: boolean;
  onNameTheRest: (indices: number[]) => void;
}) {
  const { download } = preview;
  const wasteful = download.extra_objects > download.in_named_folders;
  const undeclared = preview.undeclared ?? [];
  const allIndices = Array.from(new Set(undeclared.flatMap((u) => u.indices))).sort((a, b) => a - b);
  return (
    <div className="mt-3 rounded-md border border-black/15 p-3">
      {/* LABELS THE YAML CANNOT NAME. A warning with its two answers in
          the same place, not a refusal somewhere else: the thing being
          decided is "what happens to those boxes", and both answers are
          one click. Writing is refused until one is taken, unless the
          filter is on. */}
      {undeclared.length > 0 && (
        <div className={`mb-3 rounded-md border px-3 py-2 text-xs ${
          dropUndeclared ? "border-black/15 bg-page text-ink-secondary"
                         : "border-critical/40 bg-critical-subtle text-critical"}`}>
          <p>
            <span className="font-medium">
              {dropUndeclared ? "Labels use classes the yaml does not name -- they will be filtered." : "Labels use classes the yaml does not name."}
            </span>{" "}
            {undeclared.map((u) => `${u.entry}: class ${u.indices.join(", ")}`).join("; ")}.
            {" "}Ultralytics discards the <em>whole image</em> for any box it has no name for.
          </p>
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => onNameTheRest(allIndices)}
              className="rounded-md border border-black/15 bg-page px-2.5 py-1 font-medium text-ink"
            >
              Name the rest as class_{allIndices[0] ?? "N"}{allIndices.length > 1 ? "..." : ""}
            </button>
            <span>
              {dropUndeclared
                ? "The filter is on: the run drops these boxes itself."
                : <>or tick <span className="font-medium">Filter undeclared classes</span> under Label classes above.</>}
            </span>
          </div>
        </div>
      )}
      <p className="text-xs font-medium text-ink">
        {preview.written ? "Written to" : "Would be written to"}{" "}
        <span className="tabular font-normal text-ink-secondary">{preview.uri}</span>
      </p>
      <pre className="tabular mt-2 max-h-64 overflow-auto rounded-sm border border-hairline bg-page p-2 text-xs leading-relaxed text-ink-secondary">
        {preview.text}
      </pre>
      {/* WHAT THE RUN READS, in images, per folder. The download lines
          below count objects under one prefix; they cannot describe a
          run that reads several, and showed "0 objects" for training and
          "55 objects" for a validation set of 1,748 images. */}
      {preview.image_counts && (
        <div className="tabular mt-2 grid gap-x-4 gap-y-0.5 text-xs text-ink-secondary sm:grid-cols-2">
          {(["train", "val"] as const).map((split) => (
            <div key={split}>
              <p className="font-medium text-ink">
                {split === "train" ? "Training" : "Validation"}:{" "}
                {preview.image_counts?.[split].total?.toLocaleString() ?? "?"} images
              </p>
              {preview.image_counts?.[split].folders.map((row) => (
                <p key={row.folder} className="truncate" title={row.folder}>
                  {row.images === null ? "?" : row.images.toLocaleString()} &middot; {row.folder}
                </p>
              ))}
            </div>
          ))}
        </div>
      )}
      <p className="mt-2 text-xs text-ink-secondary">
        The run downloads{" "}
        <span className="tabular text-ink">{preview.download_prefix}</span> before
        training &mdash; all of it, not only the folders named above.
      </p>
      {!preview.many_roots && <p className={`tabular mt-1 text-xs ${wasteful ? "text-critical" : "text-ink-muted"}`}>
        {download.in_named_folders.toLocaleString()} objects in the folders you
        picked ({mb(download.in_named_bytes)});{" "}
        {download.extra_objects.toLocaleString()} more under that prefix
        ({mb(download.extra_bytes)}) that the run downloads and does not train on.
      </p>}
      {wasteful && !preview.many_roots && (
        <p className="mt-1 text-xs text-critical">
          More than half of that download is unused. Picking folders that sit
          closer together, or moving them under a tighter parent, would make
          every run start faster.
        </p>
      )}
      {preview.eval && (
        <div className="mt-3 border-t border-hairline pt-2">
          <p className="text-xs font-medium text-ink">
            Validation is a second download{" "}
            <span className="tabular font-normal text-ink-secondary">
              {preview.eval.uri}
            </span>
          </p>
          <p className="mt-1 text-xs text-ink-secondary">
            It lives under a different prefix, which is fine &mdash; the
            evaluation step reads it in its own container. That is where the
            held-out numbers on the registered model come from. Folders
            under other prefixes are fetched by the evaluator itself; the
            image counts above include them.
          </p>
          <pre className="tabular mt-2 max-h-40 overflow-auto rounded-sm border border-hairline bg-page p-2 text-xs leading-relaxed text-ink-secondary">
            {preview.eval.text}
          </pre>
        </div>
      )}
      {preview.existing?.ours && !preview.written && (
        <p className="mt-1 text-xs text-ink-muted">
          A data.yaml from an earlier build is already there; writing replaces it.
        </p>
      )}
    </div>
  );
}

export function Track2DatasetAssemblePanel({
  onAssembled,
  multiSource = false,
  only,
  onOnlyChange,
  modelSlug = "",
}: {
  /** The folder to train from, and the separate one to evaluate
   * against when validation lives under another prefix. */
  onAssembled?: (folder: string, evalFolder?: string) => void;
  /** Several folders per split, merged into one dataset. A site's images
   * are rarely all in one place, and copying them together by hand first
   * is the step this removes. */
  multiSource?: boolean;
  /** Whether this card is the ONLY source of training data -- which
   * switches Pull Control, Auto Training and Retention off.
   *
   * Offered only when `onOnlyChange` is given, so the standard tab, where
   * none of those three cards exist, does not show a tick that would turn
   * nothing off. */
  only?: boolean;
  onOnlyChange?: (on: boolean) => void;
  /** The Model Name chosen in "Your Model Names", passed down rather
   * than asked again.
   *
   * The classes written into data.yaml are the DATASET's, and the only
   * record of what index 0 means is a Model Name's class list -- so the
   * one already chosen for the run IS the answer, and a second dropdown
   * here was the same question with a second answer that could differ.
   * Only one of the two ever reached the yaml, and nothing said which. */
  modelSlug?: string;
}) {
  // Multi-source mode keeps its own lists; the four single fields below
  // stay for the original mode so both can share everything downstream.
  const [trainSources, setTrainSources] = useState<AssembleSource[]>([
    { images: "", labels: "" },
  ]);
  // Starts with a row, like training: the evaluation set is required
  // here. Without one the run evaluates on its own training images and
  // the reported metrics are not held out -- which looks like a good
  // result and is not one.
  const [evalSources, setEvalSources] = useState<AssembleSource[]>([
    { images: "", labels: "" },
  ]);
  // Which field a folder pick applies to. One browser on screen at a
  // time, rather than four per row -- with several sources that would be
  // a dozen browsers open at once.
  const [picking, setPicking] = useState<
    { side: "train" | "eval"; index: number; field: "images" | "labels" } | null
  >(null);
  const [pickPrefix, setPickPrefix] = useState("");
  const [pickFolders, setPickFolders] = useState<string[]>([]);
  /** Which bucket the picker browses. Folders from the home bucket are
   * stored as bare keys, exactly as before; from any other as
   * `s3://bucket/key`, which the builder and the run both understand. */
  const [buckets, setBuckets] = useState<string[]>([]);
  const [homeBucket, setHomeBucket] = useState("");
  const [pickBucket, setPickBucket] = useState("");
  const [folders, setFolders] = useState<Track2DatasetFolders>({
    train_images: "", train_labels: "", eval_images: "", eval_labels: "",
  });
  const [layout, setLayout] = useState<Track2DatasetLayout | null>(null);
  // Shown while a preview request is in flight, so the panel reads as
  // live rather than as a stale answer to the previous pick.
  const [previewing, setPreviewing] = useState(false);
  const [detectors, setDetectors] = useState<Track2Detector[]>([]);
  const [detectorSlug, setDetectorSlug] = useState("");
  // `scheduled` is gone. The Training data radios in step 6 decide
  // what a scheduled run builds from, and "Enabled for scheduling"
  // beside them turns it on -- so a copy of that answer here could only
  // ever disagree with it.
  /** The refusal, when a data.yaml this card did not write is already
   * there. Held as the dialog's content so the server's own sentence is
   * what the operator reads. */
  const [confirming, setConfirming] = useState<Track2DatasetYamlInPlace | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState("");
  // The yaml as it would be written, plus what the run would
  // download. Null until Preview has been pressed.
  const [preview, setPreview] = useState<Track2DatasetYamlInPlace | null>(null);
  // Replace a data.yaml this card did not write. Off unless ticked.
  const [overwrite, setOverwrite] = useState(false);
  // Open by default, unlike Pull control: this is the card most visits to
  // the page are for, so it would be shut-then-opened every time.
  const [open, setOpen] = useCardOpen("assemble", true);

  // One browser's worth of state per picker, so each can be navigated
  // independently -- they are usually in different parts of the bucket.
  const [prefixes, setPrefixes] = useState<Record<keyof Track2DatasetFolders, string>>({
    train_images: "", train_labels: "", eval_images: "", eval_labels: "",
  });
  const [listing, setListing] = useState<Record<string, string[]>>({});
  // Whether each pair's images and labels actually describe each other,
  // keyed `side-index`. An Ultralytics label is matched to its image by
  // STEM, so folders can be picked independently -- and nothing else in
  // the pipeline checks that they line up. Picking images/train beside
  // labels/val produces a dataset where nothing pairs, which trains
  // happily and learns nothing.
  const [pairing, setPairing] = useState<Record<string, Track2Pairing | null>>({});
  const [pairingBusy, setPairingBusy] = useState<Record<string, boolean>>({});
  // Whether the saved settings have been read yet. The writer below must
  // not run before this is true, or it would save the empty defaults over
  // the settings it is still waiting for -- the card would clear itself
  // on every page load.
  const [settingsRead, setSettingsRead] = useState(false);
  // The saved "only these folders" tick, as read at mount.
  //
  // This card is rendered on BOTH tabs, and only the Multi-Auto Training
  // one passes `onOnlyChange` -- it owns the switch, because it owns the
  // three cards the switch greys out. The other instance must therefore
  // write back what it found rather than its own `only`, which is
  // always undefined there: saving `false` from the tab that cannot see
  // the tick would silently turn it off for the tab that can.
  const [savedOnly, setSavedOnly] = useState(false);
  /** Whether THIS card's folders are one of the sources.
   *
   * Defaults to true: every configuration that existed before the
   * three ticks did was using its folders, and a default of false
   * would quietly empty their training set on the next run. */
  const [useFolders, setUseFolders] = useState(true);
  /** What the LABEL FILES mean, as typed: every class in index order.
   * Distinct from the Model Name, which is what to TRAIN. When this is
   * set and contains the Model Name's class by name, the yaml declares
   * it and the run filters at load time, keeping every image. */
  const [labelClassesText, setLabelClassesText] = useState("");
  /** The filter tick: boxes whose class the yaml does not name are dropped
   * by the training container from its downloaded copy. Saved like every
   * other setting; the chain reads it. Off by default. */
  const [dropUndeclared, setDropUndeclared] = useState(false);
  /** Where the yaml build renders: a slot the page places just above
   * "Start the training run". Read after mount, once the page's DOM exists. */
  const [buildSlot, setBuildSlot] = useState<HTMLElement | null>(null);
  useEffect(() => {
    setBuildSlot(document.getElementById("track2-yaml-build-slot"));
  }, []);
  /** The Golden Set card's switch, `retention.use_golden`. READ here,
   * never written: that card owns it and is where it is ticked. This
   * card needs it only to say what the next run will be built from.
   *
   * `null` until it has been read, so the line does not report a golden
   * set as unused before the answer is back. */
  const [useGolden, setUseGolden] = useState<boolean | null>(null);
  // What this card last built, so the training card can offer "use the
  // assembled dataset" after a reload. Without it the folder lived in
  // this component and the option would be empty every time the page
  // was opened.
  const [built, setBuilt] = useState<
    { folder: string; eval_folder?: string; at?: string } | null>(null);

  // --- the card's settings ------------------------------------------------
  //
  // These used to live only in this component, so they were gone on
  // reload and a settings file saved while they were on screen did not
  // contain them. The save was never broken: it copies the live config,
  // and this card never wrote anything into it.
  useEffect(() => {
    getTrack2PullControl()
      .then((status) => {
        const saved = status.config.dataset_assemble || {};
        if (saved.train_sources?.length) setTrainSources(saved.train_sources);
        if (saved.eval_sources?.length) setEvalSources(saved.eval_sources);
        if (saved.folders) setFolders(saved.folders);
        // Only as a starting value, and only until the page says
        // otherwise: the effect below keeps this following the Model
        // Name card, including a change made after this card mounted.
        const inherited = String(status.config.default_detector?.slug || "");
        if (inherited) setDetectorSlug(inherited);
        else if (saved.detector_slug) setDetectorSlug(saved.detector_slug);
        if (typeof saved.overwrite === "boolean") setOverwrite(saved.overwrite);
        if (typeof saved.use === "boolean") setUseFolders(saved.use);
        if (saved.built?.folder) setBuilt(saved.built);
        // Restored through the page, which owns the switch and uses it to
        // grey out the three cards this one replaces.
        if (typeof saved.only === "boolean") {
          setSavedOnly(saved.only);
          onOnlyChange?.(saved.only);
        }
        if (saved.label_classes && typeof saved.label_classes === "object") {
          setLabelClassesText(formatLabelClasses(saved.label_classes));
        }
        if (typeof saved.drop_undeclared_labels === "boolean") {
          setDropUndeclared(saved.drop_undeclared_labels);
        }
      })
      .catch(() => undefined)
      .finally(() => setSettingsRead(true));
    // Mount only: this reads the saved settings, and re-reading them
    // after an edit would fight the writer below.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The golden set's switch, from the card that owns it. Only on the
  // tab that has one -- the standard tab has no Golden Set card.
  useEffect(() => {
    if (!onOnlyChange) return;
    getTrack2Retention()
      .then((r) => setUseGolden(Boolean(r.config.use_golden)))
      .catch(() => setUseGolden(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Ticked on the Golden Set card, which is on this page -- so the
  // summary line below follows it without waiting for a reload.
  useEffect(() => onGoldenChanged(setUseGolden), []);

  // One writer for the whole card rather than a save call bolted onto
  // each setter -- a setter added later cannot forget to call it.
  // Debounced, because picking a folder changes several of these at once
  // and each keystroke in a prefix box would otherwise be a request.
  useEffect(() => {
    if (!settingsRead) return;
    const timer = setTimeout(() => {
      void saveTrack2PullConfig({
        dataset_assemble: {
          only: onOnlyChange ? !!only : savedOnly,
          use: useFolders,
          label_classes: parseLabelClasses(labelClassesText),
          drop_undeclared_labels: dropUndeclared,
          train_sources: trainSources,
          eval_sources: evalSources,
          folders,
          detector_slug: detectorSlug,
          overwrite,
          // Carried through the writer rather than written separately:
          // this effect sends the whole `dataset_assemble` object, so a
          // second writer would be overwritten by whichever ran last.
          built,
        },
      }).catch(() => undefined);
    }, 600);
    return () => clearTimeout(timer);
  }, [settingsRead, only, onOnlyChange, savedOnly, useFolders, labelClassesText,
      dropUndeclared, trainSources, evalSources, folders, detectorSlug, overwrite, built]);

  // Check each filled pair as it is filled. Debounced, because picking a
  // folder changes a field per click and every keystroke in a prefix box
  // would otherwise be a listing of two folders.
  useEffect(() => {
    const pairs: { key: string; images: string; labels: string }[] = [];
    for (const [side, list] of [["train", trainSources], ["eval", evalSources]] as const) {
      list.forEach((source, index) => {
        if (source.images.trim() && source.labels.trim()) {
          pairs.push({
            key: `${side}-${index}`,
            images: source.images.trim(),
            labels: source.labels.trim(),
          });
        }
      });
    }
    if (!pairs.length) return;

    let cancelled = false;
    const timer = setTimeout(() => {
      setPairingBusy(Object.fromEntries(pairs.map((p) => [p.key, true])));
      void Promise.all(pairs.map((pair) =>
        // The bucket is left to the server, which falls back to the one
        // everything else here reads.
        checkTrack2Pairing("", [pair.images], [pair.labels])
          .then((result) => [pair.key, result] as const)
          .catch(() => [pair.key, null] as const),
      )).then((results) => {
        if (cancelled) return;
        setPairing((current) => ({ ...current, ...Object.fromEntries(results) }));
        setPairingBusy({});
      });
    }, 700);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [trainSources, evalSources]);

  useEffect(() => {
    // The built-in PPE list is filtered out here as it is everywhere
    // else -- "Your Model Names" filters it the same way, so the slug
    // this card is handed can never be a built-in one and the two lists
    // agree. Kept as a LOOKUP now rather than a dropdown's options: the
    // classes are shown, not chosen, and this is what resolves the slug
    // into the names the yaml declares.
    getTrack2Detectors()
      .then((res) => setDetectors(res.detectors.filter((d) => !d.builtin)))
      .catch(() => undefined);
  }, []);

  // Each picker lists its own prefix. The browse endpoint takes two
  // prefixes per call, so the four are fetched as two calls.
  useEffect(() => {
    let cancelled = false;
    getTrack2Models(prefixes.train_images, undefined, prefixes.train_labels)
      .then((res) => {
        if (cancelled) return;
        setListing((c) => ({
          ...c,
          train_images: res.dataset_browser.folders,
          train_labels: res.eval_browser.folders,
        }));
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [prefixes.train_images, prefixes.train_labels]);

  useEffect(() => {
    let cancelled = false;
    getTrack2Models(prefixes.eval_images, undefined, prefixes.eval_labels)
      .then((res) => {
        if (cancelled) return;
        setListing((c) => ({
          ...c,
          eval_images: res.dataset_browser.folders,
          eval_labels: res.eval_browser.folders,
        }));
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [prefixes.eval_images, prefixes.eval_labels]);

  // Re-previewed whenever a folder changes: the pairing is the thing worth
  // seeing, and seeing it late is the failure this panel exists to stop.
  const hasTraining = multiSource
    ? trainSources.some((s) => s.images)
    : Boolean(folders.train_images);
  // Required in multi-source mode. Evaluating on the training images
  // reports metrics that are not held out, and they look like good ones.
  const hasEval = multiSource
    ? evalSources.some((s) => s.images)
    : true;
  const complete = hasTraining && hasEval;

  useEffect(() => {
    if (!hasTraining) {
      setLayout(null);
      return;
    }
    let cancelled = false;
    setPreviewing(true);
    getTrack2DatasetLayout(currentFolders())
      .then((res) => !cancelled && setLayout(res))
      .catch(() => !cancelled && setLayout(null))
      .finally(() => !cancelled && setPreviewing(false));
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [folders, trainSources, evalSources, multiSource, hasTraining]);

  // The Model Name card is on this same page, so its choice can change
  // while this card is open. Without this the classes here would be
  // whatever was chosen when the card mounted -- stale, and silently so,
  // because a class list that is merely wrong still looks like a list.
  useEffect(() => {
    if (modelSlug) setDetectorSlug(modelSlug);
  }, [modelSlug]);

  const chosen = detectors.find((d) => d.slug === detectorSlug);

  /** The class names as the API wants them, from the chosen Model Name. */
  /** The Model Name's own list, as `{index: name}`. */
  function modelClasses(): Record<string, string> {
    return Object.fromEntries(
      Object.entries(chosen?.classes ?? {}).map(([index, name]) => [String(index), name]),
    );
  }

  /** The label taxonomy, when it has been typed AND carries the Model
   * Name's class by name; otherwise the Model Name's list. Without the
   * name check a typo would silently produce a yaml with nothing to
   * filter to -- the chain refuses that case server-side as well. */
  function labelledClasses(): Record<string, string> | null {
    const typed = parseLabelClasses(labelClassesText);
    if (!Object.keys(typed).length) return null;
    const own = Object.values(modelClasses());
    const have = new Set(Object.values(typed));
    return own.every((n) => have.has(n)) ? typed : null;
  }

  /** What the yaml declares.
   *
   * Two genuine alternatives, not a layering:
   *
   *   filter OFF -- the yaml must name every class the label files use,
   *                 or Ultralytics drops every image holding a box it
   *                 has no name for. Label classes wins when it covers
   *                 the Model Name's class.
   *   filter ON  -- the yaml names ONLY what the model trains, and the
   *                 training container drops every other box from its
   *                 downloaded copy. Label classes is not read.
   *
   * It was a layering at first: six names typed AND the filter ticked
   * gave a six-name yaml with nothing left to filter, and the tick looked
   * like it did nothing. "Why does the yaml show six when I asked for
   * helmet" is the right question, and this is the answer. */
  function classesOf(): Record<string, string> {
    if (dropUndeclared) return modelClasses();
    return labelledClasses() ?? modelClasses();
  }

  /** "Name the rest": every undeclared index gets a placeholder name so
   * the yaml covers the label files and nothing is discarded. Placeholders
   * are honest -- `class_4` says it is one -- and can be typed over any
   * time. The Model Name's own classes keep their names. */
  function nameTheRest(indices: number[]) {
    const merged: Record<string, string> = { ...modelClasses(), ...parseLabelClasses(labelClassesText) };
    for (const i of indices) if (!merged[String(i)]) merged[String(i)] = `class_${i}`;
    setLabelClassesText(formatLabelClasses(merged));
  }

  /** Writes it.
   *
   * This used to be disabled until "Preview the yaml" had been pressed,
   * and said nothing about why -- so a complete card showed a dead
   * button and the control that would have woken it was a separate
   * press with no arrow pointing at it. The preview is still there for
   * anyone who wants to read the file first; it is no longer a toll
   * gate.
   *
   * `force` is the answer the overwrite dialog sends back.
   */
  async function handleWrite(force = false) {
    if (!chosen || !complete) return;
    setBusy(true);
    setError("");
    try {
      if (!force) {
        // Asked BEFORE writing, so a data.yaml this card did not write
        // becomes a dialog naming it -- rather than a refusal arriving
        // after the press, from a server the button cannot quote.
        const check = await previewTrack2DatasetYaml(
          await foldersForBuild(), classesOf(), false, dropUndeclared);
        setPreview(check);
        if (check.needs_overwrite) {
          setConfirming(check);
          setBusy(false);
          return;
        }
      }
      const out = await writeTrack2DatasetYaml(await foldersForBuild(), classesOf(), force || overwrite, dropUndeclared);
      setPreview(out);
      setResult(
        (out.written_uris ?? [out.uri]).join("  +  ") + " written -- nothing copied.",
      );
      // The folder the run trains from is the PARENT, because that is the
      // one prefix the trainer downloads and the yaml sits at its root.
      // The eval side is passed separately when it has its own prefix:
      // that is EVAL_DATA_S3_URI, a different download entirely.
      const record = {
        folder: out.folder,
        eval_folder: out.eval?.folder,
        at: new Date().toISOString(),
      };
      setBuilt(record);
      // Written NOW, not by the debounced writer below.
      //
      // Step 6 fills "Use the assembled dataset" by re-reading
      // `dataset_assemble.built` from the config, and it re-reads as
      // soon as `onAssembled` fires. A 600ms debounce between the two
      // means that read lands BEFORE the value it is looking for --
      // the option stays greyed out saying "Nothing assembled yet"
      // beside a yaml that exists, and nothing on screen says a reload
      // is what it wants. Awaited for the same reason: the order is the
      // point, not the saving.
      await saveTrack2PullConfig({
        dataset_assemble: {
          only: onOnlyChange ? !!only : savedOnly,
          use: useFolders,
          label_classes: parseLabelClasses(labelClassesText),
          drop_undeclared_labels: dropUndeclared,
          train_sources: trainSources,
          eval_sources: evalSources,
          folders,
          detector_slug: detectorSlug,
          overwrite,
          built: record,
        },
      }).catch(() => undefined);
      onAssembled?.(out.folder, out.eval?.folder);
      setConfirming(null);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not write the yaml.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    getTrack2Buckets()
      .then((res) => {
        setBuckets(res.buckets.filter((b) => b.readable).map((b) => b.name));
        setHomeBucket(res.default);
        setPickBucket((current) => current || res.default);
      })
      .catch(() => undefined);
  }, []);

  // The browser for whichever field is being picked -- in the home bucket
  // through the same listing as always, in any other through the generic
  // folder listing.
  useEffect(() => {
    if (!picking) return;
    let cancelled = false;
    const listing = !pickBucket || pickBucket === homeBucket
      ? getTrack2Models(pickPrefix, undefined, undefined).then((res) => res.dataset_browser.folders)
      : listTrack2Folders(pickBucket, pickPrefix).then((res) => res.folders);
    listing
      .then((folders) => !cancelled && setPickFolders(folders))
      .catch(() => !cancelled && setPickFolders([]));
    return () => { cancelled = true; };
  }, [picking, pickPrefix, pickBucket, homeBucket]);

  /** A picked folder as it is stored: a bare key at home, a URI elsewhere. */
  function qualified(path: string): string {
    return pickBucket && homeBucket && pickBucket !== homeBucket
      ? `s3://${pickBucket}/${path.replace(/^\/+/, "")}`
      : path;
  }

  function setSource(side: "train" | "eval", index: number, field: "images" | "labels", value: string) {
    const setter = side === "train" ? setTrainSources : setEvalSources;
    setter((current) =>
      current.map((source, i) => (i === index ? { ...source, [field]: value } : source)),
    );
  }

  /** Empty one field without touching the other.
   *
   * The only removal this card offers now. A row-level "remove the
   * pair" was the wrong tool for the common mistake -- images right,
   * labels pointing at the wrong split -- and an emptied row costs
   * nothing: `currentFolders` drops any pair with no images, so a blank
   * one is ignored rather than assembled. */
  function clearField(side: "train" | "eval", index: number,
                      field: "images" | "labels") {
    setSource(side, index, field, "");
    setPairing((current) => ({ ...current, [`${side}-${index}`]: null }));
  }

  function addSource(side: "train" | "eval") {
    const setter = side === "train" ? setTrainSources : setEvalSources;
    setter((current) => [...current, { images: "", labels: "" }]);
  }

  /** What gets sent, in whichever mode is active. */
  function currentFolders(): Track2DatasetRequest {
    if (!multiSource) return folders;
    return {
      ...folders,
      train_sources: trainSources.filter((s) => s.images),
      eval_sources: evalSources.filter((s) => s.images),
    };
  }

  /** What the yaml is BUILT from: these folders, plus the golden set when
   * it is ticked -- the same composition the chain makes at run time, so
   * the yaml previewed and written here is the one the run trains on.
   *
   * Golden's train side joins training; its val side REPLACES the val
   * folders as the held-out set. Pull Control is never added here: this
   * card is the hand-picked path, and the pull's week folders are not
   * part of it.
   *
   * Read fresh from the config each time, because the Golden Set card
   * saves its folder and split independently of this one. */
  async function foldersForBuild(): Promise<Track2DatasetRequest> {
    const base = currentFolders();
    if (!multiSource || !useGolden) return base;
    let golden: { target_folder?: string; split?: boolean } = {};
    try {
      golden = (await getTrack2PullControl()).config.golden_set || {};
    } catch {
      return base;
    }
    const target = (golden.target_folder || "").replace(/^\/+|\/+$/g, "");
    if (!target) return base;
    const split = golden.split !== false;
    const goldenTrain = { images: split ? `${target}/images/train` : `${target}/images`, labels: "" };
    const goldenVal = split ? [{ images: `${target}/images/val`, labels: "" }] : null;
    return {
      ...base,
      train_sources: [...(useFolders ? base.train_sources ?? [] : []), goldenTrain],
      // Golden val is ADDED to the dataset's own val: both are scored.
      eval_sources: [...(useFolders ? base.eval_sources ?? [] : []), ...(goldenVal ?? [])],
    };
  }

  const pickers: { key: keyof Track2DatasetFolders; label: string; hint: string }[] = [
    { key: "train_images", label: "Training images", hint: "required" },
    { key: "train_labels", label: "Training labels", hint: ".txt files for those images" },
    { key: "eval_images", label: "Evaluation images", hint: "optional" },
    { key: "eval_labels", label: "Evaluation labels", hint: "optional" },
  ];

  // What the card says when it is shut. A count of chosen folders and
  // whether anything has been built is enough to know if it needs
  // opening; "Press Show" alone would make it a door with no label.
  const sourcesChosen = [...trainSources, ...evalSources]
    .filter((source) => source.images || source.labels).length;

  /* THE YAML BUILD, rendered just above "Start the training run".
   *
   * Asked for there because it is the last thing done before a run:
   * folders are picked up here, the yaml is written down there, next to
   * the button that trains on it. A portal rather than a second
   * component, so every piece of state -- folders, classes, the filter,
   * the preview -- stays in this one card and nothing is duplicated.
   * Rendered in place when the page offers no slot. */
  const buildSection = (
    <>
      {/* Shown, not asked. The Model Name card above is where this is
          decided; a dropdown here was the same question twice, and only
          one of the two answers ever reached the yaml. */}
      <div className="mt-3">
        <span className="mb-1 block text-xs text-ink-muted">
          Classes{" "}
          <span className="text-ink-muted/70">
            (from your Model Name, in index order)
          </span>
        </span>
        {chosen ? (
          <p className="tabular text-sm text-ink-secondary">
            <span className="text-ink">{chosen.label}</span> detects{" "}
            {Object.entries(chosen.classes)
              .sort((a, b) => Number(a[0]) - Number(b[0]))
              .map(([index, name], i) => (
                <span key={index}>
                  {i > 0 && ", "}
                  <span className="text-ink-muted">{index}:</span>
                  {name}
                </span>
              ))}
          </p>
        ) : (
          <p className="text-sm text-info">
            Pick a Model Name in <strong>Your Model Names</strong> above &mdash;
            its class list is what this{" "}
            <span className="tabular">data.yaml</span> declares.
          </p>
        )}

        {/* WHAT THE LABEL FILES MEAN, when that is more than the Model
            Name trains. A site dataset is labelled once with everything
            on it; a helmet Model Name has one class. Writing the one
            class into the yaml made Ultralytics drop every image with a
            box it had no name for -- silently on the training side,
            and a refusal to start on validation. Said here, under the
            Model Name's list, because this is the other half of the
            same question: that list is what to TRAIN, this is what was
            LABELLED. */}
        <label className="mt-2 block text-xs text-ink-muted">
          Label classes{" "}
          <span className="text-ink-muted/70">
            (every class the label files use, in index order &mdash; leave
            blank if they are exactly the Model Name&apos;s)
          </span>
          <input
            type="text"
            value={labelClassesText}
            onChange={(e) => setLabelClassesText(e.target.value)}
            placeholder="0 helmet, 1 gloves, 2 vest, 3 boots"
            className="tabular mt-1 w-full rounded border border-black/15 bg-page px-2 py-1 text-sm text-ink"
          />
        </label>
        {/* ALWAYS VISIBLE, not only when a mismatch has been found. It is
            a saved setting, and a setting that only shows itself after a
            preview on a dataset that happens to need it is one nobody can
            find on a dataset that does not. The warning in the preview
            points here. */}
        <label className="mt-1.5 flex items-start gap-1.5 text-xs text-ink-secondary">
          <input
            type="checkbox"
            checked={dropUndeclared}
            onChange={(e) => setDropUndeclared(e.target.checked)}
            className="mt-0.5 h-3.5 w-3.5 shrink-0"
          />
          <span>
            <span className="text-ink">Filter undeclared classes in the training container</span>{" "}
            <span className="text-ink-muted">(label files in S3 unchanged)</span>. Boxes whose
            class the yaml does not name are dropped from the run&apos;s downloaded copy of
            each label file before Ultralytics reads it. Off, a yaml naming fewer classes
            than the labels use is refused instead.
          </span>
        </label>
        {dropUndeclared && chosen && (
          <p className="mt-1 text-xs text-ink-secondary">
            Filter on: the yaml declares only{" "}
            <span className="text-ink">{Object.values(modelClasses()).join(", ")}</span>;
            every other box is dropped from the run&apos;s downloaded label files
            before Ultralytics reads them. Label classes is not used while this
            is ticked.
          </p>
        )}
        {!dropUndeclared && labelClassesText.trim() && chosen && (
          labelledClasses() ? (
            <p className="mt-1 text-xs text-ink-secondary">
              The yaml declares {Object.keys(labelledClasses() ?? {}).length} classes;
              the run keeps every image and trains only{" "}
              <span className="text-ink">{Object.values(modelClasses()).join(", ")}</span>,
              filtered at load time.
            </p>
          ) : (
            <p className="mt-1 text-xs text-critical">
              These do not include{" "}
              {Object.values(modelClasses()).join(", ")} by name, so the run
              would have nothing to filter to. Add it at the index the label
              files use for it.
            </p>
          )
        )}
      </div>

      <p className="mt-2 text-xs text-ink-muted">
        {/* No Destination field any more: nothing is copied, so there is
            nowhere to copy it to. The yaml is written at the common
            parent of the folders above, because that is the one prefix
            the run downloads and a data.yaml has to sit at its root. */}
        Nothing is copied. One <span className="tabular">data.yaml</span> is
        written naming these folders where they already are, and the class
        indices in your .txt files are not rewritten &mdash; so they must
        already match the order above. Labels are never named in the file:
        Ultralytics finds them by swapping{" "}
        <span className="tabular">/images/</span> for{" "}
        <span className="tabular">/labels/</span>, so each folder above needs
        its labels sibling in that exact shape.
      </p>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {/* ONE button. It builds the yaml (what the old Preview did),
            shows it below, and writes it -- or, when something must be
            decided first (an existing hand-written yaml, labels the yaml
            cannot name, no validation folder), stops and shows that
            instead, with its answer beside it. */}
        <button
          type="button"
          disabled={busy || !complete || !detectorSlug}
          onClick={() => void handleWrite()}
          className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy ? "Generating..." : "Generate yaml training file"}
        </button>
        {/* The scheduling tick that used to live here has moved to
            "Enabled for scheduling" beside the Training data radios in
            step 6.

            It was a SECOND answer to a question that group already asks:
            pick "Use the assembled dataset" there and this card is what
            the schedule builds from. Two switches that can disagree is
            how an unattended run trains something nobody chose, and
            nothing on screen said which one won. */}
      </div>
      {/* Where the button is, not three hundred pixels above it. Both of
          these answer "Preview the yaml" and "Write it", and nothing
          else on this card sets them -- rendered at the top, a failed
          write scrolled out of view and read as the button doing
          nothing, which is exactly how it was reported. */}
      {error && <p className="mt-2 text-xs text-critical">{error}</p>}
      {result && (
        <p className="tabular mt-2 text-xs text-ink-secondary">{result}</p>
      )}
      {!detectorSlug && complete && (
        <p className="mt-1 text-xs text-info">
          Pick a Model Name above &mdash; its classes are what the yaml declares.
        </p>
      )}
      {!complete && (
        <p className="mt-1 text-xs text-ink-muted">
          {hasTraining
            ? "Pick an evaluation folder -- evaluating on the training images reports metrics that are not held out."
            : "Pick a training folder."}
        </p>
      )}

      {preview && (
        <YamlPreview
          preview={preview}
          dropUndeclared={dropUndeclared}
          onNameTheRest={nameTheRest}
        />
      )}

      {/* The refusal IS the dialog, rather than a red line somewhere else
          and a second control to find. It names the file, says what is
          kept, and carries the harsher answer as its own button. */}
      <ConfirmDialog
        open={!!confirming}
        title="Replace the data.yaml already there?"
        destructive
        busy={busy}
        confirmLabel="Replace it"
        cancelLabel="Cancel"
        body={
          <>
            <p>
              <span className="tabular">{confirming?.key}</span> already exists
              and was not written by this card, so it is somebody's own file.
            </p>
            <p className="mt-2">
              Replacing it changes which folders and class names the next run
              trains on. <strong>No images or labels are touched</strong>, and
              no dataset or trained weights are affected &mdash; only this one
              file.
            </p>
            {confirming?.existing?.bytes != null && (
              <p className="mt-2 text-xs text-ink-muted">
                The file there now is {confirming.existing.bytes} bytes.
              </p>
            )}
          </>
        }
        onConfirm={() => void handleWrite(true)}
        onCancel={() => setConfirming(null)}
      />
    </>
  );

  return (
    // collapsible={false}: this card runs its own Show/Hide,
    // because it shows a summary of live state when shut and
    // the generic one in Card cannot know what to put there.
    <Card
      title="Assemble a dataset"
      collapsible={false}
      caption="Point at your images and labels wherever they live; this copies them into the images/ + labels/ layout the trainer requires."
      titleAside={onOnlyChange ? (
        // THIS card's own source, and only this one. The other two are
        // ticked on the cards they belong to -- Pick Golden Set / Edge
        // Cases and Pull control each carry their switch in their own
        // heading, which is where somebody looking for it goes. Three
        // ticks here meant one question asked in two places; the
        // summary line below says what they add up to without claiming
        // to be where they are set.
        <label className="ml-3 inline-flex items-center gap-1.5 align-middle text-xs font-normal text-ink-secondary">
          <input
            type="checkbox"
            checked={useFolders}
            onChange={(e) => setUseFolders(e.target.checked)}
            className="h-3.5 w-3.5"
          />
          Use these folders
        </label>
      ) : undefined}
      action={
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
        >
          {open ? "Hide" : "Show"}
        </button>
      }
    >
      {/* Shown whether the card is open or shut, because a shut card is
          still the reason those three are off. It NAMES where the tick
          lives rather than carrying one: the switch belongs on the card
          it switches. */}
      {/* What the next run will actually train on. The three ticks sit
          on three different cards now, which is right -- each with the
          thing it switches -- but it means nowhere else on the page
          shows them added up. This is a reading of them, not a second
          place to set them. */}
      {onOnlyChange && (
        <p className="mb-2 text-xs text-ink-secondary">
          <span className="font-medium text-ink">data.yaml will be built from:</span>{" "}
          {[
            useFolders ? "these folders" : "",
            useGolden ? "the golden set / edge cases" : "",
            !only ? "Pull Control's week folders" : "",
          ].filter(Boolean).join(" + ") || (
            <span className="text-critical">
              nothing &mdash; tick at least one, or the run has no training data
            </span>
          )}
          {useGolden ? " (the golden set's val side is the held-out evaluation set)" : ""}
        </p>
      )}
      {only && onOnlyChange && (
        <p className="mb-3 rounded-md border border-accent/40 bg-accent-subtle px-3 py-2 text-xs text-ink-secondary">
          This run takes its data from the folders on this card.{" "}
          <span className="font-medium text-ink">Pull Control, Auto Training
          and Retention are switched off</span> while that is ticked &mdash;
          they are the other way of deciding where training data comes from,
          and two answers to that question is one too many. Tick{" "}
          <span className="font-medium text-ink">Use Pull Control</span> in the
          Pull control heading to put them back.
        </p>
      )}
      {!open ? (
        <p className="tabular text-xs text-ink-muted">
          {result
            ? `Wrote ${result}`
            : sourcesChosen
              ? `${sourcesChosen} source${sourcesChosen === 1 ? "" : "s"} chosen`
                + `${layout?.total_paired ? `, ${layout.total_paired.toLocaleString()} paired` : ""}`
                + ", not assembled yet"
              : "Nothing chosen yet."}
        </p>
      ) : (
      <>
      {multiSource ? (
        <div className="flex flex-col gap-3">
          {([
            ["train", "Training sources", trainSources],
            ["eval", "Evaluation sources", evalSources],
          ] as const).map(([side, heading, list]) => (
            <div key={side}>
              <div className="mb-1 flex items-center justify-between gap-2">
                <span className="text-xs font-medium text-ink">
                  {heading}{" "}
                  <span className="font-normal text-ink-muted">
                    (at least one)
                  </span>
                </span>
                <button
                  type="button"
                  onClick={() => addSource(side)}
                  className="rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary"
                >
                  Add folder pair
                </button>
              </div>
              {list.length === 0 && (
                <p className="text-xs text-ink-muted">
                  None yet -- add at least one folder pair.
                </p>
              )}
              <div className="flex flex-col gap-1.5">
                {list.map((source, index) => (
                  <div
                    key={`${side}-${index}`}
                    className="rounded-md border border-hairline bg-page p-2.5"
                  >
                    <div className="grid gap-2 sm:grid-cols-2">
                      {(["images", "labels"] as const).map((field) => {
                        const active = picking?.side === side
                          && picking.index === index && picking.field === field;
                        return (
                          <div key={field} className="min-w-0">
                            <span className="mb-1 block text-xs text-ink-muted">
                              {field === "images" ? "Images" : "Labels"}
                            </span>
                            <button
                              type="button"
                              onClick={() => {
                                setPicking(active ? null : { side, index, field });
                                setPickPrefix("");
                              }}
                              className={`tabular w-full truncate rounded-md border px-3 py-2 text-left text-xs ${
                                active ? "border-accent bg-accent-subtle" : "border-black/15"
                              }`}
                              title={source[field] || undefined}
                            >
                              {source[field] || "Choose a folder..."}
                            </button>
                            {/* Under its own field, so which of the two
                                it empties is not a guess. Rendered for
                                both fields and for Training and
                                Evaluation alike -- the loops above run
                                the same markup for all four.

                                Always shown, not only when the field
                                holds something: a control that appears
                                and disappears is one you have to hunt
                                for, and on an empty field it simply does
                                nothing. */}
                            <button
                              type="button"
                              onClick={() => clearField(side, index, field)}
                              className="mt-1 text-xs font-medium text-critical underline"
                            >
                              Remove this folder
                            </button>
                          </div>
                        );
                      })}
                    </div>
                    {/* Do the two folders describe each other? A label is
                        matched to its image by STEM, so independently
                        chosen folders can look fine and pair with
                        nothing -- images/train beside labels/val trains
                        happily and learns nothing. */}
                    {(() => {
                      const key = `${side}-${index}`;
                      if (pairingBusy[key]) {
                        return <p className="mt-1.5 text-xs text-ink-muted">Checking the pairing...</p>;
                      }
                      const check = pairing[key];
                      if (!check || !check.checked) return null;
                      const orphanLabels = check.labels_without_images;
                      const bare = check.images_without_labels;
                      const nothingPairs = check.paired === 0;
                      if (nothingPairs) {
                        return (
                          <p className="mt-1.5 text-xs text-critical">
                            <strong>Nothing pairs.</strong> {check.images.toLocaleString()} images
                            and {check.labels.toLocaleString()} labels, and not one filename
                            matches. These two folders describe different sets of
                            images &mdash; the usual cause is one pointing at train
                            and the other at val.
                          </p>
                        );
                      }
                      if (orphanLabels > 0) {
                        return (
                          <p className="mt-1.5 text-xs text-critical">
                            <strong>{orphanLabels.toLocaleString()} labels have no image.</strong>{" "}
                            {check.paired.toLocaleString()} of {check.labels.toLocaleString()} pair.
                            A label with no image describes nothing and never reaches
                            the trainer, so whatever it annotated is missing from the run.
                            {check.sample_labels_without_images.length > 0 && (
                              <span className="tabular block text-ink-secondary">
                                e.g. {check.sample_labels_without_images.slice(0, 3).join(", ")}
                              </span>
                            )}
                          </p>
                        );
                      }
                      if (bare === check.images && check.images > 0) {
                        return (
                          <p className="mt-1.5 text-xs text-critical">
                            <strong>No image has a label.</strong> All{" "}
                            {check.images.toLocaleString()} would train as background.
                            The labels folder is almost certainly the wrong one.
                          </p>
                        );
                      }
                      return (
                        <p className="mt-1.5 text-xs text-ink-secondary">
                          {check.paired.toLocaleString()} of {check.images.toLocaleString()} images
                          are labelled
                          {bare > 0 && (
                            <span className="text-ink-muted">
                              {" "}&middot; {bare.toLocaleString()} without labels, which train
                              as background examples
                            </span>
                          )}
                          .
                        </p>
                      );
                    })()}

                  </div>
                ))}
              </div>

            {/* A second one, here as well as in the header: after filling
                a pair the eye is at the bottom of the list, not back up
                at the heading. There is no cap on how many. */}
            <button
              type="button"
              onClick={() => addSource(side)}
              className="mt-1.5 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary"
            >
              + Add another {side === "train" ? "training" : "evaluation"} pair
            </button>

            {/* Rendered inside the section it belongs to, so the
                browser opens directly under the pair being edited
                rather than at the foot of the whole card. */}
            {picking?.side === side && (
              <div className="rounded-md border border-accent/40 p-3">
                <p className="mb-1.5 text-xs text-ink-secondary">
                  Choose the <span className="font-medium text-ink">{picking.field}</span> folder
                  for {picking.side === "train" ? "training" : "evaluation"} source{" "}
                  {picking.index + 1}
                </p>
                {buckets.length > 1 && (
                  <label className="mb-1.5 flex items-center gap-2 text-xs text-ink-secondary">
                    Bucket
                    <select
                      value={pickBucket}
                      onChange={(e) => { setPickBucket(e.target.value); setPickPrefix(""); }}
                      className="tabular rounded border border-black/15 bg-page px-1.5 py-0.5 text-xs text-ink"
                    >
                      {buckets.map((b) => (
                        <option key={b} value={b}>{b}{b === homeBucket ? " (home)" : ""}</option>
                      ))}
                    </select>
                  </label>
                )}
                <Track2S3FolderBrowser
                  prefix={pickPrefix}
                  folders={pickFolders}
                  onNavigate={setPickPrefix}
                  onSelect={(p) => {
                    setSource(picking.side, picking.index, picking.field, qualified(p));
                    // Straight on to the labels for this pair. A pair with
                    // images and no labels trains on unannotated data and
                    // reports a confident zero, so it is worth walking the
                    // operator through rather than leaving it half-filled.
                    setPicking(
                      picking.field === "images"
                        ? { ...picking, field: "labels" }
                        : null,
                    );
                    setPickPrefix("");
                  }}
                  selected={
                    // The side being picked, not always the training one --
                    // reading trainSources here highlighted the wrong folder
                    // whenever an evaluation pair was being chosen.
                    (picking.side === "train" ? trainSources : evalSources)[picking.index]
                      ?.[picking.field] ?? ""
                  }
                />
                <button
                  type="button"
                  onClick={() => setPicking(null)}
                  className="mt-2 rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
                >
                  Cancel
                </button>
              </div>
            )}
            </div>
          ))}

          {/* Every chosen path in full. The rows above truncate to fit
              two per line, and an S3 key here runs to ~90 characters --
              so two folders that differ only near the end look identical
              there. This is the copy you can actually check. */}
          <div className="rounded-md border border-black/15 bg-page p-3">
            <p className="mb-1.5 text-xs font-medium text-ink">Selected folders</p>
            {([
              ["Training", trainSources],
              ["Evaluation", evalSources],
            ] as const).map(([heading, list]) => {
              const filled = list.filter((s) => s.images || s.labels);
              return (
                <div key={heading} className="mb-2 last:mb-0">
                  <p className="text-xs text-ink-muted">
                    {heading}{" "}
                    <span className="text-ink-muted/70">
                      ({filled.length} {filled.length === 1 ? "pair" : "pairs"})
                    </span>
                  </p>
                  {filled.length === 0 ? (
                    <p className="text-xs text-critical">
                      None chosen -- {heading.toLowerCase()} needs at least one pair.
                    </p>
                  ) : (
                    filled.map((source, index) => (
                      <div key={index} className="tabular mt-1 text-xs">
                        <p className="break-all text-ink-secondary">
                          <span className="text-ink-muted">{index + 1}. images </span>
                          {source.images || <span className="text-critical">not chosen</span>}
                        </p>
                        <p className="break-all text-ink-secondary">
                          <span className="text-ink-muted">&nbsp;&nbsp; labels </span>
                          {source.labels || <span className="text-critical">not chosen</span>}
                        </p>
                      </div>
                    ))
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ) : (
      <div className="grid gap-3 sm:grid-cols-2">
        {pickers.map(({ key, label, hint }) => (
          <div key={key}>
            <span className="mb-1 block text-xs text-ink-muted">
              {label} <span className="text-ink-muted/70">({hint})</span>
            </span>
            <Track2S3FolderBrowser
              prefix={prefixes[key]}
              folders={listing[key] ?? []}
              onNavigate={(p) => setPrefixes((c) => ({ ...c, [key]: p }))}
              onSelect={(p) => setFolders((c) => ({ ...c, [key]: p }))}
              selected={folders[key]}
            />
            {folders[key] && (
              <p className="tabular mt-1 truncate text-xs text-ink-secondary" title={folders[key]}>
                {folders[key]}
              </p>
            )}
          </div>
        ))}
      </div>
      )}

      {/* The live build preview. Updates on every pick, so the shape
          being produced is visible while the folders are still being
          chosen rather than only after pressing Assemble. */}
      <div className="mt-3 rounded-md border border-black/15 p-3">
        <div className="mb-2 flex items-center gap-2">
          <p className="text-xs font-medium text-ink">What will be built</p>
          {previewing && (
            <span className="text-xs text-ink-muted">checking folders...</span>
          )}
        </div>

        {!layout ? (
          <p className="text-xs text-ink-muted">
            Pick a training images folder to see the dataset take shape.
          </p>
        ) : (
          <>
            <LayoutTree
              layout={layout}
              destination={preview?.folder ?? ""}
              classes={chosen ? Object.values(chosen.classes).join(", ") : ""}
            />

            {/* One row per SOURCE. A single total would hide an empty
                folder sitting among full ones, which is the mistake this
                panel exists to catch. */}
            <div className="mt-2 flex flex-col gap-0.5">
              {layout.splits.map((split) => {
                const bad = split.image_count === 0 || split.paired === 0;
                return (
                  <p
                    key={`${split.split}-${split.source_index ?? 0}`}
                    className={`tabular text-xs ${bad ? "text-critical" : "text-ink-secondary"}`}
                  >
                    <span className="font-medium text-ink">
                      {split.split} {(split.source_index ?? 0) + 1}
                    </span>{" "}
                    <span className="break-all">{split.images_prefix || "(no folder)"}</span>
                    {" -- "}
                    {split.image_count.toLocaleString()} images,{" "}
                    <span className="font-medium">{split.paired.toLocaleString()} paired</span>
                    {split.unlabelled > 0 && `, ${split.unlabelled.toLocaleString()} without labels`}
                    {split.orphan_labels > 0
                      && `, ${split.orphan_labels.toLocaleString()} labels with no image`}
                  </p>
                );
              })}
            </div>

            <p className="tabular mt-2 text-xs text-ink">
              Total paired:{" "}
              <span className="font-medium">{layout.total_paired.toLocaleString()}</span>
            </p>

            {layout.warnings.map((warning) => (
              <p key={warning} className="mt-1 text-xs text-critical">{warning}</p>
            ))}
          </>
        )}
      </div>

      {!buildSlot && buildSection}
      </>
      )}
      {buildSlot && createPortal(
        <Card title="Build the data.yaml" collapsible={false}
              caption="From the folders picked in Assemble a dataset. Generate it before starting the run.">
          {buildSection}
        </Card>,
        buildSlot,
      )}
    </Card>
  );
}
