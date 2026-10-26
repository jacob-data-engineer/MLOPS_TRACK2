import { useEffect, useRef, useState } from "react";
import {
  ApiError,
  deleteTrack2Model,
  getTrack2Architectures,
  getTrack2Detectors,
  getTrack2Metrics,
  getTrack2Models,
  getTrack2PullControl,
  getTrack2WeightsLibrary,
  registerTrack2BaseModel,
  saveTrack2Detector,
  saveTrack2PullConfig,
  uploadTrack2BaseModelViaApi,
  verifyTrack2BaseModel,
  type Track2CheckpointInspection,
  type Track2Detector,
  type Track2ImportedModel,
  type Track2LibraryWeight,
  type Track2PendingImport,
  type Track2MetricsRow,
} from "../lib/api";
import { groupNameFor, slugFromName, slugProblem } from "../lib/slug";
import { Card } from "./Card";
import { ConfirmDialog } from "./ConfirmDialog";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

/** A class list as a stable string, for comparing a checkpoint's own
 * taxonomy against a detector's. Index order is part of the identity:
 * inference reads index 0 by position, so a swapped pair mislabels every
 * prediction without ever failing. */
function classSignature(classes: Record<string, string>): string {
  return Object.entries(classes)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([index, name]) => `${index}:${name}`)
    .join(", ");
}

/** Brings a model that was not trained here into a
 * detector's registry.
 *
 * Two steps on purpose. The checkpoint knows what it detects, so it is
 * read and shown FIRST -- picking a detector before seeing the classes
 * meant finding out they disagreed only when registration was refused.
 */
export function Track2ImportCheckpointPanel({ onImported,
  defaultArchitecture,
  lockedByPick = false,
  onPendingImportChange,
}: {
  /** The checkpoint waiting to be filed, raised so "Define a new Class"
   * below can register it into the Class it is about to create -- which
   * is the whole point of unticking the box. Null when nothing is
   * inspected, or when its own class names are being used. */
  onPendingImportChange?: (pending: Track2PendingImport | null) => void;
  /** What step 1 is set to. Used as the initial architecture so an
   * import lands in the size the base-model picker is showing -- with
   * neither, you could import yolo11s while step 1 said yolo11m and be
   * told "nothing to start from; import a checkpoint", pointing you
   * back at what you just did. */
  defaultArchitecture?: string;
  /** A Model Name is already chosen, so importing a checkpoint to create
   * one is not the next thing to do. The body dims; the button and the
   * reason stay OUTSIDE that, so the reason is still readable and the
   * dead button still says what it would have done. */
  lockedByPick?: boolean;
  /** Called after a registration or a delete. The payload is present
   * only on a registration, and is what the page needs to move step 1
   * onto the version that just appeared -- without it an import
   * succeeded and then vanished, because the base-model picker filters
   * on step 1 and step 1 was still pointing somewhere else. */
  onImported?: (registered?: {
    slug: string; architecture: string; arn: string;
  }) => void;
}) {
  const [detectors, setDetectors] = useState<Track2Detector[]>([]);
  const [architectures, setArchitectures] = useState<string[]>([]);
  const [architecture, setArchitecture] = useState("");
  const [source, setSource] = useState<"upload" | "bucket" | "library">("upload");
  // The mirrored official releases. Listed from S3 rather than hardcoded:
  // the folder is seeded by a script, and offering a file whose upload
  // failed sends somebody looking for something that is not there.
  const [library, setLibrary] = useState<Track2LibraryWeight[]>([]);
  const [libraryError, setLibraryError] = useState("");
  const [file, setFile] = useState<File | null>(null);
  // Browsing the bucket for a .pt that is already there.
  const [bucketKey, setBucketKey] = useState("");
  const [prefix, setPrefix] = useState("");
  const [folders, setFolders] = useState<string[]>([]);
  const [weightFiles, setWeightFiles] = useState<string[]>([]);
  const [inspection, setInspection] = useState<Track2CheckpointInspection | null>(null);
  const [detectorSlug, setDetectorSlug] = useState("");
  const [description, setDescription] = useState("");
  // ONE field, as in "Define a new Class". The identifier is derived
  // from the name by the same `slugFromName` that card uses -- two
  // derivations would file the same typed name under two different
  // registry groups depending on which card created it.
  const [newLabel, setNewLabel] = useState("");
  // Either the Class comes from the names baked into the .pt, or it is
  // defined by hand in "Define a new Class" below. Exclusive, because
  // they are two answers to one question -- what does this model detect
  // -- and the registry only records one of them.
  const [useFileClasses, setUseFileClasses] = useState(true);
  // See the matching block in Track2DatasetAssemblePanel. Only the
  // STANDING choices are saved: a chosen file, an inspection and a
  // registration result belong to one action, and restoring them would
  // show a finished import that never happened on this machine.
  const [settingsRead, setSettingsRead] = useState(false);
  // Which of the file's classes the new Model Name should declare.
  //
  // A COCO checkpoint carries eighty and you almost never want eighty: the
  // point of importing one is usually "start from this, but I only care
  // about person and truck". Empty means nothing picked yet; it is filled
  // with every index the moment a file is read, so the default is still
  // "all of them" and ticking nothing is a state you have to choose.
  const [pickedClasses, setPickedClasses] = useState<string[]>([]);
  //: The class ticks a restore is carrying, handed to the effect that
  //: would otherwise tick every class the moment an inspection lands.
  //: A ref, not state: it is read once and cleared, and a re-render in
  //: between would be a re-render that undid the restore.
  const restoredPicks = useRef<string[] | null>(null);
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState<Track2ImportedModel | null>(null);
  // What is already registered, so a failed or unwanted import can be seen
  // and removed here rather than hunted for on another page.
  const [existing, setExisting] = useState<Track2MetricsRow[]>([]);
  const [registryVersion, setRegistryVersion] = useState(0);
  // The registered version the delete dialog is asking about. Deleting
  // one is not recoverable, and the row it is on is a line of small
  // text -- so the dialog names the version and the registry rather
  // than asking "are you sure".
  const [deleteRow, setDeleteRow] = useState<Track2MetricsRow | null>(null);

  function loadDetectors() {
    return getTrack2Detectors()
      .then((res) => {
        const own = res.detectors.filter((d) => !d.builtin);
        setDetectors(own);
        return own;
      })
      .catch(() => [] as Track2Detector[]);
  }

  useEffect(() => {
    getTrack2PullControl()
      .then((status) => {
        const saved = status.config.import_checkpoint || {};
        if (saved.architecture) setArchitecture(saved.architecture);
        if (saved.source === "upload" || saved.source === "bucket"
            || saved.source === "library") setSource(saved.source);
        if (saved.bucket_key) setBucketKey(saved.bucket_key);
        if (saved.prefix) setPrefix(saved.prefix);
        if (saved.detector_slug) setDetectorSlug(saved.detector_slug);
        if (saved.new_label) setNewLabel(saved.new_label);
        if (saved.description) setDescription(saved.description);
        if (typeof saved.use_file_classes === "boolean") {
          setUseFileClasses(saved.use_file_classes);
        }

        // Read the checkpoint again rather than storing what it said.
        //
        // The classes a .pt declares are a property of the file, so
        // re-reading cannot be stale -- and it means the card comes back
        // showing them without anybody pressing Read classes, which is
        // the step that made a reloaded page feel unsaved.
        //
        // `read_key` and not `bucket_key`: after an upload the file IS in
        // the bucket, so the key the read returned restores all three
        // sources, including the one whose original file lives on
        // somebody's laptop and cannot come back.
        const key = saved.read_key || saved.bucket_key || "";
        const architectureFor = saved.architecture || "";
        if (key && architectureFor) {
          // Held for the effect below, which otherwise ticks every class
          // the moment an inspection appears.
          restoredPicks.current = saved.picked_classes ?? null;
          verifyTrack2BaseModel(key, architectureFor)
            .then(setInspection)
            .catch(() => {
              // Moved, deleted, or renamed since. The card falls back to
              // "pick a file", which is the honest state -- better than
              // showing classes for something that is no longer there.
              restoredPicks.current = null;
            });
        }
      })
      .catch(() => undefined)
      .finally(() => setSettingsRead(true));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!settingsRead) return;
    const timer = setTimeout(() => {
      void saveTrack2PullConfig({
        import_checkpoint: {
          architecture, source, bucket_key: bucketKey, prefix,
          detector_slug: detectorSlug, use_file_classes: useFileClasses,
          // What was read and what was ticked. The classes are a CHOICE,
          // not a result: "these four of the eighty" is the whole point
          // of the picker, and losing it on reload meant making it again.
          read_key: inspection?.key || "",
          picked_classes: pickedClasses,
          new_label: newLabel,
          description,
        },
      }).catch(() => undefined);
    }, 600);
    return () => clearTimeout(timer);
  }, [settingsRead, architecture, source, bucketKey, prefix, detectorSlug,
      useFileClasses, inspection, pickedClasses, newLabel, description]);

  useEffect(() => {
    loadDetectors();
    getTrack2Architectures()
      .then((res) => {
        setArchitectures(res.architectures);
        setArchitecture((current) =>
          current || defaultArchitecture || res.architectures[0] || "");
      })
      .catch(() => undefined);
  }, []);

  // Only while browsing: the listing is per prefix, so navigating has to
  // refetch or the folder never actually opens.
  useEffect(() => {
    getTrack2WeightsLibrary()
      .then((r) => { setLibrary(r.weights); setLibraryError(""); })
      .catch((err) => setLibraryError(
        err instanceof ApiError ? err.detail : "Could not read the weights library."));
  }, []);

  useEffect(() => {
    if (source !== "bucket") return;
    let cancelled = false;
    getTrack2Models(undefined, prefix, undefined)
      .then((res) => {
        if (cancelled) return;
        setFolders(res.weights_browser.folders);
        setWeightFiles(res.weights_browser.weight_files ?? []);
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [source, prefix]);

  // Both halves, parsed. Stripping the literal "yolo11" is a no-op for
  // any other generation, so a yolo26 import yielded a size of
  // "yolo26m" -- and the base-model picker, which filters on step 1's
  // generation AND size, could never show the version just imported.
  const parsedArch = /^yolo(\d+)([a-z]+)$/.exec(architecture.trim().toLowerCase());
  const generation = parsedArch?.[1] ?? "";
  const size = parsedArch?.[2] ?? "";

  // The identifier the typed name becomes, from the same helper "Define
  // a new Class" uses. One derivation, or the same typed name would
  // land in two different registry groups depending on which card
  // created the Class.
  const newSlug = slugFromName(newLabel);
  const newLabelProblem = slugProblem(newLabel);

  // The size dropdown stays editable after a file is read, so the
  // declaration can now drift from the file -- read yolo11s.pt, switch
  // the dropdown to yolo11m, and the checkpoint would be filed in the
  // yolo11m group. Freezing the control was the only thing preventing
  // that before; this is what replaces it.
  //
  // A file that does not RECORD its architecture is not a mismatch: a
  // fork or an older Ultralytics may simply not say, and refusing over
  // silence would be worse than recording what we do know.
  const architectureMismatch = Boolean(
    inspection?.detected_architecture
    && inspection.detected_architecture !== architecture);


  // Raised only while the box is OFF: with it on, this card registers the
  // checkpoint itself and a second button below doing the same thing
  // would be two ways to perform one action, which is how they drift.
  const pendingImport: Track2PendingImport | null =
    inspection && !useFileClasses
      ? {
          key: inspection.key,
          architecture,
          description,
          checkpointClasses: Object.entries(inspection.classes)
            .sort((a, b) => Number(a[0]) - Number(b[0]))
            .map(([, name]) => name),
        }
      : null;
  // Every class, whenever a different file is read. Done here rather than
  // beside each setInspection call: there are three sources (upload,
  // bucket, stock release) and a fourth would have been added without
  // this.
  //
  // Except on a RESTORE, where the whole point is the classes that were
  // ticked when the file was saved. Without the guard, re-reading the
  // checkpoint on load would tick all eighty again and the selection
  // would be lost in the act of restoring it.
  useEffect(() => {
    const available = Object.keys(inspection?.classes ?? {});
    const restored = restoredPicks.current;
    if (restored) {
      restoredPicks.current = null;
      // Intersected, not trusted: a checkpoint can be replaced at the
      // same key by one with different classes, and ticking an index it
      // no longer has would train against a class that is not there.
      const kept = restored.filter((index) => available.includes(index));
      setPickedClasses(kept.length ? kept : available);
      return;
    }
    setPickedClasses(available);
  }, [inspection]);

  const pendingKey = pendingImport
    ? `${pendingImport.key}|${pendingImport.architecture}|${pendingImport.description}`
    : "";
  useEffect(() => {
    onPendingImportChange?.(pendingImport);
    // Keyed on the VALUE, not the object: rebuilt every render, it would
    // otherwise fire this effect on every keystroke anywhere in the card.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingKey, onPendingImportChange]);

  // Scoped to the detector and size in the form, which together name one
  // group -- listing everything would bury the one being imported into.
  useEffect(() => {
    if (!detectorSlug) {
      setExisting([]);
      return;
    }
    let cancelled = false;
    getTrack2Metrics()
      .then((res) => {
        if (cancelled) return;
        setExisting(res.rows.filter((r) => r.registry === detectorSlug && r.size === size));
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [detectorSlug, size, registryVersion]);

  async function handleDeleteExisting(row: Track2MetricsRow) {
    setBusy("Deleting...");
    setError("");
    try {
      await deleteTrack2Model(row.model_package_arn);
      setDeleteRow(null);
      setRegistryVersion((n) => n + 1);
      // No payload: a delete leaves nothing to select.
      onImported?.();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not delete that version.");
    } finally {
      setBusy("");
    }
  }

  // The classes the new Model Name will declare: the ones ticked,
  // RENUMBERED from zero.
  //
  // Renumbered because a Model Name's indices are its own. Keeping COCO's
  // numbering would declare a taxonomy with holes in it -- {0: person,
  // 7: truck} -- and every label file written against it would have to
  // know that 1 through 6 are missing. Training derives a filtered copy
  // of the dataset to match, which is what `class_filter_for` does
  // server-side.
  //
  // The consequence is worth being clear about rather than hiding: once
  // this is a subset, the WEIGHTS still predict all eighty in COCO order
  // while the registry says it detects two. That is a seed, and the card
  // says so below. The head is rebuilt on the first training run.
  const allFileClasses = inspection?.classes ?? {};
  const found: Record<string, string> = Object.fromEntries(
    Object.keys(allFileClasses)
      .filter((index) => pickedClasses.includes(index))
      .sort((a, b) => Number(a) - Number(b))
      .map((index, position) => [String(position), allFileClasses[index]]),
  );
  const foundSignature = classSignature(found);
  // Only detectors whose taxonomy actually matches the file can take it.
  // The Class chosen in the dropdown, and whether filing the checkpoint
  // into it is a SEED -- its weights detecting something other than what
  // that Class declares. Not a reason to refuse: training rebuilds the
  // head for whatever the dataset says, and giving an existing Class its
  // first checkpoint to train from is exactly what a seed is for. It IS
  // a reason to say so, and to register it Pending.
  const chosenDetector = detectors.find((d) => d.slug === detectorSlug);
  //: A subset is a mismatch too.
  //
  //: The server refuses to file a checkpoint under a Model Name whose
  //: classes differ from the file's, unless it is told the mismatch is
  //: deliberate. Creating a NEW Model Name from the whole file never
  //: tripped that -- the two were equal by construction. Picking a
  //: SUBSET breaks that equality: the Model Name says two, the weights
  //: still predict eighty. Without this, Register came back with "this
  //: checkpoint detects person, bicycle, car ... but the detector is
  //: person, truck" and no way forward.
  const subsetChosen = useFileClasses
    && inspection != null
    && pickedClasses.length > 0
    && pickedClasses.length < Object.keys(inspection.classes).length;

  // A subset is NOT a seed any more. The server cuts the head down to the
  // picked classes before registering, so the weights and the Model Name
  // agree by construction. It was a seed only while the file was
  // registered whole and the disagreement left for a training run to fix.
  const seedForChosen = Boolean(
    chosenDetector && classSignature(chosenDetector.classes) !== foundSignature);

  async function handleRead() {
    if (!architecture) return;
    if (source === "upload" ? !file : !bucketKey) return;
    setError("");
    setResult(null);
    setBusy(source === "upload" ? "Uploading and reading..." : "Reading...");
    try {
      const inspected = source === "upload"
        ? await uploadTrack2BaseModelViaApi(file as File, architecture)
        // Already in the bucket, so there is nothing to upload -- it is
        // read in place.
        : await verifyTrack2BaseModel(bucketKey, architecture);
      setInspection(inspected);
      // Left EMPTY, which is "create a new Class from this file".
      // Preselecting the first match filed a checkpoint into a Class
      // nobody had looked at, and now that every Class is offered the
      // same slip would reach one with different classes entirely.
      setDetectorSlug("");
      const fits = detectors.some(
        (d) => classSignature(d.classes) === classSignature(inspected.classes),
      );
      if (!fits) {
        // Suggest a name from the classes themselves, so the common case
        // -- a single-class checkpoint with no Class yet -- is one click
        // rather than a naming decision.
        //
        // Only when there is exactly ONE class. A name built from four of
        // them -- "horizontal_plane, vertical_plane, hatch_wall,
        // dangerous_plane" -- is unreadable in the picker and would
        // derive an equally unreadable registry group, and the class
        // list is shown just above this box anyway.
        const names = Object.entries(inspected.classes)
          .sort((a, b) => Number(a[0]) - Number(b[0]))
          .map(([, name]) => name);
        setNewLabel(names.length === 1 ? names[0] : "");
      }
      // A sensible default the operator can overwrite -- an imported model
      // with no description is indistinguishable from a trained one whose
      // notes were forgotten.
      setDescription(
        `Imported ${inspected.detected_architecture || architecture} checkpoint, ` +
        `${Object.keys(inspected.classes).length} classes.`,
      );
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not read that checkpoint.");
    } finally {
      setBusy("");
    }
  }

  /** Creates the detector when there isn't one, then registers -- one
   * action, not two. Splitting them left a disabled Register button whose
   * only cue was a line of text, which read as a broken button. */
  async function handleRegister() {
    if (!inspection) return;
    if (!detectorSlug && !newSlug) return;
    setError("");
    try {
      let slug = detectorSlug;
      if (!slug) {
        setBusy("Creating the Model Name...");
        // The server normalises the slug, so take its answer rather than
        // assuming what was typed survived unchanged.
        const created = await saveTrack2Detector(newSlug, newLabel.trim() || newSlug, found);
        slug = created.slug;
        await loadDetectors();
        setDetectorSlug(slug);
        // Clearing the name clears the derived identifier with it.
        setNewLabel("");
      }
      setBusy("Registering...");
      const registered = await registerTrack2BaseModel(
        inspection.key, slug, architecture, description,
        // Only when the chosen Class really does declare something else.
        // A new Class is built FROM this file's classes, so it never
        // mismatches and must not be waved through as though it might.
        seedForChosen,
        // The ORIGINAL indices, so the server can cut the head down to
        // them. A strict subset then detects exactly what the Model Name
        // declares -- no seed, no training run to reconcile them.
        subsetChosen ? pickedClasses.map(Number).sort((a, b) => a - b) : [],
      );
      setResult(registered);
      setInspection(null);
      setFile(null);
      setBucketKey("");
      setDescription("");
      setRegistryVersion((n) => n + 1);
      // What was registered, so step 1 can move onto it. An import that
      // succeeded and then could not be found is the same to the
      // operator as one that failed.
      onImported?.({ slug, architecture, arn: registered.model_package_arn });
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not register the checkpoint.");
    } finally {
      setBusy("");
    }
  }

  return (
    // Collapsed by default. Importing a checkpoint is occasional -- the
    // caption already says optional -- and this card is tall enough that
    // open it pushes the training cards off the screen.
    //
    // The collapse is Card's. This ran its own as well, so the header
    // carried two Show/Hide buttons; the same duplication was reported
    // on the Register card and is the same fix.
    <Card
      title="Import a pretrained COCO or an already-trained model"
      caption="Reads a .pt trained elsewhere -- an official COCO release like yolo11s.pt, or weights somebody already trained -- shows what it detects, and registers it as a version in a Model Name's registry. Optional: skip it if the Model Name already has a version to continue from."
      defaultOpen={false}
    >
      {/* NOT one of the exclusive options, and never greyed -- the same
          treatment as "Define a new Model Name" above it.

          Importing does not TRAIN. It files a .pt as a version so that
          "Train on an existing model" has something to offer, which is
          why a successful import moves the selection there. Holding a
          "Use this" tick made it a third way of training, so the card
          sat greyed out until you ticked it, and ticking it greyed out
          the two modes that do train -- to do something that is not a
          mode at all. */}
      <div className={lockedByPick ? "pointer-events-none opacity-40" : ""}>

      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {result && (
        <p className="tabular mb-2 text-xs text-ink-secondary">
          Registered as <span className="font-medium text-ink">v{result.version}</span> in{" "}
          {result.group_name}.
        </p>
      )}

      {/* --- step 1: read it ------------------------------------------- */}
      <div className="mb-2 flex flex-wrap gap-3">
        {([
          ["upload", "Upload from my computer"],
          ["bucket", "Pick one already in the bucket"],
          // The usual starting point for a brand new Class is the stock
          // COCO checkpoint, and before this that meant finding it on
          // the internet and uploading it by hand every time.
          ["library", "Use a stock YOLO release"],
        ] as const).map(([value, label]) => (
          <label key={value} className="flex items-center gap-1.5 text-xs text-ink-secondary">
            <input
              type="radio"
              name="importSource"
              checked={source === value}
              onChange={() => {
                setSource(value);
                setFile(null);
                setBucketKey("");
                // A different source is a different FILE, so whatever
                // was read no longer describes what is selected. Cleared
                // rather than left on screen: a panel of classes that
                // belong to a file you are no longer pointing at is the
                // worst of the three states this could be in.
                setInspection(null);
                setError("");
              }}
            />
            {label}
          </label>
        ))}
      </div>

      <label className="block sm:w-40">
        <span className="mb-1 block text-xs text-ink-muted">Model size</span>
        <select
          className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
          value={architecture}
          // NOT cleared by this one. The size is a DECLARATION about the
          // file, not a choice of file -- changing it leaves the reading
          // valid and only changes what it is being compared against,
          // which the mismatch line below recomputes live.
          onChange={(e) => setArchitecture(e.target.value)}
        >
          {architectures.map((a) => (
            <option key={a} value={a}>{a}</option>
          ))}
        </select>
      </label>

      {source === "library" ? (
        <div className="mt-2">
          <span className="mb-1 block text-xs text-ink-muted">
            Stock release{" "}
            <span className="text-ink-muted/70">
              (official COCO weights, mirrored into this bucket)
            </span>
          </span>
          {libraryError ? (
            <p className="text-xs text-critical">{libraryError}</p>
          ) : library.length === 0 ? (
            <p className="text-xs text-ink-muted">
              Nothing mirrored yet. Seed the folder with{" "}
              <span className="tabular">scripts/mirror_yolo_weights.py</span>,
              or upload a checkpoint instead.
            </p>
          ) : (
            <select
              className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
              value={bucketKey}
              onChange={(e) => {
                setBucketKey(e.target.value);
                setInspection(null);
                setError("");
                // The release IS its architecture -- `yolo11m.pt` is a
                // yolo11m. Setting it from the file rather than leaving
                // the size dropdown on whatever it was removes the one
                // way this source can be got wrong.
                const picked = library.find((w) => w.key === e.target.value);
                if (picked) setArchitecture(picked.architecture);
              }}
            >
              <option value="">Pick a release</option>
              {library.map((w) => (
                <option key={w.key} value={w.key}>
                  {w.architecture}.pt &middot;{" "}
                  {(w.size_bytes / (1024 * 1024)).toFixed(1)} MB
                </option>
              ))}
            </select>
          )}
        </div>
      ) : source === "upload" ? (
        <label className="mt-2 block">
          <span className="mb-1 block text-xs text-ink-muted">
            Checkpoint <span className="text-ink-muted/70">(.pt)</span>
          </span>
          <input
            type="file"
            accept=".pt"
            className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
            onChange={(e) => {
              setFile(e.target.files?.[0] ?? null);
              setInspection(null);
              setError("");
            }}
          />
        </label>
      ) : (
        <div className="mt-2">
          <span className="mb-1 block text-xs text-ink-muted">
            Checkpoint in the bucket{" "}
            <span className="text-ink-muted/70">
              (navigate or search, then pick a .pt)
            </span>
          </span>
          {!inspection && (
            <Track2S3FolderBrowser
              prefix={prefix}
              folders={folders}
              onNavigate={setPrefix}
              files={weightFiles}
              onSelectFile={(key) => {
                setBucketKey(key);
                // Same reason as the other sources: a different file
                // means the reading on screen is about something else.
                setInspection(null);
                setError("");
              }}
              selectedFile={bucketKey}
            />
          )}
          {bucketKey && (
            <p className="tabular mt-1 text-xs text-ink-secondary">{bucketKey}</p>
          )}
        </div>
      )}

      {!inspection ? (
        <>
          <button
            type="button"
            disabled={!!busy || !architecture || (source === "upload" ? !file : !bucketKey)}
            onClick={handleRead}
            className="mt-2 rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
          >
            {busy || (source === "upload" ? "Upload and read classes" : "Read classes")}
          </button>
          {busy && source === "upload" && (
            <p className="mt-1 text-xs text-ink-muted">
              The file goes through this app rather than straight to S3 -- the
              bucket only accepts browser uploads from SageMaker origins -- so a
              large checkpoint takes a while.
            </p>
          )}
        </>
      ) : (
        <>
          {/* --- what the file says about itself ------------------------ */}
          <div className="mt-3 rounded-md border border-black/15 bg-page p-3">
            <p className="text-xs font-medium text-ink">
              Available classes in the .pt file
            </p>
            {/* A TABLE, not the comma-separated run this was. A COCO
                checkpoint carries eighty classes, and
                "0:person, 1:bicycle, 2:car, ..." wrapped across six
                lines is unreadable in exactly the place where the index
                matters most -- a label row carries the NUMBER, never the
                name, so index 2 meaning "car" here is the fact being
                checked.

                Scrolls past about ten rows rather than pushing the rest
                of the card off the screen. The usual case is one to six
                classes, which fits without scrolling at all. */}
            <div className="mt-1 max-h-56 overflow-y-auto rounded-sm border border-hairline">
              <table className="w-full text-left text-xs">
                <thead className="sticky top-0 bg-page text-ink-muted">
                  <tr className="border-b border-hairline">
                    <th className="w-8 py-1 pl-2 pr-1 font-medium">
                      {/* All or none. Indeterminate when it is neither,
                          so the header reports the state rather than
                          guessing at one. */}
                      <input
                        type="checkbox"
                        className="h-3.5 w-3.5"
                        disabled={!useFileClasses}
                        checked={pickedClasses.length === Object.keys(inspection.classes).length}
                        ref={(box) => {
                          if (box) {
                            box.indeterminate =
                              pickedClasses.length > 0
                              && pickedClasses.length < Object.keys(inspection.classes).length;
                          }
                        }}
                        onChange={(e) => setPickedClasses(
                          e.target.checked ? Object.keys(inspection.classes) : [],
                        )}
                        title="Use all of them, or none"
                      />
                    </th>
                    <th className="w-12 py-1 pl-1 pr-3 font-medium">No</th>
                    {/* The LABELS inside the .pt, not the Model Name it is
                        being filed under -- those are two different things
                        and this column is the first one. */}
                    <th className="py-1 pr-2 font-medium">Class</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(inspection.classes)
                    .sort((a, b) => Number(a[0]) - Number(b[0]))
                    .map(([index, name]) => {
                      const on = pickedClasses.includes(index);
                      return (
                        <tr key={index}
                            className={`border-b border-hairline/50 last:border-0 ${
                              useFileClasses && !on ? "opacity-40" : ""}`}>
                          <td className="py-1 pl-2 pr-1">
                            <input
                              type="checkbox"
                              className="h-3.5 w-3.5"
                              disabled={!useFileClasses}
                              checked={on}
                              onChange={(e) => setPickedClasses((current) => (
                                e.target.checked
                                  ? [...current, index]
                                  : current.filter((i) => i !== index)
                              ))}
                            />
                          </td>
                          <td className="tabular py-1 pl-1 pr-3 text-ink-muted">{index}</td>
                          <td className="tabular py-1 pr-2 text-ink">{name}</td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
            <p className="mt-1 text-xs text-ink-muted">
              {useFileClasses ? (
                <>
                  <span className="tabular text-ink">{pickedClasses.length}</span>
                  {" of "}
                  <span className="tabular">{Object.keys(inspection.classes).length}</span>
                  {" selected"}
                </>
              ) : (
                <>
                  {Object.keys(inspection.classes).length} class
                  {Object.keys(inspection.classes).length === 1 ? "" : "es"}
                </>
              )}
            </p>
            {/* Two states worth saying out loud rather than leaving the
                Register button mysteriously dead or quietly producing
                something other than what the table shows. */}
            {useFileClasses && pickedClasses.length === 0 && (
              <p className="mt-1 text-xs text-critical">
                Nothing selected, so there is no taxonomy to create. Tick at
                least one class.
              </p>
            )}
            {useFileClasses
              && pickedClasses.length > 0
              && pickedClasses.length < Object.keys(inspection.classes).length && (
              <p className="mt-1 rounded-md border border-accent/40 bg-accent-subtle px-2.5 py-2 text-xs text-ink-secondary">
                A <strong>new checkpoint</strong> will be written detecting
                just these{" "}
                <span className="tabular text-ink">{pickedClasses.length}</span>,
                renumbered from 0 &mdash;{" "}
                <span className="tabular">
                  {Object.values(found).slice(0, 6).join(", ")}
                  {Object.keys(found).length > 6 ? ", ..." : ""}
                </span>.
                {" "}The head is cut to them, so those classes predict exactly
                what they predict now &mdash; nothing is learned or lost, the
                other outputs are simply removed. It registers as a normal
                version, <strong>not a seed</strong>, and can be deployed
                without a training run. The original file is left alone.
              </p>
            )}
            <p className="tabular mt-1.5 text-xs text-ink-muted">
              {inspection.detected_architecture
                ? `Architecture ${inspection.detected_architecture}`
                : "Architecture not recorded in the file"}
              {" · "}
              {(inspection.size_bytes / (1024 * 1024)).toFixed(1)} MB
            </p>
            {/* Recomputed here, not read off the inspection. That field
                was baked server-side against the size as it stood when
                the file was READ -- and the size dropdown now stays
                editable, so a stale boolean would keep saying "matches"
                about a comparison nobody is making any more.

                `detected_architecture` IS a fact about the file, so it
                is the half that keeps coming from the server. An empty
                one means the file does not say, which is not a
                mismatch. */}
            {architectureMismatch && (
              <p className="mt-1 text-xs text-critical">
                This file is a {inspection.detected_architecture}, but{" "}
                <em>Model size</em> says {architecture}. Registering it under
                the wrong size follows it through the registry for ever, so
                that is refused &mdash; set the size to{" "}
                {inspection.detected_architecture} and it will register.
              </p>
            )}

            {/* The either/or. Ticked, the Model Name is these names; unticked,
                "Define a new Model Name" below comes alive and you type them
                yourself. Two answers to one question -- what does this
                model detect -- and the registry records only one, so
                both being live at once is the state that produces a
                version filed under a taxonomy nobody chose. */}
            <label className="mt-2 flex items-start gap-2 border-t border-hairline pt-2 text-xs text-ink-secondary">
              <input
                type="checkbox"
                className="mt-0.5 shrink-0"
                checked={useFileClasses}
                onChange={(e) => setUseFileClasses(e.target.checked)}
              />
              <span>
                <span className="font-medium text-ink">
                  Use the classes in this file
                </span>
                <span className="ml-1.5 text-ink-muted">
                  {useFileClasses
                    ? "-- the Model Name is named below, and detects exactly these."
                    : "-- off, so define the Model Name yourself in Define a new Model Name below."}
                </span>
              </span>
            </label>
          </div>

          {/* --- step 2: where it goes ---------------------------------- */}
          {/* EVERY Model Name, not only the ones whose class names match.
              Filtering to matches made the whole point of a seed
              unreachable: a COCO checkpoint detects eighty things and
              matches nothing, so the only offer was "create a new
              eighty-class Class" -- and giving an EXISTING Class its
              first checkpoint to train from, which is what a seed IS,
              could not be done at all.

              A mismatch is not hidden, it is labelled: the option says
              so, the panel below says so, and the server marks the
              version a seed and registers it Pending. */}
          {useFileClasses ? (
            <label className="mt-2 block">
              <span className="mb-1 block text-xs text-ink-muted">Model Name</span>
              <select
                className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                value={detectorSlug}
                onChange={(e) => setDetectorSlug(e.target.value)}
              >
                <option value="">Create a new Model Name from this file</option>
                {detectors.filter((d) => !d.builtin).map((d) => (
                  <option key={d.slug} value={d.slug}>
                    {d.label}
                    {classSignature(d.classes) === foundSignature
                      ? " -- same classes"
                      : ` -- SEED, it detects ${Object.keys(d.classes).length} other class`
                        + `${Object.keys(d.classes).length === 1 ? "" : "es"}`}
                  </option>
                ))}
              </select>
            </label>
          ) : null}

          {detectorSlug && seedForChosen && (
            <div className="mt-2 rounded-md border border-critical/30 bg-critical-subtle p-3 text-xs text-critical">
              <p>
                <strong>This will register as a seed.</strong> The file detects{" "}
                <span className="tabular">{Object.keys(inspection.classes).length} classes</span>,
                and <span className="tabular">{chosenDetector?.label}</span> is{" "}
                <span className="tabular">
                  {Object.values(chosenDetector?.classes ?? {}).join(", ")}
                </span>.
              </p>
              <p className="mt-1">
                The head is rebuilt on the first training run, so those names do
                not survive it &mdash; but until that run the weights and the
                registry disagree. It is marked in its description and left
                Pending rather than Ready. Do not deploy it as it stands.
              </p>
            </div>
          )}

          {useFileClasses && !detectorSlug ? (
            // Nothing matches, so there is nowhere to put it. Building
            // the Class from the file's own classes is the only route
            // that cannot produce a mismatch.
            <div className="mt-2 rounded-md border border-black/15 p-3">
              <p className="text-xs text-ink-secondary">
                Input a name for the model with these classes &mdash; for
                example <span className="tabular text-ink">Helmet</span> or{" "}
                <span className="tabular text-ink">Forklift</span>.
              </p>
              <input
                className="mt-2 w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                value={newLabel}
                placeholder="Name, e.g. Forklift"
                onChange={(e) => setNewLabel(e.target.value)}
              />
              {newLabelProblem ? (
                <p className="mt-1 text-xs text-critical">{newLabelProblem}</p>
              ) : newSlug ? (
                <p className="tabular mt-1 text-xs text-ink-muted">
                  Registers into{" "}
                  <span className="text-ink">
                    {groupNameFor(newSlug, generation || "11", size || "s")}
                  </span>
                </p>
              ) : null}
            </div>
          ) : (
            // The box is off, so the Class is being defined by hand below
            // and this half stands down. Said rather than left blank: a
            // section that simply vanished would read as the card having
            // lost its way to register anything.
            <div className="mt-2 rounded-md border border-black/15 p-3">
              <p className="text-xs text-ink-secondary">
                <strong>Define a new Model Name</strong> below is now live. Give it
                a name and its class names, in index order, and press{" "}
                <em>Save classes</em> &mdash; then come back here and it will
                be offered above.
              </p>
              <p className="mt-1 text-xs text-ink-muted">
                Its class names have to match what this file detects, in the
                same order. A label row carries{" "}
                <span className="tabular">2</span> and never
                &ldquo;vehicle&rdquo;, so a Model Name whose order differs
                mislabels every box this model draws.
              </p>
            </div>
          )}

          <label className="mt-2 block">
            <span className="mb-1 block text-xs text-ink-muted">Description</span>
            <input
              className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
              value={description}
              placeholder="Where these weights came from"
              onChange={(e) => setDescription(e.target.value)}
            />
          </label>

          {!lockedByPick && !detectorSlug && (useFileClasses ? !newSlug : true) && (() => {
            // Whether any Model Name exists at all. The message used to
            // say "define one and save it" whatever the answer was, so
            // somebody with six already defined was told to go and make
            // another. What is actually missing in that case is the
            // CHOICE above, not the Model Name.
            const defined = detectors.filter((d) => !d.builtin);
            if (useFileClasses) {
              return (
                <p className="mt-2 text-xs text-critical">
                  Name the model above -- every registered version belongs to a
                  Model Name, and this is what creates it.
                </p>
              );
            }
            if (defined.length) {
              // The same sentence the define card shows, word for word.
              // Two different wordings for one state read as two
              // different states.
              return (
                <p className="mt-2 text-xs font-medium text-info">
                  A model has already been selected.
                </p>
              );
            }
            return (
              <p className="mt-2 text-xs text-critical">
                Define the Model Name below and save it, then choose it above --
                every registered version belongs to one.
              </p>
            );
          })()}
          {/* Already there. Shown before registering so a duplicate is
              obvious, and removable here so a bad import does not have to
              be hunted down on another page. */}
          {existing.length > 0 && (
            <div className="mt-2 rounded-md border border-black/15 p-2.5">
              <p className="text-xs text-ink-muted">
                Already in {existing[0].group_name}:
              </p>
              {existing.map((row) => (
                <div key={row.model_package_arn} className="mt-1 flex items-center justify-between gap-3">
                  <span className="tabular min-w-0 flex-1 truncate text-xs text-ink-secondary">
                    v{row.version} &middot; {row.status}
                    {row.description ? ` -- ${row.description}` : ""}
                  </span>
                  <button
                    type="button"
                    disabled={!!busy}
                    onClick={() => setDeleteRow(row)}
                    className="shrink-0 text-xs font-medium text-critical underline disabled:opacity-40"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          )}

          {!lockedByPick && (
          <div className="mt-2 flex gap-2">
            <button
              type="button"
              disabled={!!busy || architectureMismatch
                // Nothing ticked is no taxonomy to register. Said above
                // the button too, so a dead button has a reason on
                // screen rather than only in a tooltip nobody hovers.
                || (useFileClasses && !!inspection && pickedClasses.length === 0)
                || (!detectorSlug
                  && (!useFileClasses || !newSlug || !!newLabelProblem))}
              onClick={handleRegister}
              className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
            >
              {busy || (detectorSlug ? "Register this version" : "Register New Model Name")}
            </button>
          </div>
          )}
        </>
      )}

      </div>

      {/* OUTSIDE the dimmed body above, so it stays readable while
          everything else fades. The button is kept and dead rather than
          hidden: a button that disappears leaves nothing to explain why
          the card does nothing. */}
      {lockedByPick && (
        <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
          <button
            type="button"
            disabled
            className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white opacity-40"
          >
            {detectorSlug ? "Register this version" : "Register New Model Name"}
          </button>
          <p className="text-xs font-medium text-info">
            A model has already been selected.
          </p>
        </div>
      )}
      <ConfirmDialog
        open={Boolean(deleteRow)}
        title={`Delete version ${deleteRow?.version}?`}
        destructive
        busy={busy === "Deleting..."}
        confirmLabel="Delete this version"
        onCancel={() => setDeleteRow(null)}
        onConfirm={() => { if (deleteRow) void handleDeleteExisting(deleteRow); }}
        body={
          <>
            <p>
              Removes version <span className="tabular">{deleteRow?.version}</span>{" "}
              from <span className="tabular">{deleteRow?.group_name}</span>.
            </p>
            <p className="mt-2">
              The weights in S3 stay where they are; what goes is the
              registered version pointing at them, so no run can continue
              from it and nothing can be rolled back to it.
            </p>
            <p className="mt-2">This cannot be undone.</p>
          </>
        }
      />

    </Card>
  );
}
