import { useCallback, useEffect, useState, type ReactNode } from "react";
import { groupNameFor, slugFromName, slugProblem } from "../lib/slug";
import {
  ApiError,
  deleteTrack2Detector,
  listTrack2YamlFiles,
  readTrack2ClassesFromYamlText,
  readTrack2YamlClasses,
  getTrack2Detectors,
  getTrack2PullControl,
  saveTrack2PullConfig,
  getTrack2Buckets,
  registerTrack2BaseModel,
  saveTrack2Detector,
  renameTrack2Detector,
  type Track2Bucket,
  type Track2Detector,
  type Track2PendingImport,
  type Track2Generation,
} from "../lib/api";
import { Card } from "./Card";
import { ConfirmDialog } from "./ConfirmDialog";
import { Track2TrainingAdvanced, effectiveHyperparams }
  from "./Track2TrainingAdvanced";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";

/** What a detector's model is, when nothing has been chosen. */
const FALLBACK_GENERATIONS: Track2Generation[] = [
  {
    generation: "11",
    label: "YOLO11",
    sizes: ["s", "m", "l", "x"],
    size_labels: { s: "Small", m: "Medium", l: "Large", x: "Extra-Large" },
  },
];

/** Defines what each YOLO model type detects.
 *
 * A detector is a name plus a class list. The class list is the thing that
 * stops a vehicle model being read as person/helmet: a run is refused
 * unless its dataset declares exactly these classes, and every version
 * trained for a detector registers into that detector's own per-size
 * groups, so version numbers mean something per model type.
 */
export function Track2DetectorsPanel({
  onChanged,
  onChoose,
  onTraining,
  children,
  aboveDefine,
  defineDimmed = false,
  lockedByPick = false,
  pendingImport = null,
  selectVersion = null,
  onVersionSelected,
  onBaseVersion,
  onImported,
}: {
  onChanged?: () => void;
  /** Fired whenever the Class, model or size dropdown changes, and
   * once on load. The page holds it so the base-model picker and the
   * training form below both follow one choice instead of each keeping
   * their own copy of it. */
  /** Fired when the run's shape changes -- epochs and the rest. Separate
   * from `onChoose` because they are separate decisions that change at
   * separate times; folding them into one callback would mean every
   * keystroke in Epochs re-announced the detector. */
  onTraining?: (shape: {
    epochs: number; imgSize: number; batch: number;
    seed: number; patience: number; notes: string;
    /** What the run is sent, typed -- numbers, booleans, freeze as a
     * count or list. Not the text in the inputs. */
    hyperparams: ReturnType<typeof effectiveHyperparams>;
  }) => void;
  onChoose?: (
    choice: {
      slug: string;
      generation: string;
      size: string;
      architecture: string;
      group: string;
    },
    /** True when this is the saved choice being put back on load, false
     * when somebody picked it just now.
     *
     * The page clears the chosen base VERSION whenever the Class or
     * architecture changes, which is right -- a version belongs to one
     * registry group, so it cannot survive a move to another. On a
     * restore it is wrong: the version being restored is the one that
     * belongs to exactly this Class. Without this flag the two are
     * indistinguishable, and restoring the Class wiped the version that
     * was restored alongside it. */
    restored?: boolean,
  ) => void;
  /** The other way to get a model: import a COCO release or weights
   * somebody already trained. Passed in rather than imported here so
   * this card keeps knowing only about detectors, and the page decides
   * what a model can be made from. */
  children?: ReactNode;
  /** Drawn directly above "Define a new Class".
   *
   * Importing a checkpoint belongs here rather than with the training
   * modes at the foot of the card: it is something you do BEFORE there
   * is anything to train on, in the same way defining a Class is. */
  aboveDefine?: ReactNode;
  /** True while the import card above is taking its Class from a
   * checkpoint's own class names. The two are answers to one question --
   * what does this model detect -- and the registry records only one, so
   * both being live at once is the state that produces a version filed
   * under a taxonomy nobody chose.
   *
   * Only ever true DURING an import. This card spent a release greyed
   * out for most of the time it was on screen, because it was in an
   * exclusive group it did not belong in; dimming it whenever the import
   * card merely exists would be that bug again. */
  defineDimmed?: boolean;
  /** A Model Name has been chosen below, so defining another is not the
   * next thing to do. The card stays on screen and out of play: a card
   * that vanishes teaches nobody that it exists.
   *
   * The way back is the Model Name dropdown itself -- its blank option
   * clears the choice -- rather than a button on a card that is dimmed. */
  lockedByPick?: boolean;
  /** A checkpoint inspected in the import card above, waiting for the
   * Class being defined here. Its presence is what replaces "Save
   * classes" with "Register New Model/Class", because saving a Class and
   * then going back up to file the checkpoint into it is two halves of
   * one decision, with a dropdown in between that can be got wrong. */
  pendingImport?: Track2PendingImport | null;
  /** A version that has just been registered. The panel moves its own
   * Class / model / size dropdowns onto it and preselects it, which is
   * the difference between an import that landed somewhere findable and
   * one that disappeared. */
  selectVersion?: { slug: string; architecture: string; arn: string } | null;
  onVersionSelected?: () => void;
  /** Preselect a registered version as the one a run starts from. Used
   * only by the landing above -- the picker itself reports its own
   * choice straight to the page. */
  onBaseVersion?: (arn: string) => void;
  onImported?: () => void;
}) {
  const [detectors, setDetectors] = useState<Track2Detector[]>([]);
  const [sizes, setSizes] = useState<string[]>([]);
  const [error, setError] = useState("");
  // Set when a delete was refused for holding versions, so the
  // destructive follow-up is offered against that exact Class and
  // cannot be aimed at a different one by a dropdown change.
  // The Model Name the delete dialog is asking about, and the server's
  // refusal if it gave one. Both live here because the dialog is the
  // only place either is shown.
  const [deleteTarget, setDeleteTarget] = useState<Track2Detector | null>(null);
  const [deleteBlocked, setDeleteBlocked] = useState("");
  const [deleting, setDeleting] = useState(false);
  // Renaming a Model Name. `renameTarget` is the one being renamed and
  // `renameTo` the new display name; the identifier is derived from it,
  // exactly as it is when one is first defined.
  const [renameTarget, setRenameTarget] = useState<Track2Detector | null>(null);
  const [renameTo, setRenameTo] = useState("");
  const [renaming, setRenaming] = useState(false);
  const [renameNote, setRenameNote] = useState("");
  const [busy, setBusy] = useState(false);
  // The slug of the Class being EDITED, or "" when defining a new one.
  // Not a field any more: for a new Class it is derived from the name,
  // and for an existing one it must not move at all -- the slug is in
  // the registry group name, so changing it would strand every version
  // already filed under the old one and quietly create a second Class.
  const [editingSlug, setEditingSlug] = useState("");
  const [label, setLabel] = useState("");
  // One class per line, index taken from position -- which is exactly how
  // data.yaml orders them, and the order the model's outputs use.
  const [classText, setClassText] = useState("");
  // Reading the taxonomy out of a dataset instead of retyping it. The
  // names live ONLY in data.yaml -- a label file carries the index and
  // nothing else -- and retyping them is where index 0 quietly becomes
  // the wrong name for every model in that group afterwards.
  // Dimmed unless a checkpoint has declined its own classes -- OR an
  // existing Class is open for editing. Edit on a row loads it into
  // this same form, so without the exception that button would appear
  // to do nothing at all.
  //
  // Declared HERE rather than beside `slug` below, because the effects
  // that list buckets and yaml files gate on it: a dependency array is
  // evaluated during render, so a const declared after them would be in
  // its temporal dead zone.
  const bodyDimmed = (defineDimmed && !editingSlug) || lockedByPick;
  const [prefix, setPrefix] = useState("");
  const [folders, setFolders] = useState<string[]>([]);
  // The .yaml files at the level being browsed. Shown and selected the
  // same way the checkpoint picker shows .pt files, so one browser
  // behaves one way wherever it appears.
  const [yamlFiles, setYamlFiles] = useState<string[]>([]);
  // Where the data.yaml is. A dataset exported from Roboflow arrives in
  // Downloads, and uploading it to S3 just to read six class names off
  // it is a detour.
  const [yamlSource, setYamlSource] = useState<"computer" | "bucket">("computer");
  const [localYaml, setLocalYaml] = useState<File | null>(null);
  // WHICH bucket. The browser was bound to this app's own, and the
  // datasets are as often in another one -- which made "From the
  // bucket" mean "from the one bucket we happen to write to".
  const [yamlBucket, setYamlBucket] = useState("");
  const [buckets, setBuckets] = useState<Track2Bucket[]>([]);
  const [picked, setPicked] = useState("");
  const [readNote, setReadNote] = useState("");
  // What data.yaml declared, and which of those to keep. Reading used to
  // dump every class straight into the box; a dataset that carries ten
  // classes when the detector wants three then had to be edited by hand,
  // which is exactly the retyping this button exists to avoid.
  // What everything trains as, chosen here and read everywhere else.
  // Server-side, because the training form's own choice lives in a
  // browser tab -- which is why the unattended retrain had nothing to
  // read and blocked on "no detector chosen".
  const [chosenSlug, setChosenSlug] = useState("");
  const [chosenSize, setChosenSize] = useState("s");
  // Which YOLO generation a run is built from. Separate from the size
  // because they are separate decisions -- "a bigger model" and "a newer
  // architecture" are not the same question, and pairing them in one
  // dropdown of eight makes neither legible.
  const [chosenGeneration, setChosenGeneration] = useState("11");
  const [generations, setGenerations] = useState<Track2Generation[]>(FALLBACK_GENERATIONS);
  // How every run is shaped. Here rather than in the training form, for
  // the same reason the detector is: a number in a browser tab is a
  // number the scheduled retrain cannot read.
  const [shape, setShape] = useState({
    epochs: 50, imgSize: 640, batch: 16, seed: 42, patience: 0, notes: "",
  });
  const [shapeSaved, setShapeSaved] = useState(true);
  // The rest of what Ultralytics takes. OVERRIDES ONLY: a key is absent
  // unless it was changed, so a run that touches none of this trains
  // exactly as it did before any of it existed.
  const [hyperparams, setHyperparams] = useState<Record<string, string>>({});
  // Whether the rest of the training parameters are sent at all. OFF by
  // default, and that default matters: every settings file written
  // before this existed has no such key, and those runs must go on
  // behaving exactly as they did.
  const [hyperparamsOn, setHyperparamsOn] = useState(false);

  const [found, setFound] = useState<{ index: number; name: string }[]>([]);
  const [keep, setKeep] = useState<number[]>([]);

  useEffect(() => {
    getTrack2PullControl()
      .then((status) => {
        const training = status.config.default_training || {};
        if (Object.keys(training).length) {
          setHyperparams(
            (training.hyperparams as Record<string, string> | undefined) ?? {});
          setHyperparamsOn(Boolean(training.hyperparams_enabled));
          const loaded = {
            epochs: Number(training.epochs ?? 50),
            imgSize: Number(training.img_size ?? 640),
            batch: Number(training.batch ?? 16),
            seed: Number(training.seed ?? 42),
            patience: Number(training.patience ?? 0),
            notes: String(training.notes ?? ""),
          };
          setShape(loaded);
          onTraining?.({
            ...loaded,
            // What the RUN gets, not what is typed: expanded to every
            // field when the box is ticked, empty when it is not.
            hyperparams: effectiveHyperparams(
              (training.hyperparams as Record<string, string> | undefined) ?? {},
              Boolean(training.hyperparams_enabled)),
          });
        } else {
          // Nothing recorded yet. The shape was only ever written on
          // BLUR, so a run whose numbers were left at their defaults
          // saved `default_training: {}` -- and a settings file holding
          // an empty object does not restore 50/640/16, it leaves
          // whatever the page happens to be showing. Writing what is on
          // screen once makes the saved file a complete answer rather
          // than a partial one. Runs once: after this the key exists.
          void saveTrack2PullConfig({
            default_training: {
              epochs: 50, img_size: 640, batch: 16,
              seed: 42, patience: 0, notes: "",
            },
          }).catch(() => undefined);
        }
        const picked = status.config.default_detector || {};
        if (picked.slug) setChosenSlug(picked.slug);
        if (picked.architecture) {
          // "yolo26m" -> generation 26, size m. Both halves, because
          // taking only the size would silently move a YOLO26 selection
          // back to YOLO11 on the next save.
          const parts = /^yolo(\d+)([a-z]+)$/.exec(picked.architecture);
          if (parts) {
            setChosenGeneration(parts[1]);
            setChosenSize(parts[2]);
            onChoose?.({
              slug: picked.slug || "",
              generation: parts[1],
              size: parts[2],
              architecture: picked.architecture,
              group: `yolo${parts[1]}${parts[2]}-${picked.slug}-detector`,
            }, true);
          }
        }
      })
      .catch(() => undefined);
  }, []);

  /** Make this detector the one everything trains as. */
  // A version that has just been registered: move the dropdowns onto its
  // Class, model and size so the base-model picker -- which filters on
  // exactly those three -- is showing the list it is in.
  //
  // `onVersionSelected` clears the request, so this runs once per
  // registration rather than fighting whatever the operator picks next.
  useEffect(() => {
    if (!selectVersion) return;
    const parsed = /^yolo(\d+)([smlx])$/.exec(
      selectVersion.architecture.trim().toLowerCase());
    if (!parsed) { onVersionSelected?.(); return; }
    void chooseDetector(selectVersion.slug, parsed[2], parsed[1]);
    // The version itself, not just its Class. `restored` is true: this
    // is not somebody PICKING a different Class, so the version must not
    // be cleared as a stale selection -- it is the one that was just
    // created for exactly this Class.
    onChoose?.({
      slug: selectVersion.slug,
      generation: parsed[1],
      size: parsed[2],
      architecture: `yolo${parsed[1]}${parsed[2]}`,
      group: `yolo${parsed[1]}${parsed[2]}-${selectVersion.slug}-detector`,
    }, true);
    onBaseVersion?.(selectVersion.arn);
    onVersionSelected?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectVersion]);

  async function chooseDetector(slug: string, size: string, generation = chosenGeneration) {
    setChosenSlug(slug);
    setChosenSize(size);
    setChosenGeneration(generation);
    onChoose?.({
      slug, generation, size,
      architecture: `yolo${generation}${size}`,
      group: `yolo${generation}${size}-${slug}-detector`,
    });
    try {
      await saveTrack2PullConfig({
        default_detector: { slug, architecture: `yolo${generation}${size}` },
      });
    } catch {
      // The card still reads correctly; the next save will carry it.
    }
  }

  /** Change one field of the run's shape. Announced immediately so the
   * form below follows as you type; SAVED on blur, because writing the
   * config on every keystroke turns "120" into three saves, one of which
   * persists "1". */
  function editShape(patch: Partial<typeof shape>) {
    const next = { ...shape, ...patch };
    setShape(next);
    setShapeSaved(false);
    onTraining?.({ ...next, hyperparams: effectiveHyperparams(hyperparams, hyperparamsOn) });
  }

  async function saveShape() {
    if (shapeSaved) return;
    try {
      await saveTrack2PullConfig({
        default_training: {
          epochs: shape.epochs, img_size: shape.imgSize, batch: shape.batch,
          seed: shape.seed, patience: shape.patience, notes: shape.notes,
          // Saved with the rest, so the form, "Train on this" and the
          // scheduled retrain read ONE set of numbers rather than keeping
          // copies that can differ without anything saying so.
          // The sparse overrides, plus whether they are in play. Saved
          // sparse deliberately: the file then reads as the decisions
          // somebody made, not thirty-seven restatements of a default.
          hyperparams,
          hyperparams_enabled: hyperparamsOn,
        },
      });
      setShapeSaved(true);
    } catch {
      // Left unsaved rather than silently marked saved -- the caption
      // says so, and the next blur tries again.
    }
  }

  /** The sizes this generation actually offers. */
  const generationSizes =
    generations.find((g) => g.generation === chosenGeneration)?.sizes
    ?? (sizes.length ? sizes : ["s", "m", "l", "x"]);

  // Every bucket in the account, once the picker is open.
  useEffect(() => {
    if (bodyDimmed || buckets.length) return;
    getTrack2Buckets()
      .then((res) => {
        setBuckets(res.buckets);
        // Only when nothing is chosen, and this app's own first --
        // which is where the datasets this platform pulls into live.
        setYamlBucket((current) => current || res.default || res.buckets[0]?.name || "");
      })
      .catch(() => undefined);
  }, [bodyDimmed, buckets.length]);

  useEffect(() => {
    if (bodyDimmed || yamlSource !== "bucket" || !yamlBucket) return;
    let cancelled = false;
    listTrack2YamlFiles(yamlBucket, prefix)
      .then((res) => {
        if (cancelled) return;
        setFolders(res.folders);
        setYamlFiles(res.yaml_files ?? []);
      })
      .catch(() => {
        if (cancelled) return;
        // A bucket the role cannot read lists as empty rather than
        // leaving the previous bucket's folders on screen under the new
        // bucket's name.
        setFolders([]);
        setYamlFiles([]);
      });
    return () => { cancelled = true; };
  }, [bodyDimmed, yamlSource, yamlBucket, prefix]);

  /** A data.yaml off the operator's own machine.
   *
   * The file is read in the browser and its TEXT posted; nothing is
   * uploaded to S3, because reading class names is not a reason to
   * leave a file in the bucket. The parsing happens server-side -- see
   * `classes_from_yaml_text`.
   */
  async function readFromLocalYaml() {
    if (!localYaml) return;
    setReadNote("");
    setError("");
    try {
      const text = await localYaml.text();
      applyReadClasses(
        await readTrack2ClassesFromYamlText(text, localYaml.name),
        localYaml.name,
      );
    } catch (err) {
      setError(
        err instanceof ApiError ? err.detail : `Could not read ${localYaml.name}.`,
      );
    }
  }

  /** `key` is the exact yaml, e.g. `retraining/data.yaml`. The browser
   * selects the FILE now, so there is no folder convention to honour and
   * `dataset.yaml` reads as itself. */
  /** What was read, named once. It was spelled three ways -- and one of
   * them appended "/data.yaml" to a value that is already a full key,
   * and said it about a local file too. */
  const yamlSourceLabel = yamlSource === "computer"
    ? (localYaml?.name ?? "")
    : (picked ? `s3://${yamlBucket}/${picked}` : "");

  /** What both sources do with what came back. One place, so a yaml off
   * a laptop and one in the bucket cannot be interpreted differently. */
  function applyReadClasses(res: { classes: Record<string, string> }, from: string) {
    const names = Object.entries(res.classes ?? {})
      .sort((a, b) => Number(a[0]) - Number(b[0]))
      .map(([index, name]) => ({ index: Number(index), name: String(name) }));
    if (names.length === 0) {
      setError(`${from} declares no classes.`);
      return;
    }
    // Everything ticked to begin with: taking the whole taxonomy is the
    // common case, and is the only selection whose indices are already
    // the ones the label files use.
    setFound(names);
    setKeep(names.map((entry) => entry.index));
    setReadNote(
      `Read ${names.length} class${names.length === 1 ? "" : "es"} from ${from}`,
    );
  }

  async function readFromDataset(key: string) {
    setReadNote("");
    setError("");
    try {
      applyReadClasses(await readTrack2YamlClasses(yamlBucket, key), yamlSourceLabel);
    } catch (err) {
      setError(
        err instanceof ApiError ? err.detail : `Could not read ${key}.`,
      );
    }
  }

  const load = useCallback(() => {
    getTrack2Detectors()
      .then((res) => {
        // The built-in PPE pair is hidden here. It is merged in from code
        // rather than stored, cannot be edited or deleted, and owns the
        // legacy yolo11{size}-detector groups -- so listing it among the
        // class lists you define implies a control that does not exist.
        // It is still returned by the API and still usable elsewhere.
        setDetectors(res.detectors.filter((d) => !d.builtin));
        setSizes(res.sizes);
        // Server-driven, so adding a generation server-side is the whole
        // change -- there is no matching list to forget to update here.
        if (res.generations?.length) setGenerations(res.generations);
      })
      .catch(() => setError("Could not load the Model Names."));
  }, []);

  useEffect(load, [load]);

  // What the typed name becomes. Editing keeps the original -- see the
  // note on `editingSlug`.
  /** The chosen Class's taxonomy, in index order -- numerically,
   * because as text 10 sorts before 2 and the list goes wrong at the
   * eleventh.
   *
   * Declared HERE, after `chosenSlug`. It was up with `bodyDimmed`,
   * twenty-nine lines ABOVE the `useState` that creates `chosenSlug` --
   * a temporal dead zone, so every render threw
   * "Cannot access 'chosenSlug' before initialization" and the page went
   * blank. `tsc` passed it: the reference sits inside the `find`
   * callback, and TS2448 only flags a DIRECT use before declaration, not
   * one inside a function body it cannot prove runs immediately.
   */
  const chosenClasses = Object.entries(
    detectors.find((d) => d.slug === chosenSlug)?.classes ?? {},
  ).sort((a, b) => Number(a[0]) - Number(b[0]));

  const slug = editingSlug || slugFromName(label);
  const nameProblem = editingSlug ? "" : slugProblem(label);
  // Two different names can derive to one identifier ("Vehicle
  // detection" and "vehicle-detection"), and the server upserts, so the
  // second one SILENTLY REPLACES the first's class list. With the slug
  // on screen that was at least visible; with one field it has to be
  // said.
  const clash = editingSlug
    ? undefined
    : detectors.find((d) => d.slug === slug && slug !== "");
  // The single group a waiting checkpoint lands in: its own architecture,
  // not the eight the Class will eventually own. `yolo11s` -> 11 and s.
  const pendingArch = /^yolo(\d+)([smlx])$/.exec(
    (pendingImport?.architecture || "").trim().toLowerCase());
  const pendingGroup = pendingArch && slug && !nameProblem
    ? groupNameFor(slug, pendingArch[1], pendingArch[2])
    : "";

  /** Save the Class, then file the waiting checkpoint into it.
   *
   * One action, not two. The alternative is save here, scroll up, find
   * the new Class in a dropdown and press Register -- and the dropdown
   * is exactly where it gets filed under the wrong one. The Class is
   * saved FIRST and only then registered into, because there is nothing
   * to register into until it exists; a failure in between leaves a
   * saved Class and no version, which is visible and recoverable, rather
   * than a version filed under a Class that was never written.
   */
  // Which of the checkpoint's own classes the typed names correspond to,
  // in the order they were typed.
  //
  // This is what decides whether the head can be CUT or has to be
  // rebuilt, and the distinction is not cosmetic: cutting keeps the
  // weights that already detect those things, so the version works
  // immediately. Rebuilding throws those outputs away and waits for a
  // training run to learn new ones.
  //
  // A name the file has never heard of cannot be cut to. No amount of
  // head surgery turns a COCO checkpoint into a forklift detector -- it
  // has no forklift outputs to keep. That is a seed, honestly.
  const typedNames = classText.split("\n").map((line) => line.trim()).filter(Boolean);
  const fileNames = pendingImport?.checkpointClasses ?? [];
  const fileLower = fileNames.map((name) => name.toLowerCase());
  const matchedIndices = typedNames.map((name) => fileLower.indexOf(name.toLowerCase()));
  const unmatchedNames = typedNames.filter((_n, i) => matchedIndices[i] < 0);
  const everyNameInFile = typedNames.length > 0 && unmatchedNames.length === 0;
  // Nothing to do when the ask is already exactly what the file is, in
  // order -- a copy under a new key would say the same thing twice.
  const sliceWouldChangeIt = everyNameInFile
    && (typedNames.length !== fileNames.length
      || matchedIndices.some((original, position) => original !== position));
  const canCutHead = Boolean(pendingImport) && everyNameInFile && sliceWouldChangeIt;

  async function handleSaveAndRegister() {
    if (!pendingImport) return;
    const saved = await handleSave({ keepForm: true });
    if (!saved) return;
    setBusy(true);
    setError("");
    try {
      await registerTrack2BaseModel(
        pendingImport.key, saved.slug, pendingImport.architecture,
        pendingImport.description,
        // Deliberate: the checkpoint detects something else, which is the
        // normal case here and why the server marks it a seed. NOT when
        // the head is being cut -- then it matches by construction, and
        // handing it the exception as well would wave a real mismatch
        // through alongside it.
        !canCutHead,
        // The file's own indices, in the order the names were typed, so
        // the first line really is class 0.
        canCutHead ? matchedIndices : [],
      );
      setEditingSlug("");
      setLabel("");
      setClassText("");
      load();
      onChanged?.();
      onImported?.();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail
        : "The Model Name was saved, but the checkpoint could not be registered into it.");
    } finally {
      setBusy(false);
    }
  }

  async function handleSave(
    { keepForm = false }: { keepForm?: boolean } = {},
  ): Promise<{ slug: string; saved: boolean } | null> {
    const names = classText.split("\n").map((line) => line.trim()).filter(Boolean);
    if (names.length === 0) {
      setError("A Model Name needs at least one class name.");
      return null;
    }
    setBusy(true);
    setError("");
    try {
      // Index by position: line 1 is class 0. The server rejects anything
      // that is not a contiguous 0..n-1 range, which is what Ultralytics
      // requires and what every consumer reading index 0 assumes.
      const classes: Record<string, string> = {};
      names.forEach((name, index) => {
        classes[String(index)] = name;
      });
      const saved = await saveTrack2Detector(slug, label, classes);
      // Saved, so the useful next state is one where it can be trained --
      // unless a registration is about to follow, which needs the form
      // intact if it fails.
      if (!keepForm) {
        setEditingSlug("");
        setLabel("");
        setClassText("");
      }
      load();
      onChanged?.();
      return saved;
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not save the Model Name.");
      return null;
    } finally {
      setBusy(false);
    }
  }

  async function handleRename() {
    if (!renameTarget) return;
    const nextSlug = slugFromName(renameTo);
    setRenaming(true);
    setError("");
    try {
      const out = await renameTrack2Detector(
        renameTarget.slug, nextSlug, renameTo.trim());
      // Follow the rename rather than leaving the page on an identifier
      // that no longer exists.
      if (chosenSlug === renameTarget.slug) setChosenSlug(out.detector.slug);
      setRenameTarget(null);
      setRenameTo("");
      // Said out loud, because the whole point is that it did not stop
      // at the registry: the operator should see what moved with it.
      setRenameNote(
        out.cascaded.length
          ? `Renamed. Now following it: ${out.cascaded.join(", ")}.`
          : "Renamed.");
      load();
      onChanged?.();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not rename it.");
    } finally {
      setRenaming(false);
    }
  }

  /** Runs the delete. The dialog asks; this only acts.
   *
   * A refusal does NOT close the dialog and does not put anything on the
   * page behind it. The server's reason and the harsher option it
   * implies belong in the dialog the operator is already looking at --
   * they used to be a red line at the top of the card plus a separate
   * button, which meant the answer to "can I delete this" appeared
   * several hundred pixels from the row that asked.
   */
  async function handleDelete(detector: Track2Detector, force = false) {
    setDeleting(true);
    setError("");
    try {
      await deleteTrack2Detector(detector.slug, force);
      setDeleteTarget(null);
      setDeleteBlocked("");
      if (chosenSlug === detector.slug) setChosenSlug("");
      load();
      onChanged?.();
    } catch (err) {
      // The server refuses while its registries hold versions, and names
      // the groups holding them -- the actionable part, so it is shown
      // verbatim, inside the dialog.
      setDeleteBlocked(
        err instanceof ApiError ? err.detail : "Could not delete the Model Name.");
    } finally {
      setDeleting(false);
    }
  }

  // In data.yaml order; a class's POSITION in this list is the index it
  // gets, which is the whole reason leaving one out matters.
  const chosen = found.filter((entry) => keep.includes(entry.index));
  const renumbered = chosen.some((entry, position) => entry.index !== position);

  return (
    // Collapsed by default. Defining a taxonomy is a setup step you touch
    // once and then leave alone, and this card is the first thing on the
    // page -- open, it pushes the actual work below the fold. The count
    // stays in the caption so it is legible while closed.
    //
    // The collapse is Card's, not this card's. It used to run its own as
    // well, so the header carried TWO Show/Hide buttons doing the same
    // thing to different halves of the card.
    <Card
      title="Model types and Model Names"
      defaultOpen={false}
      step="1"
      help={
        <>
          <p>
            A <strong>Model Name</strong> is a name plus the list of class names it
            detects, in order. Registering one decides two things later: which
            registry a trained model is versioned into, and what the dataset&apos;s
            class names are checked against.
          </p>
          <p className="mt-2">
            The order matters and cannot be changed afterwards without
            retraining. A label file says <span className="tabular">2</span>, not
            &ldquo;person&rdquo;, so if index 2 means something different here
            than it does in your data, every box trains under the wrong name and
            nothing errors.
          </p>
          <p className="mt-2">
            Everything about <strong>which model</strong> is here: define a
            Model Name, choose the YOLO generation and size it trains as, import
            a COCO release or weights somebody already trained, and pick the
            registered version a run continues from. Step 6 only starts the
            run.
          </p>
          <p className="mt-2">
            <strong>Train as this</strong> on a row is the choice the rest of
            the page follows -- the training form pre-selects it, and the
            scheduled retrain always uses it.
          </p>
        </>
      }
      next="Step 2, Pick Golden Set / Edge Case -- or skip it and go straight to Step 3, Pull Control."
    >

      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {renameNote && <p className="mb-2 text-xs text-ink-secondary">{renameNote}</p>}

      {/* Renaming. The identifier derives from the name, exactly as it
          does when a Model Name is first defined -- so the dialog shows
          what it will become rather than asking for it twice. */}
      <ConfirmDialog
        open={Boolean(renameTarget)}
        title={`Rename "${renameTarget?.label}"`}
        busy={renaming}
        confirmLabel="Rename and update the workflow"
        onCancel={() => { setRenameTarget(null); setRenameTo(""); }}
        onConfirm={() => void handleRename()}
        body={
          <>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">New name</span>
              <input
                type="text"
                value={renameTo}
                autoFocus
                onChange={(e) => setRenameTo(e.target.value)}
                className="w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-xs"
              />
            </label>

            {renameTo.trim() && (
              <p className="tabular mt-2 text-xs text-ink-muted">
                Identifier <span className="text-ink">{slugFromName(renameTo)}</span>
                {" "}&middot; registers into{" "}
                <span className="text-ink">
                  yolo11s-{slugFromName(renameTo)}-detector
                </span>
              </p>
            )}
            {slugProblem(renameTo) && (
              <p className="mt-1 text-xs text-critical">{slugProblem(renameTo)}</p>
            )}

            <p className="mt-3">
              <strong>Everything downstream follows it</strong> — what runs
              train as, the Assemble and Import cards, the unattended
              retrain, and anything already queued.
            </p>
            <p className="mt-2">
              Versions already registered <strong>stay where they are</strong>,
              under{" "}
              <span className="tabular">
                yolo11s-{renameTarget?.slug}-detector
              </span>
              . A registry group cannot be renamed — SageMaker has no such
              operation — so they are kept readable under the old name
              rather than moved. They still show here and a run can still
              continue from them.
            </p>
            <p className="mt-2 text-ink-muted">
              Nothing is deleted and no weights move.
            </p>
          </>
        }
      />

      {/* Everything about deleting a Model Name happens in here. There
          used to be a red line and a "Delete it and its versions" button
          at the top of the card -- the answer to a question asked at the
          bottom, shown several hundred pixels away from it. */}
      <ConfirmDialog
        open={Boolean(deleteTarget)}
        title={deleteBlocked
          ? `"${deleteTarget?.label}" has registered versions`
          : `Delete "${deleteTarget?.label}"?`}
        destructive
        busy={deleting}
        confirmLabel={deleteBlocked ? "Delete it and its versions" : "Delete"}
        cancelLabel={deleteBlocked ? "Leave it alone" : "Cancel"}
        onCancel={() => { setDeleteTarget(null); setDeleteBlocked(""); }}
        onConfirm={() => {
          if (deleteTarget) void handleDelete(deleteTarget, Boolean(deleteBlocked));
        }}
        body={deleteBlocked ? (
          <>
            {/* The server's own words: it names the registries and how
                many versions each holds, which is the part that decides
                whether this is safe. */}
            <p className="text-critical">{deleteBlocked}</p>
            <p className="mt-2">
              Deleting it anyway removes <strong>every registered version</strong> in
              those registries as well. Models already deployed keep running;
              what goes is the record of where they came from, and nothing
              can be rolled back to them afterwards.
            </p>
            <p className="mt-2">This cannot be undone.</p>
          </>
        ) : (
          <>
            <p>
              Removes the Model Name and its class list. Datasets and
              trained weights are untouched.
            </p>
            {deleteTarget && Object.keys(deleteTarget.classes || {}).length > 0 && (
              <p className="tabular mt-2 text-ink-muted">
                Detects {Object.values(deleteTarget.classes).join(", ")}.
              </p>
            )}
          </>
        )}
      />

      {aboveDefine}

      {/* NOT one of the exclusive options, and never greyed. Defining a
          Model Name is not a way of getting a model -- it is what the other
          ways all need to exist first. Putting it in the group meant
          ticking a box before you could type a name, and meant the card
          sat greyed out for most of the time it was on screen.

          Like "Your Model Names" below it and "How every run trains" at the
          foot of the card: always available, whatever is selected. */}
      {bodyDimmed && (
        <p className="mb-2 rounded-sm border border-hairline bg-page px-2.5 py-1.5 text-xs text-ink-secondary">
          A Model Name is defined from an imported checkpoint. Read one above and
          untick <em>Use the classes in this file</em> to name it here &mdash;
          or type an existing Model Name below to change what it detects.
        </p>
      )}
      <Card
        title="Define a new Model Name"
        // SHUT at the start, with the card's own Show button as the way
        // in. There was briefly a tick beside the title doing this, and
        // it was one control too many: Show/Hide already says whether a
        // card is in play, and a second thing saying the same thing is
        // another thing to read.
        defaultOpen={false}
        // What a shut card says. Without one it is a title and nothing
        // else -- and the one moment it matters is when a checkpoint is
        // waiting for a Class that has not been named yet, which would
        // otherwise be invisible behind a closed card.
        summary={pendingImport ? (
          // Red only when there is genuinely nothing to file it into.
          // With a Model Name already picked this card is not the
          // blocker -- the choice in the import card above is -- and a
          // red "name one" sends somebody to define a second.
          chosenSlug ? (
            <p className="text-xs text-info">
              <strong>A model has already been selected.</strong>{" "}
              <em>
                {detectors.find((d) => d.slug === chosenSlug)?.label || chosenSlug}
              </em>{" "}
              is in <em>Your Model Names</em> below &mdash; choose it in the
              import card above to register the waiting checkpoint into it.
            </p>
          ) : (
            <p className="text-xs text-critical">
              A checkpoint is waiting for a Model Name. Press <em>Show</em> to name
              one and register it.
            </p>
          )
        ) : (
          <p className="text-xs text-ink-muted">
            Open this to define a Model Name &mdash; a name plus the class names it
            detects, in order.
          </p>
        )}
        caption="A name plus the class names it detects, in order. This is what a trained model is filed under, and what a dataset is checked against."
        help={
          <>
            <p>
              A <strong>Model Name</strong> is the taxonomy a model is trained
              for: <span className="tabular">0: helmet, 1: gloves</span> and
              so on. The ORDER is part of it and cannot be changed later
              without retraining &mdash; a label file says{" "}
              <span className="tabular">2</span>, never
              &ldquo;vehicle&rdquo;.
            </p>
            <p className="mt-2">
              <strong>Reading the names from a data.yaml</strong> is the safe
              way to fill it in, from your computer or from a bucket. They
              live only in that file, so retyping them is where index 0
              quietly becomes the wrong name for every model in the group
              afterwards.
            </p>
            <p className="mt-2">
              Registry groups are named after it, so the name is permanent
              in practice.
            </p>
          </>
        }
      >
        {/* The body, and only the body: the card's own Show/Hide stays
            usable so it can be looked at while it is out of play. */}
        <div className={bodyDimmed ? "pointer-events-none opacity-40" : ""}>
        {/* Edit on a row loads that Model Name into this same form, so without
            a way back "new" would show the last one edited. */}
        {editingSlug && (
          <div className="mb-2 flex items-center justify-between gap-2 rounded-sm border border-black/15 bg-page px-2.5 py-1.5">
            <span className="text-xs text-ink-secondary">
              Editing{" "}
              <span className="font-medium text-ink">{label || editingSlug}</span>
            </span>
            <button
              type="button"
              onClick={() => {
                setEditingSlug("");
                setLabel("");
                setClassText("");
                setError("");
              }}
              className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary"
            >
              Start a new one
            </button>
          </div>
        )}
        <div className="rounded-md border border-black/15 p-3">
          {/* Two parts, titled. The card ran Name, then a collapsed
              "Read from a data.yaml", then the class list, as one
              undifferentiated run -- so "set the Model Name" and "where do
              its class names come from" read as one form rather than
              two decisions, and the second was behind a button. */}
          <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-ink-muted">
            1 &middot; Name the Model Name
          </p>

          {/* ONE field. It was Slug and Name side by side: the same
              question asked twice, where one of the two silently decided
              where every version of this Model Name would be filed for the
              rest of its life and the other was a label. The identifier
              is derived from the name and shown below, so the
              consequence is visible without being a second thing to
              fill in. */}
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">
              Name{" "}
              <span className="text-ink-muted/70">
                (what this Model Name is called, and what its registry group is
                named after)
              </span>
            </span>
            <input
              className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
              value={label}
              placeholder="Vehicle detection"
              onChange={(e) => setLabel(e.target.value)}
            />
          </label>
          {nameProblem && (
            <p className="mt-1 text-xs text-critical">{nameProblem}</p>
          )}
          {/* An existing Model Name keeps the identifier it was created with,
              whatever the name is changed to. Renaming the group would
              orphan every version already registered under the old
              name -- so the rename is a label change, and this says so
              rather than letting the group names below look wrong. */}
          {editingSlug && slugFromName(label) !== editingSlug && (
            <p className="mt-1 text-xs text-ink-muted">
              Keeps its original identifier{" "}
              <span className="tabular text-ink">{editingSlug}</span>, because
              versions are already filed under it. The name is what changes.
            </p>
          )}
          {!editingSlug && clash && (
            // The built-in is a different refusal, not a stronger one:
            // the server will not redefine it at all, so promising a
            // replace would be a 422 a moment later -- and the derived
            // identifier is no longer on screen to explain why "PPE"
            // collided with anything.
            <p className="mt-1 text-xs text-critical">
              {clash.builtin ? (
                <>
                  <span className="tabular">{slug}</span> already exists
                  &mdash; unable to comply. It is the built-in{" "}
                  <span className="tabular">{clash.label}</span>, which owns
                  the production registries. Pick another name.
                </>
              ) : (
                <>
                  A Model Name with this identifier already exists
                  (<span className="tabular">{clash.label}</span>). Saving will
                  replace its class list. Use a different name, or edit that
                  one from the list below.
                </>
              )}
            </p>
          )}
          {/* Not behind a button any more. The names exist ONLY in
              data.yaml -- a label file carries "0 0.51 0.22 ..." and no
              name at all -- so reading them IS the way to fill this in
              and typing them is the fallback. A collapsed toggle made
              the correct route the one you had to go looking for. */}
          <p className="mb-1 mt-4 text-xs font-semibold uppercase tracking-wide text-ink-muted">
            2 &middot; Read its classes from a data.yaml
          </p>
          <p className="mb-2 text-xs text-ink-muted">
            The class names live only in that file, in index order. Reading
            them is safer than retyping: get the order wrong and every model
            in this group is mislabelled, and nothing errors.
          </p>

          <label className="mt-2 block">
            <span className="mb-1 block text-xs text-ink-muted">
              Classes, one per line{" "}
              <span className="text-ink-muted/70">
                (order matters -- the first line is class 0, and must match the
                dataset's data.yaml exactly)
              </span>
            </span>
            <textarea
              className="tabular w-full resize-y rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
              rows={4}
              value={classText}
              placeholder={"vehicle\nforklift\ntruck"}
              onChange={(e) => setClassText(e.target.value)}
            />
          </label>

          {(
            <div>
              {/* The file is on a laptop as often as it is in the
                  bucket -- a dataset exported from Roboflow arrives in
                  Downloads, and uploading it to S3 just to read six
                  class names off it is a detour. */}
              <div className="mb-2 flex flex-wrap gap-3">
                {([
                  ["computer", "From my computer"],
                  ["bucket", "From the bucket"],
                ] as const).map(([value, optionLabel]) => (
                  <label key={value} className="flex items-center gap-1.5 text-xs text-ink-secondary">
                    <input
                      type="radio"
                      name="yamlSource"
                      checked={yamlSource === value}
                      onChange={() => {
                        setYamlSource(value);
                        setPicked("");
                        setReadNote("");
                        setError("");
                      }}
                    />
                    {optionLabel}
                  </label>
                ))}
              </div>

              {yamlSource === "computer" ? (
                <label className="block">
                  <span className="mb-1 block text-xs text-ink-muted">
                    data.yaml <span className="text-ink-muted/70">(.yaml or .yml)</span>
                  </span>
                  <input
                    type="file"
                    // Both spellings, and only those: Ultralytics writes
                    // `.yaml`, plenty of exports write `.yml`, and this
                    // card reads nothing else.
                    accept=".yaml,.yml,application/x-yaml,text/yaml"
                    className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                    onChange={(e) => {
                      const chosen = e.target.files?.[0] ?? null;
                      setLocalYaml(chosen);
                      setReadNote("");
                      setError("");
                    }}
                  />
                </label>
              ) : (
                <>
                  <label className="mb-2 block">
                    <span className="mb-1 block text-xs text-ink-muted">S3 bucket</span>
                    <select
                      className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                      value={yamlBucket}
                      onChange={(e) => {
                        setYamlBucket(e.target.value);
                        // A new bucket is a new tree: keeping the old
                        // prefix would walk to a path that does not
                        // exist here and read as an empty bucket.
                        setPrefix("");
                        setPicked("");
                      }}
                    >
                      {buckets.length === 0 && <option value="">loading...</option>}
                      {buckets.map((b) => (
                        <option key={b.name} value={b.name} disabled={!b.readable}>
                          {b.name}{b.readable ? "" : "  (no read access)"}
                        </option>
                      ))}
                    </select>
                  </label>
                <Track2S3FolderBrowser
                  prefix={prefix}
                  folders={folders}
                  onNavigate={setPrefix}
                  files={yamlFiles}
                  onSelectFile={setPicked}
                  selectedFile={picked}
                />
                </>
              )}
              <div className="mt-1.5 flex items-center gap-2">
                <button
                  type="button"
                  disabled={yamlSource === "computer" ? !localYaml : !picked}
                  onClick={() => (yamlSource === "computer"
                    ? readFromLocalYaml()
                    : readFromDataset(picked))}
                  className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40"
                >
                  Read this file
                </button>
                {yamlSourceLabel && (
                  <span
                    className="tabular truncate text-xs text-ink-muted"
                    title={yamlSourceLabel}
                  >
                    {yamlSourceLabel}
                  </span>
                )}
              </div>
              {/* What came back, and from where. It lived in the block
                  the toggle used to open, so with that gone the read
                  succeeded and said nothing. */}
              {readNote && (
                <p className="tabular mt-1 text-xs text-ink-secondary">{readNote}</p>
              )}

              {/* Which of them to keep. Untick one and everything below it
                  shifts up, because position IS the index and a label file
                  carries nothing but that number -- so the new numbering is
                  spelled out here rather than found out after a run has
                  trained every class under the wrong name. */}
              {found.length > 0 && (
                <div className="mt-2 rounded-md border border-hairline p-2">
                  <div className="mb-1 flex items-center gap-3">
                    <span className="text-xs text-ink-secondary">
                      {keep.length} of {found.length} selected
                    </span>
                    <button
                      type="button"
                      onClick={() => setKeep(found.map((entry) => entry.index))}
                      className="text-xs font-medium text-ink-secondary underline hover:text-ink"
                    >
                      Select all
                    </button>
                    <button
                      type="button"
                      onClick={() => setKeep([])}
                      className="text-xs font-medium text-ink-secondary underline hover:text-ink"
                    >
                      Select none
                    </button>
                  </div>
                  <div className="flex flex-col gap-0.5">
                    {found.map((entry) => {
                      const on = keep.includes(entry.index);
                      const position = chosen.findIndex((c) => c.index === entry.index);
                      return (
                        <label
                          key={entry.index}
                          className="flex items-center gap-2 text-xs text-ink-secondary"
                        >
                          <input
                            type="checkbox"
                            className="shrink-0"
                            checked={on}
                            onChange={() =>
                              setKeep(
                                on
                                  ? keep.filter((index) => index !== entry.index)
                                  : [...keep, entry.index],
                              )
                            }
                          />
                          <span className="tabular">
                            <span className="text-ink-muted">{entry.index}</span>{" "}
                            <span className="font-medium text-ink">{entry.name}</span>
                            {on && position !== entry.index && (
                              <span className="text-critical">
                                {" "}-- becomes class {position}
                              </span>
                            )}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                  {renumbered && (
                    <p className="mt-1.5 text-xs text-critical">
                      Leaving a class out re-numbers the ones after it. The label
                      files still carry the data.yaml numbers, so the dataset has
                      to be remapped before a run uses this list.
                    </p>
                  )}
                  <button
                    type="button"
                    disabled={keep.length === 0}
                    onClick={() => {
                      setClassText(chosen.map((entry) => entry.name).join(String.fromCharCode(10)));
                      setReadNote(
                        `Took ${chosen.length} of ${found.length} class`
                        + `${found.length === 1 ? "" : "es"} from ${yamlSourceLabel}`,
                      );
                      // No toggle to close any more; clearing the list
                      // is what ends the step.
                      setFound([]);
                    }}
                    className="mt-1.5 rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40"
                  >
                    {keep.length === found.length
                      ? `Use all ${found.length}`
                      : `Use these ${keep.length}`}
                  </button>
                </div>
              )}
            </div>
          )}
          {slug && !nameProblem && (
            <div className="mt-1 text-xs text-ink-muted">
              {/* A Model Name OWNS a group at every generation and size,
                  because a future run can train it at any of them. But
                  when a checkpoint is waiting, THIS press files it into
                  exactly one -- the one the file is built for -- and
                  listing all eight under a button about to use one of
                  them reads as if it landed in all eight. */}
              {pendingGroup ? (
                <>
                  <p>
                    <span className="font-medium text-ink">Register</span> files
                    this checkpoint into
                  </p>
                  <p className="tabular text-ink">{pendingGroup}</p>
                  <p className="mt-1">
                    &mdash; the one group{" "}
                    <span className="tabular">{pendingImport?.architecture}</span>{" "}
                    belongs to. The Model Name owns the others too, for runs at a
                    different model or size:
                  </p>
                </>
              ) : (
                <p>Registers into:</p>
              )}
              {generations.map((gen) => (
                <p key={gen.generation} className="tabular">
                  {gen.label}
                  {"  "}
                  {gen.sizes
                    .map((size) => `yolo${gen.generation}${size}-${slug}-detector`)
                    .map((name) => (name === pendingGroup ? `${name} *` : name))
                    .join(", ")}
                </p>
              ))}
            </div>
          )}
          {/* What the checkpoint upstairs actually detects, beside what
              is being typed. The two differing is the NORMAL case here --
              a COCO file detects eighty things and this Model Name is three --
              and saying so is the difference between a deliberate seed
              and a mislabelled version. */}
          {pendingImport && (
            <div className="mt-2 rounded-md border border-black/15 bg-page p-2.5 text-xs">
              <p className="font-medium text-ink">
                A checkpoint is waiting for this Model Name
              </p>
              <p className="tabular mt-1 break-all text-ink-secondary">
                {pendingImport.key.split("/").slice(-1)[0]} &middot;{" "}
                {pendingImport.architecture}
              </p>
              {/* The COUNT and a few names, not all eighty. This is a
                  warning box, not the place to read a taxonomy -- the
                  import card above lists them in full, in a table. Eighty
                  names wrapping over six lines here buried the sentence
                  underneath them, which is the part that matters. */}
              <p className="mt-1 text-ink-muted">
                Its weights currently detect{" "}
                <span className="text-ink">
                  {pendingImport.checkpointClasses.length === 0
                    ? "nothing recorded"
                    : `${pendingImport.checkpointClasses.length} class${
                        pendingImport.checkpointClasses.length === 1 ? "" : "es"}`}
                </span>
                {pendingImport.checkpointClasses.length > 0 && (
                  <span className="tabular text-ink-muted">
                    {" ("}
                    {pendingImport.checkpointClasses.slice(0, 4).join(", ")}
                    {pendingImport.checkpointClasses.length > 4 ? ", ..." : ""}
                    {")"}
                  </span>
                )}
                . Registering it here files it as this Model Name&rsquo;s first
                version and <strong>the head is rebuilt on the first training
                run</strong>, so those names do not survive it.
              </p>
              <p className="mt-1 text-critical">
                Until that run the weights and the registry disagree, so it is
                registered as a <strong>seed</strong> &mdash; marked in its
                description and left Pending rather than Approved. Do not
                deploy it as it stands.
              </p>
            </div>
          )}

          {/* The other outcome, and the one worth aiming for: every name
              typed is a class the file already detects, so the head can
              be CUT to them instead of rebuilt. Shown only with a
              checkpoint waiting -- without one there is nothing to cut. */}
          {pendingImport && canCutHead && (
            <div className="mt-2 rounded-md border border-accent/40 bg-accent-subtle p-3 text-xs text-ink-secondary">
              <p>
                <strong>All {typedNames.length} of these are in the file.</strong>{" "}
                The head will be cut to them, in this order &mdash; so{" "}
                <span className="tabular text-ink">{typedNames[0]}</span> really
                is class 0. Those classes predict exactly what they predict
                now; nothing is learned or lost, the other outputs are removed.
              </p>
              <p className="mt-1">
                It registers as an ordinary version,{" "}
                <strong>not a seed</strong>, and can be deployed without a
                training run. The original file is left alone.
              </p>
            </div>
          )}

          {/* Why it CANNOT be cut, when only some names match. Said plainly
              rather than leaving "seed" to be inferred: the reason is not a
              policy, it is that the weights have no output for a thing they
              were never trained on. */}
          {pendingImport && typedNames.length > 0 && !everyNameInFile
            && unmatchedNames.length < typedNames.length && (
            <p className="mt-2 text-xs text-ink-muted">
              <span className="tabular text-ink">{unmatchedNames.join(", ")}</span>
              {unmatchedNames.length === 1 ? " is" : " are"} not in the file, so
              the head cannot be cut to this list &mdash; there is no output to
              keep for something the weights never learned. It registers as a
              seed and the head is rebuilt on the first training run.
            </p>
          )}

          {/* `block`, not the default inline-block: a <button> otherwise
              shares a line with the read-from-dataset toggle above it and
              the two overlap, hiding the end of that one's label. */}
          {/* ONE button. With a checkpoint waiting, Register saves the
              Model Name and then files the checkpoint into it -- so a second
              button that does the first half alone is a way to do a
              worse version of what the one beside it already does, and
              the obvious mistake is pressing it and wondering where the
              version went. Without a checkpoint there is nothing to
              register and Save is the only thing there is.

              NEITHER while the card is dimmed. The dimming already
              stops the press, but a greyed-out button still reads
              as the thing to reach for, and the answer there is
              upstairs rather than here. */}
          {!bodyDimmed && (
          <div className="mt-2">
            {pendingImport ? (
              <button
                type="button"
                disabled={busy || !slug || !!nameProblem || Boolean(clash?.builtin)}
                onClick={handleSaveAndRegister}
                className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
              >
                {busy ? "Saving and registering..." : "Register New Model Name"}
              </button>
            ) : (
              <button
                type="button"
                disabled={busy || !slug || !!nameProblem || Boolean(clash?.builtin)}
                onClick={() => void handleSave()}
                className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
              >
                {busy ? "Saving..." : "Save classes"}
              </button>
            )}
          </div>
          )}
        </div>

        </div>

        {/* Where the Register button would be. It is hidden while the
            card is out of play, so this takes its place rather than
            sitting beside a button that is not there.

            OUTSIDE the dimmed subtree above, deliberately: the button
            that puts the card back in play cannot itself be inert. */}
        {lockedByPick && (
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5">
            {/* KEPT and greyed, rather than hidden. A button that
                disappears leaves nothing to explain why this card does
                nothing; a dead one with the reason beside it says both
                at once. Disabled for real, not merely faded. */}
            <button
              type="button"
              disabled
              className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white opacity-40"
            >
              {pendingImport ? "Register New Model Name" : "Save classes"}
            </button>
            <p className="text-xs text-info">
              <strong>A model has already been selected.</strong>
            </p>
          </div>
        )}
      </Card>

      {/* Above the three ways, not below them. "Which Model Name" is what
          every one of them asks about, so showing the list after the
          choice is the wrong way round -- and the base-model picker
          asks for a registered version of a Model Name it had not shown. */}
      <p className="mb-1.5 mt-4 text-xs font-medium text-ink">
        Your Model Names{" "}
        <span className="font-normal text-ink-muted">
          (what everything below follows)
        </span>
      </p>

      {/* ONE set of controls rather than a copy on every row. As a radio
          per row plus a model and size select that appeared only on the
          chosen one, the same three facts were spread across as many
          rows as there were Model Names, and only one row ever showed them.
          Three dropdowns say the whole answer in one line. */}
      <div className="mb-2 rounded-md border border-black/15 bg-page p-2.5">
        <div className="flex flex-wrap items-end gap-3">
          <label className="min-w-40 flex-1">
            <span className="mb-1 block text-xs text-ink-muted">Model Name</span>
            <select
              className="w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-sm"
              value={chosenSlug}
              onChange={(e) => void chooseDetector(e.target.value, chosenSize)}
            >
              {/* Selectable once something is chosen, which is what puts
                  the two cards above back in play. It used to be
                  permanently disabled, so a Model Name could be picked
                  and never unpicked -- and with the cards that MAKE one
                  greyed out behind that choice, there was no way back at
                  all short of editing the settings file. */}
              <option value="" disabled={!chosenSlug}>
                {!detectors.length
                  ? "No models defined yet"
                  : chosenSlug
                    ? "Clear the selection"
                    : "Pick a Model"}
              </option>
              {detectors.map((d) => (
                <option key={d.slug} value={d.slug}>
                  {d.label} ({Object.keys(d.classes).length} class
                  {Object.keys(d.classes).length === 1 ? "" : "es"})
                </option>
              ))}
            </select>
          </label>
          <label>
            <span className="mb-1 block text-xs text-ink-muted">Model type</span>
            <select
              className="rounded-md border border-black/15 bg-page px-2 py-1.5 text-sm"
              value={chosenGeneration}
              onChange={(e) => {
                // A generation need not offer the size currently picked,
                // so fall back to its first rather than writing an
                // architecture that does not exist.
                const next = generations.find((g) => g.generation === e.target.value);
                const size = next?.sizes.includes(chosenSize)
                  ? chosenSize
                  : next?.sizes[0] ?? "s";
                void chooseDetector(chosenSlug, size, e.target.value);
              }}
            >
              {generations.map((gen) => (
                <option key={gen.generation} value={gen.generation}>{gen.label}</option>
              ))}
            </select>
          </label>
          <label>
            <span className="mb-1 block text-xs text-ink-muted">Size</span>
            <select
              className="rounded-md border border-black/15 bg-page px-2 py-1.5 text-sm"
              value={chosenSize}
              onChange={(e) => void chooseDetector(chosenSlug, e.target.value)}
            >
              {generationSizes.map((size) => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </label>
        </div>
        {/* WHAT it detects, under the dropdown that names it. The row
            said "PPE2 (1 classes)" and where its versions are filed, and
            nothing at all about the taxonomy -- which is the thing every
            dataset is checked against and the thing a training run
            inherits.

            In INDEX ORDER, with the index shown, because that is what a
            label file carries: a row reads `2 0.51 0.22 ...` and never
            "vehicle". */}
        {chosenClasses.length > 0 && (
          <p className="mt-1.5 text-xs text-ink-muted">
            Detects{" "}
            {chosenClasses.map(([index, name], position) => (
              <span key={index}>
                {position > 0 ? ", " : ""}
                <span className="tabular text-ink-muted/70">{index}</span>
                <span className="tabular text-ink">:{name}</span>
              </span>
            ))}
          </p>
        )}
        <div className="mt-1.5 flex flex-wrap items-center justify-between gap-2">
          <p className="tabular text-xs text-ink-muted">
            {chosenSlug
              ? `Registers into ${
                  detectors.find((d) => d.slug === chosenSlug)
                    ?.groups_by_generation?.[chosenGeneration]?.[chosenSize]
                  ?? `yolo${chosenGeneration}${chosenSize}-${chosenSlug}-detector`
                }`
              : "Pick a Model Name to see where its versions are filed."}
          </p>
          {/* Act on whatever the dropdowns name. The per-Model Name rows that
              used to carry these are gone: four rows of class lists to
              scroll past said nothing the dropdown does not, and the
              buttons were the only part anybody used. */}
          <div className="flex shrink-0 gap-1.5">
            <button
              type="button"
              disabled={busy || !chosenSlug}
              onClick={() => {
                const d = detectors.find((x) => x.slug === chosenSlug);
                if (d) { setRenameTarget(d); setRenameTo(d.label); setRenameNote(""); }
              }}
              className="rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40"
            >
              Rename
            </button>
            <button
              type="button"
              disabled={busy || !chosenSlug}
              onClick={() => {
                const d = detectors.find((x) => x.slug === chosenSlug);
                if (d) { setDeleteTarget(d); setDeleteBlocked(""); }
              }}
              className="rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-critical disabled:opacity-40"
            >
              Delete
            </button>
          </div>
        </div>
      </div>
      {children && <div className="space-y-3">{children}</div>}
      {/* How every run trains. Moved out of the training form so that
          ONE set of numbers is used by the form, by "train on this" on a
          snapshot, and by the scheduled retrain. They used to live in
          the form, which meant the schedule kept its own copy and the
          two could differ without anything saying so. */}
      <div className="mt-4 rounded-md border border-black/15 p-3">
        <div className="mb-2 flex items-baseline justify-between gap-2">
          <span className="text-xs font-medium text-ink">How every run trains</span>
          <span className="text-[11px] text-ink-muted">
            {shapeSaved ? "saved" : "unsaved -- click away to save"}
          </span>
        </div>
        <p className="mb-2 text-xs text-ink-muted">
          Used by the training form, by <em>Train on this</em> on a frozen
          snapshot, and by the scheduled retrain. Change it once here.
        </p>
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-5">
          {([
            ["Epochs", "epochs", shape.epochs],
            ["Image size", "imgSize", shape.imgSize],
            ["Batch", "batch", shape.batch],
            ["Seed", "seed", shape.seed],
            ["Patience", "patience", shape.patience],
          ] as [string, keyof typeof shape, number][]).map(([label, key, value]) => (
            <label key={key} className="block">
              <span className="mb-1 block text-xs text-ink-muted">{label}</span>
              <input
                type="number"
                className="tabular w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-sm"
                value={value}
                onChange={(e) => editShape({ [key]: Number(e.target.value) } as Partial<typeof shape>)}
                onBlur={() => void saveShape()}
              />
            </label>
          ))}
        </div>
        <label className="mt-2 block">
          <span className="mb-1 block text-xs text-ink-muted">
            Notes{" "}
            <span className="text-ink-muted/70">
              (carried onto every run, including unattended ones)
            </span>
          </span>
          <input
            type="text"
            className="w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-sm"
            value={shape.notes}
            onChange={(e) => editShape({ notes: e.target.value })}
            onBlur={() => void saveShape()}
          />
        </label>
        <p className="mt-1.5 text-xs text-ink-muted">
          <span className="font-medium">Patience 0</span> means never stop
          early, which is what makes two runs comparable. A non-zero value
          stops once the score has not improved for that many epochs.
        </p>
        <Track2TrainingAdvanced
          value={hyperparams}
          enabled={hyperparamsOn}
          onEnabledChange={(on) => {
            setHyperparamsOn(on);
            setShapeSaved(false);
            onTraining?.({ ...shape, hyperparams: effectiveHyperparams(hyperparams, on) });
          }}
          onChange={(next) => {
            setHyperparams(next);
            setShapeSaved(false);
            // Straight to the form as well. Saving it to the config and
            // not telling the form is how the card and the run come to
            // disagree about what a run was given.
            onTraining?.({ ...shape, hyperparams: effectiveHyperparams(next, hyperparamsOn) });
          }}
          onCommit={() => void saveShape()}
        />
      </div>

      {/* Below the list, because a checkpoint you import needs a Model Name
          to belong to -- the order on screen is the order of the
          decisions. */}
    </Card>
  );
}
