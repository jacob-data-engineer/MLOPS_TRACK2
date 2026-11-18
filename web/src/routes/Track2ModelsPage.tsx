import { Track2TrainingDataFile } from "../components/Track2TrainingDataFile";
import { Track2TrainingDataChoice, type TrainingDataChoice } from "../components/Track2TrainingDataChoice";
import { AlertCircle, Cpu, Loader2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { listen } from "../lib/bus";
import { useCardOpen } from "../lib/cardState";
import { formatDayTime } from "../lib/format";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { Track2DatasetVersionsPanel } from "../components/Track2DatasetVersionsPanel";
import { ProgressBar } from "../components/ProgressBar";
import { Track2TrainingRunProgress } from "../components/Track2TrainingRunProgress";
import { StatTile } from "../components/StatTile";
import { Track2DatasetAssemblePanel } from "../components/Track2DatasetAssemblePanel";
import { Track2DetectorsPanel } from "../components/Track2DetectorsPanel";
import { Track2BaseModelPicker } from "../components/Track2BaseModelPicker";
import { Track2ImportCheckpointPanel } from "../components/Track2ImportCheckpointPanel";
import { Track2PinnedBar } from "../components/Track2PinnedBar";
import { Track2TrainingHistory } from "../components/Track2TrainingAuditPanel";
import {
  runTrack2Retrain,
  getTrack2PreparationJob,
  type Track2PreparationJob,
  ApiError,
  getTrack2Metrics,
  getTrack2QueueAfterPull,
  saveTrack2QueueAfterPull,
  getTrack2Retrain,
  type Track2QueueAfterPull,
  type Track2RetrainStatus,
  getTrack2DatasetClasses,
  getTrack2Detectors,
  getTrack2PullControl,
  saveTrack2PullConfig,
  getTrack2Models,
  getTrack2ModelsLive,
  removeTrack2Queue,
  startTrack2Queue,
  stopTrack2Queue,
  stopTrack2Training,
  type Track2DatasetClasses,
  type Track2Detector,
  type Track2PullConfig,
  type Track2MetricsRow,
  type Track2ModelsResponse,
  type Track2PendingImport,
  type Track2QueueEntry,
  type Track2TrainingParams,
  type Track2TrainingProgress,
} from "../lib/api";

const DEFAULT_ARCHITECTURE = "yolo11s";

/** "yolo26m" -> { generation: "26", size: "m" }.
 *
 * Both halves, parsed. The previous version stripped the literal
 * "yolo11" prefix, which was correct while YOLO11 was the only
 * generation and silently wrong the moment it was not: "yolo26m" kept
 * its prefix and became a SIZE called "yolo26m", so the run would have
 * been submitted with a size no registry has. */
function splitArchitecture(architecture: string): { generation: string; size: string } {
  const parts = /^yolo(\d+)([a-z]+)$/.exec((architecture || "").trim().toLowerCase());
  return parts
    ? { generation: parts[1], size: parts[2] }
    : { generation: "11", size: "s" };
}

/** Same mapping the Deploy and Metrics tables use for approval badges. */
/** One queued or running run, as its own block.
 *
 * Running and pending entries render through the same component so a run
 * that STARTS does not appear to vanish: it keeps its block and changes
 * state, rather than dropping out of the pending list into a one-line
 * note. Each block carries its own status, so two queued runs update
 * independently.
 */
/** Which stage of the pipeline a running entry is actually in.
 *
 * "Executing" is the EXECUTION's status and says nothing about what the
 * GPU is doing -- training and evaluating both report it, and the epoch
 * bar sits at 1/1 through evaluation as if nothing were happening. The
 * per-step statuses were already on the payload; only nothing rendered
 * them.
 */
function runPhase(progress?: Track2TrainingProgress | null): string {
  const steps = progress?.steps;
  if (!steps) return "";
  // Only ever report a step that is ACTUALLY Executing. Inferring the
  // next one from the last Succeeded would claim "Registering the model"
  // on the custom pipeline, which has no register step at all -- its
  // RegisterYoloModel simply stays NotStarted forever.
  if (steps.RegisterYoloModel === "Executing") return "Registering the model";
  if (steps.EvaluateYolo === "Executing") return "Evaluating";
  if (steps.TrainYolo === "Executing") return "Training";
  if (steps.TrainYolo === "Succeeded") return "Finishing up";
  return "Starting up";
}

/** Where a run's images actually came from.
 *
 * `dataset_folder` is the assembled copy -- a dated name nobody chose and
 * nobody recognises afterwards. The folders the operator picked are what
 * belong on the card; the copy is an implementation detail of getting
 * them into the trainer's layout.
 */
/** A queue card's folders, training and evaluation in ONE scrolling box.
 *
 * One folder per line, with how many of its images the run uses beside it
 * -- for a week folder that is what retention kept. It was a comma-joined
 * line truncated after two folders, then two boxes; one box reads as the
 * one dataset it is. */
function plural(n: number, one: string, many: string): string {
  return `${n.toLocaleString()} ${n === 1 ? one : many}`;
}

function FolderBox({ sides, note }: {
  sides: { title: string; folders: string[]; counts?: Record<string, number> }[];
  note?: ReactNode;
}) {
  return (
    <div className="col-span-2 sm:col-span-3">
      <span>
        Images:{" "}
        {sides.map((side) => `${plural(side.folders.length, "folder", "folders")} for ${side.title.toLowerCase()}`)
          .join(", ")}
        {note && <> {note}</>}
      </span>
      <div className="mt-0.5 max-h-[7.5rem] overflow-auto rounded-sm border border-hairline bg-page px-2 py-0.5">
        {sides.map((side) => (
          <div key={side.title}>
            <p className="mt-0.5 font-medium text-ink-secondary">
              {side.title} &middot; {plural(side.folders.length, "folder", "folders")}
            </p>
            {side.folders.map((folder) => (
              <p key={folder} className="whitespace-nowrap">
                <span className="tabular inline-block w-24 text-right text-ink-secondary">
                  {side.counts?.[folder] !== undefined
                    ? plural(side.counts[folder], "image", "images")
                    : ""}
                </span>
                {"  "}{folder}
              </p>
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

function QueueEntryBlock({
  entry,
  state,
  position,
  progress,
  onRemove,
}: {
  entry: Track2QueueEntry;
  state: "running" | "queued";
  position?: number;
  /** Live epoch/download progress -- only ever available for the entry
   * that is actually on the GPU. */
  progress?: Track2TrainingProgress | null;
  onRemove?: () => void;
}) {
  const running = state === "running";
  return (
    <div
      className={`flex items-start justify-between gap-3 rounded-sm border p-2.5 ${
        running ? "border-accent/40 bg-accent-subtle" : "border-hairline bg-page"
      }`}
    >
      <div className="min-w-0 flex-1">
        <div className="mb-1.5 flex items-center gap-2">
          <Badge tone={running ? "pending" : "neutral"} label={running ? "Running" : `Queued #${position}`} />
          {running && runPhase(progress) && (
            <span className="text-xs font-medium text-ink">{runPhase(progress)}</span>
          )}
          <span className="tabular text-xs text-ink-secondary">
            {entry.custom_classes ? "Custom classes" : `YOLO${entry.generation}-${entry.size}`}
          </span>
          {running && entry.run_id && (
            <span className="tabular text-xs text-ink-muted">{entry.run_id}</span>
          )}
        </div>
        <div className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-3">
          {/* FIRST, and spanning: it is the run's identity, not one of
              its numbers. The registry group is derived from this pair,
              while the class NAMES come from the dataset's own
              data.yaml -- and nothing forces the two to agree. A run
              queued against the wrong Model Name trains happily,
              reports metrics, and files them under a taxonomy that does
              not describe it. Reading the queue should be enough to
              catch that before the GPU is spent. */}
          {entry.detector_slug && (
            <span className="col-span-2 truncate sm:col-span-3"
                  title={`Registers into ${entry.architecture || ""}-${entry.detector_slug}-detector`}>
              Model: <span className="font-medium">{entry.detector_slug}</span>
              {entry.architecture && (
                <span className="text-ink-muted"> &middot; {entry.architecture}</span>
              )}
            </span>
          )}
          <span>Epochs: {entry.epochs}</span>
          <span>Image size: {entry.img_size}</span>
          <span>Batch: {entry.batch}</span>
          <span>Seed: {entry.seed}</span>
          {/* A versioned entry trains on lists frozen at queue time; its
              own folder (`_datasets/<id>`) holds only those lists, so the
              lines below name where the images COME FROM. */}
          {entry.dataset_version_id && (
            <span className="col-span-2 truncate sm:col-span-3" title={entry.dataset_version_id}>
              Dataset version: <span className="font-medium">{entry.dataset_version_id}</span>
              {entry.dataset_version_train_images !== undefined && (
                <> &middot; {entry.dataset_version_train_images.toLocaleString()} train /{" "}
                  {(entry.dataset_version_val_images ?? 0).toLocaleString()} val images,
                  pinned by name when queued</>
              )}
            </span>
          )}
          <FolderBox
            sides={[
              {
                title: "Training",
                folders: entry.dataset_source_folders?.length
                  ? entry.dataset_source_folders : [entry.dataset_folder].filter(Boolean),
                counts: entry.dataset_source_counts,
              },
              {
                title: "Evaluation",
                folders: (entry.eval_source_folders ?? entry.dataset_source_folders)?.length
                  ? (entry.eval_source_folders ?? entry.dataset_source_folders) as string[]
                  : [entry.eval_dataset_folder || entry.dataset_folder].filter(Boolean),
                counts: entry.eval_source_folders ? entry.eval_source_counts : entry.dataset_source_counts,
              },
            ]}
            note={!entry.dataset_version_id && (
              <span className="text-ink-muted">(evaluation is the val split its data.yaml declares)</span>
            )}
          />
          {/* Which frozen definition, if any. Without it a run on a
              deliberately chosen snapshot was indistinguishable
              afterwards from one on the live corpus -- same folder name,
              same epochs, different images. */}
          {entry.snapshot_id && (
            <span
              className="col-span-2 truncate sm:col-span-3"
              title={entry.snapshot_note || entry.snapshot_id}
            >
              Frozen snapshot:{" "}
              <span className="font-medium">{entry.snapshot_id}</span>
              {entry.snapshot_created_at && (
                <> &middot; frozen {localDate(entry.snapshot_created_at)}</>
              )}
              {entry.snapshot_files
                ? ` · ${entry.snapshot_files.toLocaleString()} images`
                : ""}
              {entry.dataset_version_id
                ? " (a record only -- this run trains on the dataset version above)"
                : entry.snapshot_pinned
                ? " pinned by name"
                : " (recorded automatically: the exact files this run used. The yaml names folders, so a rerun would also pick up files added since)"}
            </span>
          )}
          {entry.base_model_folder && (
            <span className="col-span-2 truncate sm:col-span-3" title={entry.base_model_folder}>
              Base model: {entry.base_model_folder}
            </span>
          )}
          {entry.notes && (
            <span className="col-span-2 truncate sm:col-span-3" title={entry.notes}>
              Notes: {entry.notes}
            </span>
          )}
        </div>
        {running && progress?.status === "Executing" && <Track2TrainingRunProgress progress={progress} />}
      </div>
      {onRemove && (
        <button
          type="button"
          onClick={onRemove}
          className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
        >
          Remove
        </button>
      )}
    </div>
  );
}

/** What the training actions (Start Training / Add to Queue) submit --
 * held at the page rather than inside the form so it survives the page's
 * own refreshes while a run is in flight. */
interface TrainingSelection {
  epochs: number;
  imgSize: number;
  batch: number;
  seed: number;
  /** Early stopping: epochs without improvement before the run stops
   * itself. 0 leaves it to Ultralytics. */
  patience: number;
  /** Typed, as the run receives them -- see Track2TrainingParams. */
  hyperparams: NonNullable<Track2TrainingParams["hyperparams"]>;
  datasetFolder: string;
  /** Empty means "evaluate on the training dataset", which is what the
   * pipeline does when no separate set is given. */
  evalFolder: string;
  /** Which detector this run trains. Empty means the shared open groups,
   * which is how every custom run worked before detectors existed. */
  detectorSlug: string;
  /** Starting weights are always a registered version now. Picking a
   * loose .pt from the bucket was removed: it had no version, no metrics
   * and no lineage, so a run started from one could not say what it began
   * with. Import it first and it appears here like anything else. The
   * backend still accepts `base_model_key` and `base_model_scratch` (no
   * card sends either now), so
   * re-offering either is a UI change rather than a rebuild. */
  /** Which YOLO model this is: size, architecture, and the per-size
   * registry group the trained version lands in. */
  architecture: string;
  basePackageArn: string;
  notes: string;
}

/** A server timestamp as somebody in Singapore reads it.
 *
 * These are written UTC-aware, and printing the raw ISO string is how a
 * 12:37 run came to read as 04:37 elsewhere on this page. */
/** Named for what it used to be -- the viewer's local clock -- and kept
 * because eight call sites read it, but it renders Singapore time now.
 *
 * Every value passed to it is an instant the SERVER produced: when a
 * snapshot was frozen, when a dataset was built, when the last pull ran,
 * when the schedule next fires. In the viewer's zone, two people reading
 * one page side by side could see different times for the same event, and
 * the next-run line could disagree with the schedule that computed it. The
 * zone is named in the output so the reading is checkable. */
/** The Assemble card's saved build inputs, parsed the way
 * `retrain_schedule.assemble_plan` parses them: blank names dropped,
 * indices as written. `configured` is false when no folder is picked,
 * which is when the chain falls back to the root-and-week-folders build
 * and the folder's own yaml is the thing to check. */
function readAssembleSaved(saved: Track2PullConfig["dataset_assemble"]) {
  const labelClasses: Record<string, string> = {};
  for (const [key, value] of Object.entries(saved?.label_classes ?? {})) {
    const name = String(value ?? "").trim();
    if (name && Number.isInteger(Number(key))) labelClasses[String(Number(key))] = name;
  }
  return {
    configured: Boolean(saved?.train_sources?.length || saved?.eval_sources?.length),
    slug: String(saved?.detector_slug ?? "").trim(),
    labelClasses,
    dropUndeclared: false,
  };
}

function localDate(iso: string): string {
  return iso ? formatDayTime(iso) : "";
}

export function TrainingForm({
  selection,
  setSelection,
  onSubmitted,
  detectorsVersion,
}: {
  selection: TrainingSelection;
  setSelection: (patch: Partial<TrainingSelection>) => void;
  /** Called after a successful add. Takes no message: the entry appears
   * in the Training queue below, which is the record. A banner saying
   * "Added to queue (YOLO11-s, 1 epochs, retraining)" restated what the
   * queue card shows better, and sat there afterwards describing an
   * entry that might since have started, finished or been removed. */
  onSubmitted: () => void;
  /** Bumped when the Detectors panel saves or deletes one, so the picker
   * below is never offering a list that no longer exists. */
  detectorsVersion: number;
}) {
  const { epochs, imgSize, batch, seed, patience, hyperparams, datasetFolder, detectorSlug, architecture, basePackageArn, notes } = selection;
  const [busy, setBusy] = useState(false);
  const [deleteDecayAfterRun, setDeleteDecayAfterRun] = useState(false);
  const [dataChoice, setDataChoice] = useState<TrainingDataChoice>({mode: 'prepared', id: '', checked: false});
  const [, openTrainingCard] = useCardOpen('start the training run', true);
  useEffect(() => {
    const select = (event: Event) => {
      setDataChoice({mode: 'saved', id: (event as CustomEvent<string>).detail, checked: false});
      openTrainingCard(true);
    };
    window.addEventListener('track2-use-saved-dataset', select);
    return () => window.removeEventListener('track2-use-saved-dataset', select);
  }, []);
  const [preparationJob, setPreparationJob] = useState<Track2PreparationJob | null>(null);
  const completedJob = useRef('');
  const preparing = preparationJob?.state === 'pending' || preparationJob?.state === 'running';
  useEffect(() => {
    let cancelled = false;
    const poll = async () => {
      try {
        const {job} = await getTrack2PreparationJob();
        if (cancelled) return;
        setPreparationJob(job);
        if (job?.state === 'completed' && completedJob.current !== job.id) {
          completedJob.current = job.id;
          onSubmitted();
        }
      } catch { /* Keep the saved status while the server reconnects. */ }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 2000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);
  const [error, setError] = useState<string | null>(null);
  const [datasetClasses, setDatasetClasses] = useState<Track2DatasetClasses | null>(null);
  // Saved dataset taxonomy is checked against the selected model.
  const [assembleSaved, setAssembleSaved] = useState<{
    configured: boolean;
    slug: string;
    labelClasses: Record<string, string>;
    dropUndeclared: boolean;
  } | null>(null);
  // Refresh compatibility when Data Preparation changes in another tab.
  useEffect(() => listen("data-sources-changed", () => {
    getTrack2PullControl()
      .then((status) => {
        setAssembleSaved(readAssembleSaved(status.config.dataset_assemble));
      })
      .catch(() => undefined);
  }), []);
  // Choosing the dataset from this card, for when nothing upstream has.
  const [detectors, setDetectors] = useState<Track2Detector[]>([]);
  useEffect(() => {
    let cancelled = false;
    getTrack2Detectors()
      .then((res) => {
        if (cancelled) return;
        setDetectors(res.detectors);
        // A detector deleted while it was selected would otherwise sit in
        // the form invisibly and fail the run server-side.
        if (detectorSlug && !res.detectors.some((d) => d.slug === detectorSlug)) {
          setSelection({ detectorSlug: "" });
        }
        // Preselect the one chosen in step 1, falling back to the first
        // so the registry below is populated rather than starting on
        // "Pick a detector above".
        //
        // Step 1's choice, not this form's, because this form's lives in
        // a browser tab -- which is why nothing unattended could read it.
        // Overriding here still works and affects THIS run only: the
        // schedule always reads step 1, so a one-off cannot silently
        // change what the scheduled runs do later.
        const own = res.detectors.filter((d) => !d.builtin);
        if (!detectorSlug) {
          void getTrack2PullControl()
            .then((status) => {
              if (cancelled) return;
              const picked = status.config.default_detector || {};
              const wanted = picked.slug
                && res.detectors.some((d) => d.slug === picked.slug)
                ? picked.slug
                : own[0]?.slug;
              if (wanted) {
                setSelection({
                  detectorSlug: wanted,
                  ...(picked.architecture
                    ? { architecture: picked.architecture } : {}),
                });
              }
            })
            .catch(() => {
              if (!cancelled && own.length) {
                setSelection({ detectorSlug: own[0].slug });
              }
            });
        }
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectorsVersion]);

  // The detector's own class list, for comparing against the dataset's.
  const chosenDetector = detectors.find((d) => d.slug === detectorSlug);
  // Two different situations, with opposite outcomes. A dataset that
  // CONTAINS the detector's classes is filtered down to them before
  // training -- the normal case, since a site dataset is labelled once
  // with everything on it. A dataset MISSING one of them cannot train it
  // at all, and that is the only real error.
  //
  const assembledBuild = (() => {
    if (!assembleSaved?.configured) return null;
    return { classes: assembleSaved.labelClasses };
  })();
  const checkedClasses: Record<string, string> | null = assembledBuild
    ? assembledBuild.classes
    : datasetClasses?.classes ?? null;
  const datasetNames = checkedClasses ? Object.values(checkedClasses) : [];
  const detectorNames = chosenDetector ? Object.values(chosenDetector.classes) : [];
  const absentClasses =
    chosenDetector && checkedClasses
      ? detectorNames.filter((name) => !datasetNames.includes(name))
      : [];
  const willFilter =
    chosenDetector && checkedClasses && absentClasses.length === 0 &&
    JSON.stringify(chosenDetector.classes) !== JSON.stringify(checkedClasses);
  // The same rows the Deploy and Metrics tables use. The package list has no
  // metrics, so picking a base model from it meant choosing on a version
  // number alone with no idea how good that version was.
  const [registryRows, setRegistryRows] = useState<Track2MetricsRow[]>([]);
  // The unattended half: queue a run whenever a pull finishes, leaving a
  // person to release it.
  const [afterPull, setAfterPull] = useState<Track2QueueAfterPull | null>(null);
  const [afterPullBusy, setAfterPullBusy] = useState(false);
  /** The retrain schedule's own clock -- read even when the pull is the
   * trigger, because `follows_pull` on the response is what decides
   * whether the cadence fields are shown at all. */
  const [retrain, setRetrain] = useState<Track2RetrainStatus | null>(null);
  // No cadence state here any more: the interval and the time are set
  // ONCE, in the PULL, BUILD, QUEUE, TRAIN strip, which mirrors them
  // onto this clock server-side. `retrain` is still read so the card
  // can say when the next run is.
  // Read the chosen dataset's taxonomy and PRESELECT the pipeline from it,
  // rather than letting the operator discover from a rejection that their
  // data is not person/helmet. Advisory only: it never routes the run,
  // because the production path carries meaning beyond its taxonomy and a
  // person/helmet dataset can legitimately want either pipeline.
  useEffect(() => {
    if (!datasetFolder) { setDatasetClasses(null); return; }
    let cancelled = false;
    getTrack2DatasetClasses(datasetFolder)
      .then((res) => {
        if (cancelled) return;
        setDatasetClasses(res);
      })
      .catch(() => { if (!cancelled) setDatasetClasses(null); });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetFolder]);


  // The architecture list moved with the size dropdown, into the Models
  // card. What is still needed here is the registry rows, and only to
  // name the version the run says it is starting from.
  useEffect(() => {
    let cancelled = false;
    getTrack2PullControl()
      .then((status) => {
        if (cancelled) return;
        // The saved choice, so a reloaded page -- or a loaded settings
        setAssembleSaved(readAssembleSaved(status.config.dataset_assemble));
        const store = status.config.central_store?.folder;
        if (store) setSelection({ datasetFolder: `${store}_decay`, evalFolder: `${store}_decay` });

      })
      .catch(() => undefined);
    getTrack2Retrain()
      .then((res) => !cancelled && setRetrain(res))
      .catch(() => undefined);
    getTrack2QueueAfterPull()
      .then((res) => !cancelled && setAfterPull(res))
      .catch(() => undefined);
    // The metrics rows carry everything the package list did, plus the
    // measured quality, so there is one source rather than two that can
    // disagree about which versions exist.
    getTrack2Metrics()
      .then((res) => !cancelled && setRegistryRows(res.rows))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
    // Refetched on an import too: importing a checkpoint adds a version to
    // the very table this populates, and without this it would not appear
    // until the page was reloaded.
  }, [detectorsVersion]);


  // The folder browsers that used to live here are gone: the dataset is
  // chosen once, in "Assemble a dataset", and both training cards use
  // that. Nothing in this card browses the bucket any more.

  // Derived from the architecture rather than chosen separately: the size
  // IS the tail of the architecture name, and a second control for the same
  // fact is one that can disagree with it.
  const { generation, size } = splitArchitecture(architecture);

  // Which registry is shown is DERIVED from the choice in step 1 -- the
  // detector, the generation and the size name exactly one group, so a
  // control for it here would be a second place to say one thing.
  //
  // BY GENERATION, not the flat `groups` map. That map is generation 11
  // only, so a yolo26s run reported "registers into
  // yolo11s-helmet-detector" while the server correctly used
  // yolo26s-helmet-detector -- the card and the run disagreeing about
  // where a model was about to be filed.
  const activeGroupName = chosenDetector && size
    ? chosenDetector.groups_by_generation?.[generation]?.[size]
      ?? (generation === "11" ? chosenDetector.groups[size] ?? "" : "")
    : "";

  // A source chosen but not completed. "registry" with nothing picked is
  // NOT incomplete on the production path -- it means "leave it to the
  // coordinator" -- but it is on the open one, which has no default.
  // There is one source now: a registered version. The exemption that
  // used to sit here was for random weights, which needed none.
  const baseSelectionIncomplete = !basePackageArn;

  /** Everything that decides whether this run can work, as one verdict.
   *
   * The compatibility that matters is between the DATASET's classes and
   * the DETECTOR's -- and by extension the registry the result lands in,
   * because a version filed under Helmet whose model detects something
   * else is mislabelled for the rest of its life. `blocking` is reserved
   * for what the server would actually refuse; everything else informs
   * without standing in the way.
   */
  const analysis = (() => {
    const lines: { tone: "ok" | "warn" | "bad"; text: string }[] = [];
    let blocking = false;

    if (dataChoice.mode === 'saved') {
      blocking = !dataChoice.checked || !chosenDetector || !basePackageArn;
      lines.push({tone: blocking ? 'bad' : 'ok', text: !chosenDetector || !basePackageArn
        ? 'Choose a model and starting version in step 1.' : !dataChoice.checked
        ? 'Choose and check a saved SQL dataset before queueing.'
        : 'The saved dataset will be rebuilt from its original central store using the current model and training settings. Files are checked again before queueing.'});
      return {lines, blocking};
    }

    lines.push({ tone: "ok", text: "Training uses the already-prepared decayed dataset. Its exact files and settings are preserved before queueing." });

    if (!chosenDetector) {
      lines.push({
        tone: "warn",
        text: "No Model Name chosen -- the model lands in the shared open registry, "
          + "where version numbers are not per model type.",
      });
    } else if (assembledBuild && !Object.keys(assembledBuild.classes).length) {
      // The server refuses this too (`assemble_plan`), so it blocks here.
      blocking = true;
      lines.push({
        tone: "bad",
        text: "Enter all dataset Label classes in Data Preparation before queueing training.",
      });
    } else if (checkedClasses) {
      if (absentClasses.length > 0) {
        blocking = true;
        // "Does not declare", never "has no labels": both sides come from
        // a class LIST, and no .txt label file is opened to decide this.
        lines.push({
          tone: "bad",
          text: `The prepared dataset declares ${datasetNames.join(", ") || "nothing"}. `
            + `It does not declare ${absentClasses.join(", ")}, so a ${chosenDetector.label} model cannot train from it.`,
        });
      } else if (willFilter) {
        lines.push({
          tone: "warn",
          text: `The dataset declares ${datasetNames.join(", ")}; a copy holding only `
            + `${detectorNames.join(", ")} will be derived first, for both the training `
            + "and evaluation sets. Images without those classes are kept as backgrounds.",
        });
      } else {
        lines.push({
          tone: "ok",
          text: `Dataset classes match ${chosenDetector.label} exactly `
            + `(${detectorNames.join(", ")}).`,
        });
      }
    }

    // The base model's own classes do not have to match: the head is
    // rebuilt. What matters is that one was chosen at all.
    const base = registryRows.find((r) => r.model_package_arn === basePackageArn);
    if (!basePackageArn) {
      blocking = true;
      lines.push({ tone: "bad", text: "No starting version selected." });
    } else if (base) {
      const sameRegistry = base.registry === detectorSlug;
      lines.push({
        tone: "ok",
        text: `Starting from ${base.group_name} v${base.version}`
          + (sameRegistry
            ? ` -- continues this detector, so the result is v${Number(base.version) + 1}.`
            : " -- a seed from another registry; the head is rebuilt for these classes."),
      });
    }

    if (chosenDetector && activeGroupName) {
      lines.push({ tone: "ok", text: `Result registers into ${activeGroupName}.` });
    }
    return { lines, blocking };
  })();

  /** Clears every choice in this card back to its default. */
  function handleStartAllOver() {
    setDataChoice({mode: 'prepared', id: '', checked: false});
    setSelection({
      epochs: 50, imgSize: 640, batch: 16, seed: 42, patience: 0,
      hyperparams: {},
      datasetFolder: "", evalFolder: "",
      detectorSlug: "", architecture: DEFAULT_ARCHITECTURE,
      basePackageArn: "", notes: "",
    });
    setError(null);
  }


  // Starting a run directly was removed with the button that did it: both
  // cards queue, and the queue is what gets started. The backend still
  // exposes both start endpoints, so this is a UI decision, not a
  // capability that has gone.
  async function handleAddToQueue() {
    setBusy(true);
    setError(null);
    try {
      const got = await runTrack2Retrain("enqueue", true, {
        epochs, img_size: imgSize, batch, seed, patience, hyperparams,
        detector_slug: detectorSlug, architecture, notes,
        base_version: basePackageArn,
        delete_decay_after_run: deleteDecayAfterRun,
        ...(dataChoice.mode === 'saved' ? {dataset_version_id: dataChoice.id, delete_decay_after_run: false} : {}),
      });
      const failed = got.steps.find((step) => !step.ok);
      if (!got.ok || failed) {
        setError(failed?.detail || "Could not preserve and queue the prepared dataset.");
        return;
      }
      if (got.job) setPreparationJob(got.job);
      else onSubmitted();
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Add to queue failed.");
    } finally {
      setBusy(false);
    }
  }

  // Either pipeline. Keyed only on training_progress, the Stop button was
  return (
    <>
    {/* Its own card: the base model is chosen
        per detector and size, and those two choices belong together
        rather than buried among the dataset and hyperparameters. */}
    <Card
      title="Start the training run"
      step="6"
      help={
        <>
          <p>
            Choose newly prepared data, a saved SQL dataset with your current
            step 1 settings, or a previous run with its recorded settings.
          </p>
          <p className="mt-2">
            Newly prepared data uses the decayed dataset already produced in Data Preparation.
            A saved dataset uses its recorded central store, even if today’s store has changed.
            Check saved dataset verifies the files and class names. Matching names in a
            different order are remapped automatically; selecting a subset is not supported here.
            Add to Queue saves a new SQL recipe before training starts.
          </p>
          <p className="mt-2">
            When the validation set lives under a different prefix &mdash; its
            own download, read by the evaluation step &mdash; it is listed
            separately as <em>Evaluation</em>. The training job validates on
            its own images in that case, because folders under another prefix
            are not on its disk, so the held-out numbers are the ones the
            evaluation step reports.
          </p>
          <p className="mt-2">
            The model comes from step 1, where the <em>Model Name</em>,{" "}
            <em>Model</em> and <em>Size</em> dropdowns pick it. The Model Name
            decides <strong>which registry</strong> the trained model is
            versioned into, and its class names are what the dataset is checked
            against. If the dataset declares more class names than the Model Name
            wants, the run trains on just that Model Name&apos;s names and nothing
            in S3 is rewritten.
          </p>
          <p className="mt-2">
            Submitting adds it to the queue below. A run takes GPU time, so
            check both lines say what you expect before pressing it.
          </p>
        </>
      }
      next="The Training queue below -- press Start Queue to let it run."
    >
      {/* No arming tick here any more. It existed to grey this card out
          while IMPORT was selected, and import is no longer one of the
          modes -- so the box could only ever be ticked, and clicking it
          did nothing. Both remaining modes submit from here. */}
      <Track2TrainingDataChoice value={dataChoice} onChange={setDataChoice}
        detectorSlug={detectorSlug} classKey={JSON.stringify(chosenDetector?.classes || {})}
        disabled={busy || !!preparing} onSubmitted={onSubmitted} />
      {dataChoice.mode !== 'repeat' && <fieldset>

      {/* The detector, the generation, the size and the version to start
          from are all chosen in step 1 now. Repeating those controls here
          was the confusing part: two places to say one thing, with nothing
          forcing them to agree. What is left is the read-back, so what is
          about to run is still visible at the moment of pressing the
          button. */}
      <div className="tabular mb-3 rounded-md border border-black/15 bg-page p-2.5 text-xs">
        <p className="text-ink-muted">
          Model{" "}
          <span className="text-ink-muted/70">(chosen in step 1, Model types and Model Names)</span>
        </p>
        {detectorSlug ? (
          <>
            <p className="mt-0.5 text-ink">
              {detectorSlug} &middot; {architecture}
            </p>
            <p className="mt-0.5 text-ink-secondary">
              Starting from{" "}
              {basePackageArn
                ? basePackageArn.split("/").slice(-2).join(" v")
                : "nothing chosen yet"}
            </p>
          </>
        ) : (
          <p className="mt-0.5 text-ink">
            Nothing chosen yet. Open step 1 and pick one in the{" "}
            <em>Model Name</em> dropdown.
          </p>
        )}
      </div>


      {/* Epochs, image size, batch, seed, patience and the note all moved
          to step 1, "How every run trains". They were never per-run in
          practice -- the same numbers were retyped for every run, and the
          scheduled retrain kept its own copy that could differ without
          anything saying so. One set now feeds the form, "Train on this"
          on a snapshot, and the schedule. This reads them back. */}
      <div className="tabular mb-3 rounded-md border border-black/15 bg-page p-2.5 text-xs">
        <p className="text-ink-muted">
          How it trains{" "}
          <span className="text-ink-muted/70">(chosen in step 1)</span>
        </p>
        <p className="mt-0.5 text-ink">
          {epochs} epochs &middot; {imgSize}px &middot; batch {batch} &middot;
          {" "}seed {seed} &middot;{" "}
          {patience ? `patience ${patience}` : "no early stop"}
        </p>
        {notes && <p className="mt-0.5 truncate text-ink-secondary">{notes}</p>}
      </div>

      {/* The analysis, immediately above the action it governs. The
          question it answers -- can this dataset train this detector --
          was previously answered by a rejection after the fact. */}
      <div className={`mt-3 rounded-md border p-3 ${
        analysis.blocking ? "border-critical/40 bg-critical-subtle" : "border-black/15 bg-page"
      }`}>
        <p className="mb-1 text-xs font-medium text-ink">Analysis</p>
        {analysis.lines.length === 0 ? (
          <p className="text-xs text-ink-muted">
            Pick a dataset and a Model Name in step 1 to check them.
          </p>
        ) : (
          analysis.lines.map((line) => (
            <p
              key={line.text}
              className={`text-xs ${
                line.tone === "bad" ? "text-critical"
                  : line.tone === "warn" ? "text-ink-secondary"
                  : "text-ink-muted"
              }`}
            >
              {line.tone === "bad" ? "✗ " : line.tone === "warn" ? "! " : "✓ "}
              {line.text}
            </p>
          ))
        )}
      </div>

      {dataChoice.mode === 'prepared' && <label className="mt-3 flex items-start gap-2 text-xs text-ink">
        <input type="checkbox" checked={deleteDecayAfterRun} disabled={busy || preparing}
          onChange={(event) => setDeleteDecayAfterRun(event.target.checked)} />
        <span>Delete the decayed dataset after this run
          <span className="mt-1 block text-ink-muted">
            Removes the working decay copy once training finishes and the queue is clear.
            Preserved run data stays available for repeat runs and exports.
            Prepare a new decayed dataset before adding another new run.
          </span>
        </span>
      </label>}
      {preparationJob && (
        <div className="mt-3 rounded-md border border-black/15 bg-page p-3 text-xs"
          role="status" aria-live="polite">
          <div className="mb-2 flex justify-between gap-3">
            <span>{preparationJob.stage}</span><span>{preparationJob.percent}%</span>
          </div>
          <div role="progressbar" aria-label="Dataset preparation" aria-valuemin={0}
            aria-valuemax={100} aria-valuenow={preparationJob.percent}>
            <ProgressBar processed={preparationJob.percent} total={100} />
          </div>
          {preparationJob.total > 0 && <p className="mt-1">
            {preparationJob.completed.toLocaleString()} / {preparationJob.total.toLocaleString()} image and label pairs
          </p>}
          {preparing && <p className="mt-1 text-ink-muted">You can close this page. The background worker will continue; progress returns when you reopen it.</p>}
          {preparationJob.error && <p className="mt-1 text-critical">{preparationJob.error}</p>}
        </div>
      )}
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={handleAddToQueue}
          disabled={busy || preparing || baseSelectionIncomplete || analysis.blocking}
          className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy || preparing
            ? "Preserving the prepared dataset and queueing..."
            : "Add to Queue"}
        </button>
        <button
          type="button"
          onClick={handleStartAllOver}
          disabled={busy || preparing}
          className="rounded-md border border-black/15 px-3 py-2 text-sm font-medium text-ink-secondary disabled:opacity-40"
        >
          Start All Over
        </button>
      </div>
      </fieldset>}

      {/* Automatic queueing uses the prepared dataset too. */}
      <div className="mt-3 rounded-md border border-black/15 p-3">
        {/* READ-ONLY (2026-10-26). "Enabled for scheduling" was a tick
            here, a third switch for the chain beside the Retention card's
            and the strip's mode. The strip is the one switch now:
            `queue_after_pull.settings` derives on/off from the pull mode. */}
        <p className="text-sm">
          <span className="font-medium text-ink">Runs automatically: </span>
          <span className={afterPull?.settings.enabled ? "text-ink" : "text-ink-muted"}>
            {afterPull?.settings.enabled
              ? (afterPull.settings.triggers ?? []).includes("arrival")
                ? "when files arrive"
                : "after each scheduled pull"
              : "off"}
          </span>
          <span className="mt-0.5 block text-xs text-ink-muted">
            Switched on and off by <strong>TRAINING SCHEDULE</strong> at
            the top. The background worker pulls fresh data, rebuilds and preserves the
            decayed dataset, then queues the run and starts it, even when this page is closed.
            The cleanup option above applies
            to runs you add here with Add to Queue.
          </span>
          {/* When the training clock is in charge -- Pull Control is not
              a source, so no sweep runs -- its next slot, from the same
              settings the strip writes. */}
          {retrain && retrain.follows_pull === false && (
            <span className="mt-2 block text-xs text-ink-muted">
              {retrain.settings.enabled && retrain.next_runs?.length
                ? `Next run ${localDate(retrain.next_runs[0])}.`
                : "Not armed -- set “On a schedule” in TRAINING SCHEDULE at the top."}
              {retrain.problems?.some((p) => p.level === "blocking") && (
                <span className="mt-1 block text-critical">
                  {retrain.problems.filter((p) => p.level === "blocking")[0].text}
                </span>
              )}
            </span>
          )}
        </p>

        {afterPull?.settings.enabled && (
          <div className="mt-2 space-y-1 border-t border-hairline pt-2">
            <label className="flex items-start gap-2 text-xs text-ink-secondary">
              <input
                type="checkbox"
                className="mt-0.5 shrink-0"
                disabled={afterPullBusy}
                checked={!!afterPull.settings.even_if_nothing_moved}
                onChange={async (e) => {
                  setAfterPullBusy(true);
                  try {
                    setAfterPull(await saveTrack2QueueAfterPull({
                      even_if_nothing_moved: e.target.checked,
                    }));
                  } catch {
                    /* keep what is on screen */
                  } finally {
                    setAfterPullBusy(false);
                  }
                }}
              />
              <span>
                Queue even when the pull moved no files
                <span className="block text-ink-muted">
                  Off, a pull that found nothing is skipped -- the corpus is
                  unchanged, so the run would repeat the last model and bill
                  for it.
                </span>
              </span>
            </label>

            <p className="text-xs text-ink-muted">
              <span className="font-medium">Next:</span>{" "}
              {afterPull.next.act
                ? `would queue a run for pull ${afterPull.next.run?.run_id ?? ""}`
                : afterPull.next.reason}
            </p>
            {afterPull.settings.last_result?.at && (
              <p className="tabular text-xs text-ink-muted">
                <span className="font-medium">Last:</span>{" "}
                {afterPull.settings.last_result.ok ? "queued" : "did not queue"}{" "}
                after pull {afterPull.settings.last_result.run_id} --{" "}
                {afterPull.settings.last_result.detail}
              </p>
            )}
          </div>
        )}
      </div>

    </Card>

    {/* LAST in this form, which puts it immediately above the Training
        queue -- where a refused add is looked for, and where the entry
        would have appeared had it worked.

        It began at the top of the form, several cards up, so pressing
        Add to Queue and being refused looked like nothing happening at
        all -- the server's reason was complete and actionable and
        reached nobody. */}
    {error && (
      <div className="flex items-start gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-sm text-critical">
        <AlertCircle size={16} className="mt-0.5 shrink-0" />
        <span>{error}</span>
      </div>
    )}

    </>
  );
}

/** The training page.
 *
 * Two routes render THIS component, so the GPU status, the training
 * queue and the run history are not merely similar between them -- they
 * are the same component over the same state. Anything added to one is
 * in the other by construction.
 *
 * `variant` only decides which cards appear:
 *   standard  -- everything
 *   multi     -- multi-source assemble, pull control and the golden set.
 *                Import is on BOTH tabs: it was
 *                standard-only siblings once, which left the tab the
 *                guided 1-6 flow lives on with no way to bring in a
 *                checkpoint at all.
 */
export function Track2ModelsPage({ variant = "standard" }: { variant?: "standard" | "multi" } = {}) {
  const multi = variant === "multi";
  const [data, setData] = useState<Track2ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  // `message` went with the queue buttons that were its only writer --
  // their replies now render inside the queue card, beside the controls
  // that produced them. `error` stays: the page-level load failure still
  // uses it.
  // Bumped whenever a run is submitted, so the history below the queue
  // picks up the new row without the operator having to reload the page.
  const [historyKey, setHistoryKey] = useState(0);
  // Same idea for the detector list the training form below holds.
  const [detectorsVersion, setDetectorsVersion] = useState(0);
  // The checkpoint the import card has inspected and is holding, while
  // the Class it will be filed under is defined by hand below. Held here
  // because the two cards are siblings: one knows what was inspected,
  // the other knows the Class about to exist.
  const [pendingImport, setPendingImport] = useState<Track2PendingImport | null>(null);
  // A version that was JUST registered, so the page can move step 1 onto
  // its Class and size and preselect it. Without this an import
  // succeeded and then vanished: the base-model picker filters on step
  // 1, which was still pointing at whatever was selected before, so the
  // version that had just been created was on no screen anywhere.
  const [justRegistered, setJustRegistered] =
    useState<{ slug: string; architecture: string; arn: string } | null>(null);
  // There is no longer a choice of where a run's starting weights come
  // from: it continues a registered version, always. `modelSource` held
  // "scratch" | "existing" and the whole machine around it -- a saved
  // `default_source`, a restore, a refusal lookup per Class, and an
  // effect to unstick scratch when a Class could not use it -- existed
  // only to pick between the two.
  //
  // The SERVER still accepts `base_model_scratch`, still resolves the
  // random-init seeds and still refuses scratch on a Class with a
  // lineage. Removed from the page, not from the platform: the seeds are
  // real objects in S3 and the refusal is the rule that keeps a v4
  // continuing v3, neither of which becomes wrong because a card went.
  // What the Models card says everything trains as. Held here rather than
  // in either card, because two of them read it.
  const [modelChoice, setModelChoice] = useState({
    slug: "", generation: "11", size: "s", architecture: "yolo11s", group: "",
  });
  /** A Model Name is chosen, so defining one and importing a checkpoint
   * are both out of play: they are the two ways to CREATE the thing that
   * already exists.
   *
   * Cleared by clearing the choice -- the Model Name dropdown's blank
   * option -- rather than by a button, so there is one control deciding
   * this rather than two that can disagree. */
  const modelLocked = Boolean(modelChoice.slug);
  const [selection, setSelectionState] = useState<TrainingSelection>({
    epochs: 50,
    imgSize: 640,
    batch: 16,
    seed: 42,
    patience: 0,
    hyperparams: {},
    datasetFolder: "",
    evalFolder: "",
    detectorSlug: "",
    architecture: DEFAULT_ARCHITECTURE,
    basePackageArn: "",
    notes: "",
  });

  function setSelection(patch: Partial<TrainingSelection>) {
    setSelectionState((current) => ({ ...current, ...patch }));
  }

  // The starting version a settings file saved. Restored HERE, not in the
  // training form: the setter is this component's. Which version a run
  // starts FROM is half of what defines the run, so a file holding the
  // Class but not the version restores a different run.
  useEffect(() => {
    let cancelled = false;
    getTrack2PullControl()
      .then((status) => {
        if (cancelled) return;
        if (status.config.default_version) {
          setSelection({ basePackageArn: status.config.default_version });
        }
      })
      .catch(() => undefined);
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function load() {
    getTrack2Models()
      .then((res) => {
        setData(res);
        // No dataset prefill. It used to seed `default_dataset` whenever
        // the field was empty -- and because this runs on a 15s poll, not
        // just at mount, clearing the dataset (or pressing Start All Over)
        // silently refilled it moments later with a folder nobody chose.
        // The analysis then read THAT folder's classes, which is how a
        // six-class dataset appeared under a helmet detector.
        //
        // Beyond the bug: the dataset now comes from "Assemble a dataset",
        // so a raw folder appearing by default contradicts the one rule
        // both training cards follow.
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load training status."));
  }

  useEffect(load, []);

  // Auto-refresh every 15s while a run or queue is active -- mirrors the
  // old app's <meta refresh>, done here as a poll instead.
  const shouldPoll = useMemo(
    // `custom_run_id` belongs here as much as the other three.
    //
    // `training_progress` reports the PRODUCTION pipeline, which never
    // goes Executing for a custom-class run, and `queue.started` is set
    // false in memory the moment the queue empties -- which is exactly
    // when the last entry has just STARTED. So from the instant a custom
    // run began, nothing here was true: the page stopped polling, and
    // History sat on "Executing" until it was reloaded by hand.
    () => Boolean(
      data?.training_progress?.status === "Executing"
      || data?.custom_run_id
      || data?.queue.started
      || data?.pending_message,
    ),
    [data],
  );
  useEffect(() => {
    if (!shouldPoll) return;
    // The history reloads on this tick too. `load()` refreshes the queue
    // and the run progress but not the history list, whose only other
    // trigger is handleSubmitted -- a MANUAL submit. A queue working
    // through its entries in the background never calls that, so the
    // second run of a queued pair started, ran and finished while the
    // History section still showed only the first, until the page was
    // reloaded by hand. Bounded: shouldPoll is false once nothing is
    // running, so an idle page makes no extra requests.
    const id = setInterval(() => {
      load();
      setHistoryKey((k) => k + 1);
    }, 15000);
    return () => clearInterval(id);
  }, [shouldPoll]);

  // The LIVE bits, polled unconditionally. `shouldPoll` above only runs
  // while something is already known to be happening, which cannot be
  // true of a page loaded before anything started -- and that is exactly
  // the case that broke: a scheduled chain began minutes after the page
  // was opened, and the GPU card went on saying "Available" in green
  // while a job downloaded its dataset, because nothing ever looked
  // again. This endpoint is three cheap calls, so it can just always
  // look; the heavy payload keeps its conditional poll.
  // Held across ticks so the moment a run FINISHES can be noticed. The
  // queue showed a run as going until somebody pressed refresh: the live
  // poll updated the status, but History is loaded by its own component
  // on `historyKey`, and the registries behind "which version is newest"
  // come from the heavy payload. So the finished run existed in three
  // places and only one of them had heard.
  const lastStatus = useRef<string | null>(null);
  /** The custom pipeline's equivalent. `training_progress` describes the
   * production pipeline, so a custom run finishing moved nothing here --
   * and the finish is the moment History most needs re-reading. */
  const lastCustomRun = useRef<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    const look = () => {
      getTrack2ModelsLive()
        .then((live) => {
          if (cancelled) return;
          setData((current) => (current ? { ...current, ...live } : current));
          const now = live.training_progress?.status ?? null;
          const before = lastStatus.current;
          lastStatus.current = now;
          // Executing -> anything else. Not "is terminal": that would
          // fire on every tick of an already-finished run and reload the
          // page's heaviest endpoint every ten seconds forever.
          if (before === "Executing" && now && now !== "Executing") {
            setHistoryKey((k) => k + 1);
            load();
          }
          // A custom run ending: id -> null. Same reload, different
          // pipeline. Without it a stopped or failed custom run stayed
          // "Executing" in History for as long as the page was open --
          // the audit row said Stopped, nothing asked it again.
          const customNow = live.custom_run_id ?? null;
          const customBefore = lastCustomRun.current;
          lastCustomRun.current = customNow;
          if (customBefore && !customNow) {
            setHistoryKey((k) => k + 1);
            load();
          }
        })
        .catch(() => undefined);
    };
    look();
    const id = setInterval(look, 10000);
    return () => { cancelled = true; clearInterval(id); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function handleQueueRemove(id: string) {
    await removeTrack2Queue(id);
    load();
  }

  // No banner on either. "Queue started" restated what the Training
  // queue card shows -- its buttons, its running entry, its counts --
  // and then stayed on screen after the queue had moved on. The queue is
  // the record; `load()` refreshes it so it is right by the time the
  // button finishes.
  /** Which of the three queue buttons is mid-flight.
   *
   * Each of them is a round trip that can take a moment -- Stop Training
   * calls SageMaker -- and all three left the button looking exactly as
   * it did before. Pressing again was the only way to find out whether
   * the first press had registered, and "Start Queue" was pressed eleven
   * times in a row that way. */
  const [queueBusy, setQueueBusy] =
    useState<"" | "start" | "stop" | "stop-training">("");
  /** What the last queue action replied, shown IN the queue card.
   *
   * The page's own banners sit nearly three hundred lines above these
   * buttons, off the top of the screen. "Queue is empty -- add at least
   * one run first" delivered up there is the refusal arriving somewhere
   * the person pressing is not looking. */
  const [queueNote, setQueueNote] =
    useState<{ text: string; bad: boolean } | null>(null);
  /** Stop Training asks first, because it can now reach jobs this app
   * did not start. `alsoForeign` is the harsher answer, carried in the
   * same dialog rather than behind a second control somewhere else. */
  const [stopOpen, setStopOpen] = useState(false);
  const [alsoForeign, setAlsoForeign] = useState(false);

  async function handleQueueStart() {
    if (queueBusy) return;
    setQueueBusy("start");
    setQueueNote(null);
    try {
      // The reply, SHOWN. This used to be awaited and dropped, so a
      // refusal -- "Queue is empty -- add at least one run first" --
      // arrived as a 400 the page never mentioned. The button simply
      // did nothing, eleven times.
      setQueueNote({ text: (await startTrack2Queue()).message, bad: false });
    } catch (err) {
      setQueueNote({
        text: err instanceof ApiError ? err.detail : "Could not start the queue.",
        bad: true,
      });
    } finally {
      setQueueBusy("");
      load();
    }
  }

  async function handleQueueStop() {
    if (queueBusy) return;
    setQueueBusy("stop");
    setQueueNote(null);
    try {
      setQueueNote({ text: (await stopTrack2Queue()).message, bad: false });
    } catch (err) {
      setQueueNote({
        text: err instanceof ApiError ? err.detail : "Could not stop the queue.",
        bad: true,
      });
    } finally {
      setQueueBusy("");
      load();
    }
  }


  /** Kills the run currently on the GPU. NOT the same as Stop Queue:
   * /queue/stop only halts the loop and lets the in-flight run finish,
   * so without this there is no way to end a run early and the
   * single-GPU guard blocks everything else until it times out. */
  async function handleStopTraining(includeForeign: boolean) {
    if (queueBusy) return;
    setQueueBusy("stop-training");
    setQueueNote(null);
    try {
      const res = await stopTrack2Training(includeForeign);
      setStopOpen(false);
      setAlsoForeign(false);
      setQueueNote({ text: res.message, bad: !!res.problems?.length });
    } catch (err) {
      setQueueNote({
        text: err instanceof ApiError ? err.detail : "Stop failed.",
        bad: true,
      });
    } finally {
      // In `finally`, not after the await: this one calls SageMaker, so
      // a failure is exactly when the button must stop saying
      // "Stopping training..." forever.
      setQueueBusy("");
      load();
    }
  }

  function handleSubmitted() {
    // No banner. The queue below is where a queued run is read, and it
    // is refreshed here so the entry is there by the time the button
    // finishes.
    setHistoryKey((k) => k + 1);
    load();
  }

  /** GPU jobs this app did not start. `own_pipeline` ones belong to a
   * pipeline execution and are stopped through THAT -- stopping the job
   * under a running execution leaves the execution to fail its step
   * rather than stop cleanly. */
  const foreignGpuJobs = (data?.gpu_jobs ?? []).filter((j) => !j.own_pipeline);
  /** Whether there is a run of ours to stop at all. The button is also
   * reachable with only a foreign job on the GPU, and the dialog has to
   * say which of the two situations it is. */
  const stoppableOwnRun = Boolean(
    data?.queue.running
    || data?.training_progress?.status === "Executing"
    || data?.custom_run_id,
  );

  return (
    <div>
      {/* The title and the toolbar travel together and do not move --
          a Word-style ribbon. Everything below scrolls under it.

          `-mx-4 px-4` (and the md pair) cancels the page gutter so the
          band reaches both edges, otherwise content scrolls through the
          margins beside it. The background is opaque for the same
          reason: a translucent strip with text sliding under it is
          unreadable exactly when you are using it. */}
      <div className="sticky top-0 z-30 -mx-4 -mt-6 mb-4 border-b border-hairline bg-page px-4 pb-2 pt-6 md:-mx-8 md:-mt-8 md:px-8 md:pt-8">
        <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
          <Cpu size={20} className="text-accent" strokeWidth={2} />
          Track2 Training
        </h1>
        <p className="mb-2 max-w-3xl text-sm text-ink-secondary">
          Dataset/weights browsing, GPU status, training progress, and the
          training queue for the Track2 YOLO pipeline.
        </p>
        <Track2PinnedBar scope="at" />
      </div>

      <Track2TrainingDataFile />

      {/* The cost panel moved INSIDE the pinned bar, directly under the
          GPU line -- "what is the GPU doing" and "what has it cost
          today" are the same glance, and it now travels with the rest
          of the always-visible strip. */}

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}

      {!data && !error && (
        <div className="flex items-center gap-2 text-sm text-ink-muted">
          <Loader2 size={15} className="animate-spin" />
          Loading...
        </div>
      )}

      {data && (
        <div className="flex flex-col gap-5">
          {/* The "SageMaker GPU resource" card used to sit here. It is
              gone: it rendered from the page-load snapshot, so it showed
              "Available" in green while a training job was downloading
              its dataset, and the one person who reads that card is the
              one about to press Start. The GPU indicator in the pinned
              bar above is the same information, polled, with this card's
              job list kept as its tooltip -- and it now has the CPU
              beside it, which is what a run is actually using for the
              seven or eight minutes after training releases the GPU. */}

          {/* Before the training cards: a dataset has to be in the shape
              the trainer expects before any of them can use it. Assembling
              one selects it as the training folder below. */}
          {/* Both fields, same folder. An assembled dataset carries its
              own train and val splits, so the "evaluation folder" is that
              same folder -- setting it explicitly stops the UI reporting
              an empty eval folder as "same as training", which it is not. */}
          {/* First on the page: a class list is the taxonomy everything
              downstream is checked against -- the dataset must declare
              exactly these, and the registry group is named after it.
              Collapsed by default, because it is defined once and then
              left alone. */}
          {/* One card for "which model", on BOTH tabs. Import and scratch
              used to be standard-tab-only siblings, so the multi tab --
              the one the whole guided 1-6 flow lives on -- had no way to
              bring in a checkpoint at all. Nested here they are the same
              implementation on both, which is the only way the two tabs
              cannot drift. */}
          <Track2DetectorsPanel
            // Dimmed UNLESS a checkpoint has been read with its own
            // class names declined. That is the one state in which
            // defining a Class by hand is the next thing to do -- and
            // every other state has an answer somewhere else: no
            // checkpoint means there is nothing for the Class to hold,
            // and a checkpoint whose classes are accepted has already
            // named them.
            //
            // This replaced `fileClassesInUse`, which said "the import
            // card is supplying the classes" -- only one of the ways
            // this card is not the place to be.
            defineDimmed={!pendingImport}
            lockedByPick={modelLocked}
            selectVersion={justRegistered}
            onVersionSelected={() => setJustRegistered(null)}
            onBaseVersion={(arn) => {
              setSelection({ basePackageArn: arn });
              // Saved server-side like any other choice of starting
              // version, so the settings file carries it.
              void saveTrack2PullConfig({ default_version: arn })
                .catch(() => undefined);
            }}
            pendingImport={pendingImport}
            onImported={() => setDetectorsVersion((n) => n + 1)}
            onChanged={() => setDetectorsVersion((n) => n + 1)}
            // The run's shape flows the same way the model does: chosen
            // once in step 1, written into the selection the form
            // submits, so there is no second copy to disagree.
            onTraining={(t) => setSelection({
              epochs: t.epochs, imgSize: t.imgSize, batch: t.batch,
              seed: t.seed, patience: t.patience, notes: t.notes,
              hyperparams: t.hyperparams,
            })}
            // Above "Define a new Class", not down with the training
            // modes: importing is something you do before there is
            // anything to train on, like defining the Class itself.
            aboveDefine={
              <Track2ImportCheckpointPanel
                onPendingImportChange={setPendingImport}
                onImported={(registered) => {
                  // The picker re-reads the registry...
                  setDetectorsVersion((n) => n + 1);
                  // ...and step 1 moves onto what was just registered,
                  // because the picker filters on step 1. Without this
                  // the version appeared in a list nobody was looking
                  // at: an import that succeeds and cannot be found is
                  // the same to the operator as one that failed.
                  if (registered) setJustRegistered(registered);
                }}
                defaultArchitecture={modelChoice.architecture}
                // Dimmed from the inside rather than wrapped in
                // `SwitchedOff`. The wrapper's banner sat above the card
                // as a second line saying what the card already says,
                // and everything it wrapped -- the reason included --
                // faded with it.
                lockedByPick={modelLocked}
              />
            }
            // One choice, two readers. Writing it straight into the
            // training selection is what makes the form below inherit it
            // rather than keep a second copy that can disagree.
            onChoose={(choice, restored) => {
              setModelChoice(choice);
              setSelection({
                detectorSlug: choice.slug,
                architecture: choice.architecture,
                // Clearing the version is right when somebody PICKS a
                // different Class -- a version belongs to one registry
                // group and cannot follow. It is wrong on a restore,
                // where the version being put back is the one that
                // belongs to this Class: this card and the picker each
                // fetch the config separately, so whichever landed last
                // won, and when it was this one it wiped the version the
                // other had just restored. That is why a saved file came
                // back with the Class and size right and the version
                // empty.
                ...(restored ? {} : { basePackageArn: "" }),
              });
            }}
          >
            {/* The one way a run starts. "Train from scratch" was the
                other, and was removed on request: a from-scratch run
                needs far more data and many more epochs than fine-tuning
                a registered version, and takes correspondingly longer on
                the single GPU slot everything here queues behind.
                Importing is neither mode -- it moved up beside "Define a
                new Model Name". */}
            <Track2BaseModelPicker
              detectorSlug={modelChoice.slug}
              detectorLabel={modelChoice.slug || "this Model Name"}
              architecture={modelChoice.architecture}
              generation={modelChoice.generation}
              size={modelChoice.size}
              groupName={modelChoice.group}
              value={selection.basePackageArn}
              onChange={(arn) => {
                setSelection({ basePackageArn: arn });
                // Saved server-side, so the central Save carries it.
                // Which version a run starts FROM is half of what
                // defines the run; a settings file holding the Class
                // but not the version restores a different run.
                void saveTrack2PullConfig({ default_version: arn })
                  .catch(() => undefined);
              }}
              refreshKey={detectorsVersion}
            />
          </Track2DetectorsPanel>

          {/* The standard tab's copy. Multi has its own below, with the
              tick that switches the week-by-week cards off -- a tick that
              would turn nothing off here, because none of those three
              cards exist on this tab. Same component either way, so the
              two cannot drift. */}
          {!multi && (
            <Track2DatasetAssemblePanel
              multiSource={multi}
              // The Model Name chosen in step 1, so the classes written
              // into data.yaml are the ones the run registers under.

            />
          )}

          {/* The data cards -- Assemble, Golden Set, Pull Control, the
              weeks with Retention inside, and the pull queue -- moved to
              the Data Preparation tab with the data/training split
              (2026-10-28). They prepare what a run trains on; this tab
              decides what to train and trains it. What they chose reaches
              this tab through the saved settings, and the notice board
              (lib/bus.ts) says when it changed. */}

          {/* The data.yaml the run will build, next to the button that
              trains on it. On the standard tab the Assemble card above
              renders it here by portal. On this tab the card is on the
              Data Preparation tab, so its preview-only mode draws the same
              preview from the saved settings -- the same code, so the two
              cannot disagree. */}
          {multi
            ? <Track2DatasetAssemblePanel multiSource previewOnly />
            : <div id="track2-yaml-build-slot" />}

          <TrainingForm
            selection={selection}
            setSelection={setSelection}
            onSubmitted={handleSubmitted}
            detectorsVersion={detectorsVersion}
          />

          {/* The pull queue is on the Data Preparation tab now, with the
              Pull Control card whose settings it queues. */}

          <Card
            title="Training queue"
            caption={
              // Counts the running entry too. Counting only `pending` made
              // a run that STARTED look like one that disappeared: press
              // Start with two queued and the caption drops to "1 of 10",
              // with the one that began reduced to a single line below.
              data.queue.running
                ? `1 running, ${data.queue.pending.length} queued (max ${data.queue.max_size})`
                : `${data.queue.pending.length} of ${data.queue.max_size} max`
            }
            action={
              <div className="flex shrink-0 gap-2">
                {/* The label says what is happening, and every button
                    goes down while any one of them is in flight -- they
                    all act on the same queue, so a second press during
                    the first is never the next thing to do. */}
                <button
                  type="button"
                  onClick={handleQueueStart}
                  disabled={
                    !!queueBusy
                    || data.queue.started
                    || (data.queue.pending.length === 0 && !data.queue.running)
                  }
                  className="rounded-md border border-black/15 px-3 py-2 text-xs font-medium text-ink-secondary disabled:opacity-40"
                >
                  {queueBusy === "start" ? "Starting queue..." : "Start Queue"}
                </button>
                <button
                  type="button"
                  onClick={handleQueueStop}
                  disabled={!!queueBusy || !data.queue.started}
                  className="rounded-md bg-red-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-40"
                >
                  {queueBusy === "stop" ? "Stopping queue..." : "Stop Queue"}
                </button>
                <button
                  type="button"
                  onClick={() => { setAlsoForeign(false); setStopOpen(true); }}
                  // custom_run_id matters: during a custom run
                  // training_progress reports the last PRODUCTION
                  // execution, which is a finished one -- keying off it
                  // alone greys this out exactly when the single-GPU
                  // guard makes stopping most necessary.
                  // Reachable when a FOREIGN job holds the GPU as well.
                  // The single-GPU guard refuses to start anything while
                  // one does, and this was greyed out in exactly that
                  // state -- the app could name the job in the status
                  // bar and offered no way to end it.
                  disabled={
                    !!queueBusy || (
                      !data.queue.running &&
                      data.training_progress?.status !== "Executing" &&
                      !data.custom_run_id &&
                      !data.gpu_jobs.some((j) => !j.own_pipeline)
                    )
                  }
                  className="rounded-md bg-red-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-40"
                >
                  {queueBusy === "stop-training" ? "Stopping training..." : "Stop Training"}
                </button>
              </div>
            }
          >
            {/* The answer, where the question was asked. The buttons are
                in this card's header; the page's own banners are nearly
                three hundred lines above, off the top of the screen. */}
            {queueNote && (
              <p className={`mb-2 text-xs ${queueNote.bad ? "text-critical" : "text-ink-secondary"}`}>
                {queueNote.text}
              </p>
            )}
            {/* Run status lives here rather than in its own card. The old
                "Latest training run" panel read from training_progress,
                which covers the PRODUCTION pipeline only -- so during a
                custom run it showed a finished production execution and
                said "Stopped" while training was underway. */}
            {/* Suppressed while the queue has its own running block, which
                already shows this run's status and progress -- otherwise
                the same run is reported twice, a few lines apart. */}
            {data.pending_message ? (
              <p className="mb-2 text-sm font-medium text-ink">{data.pending_message}</p>
            ) : data.queue.running ? null : data.custom_run_id && !data.training_progress ? (
              <div className="mb-2 flex flex-col gap-1">
                <Badge tone="pending" label="Executing" />
                <p className="tabular text-xs text-ink-secondary">
                  Custom-class run {data.custom_run_id}
                </p>
              </div>
            ) : data.training_progress?.status === "Executing" ? (
              <div className="mb-2 flex flex-col gap-2">
                <Badge
                  tone="pending"
                  label={data.training_progress.status}
                />
                <Track2TrainingRunProgress progress={data.training_progress} />
              </div>
            ) : null}
            {data.queue.last_error && (
              <p className="mb-2 text-xs text-critical">{data.queue.last_error}</p>
            )}
            <div className="mb-3">
              <StatTile label="Completed" value={data.queue.completed_count} />
            </div>
            {!data.queue.running && data.queue.pending.length === 0 ? (
              <p className="text-sm text-ink-muted">Queue is empty.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {/* Running first, then the queue in order. One list, so a
                    run that starts keeps its place instead of disappearing
                    from the pending count. */}
                {data.queue.running && (
                  <QueueEntryBlock
                    key={data.queue.running.id}
                    entry={data.queue.running}
                    state="running"
                    progress={data.training_progress}
                  />
                )}
                {data.queue.pending.map((e, index) => (
                  <QueueEntryBlock
                    key={e.id}
                    entry={e}
                    state="queued"
                    position={index + 1}
                    onRemove={() => handleQueueRemove(e.id)}
                  />
                ))}
              </div>
            )}
            <Track2TrainingHistory reloadKey={historyKey} />
          </Card>

          {/* Every queued run trains on a frozen dataset version
              (2026-10-26) -- listed here, readable, and queueable again. */}
          <Track2DatasetVersionsPanel />

          {/* Destructive, and it can now reach outside this app -- so it
              names what goes, says what does not, and carries the
              harsher choice itself rather than sending the operator to
              find a second control. */}
          <ConfirmDialog
            open={stopOpen}
            title="Stop training?"
            destructive
            busy={queueBusy === "stop-training"}
            confirmLabel={
              alsoForeign
                ? `Stop ${foreignGpuJobs.length + (stoppableOwnRun ? 1 : 0)} job${
                    foreignGpuJobs.length + (stoppableOwnRun ? 1 : 0) === 1 ? "" : "s"
                  }`
                : "Stop training"
            }
            cancelLabel="Cancel"
            body={
              <>
                {stoppableOwnRun ? (
                  <p>
                    The run started here stops now. SageMaker cannot resume a
                    stopped execution &mdash; a fresh run has to be started
                    instead.
                  </p>
                ) : (
                  <p>No run started here is executing.</p>
                )}
                {foreignGpuJobs.length > 0 && (
                  <div className="mt-3 rounded-md border border-hairline bg-page p-2.5">
                    <p className="text-xs text-ink-secondary">
                      {foreignGpuJobs.length === 1 ? "This job is" : "These jobs are"}{" "}
                      on the GPU and {foreignGpuJobs.length === 1 ? "was" : "were"} not
                      started here:
                    </p>
                    <ul className="tabular mt-1 text-xs text-ink">
                      {foreignGpuJobs.map((j) => (
                        <li key={j.name}>
                          {j.name}
                          {j.spot && <span className="text-ink-muted"> &middot; spot</span>}
                          {!j.started && (
                            <span className="text-ink-muted">
                              {" "}&middot; no instance allocated yet
                            </span>
                          )}
                        </li>
                      ))}
                    </ul>
                    <label className="mt-2 inline-flex items-start gap-1.5 text-xs text-critical">
                      <input
                        type="checkbox"
                        className="mt-0.5 h-3.5 w-3.5"
                        checked={alsoForeign}
                        onChange={(e) => setAlsoForeign(e.target.checked)}
                      />
                      Stop {foreignGpuJobs.length === 1 ? "it" : "them"} too &mdash;{" "}
                      {foreignGpuJobs.length === 1 ? "it belongs" : "they belong"} to
                      another system.
                    </label>
                  </div>
                )}
                <p className="mt-3 text-xs text-ink-muted">
                  <strong>Nothing else is touched.</strong> No dataset, no
                  snapshot and no registered model changes, and the training
                  queue keeps its entries &mdash; stopping a run is not the
                  same as stopping the queue.
                </p>
              </>
            }
            onConfirm={() => void handleStopTraining(alsoForeign)}
            onCancel={() => { setStopOpen(false); setAlsoForeign(false); }}
          />

        </div>
      )}
    </div>
  );
}
