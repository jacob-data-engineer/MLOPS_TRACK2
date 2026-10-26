import { AlertCircle, Cpu, Loader2 } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { formatDayTime } from "../lib/format";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { ConfirmDialog } from "../components/ConfirmDialog";
import { ProgressBar } from "../components/ProgressBar";
import { StatTile } from "../components/StatTile";
import { Track2AutoTrainingPanel } from "../components/Track2AutoTrainingPanel";
import { Track2DatasetAssemblePanel } from "../components/Track2DatasetAssemblePanel";
import { Track2GoldenSetPanel } from "../components/Track2GoldenSetPanel";
import { SwitchedOff } from "../components/SwitchedOff";
import { Track2DetectorsPanel } from "../components/Track2DetectorsPanel";
import { Track2BaseModelPicker } from "../components/Track2BaseModelPicker";
import { Track2ImportCheckpointPanel } from "../components/Track2ImportCheckpointPanel";
import { Track2PullControlPanel } from "../components/Track2PullControlPanel";
import { Track2PullQueuePanel } from "../components/Track2PullQueuePanel";
import { Track2PinnedBar } from "../components/Track2PinnedBar";
import { Track2TrainingHistory } from "../components/Track2TrainingAuditPanel";
import {
  addTrack2Queue,
  runTrack2Retrain,
  ApiError,
  listTrack2Snapshots,
  getTrack2LastPullDataset,
  restoreTrack2Snapshot,
  deleteTrack2Snapshot,
  trainOnTrack2Snapshot,
  type Track2DatasetSnapshot,
  getTrack2Metrics,
  getTrack2QueueAfterPull,
  saveTrack2QueueAfterPull,
  getTrack2Retrain,
  saveTrack2Retrain,
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
function sourceLabel(folders: string[] | undefined, fallback: string): string {
  return folders && folders.length > 0 ? folders.join(", ") : fallback;
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
          <span
            className="col-span-2 truncate sm:col-span-3"
            title={sourceLabel(entry.dataset_source_folders, entry.dataset_folder)}
          >
            Training images: {sourceLabel(entry.dataset_source_folders, entry.dataset_folder)}
          </span>
          {/* Never "same as training" any more. The evaluator runs
              model.val(split="val"), so what it measures on is whichever
              split the folder's data.yaml declares as val -- for an
              assembled dataset that is a held-out images/val, and calling
              it the training set was simply wrong. */}
          <span
            className="col-span-2 truncate sm:col-span-3"
            title={sourceLabel(
              entry.eval_source_folders ?? entry.dataset_source_folders,
              entry.eval_dataset_folder || entry.dataset_folder,
            )}
          >
            Evaluation images:{" "}
            {sourceLabel(
              entry.eval_source_folders ?? entry.dataset_source_folders,
              entry.eval_dataset_folder || entry.dataset_folder,
            )}
            {" "}
            <span className="text-ink-muted">(the val split its data.yaml declares)</span>
          </span>
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
              {entry.snapshot_pinned
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
        {running && progress?.epoch_progress && (
          <div className="mt-2">
            <p className="tabular mb-1 text-xs text-ink-muted">
              Epoch {progress.epoch_progress.current} / {progress.epoch_progress.total}
              {/* The number actually being asked for -- "how much longer"
                  is not a subtraction the reader should have to do. Only
                  while training: once evaluation starts the epoch count
                  is finished and "0 left" would read as a stall. */}
              {runPhase(progress) === "Training" &&
                progress.epoch_progress.total > progress.epoch_progress.current &&
                ` -- ${progress.epoch_progress.total - progress.epoch_progress.current} to go`}
            </p>
            <ProgressBar processed={progress.epoch_progress.current} total={progress.epoch_progress.total} />
          </div>
        )}
        {running && progress?.download_progress && (
          <div className="mt-2">
            <p className="tabular mb-1 text-xs text-ink-muted">
              Downloading {progress.download_progress.current} / {progress.download_progress.total}
            </p>
            <ProgressBar processed={progress.download_progress.current} total={progress.download_progress.total} />
          </div>
        )}
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
  hyperparams: Record<string, string>;
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
/** The radio's own words, so the scheduling tick and the "Training
 * data" group above cannot describe the source differently. */
function sourceLabelFor(source: "assembled" | "snapshot" | "pull"): string {
  if (source === "snapshot") return "the frozen snapshot";
  if (source === "pull") return "the snapshot generated during auto pull";
  return "the assembled dataset";
}

/** Named for what it used to be -- the viewer's local clock -- and kept
 * because eight call sites read it, but it renders Singapore time now.
 *
 * Every value passed to it is an instant the SERVER produced: when a
 * snapshot was frozen, when a dataset was built, when the last pull ran,
 * when the schedule next fires. In the viewer's zone, two people reading
 * one page side by side could see different times for the same event, and
 * the next-run line could disagree with the schedule that computed it. The
 * zone is named in the output so the reading is checkable. */
function localDate(iso: string): string {
  return iso ? formatDayTime(iso) : "";
}

function TrainingForm({
  selection,
  setSelection,
  onSubmitted,
  detectorsVersion,
  assembledVersion,
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
  /** Bumped when the Assemble card writes a data.yaml.
   *
   * `assembled` is read from the config, and that read happened once on
   * mount -- so writing a yaml left "Use the assembled dataset" greyed
   * out, saying "Nothing assembled yet" beside a file that existed,
   * until the page was reloaded. Nothing on screen said a reload was
   * what it wanted. */
  assembledVersion: number;
}) {
  const { epochs, imgSize, batch, seed, patience, hyperparams, datasetFolder, evalFolder, detectorSlug, architecture, basePackageArn, notes } = selection;
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [datasetClasses, setDatasetClasses] = useState<Track2DatasetClasses | null>(null);
  /** The three source ticks, as the cards left them. Read here rather
   * than passed down: they live on three different cards, and this one
   * only reports them. */
  const [sources, setSources] = useState({ folders: true, golden: false, pull: false });
  // Choosing the dataset from this card, for when nothing upstream has.
  // The frozen snapshots of the training root, offered HERE as well as
  // in Retention. Repeating a past dataset is a thing you decide at the
  // moment of starting a run, and it lived three cards away -- so doing
  // it meant scrolling up, restoring, scrolling back, and trusting that
  // the folder in this box was still the one just restored.
  // Which KIND of dataset this run uses. Two mutually exclusive answers,
  // and the controls for the one not chosen are disabled rather than
  // hidden -- a greyed button still says the option exists, where a
  // vanished one reads as a bug.
  //
  // "live" is the week folders as they stand, which is what the chain
  // builds and what a scheduled retrain trains on. "snapshot" is a
  // frozen definition: the same images every time, whatever has been
  // pulled since. They are genuinely different datasets, which is why
  // picking one has to DO something rather than only tick a box --
  // choosing live rebuilds the week-folder yaml over whatever a restore
  // left in the live slot.
  // Three ways a run gets its images, and they are different questions:
  // what this page assembled, a snapshot somebody froze, and whatever the
  // pull chain last produced. Each has its own record server-side,
  // because one slot cannot answer all three -- loading a snapshot by
  // hand would otherwise erase what the pull built.
  const [dataSource, setDataSource] =
    useState<"assembled" | "snapshot" | "pull">("assembled");
  /** What the Assemble card last wrote a data.yaml for. */
  const [assembled, setAssembled] = useState<
    { folder: string; eval_folder?: string; at?: string } | null>(null);
  /** What the pull chain last froze. Never overwritten by a manual pick. */
  const [lastPull, setLastPull] = useState<{
    folder?: string; snapshot_id?: string; files?: number;
    set_by?: string; at?: string;
  } | null>(null);
  const [snapOpen, setSnapOpen] = useState(false);
  const [snaps, setSnaps] = useState<Track2DatasetSnapshot[]>([]);
  const [snapBusy, setSnapBusy] = useState("");
  /** The snapshot a Delete was pressed on, while the dialog asks. */
  const [snapToDelete, setSnapToDelete] = useState<Track2DatasetSnapshot | null>(null);
  const [snapDeleting, setSnapDeleting] = useState(false);
  const [snapDeleteError, setSnapDeleteError] = useState("");
  const [snapNote, setSnapNote] = useState("");
  // What the last automatic run built and froze. Read so the card opens
  // pointing at it rather than saying "Nothing chosen yet" after the
  // schedule has just produced a dataset.
  const [autoDataset, setAutoDataset] = useState<{
    folder?: string; snapshot_id?: string; files?: number;
    set_by?: string; at?: string;
    snapshot_created_at?: string; snapshot_note?: string; pinned?: boolean;
  } | null>(null);
  /** Re-read which definition is live. Called after anything that moves
   * the live slot: a restore writes `default_dataset`, and without
   * reading it back the card goes on naming the snapshot the last chain
   * froze over a slot now holding a different one. */
  const refreshLiveDataset = useCallback(async () => {
    try {
      const status = await getTrack2PullControl();
      const built = status.config.default_dataset || {};
      if (built.folder) setAutoDataset(built);
      return built;
    } catch {
      return null;
    }
  }, []);
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
  const datasetNames = datasetClasses ? Object.values(datasetClasses.classes) : [];
  const detectorNames = chosenDetector ? Object.values(chosenDetector.classes) : [];
  const absentClasses =
    chosenDetector && datasetClasses
      ? detectorNames.filter((name) => !datasetNames.includes(name))
      : [];
  const willFilter =
    chosenDetector && datasetClasses && absentClasses.length === 0 &&
    JSON.stringify(chosenDetector.classes) !== JSON.stringify(datasetClasses.classes);
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
        // file -- comes back on the same side of live/frozen it was left
        // on. Without it the radio always read "live" while the live
        // slot could be holding a restored snapshot.
        // "live" is the old name for this setting, from when there were
        // two choices and the first was "the week folders as they
        // stand". A saved file still carrying it maps onto the nearest
        // of the three rather than being ignored, which would silently
        // put the card back on Assembled.
        const saved = status.config.data_source;
        const source: "assembled" | "snapshot" | "pull" =
          saved === "snapshot" || saved === "pull" ? saved : "assembled";
        setDataSource(source);

        const assembledRecord = status.config.dataset_assemble?.built ?? null;
        setAssembled(assembledRecord);
        // WHICH SOURCES ARE TICKED, read from the same three fields the
        // chain reads. The assembled folder alone stopped describing
        // this run the moment the yaml could be composed from more than
        // one place: with the golden set on, that folder is one of two
        // the run trains from, and the line under the radio said only
        // the first.
        setSources({
          folders: status.config.dataset_assemble?.use !== false,
          golden: Boolean(status.config.retention?.use_golden),
          pull: !status.config.dataset_assemble?.only,
        });
        setLastPull(status.config.last_pull_dataset ?? null);
        const built = status.config.default_dataset || {};
        if (built.folder) setAutoDataset(built);

        // THE CHOSEN SOURCE DECIDES THE FOLDER.
        //
        // This used to restore the radio from `data_source` and then set
        // the folder from `default_dataset` whatever the radio said. So
        // every reload came back with "Use the assembled dataset" ticked
        // and `retraining` -- the frozen-snapshot slot -- underneath it,
        // and the run that followed trained `retraining`. Clicking the
        // radio fixed it; reloading broke it again, and nothing on
        // screen said which of the two to believe.
        //
        // Each option now fills from its OWN record, the same mapping
        // the radio's onChange uses.
        const recordFor = (key: string) =>
          key === "assembled" ? assembledRecord
            : key === "pull" ? (status.config.last_pull_dataset ?? null)
              : (built.folder ? built : null);

        let settled = recordFor(source);
        let settledSource = source;
        // Nothing saved under the chosen source. Fall back to the frozen
        // slot AND move the radio with it, rather than showing one
        // answer and running another.
        if (!settled?.folder && built.folder) {
          settled = built;
          settledSource = "snapshot";
          setDataSource("snapshot");
        }
        // `datasetFolder` is the value at the time this effect ran: ""
        // on mount, current on any later re-run. An explicit pick made
        // in this tab wins, so a refresh cannot move the folder out from
        // under someone mid-setup.
        if (!datasetFolder && settled?.folder) {
          setSelection({
            datasetFolder: settled.folder,
            evalFolder: (settled as { eval_folder?: string }).eval_folder || settled.folder,
          });
        }

        // The config field is only written by chains that ran after it
        // existed. The route also reads the snapshot store, so pulls
        // from before it are still offered -- and it answers AFTER the
        // block above, so the pull option fills its folder here when it
        // is the one chosen.
        void getTrack2LastPullDataset()
          .then((res) => {
            if (!res.dataset?.folder) return;
            setLastPull(res.dataset);
            if (settledSource === "pull" && !datasetFolder) {
              setSelection({
                datasetFolder: res.dataset.folder,
                evalFolder: res.dataset.folder,
              });
            }
          })
          .catch(() => undefined);
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
    // until the page was reloaded. Same for the Assemble card writing a
    // yaml: `dataset_assemble.built` is what fills the "assembled"
    // option, and it is read right here.
  }, [detectorsVersion, assembledVersion]);


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

    if (!datasetFolder) {
      lines.push({
        tone: "warn",
        text: "No dataset chosen yet -- assemble one, or press \"Use for "
              + "training\" under a built data.yaml.",
      });
    } else if (!datasetClasses) {
      lines.push({ tone: "warn", text: `Could not read data.yaml in ${datasetFolder}.` });
    }

    if (!chosenDetector) {
      lines.push({
        tone: "warn",
        text: "No Model Name chosen -- the model lands in the shared open registry, "
          + "where version numbers are not per model type.",
      });
    } else if (datasetClasses) {
      if (absentClasses.length > 0) {
        blocking = true;
        lines.push({
          tone: "bad",
          text: `The dataset has no labels for ${absentClasses.join(", ")}, so a `
            + `${chosenDetector.label} model cannot be trained from it. It declares `
            + `${datasetNames.join(", ") || "nothing"}.`,
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
    setSelection({
      epochs: 50, imgSize: 640, batch: 16, seed: 42, patience: 0,
      hyperparams: {},
      datasetFolder: "", evalFolder: "",
      detectorSlug: "", architecture: DEFAULT_ARCHITECTURE,
      basePackageArn: "", notes: "",
    });
    setError(null);
  }


  const params = (): Track2TrainingParams => ({
    generation,
    size,
    epochs,
    img_size: imgSize,
    batch,
    seed,
    patience,
    // Overrides only -- see the request type. Empty means every
    // Ultralytics default, which is how a run behaved before this.
    hyperparams,
    dataset_folder: datasetFolder,
    eval_dataset_folder: evalFolder,
    // A custom run picks EXACTLY one source and the backend rejects two,
    // so the folder is never sent alongside a key -- custom runs select a
    // file now, not a folder. The production run keeps its original
    // browse-or-latest-approved behaviour untouched.
    // Exactly one source, on either pipeline. `registry` on the production
    // path means "leave it to the coordinator's production pointer", which
    // is what sending nothing does -- so it deliberately sends nothing.
    base_model_folder: "",
    // EXACTLY ONE base model source, which the backend enforces. With
    // scratch gone there is only one to send.
    base_model_package_arn: basePackageArn,
    // Retired: starting weights are always a registered version now. Still
    // sent, and still empty, because the backend's "exactly one source"
    // check reads it and the queue replays entries that carry it.
    base_model_key: "",
    // Always false from this page now. Still SENT, because the backend's
    // "exactly one base model source" check reads it and the queue
    // replays entries that carry it.
    base_model_scratch: false,
    architecture,
    // Sent on every submission, not just Start Training: Add to Queue used
    // to drop it, so a queued custom run silently ran as a production one.
    // Always: this card trains a detector's classes through the open
    // pipeline. The production person/helmet path is not reachable
    // from here any more, by design.
    custom_classes: true,
    // Only meaningful on the custom path -- the production pipeline's
    // taxonomy and registry group are fixed.
    detector_slug: detectorSlug,
    notes,
  });

  // Starting a run directly was removed with the button that did it: both
  // cards queue, and the queue is what gets started. The backend still
  // exposes both start endpoints, so this is a UI decision, not a
  // capability that has gone.
  /** A run built from MORE than the assembled folder.
   *
   * The queue entry this card writes names one folder, and one folder
   * cannot carry a golden set that lives under another root. Only the
   * chain composes that yaml -- builds the golden set, names every
   * source, fetches every prefix -- so when the radio above says the
   * run is composed, this button has to run the chain or the label
   * lies: it would say "plus the golden set" and train without it.
   *
   * The chain trains with the SAME shape: step 1 writes
   * `default_training`, `settings()` reads it, and `onTraining` copies
   * it into this form. There is no second copy to disagree. */
  const composed = dataSource === "assembled" && (sources.golden || sources.pull);

  async function handleAddToQueue() {
    setBusy(true);
    setError(null);
    try {
      if (composed) {
        // `queue`, not `start`: the entry lands in the queue card below
        // with every other run, and the queue releases it -- the same
        // hand-over the schedule uses, so pressing this says something
        // about whether the schedule works.
        const got = await runTrack2Retrain("queue");
        const failed = got.steps.find((step) => !step.ok);
        if (!got.ok || failed) {
          // The step that stopped it, in its own words. "Add to queue
          // failed" would hide the one sentence that says what to fix.
          setError(`Stopped at "${failed?.step ?? "?"}": ${failed?.detail ?? "no detail"}`);
          return;
        }
        onSubmitted();
        return;
      }
      await addTrack2Queue(params());
      onSubmitted();
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
            Starts the run. Everything about <strong>which model</strong> was
            decided in step 1; everything about <strong>which data</strong>{" "}
            came from steps 2 to 5. This card only sets how long it trains for
            and sends it.
          </p>
          <p className="mt-2">
            The dataset arrives here by itself. Pressing <em>Write it</em> on{" "}
            <em>Assemble a dataset</em> fills it in, and so does{" "}
            <em>Use for training</em> under a data.yaml in step 5, or{" "}
            <em>Train on this</em> on a frozen snapshot. Until one of those
            happens this says &ldquo;Nothing chosen yet&rdquo;, and you can
            always pick a folder here instead.
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
      <fieldset>

      <div className="mb-3 rounded-md border border-black/15 bg-page p-2.5 text-xs">
        <p className="tabular text-ink-muted">
          Training data{" "}
          <span className="text-ink-muted/70">
            (from <em>Assemble a dataset</em> above, from Auto Training,
            &ldquo;Use for training&rdquo; on a data.yaml, or pick one here)
          </span>
        </p>
        {/* The choice, before anything that depends on it. Radios rather
            than two checkboxes: they are mutually exclusive, and a pair
            of checkboxes that cannot both be ticked is a radio group
            wearing the wrong control. */}
        {/* Three sources, each with its own record. A greyed one says
            what would fill it rather than disappearing: an option you
            cannot see is one you cannot plan for. */}
        <div className="mt-1 flex flex-col gap-1">
          {([
            {
              key: "assembled" as const,
              label: sources.golden || sources.pull
                ? "Use the assembled dataset + the other ticked sources"
                : "Use the assembled dataset",
              why: [
                sources.folders ? "the folders you picked in Assemble a dataset" : "",
                sources.golden ? "the golden set / edge cases" : "",
                sources.pull ? "Pull Control's week folders" : "",
              ].filter(Boolean).join(" + ")
                ? `Built from ${[
                    sources.folders ? "the folders you picked in Assemble a dataset" : "",
                    sources.golden ? "the golden set / edge cases" : "",
                    sources.pull ? "Pull Control's week folders" : "",
                  ].filter(Boolean).join(" + ")}.`
                  + (sources.golden
                    ? " The golden set's val side is the held-out evaluation set."
                    : "")
                : "No source is ticked. Tick at least one on the cards above.",
              folder: assembled?.folder,
              detail: assembled?.folder
                ? `${assembled.folder}${assembled.at ? ` · built ${localDate(assembled.at)}` : ""}`
                  + (sources.golden || sources.pull
                    ? " · plus the other ticked sources, composed at run time"
                    : "")
                : "",
              missing: "Nothing assembled yet. Pick folders in Assemble a dataset above and press Write data.yaml.",
            },
            {
              key: "snapshot" as const,
              label: "Use a frozen snapshot",
              why: "The same images every time, whatever has been pulled since.",
              folder: autoDataset?.folder,
              detail: "",
              missing: "",
            },
            {
              key: "pull" as const,
              label: "Use snapshot generated during auto pull",
              why: "The snapshot the chain froze when the pull last ran on its own.",
              folder: lastPull?.folder,
              detail: lastPull?.folder
                ? `${lastPull.folder} · ${lastPull.snapshot_id ?? "snapshot"}`
                  + (lastPull.files ? ` · ${lastPull.files.toLocaleString()} images` : "")
                  + (lastPull.at ? ` · ${localDate(lastPull.at)}` : "")
                : "",
              missing: "No automatic pull has frozen one yet. Switch on “Run after every pull” in Auto Training, and run a pull.",
            },
          ]).map((option) => {
            // The snapshot option is always offered: picking one is how
            // it gets filled, so greying it until it has a value would
            // be a control that can never be reached.
            const ready = option.key === "snapshot" || Boolean(option.folder);
            const chosen = dataSource === option.key;
            // Keyed here as well as on the wrapper below: whichever of
            // the two is returned IS the list item, so both need one.
            const row = (
              <label key={option.key}
                     className={`flex items-start gap-1.5 ${ready ? "" : "opacity-55"}`}>
                <input
                  type="radio"
                  name="track2-data-source"
                  className="mt-0.5 shrink-0"
                  checked={chosen}
                  disabled={!ready || !!snapBusy}
                  onChange={() => {
                    setDataSource(option.key);
                    setSnapOpen(false);
                    setSnapNote("");
                    void saveTrack2PullConfig({ data_source: option.key })
                      .catch(() => undefined);
                    // Point the run at that source's own folder, so the
                    // radio and the line under it cannot disagree.
                    if (option.key === "assembled" && assembled?.folder) {
                      setSelection({
                        datasetFolder: assembled.folder,
                        evalFolder: assembled.eval_folder || assembled.folder,
                      });
                    } else if (option.key === "pull" && lastPull?.folder) {
                      setSelection({
                        datasetFolder: lastPull.folder,
                        evalFolder: lastPull.folder,
                      });
                      // RESTORED, not merely pointed at.
                      //
                      // A snapshot's folder is the live root -- the same
                      // `retraining` the week folders fill. Pointing at
                      // it without putting the frozen file lists back
                      // would train on whatever has been pulled since,
                      // under a label promising the images from the
                      // pull. The restore writes those lists back, which
                      // is what makes the two agree.
                      if (lastPull.snapshot_id) {
                        setSnapBusy("Restoring...");
                        void restoreTrack2Snapshot(lastPull.snapshot_id)
                          .then(async (got) => {
                            await refreshLiveDataset();
                            setSnapNote(
                              `Put back ${got.restored.length} file lists from `
                              + `${lastPull.snapshot_id}.`);
                          })
                          .catch((err) => setSnapNote(
                            err instanceof ApiError ? err.detail
                              : "Could not put that snapshot back."))
                          .finally(() => setSnapBusy(""));
                      }
                    }
                  }}
                />
                <span>
                  <span className={chosen ? "font-medium text-ink" : "text-ink-muted"}>
                    {option.label}
                  </span>
                  <span className="block text-ink-muted/80">{option.why}</span>
                  {ready && option.detail && (
                    <span className="tabular block text-ink-secondary">{option.detail}</span>
                  )}
                  {!ready && (
                    <span className="block text-ink-muted">{option.missing}</span>
                  )}
                </span>
              </label>
            );
            // The picker belongs ON this row, not three lines below the
            // last option: it only acts on THIS choice, and sitting
            // under all three it read as belonging to whichever was
            // ticked. A sibling of the label rather than a child of it
            // -- a button inside a label is also the label's control,
            // so clicking it would tick the radio as well.
            if (option.key !== "snapshot") return row;
            return (
              <div key={option.key} className="flex items-start gap-2">
                {row}
                <button
                  type="button"
                  onClick={() => {
                    const next = !snapOpen;
                    setSnapOpen(next);
                    setSnapNote("");
                    if (!next) return;
                    const root = (autoDataset?.folder || datasetFolder || "").trim();
                    setSnapBusy("Loading...");
                    listTrack2Snapshots(root, 25)
                      .then((res) => setSnaps(res.snapshots ?? []))
                      .catch(() => setSnapNote("Could not read the snapshots."))
                      .finally(() => setSnapBusy(""));
                  }}
                  disabled={dataSource !== "snapshot"}
                  title={dataSource !== "snapshot"
                    ? "Switch to “Use a frozen snapshot” to pick one."
                    : undefined}
                  className="mt-0.5 shrink-0 rounded-md border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40"
                >
                  {snapOpen ? "Hide snapshots" : "Pick a snapshot"}
                </button>
              </div>
            );
          })}
        </div>

        <p className="tabular mt-0.5 text-ink">
          {datasetFolder || "Nothing chosen yet."}
        </p>
        {evalFolder && (
          <p className="tabular mt-0.5 text-ink-secondary">Evaluation: {evalFolder}</p>
        )}

        {/* Pickable HERE, not only upstream. The dataset could previously
            be set in exactly one place -- "Use for training" in
            Retention, which appears only once a data.yaml has been built
            -- so anyone who had not built one reached this card, read
            "no dataset chosen", and had nothing on screen to act on. The
            run was unreachable from the card that starts it. */}
        {/* Only under the radio it describes. This records
            `default_dataset` -- the FROZEN slot -- and it was rendered
            whenever that slot had anything in it, so "Built by the
            snapshot run ... frozen as 532f55d72b51a774 (451 files)" sat
            under a ticked "Use the assembled dataset" describing a
            different dataset entirely. */}
        {dataSource === "snapshot" && autoDataset?.folder && (
          <p className="tabular mt-1 text-xs text-ink-secondary">
            {/* Two dates, and they are not the same thing. `at` is when
                this definition was PUT BACK; `snapshot_created_at` is
                when it was frozen. Showing only the first made a
                snapshot taken on 30 September read as "at 2026-10-06
                06:59" -- the one fact that tells one frozen dataset from
                another, replaced by the time you clicked Load. */}
            {/* Only while the frozen side is chosen. The record can say
                "snapshot" for a moment after switching to live -- the
                rebuild clears it, but the card renders first -- and a
                line reading "Loaded snapshot ..." under a ticked "Use
                live data" is the card contradicting itself. The radio is
                what the person just chose; it wins. */}
            {autoDataset.set_by === "snapshot" && dataSource === "snapshot" ? (
              <>
                {/* The date it was FROZEN, and nothing about when it was
                    put back. "Built by the snapshot run at <now>" was
                    the restore time wearing the creation time's place --
                    it changed every time you pressed Load, so the one
                    fact that tells one frozen dataset from another was
                    the one fact it never showed. */}
                Loaded snapshot{" "}
                <span className="font-medium">{autoDataset.snapshot_id}</span>
                {autoDataset.snapshot_created_at && (
                  <> &middot; frozen {localDate(autoDataset.snapshot_created_at)}</>
                )}
                {autoDataset.files
                  ? ` · ${autoDataset.files.toLocaleString()} images`
                  : ""}
                {autoDataset.pinned && " pinned by name from its manifest"}
                {autoDataset.snapshot_note && (
                  <span className="block text-ink-muted">{autoDataset.snapshot_note}</span>
                )}
              </>
            ) : (
              <>
                Built by the {autoDataset.set_by ?? "automatic"} run at{" "}
                {localDate(autoDataset.at ?? "")}
                {autoDataset.snapshot_id && (
                  <>
                    {" "}&middot; frozen as{" "}
                    <span className="font-medium">{autoDataset.snapshot_id}</span>
                    {autoDataset.files ? ` (${autoDataset.files.toLocaleString()} files)` : ""}
                  </>
                )}
              </>
            )}
            {datasetFolder && datasetFolder !== autoDataset.folder && (
              <span className="block text-critical">
                You are training {datasetFolder}, not the folder the schedule
                built.
              </span>
            )}
          </p>
        )}
        {snapNote && (
          <p className={`mt-1 text-xs ${
            /could not|stopped at/i.test(snapNote) ? "text-critical" : "text-ink-secondary"
          }`}>
            {snapNote}
          </p>
        )}

        {snapOpen && (
          <div className="mt-2 rounded-sm border border-hairline">
            <p className="border-b border-hairline px-2 py-1.5 text-xs text-ink-muted">
              Every frozen definition of this root, newest first. <strong>Load</strong>{" "}
              puts one back and points this form at it; <strong>Run</strong> does that
              and queues the run.
            </p>
            {snapBusy && <p className="px-2 py-1.5 text-xs text-ink-muted">{snapBusy}</p>}
            {!snapBusy && snaps.length === 0 && (
              <p className="px-2 py-1.5 text-xs text-ink-muted">
                No snapshots yet. The automatic chain freezes one on every run.
              </p>
            )}
            <div className="max-h-56 overflow-y-auto">
              {snaps.map((snap) => (
                <div key={snap.snapshot_id}
                     className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-hairline px-2 py-1.5 last:border-b-0">
                  <span className="tabular text-xs font-medium text-ink">
                    {snap.snapshot_id}
                  </span>
                  <span className="tabular text-xs text-ink-muted">
                    {snap.created_at ? localDate(snap.created_at) : ""}
                    {" · "}
                    {snap.file_count.toLocaleString()} files
                  </span>
                  {snap.note && (
                    <span className="text-xs text-ink-secondary">{snap.note}</span>
                  )}
                  <span className="ml-auto flex gap-3">
                    <button type="button" disabled={!!snapBusy}
                      onClick={async () => {
                        setSnapBusy("Loading..."); setSnapNote("");
                        try {
                          const got = await restoreTrack2Snapshot(snap.snapshot_id);
                          setSelection({ datasetFolder: got.root, evalFolder: got.root });
                          // Read back WHICH definition is live. The
                          // folder name does not change on a restore --
                          // there is one live slot per root -- so
                          // without this the box looked identical
                          // afterwards and loading appeared to do
                          // nothing, while still naming the snapshot the
                          // last chain froze.
                          await refreshLiveDataset();
                          const frozen = got.created_at
                            ? ` (frozen ${localDate(got.created_at)})` : "";
                          setSnapNote(got.pinned_from_manifest
                            ? `Loaded ${got.snapshot_id}${frozen} -- `
                              + `${got.selected.toLocaleString()} images pinned by `
                              + `name from its manifest.`
                            : `Loaded ${got.snapshot_id}${frozen}.`);
                          setSnapOpen(false);
                        } catch (err) {
                          setSnapNote(err instanceof ApiError ? err.detail : "Could not load it.");
                        } finally { setSnapBusy(""); }
                      }}
                      className="text-xs font-medium text-ink-secondary underline disabled:opacity-40">
                      Load
                    </button>
                    <button type="button" disabled={!!snapBusy}
                      onClick={async () => {
                        setSnapBusy("Starting..."); setSnapNote("");
                        try {
                          const got = await trainOnTrack2Snapshot(snap.snapshot_id);
                          setSelection({ datasetFolder: got.restore.root,
                                         evalFolder: got.restore.root });
                          await refreshLiveDataset();
                          const failed = (got.run.steps || []).find((st) => !st.ok);
                          setSnapNote(failed
                            ? `Stopped at ${failed.step}: ${failed.detail}`
                            : `Queued a run on ${got.restore.snapshot_id}`
                              + (got.restore.created_at
                                ? ` (frozen ${localDate(got.restore.created_at)})` : "")
                              + `. It starts when the GPU is free.`);
                          if (!failed) setSnapOpen(false);
                        } catch (err) {
                          setSnapNote(err instanceof ApiError ? err.detail : "Could not start it.");
                        } finally { setSnapBusy(""); }
                      }}
                      className="text-xs font-medium text-accent underline disabled:opacity-40">
                      Run
                    </button>
                    <button type="button" disabled={!!snapBusy}
                      onClick={() => { setSnapDeleteError(""); setSnapToDelete(snap); }}
                      className="rounded bg-red-600 px-2 py-0.5 text-xs font-medium text-white hover:bg-red-700 disabled:opacity-40">
                      Delete
                    </button>
                  </span>
                </div>
              ))}
              {/* Destructive, so a dialog that says what goes and what
                  stays; Cancel is the default. */}
              <ConfirmDialog
                open={!!snapToDelete}
                title={`Delete snapshot ${snapToDelete?.snapshot_id ?? ""}?`}
                destructive
                busy={snapDeleting}
                confirmLabel="Delete it"
                cancelLabel="Cancel"
                body={
                  <>
                    <p>
                      Frozen {snapToDelete?.created_at ? localDate(snapToDelete.created_at) : ""}
                      {" · "}{snapToDelete?.file_count.toLocaleString()} files.
                      Its record, its frozen <span className="tabular">data.yaml</span> and
                      its file list are deleted. <strong>The images and labels are not
                      touched</strong>, and no trained model is affected.
                    </p>
                    <p className="mt-2">
                      What is lost is the record of exactly which files this selection
                      held, so a run trained from it can no longer be repeated from it.
                      This cannot be undone.
                    </p>
                    {snapDeleteError && (
                      <p className="mt-2 text-critical">{snapDeleteError}</p>
                    )}
                  </>
                }
                onConfirm={async () => {
                  if (!snapToDelete) return;
                  setSnapDeleting(true);
                  setSnapDeleteError("");
                  try {
                    await deleteTrack2Snapshot(snapToDelete.snapshot_id);
                    setSnaps((list) => list.filter((x) => x.snapshot_id !== snapToDelete.snapshot_id));
                    setSnapNote(`Deleted snapshot ${snapToDelete.snapshot_id}. The images are untouched.`);
                    setSnapToDelete(null);
                  } catch (err) {
                    setSnapDeleteError(err instanceof ApiError ? err.detail : "Could not delete it.");
                  } finally {
                    setSnapDeleting(false);
                  }
                }}
                onCancel={() => setSnapToDelete(null)}
              />
            </div>
          </div>
        )}
      </div>

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

      {/* Queue only. Starting is done once, from the card below, so two
          cards cannot each kick off a run against the single GPU.

          EXCEPT for a composed run, which runs the chain: it builds the
          golden set, composes the yaml, freezes a snapshot, queues the
          entry and releases the queue -- the same hand-over the
          schedule uses. Said here so the label change is not a
          surprise. */}
      {composed && (
        <p className="mt-3 text-xs text-ink-muted">
          This run is composed from more than one source, so the button
          runs the full chain: build the golden set &rarr; compose the
          data.yaml &rarr; freeze a snapshot &rarr; queue it &rarr; start
          it. Exactly what the schedule does when it fires, done now.
        </p>
      )}
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          onClick={handleAddToQueue}
          disabled={busy || !datasetFolder || baseSelectionIncomplete || analysis.blocking}
          className="rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy
            ? (composed ? "Building and queueing..." : "Working...")
            : (composed ? "Build, Queue and Start" : "Add to Queue")}
        </button>
        <button
          type="button"
          onClick={handleStartAllOver}
          disabled={busy}
          className="rounded-md border border-black/15 px-3 py-2 text-sm font-medium text-ink-secondary disabled:opacity-40"
        >
          Start All Over
        </button>
      </div>
      </fieldset>

      {/* Queue a run whenever a pull lands, and stop there. Starting one
          outright commits GPU hours with nobody watching; doing nothing
          means the rebuild is the step that gets forgotten. This does
          the rebuild, the freeze and the queueing, and leaves the
          expensive irreversible part a human decision. */}
      <div className="mt-3 rounded-md border border-black/15 p-3">
        <label className="flex items-start gap-2 text-sm">
          <input
            type="checkbox"
            className="mt-0.5 shrink-0"
            disabled={afterPullBusy || !afterPull}
            checked={!!afterPull?.settings.enabled}
            onChange={async (e) => {
              const on = e.target.checked;
              setAfterPullBusy(true);
              try {
                // BOTH, to the same value. Which mechanism is in charge
                // depends on "Use these folders only", and a tick that
                // armed one of them would be a switch that does nothing
                // in half the configurations -- silently, because the
                // other one is what the schedule actually reads.
                setAfterPull(await saveTrack2QueueAfterPull({ enabled: on }));
                setRetrain(await saveTrack2Retrain({ enabled: on }));
              } catch {
                // Left as it was; the next click tries again.
              } finally {
                setAfterPullBusy(false);
              }
            }}
          />
          <span>
            <span className="font-medium text-ink">
              Enabled for scheduling
            </span>
            <span className="mt-0.5 block text-xs text-ink-muted">
              {/* It DOES start it. `run_chain(submit="queue")` calls
                  `queue_start()`, and has since the chain was written --
                  so the old "It does NOT start it" was describing
                  something the code has never done. A schedule that
                  stopped at a queued entry would produce a to-do item,
                  not a model. */}
              Each scheduled pull builds the training data from{" "}
              <strong>{sourceLabelFor(dataSource)}</strong> above, queues one
              run and starts it &mdash; through to a trained model, with
              nobody watching.
            </span>
            <span className="mt-0.5 block text-xs text-ink-muted">
              {dataSource === "assembled"
                ? "The folders on the Assemble card are rebuilt into a data.yaml each time, then frozen."
                : "That snapshot is trained exactly as it was frozen -- nothing is rebuilt or re-frozen, so every run reads the same images."}
            </span>
            {/* The CADENCE is not set here. It lives in one place --
                the PULL, BUILD, QUEUE, TRAIN strip at the top, which
                mirrors its interval and time onto this clock when the
                training clock is the one in charge. Two interval fields
                on one page, one of them deciding nothing, is how
                somebody sets the wrong one.

                What stays is the ANSWER: when it will next run, read
                from the same settings the strip just wrote. */}
            {retrain && retrain.follows_pull === false && (
              <span className="mt-2 block text-xs text-ink-muted">
                {retrain.settings.enabled && retrain.next_runs?.length
                  ? `Next run ${localDate(retrain.next_runs[0])} · set it in `
                    + `PULL, BUILD, QUEUE, TRAIN at the top.`
                  : "Not armed -- set “On a schedule” in PULL, BUILD, "
                    + "QUEUE, TRAIN at the top."}
                {retrain.problems?.some((p) => p.level === "blocking") && (
                  <span className="mt-1 block text-critical">
                    {retrain.problems.filter((p) => p.level === "blocking")[0].text}
                  </span>
                )}
              </span>
            )}
          </span>
        </label>

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
  /** Bumped when the Assemble card writes a data.yaml, so step 6 re-reads
   * `dataset_assemble.built` and can offer it. */
  const [assembledVersion, setAssembledVersion] = useState(0);
  // Whether the dataset is picked by hand on the Assemble card, which
  // switches Pull Control, Auto Training and Retention off.
  //
  // Declared up here with the rest of the state, not beside the cards
  // that read it near the bottom of this file. A `const` that reads a
  // `useState` declared BELOW it throws on every render and unmounts the
  // whole tree -- the page went white that way once already, and `tsc`
  // does not catch it when the read is inside a callback.
  //
  // Not persisted: it is off on every load, so a reload always leaves
  // every card usable rather than leaving someone with three greyed-out
  // panels and nothing on screen saying which tick did it.
  const [pickFoldersByHand, setPickFoldersByHand] = useState(false);
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
        <Track2PinnedBar />
      </div>

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
              modelSlug={modelChoice.slug}
              onAssembled={(folder) => {
                setSelection({ datasetFolder: folder, evalFolder: folder });
                // So step 6 re-reads what was just written, rather than
                // staying on the answer it read when the page loaded.
                setAssembledVersion((n) => n + 1);
              }}
            />
          )}

          {/* Pick the data by hand, instead of collecting it week by week.
              ABOVE the golden set and everything below it, because it is
              the first question -- where does this run's data come from --
              and the three cards underneath are the other answer to it.
              Ticking "Use these folders only" switches those three off
              rather than letting both decide. */}
          {multi && (
            <Track2DatasetAssemblePanel
              multiSource
              // Same one choice, cascaded: the Model Name decides the
              // registry the run versions into AND the class names the
              // yaml declares, and those must be one list.
              modelSlug={modelChoice.slug}
              only={pickFoldersByHand}
              onOnlyChange={setPickFoldersByHand}
              // Two folders, not one. The eval set is a SEPARATE download
              // (EVAL_DATA_S3_URI, read by a different container), so when
              // it has its own prefix the run must be told about it --
              // falling back to the training folder would evaluate on the
              // images it trained on and report that as held-out.
              onAssembled={(folder, evalFolder) => {
                setSelection({
                  datasetFolder: folder, evalFolder: evalFolder || folder,
                });
                setAssembledVersion((n) => n + 1);
              }}
            />
          )}

          {/* Divides a set somebody curated by hand. Ahead of the pull
              rather than after it: what belongs in a golden set is a
              judgement made on images you already have, so it owes the
              pull nothing and reads better settled before the collecting
              starts. */}
          {multi && <Track2GoldenSetPanel />}

          {/* Upstream of everything below it: pull control is what puts
              files into the week folders that Auto Training then discovers.
              Multi-auto-training only, for the same reason Auto Training
              is -- the standard tab works one folder at a time. */}
          {multi && (
            // Not wrapped in `SwitchedOff`: that makes the whole subtree
            // inert, and the tick that turns this card back on now lives
            // in its own heading. The card dims its own body instead, so
            // the switch stays reachable either way.
            <Track2PullControlPanel
              only={pickFoldersByHand}
              onOnlyChange={setPickFoldersByHand}
            />
          )}

          {/* Multi-auto-training only. The standard tab assembles one
              folder at a time; discovering a whole root of week folders is
              the thing that tab exists for. */}
          {/* Carries Retention inside it now. Retention decides how much
              of each older week an assembly takes, so it was being read
              on one card and applied on another, three cards apart. */}
          {multi && (
            <SwitchedOff
              off={pickFoldersByHand}
              reason={<>The run is using the folders picked above. Retention
                lives inside this card, so it is off with it.</>}
            >
              <Track2AutoTrainingPanel
                onAssembled={(folder) => setSelection({ datasetFolder: folder, evalFolder: folder })}
              />
            </SwitchedOff>
          )}

          {/* The Assemble card's yaml build renders here, by portal --
              the last step before the run, next to the button that trains
              on it. Its state stays in the Assemble card. */}
          <div id="track2-yaml-build-slot" />

          <TrainingForm
            selection={selection}
            setSelection={setSelection}
            onSubmitted={handleSubmitted}
            detectorsVersion={detectorsVersion}
            assembledVersion={assembledVersion}
          />

          {/* The two queues, side by side. Pull first because it is
              upstream: a sweep puts files where a training run then reads
              them, so reading left to right follows the work.

              Multi only -- the standard tab has no Pull Control card, so
              a pull queue there would queue settings nothing on that tab
              can see or change. */}
          {multi && <Track2PullQueuePanel />}

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
            ) : data.queue.running ? null : data.custom_run_id ? (
              <div className="mb-2 flex flex-col gap-1">
                <Badge tone="pending" label="Executing" />
                <p className="tabular text-xs text-ink-secondary">
                  Custom-class run {data.custom_run_id}
                </p>
              </div>
            ) : data.training_progress ? (
              <div className="mb-2 flex flex-col gap-2">
                <Badge
                  tone={
                    data.training_progress.status === "Executing" ? "pending" :
                    data.training_progress.status === "Succeeded" ? "confirmed" : "neutral"
                  }
                  label={data.training_progress.status}
                />
                {data.training_progress.epoch_progress && (
                  <div>
                    <p className="tabular mb-1 text-xs text-ink-muted">
                      Epoch {data.training_progress.epoch_progress.current} / {data.training_progress.epoch_progress.total}
                    </p>
                    <ProgressBar processed={data.training_progress.epoch_progress.current} total={data.training_progress.epoch_progress.total} />
                  </div>
                )}
                {data.training_progress.download_progress && (
                  <div>
                    <p className="tabular mb-1 text-xs text-ink-muted">
                      Downloading {data.training_progress.download_progress.current} / {data.training_progress.download_progress.total}
                    </p>
                    <ProgressBar processed={data.training_progress.download_progress.current} total={data.training_progress.download_progress.total} />
                  </div>
                )}
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
