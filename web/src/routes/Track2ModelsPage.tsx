import { AlertCircle, CheckCircle2, Cpu, Loader2, XCircle } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { ProgressBar } from "../components/ProgressBar";
import { StatTile } from "../components/StatTile";
import { Track2AutomatedQueuesPanel } from "../components/Track2AutomatedQueuesPanel";
import { Track2S3FolderBrowser } from "../components/Track2S3FolderBrowser";
import {
  addTrack2AutomatedQueue,
  addTrack2Queue,
  ApiError,
  deleteTrack2AutomatedQueue,
  getTrack2Models,
  removeTrack2Queue,
  startTrack2Queue,
  startTrack2Training,
  stopTrack2Queue,
  stopTrack2Training,
  type Track2AutomatedQueueParams,
  type Track2ModelsResponse,
  type Track2TrainingParams,
} from "../lib/api";

const WEEKDAYS = ["SUN", "MON", "TUE", "WED", "THU", "FRI", "SAT"];
const GENERATION = "11";

/** Everything both the immediate training actions (Start/Add to Queue) and
 * the automated-queue form need -- lifted up to the page so the automated
 * queue reads the same generation/size/dataset/etc. the user already set
 * above it, instead of holding its own separate duplicate copy of those
 * fields. */
interface TrainingSelection {
  size: string;
  epochs: number;
  imgSize: number;
  batch: number;
  seed: number;
  datasetFolder: string;
  baseModelFolder: string;
  noBaseModel: boolean;
  notes: string;
}

function TrainingForm({
  data,
  selection,
  setSelection,
  onSubmitted,
}: {
  data: Track2ModelsResponse;
  selection: TrainingSelection;
  setSelection: (patch: Partial<TrainingSelection>) => void;
  onSubmitted: (message: string) => void;
}) {
  const sizes = data.generation_sizes[GENERATION] ?? [];
  const { size, epochs, imgSize, batch, seed, datasetFolder, baseModelFolder, noBaseModel, notes } = selection;
  const [datasetPrefix, setDatasetPrefix] = useState(data.dataset_browser.prefix);
  const [weightsPrefix, setWeightsPrefix] = useState("");
  const [datasetFolders, setDatasetFolders] = useState(data.dataset_browser.folders);
  const [weightsFolders, setWeightsFolders] = useState(data.weights_browser.folders);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The page-level `data` only ever reflects the DEFAULT browse prefixes
  // (fetched once on load) -- navigating into a subfolder only moved this
  // component's own prefix state, never actually fetched THAT folder's
  // contents, so the list silently stayed stuck showing the original
  // level no matter how deep you navigated. Re-fetch for whichever
  // prefixes are actually being browsed right now instead.
  useEffect(() => {
    let cancelled = false;
    getTrack2Models(datasetPrefix, weightsPrefix)
      .then((res) => {
        if (cancelled) return;
        setDatasetFolders(res.dataset_browser.folders);
        setWeightsFolders(res.weights_browser.folders);
      })
      .catch(() => {
        // A browse-only refresh failing isn't worth surfacing as a page
        // error -- the existing listing just stays as it was.
      });
    return () => {
      cancelled = true;
    };
  }, [datasetPrefix, weightsPrefix]);

  const latestApproved = data.latest_approved[GENERATION]?.[size] ?? null;

  const params = (): Track2TrainingParams => ({
    generation: GENERATION,
    size,
    epochs,
    img_size: imgSize,
    batch,
    seed,
    dataset_folder: datasetFolder,
    base_model_folder: noBaseModel ? "" : baseModelFolder,
    notes,
  });

  async function handleStart() {
    setBusy(true);
    setError(null);
    try {
      const res = await startTrack2Training(params());
      onSubmitted(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Training trigger failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleAddToQueue() {
    setBusy(true);
    setError(null);
    try {
      const res = await addTrack2Queue(params());
      onSubmitted(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Add to queue failed.");
    } finally {
      setBusy(false);
    }
  }

  const isExecuting = data.training_progress?.status === "Executing";

  async function handleStop() {
    setBusy(true);
    setError(null);
    try {
      const res = await stopTrack2Training();
      onSubmitted(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Stop failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Start training">
      {error && (
        <div className="mb-3 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical">
          <AlertCircle size={14} />
          {error}
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="flex flex-col gap-3">
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Size (YOLO{GENERATION})</span>
            <select className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={size} onChange={(e) => setSelection({ size: e.target.value })}>
              {sizes.map((s) => (
                <option key={s} value={s}>{data.size_labels[s] ?? s}</option>
              ))}
            </select>
          </label>
          <div className="grid grid-cols-2 gap-3">
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Epochs</span>
              <input type="number" className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={epochs} onChange={(e) => setSelection({ epochs: Number(e.target.value) })} />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Image size</span>
              <input type="number" className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={imgSize} onChange={(e) => setSelection({ imgSize: Number(e.target.value) })} />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Batch</span>
              <input type="number" className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={batch} onChange={(e) => setSelection({ batch: Number(e.target.value) })} />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Seed</span>
              <input type="number" className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={seed} onChange={(e) => setSelection({ seed: Number(e.target.value) })} />
            </label>
          </div>
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Notes</span>
            <textarea
              className="w-full resize-vertical rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
              rows={10}
              value={notes}
              onChange={(e) => setSelection({ notes: e.target.value })}
              maxLength={2048}
            />
          </label>
        </div>

        <div className="flex flex-col gap-3">
          <div>
            <span className="mb-1 block text-xs text-ink-muted">Dataset folder</span>
            <Track2S3FolderBrowser
              prefix={datasetPrefix}
              folders={datasetFolders}
              onNavigate={setDatasetPrefix}
              onSelect={(folder) => setSelection({ datasetFolder: folder })}
              selected={datasetFolder}
              allowCreate
            />
          </div>
          <div>
            <span className="mb-1 block text-xs text-ink-muted">Base model .pt file (browse the whole bucket)</span>
            <div className={noBaseModel ? "pointer-events-none opacity-40" : ""}>
              <Track2S3FolderBrowser
                prefix={weightsPrefix}
                folders={weightsFolders}
                onNavigate={setWeightsPrefix}
                onSelect={(folder) => setSelection({ baseModelFolder: folder })}
                selected={baseModelFolder}
                allowCreate
              />
            </div>
            <label className="mt-1.5 flex items-center gap-1.5 text-xs font-normal text-ink-secondary">
              <input
                type="checkbox"
                checked={noBaseModel}
                onChange={(e) => setSelection({ noBaseModel: e.target.checked })}
              />
              No .pt file available -- use the registry's latest Approved package instead
            </label>
            {noBaseModel && (
              latestApproved ? (
                <p className="tabular mt-1 text-xs text-ink-muted">
                  Latest Approved: {latestApproved.group} v{latestApproved.version}
                </p>
              ) : (
                <p className="mt-1 text-xs text-critical">
                  No Approved package in yolo{GENERATION}{size}-detector yet -- training will fail until one is approved.
                </p>
              )
            )}
          </div>
        </div>
      </div>

      <div className="mt-4 flex gap-2">
        <button type="button" onClick={handleStart} disabled={busy || !datasetFolder} className="flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40">
          {busy && <Loader2 size={14} className="animate-spin" />}
          Start Training
        </button>
        <button type="button" onClick={handleAddToQueue} disabled={busy || !datasetFolder} className="rounded-md border border-black/15 px-3 py-2 text-sm font-medium text-ink-secondary disabled:opacity-40">
          Add to Queue
        </button>
        <button
          type="button"
          onClick={handleStop}
          disabled={busy || !isExecuting}
          className="flex items-center gap-1.5 rounded-md bg-red-600 px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
        >
          {busy && <Loader2 size={14} className="animate-spin" />}
          Stop Training
        </button>
      </div>
    </Card>
  );
}

function AutomatedQueueForm({
  selection,
  onAdded,
}: {
  selection: TrainingSelection;
  onAdded: (message: string) => void;
}) {
  const { size, epochs, imgSize, batch, seed, datasetFolder, baseModelFolder, noBaseModel, notes } = selection;
  const [mode, setMode] = useState<"daily" | "weekly" | "monthly">("weekly");
  const [hour, setHour] = useState(9);
  const [day, setDay] = useState("SUN");
  const [dom, setDom] = useState(1);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleAdd() {
    setBusy(true);
    setError(null);
    try {
      const params: Track2AutomatedQueueParams = {
        generation: GENERATION, size, epochs, img_size: imgSize, batch, seed,
        dataset_folder: datasetFolder, base_model_folder: noBaseModel ? "" : baseModelFolder, notes,
        mode,
        daily_hour: mode === "daily" ? hour : undefined,
        weekly_day: mode === "weekly" ? day : undefined,
        weekly_hour: mode === "weekly" ? hour : undefined,
        monthly_day: mode === "monthly" ? dom : undefined,
        monthly_hour: mode === "monthly" ? hour : undefined,
      };
      const res = await addTrack2AutomatedQueue(params);
      onAdded(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Add automated queue failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card title="Add an automated queue" caption="A persistent, individually-scheduled recipe -- fires via its own EventBridge rule. Uses the model size, dataset, base model, and hyperparameters set above in Start Training.">
      {error && (
        <div className="mb-3 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical">
          <AlertCircle size={14} />
          {error}
        </div>
      )}
      <p className="tabular mb-3 truncate text-xs text-ink-muted" title={datasetFolder}>
        Will use: YOLO{GENERATION}-{size.toUpperCase()} &middot; {epochs} epochs &middot; {datasetFolder || "no dataset selected above"}
      </p>
      <div className="flex flex-wrap items-end gap-3">
        <label className="block">
          <span className="mb-1 block text-xs text-ink-muted">Mode</span>
          <select className="rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={mode} onChange={(e) => setMode(e.target.value as typeof mode)}>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
            <option value="monthly">Monthly</option>
          </select>
        </label>
        {mode === "weekly" && (
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Day</span>
            <select className="rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={day} onChange={(e) => setDay(e.target.value)}>
              {WEEKDAYS.map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>
        )}
        {mode === "monthly" && (
          <label className="block">
            <span className="mb-1 block text-xs text-ink-muted">Day of month</span>
            <input type="number" min={1} max={28} className="tabular w-20 rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={dom} onChange={(e) => setDom(Number(e.target.value))} />
          </label>
        )}
        <label className="block">
          <span className="mb-1 block text-xs text-ink-muted">Hour (SGT)</span>
          <input type="number" min={0} max={23} className="tabular w-20 rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={hour} onChange={(e) => setHour(Number(e.target.value))} />
        </label>
        <button type="button" onClick={handleAdd} disabled={busy || !datasetFolder} className="flex items-center gap-1.5 rounded-md border border-black/15 px-3 py-2 text-sm font-medium text-ink-secondary disabled:opacity-40">
          {busy && <Loader2 size={14} className="animate-spin" />}
          Add to Automated Queue
        </button>
      </div>
    </Card>
  );
}

export function Track2ModelsPage() {
  const [data, setData] = useState<Track2ModelsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [selection, setSelectionState] = useState<TrainingSelection>({
    size: "s",
    epochs: 50,
    imgSize: 640,
    batch: 16,
    seed: 42,
    datasetFolder: "",
    baseModelFolder: "",
    noBaseModel: true,
    notes: "",
  });

  function setSelection(patch: Partial<TrainingSelection>) {
    setSelectionState((current) => ({ ...current, ...patch }));
  }

  function load() {
    getTrack2Models()
      .then((res) => {
        setData(res);
        // Only seed defaults once (on first load) -- an in-progress
        // selection shouldn't get stomped by every 15s auto-refresh poll.
        setSelectionState((current) =>
          current.datasetFolder ? current : { ...current, size: (res.generation_sizes[GENERATION] ?? ["s"])[0], datasetFolder: res.default_dataset },
        );
      })
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load training status."));
  }

  useEffect(load, []);

  // Auto-refresh every 15s while a run or queue is active -- mirrors the
  // old app's <meta refresh>, done here as a poll instead.
  const shouldPoll = useMemo(
    () => Boolean(data?.training_progress?.status === "Executing" || data?.queue.started || data?.pending_message),
    [data],
  );
  useEffect(() => {
    if (!shouldPoll) return;
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [shouldPoll]);

  async function handleQueueRemove(id: string) {
    await removeTrack2Queue(id);
    load();
  }

  async function handleQueueStart() {
    const res = await startTrack2Queue();
    setMessage(res.message);
    load();
  }

  async function handleQueueStop() {
    const res = await stopTrack2Queue();
    setMessage(res.message);
    load();
  }

  async function handleDeleteAutomatedQueue(id: string) {
    await deleteTrack2AutomatedQueue(id);
    load();
  }

  function handleSubmitted(msg: string) {
    setMessage(msg);
    load();
  }

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Cpu size={20} className="text-accent" strokeWidth={2} />
        Track2 Training
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Dataset/weights browsing, GPU status, training progress, and the
        training queue for the Track2 YOLO pipeline.
      </p>

      {error && (
        <div className="mb-6 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-4 py-3 text-sm text-critical">
          <AlertCircle size={16} />
          {error}
        </div>
      )}
      {message && (
        <div className="mb-6 rounded-sm border border-hairline bg-accent-subtle px-4 py-3 text-sm text-ink">
          {message}
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
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Card title="SageMaker GPU resource">
              {data.gpu_jobs.length === 0 ? (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-600 px-2.5 py-1 text-xs font-medium text-white">
                  <CheckCircle2 size={13} strokeWidth={2.25} />
                  Available
                </span>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <span className="inline-flex w-fit items-center gap-1.5 rounded-full bg-red-600 px-2.5 py-1 text-xs font-medium text-white">
                    <XCircle size={13} strokeWidth={2.25} />
                    In use
                  </span>
                  {data.gpu_jobs.map((j) => (
                    <p key={j.name} className="tabular text-xs text-ink-muted">
                      {j.name} &middot; {j.own_pipeline ? "this app's own pipeline" : "a different system"}
                    </p>
                  ))}
                </div>
              )}
            </Card>

            <Card title="Latest training run">
              {data.pending_message ? (
                <p className="text-sm font-medium text-ink">{data.pending_message}</p>
              ) : data.training_progress ? (
                <div className="flex flex-col gap-2">
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
              ) : (
                <p className="text-sm text-ink-muted">No pipeline execution has ever run.</p>
              )}
            </Card>
          </div>

          <TrainingForm data={data} selection={selection} setSelection={setSelection} onSubmitted={handleSubmitted} />

          <Card
            title="Training queue"
            caption={`${data.queue.pending.length} of ${data.queue.max_size} max`}
            action={
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  onClick={handleQueueStart}
                  disabled={data.queue.started || (data.queue.pending.length === 0 && !data.queue.running)}
                  className="rounded-md border border-black/15 px-3 py-2 text-xs font-medium text-ink-secondary disabled:opacity-40"
                >
                  Start Queue
                </button>
                <button
                  type="button"
                  onClick={handleQueueStop}
                  disabled={!data.queue.started}
                  className="rounded-md bg-red-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-40"
                >
                  Stop Queue
                </button>
              </div>
            }
          >
            {data.queue.last_error && (
              <p className="mb-2 text-xs text-critical">{data.queue.last_error}</p>
            )}
            <div className="mb-3">
              <StatTile label="Completed" value={data.queue.completed_count} />
            </div>
            {data.queue.running && (
              <p className="tabular mb-2 text-xs text-ink-secondary">
                Running: YOLO{data.queue.running.generation}-{data.queue.running.size}
              </p>
            )}
            {data.queue.pending.length === 0 ? (
              <p className="text-sm text-ink-muted">Queue is empty.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {data.queue.pending.map((e) => (
                  <div key={e.id} className="flex items-start justify-between gap-3 rounded-sm border border-hairline bg-page p-2.5">
                    <div className="tabular grid flex-1 grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-3">
                      <span>Model: YOLO{e.generation}-{e.size}</span>
                      <span>Epochs: {e.epochs}</span>
                      <span>Image size: {e.img_size}</span>
                      <span>Batch: {e.batch}</span>
                      <span>Seed: {e.seed}</span>
                      <span className="col-span-2 truncate sm:col-span-3" title={e.dataset_folder}>
                        Dataset: {e.dataset_folder}
                      </span>
                      {e.base_model_folder && (
                        <span className="col-span-2 truncate sm:col-span-3" title={e.base_model_folder}>
                          Base model: {e.base_model_folder}
                        </span>
                      )}
                      {e.notes && (
                        <span className="col-span-2 truncate sm:col-span-3" title={e.notes}>
                          Notes: {e.notes}
                        </span>
                      )}
                    </div>
                    <button
                      type="button"
                      onClick={() => handleQueueRemove(e.id)}
                      className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
                    >
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
          </Card>

          <AutomatedQueueForm selection={selection} onAdded={handleSubmitted} />
          <Track2AutomatedQueuesPanel queues={data.automated_queues} onDelete={handleDeleteAutomatedQueue} />
        </div>
      )}
    </div>
  );
}
