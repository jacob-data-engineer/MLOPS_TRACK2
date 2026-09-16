import { AlertCircle, FlaskConical, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { Track2S3FolderBrowser } from "../components/Track2S3FolderBrowser";
import {
  addTrack2EvalRun,
  ApiError,
  getTrack2Eval,
  getTrack2Export,
  removeTrack2EvalRun,
  startTrack2EvalQueue,
  stopTrack2EvalQueue,
  type Track2EvalResponse,
  type Track2EvalRunStatus,
} from "../lib/api";

const GENERATION = "11";

function statusTone(status: Track2EvalRunStatus): "confirmed" | "rejected" | "pending" | "neutral" {
  if (status === "completed") return "confirmed";
  if (status === "failed") return "rejected";
  if (status === "queued") return "neutral";
  return "pending"; // deploying, running
}

function QueueForm({ data, onQueued }: { data: Track2EvalResponse; onQueued: (message: string) => void }) {
  const sizes = data.generation_sizes[GENERATION] ?? [];
  const [size, setSize] = useState(sizes[0] ?? "s");
  const [weightsKind, setWeightsKind] = useState<"registry" | "s3">("registry");
  const [selectedArn, setSelectedArn] = useState("");
  const [s3Folder, setS3Folder] = useState("");
  const [weightsPrefix, setWeightsPrefix] = useState("");
  const [weightsFolders, setWeightsFolders] = useState<string[]>([]);
  const [datasetPrefix, setDatasetPrefix] = useState("");
  const [datasetFolders, setDatasetFolders] = useState<string[]>([]);
  const [dataFolder, setDataFolder] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getTrack2Export(datasetPrefix)
      .then((res) => {
        if (!cancelled) setDatasetFolders(res.folders);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [datasetPrefix]);

  // Reuses Export's own folder-browsing endpoint again, purely for its
  // listing side effect (same convention as the dataset browser above),
  // scoped to nothing in particular -- an arbitrary bucket-relative path,
  // same as Deploy's own raw-S3 weights picker.
  useEffect(() => {
    let cancelled = false;
    getTrack2Export(weightsPrefix)
      .then((res) => {
        if (!cancelled) setWeightsFolders(res.folders);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [weightsPrefix]);

  const group = `yolo11${size}-detector`;
  const approvedPackages = data.registries[group]?.packages.filter((p) => p.approval_status === "Approved") ?? [];

  async function handleAdd() {
    if (!dataFolder) return;
    setBusy(true);
    setError(null);
    try {
      const res = await addTrack2EvalRun({
        size,
        data_folder: dataFolder,
        weights_kind: weightsKind,
        model_package_arn: weightsKind === "registry" ? selectedArn : undefined,
        s3_folder: weightsKind === "s3" ? s3Folder.trim() : undefined,
      });
      onQueued(res.message);
      setSelectedArn("");
      setS3Folder("");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Add to queue failed.");
    } finally {
      setBusy(false);
    }
  }

  const canAdd = Boolean(dataFolder) && (weightsKind === "registry" ? Boolean(selectedArn) : Boolean(s3Folder.trim()));

  return (
    <Card title="Queue an evaluation" caption="Deploys the chosen weights (temporarily occupying that size's endpoint), runs inference against every image in the dataset folder, and scores it against the matching YOLO-format labels.">
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
            <select className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-sm" value={size} onChange={(e) => { setSize(e.target.value); setSelectedArn(""); }}>
              {sizes.map((s) => (
                <option key={s} value={s}>{data.size_labels[s] ?? s}</option>
              ))}
            </select>
          </label>

          <div className="flex gap-2">
            {(["registry", "s3"] as const).map((kind) => (
              <button
                key={kind}
                type="button"
                onClick={() => setWeightsKind(kind)}
                className={`rounded-sm border px-3 py-1.5 text-xs font-medium ${
                  weightsKind === kind ? "border-accent bg-accent-subtle text-ink" : "border-black/15 text-ink-secondary"
                }`}
              >
                {kind === "registry" ? "Registry version" : "My own .pt weights"}
              </button>
            ))}
          </div>

          {weightsKind === "registry" ? (
            <div className="flex flex-col gap-1.5">
              {approvedPackages.length === 0 ? (
                <p className="text-sm text-ink-muted">No Approved versions for this size.</p>
              ) : (
                approvedPackages.map((p) => (
                  <label key={p.model_package_arn} className="flex items-center gap-2 rounded-sm border border-hairline bg-page px-3 py-2 text-sm">
                    <input
                      type="checkbox"
                      checked={selectedArn === p.model_package_arn}
                      onChange={(e) => setSelectedArn(e.target.checked ? p.model_package_arn : "")}
                    />
                    <span className="tabular">v{p.version}</span>
                  </label>
                ))
              )}
            </div>
          ) : (
            <div>
              <p className="mb-1 text-xs text-ink-muted">
                Not registered yet, and doesn't need to be Approved -- this is exactly how you'd evaluate a
                candidate before deciding whether to approve it.
              </p>
              <Track2S3FolderBrowser
                prefix={weightsPrefix}
                folders={weightsFolders}
                onNavigate={setWeightsPrefix}
                onSelect={setS3Folder}
                selected={s3Folder}
                allowCreate
              />
            </div>
          )}
        </div>

        <div>
          <span className="mb-1 block text-xs text-ink-muted">Dataset folder (needs images/ and labels/ inside)</span>
          <Track2S3FolderBrowser
            prefix={datasetPrefix}
            folders={datasetFolders}
            onNavigate={setDatasetPrefix}
            onSelect={setDataFolder}
            selected={dataFolder}
            allowCreate
          />
        </div>
      </div>

      <button
        type="button"
        onClick={handleAdd}
        disabled={busy || !canAdd}
        className="mt-4 flex items-center gap-1.5 rounded-md bg-accent px-3 py-2 text-sm font-medium text-white disabled:opacity-40"
      >
        {busy && <Loader2 size={14} className="animate-spin" />}
        Add to Queue
      </button>
    </Card>
  );
}

export function Track2EvalPage() {
  const [data, setData] = useState<Track2EvalResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  function load() {
    getTrack2Eval()
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load the evaluation queue."));
  }

  useEffect(load, []);

  useEffect(() => {
    const active = data?.runs.some((r) => r.status === "queued" || r.status === "deploying" || r.status === "running");
    if (!active) return;
    const id = setInterval(load, 15000);
    return () => clearInterval(id);
  }, [data]);

  async function handleQueued(msg: string) {
    setMessage(msg);
    load();
  }

  async function handleRemove(id: string) {
    await removeTrack2EvalRun(id);
    load();
  }

  async function handleStart() {
    try {
      const res = await startTrack2EvalQueue();
      setMessage(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Start failed.");
    }
    load();
  }

  async function handleStop() {
    const res = await stopTrack2EvalQueue();
    setMessage(res.message);
    load();
  }

  const pendingRuns = data?.runs.filter((r) => r.status === "queued" || r.status === "deploying" || r.status === "running") ?? [];
  const finishedRuns = data?.runs.filter((r) => r.status === "completed" || r.status === "failed") ?? [];

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <FlaskConical size={20} className="text-accent" strokeWidth={2} />
        Track2 Evaluation Queue
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Score a registry version -- or your own candidate weights -- against a
        chosen dataset, with real precision/recall/mAP metrics a reviewer can
        judge performance from. Runs one at a time; results persist here.
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
          <QueueForm data={data} onQueued={handleQueued} />

          <Card
            title="Queue"
            caption={`${pendingRuns.length} pending/running`}
            action={
              <div className="flex shrink-0 gap-2">
                <button
                  type="button"
                  onClick={handleStart}
                  disabled={data.queue.started || pendingRuns.length === 0}
                  className="rounded-md border border-black/15 px-3 py-2 text-xs font-medium text-ink-secondary disabled:opacity-40"
                >
                  Start Queue
                </button>
                <button
                  type="button"
                  onClick={handleStop}
                  disabled={!data.queue.started}
                  className="rounded-md bg-red-600 px-3 py-2 text-xs font-medium text-white disabled:opacity-40"
                >
                  Stop Queue
                </button>
              </div>
            }
          >
            {pendingRuns.length === 0 ? (
              <p className="text-sm text-ink-muted">Nothing queued.</p>
            ) : (
              <div className="flex flex-col gap-2">
                {pendingRuns.map((run) => (
                  <div key={run.id} className="flex items-center justify-between gap-3 rounded-sm border border-hairline bg-page p-2.5">
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <Badge tone={statusTone(run.status)} label={run.status} />
                        <span className="tabular text-sm text-ink">{run.weights_label}</span>
                      </div>
                      <p className="tabular mt-0.5 truncate text-xs text-ink-muted" title={run.data_folder}>
                        {run.data_folder}
                      </p>
                    </div>
                    {run.status === "queued" && (
                      <button
                        type="button"
                        onClick={() => handleRemove(run.id)}
                        className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
                      >
                        Remove
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </Card>

          <Card title="Results">
            {finishedRuns.length === 0 ? (
              <p className="text-sm text-ink-muted">No completed evaluations yet.</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead>
                    <tr className="border-b border-hairline text-xs text-ink-muted">
                      <th className="py-1.5 pr-4 font-medium">Model</th>
                      <th className="py-1.5 pr-4 font-medium">Status</th>
                      <th className="py-1.5 pr-4 font-medium">Dataset</th>
                      <th className="tabular py-1.5 pr-4 font-medium">mAP50</th>
                      <th className="tabular py-1.5 pr-4 font-medium">mAP50-95</th>
                      <th className="tabular py-1.5 pr-4 font-medium">Precision</th>
                      <th className="tabular py-1.5 pr-4 font-medium">Recall</th>
                      <th className="tabular py-1.5 pr-4 font-medium">Accuracy</th>
                      <th className="tabular py-1.5 font-medium">Images</th>
                    </tr>
                  </thead>
                  <tbody>
                    {finishedRuns.map((run) => (
                      <tr key={run.id} className="border-b border-hairline last:border-0">
                        <td className="tabular py-1.5 pr-4">{run.weights_label}</td>
                        <td className="py-1.5 pr-4">
                          <Badge tone={statusTone(run.status)} label={run.status} />
                          {run.status === "failed" && run.error && (
                            <p className="mt-0.5 max-w-[16rem] truncate text-xs text-critical" title={run.error}>
                              {run.error}
                            </p>
                          )}
                        </td>
                        <td className="max-w-[12rem] truncate py-1.5 pr-4 text-xs text-ink-secondary" title={run.data_folder}>
                          {run.data_folder}
                        </td>
                        <td className="tabular py-1.5 pr-4">{run.metrics?.mAP50?.toFixed(3) ?? "--"}</td>
                        <td className="tabular py-1.5 pr-4">{run.metrics?.["mAP50-95"]?.toFixed(3) ?? "--"}</td>
                        <td className="tabular py-1.5 pr-4">{run.metrics?.precision?.toFixed(3) ?? "--"}</td>
                        <td className="tabular py-1.5 pr-4">{run.metrics?.recall?.toFixed(3) ?? "--"}</td>
                        <td className="tabular py-1.5 pr-4">{run.metrics?.accuracy?.toFixed(3) ?? "--"}</td>
                        <td className="tabular py-1.5">{run.metrics?.num_images ?? "--"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      )}
    </div>
  );
}
