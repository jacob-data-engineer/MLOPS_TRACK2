import { AlertCircle, ChevronDown, ChevronRight, Loader2, Power, Rocket } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { Badge } from "../components/Badge";
import { Card } from "../components/Card";
import { Track2S3FolderBrowser } from "../components/Track2S3FolderBrowser";
import {
  addTrack2InferenceQueueItem,
  ApiError,
  deleteTrack2Endpoint,
  getTrack2Cost,
  getTrack2Deploy,
  getTrack2DeployBrowse,
  getTrack2Metrics,
  getTrack2RunLineage,
  track2HostingTotal,
  removeTrack2InferenceQueueItem,
  startTrack2InferenceQueue,
  stopTrack2InferenceQueue,
  type Track2DeployedComponent,
  type Track2DeployResponse,
  type Track2EndpointStatus,
  type Track2InferenceQueueItem,
  type Track2InferenceQueueItemStatus,
  type Track2MetricsRow,
  type Track2RunLineage,
} from "../lib/api";
import { formatElapsed, formatTimestamp } from "../lib/format";

const SIZES = ["s", "m", "l", "x"] as const;

type EndpointPhase = "ready" | "inferencing" | "deploying_endpoint" | "spinning_up" | "not_in_use" | "not_deployed";

/** Derived entirely from fields SageMaker (and this app's own
 * invoke_inference tracking) already report -- no guessing, and no more
 * manual "Spin Up" action: every deploy already requests CopyCount: 1
 * regardless of the count beforehand, so starting a queued job is the
 * only action that's ever needed, and this fully reflects where that
 * process currently is. */
function phaseOf(d: Track2DeployedComponent | null): EndpointPhase {
  if (!d) return "not_deployed";
  if (d.running_inference) return "inferencing";
  if (d.status === "Creating") return "deploying_endpoint";
  if (d.status === "Updating") {
    const desired = d.desired_copy_count ?? 0;
    const current = d.current_copy_count ?? 0;
    if (current === 0 && desired > 0) return "spinning_up";
    return "deploying_endpoint";
  }
  if ((d.current_copy_count ?? 0) === 0) return "not_in_use";
  return "ready";
}

function isTransitional(phase: EndpointPhase): boolean {
  return phase === "deploying_endpoint" || phase === "spinning_up";
}

const PHASE_LABEL: Record<EndpointPhase, string> = {
  ready: "Ready",
  inferencing: "Inferencing",
  deploying_endpoint: "Deploying endpoint",
  spinning_up: "Spinning up",
  not_in_use: "Not in use",
  not_deployed: "Not deployed",
};

const GROUP_GENERATION = /^yolo(\d+)[smlx]/;

const PHASE_PILL_CLASS: Record<EndpointPhase, string> = {
  ready: "bg-emerald-600",
  inferencing: "bg-accent",
  deploying_endpoint: "bg-amber-600",
  spinning_up: "bg-amber-600",
  not_in_use: "bg-black/40",
  not_deployed: "bg-black/40",
};

/** Today's endpoint-hosting spend, shown against the endpoint it belongs
 * to rather than only on the Cost tab -- the decision to leave this
 * running or shut it down is made here, so the number that decision costs
 * belongs here too.
 *
 * Read once on mount, never on the poll: Cost Explorer charges per request
 * and this page refreshes every few seconds. The backend caches it as
 * well, so revisits are free. */
function EndpointCostToday() {
  const [cost, setCost] = useState<{ amount: number; currency: string; date: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    getTrack2Cost(1)
      .then((res) => {
        const latest = res.days[0];
        if (!latest || cancelled) return;
        setCost({
          amount: track2HostingTotal(latest.items),
          currency: res.currency,
          date: latest.date,
        });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  if (!cost) return null;
  const value = cost.currency === "USD" ? `$${cost.amount.toFixed(2)}` : `${cost.amount.toFixed(2)} ${cost.currency}`;
  return (
    <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2 rounded-sm border border-hairline bg-page px-3 py-2">
      <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">
        Endpoint cost today &middot; {cost.date}
      </p>
      <p className="tabular text-lg font-bold text-ink">{value}</p>
      <p className="w-full text-xs text-ink-secondary">
        Hosting only, separate from training and Studio. Cost Explorer reports
        hours behind, so this lags the endpoint you can see above rather than
        contradicting it.
      </p>
    </div>
  );
}

/** Where the model itself is up to, distinct from the endpoint that hosts
 * it: the endpoint can be running and billing while the model is still
 * loading, or unloaded entirely. Sits under the shut-down button so both
 * halves of "what is live right now" read as one block. */
function PhasePill({ deployed }: { deployed: Track2DeployedComponent | null }) {
  const phase = phaseOf(deployed);
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold text-white ${PHASE_PILL_CLASS[phase]}`}
      title={deployed?.model_name ?? undefined}
    >
      {phase === "inferencing" && <span className="inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-white" />}
      {PHASE_LABEL[phase]}
    </span>
  );
}

/** "YOLO26-S", read from the model package rather than asserted.
 *
 * This said `YOLO11-${size}` for every model. The platform has been able
 * to train generation 26 since 2026-09-30, so a YOLO26 checkpoint would
 * be labelled YOLO11 on the one page that exists to say what is
 * deployed -- and the size beside it would be right, which makes the
 * whole label read as considered rather than hard-coded.
 *
 * The package ARN carries the group and the group carries the
 * generation: ".../model-package/yolo26s-helmet-detector/3". With no
 * package -- S3 weights -- the generation is genuinely unknown, and
 * saying so beats guessing. */
function lineageModelLabel(lineage: Track2RunLineage): string {
  const size = (lineage.model.size ?? "?").toUpperCase();
  const arn = lineage.model.model_package_arn ?? "";
  const group = arn.split("/").slice(-2)[0] ?? "";
  const match = GROUP_GENERATION.exec(group);
  return match ? `YOLO${match[1]}-${size}` : `YOLO-${size}`;
}

/** What produced one run's results, read back from the lineage recorded
 * in DynamoDB (falling back to the output folder's own lineage.json).
 * Fetched on expand rather than with the queue: it is a lookup per run,
 * and loading every run's file list to render collapsed rows would put
 * the cost of the whole history on every poll. */
function RunLineage({ outputFolder, runId }: { outputFolder: string; runId: string }) {
  const [open, setOpen] = useState(false);
  const [lineage, setLineage] = useState<Track2RunLineage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (!next || lineage || loading) return;
    setLoading(true);
    setError(null);
    try {
      setLineage(await getTrack2RunLineage(outputFolder, runId));
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Couldn't read the lineage for this run.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mt-1.5">
      <button
        type="button"
        onClick={toggle}
        className="inline-flex items-center gap-1 text-xs font-medium text-ink-secondary hover:text-ink"
      >
        {open ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
        Lineage
      </button>

      {open && (
        <div className="mt-1.5 rounded-sm border border-hairline bg-surface p-2.5">
          {loading && (
            <p className="flex items-center gap-1.5 text-xs text-ink-muted">
              <Loader2 size={12} className="animate-spin" />
              Reading lineage...
            </p>
          )}
          {error && <p className="text-xs text-critical">{error}</p>}
          {lineage && (
            <div className="flex flex-col gap-1.5 text-xs">
              <LineageRow label="Model">
                {lineageModelLabel(lineage)}
                {lineage.model.version ? ` · registry v${lineage.model.version}` : ""}
                {!lineage.model.version && lineage.model.weights_kind === "s3" ? " · S3 weights" : ""}
              </LineageRow>
              {lineage.model.weights_label && <LineageRow label="Weights">{lineage.model.weights_label}</LineageRow>}
              {lineage.model.model_package_arn && (
                <LineageRow label="Package">{lineage.model.model_package_arn}</LineageRow>
              )}
              {(lineage.model.weights_key || lineage.model.weights_folder) && (
                <LineageRow label="Checkpoint">
                  {lineage.model.weights_key || lineage.model.weights_folder}
                </LineageRow>
              )}
              <LineageRow label="Source">{`s3://${lineage.bucket}/${lineage.images_folder}`}</LineageRow>
              <LineageRow label="Output">{`s3://${lineage.bucket}/${lineage.output_folder}`}</LineageRow>
              {lineage.inference_started_at && (
                <LineageRow label="Inference">
                  {formatTimestamp(lineage.inference_started_at)}
                  {lineage.inference_completed_at ? ` → ${formatTimestamp(lineage.inference_completed_at)}` : ""}
                </LineageRow>
              )}
              {lineage.model.notes && <LineageRow label="Notes">{lineage.model.notes}</LineageRow>}

              <div className="mt-0.5">
                <p className="font-semibold text-ink-secondary">
                  Files ({lineage.files.length})
                </p>
                <ul className="tabular mt-1 max-h-40 space-y-0.5 overflow-y-auto text-ink-muted">
                  {lineage.files.map((f) => (
                    <li key={f} className="truncate" title={f}>
                      {f}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function LineageRow({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex gap-2">
      <span className="w-20 shrink-0 font-semibold text-ink-secondary">{label}</span>
      <span className="tabular min-w-0 flex-1 break-all text-ink">{children}</span>
    </div>
  );
}

/** Endpoint states that are still moving and will change on their own.
 * "InService" and a missing endpoint are the two resting states; anything
 * else resolves without further input, so the page should keep watching. */
const ENDPOINT_IN_FLIGHT = ["Creating", "Updating", "SystemUpdating", "RollingBack", "Deleting"];

function endpointInFlight(endpoint: Track2EndpointStatus | null): boolean {
  return endpoint !== null && ENDPOINT_IN_FLIGHT.includes(endpoint.status);
}

/** The endpoint is created on demand and deleted once every queue has
 * drained, because a real-time endpoint bills its instance for every hour
 * it EXISTS -- not per request. That makes "is it up right now" a cost
 * question, not just a readiness one, so it gets its own row here with a
 * manual shutdown for the case where something left it running. */
function EndpointLifecycleRow({
  endpoint,
  deployed,
  onChanged,
}: {
  endpoint: Track2EndpointStatus | null;
  deployed: Track2DeployedComponent | null;
  onChanged: (msg: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleShutdown() {
    setBusy(true);
    setError(null);
    try {
      const res = await deleteTrack2Endpoint();
      onChanged(res.deleted ? "Shutting down the endpoint -- billing stops once it's gone." : `Left running: ${res.reason}.`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not shut the endpoint down.");
    } finally {
      setBusy(false);
    }
  }

  if (!endpoint) {
    return (
      <div className="mb-3 rounded-sm border border-hairline bg-page p-3">
        <div className="flex items-center gap-2">
          <Power size={14} className="shrink-0 text-ink-muted" />
          <p className="text-sm font-semibold text-ink">Endpoint not running &mdash; nothing is being billed</p>
        </div>
        <p className="mt-1 text-xs text-ink-secondary">
          It starts automatically when a queued job needs it (takes a few
          minutes), and shuts down again once the queue drains.
        </p>
      </div>
    );
  }

  const live = endpoint.status === "InService";
  const inFlight = endpointInFlight(endpoint);
  const label = live
    ? "Endpoint running — billing"
    : endpoint.status === "Creating"
      ? "Endpoint starting up..."
      : endpoint.status === "Deleting"
        ? "Endpoint shutting down..."
        : `Endpoint ${endpoint.status.toLowerCase()}`;
  return (
    <div className="mb-3 rounded-sm border border-hairline bg-page p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            {inFlight ? (
              <Loader2 size={14} className="shrink-0 animate-spin text-ink-muted" />
            ) : (
              <Power size={14} className={`shrink-0 ${live ? "text-critical" : "text-ink-muted"}`} />
            )}
            <p className="text-sm font-semibold text-ink">{label}</p>
          </div>
          <p className="tabular mt-1 text-xs text-ink-muted">
            {endpoint.instance_type}
            {endpoint.instance_count ? ` × ${endpoint.instance_count}` : ""}
            {endpoint.created_at ? ` · up since ${formatTimestamp(endpoint.created_at)}` : ""}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1.5">
          {live && (
            <button
              type="button"
              onClick={handleShutdown}
              disabled={busy}
              className="rounded-sm border border-black/15 px-2.5 py-1 text-xs font-medium text-ink-secondary disabled:opacity-40"
            >
              {busy ? "Shutting down..." : "Shut down"}
            </button>
          )}
          <PhasePill deployed={deployed} />
        </div>
      </div>
      {live && (
        <p className="mt-2 text-xs text-ink-secondary">
          Billed per instance-hour for as long as it exists, whether or not
          anything is running on it.
        </p>
      )}
      {endpoint.status === "Creating" && (
        <p className="mt-2 text-xs text-ink-secondary">
          Building the instance, usually a few minutes. Billing has already
          started; the first queued job runs once this finishes.
        </p>
      )}
      {endpoint.status === "Deleting" && (
        <p className="mt-2 text-xs text-ink-secondary">
          Billing stops once it is gone. The model is removed first, so this
          takes a couple of minutes.
        </p>
      )}
      {error && <p className="mt-2 text-xs text-critical">{error}</p>}
    </div>
  );
}

/** One box, one model -- there is deliberately only ever one model live
 * at a time (four independent always-on slots read as confusing next to
 * a single "what's live right now" answer), so there's no size switcher
 * here anymore: whatever was deployed last is the whole answer. */
function EndpointStatusPanel({
  deployed,
  endpoint,
  message,
  onChanged,
}: {
  deployed: Track2DeployedComponent | null;
  endpoint: Track2EndpointStatus | null;
  message: string | null;
  onChanged: (msg: string) => void;
}) {
  return (
    <Card title="Endpoint in use">
      {message && (
        <p className="mb-3 rounded-sm border border-hairline bg-accent-subtle px-3 py-2 text-sm text-ink">{message}</p>
      )}
      <EndpointCostToday />
      <EndpointLifecycleRow endpoint={endpoint} deployed={deployed} onChanged={onChanged} />
    </Card>
  );
}

function approvalTone(status: string): "confirmed" | "rejected" | "pending" {
  if (status === "Approved") return "confirmed";
  if (status === "Rejected") return "rejected";
  return "pending";
}

function rowKey(row: Track2MetricsRow): string {
  return `${row.group_name}/${row.version}`;
}

function inferenceQueueStatusTone(status: Track2InferenceQueueItemStatus): "confirmed" | "rejected" | "pending" | "neutral" {
  if (status === "done") return "confirmed";
  if (status === "failed") return "rejected";
  if (status === "queued") return "neutral";
  return "pending"; // deploying, running
}

/** Queue up several (model, images folder, output folder) jobs -- each
 * one deploys its model (swapping out whatever's currently active, same
 * as a manual deploy), runs the whole images folder through it, and
 * moves to the next job once that's done. Runs on the backend's own
 * in-memory worker thread, one job at a time; add-and-go starts the
 * queue immediately, same UX as the Deploy section above. */
function InferenceQueueSection({
  queue,
  queueStatus,
  weightsPrefix,
  weightsFolders,
  weightFiles,
  onNavigateWeights,
  imagesPrefix,
  imagesFolders,
  onNavigateImages,
  outputPrefix,
  outputFolders,
  onNavigateOutput,
  onChanged,
}: {
  queue: Track2InferenceQueueItem[];
  queueStatus: { started: boolean; worker_alive: boolean };
  weightsPrefix: string;
  weightsFolders: string[];
  weightFiles: string[];
  onNavigateWeights: (prefix: string) => void;
  imagesPrefix: string;
  imagesFolders: string[];
  onNavigateImages: (prefix: string) => void;
  outputPrefix: string;
  outputFolders: string[];
  onNavigateOutput: (prefix: string) => void;
  onChanged: (message: string) => void;
}) {
  const [rows, setRows] = useState<Track2MetricsRow[] | null>(null);
  const [selectedKey, setSelectedKey] = useState("");
  const [useS3, setUseS3] = useState(false);
  const [s3Size, setS3Size] = useState<string>(SIZES[0]);
  const [weightsKey, setWeightsKey] = useState("");
  const [notes, setNotes] = useState("");
  const [imagesFolder, setImagesFolder] = useState("");
  const [outputFolder, setOutputFolder] = useState("");
  const [comment, setComment] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getTrack2Metrics()
      .then((res) => setRows(res.rows))
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load registry versions."));
  }, []);

  const rowsBySize: { size: string; registry: string; group_name: string; rows: Track2MetricsRow[] }[] = [];
  for (const row of rows ?? []) {
    const group = rowsBySize[rowsBySize.length - 1];
    // Split on the GROUP, not the size. yolo11s-detector and
    // yolo11s-open-detector are both size "s", so grouping by size alone
    // produced two identical "SIZE S" headers -- and one "SIZE" with no
    // letter for the legacy group, which has no size at all.
    if (group && group.group_name === row.group_name) group.rows.push(row);
    else rowsBySize.push({ size: row.size, registry: row.registry, group_name: row.group_name, rows: [row] });
  }

  const selectedRow = rows?.find((r) => rowKey(r) === selectedKey);
  const canAdd = Boolean(imagesFolder) && Boolean(outputFolder) && (useS3 ? Boolean(weightsKey) : Boolean(selectedRow));

  async function handleAdd() {
    setBusy(true);
    setError(null);
    try {
      const res = await addTrack2InferenceQueueItem({
        size: useS3 ? s3Size : selectedRow!.size,
        weights_kind: useS3 ? "s3" : "registry",
        model_package_arn: useS3 ? undefined : selectedRow!.model_package_arn,
        weights_key: useS3 ? weightsKey : undefined,
        notes: useS3 && notes.trim() ? notes.trim() : undefined,
        images_folder: imagesFolder,
        output_folder: outputFolder,
        comment: comment.trim() ? comment.trim() : undefined,
      });
      // Adding only queues it -- it stays "queued" until Start Queue is
      // clicked, so several jobs can be lined up (by this person or
      // others) before any of them actually starts deploying.
      onChanged(res.message);
      setSelectedKey("");
      setWeightsKey("");
      setNotes("");
      setComment("");
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Add to queue failed.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRemove(id: string) {
    await removeTrack2InferenceQueueItem(id);
    onChanged("Removed from queue.");
  }

  async function handleStart() {
    try {
      const res = await startTrack2InferenceQueue();
      onChanged(res.message);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Start failed.");
    }
  }

  async function handleStop() {
    const res = await stopTrack2InferenceQueue();
    onChanged(res.message);
  }

  const pending = queue.filter((i) => i.status === "queued" || i.status === "deploying" || i.status === "running");
  // "started" alone must not disable Start. If the worker thread died --
  // the app restarted mid-run, say -- the flag stays true while nothing is
  // actually processing, and a disabled Start leaves the queue stuck with
  // no way out from the UI. Only a genuinely live worker should block it.
  const queueRunning = queueStatus.started && queueStatus.worker_alive;
  const canStartQueue = !queueRunning && pending.some((i) => i.status === "queued");
  const history = queue.filter((i) => i.status === "done" || i.status === "failed");

  return (
    <Card
      title="Inference queue"
      caption="Each queued job deploys its model (swapping out whatever's live), runs every image in its input folder, and moves to the next -- one at a time."
    >
      {error && (
        <div className="mb-3 flex items-center gap-2 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-xs text-critical">
          <AlertCircle size={14} />
          {error}
        </div>
      )}

      <div className="flex flex-col gap-3">
        <div className="flex gap-2">
          {[
            { key: false, label: "Registry version" },
            { key: true, label: "Weights not in the registry" },
          ].map((opt) => (
            <button
              key={String(opt.key)}
              type="button"
              onClick={() => setUseS3(opt.key)}
              className={`rounded-sm border px-3 py-1.5 text-xs font-medium ${
                useS3 === opt.key ? "border-accent bg-accent-subtle text-ink" : "border-black/15 text-ink-secondary"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {!useS3 ? (
          !rows ? (
            <div className="flex items-center gap-2 text-sm text-ink-muted">
              <Loader2 size={15} className="animate-spin" />
              Loading...
            </div>
          ) : rows.length === 0 ? (
            <p className="text-sm text-ink-muted">No registered versions yet.</p>
          ) : (
            <div className="overflow-x-auto rounded-sm border border-hairline">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-hairline text-xs text-ink-muted">
                    <th className="py-1.5 pr-2 font-medium"></th>
                    <th className="py-1.5 pr-4 font-medium">Version</th>
                    <th className="py-1.5 pr-4 font-medium">Status</th>
                    <th className="tabular py-1.5 font-medium">mAP50</th>
                  </tr>
                </thead>
                {rowsBySize.map(({ size, registry, group_name, rows: sizeRows }) => (
                  <tbody key={group_name}>
                    <tr>
                      <td colSpan={4} className="bg-page px-2 py-1.5 text-xs font-semibold uppercase tracking-wide text-ink-secondary">
                        {/* The generation comes off the group name --
                            `yolo26s-detector` is a production group too.
                            Writing YOLO11 here labelled every production
                            row as generation 11 whatever it was. */}
                        {registry === "production"
                          ? <>
                              Production &middot;{" "}
                              YOLO{GROUP_GENERATION.exec(group_name)?.[1] ?? "?"} &middot;{" "}
                              Size {size}
                            </>
                          : <>Custom &middot; {group_name} <span className="font-normal normal-case tracking-normal text-ink-muted">(own classes -- detections come back as that model's class indices)</span></>}
                      </td>
                    </tr>
                    {sizeRows.map((row) => {
                      const key = rowKey(row);
                      const disabled = row.status !== "Approved";
                      return (
                        <tr key={key} className="border-b border-hairline last:border-0">
                          <td className="py-1.5 pr-2">
                            <input
                              type="checkbox"
                              checked={selectedKey === key}
                              disabled={disabled}
                              onChange={(e) => setSelectedKey(e.target.checked ? key : "")}
                            />
                          </td>
                          <td className="tabular py-1.5 pr-4">{row.version}</td>
                          <td className="py-1.5 pr-4">
                            <Badge tone={approvalTone(row.status)} label={row.status} />
                          </td>
                          <td className="tabular py-1.5">{row.metrics?.mAP50?.toFixed(3) ?? "--"}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                ))}
              </table>
            </div>
          )
        ) : (
          <div className="flex flex-col gap-2">
            <label className="block">
              {/* No generation in the label. These are loose weights in
                  S3 -- nothing records which generation they are, and
                  "YOLO11-S" on a yolo26 checkpoint is a worse answer
                  than the size alone. */}
              <span className="mb-1 block text-xs text-ink-muted">Size</span>
              <select
                className="w-full max-w-xs rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                value={s3Size}
                onChange={(e) => setS3Size(e.target.value)}
              >
                {SIZES.map((s) => (
                  <option key={s} value={s}>{s.toUpperCase()}</option>
                ))}
              </select>
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">
                Notes <span className="text-ink-muted/70">(optional -- what is this checkpoint?)</span>
              </span>
              <textarea
                className="w-full resize-y rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
                rows={2}
                maxLength={500}
                placeholder="e.g. epoch 80 checkpoint, trained on the 2026-08 hazard-images export with extra night-shift crops"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
              />
            </label>
            <Track2S3FolderBrowser
              prefix={weightsPrefix}
              folders={weightsFolders}
              onNavigate={onNavigateWeights}
              files={weightFiles}
              onSelectFile={setWeightsKey}
              selectedFile={weightsKey}
            />
          </div>
        )}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
        <div className="min-w-0">
          <span className="mb-1 block text-xs text-ink-muted">Images folder</span>
          <Track2S3FolderBrowser
            prefix={imagesPrefix}
            folders={imagesFolders}
            onNavigate={onNavigateImages}
            onSelect={setImagesFolder}
            selected={imagesFolder}
            allowCreate
          />
        </div>

        <div className="min-w-0">
          <span className="mb-1 block text-xs text-ink-muted">Output folder</span>
          <Track2S3FolderBrowser
            prefix={outputPrefix}
            folders={outputFolders}
            onNavigate={onNavigateOutput}
            onSelect={setOutputFolder}
            selected={outputFolder}
            allowCreate
          />
        </div>
      </div>

      <label className="mt-4 block">
        <span className="mb-1 block text-xs text-ink-muted">Comment <span className="text-ink-muted/70">(optional -- notes about this run)</span></span>
        <textarea
          className="w-full resize-y rounded-md border border-black/15 bg-page px-3 py-2 text-sm"
          rows={2}
          maxLength={500}
          placeholder="e.g. re-checking the 2026-09-12 night-shift batch after the helmet-box fix"
          value={comment}
          onChange={(e) => setComment(e.target.value)}
        />
      </label>

      <button
        type="button"
        onClick={handleAdd}
        disabled={busy || !canAdd}
        className="mt-4 flex items-center justify-center gap-1.5 rounded-md bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-40"
      >
        {busy && <Loader2 size={14} className="animate-spin" />}
        Add to Queue
      </button>
      <p className="mt-1.5 text-xs text-ink-muted">
        {queueRunning
          ? "The queue is running -- anything added now runs after the job in flight, reusing the endpoint that's already up."
          : "Line up as many as you need, then press Start Queue. They run one at a time on a single endpoint, which is torn down once the last one finishes."}
      </p>

      <div className="mt-5 border-t border-hairline pt-4">
        <div className="mb-2 flex items-center justify-between">
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-muted">Queue ({pending.length})</p>
          <div className="flex gap-2">
            <button
              type="button"
              onClick={handleStart}
              disabled={!canStartQueue}
              className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary disabled:opacity-40"
            >
              Start Queue
            </button>
            <button
              type="button"
              onClick={handleStop}
              disabled={!queueStatus.started}
              className="rounded-md bg-red-600 px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
            >
              Stop Queue
            </button>
          </div>
        </div>

        {pending.length === 0 && history.length === 0 && <p className="text-sm text-ink-muted">Nothing queued yet.</p>}

        {pending.length > 0 && (
          <div className="mb-3 flex flex-col gap-2">
            {pending.map((item) => (
              <div key={item.id} className="flex items-center justify-between gap-3 rounded-sm border border-hairline bg-page p-2.5">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <Badge tone={inferenceQueueStatusTone(item.status)} label={item.status} />
                    <span className="tabular text-sm text-ink">{item.weights_label}</span>
                  </div>
                  <p className="tabular mt-0.5 truncate text-xs text-ink-muted">
                    {item.images_folder} &rarr; {item.output_folder}
                  </p>
                  {item.notes && <p className="mt-0.5 whitespace-pre-wrap text-xs text-ink-secondary">{item.notes}</p>}
                  {item.comment && <p className="mt-0.5 whitespace-pre-wrap text-xs text-ink-secondary">{item.comment}</p>}
                  {/* No fixed ETA is shown -- a deploy's length depends on
                      whether this exact model is already loaded elsewhere
                      (seconds) or needs a cold container start (minutes), so
                      a made-up number would just be wrong half the time.
                      Elapsed time is the honest version: it's real, and it
                      keeps ticking on every 3s poll while a job is active. */}
                  {item.started_at && (item.status === "deploying" || item.status === "running") && (
                    <p className="tabular mt-0.5 text-xs text-ink-muted">
                      Started {formatTimestamp(item.started_at)} &middot;{" "}
                      {item.status === "deploying" ? "deploying" : "running"} for {formatElapsed(item.started_at)}
                    </p>
                  )}
                </div>
                {item.status === "queued" && (
                  <button
                    type="button"
                    onClick={() => handleRemove(item.id)}
                    className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
                  >
                    Delete
                  </button>
                )}
              </div>
            ))}
          </div>
        )}

        {history.length > 0 && (
          <div>
            <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">History</p>
            <div className="flex flex-col gap-2">
              {history.map((item) => (
                <div key={item.id} className="flex items-start justify-between gap-3 rounded-sm border border-hairline bg-page p-2.5">
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-2">
                      <Badge tone={inferenceQueueStatusTone(item.status)} label={item.status} />
                      <span className="tabular text-sm text-ink">{item.weights_label}</span>
                    </div>
                    <p className="tabular mt-0.5 truncate text-xs text-ink-muted">
                      {item.images_folder} &rarr; {item.output_folder}
                    </p>
                    {item.notes && <p className="mt-0.5 whitespace-pre-wrap text-xs text-ink-secondary">{item.notes}</p>}
                    {item.comment && <p className="mt-0.5 whitespace-pre-wrap text-xs text-ink-secondary">{item.comment}</p>}
                    {item.started_at && (
                      <p className="tabular mt-0.5 text-xs text-ink-muted">Started {formatTimestamp(item.started_at)}</p>
                    )}
                    {item.completed_at && (
                      <p className="tabular mt-0.5 text-xs text-ink-muted">Completed {formatTimestamp(item.completed_at)}</p>
                    )}
                    {item.status === "done" && item.result && (
                      <>
                        <p className="mt-1 text-xs text-ink-secondary">
                          Processed {item.result.processed} of {item.result.total} image(s).
                          {item.result.errors.length > 0 && ` ${item.result.errors.length} error(s).`}
                          {item.started_at &&
                            item.completed_at &&
                            ` Took ${formatElapsed(item.started_at, new Date(item.completed_at))}.`}
                        </p>
                        {item.result.errors.length > 0 && (
                          <ul className="mt-1.5 space-y-1 rounded-sm border border-critical/20 bg-critical-subtle p-2 text-xs text-critical">
                            {item.result.errors.map((e, i) => (
                              <li key={i}>{e}</li>
                            ))}
                          </ul>
                        )}
                      </>
                    )}
                    {item.status === "failed" && item.error && (
                      <p className="mt-1 text-xs text-critical">
                        {item.error}
                        {item.started_at && item.completed_at && (
                          <span className="text-ink-muted"> (after {formatElapsed(item.started_at, new Date(item.completed_at))})</span>
                        )}
                      </p>
                    )}
                    {item.status === "done" && <RunLineage outputFolder={item.output_folder} runId={item.id} />}
                  </div>
                  <button
                    type="button"
                    onClick={() => handleRemove(item.id)}
                    className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
                  >
                    Delete
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

export function Track2DeployPage() {
  const [data, setData] = useState<Track2DeployResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  // SageMaker's own describe call can lag briefly right after an update
  // call is accepted -- still reporting the previous model/InService
  // status for a few seconds before it catches up. Rather than trust the
  // very next GET, keep polling for a short window after any deploy/
  // spin-up action so the card settles onto the real current state
  // instead of freezing on stale data with nothing to trigger a refresh.
  const [forcePollUntil, setForcePollUntil] = useState(0);

  function load() {
    getTrack2Deploy()
      .then(setData)
      .catch((err) => setError(err instanceof ApiError ? err.detail : "Couldn't load deploy status."));
  }

  useEffect(load, []);

  useEffect(() => {
    if (!data) return;
    const queueBusy =
      data.inference_queue_status.started ||
      data.inference_queue.some((i) => i.status === "deploying" || i.status === "running");
    const phase = phaseOf(data.deployed);
    // The endpoint moves on its own schedule, independent of both the
    // queue and the component: it is built before any job can run and
    // torn down after the queue has already drained. Leaving it out of
    // this meant polling stopped exactly when the endpoint was mid-
    // transition with nothing else in flight -- the panel froze on
    // "running, billing" through the whole teardown and never noticed it
    // had gone. Which is the one state you cannot afford to be stale
    // about, since it is the one that costs money.
    const busy = isTransitional(phase) || phase === "inferencing" || queueBusy || endpointInFlight(data.endpoint);
    const forcePolling = Date.now() < forcePollUntil;
    if (!busy && !forcePolling) return;
    // 3s while the endpoint is mid-transition: those are the states a
    // person is actively watching (waiting to start, waiting for billing
    // to stop), and 10s there reads as frozen.
    const watchClosely = (forcePolling && !busy) || endpointInFlight(data.endpoint);
    const id = setInterval(load, watchClosely ? 3000 : 10000);
    return () => clearInterval(id);
  }, [data, forcePollUntil]);

  function handleDeployed(msg: string) {
    setMessage(msg);
    setForcePollUntil(Date.now() + 30000);
    load();
  }

  // Shared "weights not in the registry" browser location -- same
  // convention as the training base-model-weights browser (an arbitrary
  // bucket-relative path).
  const [weightsPrefix, setWeightsPrefix] = useState("");
  const [weightsFolders, setWeightsFolders] = useState<string[]>([]);
  const [weightFiles, setWeightFiles] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    getTrack2DeployBrowse(weightsPrefix)
      .then((res) => {
        if (!cancelled) {
          setWeightsFolders(res.folders);
          setWeightFiles(res.weight_files);
        }
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [weightsPrefix]);

  // Separate browsing locations for batch inference's input images
  // folder and output folder -- distinct from the weights browser above,
  // since they're picked at a different point in the flow.
  const [imagesPrefix, setImagesPrefix] = useState("");
  const [imagesFolders, setImagesFolders] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    getTrack2DeployBrowse(imagesPrefix)
      .then((res) => {
        if (!cancelled) setImagesFolders(res.folders);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [imagesPrefix]);

  const [outputPrefix, setOutputPrefix] = useState("");
  const [outputFolders, setOutputFolders] = useState<string[]>([]);
  useEffect(() => {
    let cancelled = false;
    getTrack2DeployBrowse(outputPrefix)
      .then((res) => {
        if (!cancelled) setOutputFolders(res.folders);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [outputPrefix]);

  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <Rocket size={20} className="text-accent" strokeWidth={2} />
        Track2 Deploy &amp; Inference
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        Only one model is live at a time -- queuing a job below deploys its
        model (an Approved registry version, or your own S3 weights),
        replacing whatever's currently active on the shared endpoint, then
        runs inference over its input folder.
      </p>

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
        <div className="flex flex-col gap-6">
          <EndpointStatusPanel
            deployed={data.deployed}
            endpoint={data.endpoint}
            message={message}
            onChanged={handleDeployed}
          />
          <InferenceQueueSection
            queue={data.inference_queue}
            queueStatus={data.inference_queue_status}
            weightsPrefix={weightsPrefix}
            weightsFolders={weightsFolders}
            weightFiles={weightFiles}
            onNavigateWeights={setWeightsPrefix}
            imagesPrefix={imagesPrefix}
            imagesFolders={imagesFolders}
            onNavigateImages={setImagesPrefix}
            outputPrefix={outputPrefix}
            outputFolders={outputFolders}
            onNavigateOutput={setOutputPrefix}
            onChanged={handleDeployed}
          />
        </div>
      )}
    </div>
  );
}
