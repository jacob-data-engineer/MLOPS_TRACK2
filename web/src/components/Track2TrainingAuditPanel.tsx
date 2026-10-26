import { ChevronDown, ChevronRight, Loader2 } from "lucide-react";
import { useCardOpen } from "../lib/cardState";
import { useCallback, useEffect, useState } from "react";
import { Badge, type BadgeTone } from "./Badge";
import { Track2S3FolderBrowser } from "./Track2S3FolderBrowser";
import {
  ApiError,
  deleteTrack2TrainingAudit,
  exportTrack2TrainingAuditCsv,
  exportTrack2Weights,
  getTrack2Export,
  getTrack2TrainingAudit,
  getTrack2TrainingAuditWeightsUrl,
  type Track2TrainingAudit,
  type Track2TrainingAuditDataset,
} from "../lib/api";
import { ConfirmDialog } from "./ConfirmDialog";

/** Finished runs, under the queue that produced them -- the same shape as
 * Deploy & Inference: what is queued above, what has already run below.
 *
 * Each row is the full record of what that run used: both image folders
 * and the file names in them, every hyperparameter, the notes, which
 * model version came out of it, and when it ran. Rows are written at
 * submit time and completed by the server when the run ends, so a row
 * still in flight legitimately has no end time, model or metrics -- shown
 * as "--" rather than hidden, since "not yet" and "never recorded" have
 * to stay distinguishable in a record like this.
 */
export function Track2TrainingHistory({ reloadKey = 0 }: { reloadKey?: number }) {
  const [audits, setAudits] = useState<Track2TrainingAudit[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const res = await getTrack2TrainingAudit(50);
      setAudits(res.audits);
      setError(null);
    } catch {
      setError("Could not load the training history.");
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, reloadKey]);

  if (audits !== null && audits.length === 0 && !error) return null;

  return (
    <div className="mt-4">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">History</p>
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {audits === null ? (
        <p className="text-sm text-ink-muted">Loading...</p>
      ) : (
        <div className="flex flex-col gap-2">
          {audits.map((audit) => (
            <AuditRow key={audit.id} audit={audit} onDeleted={load} />
          ))}
        </div>
      )}
    </div>
  );
}

function statusTone(status: string): BadgeTone {
  if (status === "Succeeded") return "confirmed";
  if (status === "Failed" || status === "Stopped") return "rejected";
  return "pending";
}

/** Local time, and the date too -- a bare clock time is ambiguous on a
 * record that can be weeks old. */
function formatTime(value: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

/** How long the run took, or has taken so far.
 *
 * This is the PIPELINE EXECUTION's wall time -- `started_at` and
 * `ended_at` are written from `describe_pipeline_execution` -- so it
 * covers waiting for GPU capacity, training, evaluation and
 * registration. It is NOT the billable GPU time, and the gap is real:
 * on 2026-10-15 an execution started at 18:41:07 and its training job
 * did not get an instance until 18:44:04, three minutes it was not
 * charged for.
 *
 * `running` makes an unfinished run count up instead of reading "--",
 * which is what it did for the whole twenty minutes anybody actually
 * wants to know.
 */
function formatDuration(started: string | null, ended: string | null,
                        running = false): string {
  if (!started) return "--";
  const from = new Date(started).getTime();
  const to = ended ? new Date(ended).getTime() : (running ? Date.now() : NaN);
  const ms = to - from;
  if (!Number.isFinite(ms) || ms < 0) return "--";
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/** Statuses where the run has not stopped, so the clock is still going. */
const STILL_GOING = new Set(["Executing", "Stopping", "InProgress"]);

function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined) return "--";
  return typeof value === "number" ? value.toFixed(4) : String(value);
}

/** Exported so the registry tables can show the full run behind a version
 * without rebuilding this card. The history panel and the per-model detail
 * must not drift: they describe the same run, and two renderings of the
 * same submitted-vs-actual comparison would be worse than one. */
export function AuditRow({ audit, onDeleted }: { audit: Track2TrainingAudit; onDeleted: () => void }) {
  const [open, setOpen] = useCardOpen("training_audit", false);
  const [deleting, setDeleting] = useState(false);
  // Deleting a run deletes the registered model with it. The dialog
  // names the version that goes, because the row it is on shows a date
  // and a status, not what it produced.
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [downloading, setDownloading] = useState(false);
  // Copying the .pt into a bucket folder, as opposed to downloading it.
  const [exportOpen, setExportOpen] = useState(false);
  const [exportPrefix, setExportPrefix] = useState("");
  const [exportFolders, setExportFolders] = useState<string[]>([]);
  const [exportTarget, setExportTarget] = useState("");
  const [exportMessage, setExportMessage] = useState("");
  const [copying, setCopying] = useState(false);

  // Only while the picker is open: listing the bucket for every collapsed
  // run in the history would be dozens of calls for nothing.
  useEffect(() => {
    if (!exportOpen) return;
    let cancelled = false;
    getTrack2Export(exportPrefix)
      .then((res) => !cancelled && setExportFolders(res.folders))
      .catch(() => undefined);
    return () => { cancelled = true; };
  }, [exportOpen, exportPrefix]);

  async function handleCopyToFolder() {
    if (!audit.model?.model_package_arn || !exportTarget) return;
    setCopying(true);
    setExportMessage("");
    try {
      const res = await exportTrack2Weights(audit.model.model_package_arn, exportTarget);
      setExportMessage(res.message);
    } catch (err) {
      setExportMessage(err instanceof ApiError ? err.detail : "Export failed.");
    } finally {
      setCopying(false);
    }
  }

  /** Follows a presigned S3 link rather than proxying the file: a
   * checkpoint is 19MB+ and the API container is on one small box. */
  async function handleExportWeights() {
    setDownloading(true);
    setDeleteError(null);
    try {
      const { url } = await getTrack2TrainingAuditWeightsUrl(audit.id);
      const link = document.createElement("a");
      link.href = url;
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setDeleteError(
        err instanceof ApiError ? err.detail
          : err instanceof Error ? err.message : "Could not download the weights.",
      );
    } finally {
      setDownloading(false);
    }
  }

  /** Fetched rather than linked: the endpoint needs the bearer token, so
   * a plain <a href> would get a 401 and download nothing. */
  async function handleExport() {
    setExporting(true);
    setDeleteError(null);
    try {
      const blob = await exportTrack2TrainingAuditCsv(audit.id);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `training-run-${audit.submitted_at.slice(0, 10)}-${audit.id}.csv`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  }

  async function handleDelete() {
    setDeleting(true);
    setDeleteError(null);
    try {
      await deleteTrack2TrainingAudit(audit.id);
      setConfirmDelete(false);
      onDeleted();
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Delete failed.");
      setDeleting(false);
    }
  }

  return (
    <div className="rounded-sm border border-hairline bg-page">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          {open ? <ChevronDown size={14} className="shrink-0 text-ink-muted" /> : <ChevronRight size={14} className="shrink-0 text-ink-muted" />}
          <Badge tone={statusTone(audit.status)} label={audit.status} />
          {/* The MODEL NAME, not the generation/size pair.
              For a custom-class run `generation` is the literal string
              "custom" and `size` holds the architecture, so the old
              template rendered "YOLOcustom-YOLO11S" -- which names
              neither the model nor anything a person chose. The Class
              the run belongs to is what the history is scanned for, and
              it was only in the expanded detail.

              Production runs keep the old form: they have no detector,
              and their generation/size really are 11 and s. */}
          <span className="tabular shrink-0 text-sm text-ink">
            {audit.detector_slug ? (
              <>
                {audit.detector_slug}
                {audit.size && (
                  <span className="text-ink-muted"> &middot; {audit.size}</span>
                )}
              </>
            ) : (
              `YOLO${audit.generation}-${audit.size.toUpperCase()}`
            )}
          </span>
          {audit.model && (
            <span className="tabular shrink-0 text-xs text-ink-secondary">v{audit.model.version}</span>
          )}
          <span className="tabular truncate text-xs text-ink-muted">{formatTime(audit.submitted_at)}</span>
          {/* On the ROW, not only inside the expanded detail. "How long
              did it take" is the question this list is read to answer,
              and it was one click away on every single row. */}
          {audit.started_at && (
            <span className="tabular shrink-0 text-xs text-ink-secondary">
              {formatDuration(audit.started_at, audit.ended_at,
                              STILL_GOING.has(audit.status))}
            </span>
          )}
        </button>
        {/* Every field and EVERY file name, not the capped sample shown
            below -- the export reads the S3 manifest. */}
        <button
          type="button"
          onClick={handleExport}
          disabled={exporting}
          title="Export this run's full details, including every file name, as CSV"
          className="flex shrink-0 items-center gap-1.5 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
        >
          {exporting && <Loader2 size={12} className="animate-spin" />}
          Export training record as CSV
        </button>
        {/* Only where a model actually came out of the run. */}
        {audit.model && (
          <button
            type="button"
            onClick={handleExportWeights}
            disabled={downloading}
            title="Download the .pt this run produced"
            className="flex shrink-0 items-center gap-1.5 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
          >
            {downloading && <Loader2 size={12} className="animate-spin" />}
            Download .pt
          </button>
        )}
        {audit.model && (
          <button
            type="button"
            onClick={() => setExportOpen((open) => !open)}
            title="Copy the .pt into a folder in the bucket"
            className="shrink-0 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04]"
          >
            {exportOpen ? "Cancel export" : "Export .pt to S3"}
          </button>
        )}
        <button
          type="button"
          onClick={() => setConfirmDelete(true)}
          disabled={deleting}
          title={audit.model ? "Delete this run and the model version it registered" : "Delete this run"}
          className="flex shrink-0 items-center gap-1.5 rounded-sm border border-black/15 px-2 py-1 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
        >
          {deleting && <Loader2 size={12} className="animate-spin" />}
          Delete
        </button>
      </div>
      {deleteError && <p className="px-3 pb-2 text-xs text-critical">{deleteError}</p>}

      <ConfirmDialog
        open={confirmDelete}
        title={audit.model ? "Delete this run and its model?" : "Delete this run?"}
        destructive
        busy={deleting}
        confirmLabel={audit.model ? "Delete the run and the model" : "Delete the run"}
        cancelLabel="Keep it"
        onCancel={() => setConfirmDelete(false)}
        onConfirm={() => void handleDelete()}
        body={audit.model ? (
          <>
            <p>
              Removes this run from the history <strong>and deletes the
              registered version</strong>{" "}
              <span className="tabular">
                {audit.model.group} v{audit.model.version}
              </span>{" "}
              with it.
            </p>
            <p className="mt-2">
              Anything already deployed from that version keeps running.
              What goes is the record of how it was trained, so nothing can
              be rolled back to it or continued from it.
            </p>
            <p className="mt-2">This cannot be undone.</p>
          </>
        ) : (
          <p>
            Removes this run from the history. It never registered a model,
            so no weights and no registry version are affected.
          </p>
        )}
      />

      {exportOpen && audit.model && (
        <div className="border-t border-hairline px-3 py-3">
          <p className="mb-2 text-xs text-ink-secondary">
            Copy{" "}
            <span className="tabular">{audit.model.group} v{audit.model.version}</span> as{" "}
            <span className="tabular">selected.pt</span> into a folder. Use "New folder
            name" to create one that does not exist yet; any existing{" "}
            <span className="tabular">selected.pt</span> there is archived rather than
            overwritten.
          </p>
          <Track2S3FolderBrowser
            prefix={exportPrefix}
            folders={exportFolders}
            onNavigate={setExportPrefix}
            onSelect={setExportTarget}
            selected={exportTarget}
            allowCreate
          />
          <div className="mt-2 flex items-center gap-2">
            <button
              type="button"
              disabled={copying || !exportTarget}
              onClick={handleCopyToFolder}
              className="rounded-md bg-accent px-3 py-1.5 text-xs font-medium text-white disabled:opacity-40"
            >
              {copying ? "Exporting..." : "Export here"}
            </button>
            {exportTarget && (
              <span className="tabular text-xs text-ink-muted">
                -&gt; {exportTarget}/selected.pt
              </span>
            )}
          </div>
          {exportMessage && (
            <p className="tabular mt-2 break-all text-xs text-ink-secondary">{exportMessage}</p>
          )}
        </div>
      )}

      {open && (
        <div className="border-t border-hairline px-3 py-3">
          <div className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-4">
            <Field label="Started" value={formatTime(audit.started_at)} />
            <Field label="Ended" value={formatTime(audit.ended_at)} />
            <Field
              label="Duration (whole run)"
              value={formatDuration(audit.started_at, audit.ended_at,
                                    STILL_GOING.has(audit.status))}
            />
            <Field label="Submitted by" value={audit.source === "queue" ? "Training queue" : "Started manually"} />
            <Field label="Epochs" value={String(audit.parameters.epochs)} />
            <Field label="Image size" value={String(audit.parameters.img_size)} />
            <Field label="Batch" value={String(audit.parameters.batch)} />
            <Field label="Seed" value={String(audit.parameters.seed)} />
          </div>

          {/* The pipeline does not always run what was asked for, and an
              audit that only showed the request would describe a run that
              never happened. Only rendered when they actually differ. */}
          {audit.actual && <ActualRunNotice audit={audit} />}

          <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
            <DatasetBlock title="Training images" dataset={audit.train_dataset} />
            <DatasetBlock
              title="Evaluation images"
              dataset={audit.eval_dataset}
              warning={audit.eval_is_train_dataset ? "Same as the training set -- these metrics are not held out." : undefined}
            />
          </div>

          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-4">
            {/* The Model Name, outright. The registry group implies it --
                "yolo11s-helmet-detector v9" -- but only to somebody who
                knows how those names are built, and a run that never
                reached registration has no group to read it from. */}
            <Field label="Model Name" value={audit.detector_slug || "--"} />
            <Field
              label="Model version"
              value={audit.model ? `${audit.model.group} v${audit.model.version}` : "--"}
            />
            <Field label="Approval" value={audit.model?.approval_status || "--"} />
            <Field
              label="Base model"
              value={audit.base_model.folder || "Registry's latest Approved"}
            />
            <Field label="Build" value={audit.build_id || "--"} />
          </div>

          {/* The frozen dataset, when there was one. It used to reach the
              history only inside the free-text notes, which made it a
              habit rather than a record: nothing guaranteed it was
              there, and nothing could be filtered on it. */}
          {audit.snapshot?.snapshot_id && (
            <div className="mt-3 rounded-sm border border-hairline bg-page px-2.5 py-2 text-xs">
              <p className="text-ink-muted">Frozen snapshot</p>
              <p className="tabular mt-0.5 text-ink">
                <span className="font-medium">{audit.snapshot.snapshot_id}</span>
                {audit.snapshot.created_at && (
                  <> &middot; frozen {formatTime(audit.snapshot.created_at)}</>
                )}
                {audit.snapshot.files
                  ? ` · ${audit.snapshot.files.toLocaleString()} images`
                  : ""}
                {audit.snapshot.pinned
                  ? " pinned by name"
                  : " (names week folders -- not repeatable)"}
              </p>
              {audit.snapshot.note && (
                <p className="mt-0.5 text-ink-muted">{audit.snapshot.note}</p>
              )}
            </div>
          )}

          <p className="mt-3 mb-1 text-xs text-ink-muted">Metrics</p>
          {audit.metrics ? (
            <div className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-5">
              <Field label="mAP50" value={formatMetric(audit.metrics["mAP50"])} />
              <Field label="mAP50-95" value={formatMetric(audit.metrics["mAP50-95"])} />
              <Field label="Precision" value={formatMetric(audit.metrics["precision"])} />
              <Field label="Recall" value={formatMetric(audit.metrics["recall"])} />
              {/* The full-frame evaluator reports f1, the older one
                  accuracy -- show whichever this run actually produced. */}
              {audit.metrics["f1"] !== null && audit.metrics["f1"] !== undefined ? (
                <Field label="F1" value={formatMetric(audit.metrics["f1"])} />
              ) : (
                <Field label="Accuracy" value={formatMetric(audit.metrics["accuracy"])} />
              )}
            </div>
          ) : (
            <p className="text-xs text-ink-muted">
              {audit.status === "Succeeded"
                ? "No metrics recorded for this run."
                : "Recorded once the run finishes and registers a model."}
            </p>
          )}

          {audit.notes && (
            <>
              <p className="mt-3 mb-1 text-xs text-ink-muted">Notes</p>
              <p className="whitespace-pre-wrap text-xs text-ink">{audit.notes}</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** Shown only when the job that ran differs from what was submitted --
 * different hyperparameters, or a different dataset entirely. Silent when
 * they agree, so it reads as a warning rather than noise. */
function ActualRunNotice({ audit }: { audit: Track2TrainingAudit }) {
  const actual = audit.actual;
  if (!actual) return null;

  const requestedUri = audit.train_dataset.s3_uri.replace(/\/+$/, "");
  const actualUri = (actual.train_data_s3_uri || "").replace(/\/+$/, "");
  const differences: string[] = [];
  if (actual.epochs !== null && actual.epochs !== audit.parameters.epochs) {
    differences.push(`epochs ${audit.parameters.epochs} -> ${actual.epochs}`);
  }
  if (actual.batch !== null && actual.batch !== audit.parameters.batch) {
    differences.push(`batch ${audit.parameters.batch} -> ${actual.batch}`);
  }
  if (actual.img_size !== null && actual.img_size !== audit.parameters.img_size) {
    differences.push(`image size ${audit.parameters.img_size} -> ${actual.img_size}`);
  }
  if (actual.seed !== null && actual.seed !== audit.parameters.seed) {
    differences.push(`seed ${audit.parameters.seed} -> ${actual.seed}`);
  }
  const datasetDiffers = Boolean(actualUri) && Boolean(requestedUri) && actualUri !== requestedUri;
  if (!differences.length && !datasetDiffers) return null;

  return (
    <div className="mt-3 rounded-sm border border-critical/30 bg-critical-subtle p-2.5 text-xs text-critical">
      <p className="font-semibold">The pipeline did not run what was submitted.</p>
      {differences.length > 0 && <p className="tabular mt-1">Changed: {differences.join(" · ")}</p>}
      {datasetDiffers && (
        <p className="tabular mt-1 break-all">
          Trained on {actualUri} instead of the folder chosen above.
        </p>
      )}
      {actual.training_job_name && (
        <p className="tabular mt-1 break-all text-ink-secondary">
          Training job: {actual.training_job_name}
          {actual.instance_type && ` on ${actual.instance_type}`}
        </p>
      )}
    </div>
  );
}

function Field({ label, value }: { label: string; value: string }) {
  return (
    <span className="min-w-0">
      <span className="text-ink-muted">{label}: </span>
      <span className="break-words" title={value}>{value}</span>
    </span>
  );
}

function DatasetBlock({
  title,
  dataset,
  warning,
}: {
  title: string;
  dataset: Track2TrainingAuditDataset;
  warning?: string;
}) {
  const [showFiles, setShowFiles] = useState(false);
  return (
    <div className="min-w-0 rounded-sm border border-hairline bg-surface p-2.5">
      <p className="mb-1 text-xs text-ink-muted">{title}</p>
      {/* The folder someone actually picked, not the assembled copy of
          it. `data/assembled/<date>` is a staging name nobody chose, it
          is deleted once the run finishes, and showing it as the
          dataset's identity buries the only path that means anything.
          It stays in the tooltip for anyone who needs it, and is the
          visible line only when there is no origin to show -- a folder
          that was not assembled IS its own origin. */}
      <p
        className="tabular break-all text-xs text-ink"
        title={dataset.folder ? `assembled at ${dataset.folder}` : dataset.s3_uri}
      >
        {dataset.source_folders && dataset.source_folders.length > 0
          ? dataset.source_folders.join(", ")
          : dataset.folder || "--"}
      </p>
      <p className="tabular mt-1 text-xs text-ink-secondary">
        {dataset.file_count} image{dataset.file_count === 1 ? "" : "s"}
        {dataset.truncated && ` (showing the first ${dataset.file_names.length} names)`}
      </p>
      {warning && <p className="mt-1 text-xs text-critical">{warning}</p>}
      {dataset.file_names.length > 0 && (
        <>
          <button
            type="button"
            onClick={() => setShowFiles((v) => !v)}
            className="mt-1.5 text-xs text-ink-secondary underline hover:text-ink"
          >
            {showFiles ? "Hide file names" : "Show file names"}
          </button>
          {showFiles && (
            <div className="tabular mt-1.5 max-h-48 space-y-2 overflow-y-auto rounded-sm border border-hairline bg-page p-2 text-xs text-ink">
              {/* Runs audited since the paths were resolved store them
                  whole, and resolved to the ORIGIN where one was recorded
                  -- the folder someone actually picked, not the dated
                  copy. Older rows stored a name relative to the folder,
                  and those would otherwise render bare as
                  "images/train/x.jpg", naming a file in no particular
                  dataset. A leading images/ or labels/ is what a relative
                  one always starts with; that is the whole test, and it
                  is the same rule the CSV export applies. */}
              {dataset.file_names.map((name) => (
                <p key={name} className="break-all">
                  {dataset.folder && /^(images|labels)\//.test(name)
                    ? `${dataset.folder}/${name}`
                    : name}
                </p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
