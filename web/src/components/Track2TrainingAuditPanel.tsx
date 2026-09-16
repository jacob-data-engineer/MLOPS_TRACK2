import { ChevronDown, ChevronRight, Loader2, RefreshCw } from "lucide-react";
import { useEffect, useState } from "react";
import { Badge, type BadgeTone } from "./Badge";
import { Card } from "./Card";
import { getTrack2TrainingAudit, type Track2TrainingAudit, type Track2TrainingAuditDataset } from "../lib/api";

/** Full record of what each training run actually used: both image
 * folders and the file names in them, every hyperparameter, the notes,
 * which model version came out of it, and when it ran.
 *
 * Rows are written at submit time and completed by the server once the
 * run ends, so a row in flight legitimately has no end time, model or
 * metrics yet -- shown as "--" rather than hidden, since "not yet" and
 * "never recorded" have to stay distinguishable in an audit.
 */
export function Track2TrainingAuditPanel() {
  const [audits, setAudits] = useState<Track2TrainingAudit[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function load() {
    setBusy(true);
    try {
      const res = await getTrack2TrainingAudit(50);
      setAudits(res.audits);
      setError(null);
    } catch {
      setError("Could not load the training audit.");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <Card
      title="Training audit"
      caption="One record per submitted run -- inputs, parameters, outputs and timings"
      action={
        <button
          type="button"
          onClick={load}
          disabled={busy}
          className="flex shrink-0 items-center gap-1.5 rounded-md border border-black/15 px-3 py-2 text-xs font-medium text-ink-secondary disabled:opacity-40"
        >
          {busy ? <Loader2 size={13} className="animate-spin" /> : <RefreshCw size={13} />}
          Refresh
        </button>
      }
    >
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {audits === null ? (
        <p className="text-sm text-ink-muted">Loading...</p>
      ) : audits.length === 0 ? (
        <p className="text-sm text-ink-muted">
          No runs recorded yet. Every run started from here is recorded from now on.
        </p>
      ) : (
        <div className="flex flex-col gap-2">
          {audits.map((audit) => (
            <AuditRow key={audit.id} audit={audit} />
          ))}
        </div>
      )}
    </Card>
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

function formatDuration(started: string | null, ended: string | null): string {
  if (!started || !ended) return "--";
  const ms = new Date(ended).getTime() - new Date(started).getTime();
  if (!Number.isFinite(ms) || ms < 0) return "--";
  const minutes = Math.floor(ms / 60000);
  const seconds = Math.round((ms % 60000) / 1000);
  if (minutes < 60) return `${minutes}m ${seconds}s`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

function formatMetric(value: number | null | undefined): string {
  if (value === null || value === undefined) return "--";
  return typeof value === "number" ? value.toFixed(4) : String(value);
}

function AuditRow({ audit }: { audit: Track2TrainingAudit }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="rounded-sm border border-hairline bg-page">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left hover:bg-black/[0.04]"
      >
        {open ? <ChevronDown size={14} className="shrink-0 text-ink-muted" /> : <ChevronRight size={14} className="shrink-0 text-ink-muted" />}
        <span className="tabular text-sm text-ink">
          YOLO{audit.generation}-{audit.size.toUpperCase()}
        </span>
        <span className="tabular truncate text-xs text-ink-muted">{formatTime(audit.submitted_at)}</span>
        {audit.model && (
          <span className="tabular shrink-0 text-xs text-ink-secondary">v{audit.model.version}</span>
        )}
        <span className="ml-auto shrink-0">
          <Badge tone={statusTone(audit.status)} label={audit.status} />
        </span>
      </button>

      {open && (
        <div className="border-t border-hairline px-3 py-3">
          <div className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-4">
            <Field label="Started" value={formatTime(audit.started_at)} />
            <Field label="Ended" value={formatTime(audit.ended_at)} />
            <Field label="Duration" value={formatDuration(audit.started_at, audit.ended_at)} />
            <Field label="Submitted by" value={audit.source === "queue" ? "Training queue" : "Started manually"} />
            <Field label="Epochs" value={String(audit.parameters.epochs)} />
            <Field label="Image size" value={String(audit.parameters.img_size)} />
            <Field label="Batch" value={String(audit.parameters.batch)} />
            <Field label="Seed" value={String(audit.parameters.seed)} />
          </div>

          <div className="mt-3 grid grid-cols-1 gap-3 lg:grid-cols-2">
            <DatasetBlock title="Training images" dataset={audit.train_dataset} />
            <DatasetBlock
              title="Evaluation images"
              dataset={audit.eval_dataset}
              warning={audit.eval_is_train_dataset ? "Same as the training set -- these metrics are not held out." : undefined}
            />
          </div>

          <div className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-4">
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

          <p className="mt-3 mb-1 text-xs text-ink-muted">Metrics</p>
          {audit.metrics ? (
            <div className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink sm:grid-cols-5">
              <Field label="mAP50" value={formatMetric(audit.metrics["mAP50"])} />
              <Field label="mAP50-95" value={formatMetric(audit.metrics["mAP50-95"])} />
              <Field label="Precision" value={formatMetric(audit.metrics["precision"])} />
              <Field label="Recall" value={formatMetric(audit.metrics["recall"])} />
              <Field label="Accuracy" value={formatMetric(audit.metrics["accuracy"])} />
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
      <p className="tabular break-all text-xs text-ink" title={dataset.s3_uri}>
        {dataset.folder || "--"}
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
            <div className="tabular mt-1.5 max-h-48 overflow-y-auto rounded-sm border border-hairline bg-page p-2 text-xs text-ink">
              {dataset.file_names.map((name) => (
                <p key={name} className="break-all">{name}</p>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
