import {ExperimentHistory} from "./ExperimentHistory";
import {groupTrainingHistory} from "../lib/trainingHistory";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useCardOpen } from "../lib/cardState";
import { useCallback, useEffect, useRef, useState } from "react";
import { Badge, type BadgeTone } from "./Badge";
import {
  getTrack2TrainingAudit, listVariantRuns, type VariantRunRecord,
  type Track2TrainingAudit,
  type Track2TrainingAuditDataset,
} from "../lib/api";

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
  const [runs,setRuns]=useState<VariantRunRecord[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [limit,setLimit]=useState(3);
  const [loading,setLoading]=useState(false);
  const [hasMore,setHasMore]=useState(false);
  const historyRef = useRef<HTMLDivElement>(null);
  const [historyHeight, setHistoryHeight] = useState<number>();

  useEffect(() => {
    const list = historyRef.current;
    if (!list || !audits?.length) return;
    if (audits.length <= 3) { setHistoryHeight(undefined); return; }
    const sections = Array.from(list.children).slice(0, 3);
    const measure = () => {
      const gap = parseFloat(getComputedStyle(list).rowGap) || 0;
      setHistoryHeight(sections.reduce((height, section) => height + section.getBoundingClientRect().height, 0)
        + Math.max(0, sections.length - 1) * gap);
    };
    measure();
    const observer = new ResizeObserver(measure);
    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, [audits]);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const experimentsRequest=listVariantRuns();
      const experiments=await experimentsRequest;
      let requested=limit;
      let res=await getTrack2TrainingAudit(requested,experimentsRequest);
      let grouped=groupTrainingHistory(res.audits,experiments.runs);
      while(requested<200&&res.audits.length===requested&&(grouped.length<=limit||grouped.slice(0,limit).some(g=>g.run?.variants.some(v=>v.run_id&&!g.audits.some(a=>a.id===v.run_id))))){
        requested=Math.min(200,requested+5);
        res=await getTrack2TrainingAudit(requested,experimentsRequest);
        grouped=groupTrainingHistory(res.audits,experiments.runs);
      }
      setHasMore(grouped.length>limit);
      setRuns(experiments.runs);
      setAudits(grouped.slice(0,limit).flatMap(group=>group.audits));
      setError(null);
    } catch {
      setError("Could not load the training history.");
    } finally {setLoading(false);}
  }, [limit]);

  useEffect(() => {
    load();
  }, [load, reloadKey]);

  if (audits !== null && audits.length === 0 && !error) return null;

  return (
    <div className="mt-4">
      <p className="mb-2 text-xs font-semibold uppercase tracking-wide text-ink-muted">History</p><p className="mb-2 text-xs text-ink-muted">The latest 3 runs are loaded.</p>
      {error && <p className="mb-2 text-xs text-critical">{error}</p>}
      {audits === null ? (
        <p className="text-sm text-ink-muted">Loading...</p>
      ) : (
        <div ref={historyRef} role="region" aria-label="Training history" tabIndex={0}
          className="flex flex-col gap-2 overflow-y-auto overscroll-contain [&>div]:shrink-0"
          style={{ maxHeight: historyHeight }}>
          {groupTrainingHistory(audits,runs).map(group=>group.run?
            <ExperimentHistory key={group.id} run={group.run} audits={group.audits}>{group.audits.map(a=><AuditRow key={a.id} audit={a} onDeleted={load}/>)}</ExperimentHistory>:
            <AuditRow key={group.id} audit={group.audits[0]} onDeleted={load}/>)}
        </div>
      )}
      {audits!==null&&<div className="mt-3 flex items-center gap-3 text-xs"><span>{groupTrainingHistory(audits,runs).length} runs loaded.</span>{hasMore&&<button type="button" className="rounded border border-hairline px-3 py-1" disabled={loading} onClick={()=>setLimit(current=>current+5)}>{loading?'Loading...':'Load 5 more runs'}</button>}{error&&<button type="button" disabled={loading} onClick={()=>void load()}>Retry</button>}</div>}
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
/** `onDeleted` is accepted for existing callers; the row has no Delete any more. */
export function AuditRow({ audit }: { audit: Track2TrainingAudit; onDeleted?: () => void }) {
  const datasetRecordId = audit.dataset_version_id || audit.train_dataset.dataset_version_id
    || audit.eval_dataset.dataset_version_id || audit.snapshot?.snapshot_id;
  const [open, setOpen] = useCardOpen("training_audit", false);
  return (
    <div className="rounded-sm border border-hairline bg-page">
      <div data-history-summary className="flex items-center gap-2 px-3 py-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="flex min-w-0 flex-1 items-center gap-2 text-left"
        >
          {open ? <ChevronDown size={14} className="shrink-0 text-ink-muted" /> : <ChevronRight size={14} className="shrink-0 text-ink-muted" />}
          <Badge tone={statusTone(audit.status)} label={audit.status} />
          <span className="text-xs text-ink-muted">{audit.comparison_status || 'No comparisons'}</span>
          <span className="tabular shrink-0 text-sm font-semibold text-ink"
                title={audit.run_number == null ? "A run number appears after the catalog syncs." : "Experiment run number"}>
            {audit.run_number == null ? "Run number pending" : `Run ${audit.run_number}`}{audit.variant_name ? ` / ${audit.variant_name}` : ""}
          </span>
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
      </div>
      {/* The row's actions (Use this dataset, Export training record as
          CSV, Download .pt, Export .pt to S3, Delete) were removed on
          request (2026-11-24). Exports of a run's .pt, dataset and
          metadata are on Model export and Catalog export. */}

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

          <div className="mt-3 grid gap-1 break-all text-xs text-ink">
            <Field label="Dataset record ID" value={datasetRecordId || "Not recorded for this run"} />
            <Field label="Training record ID" value={audit.id} />
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
            {/* The Model Name, outright. The registry group implies it --
                "yolo11s-helmet-detector v9" -- but only to somebody who
                knows how those names are built, and a run that never
                reached registration has no group to read it from. */}
            <Field label="Model Name" value={audit.detector_slug || "--"} />
            <Field
              label="Model version"
              value={audit.model ? `${audit.model.group} v${audit.model.version}` : "--"}
            />
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
