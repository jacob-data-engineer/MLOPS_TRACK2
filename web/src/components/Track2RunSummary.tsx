import type { Track2TrainingAudit } from "../lib/api";

/** Summarize the saved run, never today's mutable dataset configuration. */
export function Track2RunSummary({ audit }: { audit: Track2TrainingAudit }) {
  const missing = "Not recorded";
  const actual = audit.actual;
  const settings = audit.parameters;
  const count = (n: number | undefined) => n == null ? missing : n.toLocaleString();
  const sources = (data: Track2TrainingAudit["train_dataset"]) =>
    data.source_folders?.length ? data.source_folders.join("; ") : data.folder || missing;
  const changedData = Boolean(actual?.train_data_s3_uri && audit.train_dataset.s3_uri
    && actual.train_data_s3_uri.replace(/\/+$/, "") !== audit.train_dataset.s3_uri.replace(/\/+$/, ""));
  const rows = [
    ["Model / run", `${audit.detector_slug || missing} · ${audit.model ? `${audit.model.group} v${audit.model.version}` : `${audit.generation || missing} ${audit.size || ""}`} · Run ${audit.run_number ?? missing}`],
    ["Training settings", `Epochs ${actual?.epochs ?? settings.epochs ?? missing} · Image size ${actual?.img_size ?? settings.img_size ?? missing} · Batch ${actual?.batch ?? settings.batch ?? missing} · Seed ${actual?.seed ?? settings.seed ?? missing}`],
    ["Training data", `${count(audit.train_dataset.file_count)} images · ${count(audit.train_dataset.label_count)} label files`],
    ["Evaluation data", `${count(audit.eval_dataset.file_count)} images · ${count(audit.eval_dataset.label_count)} label files${audit.eval_is_train_dataset ? " · Overlaps training data — not held out" : ""}`],
    ["Training sources (recorded)", sources(audit.train_dataset)],
    ["Evaluation sources (recorded)", sources(audit.eval_dataset)],
    ["Base checkpoint", actual?.base_model_s3_uri || audit.base_model.folder || missing],
    ["Dataset record", audit.dataset_version_id || audit.train_dataset.dataset_version_id || audit.eval_dataset.dataset_version_id || missing],
    ["Snapshot", audit.snapshot?.snapshot_id ? `${audit.snapshot.snapshot_id} · ${audit.snapshot.pinned ? "Files pinned by name" : "Week folders only; not repeatable"}` : missing],
  ];
  return <section className="mb-3" aria-label="Run inputs summary">
    <div className="overflow-x-auto rounded-sm border border-hairline">
      <table className="w-full text-left text-xs">
        <caption className="bg-surface px-3 py-2 text-left font-semibold">What this run used</caption>
        <thead className="border-y border-hairline bg-page"><tr><th className="px-3 py-1.5">Item</th><th className="px-3 py-1.5">Recorded value</th></tr></thead>
        <tbody>{rows.map(([label, value]) => <tr key={label} className="border-b border-hairline last:border-0"><th scope="row" className="w-44 px-3 py-1.5 align-top font-medium text-ink-secondary">{label}</th><td className="break-all px-3 py-1.5">{value}</td></tr>)}</tbody>
      </table>
    </div>
    <p className="mt-1 text-xs text-ink-muted">Counts cover the full recorded dataset, not the file-name preview. Settings use the actual job values when recorded; otherwise the submitted values. Source descriptions do not provide a complete per-source image breakdown.</p>
    {changedData && <p className="mt-1 break-all text-xs text-critical">The job used a different training location: {actual?.train_data_s3_uri}. The dataset counts above describe the saved submission; see run details for the difference.</p>}
  </section>;
}
