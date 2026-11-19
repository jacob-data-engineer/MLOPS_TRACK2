import { useState } from "react";
import { comparedParameters, TrainingSettingsCell } from "./Track2ParameterComparison";
import type { Track2MetricsRow, Track2TrainingAudit, Track2TrainingAuditDataset } from "../lib/api";
import { modelLabel } from "./Track2ModelPicker";

function DatasetSummary({ data }: { data?: Track2TrainingAuditDataset }) {
  if (!data) return <span className="text-ink-muted">Not recorded</span>;
  return <>
    <p className="font-medium">{data.file_count == null ? "Image count not recorded" : `${data.file_count.toLocaleString()} images`}</p>
    {data.label_count != null && <p>{data.label_count.toLocaleString()} label files</p>}
    <p className="mt-1 text-ink-muted">Folder / location</p>
    <p className="break-all">{data.s3_uri || data.folder || "Not recorded"}</p>
    {!!data.source_folders?.length && <><p className="mt-1 text-ink-muted">Recorded sources</p>{data.source_folders.map((source, index) => <p className="break-words" key={index}>{source}</p>)}</>}
  </>;
}

export function Track2ComparisonSummary({ rows, audits, status }: {
  rows: Track2MetricsRow[]; audits: Record<string, Track2TrainingAudit>;
  status: "loading" | "ready" | "failed";
}) {
  const [differencesOnly, setDifferencesOnly] = useState(false);
  const parameters = comparedParameters({rows, audits});
  const shownParameters = parameters.filter(p => !differencesOnly || p.different);
  if (!rows.length) return null;
  return <div className="mb-4">
    <label className="mb-2 block text-xs"><input type="checkbox" checked={differencesOnly} onChange={e=>setDifferencesOnly(e.target.checked)}/> Show differences only in training settings ({parameters.filter(p=>p.different).length})</label>
    <div className="overflow-x-auto rounded-sm border border-hairline">
      <table className="w-full min-w-[850px] table-fixed text-left text-xs">
        <caption className="px-3 py-2 text-left font-semibold">Compared models — data and training summary</caption>
        <thead className="border-y border-hairline bg-page"><tr>{["Model / run", "Training images and folders", "Evaluation images and folders", "Training settings", "Dataset record"].map(label => <th key={label} className="px-3 py-2 align-top">{label}</th>)}</tr></thead>
        <tbody>{rows.map((row, index) => {
          const audit = audits[row.model_package_arn];
          const actual = audit?.actual;
          const changedData = Boolean(actual?.train_data_s3_uri && audit?.train_dataset.s3_uri && actual.train_data_s3_uri.replace(/\/+$/, "") !== audit.train_dataset.s3_uri.replace(/\/+$/, ""));
          return <tr key={row.model_package_arn} className="border-b border-hairline last:border-0">
            <th scope="row" className="break-words px-3 py-2 align-top font-medium">{row.registry}<br/>{modelLabel(row)}<p className="mt-1">{audit ? `Run ${audit.run_number ?? "not recorded"}` : status === "loading" ? "Loading run…" : status === "failed" ? "Run lookup failed" : "No recorded run"}</p></th>
            <td className="px-3 py-2 align-top"><DatasetSummary data={audit?.train_dataset}/>{changedData && <p className="mt-2 break-all text-critical">Actual job location: {actual?.train_data_s3_uri}. Counts above are from the saved submission.</p>}</td>
            <td className="px-3 py-2 align-top"><DatasetSummary data={audit?.eval_dataset}/>{audit?.eval_is_train_dataset && <p className="mt-2 text-critical">Overlaps training data — not held out.</p>}</td>
            <td className="px-3 py-2 align-top"><TrainingSettingsCell entries={shownParameters} index={index}/>{!shownParameters.length && <p>No recorded differences.</p>}</td>
            <td className="break-all px-3 py-2 align-top"><p>{audit?.dataset_version_id || audit?.train_dataset.dataset_version_id || audit?.eval_dataset.dataset_version_id || "Dataset ID not recorded"}</p></td>
          </tr>;
        })}</tbody>
      </table>
    </div>
    <p className="mt-1 text-xs text-ink-muted">Full recorded image counts, not the capped file-name preview. Folders and sources come from each saved run; separate per-source counts may not have been recorded. Training settings show saved run parameters, including saved defaults. Highlighted values differ across recorded runs; missing values are not treated as differences.</p>
  </div>;
}
