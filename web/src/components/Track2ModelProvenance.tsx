import type { Track2MetricsRow, Track2TrainingAudit } from "../lib/api";

/** The classes a registered version detects, and where its training images
 * came from -- both collapsed.
 *
 * Shared by the Metrics table and the training page's base-model picker so
 * the two cannot drift: the same version must not describe its taxonomy or
 * its data differently depending on which page you are looking at.
 *
 * Collapsed by default because a six-class list or a full S3 URI would
 * otherwise set the column width for every row, including the two-class
 * production ones that have nothing to expand.
 */
export function Track2ModelProvenance({
  metrics,
  audit,
}: {
  metrics: Track2MetricsRow["metrics"];
  /** The training run behind this version, when one was recorded. Used as
   * the fallback source of training data: the production pipeline's older
   * versions carry a legacy flat evaluation report with no dataset fields
   * at all, but the audit recorded the folder and the image count for the
   * same run. Real data rather than a blank. */
  audit?: Track2TrainingAudit;
}) {
  const classes = metrics?.classes ?? [];
  const dataset = metrics?.dataset;
  const reportHasDataset = Boolean(
    dataset && (dataset.source_uri || dataset.dataset_version || dataset.validation_count != null),
  );
  const auditTrain = audit?.train_dataset;
  const hasDataset = reportHasDataset || Boolean(auditTrain?.folder);
  if (classes.length === 0 && !hasDataset) return null;

  return (
    <div className="flex flex-col gap-1">
      {classes.length > 0 && (
        <details>
          <summary className="cursor-pointer select-none text-ink-muted">
            {classes.length} classes
          </summary>
          <span className="tabular mt-1 block break-words">
            {classes.map((name, index) => `${index}: ${name}`).join(", ")}
          </span>
        </details>
      )}
      {hasDataset && (
        <details>
          <summary className="cursor-pointer select-none text-ink-muted">Training data</summary>
          <dl className="mt-1 space-y-0.5">
            {/* Report first, audit second. The report names the pinned
                snapshot the data was derived from; the audit names the
                folder the operator actually picked. Where both exist they
                agree, and where only one does it is the only answer. */}
            {(dataset?.source_uri || auditTrain?.s3_uri || auditTrain?.folder) && (
              <div>
                <dt className="inline text-ink-muted">Images from: </dt>
                <dd className="tabular inline break-all">
                  {dataset?.source_uri || auditTrain?.s3_uri || auditTrain?.folder}
                </dd>
              </div>
            )}
            {auditTrain?.file_count ? (
              <div>
                <dt className="inline text-ink-muted">Training images: </dt>
                <dd className="tabular inline">{auditTrain.file_count}</dd>
              </div>
            ) : null}
            {audit?.eval_dataset?.folder && (
              <div>
                <dt className="inline text-ink-muted">Evaluated on: </dt>
                <dd className="tabular inline break-all">
                  {audit.eval_dataset.folder}
                  {audit.eval_is_train_dataset && (
                    <span className="text-critical"> (same as training -- not held out)</span>
                  )}
                </dd>
              </div>
            )}
            {/* Optional chaining throughout, not `!`: this block now renders
                for versions whose report carried no dataset at all and whose
                only source is the audit, so `dataset` is genuinely absent
                rather than merely narrowed. */}
            {dataset?.dataset_version && (
              <div>
                <dt className="inline text-ink-muted">Dataset version: </dt>
                <dd className="tabular inline break-all">{dataset.dataset_version}</dd>
              </div>
            )}
            {dataset?.provenance && (
              <div>
                <dt className="inline text-ink-muted">Pinned by: </dt>
                <dd className="inline">
                  {dataset.provenance === "dataset-card"
                    ? "prepared dataset card"
                    : dataset.provenance === "file-contents"
                      ? "hash of the image and label files"
                      : dataset.provenance}
                </dd>
              </div>
            )}
            {dataset?.validation_count != null && (
              <div>
                <dt className="inline text-ink-muted">Validation images: </dt>
                <dd className="tabular inline">{dataset.validation_count}</dd>
              </div>
            )}
            {dataset?.baseline_uri && (
              <div>
                <dt className="inline text-ink-muted">Compared against: </dt>
                <dd className="tabular inline break-all">{dataset.baseline_uri}</dd>
              </div>
            )}
          </dl>
        </details>
      )}
    </div>
  );
}
