import type { Track2TrainingProgress } from "../lib/api";
import { ProgressBar } from "./ProgressBar";

export function Track2TrainingRunProgress({ progress }: { progress: Track2TrainingProgress }) {
  const epochs = progress.epoch_progress;
  const images = progress.image_progress;
  const download = progress.download_progress;
  return <div className="space-y-3 rounded-md border border-hairline bg-page p-3 text-xs" aria-label="Training progress">
    <p className="font-medium text-ink">Training progress</p>
    {epochs && <div>
      <p className="mb-1 tabular">{epochs.completed ? "Epochs completed" : "Current epoch"}: {epochs.current} / {epochs.total}</p>
      <div role="progressbar" aria-label="Epochs completed" aria-valuemin={0} aria-valuemax={epochs.total}
        aria-valuenow={epochs.completed ? epochs.current : Math.max(0, epochs.current - 1)}>
        <ProgressBar processed={epochs.completed ? epochs.current : Math.max(0, epochs.current - 1)} total={epochs.total} />
      </div>
      {progress.steps.TrainYolo === "Succeeded" && epochs.completed && epochs.current < epochs.total &&
        <p className="mt-1 text-ink-muted">Training finished before the planned epoch limit.</p>}
    </div>}
    {images ? <div>
      <p className="mb-1 tabular">Images processed: {images.current.toLocaleString()} / {images.total.toLocaleString()}</p>
      <div role="progressbar" aria-label="Training images processed" aria-valuemin={0} aria-valuemax={images.total} aria-valuenow={Math.min(images.current, images.total)}>
        <ProgressBar processed={images.current} total={images.total} />
      </div>
      <p className="mt-1 text-ink-muted">{images.per_epoch.toLocaleString()} images per epoch. Images count again each epoch; this excludes validation images.</p>
    </div> : <p className="text-ink-muted">{progress.status === "Executing"
      ? "Waiting for training image counts. Downloaded images are not counted as trained."
      : "Training image counts were not recorded for this run."}</p>}
    {download && <div>
      <p className="mb-1 tabular">Downloading dataset: {download.current.toLocaleString()} / {download.total.toLocaleString()} images</p>
      <ProgressBar processed={download.current} total={download.total} />
    </div>}
    {progress.status === "Executing" && <p className="text-ink-muted">Progress updates automatically; metrics can take about a minute to arrive.</p>}
  </div>;
}
