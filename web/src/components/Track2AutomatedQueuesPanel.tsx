import { Loader2, Trash2 } from "lucide-react";
import { useState } from "react";
import type { Track2AutomatedQueue } from "../lib/api";
import { Badge } from "./Badge";
import { Card } from "./Card";

/**
 * Read + delete view of the persistent, EventBridge-scheduled Automated
 * Training Queues -- shared by the Track2 Training and Track2 Schedule
 * pages (same data, same actions), mirroring the old app's
 * `_render_automated_queues_panel` reuse across its Models/Schedule tabs.
 *
 * Adding/editing a queue needs the dataset/weights S3 browsers, so that
 * form lives only on the Training page, not duplicated here.
 */
export function Track2AutomatedQueuesPanel({
  queues,
  onDelete,
}: {
  queues: Track2AutomatedQueue[];
  onDelete: (queueId: string) => Promise<void>;
}) {
  const [deletingId, setDeletingId] = useState<string | null>(null);

  async function handleDelete(id: string) {
    setDeletingId(id);
    try {
      await onDelete(id);
    } finally {
      setDeletingId(null);
    }
  }

  return (
    <Card
      title="Automated training queues"
      caption="Persistent, individually-scheduled recipes -- fire on their own EventBridge rule even while this page isn't open."
    >
      {queues.length === 0 ? (
        <p className="text-sm text-ink-muted">No automated queues configured.</p>
      ) : (
        <div className="flex flex-col gap-2">
          {queues.map((q) => (
            <div
              key={q.id}
              className="flex items-start justify-between gap-3 rounded-sm border border-hairline bg-page p-3"
            >
              <div className="min-w-0 flex-1">
                <div className="mb-1.5 flex items-center gap-2">
                  <span className="text-sm font-medium text-ink">
                    YOLO{q.generation}-{q.size}
                  </span>
                  <Badge tone="info" label={q.schedule_summary} />
                </div>
                <div className="tabular grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-ink-muted sm:grid-cols-3">
                  <span>Epochs: {q.epochs}</span>
                  <span>Image size: {q.img_size}</span>
                  <span>Batch: {q.batch}</span>
                  <span>Seed: {q.seed}</span>
                  <span className="col-span-2 truncate sm:col-span-3" title={q.dataset_folder}>
                    Dataset: {q.dataset_folder}
                  </span>
                  {q.base_model_folder && (
                    <span className="col-span-2 truncate sm:col-span-3" title={q.base_model_folder}>
                      Base model: {q.base_model_folder}
                    </span>
                  )}
                  {q.notes && (
                    <span className="col-span-2 truncate sm:col-span-3" title={q.notes}>
                      Notes: {q.notes}
                    </span>
                  )}
                </div>
              </div>
              <button
                type="button"
                onClick={() => handleDelete(q.id)}
                disabled={deletingId === q.id}
                className="flex shrink-0 items-center gap-1.5 rounded-sm border border-black/15 px-2.5 py-1.5 text-xs font-medium text-ink-secondary hover:bg-black/[0.04] disabled:opacity-40"
              >
                {deletingId === q.id ? (
                  <Loader2 size={13} className="animate-spin" />
                ) : (
                  <Trash2 size={13} />
                )}
                Delete
              </button>
            </div>
          ))}
        </div>
      )}
    </Card>
  );
}
