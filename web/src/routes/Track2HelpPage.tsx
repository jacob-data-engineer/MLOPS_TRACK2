import { HelpCircle } from "lucide-react";
import { Card } from "../components/Card";

/**
 * Static documentation of the required Track2 training-dataset layout --
 * ported directly from the old app's Help tab (`_render_help_tab`), which
 * made zero AWS calls. No backend endpoint for this page.
 */
export function Track2HelpPage() {
  return (
    <div>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <HelpCircle size={20} className="text-accent" strokeWidth={2} />
        Track2 Help
      </h1>
      <p className="mb-6 max-w-3xl text-sm text-ink-secondary">
        The dataset folder layout a Track2 training run expects, for anyone
        preparing a new dataset to train against.
      </p>

      <div className="flex flex-col gap-5">
        <Card title="Dataset folder layout">
          <pre className="tabular overflow-x-auto rounded-sm border border-hairline bg-page p-3 text-xs text-ink">
{`<dataset_folder>/
  data.yaml
  train/
    images/
      *.jpg | *.jpeg | *.png | *.bmp | *.tif | *.tiff | *.webp
    labels/
      *.txt
  val/
    images/
    labels/`}
          </pre>
        </Card>

        <Card title="data.yaml">
          <p className="mb-2 text-sm text-ink-secondary">
            Standard Ultralytics YOLO dataset config -- paths relative to the
            dataset folder, plus the class names in index order:
          </p>
          <pre className="tabular overflow-x-auto rounded-sm border border-hairline bg-page p-3 text-xs text-ink">
{`train: train/images
val: val/images
nc: 2
names: ["class_a", "class_b"]`}
          </pre>
        </Card>

        <Card title="Label files">
          <p className="text-sm text-ink-secondary">
            One <code className="tabular">.txt</code> file per image, same
            basename, one row per box: <code className="tabular">class_id
            x_center y_center width height</code>, all coordinates normalized
            0.0-1.0. An image with no objects has an empty (or absent) label
            file.
          </p>
        </Card>

        <Card title="Accepted image extensions">
          <p className="tabular text-sm text-ink">
            .jpg, .jpeg, .png, .bmp, .tif, .tiff, .webp
          </p>
        </Card>

        <Card title="Evaluation dataset layout (Track2 Evaluation queue)">
          <p className="mb-2 text-sm text-ink-secondary">
            Flatter than a training dataset -- no train/val split, no{" "}
            <code className="tabular">data.yaml</code>. Every image just needs
            a same-basename label file next to it:
          </p>
          <pre className="tabular overflow-x-auto rounded-sm border border-hairline bg-page p-3 text-xs text-ink">
{`<data_folder>/
  images/
    *.jpg | *.jpeg | *.png | *.bmp | *.tif | *.tiff | *.webp
  labels/
    *.txt`}
          </pre>
          <p className="mt-2 text-sm text-ink-secondary">
            Same YOLO label format as training:{" "}
            <code className="tabular">class_id x_center y_center width height</code>,
            normalized 0.0-1.0. An image with no matching label file is scored
            as having zero ground-truth objects, not treated as an error.
          </p>
        </Card>
      </div>
    </div>
  );
}
