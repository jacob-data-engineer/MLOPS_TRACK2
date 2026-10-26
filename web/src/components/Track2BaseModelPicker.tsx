import { useEffect, useState } from "react";
import { isSeedVersion, statusLabel } from "../lib/slug";
import {
  getTrack2Metrics,
  type Track2MetricsRow,
} from "../lib/api";

/** "Train on an existing model" -- which registered version a run starts from.
 *
 * Lives in the Models card rather than the training form, because it is a
 * statement about WHICH MODEL, not about this particular run. The form
 * below only starts it.
 *
 * Two kinds of row appear, in one radio group because they are one choice:
 *
 *   - this detector's own versions at this generation and size, which is
 *     how v2 continues from v1; and
 *   - a SEED from the production registry, because a detector's own
 *     registry starts empty and its first run would otherwise have
 *     nothing at all to begin from. The head is rebuilt for the dataset's
 *     classes, so a person/helmet checkpoint is a legitimate seed for any
 *     taxonomy.
 *
 * A seed must be the SAME GENERATION. The production groups are
 * `yolo11{size}-detector`, so without that filter a YOLO26 selection
 * would be offered YOLO11 checkpoints and quietly train a YOLO11 model
 * under a YOLO26 name -- the run would succeed, the registry would say
 * YOLO26, and the weights would not be.
 */
export function Track2BaseModelPicker({
  detectorSlug,
  detectorLabel,
  architecture,
  generation,
  size,
  groupName,
  value,
  onChange,
  refreshKey,
}: {
  detectorSlug: string;
  detectorLabel: string;
  architecture: string;
  generation: string;
  size: string;
  /** The registry group this run would register INTO. */
  groupName: string;
  value: string;
  onChange: (modelPackageArn: string) => void;
  /** Bumped by an import, so a checkpoint just brought in appears here
   * without a page reload. */
  refreshKey: number;
  /** True when "train on an existing model" is the chosen source. */
}) {
  const [rows, setRows] = useState<Track2MetricsRow[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getTrack2Metrics()
      .then((res) => !cancelled && setRows(res.rows))
      .catch(() => undefined)
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  /** "yolo11s-detector" -> "11". A row carries its size but not its
   * generation, and the generation is the half that matters here. */
  function rowGeneration(row: Track2MetricsRow): string {
    return /^yolo(\d+)/.exec(row.group_name)?.[1] ?? "11";
  }

  const own = rows.filter(
    (r) => r.registry === detectorSlug && r.size === size && rowGeneration(r) === generation,
  );
  const seeds = rows.filter(
    (r) => r.registry === "production" && r.size === size && rowGeneration(r) === generation,
  );
  const all = [
    ...own.map((r) => ({ row: r, seed: false })),
    ...seeds.map((r) => ({ row: r, seed: true })),
  ];

  /** `metrics` is null for a version with no evaluation report -- an
   * imported checkpoint has none, which is most of what seeds a first
   * run. "--" rather than 0.000, because "not measured" and "measured as
   * zero" are different facts about a model. */
  function metric(v: number | null | undefined): string {
    return v === null || v === undefined ? "--" : v.toFixed(3);
  }

  return (
    <div className="rounded-md border border-black/15 p-3">
      {/* A plain heading. There was briefly a tick here meaning "I am
          training, not importing", which dimmed the import card above --
          removed as not needed. This card never switches off anyway: it
          is the only way a run gets its starting weights, so a box that
          could disable it would be a control that breaks the page. */}
      <p className="mb-2 text-sm font-medium text-ink">Train on an existing model</p>
      <p className="mb-2 text-xs text-ink-muted">
        Which registered version this run starts from. Unapproved versions are
        allowed &mdash; a new taxonomy&apos;s first run often has nothing
        approved to start from.
      </p>

      <fieldset>

      {!detectorSlug ? (
        <p className="text-xs text-ink-muted">
          Pick a Model Name in the dropdown above to see its registry.
        </p>
      ) : (
        <>
          {groupName && (
            <p className="tabular mb-2 text-xs text-ink-secondary">
              {detectorLabel} &middot; {architecture} &rarr; {groupName}
            </p>
          )}

          {/* No "start from scratch" option here, deliberately. This
              path continues from a registered version, and the form
              refuses to submit without one -- so offering scratch here
              produced a selection that left "Add to Queue" disabled for
              good, with nothing on screen saying why. Scratch is the
              card directly above, which arms itself and carries its own
              warning about needing far more data and epochs. */}
          {loading ? (
            <p className="text-xs text-ink-muted">Reading the registries...</p>
          ) : all.length === 0 ? (
            /* Not an error, and the reason matters: for a brand new
               generation there is no production checkpoint of that
               generation to seed from, so the honest options are an
               imported .pt or scratch -- both of which are in this same
               card, directly above. */
            <div className="rounded-md border border-black/15 bg-page p-2.5 text-xs text-ink-secondary">
              <p className="font-medium text-ink">
                Nothing to start {architecture} from yet.
              </p>
              <p className="mt-1">
                This detector has no {architecture} version, and there is no
                production {architecture} checkpoint to seed from either --
                the production registries hold YOLO11 weights only, and a
                YOLO11 checkpoint cannot seed a{" "}
                {generation === "11" ? "different" : `YOLO${generation}`} model.
              </p>
              {/* ONE way forward, not the two this used to offer. The
                  second was "Training from scratch", removed from the
                  product -- and this is the empty state whose whole job
                  is telling somebody how to get unstuck, so a route that
                  no longer exists is the worst thing it could name. */}
              <p className="mt-1">
                <strong>Import a pretrained COCO or an already-trained model</strong> above, with a{" "}
                <span className="tabular">{architecture}.pt</span> &mdash; the
                official COCO release, or weights somebody already trained.
                Importing files it as this Model Name&rsquo;s first version, and
                this list then has something to continue from.
              </p>
            </div>
          ) : (
            <div className="max-h-72 overflow-y-auto rounded-md border border-black/15">
              <table className="w-full text-left text-sm">
                <thead className="sticky top-0 bg-page text-xs text-ink-muted">
                  <tr className="border-b border-hairline">
                    <th className="w-8 py-1.5 pl-2" />
                    <th className="py-1.5 pr-4 font-medium">Version</th>
                    <th className="py-1.5 pr-4 font-medium">Status</th>
                    <th className="py-1.5 pr-4 font-medium">mAP50</th>
                    <th className="py-1.5 pr-4 font-medium">mAP50-95</th>
                    <th className="py-1.5 pr-4 font-medium">Precision</th>
                    <th className="py-1.5 pr-4 font-medium">Recall</th>
                    <th className="py-1.5 pr-2 font-medium">F1</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-hairline">
                  {all.map(({ row, seed }) => (
                    <tr key={row.model_package_arn}>
                      <td className="py-1.5 pl-2">
                        <input
                          type="radio"
                          name="track2-base-package"
                          checked={value === row.model_package_arn}
                          onChange={() => onChange(row.model_package_arn)}
                        />
                      </td>
                      <td className="tabular py-1.5 pr-4">
                        {row.version}
                        {seed && (
                          <span className="ml-1.5 text-xs text-ink-muted">seed</span>
                        )}
                      </td>
                      {/* A seed says so, in the column whose job is
                          saying what a version is. It IS Pending, and
                          pending BECAUSE it is a seed -- the same fact,
                          and "Seed" is the half that explains the other.
                          The raw status stays on hover.

                          Keyed on the description the server wrote, not
                          on the version number: a v1 produced by a
                          training run is a real model, and calling it a
                          seed would be a lie in this exact column. */}
                      <td
                        className={`py-1.5 pr-4 text-xs ${
                          isSeedVersion(row.description)
                            ? "font-medium text-ink"
                            : "text-ink-secondary"}`}
                        title={isSeedVersion(row.description)
                          ? `${row.status} -- ${row.description}`
                          : row.status}
                      >
                        {isSeedVersion(row.description) ? "Seed" : statusLabel(row.status)}
                      </td>
                      <td className="tabular py-1.5 pr-4">{metric(row.metrics?.mAP50)}</td>
                      <td className="tabular py-1.5 pr-4">{metric(row.metrics?.["mAP50-95"])}</td>
                      <td className="tabular py-1.5 pr-4">{metric(row.metrics?.precision)}</td>
                      <td className="tabular py-1.5 pr-4">{metric(row.metrics?.recall)}</td>
                      <td className="tabular py-1.5 pr-2">
                        {metric(row.metrics?.f1 ?? row.metrics?.accuracy)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {seeds.length > 0 && (
            <p className="mt-1 text-xs text-ink-muted">
              A <span className="font-medium">seed</span> is a production
              checkpoint of the same architecture. The head is rebuilt for this
              dataset&apos;s classes, so it can seed a completely different
              taxonomy.
            </p>
          )}
        </>
      )}
      </fieldset>
    </div>
  );
}
