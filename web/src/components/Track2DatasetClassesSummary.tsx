import { useEffect, useState } from "react";
import { getTrack2PullControl } from "../lib/api";
import { listen } from "../lib/bus";

/** The saved dataset mapping, not an inferred inventory of source labels. */
export function Track2DatasetClassesSummary() {
  const [datasetClasses, setDatasetClasses] = useState<Record<string, string> | null>(null);
  const [classesError, setClassesError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let generation = 0;
    const refresh = () => {
      const current = ++generation;
      getTrack2PullControl().then((result) => {
        if (cancelled || current !== generation) return;
        setDatasetClasses(result.config.dataset_assemble?.label_classes ?? {});
        setClassesError(false);
      }).catch(() => {
        if (!cancelled && current === generation) setClassesError(true);
      });
    };
    const unsubscribe = listen("data-sources-changed", refresh);
    refresh();
    return () => { cancelled = true; unsubscribe(); };
  }, []);

  return (
<div>
            <label className="block text-xs text-ink-muted">
              Label classes (shared dataset mapping, in index order)
              <input
                type="text"
                readOnly
                value={classesError ? "Could not load dataset classes." : datasetClasses === null
                  ? "Loading classes..."
                  : Object.entries(datasetClasses).sort(([a], [b]) => Number(a) - Number(b))
                    .map(([id, name]) => `${id} ${name}`).join(", ") || "No dataset classes configured."}
                className="tabular mt-1 w-full rounded border border-black/15 bg-page px-2 py-1 text-sm text-ink"
              />
            </label>
            <p className="mt-1 text-xs text-ink-muted">
              Uses the saved Data Preparation class list; these are not classes detected from this folder.
            </p>
          </div>
  );
}
