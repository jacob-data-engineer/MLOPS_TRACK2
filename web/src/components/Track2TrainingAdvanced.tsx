import { useCardOpen } from "../lib/cardState";

/**
 * The rest of what Ultralytics takes: optimiser, schedule, loss weights,
 * augmentation.
 *
 * ONLY WHAT YOU CHANGED IS SENT. Every field shows the Ultralytics
 * default and stays empty until you type in it; the blob carries the
 * overrides and nothing else. So a run that touches none of this behaves
 * exactly as it did before any of it existed, and the audit row records
 * a decision rather than thirty-five restatements of a default.
 *
 * The defaults here are read off Ultralytics 8.4.130, the version the
 * training image installs -- not from memory. `label_smoothing` is
 * deliberately absent: it is in plenty of tutorials and not in this
 * version, and a field that silently does nothing is worse than no
 * field. The trainer validates every key against the Ultralytics it
 * actually has and REFUSES one it does not know, so a drift between this
 * list and the image fails the job in seconds instead of training an
 * hour on quiet defaults.
 */

type Kind = "number" | "text" | "bool" | "choice";

interface Param {
  key: string;
  label: string;
  kind: Kind;
  /** Ultralytics' own default, shown as the placeholder. */
  fallback: string;
  choices?: string[];
  hint?: string;
}

const GROUPS: { title: string; why: string; params: Param[] }[] = [
  {
    title: "Optimiser",
    why: "How the weights move. The first thing to reach for when a run will not converge, or converges and then wanders.",
    params: [
      { key: "optimizer", label: "Optimizer", kind: "choice", fallback: "auto",
        choices: ["auto", "SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp"],
        hint: "auto picks SGD or AdamW from the run's size" },
      { key: "lr0", label: "Initial LR", kind: "number", fallback: "0.01" },
      { key: "lrf", label: "Final LR factor", kind: "number", fallback: "0.01",
        hint: "final LR = lr0 x this" },
      { key: "momentum", label: "Momentum", kind: "number", fallback: "0.937" },
      { key: "weight_decay", label: "Weight decay", kind: "number", fallback: "0.0005" },
      { key: "warmup_epochs", label: "Warmup epochs", kind: "number", fallback: "3.0" },
      { key: "warmup_momentum", label: "Warmup momentum", kind: "number", fallback: "0.8" },
      { key: "warmup_bias_lr", label: "Warmup bias LR", kind: "number", fallback: "0.1" },
      { key: "cos_lr", label: "Cosine schedule", kind: "bool", fallback: "false" },
    ],
  },
  {
    title: "Fine-tuning",
    why: "For starting from a checkpoint that is already good. Freezing keeps what it learned; dropout and scale guard against a small dataset being memorised.",
    params: [
      { key: "freeze", label: "Freeze layers", kind: "number", fallback: "none",
        hint: "first N layers held fixed" },
      { key: "dropout", label: "Dropout", kind: "number", fallback: "0.0" },
      { key: "multi_scale", label: "Multi-scale", kind: "number", fallback: "0.0" },
      { key: "rect", label: "Rectangular batches", kind: "bool", fallback: "false" },
      { key: "amp", label: "Mixed precision", kind: "bool", fallback: "true" },
      { key: "fraction", label: "Dataset fraction", kind: "number", fallback: "1.0",
        hint: "train on a slice, for a quick check" },
    ],
  },
  {
    title: "Loss weights",
    why: "What the run is being asked to care about. Raise cls when classes are confused, box when the boxes are loose.",
    params: [
      { key: "box", label: "Box", kind: "number", fallback: "7.5" },
      { key: "cls", label: "Class", kind: "number", fallback: "0.5" },
      { key: "dfl", label: "DFL", kind: "number", fallback: "1.5" },
      { key: "nbs", label: "Nominal batch", kind: "number", fallback: "64" },
    ],
  },
  {
    title: "Augmentation",
    why: "Variation invented from the images you have. More of it when the dataset is small and the site is varied; less when it already covers the conditions.",
    params: [
      { key: "hsv_h", label: "Hue", kind: "number", fallback: "0.015" },
      { key: "hsv_s", label: "Saturation", kind: "number", fallback: "0.7" },
      { key: "hsv_v", label: "Brightness", kind: "number", fallback: "0.4" },
      { key: "degrees", label: "Rotation", kind: "number", fallback: "0.0" },
      { key: "translate", label: "Translate", kind: "number", fallback: "0.1" },
      { key: "scale", label: "Scale", kind: "number", fallback: "0.5" },
      { key: "shear", label: "Shear", kind: "number", fallback: "0.0" },
      { key: "perspective", label: "Perspective", kind: "number", fallback: "0.0" },
      { key: "flipud", label: "Flip up/down", kind: "number", fallback: "0.0" },
      { key: "fliplr", label: "Flip left/right", kind: "number", fallback: "0.5" },
      { key: "mosaic", label: "Mosaic", kind: "number", fallback: "1.0" },
      { key: "close_mosaic", label: "Close mosaic", kind: "number", fallback: "10",
        hint: "turn mosaic off for the last N epochs" },
      { key: "mixup", label: "Mixup", kind: "number", fallback: "0.0" },
      { key: "cutmix", label: "Cutmix", kind: "number", fallback: "0.0" },
      { key: "copy_paste", label: "Copy-paste", kind: "number", fallback: "0.0" },
      { key: "erasing", label: "Random erasing", kind: "number", fallback: "0.4" },
      { key: "bgr", label: "BGR flip", kind: "number", fallback: "0.0" },
      { key: "auto_augment", label: "Auto augment", kind: "choice", fallback: "randaugment",
        choices: ["randaugment", "autoaugment", "augmix"] },
    ],
  },
];

/** Every field's default, by key -- the same table the inputs show. */
export const ULTRALYTICS_DEFAULTS: Record<string, string> = Object.fromEntries(
  GROUPS.flatMap((group) => group.params.map((p) => [p.key, p.fallback])),
);

/**
 * What a run is actually sent.
 *
 * OFF: nothing. The run trains exactly as it would if none of this
 * existed, and Ultralytics applies whatever its own defaults are at the
 * time.
 *
 * ON: every field, with the shown default filled in wherever one was
 * not typed. That is the difference the tick buys -- the run PINS these
 * values and the audit records them, rather than inheriting whatever an
 * Ultralytics upgrade decides later. Two runs a year apart with the box
 * ticked are comparable; without it they are only comparable if nothing
 * upstream moved.
 *
 * `none` is skipped rather than sent: it is how this table writes "not
 * set" for `freeze`, and sending the string would be a value.
 */
export function effectiveHyperparams(
  value: Record<string, string>,
  enabled: boolean,
): Record<string, string> {
  if (!enabled) return {};
  const out: Record<string, string> = {};
  for (const [key, fallback] of Object.entries(ULTRALYTICS_DEFAULTS)) {
    const chosen = (value[key] ?? "").trim();
    const resolved = chosen || fallback;
    if (!resolved || resolved === "none") continue;
    out[key] = resolved;
  }
  return out;
}

export function Track2TrainingAdvanced({
  value,
  onChange,
  onCommit,
  enabled,
  onEnabledChange,
}: {
  /** Overrides only -- a key is absent unless it was changed. */
  value: Record<string, string>;
  onChange: (next: Record<string, string>) => void;
  /** Called on blur, so this saves the same way the row above it does. */
  onCommit: () => void;
  /** Whether these are sent at all. Off means the run is untouched by
   * any of it; on means every field is sent, defaults included. */
  enabled: boolean;
  onEnabledChange: (on: boolean) => void;
}) {
  const [open, setOpen] = useCardOpen("training_advanced", false);
  const changed = Object.keys(value).length;

  function set(key: string, raw: string) {
    const next = { ...value };
    // An emptied field is not "set to empty", it is back to the default.
    if (raw === "") delete next[key];
    else next[key] = raw;
    onChange(next);
  }

  return (
    <div className="mt-3 border-t border-hairline pt-2">
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="text-xs font-medium text-ink-secondary underline"
        >
          {open ? "Hide" : "Show"} the rest of the training parameters
        </button>
        <span className="tabular text-[11px] text-ink-muted">
          {!enabled
            ? "not sent -- Ultralytics decides"
            : changed === 0
              ? "sending all defaults"
              : `sending all, ${changed} changed`}
        </span>
      </div>

      {/* The switch that decides whether any of this reaches the run.
          Outside the fold, because it changes what a run does and a
          control that does that should not be behind a Show link. */}
      <label className="mt-2 flex items-start gap-2 text-xs text-ink-secondary">
        <input
          type="checkbox"
          className="mt-0.5 shrink-0"
          checked={enabled}
          onChange={(e) => { onEnabledChange(e.target.checked); onCommit(); }}
        />
        <span>
          Set these training parameters for the run
          <span className="block text-ink-muted">
            Off, none of this is sent and Ultralytics uses its own
            defaults. On, <strong>every field below is sent</strong> --
            the ones shown as defaults included -- so the run is pinned to
            these numbers and the audit records them. Two runs months
            apart are then comparable even if Ultralytics changed a
            default in between.
          </span>
        </span>
      </label>

      {!open && changed > 0 && (
        <p className="tabular mt-1 text-xs text-ink-secondary">
          {Object.entries(value).map(([k, v]) => `${k}=${v}`).join(", ")}
        </p>
      )}

      {open && (
        <>
          <p className="mt-2 text-xs text-ink-muted">
            {enabled
              ? "A field left blank is sent as the default shown in it. "
                + "An unknown or misspelled key is refused by the trainer "
                + "before the GPU starts."
              : "These are not being sent. Tick the box above to use them; "
                + "the values below are what would be sent."}
          </p>
          {GROUPS.map((group) => (
            // Dimmed rather than hidden when off: the numbers are still
            // worth reading, and hiding them would make the tick feel
            // like it deleted something.
            <div key={group.title}
                 className={enabled ? "mt-3" : "mt-3 opacity-50"}>
              <p className="text-xs font-medium text-ink">{group.title}</p>
              <p className="mb-1.5 text-xs text-ink-muted">{group.why}</p>
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                {group.params.map((p) => (
                  <label key={p.key} className="block">
                    <span className="mb-1 block text-xs text-ink-muted">
                      {p.label}
                      {p.hint && (
                        <span className="text-ink-muted/70"> ({p.hint})</span>
                      )}
                    </span>
                    {p.kind === "choice" ? (
                      <select
                        className="w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-xs"
                        value={value[p.key] ?? ""}
                        onChange={(e) => set(p.key, e.target.value)}
                        onBlur={onCommit}
                      >
                        <option value="">{p.fallback} (default)</option>
                        {p.choices?.filter((c) => c !== p.fallback).map((c) => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                    ) : p.kind === "bool" ? (
                      <select
                        className="w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-xs"
                        value={value[p.key] ?? ""}
                        onChange={(e) => set(p.key, e.target.value)}
                        onBlur={onCommit}
                      >
                        <option value="">{p.fallback} (default)</option>
                        <option value="true">true</option>
                        <option value="false">false</option>
                      </select>
                    ) : (
                      <input
                        type="text"
                        inputMode="decimal"
                        className="tabular w-full rounded-md border border-black/15 bg-page px-2 py-1.5 text-xs"
                        value={value[p.key] ?? ""}
                        placeholder={p.fallback}
                        onChange={(e) => set(p.key, e.target.value)}
                        onBlur={onCommit}
                      />
                    )}
                  </label>
                ))}
              </div>
            </div>
          ))}
          {changed > 0 && (
            <button
              type="button"
              onClick={() => { onChange({}); onCommit(); }}
              className="mt-3 rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
            >
              Put everything back to the defaults
            </button>
          )}
        </>
      )}
    </div>
  );
}
