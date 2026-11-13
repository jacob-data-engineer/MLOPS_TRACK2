import { useCallback, useEffect, useState } from "react";
import { post } from "../lib/bus";
import { useCardOpen } from "../lib/cardState";
import {
  ApiError,
  buildTrack2DatasetYaml,
  getTrack2PullControl,
  getTrack2SettingsFiles,
  getTrack2Retention,
  previewTrack2Retention,
  saveTrack2Retention,
  track2RetentionMatrix,
  type Track2DatasetYaml,
  type Track2SettingsFile,
  type Track2RetentionMatrix,
  type Track2RetentionPlan,
  type Track2RetentionSettings,
} from "../lib/api";
import { Card } from "./Card";

/**
 * Recency weighting: older weeks contribute fewer images.
 *
 * A detector retrained every week on everything ever collected drifts
 * towards its own history. Decaying the older weeks lets recent data
 * dominate while keeping enough history that rare scenarios survive.
 *
 * Nothing is deleted. The week folders are the record and this selects a
 * subset of them when a dataset is assembled, so any past dataset can be
 * rebuilt exactly -- which is the only thing that makes a training audit
 * worth keeping.
 */
/** A cell's colour, from how much of that week survives.
 *
 * Computed from the fraction rather than picked per 10% step, so a rate
 * of 3% a week shades as readably as a rate of 10%. Dark and saturated at
 * full weight, pale as it falls away -- the shape of the policy should be
 * readable across the grid before any number is.
 */
/** Two decimals on every share shown, fixed rather than adaptive.
 *
 * Whole percents hide the choice they are meant to be showing: at 1% a
 * week, compounding and linear are 0.2 of a point apart after eight
 * weeks, so both render as "93%" and the Curve control looks broken.
 *
 * Two everywhere rather than "as many as the gap needs" because a column
 * whose precision changes as you move a slider is its own puzzle -- the
 * numbers stop being comparable down the column, and a cell that gains a
 * decimal looks like it changed when it did not.
 */
const SHARE_DECIMALS = 2;

/** A fraction as a percentage, at the precision this card reads in. */
function share(fraction: number): string {
  return (fraction * 100).toFixed(SHARE_DECIMALS);
}

function shade(fraction: number): { background: string; color: string } {
  const at = Math.max(0, Math.min(fraction, 1));
  const light = 25 + (1 - at) * 60;
  return {
    background: `hsl(215 58% ${light}%)`,
    color: light < 55 ? "#ffffff" : "#1b2430",
  };
}

export function Track2RetentionPanel() {
  const [settings, setSettings] = useState<Track2RetentionSettings | null>(null);
  const [plan, setPlan] = useState<Track2RetentionPlan | null>(null);
  const [open, setOpen] = useCardOpen("retention", false);
  const [error, setError] = useState("");

  const [enabled, setEnabled] = useState(false);
  // Held as a percentage because that is how it was asked for; the API
  // takes a fraction.
  const [percent, setPercent] = useState(1);
  const [curve, setCurve] = useState<"exponential" | "linear">("exponential");
  const [floor, setFloor] = useState(10);
  // How many of the NEWEST weeks count in full before any decay starts.
  // 1 is the old behaviour, so an existing policy is unchanged by this
  // existing.
  const [fullWeeks, setFullWeeks] = useState(1);
  const [goldenRoot, setGoldenRoot] = useState("");
  // Blank means this app's own bucket. The curated sets are not always
  // kept beside the week folders.
  const [goldenBucket, setGoldenBucket] = useState("");
  const [useGolden, setUseGolden] = useState(false);
  const [minImages, setMinImages] = useState(25);
  // Whether the two floors below apply at all. A floor is a judgement --
  // "no week ever falls below a tenth" is a claim about what history is
  // worth -- so it is stated rather than always on underneath.
  const [useFloor, setUseFloor] = useState(true);

  // Picking the golden folder rather than typing its prefix.
  const [root, setRoot] = useState("");

  // The data.yaml PREVIEW (2026-10-26). Read through /dataset-yaml, which
  // writes nothing: the chain builds the real one when a run is queued or
  // the schedule fires, after the retention lists. Building and saving it
  // by hand here was a second copy that could disagree with what the run
  // trained on.
  const [built, setBuilt] = useState<Track2DatasetYaml | null>(null);
  const [yamlOpen, setYamlOpen] = useState(false);
  const [yamlBusy, setYamlBusy] = useState(false);
  const [yamlError, setYamlError] = useState("");

  // One file, holding every card's choices. It is written from the LIVE
  // server-side config, which is why it can cover four cards from here
  // without this one knowing anything about them.

  // The settings-file controls that lived here are gone; the pinned
  // toolbar owns them. `settingsFiles` is still LISTED, because the card
  // elsewhere says which file a policy came from.
  const [, setSettingsFiles] = useState<Track2SettingsFile[]>([]);


  // The schedule as a grid: every retrain against every week of data.

  const [matrix, setMatrix] = useState<Track2RetentionMatrix | null>(null);
  const [ahead, setAhead] = useState(10);
  // Shares or counts. Both are the same arithmetic -- `keep_count` on
  // the server decides either way -- but a share answers "how hard is
  // the decay biting" and a count answers "is there enough left to
  // train on", and those are different questions asked on different
  // days.
  const [asImages, setAsImages] = useState(false);

  /** Retention persists as it is changed, like every other card.
   *
   * Saving used to happen only inside the "Saved settings" block, which
   * has been removed -- so without this the policy on screen would never
   * reach the server at all, and the one place that reads it (the
   * unattended chain) would go on using whatever was last stored.
   *
   * `loaded` gates it so the first render cannot write the defaults back
   * over the values it is still fetching.
   */
  const [loadedPolicy, setLoadedPolicy] = useState(false);
  const policyJson = JSON.stringify(currentConfig());
  useEffect(() => {
    if (!loadedPolicy) return;
    const timer = window.setTimeout(() => {
      saveTrack2Retention(JSON.parse(policyJson))
        .then(() => post("data-sources-changed", { from: "retention" }))
        .catch(() => undefined);
    }, 800);
    return () => window.clearTimeout(timer);
  }, [loadedPolicy, policyJson]);

  const load = useCallback(async () => {
    try {
      const got = await getTrack2Retention();
      setSettings(got);
      setEnabled(got.config.enabled);
      setPercent(Math.round(got.config.weekly_rate * 1000) / 10);
      setCurve(got.config.curve);
      setFloor(Math.round(got.config.floor * 100));
      setFullWeeks(Math.max(1, got.config.full_weeks ?? 1));
      // The golden folder is the Golden Set card's (step 2) -- the one the
      // training chain builds and trains on. This card used to keep a
      // second folder of its own that the run never read (2026-11-13).
      const golden = await getTrack2PullControl()
        .then((s) => s.config.golden_set ?? {}).catch(() => null);
      if (golden) {
        setGoldenRoot(golden.target_folder || "");
        setGoldenBucket(golden.target_bucket || "");
      }
      setUseGolden(Boolean(got.config.use_golden));
      setMinImages(got.config.min_images);
      setUseFloor(got.config.use_floor ?? true);
      if (typeof got.config.forecast_weeks === "number") setAhead(got.config.forecast_weeks);
      setAsImages(Boolean(got.config.forecast_as_images));
      // Always Pull Control's root: `retrain_schedule` reads
      // `config["root"]`, so a root picked here only ever previewed a
      // different folder from the one the run trains on.
      setRoot(got.suggested_root);
      // Only now may the auto-save write: before this the controls hold
      // their initial values, and persisting those would overwrite the
      // policy being read in this very call.
      setLoadedPolicy(true);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Could not read retention settings.");
    }
  }, []);

  // On mount, not only on open. The tick beside the title says whether
  // the policy is on, and it is visible while the card is shut -- so an
  // unticked box has to mean OFF and never "not read yet". Loading only
  // on open made every shut card claim the policy was off.
  useEffect(() => { void load(); }, [open, load]);

  // Re-read whenever the policy changes, and after the auto-save has had
  // time to land: the server selects with the SAVED policy, which is the
  // one the chain uses. `enabled` decides file lists or week folders --
  // the same switch the chain reads, so the preview cannot show one
  // while the run builds the other.
  useEffect(() => {
    if (!yamlOpen || !root) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      setYamlBusy(true);
      buildTrack2DatasetYaml({ root, apply_retention: enabled })
        .then((got) => {
          if (cancelled) return;
          setBuilt(got);
          setYamlError("");
        })
        .catch((err) => {
          if (cancelled) return;
          setBuilt(null);
          setYamlError(err instanceof ApiError ? err.detail : "Could not read the weeks.");
        })
        .finally(() => { if (!cancelled) setYamlBusy(false); });
    }, 1200);
    return () => { cancelled = true; window.clearTimeout(timer); };
  }, [yamlOpen, root, enabled, policyJson]);

  const loadSettingsFiles = useCallback(async () => {
    try {
      const got = await getTrack2SettingsFiles();
      setSettingsFiles(got.files);
    } catch {
      setSettingsFiles([]);
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    void loadSettingsFiles();
  }, [open, loadSettingsFiles]);


  /** Every control on the card, as one object. Written once: it is sent
   * by Save, by the preview and by the grid, and three hand-built copies
   * of it drifting apart is how a chart ends up describing a policy
   * nobody set. */
  function currentConfig() {
    return {
      enabled,
      weekly_rate: percent / 100,
      curve,
      full_weeks: fullWeeks,
      use_floor: useFloor,
      floor: floor / 100,
      min_images: minImages,
      // use_golden is NOT here. Step 2 owns it, and this panel only read
      // its copy at page load -- writing that copy back on save would
      // silently undo a switch made on the other card. The previews below
      // add it explicitly, because the bars do depend on it.
      golden_root: goldenRoot,
      golden_bucket: goldenBucket,
      root,
      // The forecast table's choices: saved, so a settings file brings the
      // table back the way it was being read.
      forecast_weeks: ahead,
      forecast_as_images: asImages,
    };
  }


  // The grid, redrawn from the controls rather than the saved settings --
  // its whole purpose is to be looked at BEFORE a policy is committed,
  // and one captioned with last week's numbers would be worse than none.
  // Debounced, because typing "12" into a number field is two changes and
  // the first one is not worth a request.
  useEffect(() => {
    if (!open) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      track2RetentionMatrix(root, ahead, { ...currentConfig(), use_golden: useGolden })
        .then((res) => !cancelled && setMatrix(res))
        .catch(() => undefined);
    }, 250);
    return () => { cancelled = true; window.clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, root, ahead, enabled, percent, curve, floor, fullWeeks,
      useFloor, minImages, useGolden, goldenRoot, goldenBucket]);

  // Once the bars are on screen they follow the controls, same as the
  // grid. They used to hold whatever the last Preview returned, so
  // switching the curve left them where they were -- which reads as the
  // curve making no difference rather than as a stale chart.
  //
  // Always, whenever the card is open and has a root: the bars are the
  // answer to "what would a retrain use today", and a button standing
  // between somebody and that answer was one more thing to remember to
  // press. It lists S3, so it is debounced and runs only while open.
  useEffect(() => {
    if (!open || !root) return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      previewTrack2Retention(root, { ...currentConfig(), use_golden: useGolden })
        .then((got) => !cancelled && setPlan(got))
        .catch(() => undefined);
    }, 400);
    return () => { cancelled = true; window.clearTimeout(timer); };
    // `plan` is deliberately absent: it is what this sets, and including
    // it would make the effect re-run itself for ever.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, root, enabled, percent, curve, floor, fullWeeks, useFloor,
      minImages, useGolden, goldenRoot, goldenBucket]);

  // The bars are scaled against the largest thing drawn, golden set
  // included -- otherwise a golden set larger than any week would run off
  // the end of its row.
  const widest = Math.max(
    1,
    ...(plan?.weeks ?? []).map((w) => w.available),
    plan?.golden?.enabled ? plan.golden.available : 0,
  );

  return (
    // collapsible={false}: this card runs its own Show/Hide,
    // because it shows a summary of live state when shut and
    // the generic one in Card cannot know what to put there.
    <Card
      title="Retention"
      step="4b"
      help={
        <>
          <p>
            This section does four things, in order. They are worth doing in
            that order the first time.
          </p>
          <p className="mt-2">
            <strong>1. Decide the decay.</strong> Older weeks contribute fewer
            images, so a model retrained every week does not drift towards its
            own history. Off, every week counts in full. Nothing is ever
            deleted -- the week folders stay whole and only the training set is
            thinned.
          </p>
          <p className="mt-2">
            <strong>2. Check the data.yaml preview.</strong> This is the file
            the trainer reads, shown as it would be built. Nothing is written
            here: the run builds it when it is queued or the schedule fires,
            applying the retention policy first when it is on -- the exact
            image list is what makes the decay reach the run at all.
          </p>
          <p className="mt-2">
            <strong>3. Let it run.</strong> With{" "}
            <em>PULL, BUILD, QUEUE, TRAIN</em> at the top on a schedule (or
            when files arrive), each pull rebuilds the lists and the yaml from
            the ticked sources, queues a run and starts it -- nothing to press.
          </p>
          <p className="mt-2">
            <em>Save settings</em> near the bottom saves every card&apos;s
            choices -- this one, the pull control and Auto Training -- into one
            named file you can load back.
          </p>
        </>
      }
      collapsible={false}
      caption="Older weeks contribute fewer images. Nothing is deleted -- the week folders stay whole and only the assembled dataset is thinned."
      action={
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
        >
          {open ? "Hide" : "Show"}
        </button>
      }
    >
      {!open ? (
        <>
          {error && <p className="mb-1 text-xs text-critical">{error}</p>}
          <p className="text-xs text-ink-muted">
            {settings
              ? settings.config.enabled
                ? `On -- ${Math.round(settings.config.weekly_rate * 1000) / 10}% a week, ${settings.config.curve}`
                : "Off -- every week counts in full."
              : "Reading the saved policy..."}
          </p>
        </>
      ) : (
        <div className="flex flex-col gap-3">
          {error && <p className="text-xs text-critical">{error}</p>}

          {/* The one on/off for this card. There used to be a second in
              the header that wrote immediately while this one waited for
              Save -- two switches for one setting, with different
              timing. Whichever was the better control, having both was
              the worse card. */}
          <label className="flex items-start gap-2 text-xs text-ink-secondary">
            <input
              type="checkbox"
              className="mt-0.5 shrink-0"
              checked={enabled}
              onChange={(e) => setEnabled(e.target.checked)}
            />
            <span>
              Decay older weeks
              <span className="block text-ink-muted">
                Off, every week counts in full and a year-old week weighs as much
                as last week's. Takes effect on Save.
              </span>
            </span>
          </label>

          <div className="grid gap-2 sm:grid-cols-3">
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">
                Weeks at full weight
              </span>
              <input
                type="number" min={1} max={104}
                className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                value={fullWeeks}
                onChange={(e) => setFullWeeks(Math.max(1, Number(e.target.value) || 1))}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Lost per week (%)</span>
              <input
                type="number" min={0} max={100} step={0.1}
                className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                value={percent}
                onChange={(e) => setPercent(Number(e.target.value))}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs text-ink-muted">Curve</span>
              <select
                className="w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs"
                value={curve}
                onChange={(e) => setCurve(e.target.value as "exponential" | "linear")}
              >
                <option value="exponential">Compounding</option>
                <option value="linear">Linear</option>
              </select>
            </label>
          </div>

          {/* The floors, behind their own switch. They are a judgement,
              not a detail: "no week ever falls below a tenth" is a claim
              about what old data is worth, and it used to be applied
              whether or not anybody had decided it. */}
          <div className="rounded-md border border-hairline p-3">
            <label className="flex items-start gap-2 text-xs text-ink-secondary">
              <input
                type="checkbox"
                className="mt-0.5 shrink-0"
                checked={useFloor}
                onChange={(e) => setUseFloor(e.target.checked)}
              />
              <span>
                Never let a week fall below a minimum
                <span className="block text-ink-muted">
                  Off, the curve alone decides and an old enough week reaches
                  nothing at all.
                </span>
              </span>
            </label>
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-xs text-ink-muted">Never below (%)</span>
                <input
                  type="number" min={0} max={100}
                  disabled={!useFloor}
                  className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs disabled:opacity-40"
                  value={floor}
                  onChange={(e) => setFloor(Number(e.target.value))}
                />
              </label>
              <label className="block">
                <span className="mb-1 block text-xs text-ink-muted">...or images</span>
                <input
                  type="number" min={0}
                  disabled={!useFloor}
                  className="tabular w-full rounded-md border border-black/15 bg-page px-3 py-2 text-xs disabled:opacity-40"
                  value={minImages}
                  onChange={(e) => setMinImages(Number(e.target.value))}
                />
              </label>
            </div>
            <p className="mt-1.5 text-xs text-ink-muted">
              Whichever of the two is the larger. A week thinned to four images
              contributes noise, not history.
            </p>
          </div>

          <p className="text-xs text-ink-muted">
            {fullWeeks > 1 ? (
              <>
                The newest {fullWeeks} weeks all count in full; decay starts on
                the {fullWeeks + 1}
                {fullWeeks + 1 === 2 ? "nd" : fullWeeks + 1 === 3 ? "rd" : "th"}.
                Nothing much changes about a site in a month, so a month of data
                is all equally current -- decaying from the second week thins
                data that is still describing today.{" "}
              </>
            ) : null}
            {curve === "linear"
              ? "Linear reaches exactly zero: at 1% a week, a week is gone 100 steps later. The floor catches it, but a curve that needs a floor to stay sane is the wrong curve."
              : "Compounding never reaches zero and reads as a half-life -- at 1% a week, half a week's images are still in play after 69 steps."}
          </p>

          <p className="text-xs text-ink-muted">
            Weeks from <span className="tabular text-ink-secondary">{root || "(no root set)"}</span>,
            Pull Control&apos;s root.
            {useGolden && goldenRoot && (
              <> Golden set: <span className="tabular text-ink-secondary">{goldenRoot}</span>,
              set in <strong>Step 2, Pick Golden Set / Edge Cases</strong>.</>
            )}
          </p>

          {/* The policy as a schedule rather than a snapshot.
              The preview below answers "what would a retrain use today".
              This answers "what will a retrain use in two months, and is
              week 1 still in it by then" -- which is the question that
              decides whether a policy is the right one, and the one no
              single-column view can show. */}
          <div className="rounded-md border border-black/15 bg-page p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-medium text-ink">
                Every retrain, against every week of data
              </p>
              <div className="flex flex-wrap items-center gap-3">
                <label className="flex items-center gap-1.5 text-xs text-ink-muted">
                  weeks ahead
                  <input
                    type="number" min={0} max={26}
                    className="tabular w-14 rounded border border-black/15 bg-surface px-1.5 py-1 text-xs"
                    value={ahead}
                    onChange={(e) => setAhead(Math.max(0, Math.min(26, Number(e.target.value) || 0)))}
                  />
                </label>
                {/* Disabled until real counts are in hand: a count drawn
                    from an assumed week size is a guess wearing the look
                    of a measurement. */}
                <div
                  className="flex items-center gap-2 text-xs text-ink-muted"
                  title={matrix?.sized
                    ? undefined
                    : "No week folders read yet, so there are no counts to show."}
                >
                  show
                  {([["%", false], ["images", true]] as [string, boolean][]).map(
                    ([label, wantsImages]) => (
                      <label key={label} className="flex items-center gap-1">
                        <input
                          type="radio"
                          name="retention-grid-units"
                          checked={asImages === wantsImages}
                          disabled={wantsImages && !matrix?.sized}
                          onChange={() => setAsImages(wantsImages)}
                        />
                        {label}
                      </label>
                    ),
                  )}
                </div>
              </div>
            </div>

            {(
              !matrix ? (
                <p className="mt-2 text-xs text-ink-muted">Working it out...</p>
              ) : (
                <>
                  <p className="mt-1 text-xs text-ink-muted">
                    Down the side is the week a retrain is <strong>deployed</strong>;
                    across the top is the week the data was{" "}
                    <strong>collected</strong>. Each cell is how much of that week
                    the retrain trains on. Blank means that week did not exist yet;
                    an empty folder is stepped over, and does not age the weeks
                    before it -- age counts weeks of collection, not weeks of
                    calendar.
                    {matrix.ahead > 0 && (
                      <> The last {matrix.ahead} rows are retrains that have not
                      happened yet.</>
                    )}
                  </p>

                  <div className="mt-2 overflow-x-auto">
                    <table className="tabular border-collapse text-[11px]">
                      <thead>
                        <tr>
                          <th className="sticky left-0 z-10 bg-page px-2 py-1 text-left font-medium text-ink-muted">
                            deployed
                          </th>
                          {matrix.golden && (
                            <th className="px-2 py-1 text-center font-medium text-ink-muted">
                              gold
                            </th>
                          )}
                          {matrix.weeks.map((w) => (
                            <th key={w} className="px-2 py-1 text-center font-medium text-ink-muted">
                              {w}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {matrix.rows.map((row) => (
                          <tr key={row.week}>
                            <th
                              className={`sticky left-0 z-10 whitespace-nowrap bg-page px-2 py-1 text-left font-medium ${
                                row.existing ? "text-ink" : "text-ink-muted italic"
                              }`}
                              title={row.existing ? undefined : "not collected yet"}
                            >
                              {row.week}
                            </th>
                            {matrix.golden && (
                              <td
                                className="px-2 py-1 text-center font-medium"
                                style={{ background: "#f0b429", color: "#231a05" }}
                                title={`The golden set never decays.`
                                  + `${matrix.golden_available
                                    ? ` ${matrix.golden_available.toLocaleString()} images.`
                                    : ""}`}
                              >
                                {asImages
                                  ? (matrix.golden_available
                                    ? matrix.golden_available.toLocaleString()
                                    : "--")
                                  : "100%"}
                              </td>
                            )}
                            {matrix.weeks.map((w, index) => {
                              const cell = row.cells[index];
                              if (!cell) {
                                return <td key={w} className="px-2 py-1" />;
                              }
                              return (
                                <td
                                  key={w}
                                  className="px-2 py-1 text-center"
                                  style={cell.empty
                                    ? { background: "hsl(336 62% 90%)", color: "#7c2148" }
                                    : shade(cell.effective)}
                                  title={`${cell.week} is ${cell.age_weeks} week`
                                    + `${cell.age_weeks === 1 ? "" : "s"} old by ${row.week}`
                                    + `${cell.steps ? `, ${cell.steps} step${cell.steps === 1 ? "" : "s"} of decay` : ""}`
                                    + `${cell.empty ? ". That week holds no images" : ""}`
                                    + `${cell.under_minimum
                                      ? `. It holds ${cell.available} images, fewer than the`
                                        + ` ${matrix.min_images}-image minimum, so it is kept`
                                        + ` whole however old it gets`
                                      : ""}`
                                    + `${cell.floored && !cell.empty
                                      ? `. The curve alone would give ${share(cell.fraction)}%;`
                                        + ` the floor holds it at ${share(cell.effective)}%`
                                      : ""}`}
                                >
                                  {cell.empty ? (
                                    // The folder exists and holds nothing.
                                    // Named rather than shown as 0%, which
                                    // would read as a week the decay had
                                    // emptied rather than one never filled.
                                    <span className="whitespace-nowrap">
                                      &#9723; empty
                                    </span>
                                  ) : asImages ? (
                                    cell.keep.toLocaleString()
                                  ) : (
                                    `${share(cell.effective)}%`
                                  )}
                                  {/* A dot, not a colour: the cell is
                                      already coloured by its value, and a
                                      floored cell is not a different
                                      value -- it is the same value
                                      arrived at differently. */}
                                  {!cell.empty && (cell.under_minimum ? (
                                    <span className="opacity-60"> &#9723;</span>
                                  ) : cell.floored ? (
                                    <span className="opacity-60"> &bull;</span>
                                  ) : null)}

                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {!matrix.enabled && (
                    <p className="mt-2 text-xs text-ink-muted">
                      Retention is off, so every cell is 100%. Tick
                      &ldquo;Decay older weeks&rdquo; above to see the policy
                      take effect.
                    </p>
                  )}
                  {matrix.enabled && matrix.use_floor && (
                    <p className="mt-2 text-xs text-ink-muted">
                      Cells marked &bull; are held up by a floor rather than set
                      by the curve, and &#9723; marks a week holding fewer images
                      than the minimum, which is therefore kept whole however old
                      it gets
                      {" "}&mdash;{" "}
                      <span className="tabular">{share(matrix.floor)}%</span>
                      {matrix.min_images > 0 && (
                        <> or <span className="tabular">{matrix.min_images}</span> images,
                        whichever is the larger</>
                      )}
                      .
                      {matrix.sized ? (
                        <>
                          {" "}Both are applied here against what each week
                          actually holds.
                          {matrix.ahead > 0 && (
                            <> The projected weeks are taken to hold{" "}
                            <span className="tabular">
                              {matrix.assumed_size.toLocaleString()}
                            </span>{" "}
                            images each, the median of the weeks that exist.</>
                          )}
                        </>
                      ) : (
                        <>
                          {" "}No week folders could be read, so only the
                          percentage is in play here -- the image minimum depends
                          on how big a week is.
                        </>
                      )}
                    </p>
                  )}
                  {matrix.enabled && (
                    <p className="mt-2 text-xs text-ink-muted">
                      {matrix.curve === "linear" ? "Linear" : "Compounding"} is{" "}
                      <span className="tabular">
                        {(matrix.curve_gap * 100).toFixed(
                          matrix.curve_gap >= 0.01 ? 1 : 2,
                        )}
                      </span>{" "}
                      points from {matrix.other_curve} at the widest point of this
                      window
                      {matrix.curve_gap < 0.01 && (
                        <> -- at {percent}% a week the two barely part company
                        until much later, so the difference lives in the
                        decimals</>
                      )}
                      .
                    </p>
                  )}
                  {matrix.enabled && matrix.golden && (
                    <p className="mt-2 text-xs text-ink-muted">
                      The gold column{" "}
                      {matrix.golden_root ? (
                        <>
                          is <span className="tabular">{matrix.golden_root}</span>
                          {matrix.golden_bucket && (
                            <> in <span className="tabular">{matrix.golden_bucket}</span></>
                          )}.
                        </>
                      ) : (
                        <>has no folder chosen yet -- pick one above.</>
                      )}{" "}
                      It has no collection week, so it has no age and never decays.
                    </p>
                  )}
                </>
              )
            )}
          </div>

          {/* No Save here any more. There is ONE save, below the bars,
              and it saves every card -- two buttons called the same thing
              on one card, one of which saved a quarter of it, is how a
              policy gets half-written. No Preview button either: the bars
              below load on their own and follow the controls, sending
              them as they read -- nothing needs saving first. */}
          {!plan && (
            <p className="text-xs text-ink-muted">
              {root ? "Reading the week folders..." : "Pick a root folder to see what a retrain would use."}
            </p>
          )}

          {plan && (
            <div className="rounded-md border border-black/15 bg-page p-3">
              <p className="mb-2 text-xs text-ink-muted">
                Age counts newer non-empty collections, not elapsed calendar weeks.
                Empty collections and time without new data do not advance decay.
                Decay selects images; it does not change their training loss weights.
              </p>
              {plan.composition && <div className="mb-3 text-xs">
                <p className="font-medium">Training contribution: {plan.composition.total.toLocaleString()} images</p>
                <p>Recent: {plan.composition.recent.toLocaleString()} · Historical: {plan.composition.historical.toLocaleString()}
                  {" "}· Golden: {plan.composition.golden.toLocaleString()}</p>
                <p>Historical share: {(plan.composition.historical_share * 100).toFixed(1)}%.</p>
                <p className="text-ink-muted">{plan.composition.scope}</p>
                {plan.composition.historical > plan.composition.recent && plan.composition.recent > 0 &&
                  <p className="text-critical">Historical images outnumber recent images despite the decay policy.</p>}
              </div>}
              <p className="mb-2 text-xs font-medium text-ink">
                {plan.keep.toLocaleString()} of {plan.available.toLocaleString()} training images,
                {" "}{plan.dropped.toLocaleString()} left out
                {!plan.enabled && " (decay is off -- everything counts)"}
                {(plan.val_available ?? 0) > 0 && (
                  <span className="font-normal text-ink-muted">
                    {" "}· {(plan.val_available ?? 0).toLocaleString()} validation images kept in full
                  </span>
                )}
              </p>
              <div className="flex flex-col gap-1">
                {/* Drawn first and always full. It is not a week -- it has
                    no collection date, so it has no age and nothing to
                    decay -- so it sits above the series rather than in
                    it, and stays whole however far down the weeks go. */}
                {plan.golden?.enabled && (
                  <div className="flex items-center gap-2 text-xs">
                    <span
                      className="tabular w-24 shrink-0 truncate font-medium text-ink"
                      title={plan.golden.root}
                    >
                      golden set
                    </span>
                    <span className="tabular w-16 shrink-0 text-ink-muted">
                      no age
                    </span>
                    <span className="relative h-3 min-w-0 flex-1 rounded-sm bg-black/5">
                      <span
                        className="absolute inset-y-0 left-0 rounded-sm"
                        style={{
                          width: `${(plan.golden.available / widest) * 100}%`,
                          background: "#f0b429",
                        }}
                      />
                    </span>
                    <span className="tabular w-28 shrink-0 text-right text-ink-secondary">
                      {plan.golden.available.toLocaleString()} /{" "}
                      {plan.golden.available.toLocaleString()}
                    </span>
                    <span className="tabular w-16 shrink-0 text-right text-ink-muted">
                      {share(1)}%
                    </span>
                  </div>
                )}
                {plan.weeks.map((w) => (
                  <div key={w.week} className="flex items-center gap-2 text-xs">
                    <span className="tabular w-24 shrink-0 truncate font-medium text-ink">
                      {w.week}
                    </span>
                    <span className="tabular w-16 shrink-0 text-ink-muted">
                      {w.age_weeks} batches old
                    </span>
                    {/* Kept against available, so the thinning is visible
                        as a shortfall rather than a number to subtract. */}
                    <span className="relative h-3 min-w-0 flex-1 rounded-sm bg-black/5">
                      <span
                        className="absolute inset-y-0 left-0 rounded-sm bg-black/10"
                        style={{ width: `${(w.available / widest) * 100}%` }}
                      />
                      <span
                        className="absolute inset-y-0 left-0 rounded-sm bg-accent"
                        style={{ width: `${(w.keep / widest) * 100}%` }}
                      />
                    </span>
                    <span
                      className="tabular w-28 shrink-0 text-right text-ink-secondary"
                      title={(w.val ?? 0) > 0
                        ? `Training images. Plus ${w.val} validation images, kept in full.`
                        : "Training images."}
                    >
                      {w.keep.toLocaleString()} / {w.available.toLocaleString()}
                    </span>
                    <span className="tabular w-16 shrink-0 text-right text-ink-muted">
                      {(w.val ?? 0) > 0 ? `+${w.val} val` : ""}
                    </span>
                    <span
                      className="tabular w-16 shrink-0 text-right text-ink-muted"
                      title={w.effective !== undefined && w.effective > w.fraction
                        ? `The curve alone would give ${share(w.fraction)}%;`
                          + ` a floor holds it at ${share(w.effective)}%`
                        : undefined}
                    >
                      {share(w.effective ?? w.fraction)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}


          {/* ------------------------------------------- data.yaml ----- */}
          {/* A PREVIEW, like "What will be built" on Assemble a dataset
              (2026-10-26). Build it, Save to S3 and Use for training are
              gone: the run builds this file itself -- the retention lists
              first when the policy is on, then the yaml pointing at them. */}
          <div className="rounded-md border border-hairline p-3">
            <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs font-medium text-ink">
                data.yaml for training{" "}
                <span className="font-normal text-ink-muted">
                  (preview -- points at the weeks where they are)
                </span>
              </p>
              <button type="button" onClick={() => setYamlOpen((v) => !v)}
                      className="text-xs font-medium text-ink-secondary underline">
                {yamlOpen ? "Hide" : "Show"}
              </button>
            </div>

            {yamlOpen && (
              <>
                <p className="mb-2 text-xs text-ink-muted">
                  <span className="font-medium text-ink-secondary">
                    Preview only -- nothing is written.
                  </span>{" "}
                  The run builds this data.yaml itself when you press{" "}
                  <em>Add to Queue</em> and each time the schedule fires.{" "}
                  {enabled ? (
                    <>
                      The retention policy is on, so it is applied first:{" "}
                      <span className="tabular">_retained/train.txt</span> and{" "}
                      <span className="tabular">val.txt</span> are written holding
                      exactly the images the decay kept, and the data.yaml points
                      at those lists.
                    </>
                  ) : (
                    <>
                      The retention policy is off, so the data.yaml names every
                      week folder whole.
                    </>
                  )}{" "}
                  Nothing is copied: Ultralytics finds each label by swapping{" "}
                  <span className="tabular">/images/</span> for{" "}
                  <span className="tabular">/labels/</span>. Updates as the
                  policy above changes.
                </p>

                {yamlBusy && (
                  <p className="mb-1 text-xs text-ink-muted">checking the weeks...</p>
                )}
                {yamlError && <p className="mb-1 text-xs text-critical">{yamlError}</p>}
                {!root && (
                  <p className="text-xs text-ink-muted">
                    No root yet -- set the folder the weeks sit under above.
                  </p>
                )}

                {built && (
                  <>
                    <pre className="tabular overflow-x-auto rounded-md border border-black/15 bg-page p-2 text-xs leading-5 text-ink-secondary">
{yamlTree(built, root)}
                    </pre>

                    {built.problems.map((problem) => (
                      <p key={problem.text}
                         className={`mt-1 text-xs ${
                           problem.level === "blocking" ? "text-critical" : "text-ink-secondary"
                         }`}>
                        {problem.level === "blocking" ? "Blocking: " : "Note: "}
                        {problem.text}
                      </p>
                    ))}

                    {/* One row per week, as Assemble lists one per source:
                        a single total hides an empty week among full ones. */}
                    {built.retained ? (
                      <div className="mt-2 flex flex-col gap-0.5">
                        {built.retained.weeks
                          .filter((w) => w.available > 0)
                          .map((w) => (
                            <p key={w.week} className="tabular text-xs text-ink-secondary">
                              <span className="inline-block w-24 font-medium text-ink">{w.week}</span>
                              <span className="inline-block w-16">{w.age_weeks}w old</span>
                              {w.keep.toLocaleString()} of {w.available.toLocaleString()} kept
                              {w.keep < w.available && (
                                <span className="text-ink-muted">
                                  {"  "}-- {(w.available - w.keep).toLocaleString()} left out
                                </span>
                              )}
                            </p>
                          ))}
                      </div>
                    ) : (
                      <div className="mt-2 flex flex-col gap-0.5">
                        {[["train", built.train_weeks], ["val", built.val_weeks]]
                          .flatMap(([split, weeks]) =>
                            (weeks as Track2DatasetYaml["train_weeks"]).map((w) => (
                              <p key={`${split}-${w.week}`}
                                 className={`tabular text-xs ${w.unlabelled ? "text-critical" : "text-ink-secondary"}`}>
                                <span className="inline-block w-10 font-medium text-ink">{split as string}</span>
                                <span className="inline-block w-24">{w.week}</span>
                                {w.images.toLocaleString()} images, {w.labels.toLocaleString()} labels
                                {w.unlabelled && <span className="ml-2">no labels</span>}
                              </p>
                            )))}
                      </div>
                    )}

                    <p className="tabular mt-2 text-xs text-ink">
                      Total:{" "}
                      <span className="font-medium">
                        {(built.retained?.train ?? built.train_images).toLocaleString()}
                      </span>{" "}
                      training,{" "}
                      <span className="font-medium">
                        {(built.retained?.val ?? built.val_images).toLocaleString()}
                      </span>{" "}
                      evaluation images
                      {built.retained && (
                        <span className="text-ink-muted">
                          {" "}(of {built.retained.train_available.toLocaleString()} training
                          available -- the val side is taken whole, so every run is
                          scored on the same images)
                        </span>
                      )}
                    </p>

                    <details className="mt-2">
                      <summary className="cursor-pointer text-xs text-ink-secondary">
                        The data.yaml text
                      </summary>
                      <pre className="tabular mt-1 max-h-72 overflow-auto rounded-md border border-black/15 bg-page p-2 text-xs leading-5 text-ink-secondary">
{built.yaml}
                      </pre>
                    </details>
                  </>
                )}
              </>
            )}
          </div>

          {/* Saved settings removed. The pinned toolbar at the top of
              the page owns the settings file now -- New, Save as, Load
              and Delete -- and this was a second set of controls over
              one file, which is how two panels end up disagreeing about
              which file you are in. */}

        </div>
      )}
    </Card>
  );
}


/** The preview as a tree, the shape "What will be built" draws on the
 * Assemble card: the root, the yaml, and what each split names. */
function yamlTree(built: Track2DatasetYaml, root: string): string {
  const names = Object.values(built.classes.names ?? {}).join(", ") || "(none found)";
  const lines = [`${root || "<root>"}/`, `├── data.yaml        names: ${names}`];
  if (built.retained) {
    const r = built.retained;
    lines.push(
      `├── _retained/train.txt  ${r.train.toLocaleString()} of `
        + `${r.train_available.toLocaleString()} training images (the decay's pick)`,
      `├── _retained/val.txt    ${r.val.toLocaleString()} evaluation images (all of them)`,
    );
  }
  const weeks = Array.from(new Set([
    ...built.train_weeks.map((w) => w.week),
    ...built.val_weeks.map((w) => w.week),
  ])).sort();
  weeks.forEach((week, i) => {
    const last = i === weeks.length - 1;
    const train = built.train_weeks.find((w) => w.week === week);
    const val = built.val_weeks.find((w) => w.week === week);
    const parts = [
      train ? `train ${train.images.toLocaleString()}` : "",
      val ? `val ${val.images.toLocaleString()}` : "",
    ].filter(Boolean).join(" · ");
    lines.push(`${last ? "└──" : "├──"} ${week}/`.padEnd(24) + ` ${parts} images`);
  });
  if (!weeks.length) lines.push("└── (no week folders found)");
  return lines.join("\n");
}
