import { useEffect, useState } from "react";
import {
  getTrack2ChainStatus,
  type Track2ChainStatus,
  type Track2GpuJob,
} from "../lib/api";

/** The one timestamp left on this bar is written UTC-aware by the
 * server, and everyone reading it is on Singapore time. Slicing the ISO
 * string -- which is what this did -- showed 00:44 for a pull the person
 * had scheduled for 08:42, which reads as "it did not run".
 *
 * Clock only, no date: everything on this bar is from today. The dated
 * form went with the "Last retrain finished" line it existed for. */
function localTime(iso: string): string {
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return iso.slice(0, 16);
  return at.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/** How long the GPU has been held, and since when.
 *
 * Read from the holding job's `TrainingStartTime`, which is also the
 * moment SageMaker starts BILLING -- so this is elapsed paid time, not
 * time since the job was submitted. A job waiting on capacity has no
 * start time at all, holds no hardware and costs nothing, and gets said
 * so rather than being given a timer counting from a moment that has
 * not happened.
 *
 * Only the jobs that could actually block a run: a managed-spot job
 * draws on a different quota, so counting it would time something that
 * is not in the way.
 */
function gpuHeldFor(jobs: Track2GpuJob[] | undefined): string {
  const blocking = (jobs ?? []).filter((j) => !j.spot);
  if (!blocking.length) return "";
  const started = blocking
    .map((j) => (j.start_time ? new Date(j.start_time).getTime() : NaN))
    .filter((t) => !Number.isNaN(t));
  if (!started.length) return "waiting for capacity -- no instance yet";
  // The EARLIEST, when more than one is somehow running: the slot has
  // been held since the first of them, not the latest.
  const from = Math.min(...started);
  const minutes = Math.max(0, Math.floor((Date.now() - from) / 60000));
  const clock = new Date(from).toLocaleTimeString(undefined,
    { hour: "2-digit", minute: "2-digit" });
  return `${minutes}m since ${clock}`;
}

/** A thin strip across the top: what the pull is doing, is the GPU free,
 * and how far has the unattended chain got.
 *
 * Sits directly under the PULL, BUILD, QUEUE, TRAIN row, because it is
 * the answer to the question that row raises. A chain takes twenty
 * minutes end to end and the GPU slot is a single one for the whole
 * account, so "queued" and "running" are different answers to "when will
 * this finish" only if you can see which.
 *
 * The PULL is here for a different reason: it can run with no chain
 * behind it at all, and when it does, this is the only place on screen
 * that says so. The Pull Control card has a far better progress banner,
 * but it only looks for a running sweep while the card is expanded --
 * so a scheduled pull could move 1,600 files with the page showing
 * nothing.
 *
 * Polls. There is nothing to push with, and the endpoint is one config
 * read plus one SageMaker list.
 */
export function Track2ChainStatusBar() {
  const [status, setStatus] = useState<Track2ChainStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    const tick = async () => {
      try {
        const got = await getTrack2ChainStatus();
        if (!cancelled) setStatus(got);
      } catch {
        // A status bar that cannot read its own state is not worth an
        // error across the top of every page.
      }
      // Faster while something is happening, slow when idle: a chain mid
      // flight is worth watching, an idle one is not worth the calls.
      // A PULL counts as something happening -- it is the stage that
      // moves thousands of files and the one most often left unwatched.
      const live = status?.running || status?.pull?.status === "running"
        || status?.gpu?.busy || status?.cpu?.busy;
      if (!cancelled) timer = window.setTimeout(tick, live ? 3000 : 20000);
    };
    void tick();
    return () => { cancelled = true; window.clearTimeout(timer); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status?.running, status?.pull?.status, status?.gpu?.busy, status?.cpu?.busy]);

  if (!status) return null;

  // The two machines a run uses, on one line. They used to be two
  // places: this label, and a "SageMaker GPU resource" card further down
  // the page that said "Available" in green from a page-load snapshot
  // while a job was downloading its dataset. One indicator, polled, with
  // the card's detail kept as the tooltip.
  const gpu = status.gpu;
  const gpuLabel = !gpu.known
    ? "GPU unknown"
    : !gpu.busy
      ? "GPU free"
      : gpu.ours
        ? "GPU in use -- this pipeline"
        : "GPU in use -- another job";
  const gpuTone = !gpu.known
    ? "text-ink-muted"
    : !gpu.busy
      ? "text-emerald-600"
      : gpu.ours
        ? "text-accent"
        : "text-critical";
  const gpuHeld = gpuHeldFor(gpu.jobs);
  const gpuDetail = gpu.jobs?.length
    ? gpu.jobs.map((j) => `${j.name} -- ${j.own_pipeline ? "this app's own pipeline" : "a different system"}`).join("\n")
    : gpu.error || "No training job is holding the GPU.";

  // The OTHER half of a run, and the reason "a run is going" and "the
  // GPU is free" are both true for seven or eight minutes at a time:
  // evaluation is a processing job on an ml.m5.xlarge, started only
  // after training releases the GPU. Without this the two readings look
  // like a contradiction and the honest answer is invisible.
  const cpu = status.cpu;
  const cpuBusy = Boolean(cpu?.busy);
  const cpuLabel = !cpuBusy
    ? ""
    : cpu.step === "EvaluateYolo"
      ? "CPU in use -- evaluating"
      : cpu.step
        ? `CPU in use -- ${cpu.step}`
        : "CPU in use";
  const cpuDetail = cpu?.jobs?.length
    ? cpu.jobs.map((j) => `${j.name}${j.instance_type ? ` -- ${j.instance_type}` : ""}`).join("\n")
    : "";

  // The sweep itself. First on the row because it runs BEFORE every
  // chain stage and, unlike them, can run entirely on its own -- a
  // scheduled pull with nothing queued behind it used to leave this bar
  // reading "No retrain yet" while 1,600 files were moving.
  const pull = status.pull;
  const pulling = pull?.status === "running";
  const pullPct = pulling && pull.total > 0
    ? Math.min(100, Math.round((pull.done / pull.total) * 100))
    : 0;

  return (
    /* A third row of the pinned card, matching the Pull row above it --
       it used to be a full-width strip across the page, which needed its
       own background and bottom border. */
    <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-hairline pt-1.5 text-xs">
      {/* The two machines, stacked rather than side by side. A run uses
          them IN TURN -- TrainYolo holds the GPU, then EvaluateYolo runs
          for seven or eight minutes on an ml.m5.xlarge -- so "GPU free"
          and "evaluating" are consecutive states of one run, not two
          independent facts. On its own line directly under the GPU the
          pair reads as a sequence; inline among the pull and the stages
          it read as another chip in a row of chips.

          A column INSIDE the row, not a full-width break: making it
          `basis-full` would push the pull, the retrain line and all
          eight stages onto a second line, which is a bigger change to
          the bar than "put the evaluation status under the GPU one". */}
      <span className="flex flex-col gap-0.5 self-start">
        <span className={`font-medium ${gpuTone}`} title={gpuDetail}>{gpuLabel}</span>
        {/* How long it has been held. Muted and beside the label rather
            than inside it: the label answers "can I start a run", this
            answers "how much longer", and they are read at different
            moments. */}
        {gpu.busy && gpuHeld && (
          <span className="tabular text-ink-muted">{gpuHeld}</span>
        )}
        {cpuBusy && (
          <span className="font-medium text-accent" title={cpuDetail}>{cpuLabel}</span>
        )}
      </span>

      {pulling ? (
        <span className="flex items-center gap-2">
          <span className="font-medium text-accent">
            Pulling {pull.done.toLocaleString()} / {pull.total.toLocaleString()}
            {pull.trigger === "schedule" ? " (scheduled)" : ""}
          </span>
          <span className="h-1 w-20 overflow-hidden rounded-full bg-black/10">
            <span className="block h-full bg-accent" style={{ width: `${pullPct}%` }} />
          </span>
          {pull.errors > 0 && (
            <span className="text-critical">{pull.errors} failed</span>
          )}
        </span>
      ) : pull?.finished_at ? (
        <span className={pull.status === "failed" ? "text-critical" : "text-ink-secondary"}>
          Pull {pull.status === "failed" ? "failed" : pull.status === "stopped" ? "stopped" : "done"}{" "}
          {localTime(pull.finished_at)} &middot;{" "}
          {pull.moved.toLocaleString()} copied
          {pull.errors > 0 ? `, ${pull.errors} failed` : ""}
        </span>
      ) : (
        <span className="text-ink-muted">No pull yet</span>
      )}

      {/* The silent one. A pull that moved thousands of files and is
          followed by nothing looks exactly like one about to retrain. */}
      {pull && !pull.chains && (
        <span className="text-ink-muted" title="Queue after pull is off, so a pull does not start a retrain.">
          nothing follows a pull
        </span>
      )}

      {/* "Last retrain finished <date>" sat here, with "Retrain running"
          and "No retrain yet" as its other two states. Removed for the
          same reason as the stage ticks below: the chain is the fast
          part. It takes about six seconds, so "running" was never on
          screen long enough to read, and what remained was a date
          restating the pull line beside it -- the pull and the chain
          that follows it finish within a minute of each other, so two
          timestamps a few pixels apart said one thing twice.

          The chain's own card keeps the full account, step by step and
          with the reason when one fails, which is where a finished
          retrain is actually read. */}

      {/* The eight-stage tick list used to run along here: Checks,
          Retention, data.yaml, Freeze snapshot, Set as training data,
          Base model, Queue, Start. Removed deliberately.

          It was built to answer "how long left" by showing which stages
          were still to come, and it never did: the whole chain takes
          about six SECONDS, so the row was a full set of ticks before
          anybody could read it. What it actually did was take up the
          width of the bar restating that the parts which always succeed
          had succeeded, next to the two lines that carry real news --
          which machine is working, and what the pull moved.

          Nothing is lost. `chain-status` still returns `stages`, and the
          Retention card prints the same steps with their detail under
          "Last run", which is where a chain that stopped half way should
          be read anyway -- it has the room to say WHY. */}
    </div>
  );
}
