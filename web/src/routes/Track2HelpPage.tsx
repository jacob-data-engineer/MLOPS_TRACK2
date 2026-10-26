import { ExternalLink, HelpCircle } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";
import { useSearchParams } from "react-router-dom";
import { Card } from "../components/Card";

/** Track2 Help -- a walkthrough per page, written for somebody using it.
 *
 * It used to be four cards of dataset-folder reference, and all four were
 * wrong in ways nobody would catch from reading them: it documented
 * `train/images` when the layout is `images/train` (labels are found by
 * swapping `/images/` for `/labels/`, so the other way round does not
 * resolve at all), it documented `nc:`, which nothing in this codebase
 * reads or writes, and it explained the Evaluation queue, a tab removed
 * from the product. Stale reference is worse than none: it reads
 * authoritative and costs a day.
 *
 * So it is a set of walkthroughs -- one per page somebody actually has to
 * operate, plus one for the housekeeping that cuts across them -- each a
 * numbered sequence with a tick box, because the thing being asked for is
 * "walk me through it" rather than "tell me what the fields mean". The
 * ticks are a per-browser convenience and are remembered in localStorage;
 * they are not state anybody else sees, and the page renders correctly
 * when that throws.
 *
 * Rewritten 2026-10-20 against the code as it stands: the three training
 * sources are independent ticks that combine into one data.yaml, the
 * Assemble card has one "Generate yaml training file" button, frozen
 * snapshots are picked and deleted from the run card, and the unattended
 * chain's triggers are a setting with two guards.
 *
 * **Open in a window** pops this out at `/track2/help/popout`, a route
 * registered OUTSIDE the Layout so it renders without the sidebar. The
 * point is to keep it beside the page it describes rather than
 * alternating between two tabs and losing your place in both.
 */

type StepList = { title: string; body: ReactNode }[];

const TABS = [
  { key: "training", label: "Multi-Auto Training" },
  { key: "metrics", label: "Review & Metrics" },
  { key: "notifications", label: "Notifications" },
  { key: "housekeeping", label: "Housekeeping" },
] as const;

type TabKey = (typeof TABS)[number]["key"];

const K = "tabular text-ink";

/** Remembered per browser so a walkthrough survives the reload it tells
 * you to do. Wrapped, because site data can be blocked or cleared and a
 * help page must not be the thing that breaks when it is. */
function readTicks(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem("track2-help-ticks") || "{}");
  } catch {
    return {};
  }
}

function writeTicks(next: Record<string, boolean>) {
  try {
    localStorage.setItem("track2-help-ticks", JSON.stringify(next));
  } catch {
    // Private window, blocked site data. The ticks stop persisting; the
    // page still works.
  }
}

function Steps({ id, steps }: { id: string; steps: StepList }) {
  const [ticks, setTicks] = useState<Record<string, boolean>>(readTicks);

  function toggle(key: string) {
    const next = { ...ticks, [key]: !ticks[key] };
    setTicks(next);
    writeTicks(next);
  }

  const done = steps.filter((_s, i) => ticks[`${id}.${i}`]).length;

  return (
    <div className="flex flex-col gap-3">
      <p className="text-xs text-ink-muted">
        {done} of {steps.length} ticked off.{" "}
        {done > 0 && (
          <button
            type="button"
            onClick={() => {
              const next = { ...ticks };
              steps.forEach((_s, i) => delete next[`${id}.${i}`]);
              setTicks(next);
              writeTicks(next);
            }}
            className="underline"
          >
            Start again
          </button>
        )}
      </p>
      <ol className="flex flex-col gap-3">
        {steps.map((step, i) => {
          const key = `${id}.${i}`;
          const ticked = Boolean(ticks[key]);
          return (
            <li
              key={key}
              className={`rounded-sm border border-hairline bg-page p-3 ${ticked ? "opacity-60" : ""}`}
            >
              <label className="flex cursor-pointer items-start gap-2">
                <input
                  type="checkbox"
                  checked={ticked}
                  onChange={() => toggle(key)}
                  className="mt-1"
                />
                <span className="min-w-0">
                  <span className="block text-sm font-semibold text-ink">
                    {i + 1}. {step.title}
                  </span>
                  <span className="mt-1 block text-sm text-ink-secondary">{step.body}</span>
                </span>
              </label>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

// --- Multi-Auto Training -------------------------------------------------

const TRAINING_STEPS: StepList = [
  {
    title: "Set up a Model Name, and pick its model and size",
    body: (
      <>
        Step 1, <strong>Model types and Model Names</strong>. A <em>Model Name</em> is the
        thing you are detecting &mdash; helmet, vest &mdash; and its class
        list is what every dataset is checked against. Pick the model
        (YOLO11 or YOLO26) and the size (S, M, L, X), and the card names the
        registry group that combination files into, for example{" "}
        <span className={K}>yolo11s-helmet-detector</span>. Every trained
        version lands there, numbered v1, v2, v3 in order; a different size
        is a different group with its own numbering. Use{" "}
        <strong>Define a new Model Name</strong> for a new one.
      </>
    ),
  },
  {
    title: "Give it something to start from",
    body: (
      <>
        A run always continues from a version already registered for that
        Model Name &mdash; pick it under <strong>Train on an existing
        model</strong>. A brand new Model Name has none yet, so use{" "}
        <strong>Import a pretrained COCO or an already-trained model</strong>{" "}
        first: it files a <span className={K}>.pt</span> you already have
        (the COCO <span className={K}>yolo11s.pt</span>, or weights somebody
        already trained) as the first version, marked as a <em>seed</em>.
        Importing does not train; it gives the run something to continue
        from.
      </>
    ),
  },
  {
    title: "Set how every run trains",
    body: (
      <>
        Still step 1: epochs, image size, batch, seed, patience, and the
        advanced settings. They are saved server-side and are what the
        unattended runs use too, so set them here once. Check epochs before
        you expect a usable model &mdash; a handful of epochs trains and
        completes but will not produce one.{" "}
        <strong>Freeze layers is not yet effective</strong>: the field is
        there, but the value it sends is not applied by the trainer, so do
        not rely on it.
      </>
    ),
  },
  {
    title: "Choose the training data sources -- any combination",
    body: (
      <>
        There are three sources, each ticked <em>on its own card</em>, and
        every ticked one is combined into one{" "}
        <span className={K}>data.yaml</span>:
        <span className="mt-1 block">
          &bull; <strong>Use these folders</strong> on{" "}
          <em>Assemble a dataset</em> &mdash; folders you pick by hand.
        </span>
        <span className="block">
          &bull; <strong>Use Pick Golden Set / Edge Cases</strong> on step 2
          &mdash; a curated folder split into train and val by the slider.
          Its <em>val</em> side is the held-out evaluation set, fixed from
          run to run so scores stay comparable.
        </span>
        <span className="block">
          &bull; <strong>Use Pull Control</strong> on step 3 &mdash; the
          week folders the pull fills. Unticking it switches Pull Control,
          Auto Training and Retention off together.
        </span>
        <span className="mt-1 block">
          The Assemble card reads them back as &ldquo;data.yaml will be built
          from: &hellip;&rdquo;. If that says <em>nothing</em>, the run has no
          training data. <strong>Do not add a Pull Control week folder on
          the Assemble card as well</strong> &mdash; it is then listed twice
          and trains twice.
        </span>
      </>
    ),
  },
  {
    title: "Assemble a dataset: pick folders and generate the yaml",
    body: (
      <>
        Pick the image folders (labels must sit beside them as{" "}
        <span className={K}>labels/</span> in place of{" "}
        <span className={K}>images/</span>). Folders can come from more
        than one bucket, and each shows its image count. Nothing is copied:
        one <span className={K}>data.yaml</span> is written naming the
        folders where they already are. Then press{" "}
        <strong>Generate yaml training file</strong> &mdash; the one button.
        It shows the yaml and writes it, or stops and shows the decision it
        needs first (an existing yaml to replace, labels it cannot name, no
        validation folder).
      </>
    ),
  },
  {
    title: "Settle what the label files contain",
    body: (
      <>
        The yaml declares the Model Name&rsquo;s classes. If the label files
        use more classes than that, fill in <strong>Label classes</strong>{" "}
        (every class, in index order) so the yaml can name them all and the
        run trains only the Model Name&rsquo;s. Or tick{" "}
        <strong>Filter undeclared classes in the training container</strong>:
        boxes of classes the yaml does not name are dropped from the
        run&rsquo;s own downloaded copy before training &mdash; the label
        files in S3 are never changed. Unticked, a yaml that names fewer
        classes than the labels use is <em>refused</em>, not silently
        trained.
      </>
    ),
  },
  {
    title: "Point Pull Control at where new images arrive",
    body: (
      <>
        Step 3. Pick the source folders, the root they land under, the week
        prefix (<span className={K}>week_</span> by default) and the{" "}
        <strong>cut-off</strong>. The cut-off alone decides which week folder
        a file goes into: week 1 is the backlog, week 2 starts at the
        cut-off, and so on. Press <strong>Apply settings</strong> &mdash;
        nothing is armed until you do &mdash; and <strong>Pull now</strong>{" "}
        to pull by hand. <em>When</em> it pulls on its own is set in the{" "}
        <strong>PULL, BUILD, QUEUE, TRAIN</strong> row pinned at the top
        (Off, When files arrive, or On a schedule, in SGT); that is
        unrelated to the week boundary.
      </>
    ),
  },
  {
    title: "Check the weeks and the retention curve",
    body: (
      <>
        Step 4, <strong>Auto Training</strong>, shows what each week folder
        holds; it follows Pull Control&rsquo;s root, which is the root the
        unattended chain trains from. Every week is included unless you tick
        it out. <strong>Retention</strong> (4b) decides how much of each
        older week a run uses. Its bars &mdash; what a retrain would use
        today &mdash; load by themselves. Only folders starting with the
        week prefix count as weeks; anything else under the root is
        ignored. Nothing is ever deleted: decay only thins the training list.
      </>
    ),
  },
  {
    title: "Start the run",
    body: (
      <>
        Step 6, <strong>Start the training run</strong>. Pick the training
        data: <em>Use the assembled dataset</em> (plus the other ticked
        sources), <em>Use a frozen snapshot</em>, or <em>Use snapshot
        generated during auto pull</em>. Read the <strong>Analysis</strong>{" "}
        block &mdash; the button stays disabled while anything blocks. With
        one source it says <strong>Add to Queue</strong>; composed from more
        than one, it says <strong>Build, Queue and Start</strong> and runs the
        whole chain now: build the golden set, compose the yaml, freeze a
        snapshot, queue, start.
      </>
    ),
  },
  {
    title: "Release the queue",
    body: (
      <>
        Queued is not started: press <strong>Start Queue</strong> on the{" "}
        <strong>Training queue</strong> card. There is one GPU
        (ml.g4dn.xlarge), so one run at a time; the queue also waits for a
        job that is still <em>Stopping</em> to clear before it starts the
        next. <strong>Stop Queue</strong> means start no more;{" "}
        <strong>Stop Training</strong> ends the running job (it cannot be
        resumed) and leaves datasets, snapshots and models alone.
      </>
    ),
  },
  {
    title: "Let it run unattended, once you trust it",
    body: (
      <>
        Tick <strong>Enabled for scheduling</strong> on step 6 (the same
        switch as <em>Run after every scheduled pull</em> in Retention).
        After each qualifying pull it runs the whole chain:{" "}
        <em>pull &rarr; rebuild lists &rarr; data.yaml &rarr; freeze snapshot
        &rarr; queue &rarr; start</em> &mdash; through to a trained model,
        with nobody watching. <strong>Which pulls chain</strong> is a setting:
        On a schedule (the default), When files arrive, Pull now (by hand).
        Two guards sit with it: <strong>Files that count as a batch</strong>{" "}
        and <strong>Wait between runs (minutes)</strong>. Set both before
        ticking <em>When files arrive</em> &mdash; it fires once per upload
        burst, and without them one photo is one GPU run.
      </>
    ),
  },
  {
    title: "Watch it, and read the history and the cost",
    body: (
      <>
        The pinned bar at the top shows the GPU, where the chain has got to
        (PULL / BUILD / QUEUE / TRAIN), and the{" "}
        <strong>SageMaker training cost</strong> for today &mdash; its job
        list shows three rows, then scrolls. The <strong>History</strong>{" "}
        under the Training queue lists every run: status, what it was sent,
        and buttons to export its details as CSV, download its{" "}
        <span className={K}>.pt</span>, or copy it into a bucket folder.
        Times are shown in SGT.
      </>
    ),
  },
  {
    title: "Use frozen snapshots for repeatable runs",
    body: (
      <>
        A snapshot records the <span className={K}>data.yaml</span> and a
        manifest of every file name it selected &mdash; no images are
        copied &mdash; so a run can be repeated after the folders have
        grown. On step 6 choose <strong>Use a frozen snapshot</strong>, then{" "}
        <strong>Pick a snapshot</strong>: each shows when it was frozen (SGT)
        and its file count. <strong>Load</strong> points the form at it;{" "}
        <strong>Run</strong> loads it and queues the run;{" "}
        <strong>Delete</strong> asks first, and removes only the record,
        never the images. Snapshots are made two ways: <strong>Freeze the
        dataset on every run</strong> in Retention (on by default; off, runs
        train on the same images but leave no snapshot), which captures
        every ticked source; and <strong>Freeze this dataset</strong>, the
        manual button, which freezes Pull Control&rsquo;s root only.
      </>
    ),
  },
];

// --- Review & Metrics ----------------------------------------------------

const METRICS_STEPS: StepList = [
  {
    title: "Find the version you want to look at",
    body: (
      <>
        The <strong>Models</strong> card narrows by Model Name, Model and Size,
        then you pick the registry group and the version inside it. The
        version dropdown shows the score and the validation count on every
        line, so you can choose without opening anything.
      </>
    ),
  },
  {
    title: "Read the validation count, every time",
    body: (
      <>
        Below about fifty validation images a single picture moves mAP by
        whole percentage points, so a near-perfect score measured on five
        images is noise, not a score. The count turns red and says so when
        it is too small to trust.
      </>
    ),
  },
  {
    title: "Add up to four versions to the comparison",
    body: (
      <>
        Press <strong>Select this</strong>. Four is the limit because four
        is the most the colour palette separates for someone with colour
        vision deficiency. Remove one with the &times; on its chip in the{" "}
        <em>Comparing</em> row &mdash; the colour follows the version, not
        its position, so removing one does not recolour the rest.
      </>
    ),
  },
  {
    title: "Read the bars against a fixed scale",
    body: (
      <>
        The axis is 0 to 1 in every cluster, not scaled to the tallest bar,
        so a bar height means the same thing everywhere. A version with no
        recorded metrics draws as an <em>empty</em> bar, not a zero &mdash;
        usually an imported checkpoint or a run that died before it
        evaluated.
      </>
    ),
  },
  {
    title: "Take the warnings above the chart seriously",
    body: (
      <>
        Two numbers on one axis imply one question answered twice. The
        notes fire when that is not true: a different class list means a
        different task, and validation sets an order of magnitude apart mean
        one number is an estimate and the other is noise.
      </>
    ),
  },
  {
    title: "Export the best version's checkpoint",
    body: (
      <>
        <strong>Export control</strong> sends the{" "}
        <span className={K}>.pt</span> itself. Pick Model Name, Model and Size,
        then what decides &ldquo;best&rdquo;: highest mAP50, F1, recall, or
        simply the newest version. Ties go to the higher version number.
        Approval status and validation count do <em>not</em> exclude a
        version, which is why the chosen one always states both.
      </>
    ),
  },
  {
    title: "Choose where it goes, and what happens to what is already there",
    body: (
      <>
        This computer or an S3 bucket.{" "}
        <strong>Once the new one lands</strong> decides the earlier exports:
        keep them all, move them to <span className={K}>archive/</span>, or
        delete them. All three touch only <em>this Model Name&rsquo;s own</em>{" "}
        <span className={K}>.pt</span> exports directly in that folder, and
        the card lists exactly what would go before you press.
      </>
    ),
  },
  {
    title: "Switch on the automatic export if you want it unattended",
    body: (
      <>
        <strong>On</strong> watches that one group. When a run registers a
        new version, the criterion is re-applied; if the winner changed, the
        new checkpoint is written by itself. S3 only. The toolbar at the top
        saves this page&rsquo;s export settings by name; loading one exports
        nothing and never switches the watcher on.
      </>
    ),
  },
];

// --- Notifications -------------------------------------------------------

const NOTIFICATION_STEPS: StepList = [
  {
    title: "Add the address, and choose what it receives",
    body: (
      <>
        Type it into <strong>Add a subscriber</strong> and tick the kinds of
        email it should get: <strong>Training started</strong>,{" "}
        <strong>Training ended</strong> and <strong>Errors</strong>.{" "}
        <strong>Evaluation metrics</strong> is marked <em>KIV</em> &mdash;
        kept in view, not sent yet &mdash; and cannot be ticked until the
        evaluation step publishes its metrics.
      </>
    ),
  },
  {
    title: "Click the link in the confirmation email",
    body: (
      <>
        Amazon SNS sends one, and nothing is delivered until somebody clicks
        it. Until then the row reads <em>pending</em> and has no Remove
        button &mdash; a pending subscription has no identifier to cancel,
        and it expires on its own in three days.
      </>
    ),
  },
  {
    title: "Change a subscriber's choices at any time",
    body: (
      <>
        Each row in <strong>Subscribers</strong> carries the same ticks. They
        are stored on the subscription at SNS, so the choice applies to
        every sender, not only this page.
      </>
    ),
  },
  {
    title: "Know what reaches everyone regardless",
    body: (
      <>
        The ticks filter only the kinds above. Emails that have no kind
        &mdash; drift review alerts, weights approved, pull summaries
        &mdash; still go to every confirmed subscriber.
      </>
    ),
  },
  {
    title: "Do not treat 'Confirmed' as proof of delivery",
    body: (
      <>
        It means only that SNS issued an identifier. Subscriptions created
        by Terraform also appear here, can be removed here, and come back the
        next time the stack is applied.
      </>
    ),
  },
];

// --- Housekeeping --------------------------------------------------------

const HOUSEKEEPING_STEPS: StepList = [
  {
    title: "Save the whole page as a settings file",
    body: (
      <>
        The toolbar pinned at the top of Multi-Auto Training holds{" "}
        <strong>New</strong>, <strong>Save as</strong>, <strong>Load</strong>{" "}
        and <strong>Delete</strong>. One file holds every card&rsquo;s
        choices &mdash; folders, Model Name, run shape, sources, retention,
        triggers &mdash; and which cards were open. Results (previews,
        listings, run summaries) are not saved; they are re-read live on
        load. Loading a file can add Model Names it carries but never
        changes or removes the ones already here.
      </>
    ),
  },
  {
    title: "Delete an S3 folder from its field",
    body: (
      <>
        Folder fields carry a red bin icon. It opens a dialog that counts
        the files about to go and names every setting still using that
        folder &mdash; if any do, the button reads{" "}
        <strong>Delete it anyway</strong>. It refuses outright for the bucket
        root, a bucket that is not this app&rsquo;s, and the pipeline&rsquo;s
        own code and weights prefixes. Deletion is permanent.
      </>
    ),
  },
  {
    title: "Delete snapshots you no longer need",
    body: (
      <>
        From <strong>Use a frozen snapshot &rarr; Pick a snapshot</strong> on
        step 6. The record, its frozen yaml and its file list go; the images,
        labels and any trained model are untouched. What is lost is the
        ability to repeat that exact run.
      </>
    ),
  },
  {
    title: "Start over on the run card",
    body: (
      <>
        <strong>Start All Over</strong> on step 6 clears the form on screen
        &mdash; Model Name, version, dataset, and the run shape back to its
        defaults (50 epochs, 640px, batch 16) &mdash; so you can choose again
        from the top. It does not touch the queue, the registry or any
        data.
      </>
    ),
  },
];

const CONTENT: Record<TabKey, { blurb: ReactNode; steps: StepList; closing?: ReactNode }> = {
  training: {
    blurb: (
      <>
        Getting from &ldquo;new images arrived&rdquo; to &ldquo;a trained
        model is registered&rdquo;. The page is numbered 1 to 6: set up the
        Model Name, choose the data sources, build the yaml, then start,
        queue or schedule the run. Each step reads back what the ones above
        chose, so work down it in order.
      </>
    ),
    steps: TRAINING_STEPS,
    closing: (
      <>
        <strong>Two things cost GPU hours without asking.</strong>{" "}
        <em>Enabled for scheduling</em> trains all the way to a model with
        nobody watching, and <em>When files arrive</em> as a chain trigger
        without a batch size and a wait fires a run per upload. And a folder
        ticked on two sources at once trains twice.
      </>
    ),
  },
  metrics: {
    blurb: (
      <>
        Comparing what has been trained, and getting the one you want out.
        Nothing on this page starts a training run or costs GPU time; the
        only thing that writes anything is Export control.
      </>
    ),
    steps: METRICS_STEPS,
    closing: (
      <>
        <strong>A score is only as good as what it was measured on.</strong>{" "}
        Every number on this page comes from the registry, and every one is
        reported beside its validation count for that reason: 0.995 on five
        images is not a better model than 0.400 on two hundred.
      </>
    ),
  },
  notifications: {
    blurb: (
      <>
        Who gets emailed, and about what. Each subscriber chooses their own
        kinds of email; a few kinds always reach everyone.
      </>
    ),
    steps: NOTIFICATION_STEPS,
  },
  housekeeping: {
    blurb: (
      <>
        Saving your setup, and removing what is no longer wanted. Every
        delete here opens a dialog that says what goes and what stays, and
        Cancel is the default.
      </>
    ),
    steps: HOUSEKEEPING_STEPS,
  },
};

export function Track2HelpPage({ popout = false }: { popout?: boolean }) {
  const [params, setParams] = useSearchParams();
  const fromUrl = params.get("tab") as TabKey | null;
  const [tab, setTab] = useState<TabKey>(
    fromUrl && TABS.some((t) => t.key === fromUrl) ? fromUrl : "training",
  );

  // Kept in the URL so the pop-out opens on the tab you were reading, and
  // so a link to one of these walkthroughs lands on the right one.
  useEffect(() => {
    if (params.get("tab") !== tab) {
      setParams({ tab }, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  const { blurb, steps, closing } = CONTENT[tab];

  return (
    <div className={popout ? "mx-auto max-w-2xl px-4 py-5" : ""}>
      <h1 className="mb-1 flex items-center gap-2 text-xl font-semibold text-ink">
        <HelpCircle size={20} className="text-accent" strokeWidth={2} />
        Track2 Help
      </h1>
      <p className="mb-4 max-w-3xl text-sm text-ink-secondary">
        One walkthrough per page, in the order you would actually do it.
        Tick each step off as you go &mdash; the ticks are remembered in
        this browser.
      </p>

      <div className="mb-5 flex flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-md border border-black/15">
          {TABS.map(({ key, label }) => (
            <button
              key={key}
              type="button"
              onClick={() => setTab(key)}
              aria-pressed={tab === key}
              className={`border-l border-black/15 px-3 py-1.5 text-xs font-semibold first:border-l-0 ${
                tab === key ? "bg-accent text-white" : "bg-surface text-ink-secondary"}`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Beside the page it describes, rather than alternating between
            two browser tabs and losing your place in both. */}
        {!popout && (
          <button
            type="button"
            onClick={() => window.open(
              `/track2/help/popout?tab=${tab}`,
              "track2-help",
              "width=620,height=900,menubar=no,toolbar=no",
            )}
            className="flex items-center gap-1.5 rounded-md border border-black/15 px-3 py-1.5 text-xs font-medium text-ink-secondary"
          >
            <ExternalLink size={13} />
            Open in a window
          </button>
        )}
      </div>

      <Card title={TABS.find((t) => t.key === tab)!.label}>
        <p className="mb-4 text-sm text-ink-secondary">{blurb}</p>
        <Steps id={tab} steps={steps} />
        {closing && (
          <p className="mt-4 rounded-sm border border-critical/30 bg-critical-subtle px-3 py-2 text-sm text-critical">
            {closing}
          </p>
        )}
      </Card>

      {!popout && (
        <div className="mt-5">
          <Card title="The dataset layout a run expects">
            <p className="mb-2 text-sm text-ink-secondary">
              Only needed if you are preparing a folder by hand. Assemble a
              dataset, Pull Control and Retention write the yaml for you.
            </p>
            {/* Corrected. This card used to read `train/images` and
                `val/images`, which is the wrong way round and does not
                resolve: Ultralytics finds a label by swapping `/images/`
                for `/labels/` in the image path, so the split has to sit
                UNDER images, not over it. */}
            <pre className="tabular overflow-x-auto rounded-sm border border-hairline bg-page p-3 text-xs text-ink">
{`<dataset_folder>/
  data.yaml
  images/
    train/   *.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp
    val/
  labels/
    train/   *.txt   -- one per image, same basename
    val/`}
            </pre>
            <p className="mt-3 text-sm text-ink-secondary">
              A label row is{" "}
              <span className={K}>class_id x_center y_center width height</span>,
              every coordinate normalised 0.0&ndash;1.0. An image with no
              objects has an empty or absent label file. The class{" "}
              <em>names</em> live only in{" "}
              <span className={K}>data.yaml</span> &mdash; a label row
              carries an index and nothing else, so scanning labels gives
              you counts, never names.
            </p>
            <pre className="tabular mt-3 overflow-x-auto rounded-sm border border-hairline bg-page p-3 text-xs text-ink">
{`path: /opt/ml/input/data/train
train: images/train
val: images/val
names:
  0: helmet
  1: person`}
            </pre>
            <p className="mt-3 text-sm text-ink-secondary">
              A split entry may also name a list of folders, or a{" "}
              <em>file</em> listing one image path per line &mdash; which is
              how retention and several combined sources reach a run without
              copying anything. A leading <span className={K}>./</span> in
              such a file resolves against the <em>list file&rsquo;s own
              directory</em>, not against <span className={K}>path:</span>.
            </p>
            {/* `nc:` was documented here and is not written or read
                anywhere in this codebase. Left out rather than explained,
                since documenting a key nothing honours is how somebody
                comes to trust it. */}
          </Card>
        </div>
      )}
    </div>
  );
}
