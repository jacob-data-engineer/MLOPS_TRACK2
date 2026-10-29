# Continue here

State of this repo as of 2026-09-25, what was changed, and what is still
outstanding. `README.md` describes the layout; this file is the handover.

---

## The data/training split (2026-10-28)

**Built in the worktree `C:\Projects\T2split`, branch `data-training-split`.
Live only on the refactor slot, 8092. Production (8501) and dev (8090) are
untouched, and so are both shared Lambdas.**

### Server: three folders, one rule each

| Folder | Holds | Rule |
|---|---|---|
| `data/` | retention, retained lists, golden set, dataset yaml, dataset versions, snapshots, `assemble.py` (assembling, source map, class filter, `dataset_classes`), `selection.py` (`read_selection`) | imports neither `training` nor `chain` |
| `training/` | `track2_shared.py` (registry, runs, queue, audit, eval, inference, deploy, cost), hyperparams, run_artifacts, metrics_export | reaches data only through `ppe_pipeline_api.data` |
| `chain/` | `retrain_schedule.py`, `queue_after_pull.py`, `ticker.py` (the 25 s tick, out of track2_shared), `queue_runner.py` (the Lambda tick) | uses both front doors; nothing imports it |
| top level | `pull_control.py` (settings store; the pull Lambda is built from this ONE file, so it stays put), `storage.py` (bucket, S3 client, list/exists helpers), notify, auth, main, routers | shared |

- **Front doors** are `data/__init__.py` and `training/__init__.py`, each an
  `__all__` list. Each entry is `_door.through(...)`: it looks the real
  function up at call time, so there are no import cycles and a test
  patching the real function still intercepts the front door.
- **Old paths still work.** `retention.py`, `track2_shared.py`, ... at the
  top level are forwarders that replace themselves in `sys.modules` with
  the moved module (one object, not a copy), so every old import, test
  patch and the queue Lambda's handler path
  (`ppe_pipeline_api.queue_runner.lambda_handler`) keep working. Inside a
  side, using a forwarder fails the boundary test.
- `tests/test_side_boundaries.py` enforces all of it. It found 21
  crossings when first written; it passes at zero.
- `track2_shared` keeps the moved names (`assemble_dataset`,
  `dataset_classes`, ...) as front-door aliases, so routers and tests that
  say `t2.assemble_dataset` still work. A test that patches a moved helper
  to change what ANOTHER moved helper calls uses `tests/_sides.py`
  `patch_everywhere`; the conftest guard wraps the one shared S3 client
  once for all three modules that bind it.

### Screens: the Data Preparation tab

- `/track2/data-preparation` (`routes/Track2DataPrepPage.tsx`), above
  Multi-Auto Training in the sidebar: Assemble, Golden Set, Pull Control,
  the weeks (Retention inside), the pull queue. Owns the
  "use these folders only" switch.
- Training keeps the data.yaml preview beside Add to Queue, drawn by the
  Assemble card's own `previewOnly` mode from the SAVED settings. It never
  writes; its label classes and filter are read-only there.
- **Notice board** `web/src/lib/bus.ts`: typed messages, posted to the
  window AND a BroadcastChannel, so the Data tab in one window updates
  Training in another. Data cards post `data-sources-changed` after each
  save; Training re-reads its three source ticks on it (this also fixed
  the old staleness: the ticks were read only on load). `useGoldenSync`
  rides on it.

### The refactor slot, 8092

- `track2-app-refactor`, role `refactor`, **`APP_BACKGROUND=off`**: no
  startup recovery (it would fail production's in-flight inference/eval
  rows) and no poll loop. It shows pages; the stable container, dev and
  the queue Lambda do the work.
- **deploy.sh never ships the shared Lambdas from a look-only container.**
  It used to update the pull router and queue runner on EVERY deploy, from
  whatever tree it ran in -- dev deploys included.
- Deploy from the worktree, reading this repo's Terraform outputs:
  `INFRA_DIR=/c/Projects/MLOPS_TRACK2/terraform/app PORT=8092 CONTAINER=track2-app-refactor APP_ROLE=refactor APP_BACKGROUND=off bash deploy.sh`
- Port 8092 opened by hand (`sgr-0ee95b1dbcced7c2d`) and declared in
  `terraform/platform/ec2.tf`. Close it once the split is promoted.

### To promote

1. Merge `data-training-split` into `mlops-track2-only`.
2. Deploy dev normally (8090, background on). **This one updates both
   shared Lambdas** with the split code -- check the queue runner's next
   tick in CloudWatch (`/aws/lambda/track2-app-queue-runner`).
3. Promote dev's tag to 8501 as usual.
4. Remove the 8092 container, close the port, drop the ec2.tf block.

---

## "item" instead of "helmet": single_cls inherited from the base checkpoint (2026-10-26)

Ultralytics re-applies `single_cls` from the STARTING checkpoint's saved
args (`Model._reset_ckpt_args` keeps `imgsz`, `data`, `task`,
`single_cls`). v14 of `yolo11s-helmet-detector` was trained single-class,
so every continuation trained single-class with `SingleCls` off -- the
class came out as `item` and `EvaluateYolo` refused it ("candidate
checkpoint classes {0: 'item'} do not match the dataset's {0: 'helmet'}").
Fixed twice: `train.py` now always passes `single_cls` (off included), and
`evaluate_any.py` (uploaded to `s3://track2-s3/custom/code/`; the old copy
is under `custom/code/backup/`) scores a one-class `item` checkpoint
against a one-class dataset under the dataset's name.

---

## Dataset versions: every queued run trains on a frozen dataset (2026-10-26)

**Live on dev (8090), not production. Uncommitted.** Option B of the
queue-isolation design. Code: `src/ppe_pipeline_api/dataset_versions.py`;
tests: `tests/test_dataset_versions.py`.

Why: a golden-set-only run failed because `golden_set/data.yaml` -- its
TRAINING yaml -- had been overwritten by a composed run's EVALUATION yaml
while it waited in the queue. Shared, overwritable yaml slots
(`<folder>/data.yaml`, `_generated/<hash>/data.yaml`, the eval yaml in a
val folder) were read at START time, not queue time.

What happens now, whenever the chain queues/starts a run and whenever
step 6 queues an open-pipeline run: the yaml is resolved to the exact
images and written ONCE under `_datasets/<dv-id>/` (`data.yaml`,
`train/train.txt`, `train/val.txt`, `eval/data.yaml`, `eval/val.txt`,
`manifest.json`), plus a row in the snapshot table
(`record_type = dataset_version`; snapshot listings skip these). The run
is pointed at it: `TrainingDatasetUri` = data prefixes + `_datasets/<id>/train/`,
`DataYamlUri` = the version's yaml, `EvaluationDatasetUri` =
`_datasets/<id>/eval/`, `EvalExtraUris` = the val folders. **No trainer,
evaluator or pipeline change** -- lists are read by Ultralytics, the
trainer's filter/preflight and the evaluator's filter as they already were.
Folders are still downloaded whole; the lists decide what is READ.

The "Dataset versions" card under the training queue lists them, shows
both yamls, and **Queue again** replays the entry's settings on the same
version.

Open:
- The queue-runner Lambda's role has no access to the snapshot table, so a
  run started by the Lambda leaves its version's state at "queued" (logged,
  non-fatal). App-started runs mark "started".
- The app's role has no `UpdateItem` on that table; `mark()` does
  get/merge/put instead. Keep it that way or add the grant.
- Direct "Start Training" (not queued) is not versioned -- it reads the
  yaml immediately.

Same day, code-review fixes F1-F6 (remap kept via `data.source.yaml`,
anchored training clock, typed hyperparameters in `hyperparams.py`,
sibling-root and foreign-prefix layouts via `dataset_yaml.local_layout`,
one retention inventory) and Add to Queue regenerating the yaml are also
on dev. A new trainer image is optional (only for the trainer to re-type
legacy string hyperparameters itself).

---

## Enabling more GPU quota to shorten the training queue (2026-10-20)

**Not started.** Runs queue one behind another because of two limits,
and only one of them is ours.

### 1. AWS: one of each GPU training instance

Live quotas in `ap-southeast-1`, checked 2026-10-20 with
`aws service-quotas list-service-quotas --service-code sagemaker`:

| Instance | GPUs | On-demand quota | Spot quota |
|---|---|---|---|
| ml.g4dn.xlarge, what runs use today | 1 × T4 | 1 | 1 |
| ml.g4dn.2xlarge, 4xlarge, 8xlarge | 1 × T4 each | 1 each | 1 each |
| ml.g4dn.12xlarge | 4 × T4 | 1 | 1 |
| ml.g5, ml.p3 and other GPU types | none | 0 | 0 |

The CPU instance used for evaluation, ml.m5.xlarge, has a quota of 30,
so evaluation is not the bottleneck. All of these can be raised through
a Service Quotas request; approval is AWS's decision and usually takes
from a day to a few days. No request had been made as of 2026-10-20
(the change history was empty).

To request three on-demand ml.g4dn.xlarge (quota code `L-3F53BF0F`;
the spot one is `L-944F78BB`):

```bash
aws service-quotas request-service-quota-increase --region ap-southeast-1 \
  --service-code sagemaker --quota-code L-3F53BF0F --desired-value 3
aws service-quotas list-requested-service-quota-change-history \
  --service-code sagemaker --region ap-southeast-1   # to follow it
```

### 2. Ours: the queue runs one job at a time

`training_already_running_reason` in `track2_shared.py` refuses to
start a run while another is running or Stopping, and the pipeline
always asks for ml.g4dn.xlarge. Both are choices in this repo and can
be changed.

### Ways to shorten the queue

- **Ask AWS for more ml.g4dn.xlarge**, say 3, and let our queue run that
  many at once. The cleanest option. Changing the queue to allow several
  runs, while keeping the dev and stable apps from double-starting, is
  about half a day after AWS approves.
- **Use the quota already there.** Up to six jobs could run in parallel
  today, one on each g4dn size, without asking AWS. The larger sizes
  have the same single T4 GPU plus more CPU, which speeds image loading
  a little but costs more per hour. Needs the instance type to become a
  pipeline parameter and the queue to pick a free size: about a day.
- **Add spot capacity alongside on-demand.** Spot has its own quota of
  one per size and costs up to about 70% less, but AWS can interrupt
  it, so a long run needs checkpoint resuming. About a day on top of
  the change above.

**Recommendation:** request 2–3 ml.g4dn.xlarge now, since asking costs
nothing, and change the queue to start as many runs as there is quota
for. A run takes about 32 minutes on one T4, so two slots roughly halve
the wait when several are queued.

---

## Proposed: an in-app assistant that guides the operator (2026-10-20)

**Not built. Asked about, priced, and parked here.** The idea: a chat
panel on every Track2 page where somebody asks "why didn't last night's
run start?" and gets an answer about *this* setup, naming the card to
fix it in.

### How it would work

1. A server route, e.g. `POST /track2/assistant`, takes the question.
2. It sends the model the question **plus two pieces of grounding**:
   - the Help page text (`web/src/routes/Track2HelpPage.tsx`, rewritten
     2026-10-20 to match the current workflow) as the reference;
   - a **live-state snapshot** gathered server-side: which of the three
     sources are ticked, Pull Control's root, `queue_after_pull`
     triggers/guards, the queue, whether the GPU is busy or Stopping,
     the last chain run and the step it stopped at, recent errors.
3. The answer comes back in terms of the cards ("stopped at data.yaml
   because the Assemble folder has no images -- fix it in Assemble a
   dataset").

**Read-only to begin with.** It explains and points; the operator
presses the button. Letting it act later would go through the same
`ConfirmDialog` the buttons use, never around it.

### Which model, and what was checked

- **Claude Haiku 5.5 on Bedrock**, listed in `ap-southeast-1` (checked
  2026-10-20: `aws bedrock list-foundation-models --by-provider
  anthropic`; inference profile `global.anthropic.claude-haiku-5-5`).
  Fast, and a fraction of a cent per question.
- **Not a model on the review host.** The host has no GPU, and the
  account's single `ml.g4dn.xlarge` quota is the training slot.

### Cost to build

| Piece | Effort |
|---|---|
| Route: gather live state + call Bedrock | ~4 h |
| Chat panel (drawer, on every Track2 page) | ~3 h |
| `bedrock:InvokeModel` on the review-host role | one IAM change -- **needs the user's approval** (IAM edits from this session are blocked by the permission classifier) |

About a day. The grounding is the part that matters: without the live
snapshot it would give generic YOLO advice, not answers about this
platform.

### Also open from 2026-10-20

- **Notification kinds are live on dev.** Page, routes and labels in
  `src/ppe_pipeline_api/notify.py`; each subscriber's choice is their SNS
  subscription's FilterPolicy; untagged mail still reaches everyone. The
  review-host policy `track2-yolo-review-app-access` got
  `sns:GetSubscriptionAttributes`, `sns:SetSubscriptionAttributes` and
  `sns:Publish` as **v34** (user-approved 2026-10-20; `v29` deleted to
  make room -- the policy keeps 5 versions). Verified from inside the
  dev container: subscriptions read back with their kinds.
- **Still open: the queue runner's `sns:Publish`.** It is in
  `terraform/app/pull_control.tf` (`NotifyChainStopped`); the plan for
  `-target=aws_iam_policy.queue_runner` adds exactly that one statement,
  but the apply was blocked by the session's permission classifier.
  Until it is applied, a SCHEDULED chain that stops cannot send its
  error email (logged in the Lambda instead). `deploy.sh` targets only
  the Lambda functions, so it will not apply this on its own.
- **Done live:** `track2-yolo-training-completed-notify` code updated, and
  the `track2-yolo-training-pipeline-completed` rule now watches
  `track2-yolo-open-training-pipeline` too. Before this, Multi-Auto
  Training runs never sent a start or end email at all.
- **Evaluation-metrics emails are KIV** -- offered on the page, disabled,
  nothing publishes them.
- **Freeze layers does not take effect.** The advanced form sends every
  value as a string; Ultralytics ignores a string `freeze` and *rejects*
  string numbers like `lr0` with a TypeError. Fix is in `train.py`
  (coerce types) plus a trainer image build -- about an hour.
- **Training step as pasted Python** was priced (~1.5 days): run the
  pasted code in place of the one `model.train(...)` call at
  `train.py:1025`, loaded from S3 the way `evaluate_any.py` already is.
  Not started.

---

## Two composed runs trained correctly and died in the evaluation step (2026-10-20)

`custom-242a93...` (21:24 SGT, 1 epoch, 35 min) and `custom-d141cb...`
(21:59 SGT, 20 epochs, 66 min), both pressed from card 6. **TrainYolo
SUCCEEDED in both** on the six-name yaml with `classes=0, single_cls`
-- 8,555 images, helmet only; the first time this dataset was trained
whole. **EvaluateYolo failed in two seconds** at `CreateProcessingJob`:

    No S3 objects found under S3 URL
    "s3://track2-s3/golden_set/,s3://track2-s3/Testing_Dataset/object_detection_dataset/"

The trainer splits that list on commas (taught to on the 16th). A
SageMaker ProcessingInput does not -- the service receives one S3 URI
and looks under the literal string. With no evaluation folder the run
fell back to the training URI, and the many-roots builder set
`eval: None`, so there never was one. The chain had computed
`eval_root` for the FREEZE and then handed the run `""`. RegisterYoloModel
never ran; both models sit unregistered in `training-output/` (55 MB
and 58 MB).

**Fixed and deployed (`20261020-021114`, Lambda 02:15Z):**

1. `_many_roots_document` writes a second yaml at the val folders' own
   base when they share one -- `golden_set/data.yaml`, `val: images/val`
   -- exactly the single-root builder's shape for a separate val prefix.
   `write_in_place` already wrote `out["eval"]`; the chain already read
   `out["eval"]["folder"]` into `eval_root`.
2. The queue entry and the direct start carry `eval_dataset_folder=
   eval_root` instead of `""`.
3. `start_custom_training_run` refuses a comma-separated training URI
   with no evaluation folder BEFORE the pipeline starts, naming why.

Proved on the live splits: `eval folder = golden_set`, so the next run
is told `EvaluationDatasetUri = s3://track2-s3/golden_set/`.

**Open:** the audit card shows "train files: 0 / eval files: 0" for a
composed run -- `_dataset_audit_block` counts under the yaml's folder,
which holds only the yaml. Cosmetic; use the download prefixes. The
20-epoch artifact is salvageable through the Import Checkpoint path if
the operator wants it kept.

Tests: `test_composed_run_evaluates_on_one_prefix.py` (12). Suite 2065.

---

## Labels the yaml cannot name: a warning with two answers (2026-10-19, evening)

The gate from the morning refused outright. Asked for instead: a
WARNING naming the folders and indices, then the operator's choice --
**Name the rest** (placeholders into Label classes) or **Filter**: the
training container drops those boxes from its downloaded copy before
Ultralytics reads them, and the label files in S3 stay exactly as they
are. The filter is a saved checkbox; the unattended chain refuses until
one of the two has been taken.

| where | what |
|---|---|
| `dataset_yaml.build_in_place` | REPORTS `undeclared` (entry, indices, text) and never raises for it -- the preview needs the indices to offer "name the rest" |
| `dataset_yaml.write_in_place` | REFUSES when `undeclared` and not `drop_undeclared`; the chain goes through this, so a scheduled run stops with both answers named |
| `dataset_assemble.drop_undeclared_labels` | the checkbox, saved like every setting, read once per chain run, carried on the queue entry and the direct start |
| `start_custom_training_run` | sends `DropUndeclaredClasses=true` ONLY when on -- blank is omitted, the pipeline default `false` applies, so an un-updated pipeline still runs every ordinary job |
| `train.py --drop-undeclared-classes` | rewrites the downloaded label files under a `labels` path component only (a path list at the root is left alone); logs `DROPPED_UNDECLARED: files_changed=... boxes dropped per class {...}; S3 unchanged` |
| `evaluate_any.py` | the same rule from `DROP_UNDECLARED_CLASSES`; it is a processing INPUT at `s3://track2-s3/custom/code/evaluate_any.py`, not baked into the image -- uploaded |
| pipeline | `DropUndeclaredClasses` (default `"false"`) -> trainer hyperparameter + evaluator environment; TrainYolo re-pinned to `sha256:908fb768904f...` |

**The three-step trap was walked deliberately.** The image is built from
`s3://track2-s3/codebuild-source/training-container.zip`, not the tree.
The local `training-container-src.zip` was from 30 Sep while S3's was
from 16 Oct; both matched the tree file-for-file, so repacking from the
tree was safe. Repacked (md5 == ETag `ccc3b46d...`), uploaded 13:13:28Z,
build started 13:13:30Z and read that location; the zipped `train.py`
carries the flag. `pipeline.json` is the pinned file now;
`pipeline_open.json` (14 Oct) is superseded.

Card: the warning renders inside `YamlPreview` with the button and the
checkbox beside it; "Name the rest" merges placeholders `class_N` for
the undeclared indices over whatever is typed, keeping the Model Name's
own names. Both writers save the tick; it is restored on load.

For THIS dataset nothing changes -- the six real names are already in --
but the next dataset needs nobody to type names: the warning says what
it found, and either answer is one click.

Tests: `test_undeclared_classes_warning_and_filter.py` (29). Suite 2052.

---

## The pipeline as specified, mapped onto the pipeline as built (2026-10-19)

Asked for: a flexible Multi-Auto Training pipeline -- upload a base PT
(yolo11 / yolo26 / pretrained), register it as a SEED when the Model Name
has no versions; pick data by assembling + golden, or pull + golden +
retention; build the yaml automatically; snapshot pull-driven runs.
Confirmed with the operator: **all three sources stay combinable** (not
two exclusive modes), and **a first version is a SEED**.

Almost all of it existed. What did not:

| spec | before | now |
|---|---|---|
| golden set on the PULL path | ignored -- `dy.build(root=...)` knows only week folders; golden was built and used by nothing | the pull branch composes like the assembled one: retention list or week folders + golden train, golden val as held-out; `label_scan=None` |
| SEED when no versions | marker only on class MISMATCH; a matching PT into an empty Class was a plain import | `first_version` from the registry summary, decided before the group is touched; same `SEED_MARKER` so `is_seed_description` agrees |
| one picture of the pipeline | six cards, none showing the sum | `/track2/pipeline` -- "Track2 Pipeline (suggestion)": five stages, read-only, each linking to its card |

`first_version` is NOT `restart_group_numbering_if_empty`'s return: that
is False for a group that does not exist yet, the commonest first
import of all.

The page writes nothing on purpose. A second control for one setting is
how two panels end up disagreeing, and that has happened here more than
once.

Tests: `test_pipeline_spec_gaps.py` (18), `test_pipeline_overview_page.py` (13).

---

## 19 Oct, 15:34: the first composed run failed on class indices (2026-10-19)

It fired ON TIME -- 15:34:34 SGT, the clock fix working -- downloaded
both prefixes, rebuilt `golden_set/` cleanly, and died 30 minutes in:

    Label class 3 exceeds dataset class count 1. Possible class labels are 0-0
    RuntimeError: No valid images found in .../golden_set/labels/val.cache

**The yaml declared the Model Name's list, not the labels'.** The
Assemble card wrote `{0: helmet}` -- the detector's one class -- while
the label files use **0-5** on the assembled folder and **0-3** on the
golden set. Ultralytics marks the WHOLE image corrupt when any box is
outside the declared range (`data/utils.py:355`, at cache build, before
`classes=` filters anything). On validation that was all 20 images and
a refusal; on training it does not even error -- which means **every
earlier "successful" run on `Testing_Dataset/object_detection_dataset`
silently dropped every image holding a non-class-0 box.** Nobody had
confirmed what index 0 means in that labelling; the only yaml saying
`helmet` was ours. (The user has since confirmed 0 = helmet.)

**Two fixes, both deployed to dev and the Lambda (`20261019-114529`):**

1. `dataset_yaml.label_indices_used` READS the label files, and
   `build_in_place` refuses -- before any GPU time -- when they use an
   index the yaml would not declare. One sentence per folder, naming
   the indices seen and the fix. The card preview samples 200 files;
   the chain passes `label_scan=None` and reads them all. Unknown is
   not broken: a failed listing refuses nothing.
2. `dataset_assemble.label_classes` -- a new field on the Assemble card,
   "what the LABEL FILES mean, in index order", distinct from the Model
   Name which is what to TRAIN. `assemble_plan` puts it in the yaml
   when it contains the Model Name's class by name; `class_filter_for`
   then does what it was built for -- a one-class detector in a wider
   yaml becomes `classes=[i]` + `single_cls` at load time. Every image
   is kept; only helmet trains. Saved like every other setting.

**The definitive index set, from reading every label file** (not a
sample): the assembled set uses **0-5** in both splits (6,762 + 1,693
files, none unreadable); the golden set uses **0-3** (100 files). Index 0
is confirmed as helmet by the operator; 1-5 are not yet named.

**The full scan cost 751 s serial** -- twelve and a half minutes that
the chain would have paid on every run. `label_indices_used` reads 16
files at a time now (boto3 clients are thread-safe); the same folders
take about a minute. Measured: 100 files 1.4 s, 1,693 files 10.8 s.

**The names were never unknown -- they were overwritten.** The bucket is
versioned, and `Testing_Dataset/object_detection_dataset/data.yaml` at
2026-10-15 09:19 (448 bytes, written by the Assemble card itself)
declares all six:

    0 helmet, 1 gloves, 2 person, 3 working_boots, 4 man_cage, 5 safe_area

At 11:30 the same day the card rewrote it with `0: helmet` alone -- the
Model Name had been switched to the one-class `helmet` -- and every
yaml since (376 bytes) has been that. `dataset_assemble.label_classes`
now holds the six names in the live config AND in `ussing_golden_set`
and `test4` on the host, so a load cannot drop them. The gate passes
against all three real folders; `assemble_plan` declares six;
`class_filter_for` yields `classes=[0], single_cls=True`.

**How much the one-class yaml threw away, counted from the files:** of
6,762 training images, **2** contain only helmet boxes and **6,760**
contain at least one other class. Ultralytics kept the 2. Every helmet
version trained on this dataset before 19 Oct -- v2, v3, v4 with their
identical 0.995 mAP50, v5 and v6 with 0.000 -- was trained on two
images. The dataset actually holds 25,221 helmet boxes across 6,070
images (plus 30,026 person, 15,456 working_boots, 3,820 gloves, 1,088
safe_area, 931 man_cage). With six names declared and the load-time
filter, all 6,070 are used.

The operator trimmed Label classes to four on 19 Oct (reading "I only
have helmet" as a statement about the data); the card saved it; the
live config was restored to six. The two saved files already held six.

The root cause -- the card writing a poorer yaml over a richer one when
the Model Name changes -- is closed by the gate: that write is refused
now unless Label classes covers the labels. Until then every composed run
refuses in seconds with the indices it saw, rather than failing after
the download.

**Open:** the pull (week-folders) path in the chain builds its yaml via
`dy.build(root=...)` and never applies the golden set -- neither its
val side as the held-out set nor its train side. The three ticks say it
should. ~half a day.

Tests: `test_labels_must_fit_the_names.py` (27).

---

## 19 Oct: the composed run can actually be started (2026-10-19)

Four things, found in order by trying to answer "how do I trigger this
run?" with the golden set ticked. Each one blocked the next.

**1. `DataYamlUri` is on the pipeline now.** The one outstanding command
from the 16th went through. `pipeline_parameter_names` cached the
parameter list for the life of the process, so every warm container
would have gone on refusing and quoting the command just run -- it now
re-reads once when the cached answer lacks the name being sent
(`expect=`), and only on the path about to refuse.

**2. The training clock fired 11 hours late, and it was not a timezone
fault.** `next_runs` reads `at` as SGT and returns UTC correctly;
`15:34` produced `07:34Z`, which is exactly the slot that was claimed.
What was wrong: a missed slot stayed "due" for a WHOLE interval -- 24 h
at `every_days: 1` -- so a 15:34 schedule could start at any hour of
the next day. On 2026-10-18 the 15:34 slot started at 10:58 on the 19th.
It is a 30-minute grace now (`CATCH_UP_MINUTES`), a slot past it is let
go, and `overdue_training_slot` makes the tick SAY so rather than
"Nothing due". Which settings file fired it: `test4.json` -- the only
one whose config produces that slot when run through `due_training_run`.

The display WAS a timezone fault: the pinned bar rendered the viewer's
clock and appended a literal " SGT". `formatDayTime` in `format.ts` is
pinned to `Asia/Singapore` and carries its own label; `localDate` on the
models page delegates to it.

**3. A golden-set rebuild leaked train into val.** `golden_set.build`
copied and never cleared. `golden_set/` held a 5/95 split; rebuilding
to 80/20 would have left all 95 old val files in place, putting 75 of
100 images in BOTH sides -- trained on, then scored on. The build now
reconciles: works out every key it will write, deletes anything under
the target's own `images/` and `labels/` not in that list, then copies.
Scoped to those two subfolders; the source is never touched. Returns
`removed` beside `copied`.

**4. Card 6's button did not compose, and its label said it did.** The
radio read "Use the assembled dataset + the other ticked sources ...
composed at run time"; the button under it called `addTrack2Queue` with
one `dataset_folder`. One folder cannot carry a golden set under
another root -- only the chain builds that yaml. And the one button
that ran the chain, "Run it now", sits inside Auto Training, which is
`inert` in exactly the configuration that composes (Pull Control
unticked). There was no reachable manual trigger.

Now: when the radio says composed, card 6's button reads **Build, Queue
and Start** and calls `/track2/retention/retrain/run` with
`{"submit": "queue"}` -- the same hand-over the schedule uses. The
route grew a `RetrainRunRequest`; `{}` still means `start`, so the
Retention card is unchanged. The chain trains with the same shape card
6 shows: step 1 writes `default_training`, `settings()` reads it,
`onTraining` copies it into the form.

**The ticks moved to their own cards** that morning: Golden Set and
Pull control each carry their switch in their `titleAside`; the
Assemble heading keeps only "Use these folders" plus a read-only
summary line. Pull control dims its own body rather than sitting in
`SwitchedOff`, because the tick that turns it back on is in its header
and `inert` would have made that a one-way door.

Tests: `test_card6_runs_the_chain_when_composed.py` (15),
`test_golden_set_rebuild_is_clean.py` (9), clock tests in
`test_training_has_its_own_clock.py` (33), and `test_when_things_run.py`
grew the SGT-rendering checks.

**Open:** with the folders tick off AND the Assemble card empty, the
chain falls back to the pull-only path and the golden set is not
composed in -- the class names come from the Assemble card's Model
Name, and without the card there is no class source. ~half a day. Also
still two controls overlap for "does the pull feed this run": the
step-6 Training-data radio's `pull` option and the Pull control tick;
recommended the radio drops `pull`.

---

## Three sources, three ticks, one data.yaml (2026-10-16)

**The data.yaml is now the union of whatever is ticked**, not one of
three mutually exclusive modes:

| ticked | the yaml is built from |
|---|---|
| these folders | the Assemble card's folders |
| these folders + golden | both |
| these folders + golden + pull | all three |

The ticks live in the Assemble card's heading, together, because
together is how the question is actually asked. The Golden Set card's
"Use a golden set / Not using one" pair is gone; it is a single tick
with the same label as the row's, and the two stay in step through
`web/src/lib/useGoldenSync.ts` -- one field shown twice on one page is
the trap that card's own comment warned about.

**The stored fields are deliberately the old ones**, so a settings file
written before this still reads correctly:

| tick | stored as | why |
|---|---|---|
| Use these folders | `dataset_assemble.use` | new; **absent means true**, or every existing configuration empties its training set on the next run |
| Use Pick Golden Set / Edge Cases | `retention.use_golden` | unchanged, still owned by that card |
| Use Pull Control | `dataset_assemble.only`, **inverted** | unchanged meaning: it is still what greys Pull Control, Auto Training and Retention out, and still what `follows_pull` reads |

**The golden set has a train side now.** The card has always promised
"the train side is trained on in full and never decays", and until this
went in only its val side was ever named in a yaml. With the split
switched off there is one folder and no held-out set, so it is all
training data -- naming `images/val` regardless, which is what the code
did, pointed the evaluation at a prefix with nothing under it.

**Retention arrives as its path list, not as folders.** Ultralytics
reads a `.txt` as a list of image paths, and `retained_lists.write`
already writes one. Naming the week folders instead would train every
week whole and discard the decay curve. `dataset_yaml` knows a `.txt`
from a folder (`_is_path_list`) and downloads the directory it sits in
(`_download_dir_for`), because its lines are relative to the file.

### What makes this possible, and what is still missing

`train.py` takes a **comma-separated list of prefixes** and keeps each
key's full path when there is more than one; with a single prefix it
strips it, exactly as every run so far. That image is built and the
pipeline is pointed at it
(`track2-yolo-train@sha256:a1b60f30ae68...`). It also uses
`parse_known_args`, so a hyperparameter it does not know no longer
kills the job at startup -- which is what makes the pipeline and the
image safe to change in either order.

**STILL OUTSTANDING -- one command.** The pipeline does not yet declare
`DataYamlUri`, and a composed dataset's yaml lives at
`_generated/<hash>/data.yaml`, which is not at the top of any prefix
being downloaded. `model_training/custom_training/pipeline.json` now
holds the live definition **plus** that parameter and the
`data-yaml-s3-uri` hyperparameter wired to it:

```bash
aws sagemaker update-pipeline --region ap-southeast-1   --pipeline-name track2-yolo-open-training-pipeline   --pipeline-definition file://model_training/custom_training/pipeline.json
```

Until that is run, a **composed** run stops at the data.yaml step with
a message naming this command. Single-folder runs -- which is every run
so far -- are untouched either way: both new parameters are blank for
them and a blank parameter is omitted.

The check sits right after the build says `many_roots`, not in
preflight. "More than one tick" is not the same question as "did it end
up with more than one prefix", and the defaults have two ticks on. It
is also before the freeze and the queue on purpose: `try_advance_queue`
**drops** an entry it cannot start rather than putting it back, so the
chain would otherwise report "queued" and the queue would empty itself
with nothing trained.

**A composed dataset cannot be class-filtered by copying.**
`derive_filtered_dataset` copies `images/` and `labels/` from UNDER the
dataset folder, and a composed one has nothing under it but the yaml --
the copy would succeed and produce an empty dataset. Refused by name
rather than silently.

**The freeze records every prefix.** `dataset_snapshots.create` takes
`extra_roots`; the manifest holds full keys and `prefix` is empty,
which is the same form `train.py` writes to disk in once it is
fetching more than one prefix. Walking `_generated/<hash>` alone would
have frozen a manifest of one file and called it the dataset.

`tests/test_three_sources_compose.py` (34) and
`tests/test_run_takes_several_prefixes.py` (10).

---

## A root folder in use by another saved configuration says so (2026-10-16)

Four settings files on this host, three of them training out of the
same `retraining`, and nothing anywhere said so. Not an error -- the
same week folders CAN feed two Classes -- but almost never meant, and
silent: the pull writes into that root, retention prunes it, and the
chain trains from it.

`pull_control.settings_claims()` reads every settings file and returns
`{name, root, model, assembled}`; `list_settings_files()` carries it, so
it is one fetch rather than a second round trip. The warning sits with
the Root folder field -- not as a banner at the top -- names the model
and the file, says what sharing actually costs, and has a **Use it
anyway** override that is per-folder, so dismissing it for one root
does not silence a different clash.

`tests/test_root_already_in_use.py` (15).

---

## Renaming a Model Name, and what a rename cannot move (2026-10-14)

**The limit is AWS's, and it was checked before designing on it.**
SageMaker offers `create`, `delete`, `describe` and `list` for model
package groups. There is no rename and no update:

```bash
aws sagemaker help | grep -o "[a-z-]*model-package-group[a-z-]*" | sort -u
```

A detector's identifier derives its registry group name, so a rename
cannot move the versions already registered. What a rename CAN do, and
now does:

| | |
|---|---|
| Future runs | register into `yolo11{size}-<new>-detector` |
| Existing versions | stay in the old groups, still listed, still usable |
| Weights | nothing moves, nothing is deleted |

The detector carries `previous_slugs`, and `detector_history_groups`
returns current + old. **Registration uses `detector_groups` and must
keep doing so** -- a new run must never land in an old group. LISTING
uses the history version, in three places: the registry summary, the
metrics scan and the base-model picker, so a run can still continue from
a version trained before the rename. `delete_detector` uses it too, or a
renamed detector's old groups would be left behind with versions in
them and nothing pointing at them.

**The cascade is the point.** A rename that changed only the registry
would leave the training form with nothing selected, the unattended
retrain refusing with "no Model Name", and queued entries submitting
against a detector the server cannot find -- failing at the preflight,
hours after anybody was watching. `retarget_detector_references`
rewrites the live configuration: `default_detector`,
`dataset_assemble`, `import_checkpoint`, `retrain_schedule` and pending
queue entries. It rewrites ONE FIELD, not the section.

**History is deliberately not rewritten.** Audit rows and dataset
snapshots record what happened under the name it happened under, and
editing them would make the record disagree with the registry groups
the versions are really in.

`POST /track2/models/detectors/{slug}/rename`. 16 tests.

### The advanced-training defaults were only half-tested

Asked whether the greyed placeholders are what YOLO11 and YOLO26
actually use. They are -- all 37 match Ultralytics 8.4.130 exactly, and
`yolo11.yaml` / `yolo26.yaml` define only architecture (`backbone`,
`head`, `nc`, `scales`, plus `end2end`/`reg_max` for 26) with no
overlap at all. Training hyperparameters come from one global
`DEFAULT_CFG_DICT`, not from the model.

But the test only checked the KEYS existed. An Ultralytics upgrade
changing `mosaic` from 1.0 to 0.8 would have left the card advertising
1.0 while a blank field trained on 0.8, with nothing erroring. Two tests
now pin the values and the generation-independence.

---

## Most cards were never saving whether they were open (2026-10-14)

**Reported from the page**: the Import card was expanded, the
checkpoint read, classes picked -- saved, reloaded, and it came back
collapsed with nothing read.

**The open/closed work had covered the wrong cards.** Seven panels run
their own Show/Hide and those were wired. But nearly every card
collapses through the shared `Card` component -- its own comment says
"thirty-odd cards across nineteen files" -- and `Card` held a plain
`useState(defaultOpen)`. So the majority were never saving it.

Fixed in `Card` itself, keyed on the card's TITLE, so all of them get it
without thirty call sites being edited. A card with no title passes
`stateKey`. Renaming a title forgets that card once and falls back to
`defaultOpen`, which is what it did before.

`useCardOpen` had to learn two things for this:

* **the updater form.** Every call site is `setOpen((v) => !v)`. Typed
  to take only a boolean, it compiled nowhere and cost a build.
* **an empty key.** Hooks cannot be conditional, so `Card` always calls
  it; a card with no title and no key falls back to local state rather
  than every such card sharing one bucket.

### The Import card's contents

Its class ticks were being treated as per-action and dropped. That was
wrong: "these four of the eighty" is what the picker is FOR, and losing
it meant making the choice again. Saved now, with the label and
description the read generates.

**The read itself is re-run, not stored.** On load the card re-reads the
checkpoint and restores the inspection, so nobody presses Read classes.
A `.pt`'s classes are a property of the file, so a re-read cannot be
stale -- which keeps "do not store results" intact while removing the
manual step. `read_key` is saved rather than `bucket_key`, so an
UPLOADED checkpoint restores too: after the upload the file is in the
bucket, even though the original is on somebody's laptop.

**The trap in that**: re-reading produces a new inspection, and an
effect ticks every class whenever the inspection changes -- so restoring
would have wiped the selection in the act of restoring it. The restore
carries its ticks in a ref the reset consumes once, and they are
INTERSECTED with what the file now has, because a checkpoint can be
replaced at the same key and ticking an index it no longer has would
train against a class that is not there.

An existing test pinned the exact reset line that was refactored.
Behaviour was unchanged, so it now asserts behaviour, and two more cover
the restore path.

**Verified on `20261014-071741`** by fetching the served bundle and
grepping it, not by the tag: `card_state`, `picked_classes`, `read_key`,
`import_checkpoint`, `dataset_assemble` and `golden_set` are all in the
JavaScript the browser actually gets.

---

## The cards, joined up (2026-10-14)

Most of what was asked for already existed and was not wired together.
What follows is mostly linking, which was the constraint.

**The chain already ran to a model.** `submit="queue"` in `run_chain`
does not stop at a queued entry -- it queues AND starts the queue, with
a comment saying why: "a schedule that stopped at a queued entry
produced a to-do item, not a model". Nothing changed there.

The one real gap was a hardcoded line refusing every trigger but the
schedule. It was hardcoded after a real incident, recorded in its own
comment: on 2026-10-06 an ARRIVAL pull that moved ONE file started a
full GPU run, and three arrival pulls fired in ten minutes that morning.

So the trigger list is a setting now, with the two guards that make
`arrival` a deliberate choice rather than a trap:

| | Default | |
|---|---|---|
| `triggers` | `["schedule"]` | `arrival`, `manual` selectable |
| `min_files` | `1` | what counts as a batch |
| `cooldown_minutes` | `0` | caps a burst into one run |

**The defaults reproduce the old behaviour exactly.** A settings file
written before any of this has no `triggers` key, and
`allowed_triggers` falls back to `["schedule"]` -- so no existing file
starts chaining on arrival by surprise. Pinned by a test.

**The class and model registry travels in the file.** Detectors lived
only in DynamoDB, so a file named one by slug and carried nothing that
would let another machine build it. They now ride in a `registry`
section BESIDE `config` -- not inside, because `load_settings_file`
writes `config` straight into the live configuration and a registry copy
in there would be written into live config on every load, where nothing
reads it, and carried forever by `write_config`.

On load: **create missing only, never update, never delete.** This
machine's detectors always win. 11 tests, one of which wires
`delete_detector` to fail the test if a load ever reaches it.

**Card open/closed is saved**, through `useCardOpen` -- ONE shared
record, because `write_config` merges at the top level and seven cards
each writing their own key would mean six being dropped by whichever
wrote last. Results are NOT saved: previews and listings are re-fetched
live, because a stale preview is a number somebody acts on.

### Two traps this hit, both now pinned

**`QueueAfterPullRequest` would have dropped all three new fields** --
the identical Pydantic bug as `golden_set`, caught only because the rule
had just been written down.

**`useCardOpen` was typed to take a boolean**, and all seven cards call
`setOpen((v) => !v)`. It compiled nowhere and cost a build. The hook now
takes the updater form, and a test asserts it keeps doing so -- tsc is
the real gate and does not run on this machine, so the shape is pinned
where it can be checked in a second.

A third thing worth recording: the deploy monitor was watching "the
newest build for the project" and read a STALE id, so a failed build was
reported as succeeded. Watch the id the deploy script prints, not the
newest one.

**Verified on `20261014-070117`**: all five config keys in `DEFAULTS`,
four on `ConfigRequest`, the chain's five fields on its request model,
defaults still `['schedule'] / 1 / 0`, and `allowed_triggers({})` still
schedule-only.

CLAUDE.md now carries these as standing decisions under **How the cards
are meant to behave**, so they are not re-litigated.

---

## Cards whose settings were never saved (2026-10-14)

**Reported as "settings on Multi-Auto Training do not save to Test2".**
The settings file was not the problem. `Test2.json` and the live config
were compared key by key: **34 keys, no differences.** The file is a
faithful snapshot, and `save_settings_file` just copies
`read_config()`. Cards that never wrote into that config had nothing to
be copied.

Three separate failures, one symptom:

| Card | What was wrong |
|---|---|
| **Golden Set** | the card saved correctly; **the server dropped it** |
| **Assemble** | 19 pieces of state, none persisted |
| **Import Checkpoint** | 23 pieces of state, none persisted |

**The Golden Set one is the one to remember.** The panel debounces a
`golden_set` patch on every edit and always did. But `ConfigRequest`
never declared the field, and **Pydantic discards what a model does not
declare** -- so the POST returned 200, the card went on showing the
values because they were still in its own state, and the only trace was
`"golden_set": {}` in a saved file. Nothing logged, nothing failed.

**Fixed.** `golden_set`, `dataset_assemble` and `import_checkpoint` are
declared on `ConfigRequest` and present in `DEFAULTS`. Both cards load
on mount and persist through **one debounced writer each**, rather than
a save call bolted onto every setter -- a setter added later cannot
forget to call a writer that watches the values themselves.

Import Checkpoint saves only its STANDING choices (architecture, source,
prefix, detector, class-source tick). A chosen file, an inspection and a
registration result belong to one action; restoring them would show a
finished import that never happened on this machine.

**One flaw caught before it shipped.** The Assemble card renders on BOTH
tabs, and only Multi-Auto Training passes `onOnlyChange` -- it owns the
"use these folders only" switch because it owns the three cards that
switch greys out. The other instance would have written `only: false`
and silently turned it off. It now writes back what it read.

### The rule is in CLAUDE.md, and a test enforces it

`## A new card's settings go in the settings file -- always`: add the key
to `DEFAULTS`, declare it on `ConfigRequest`, load and save it in the
panel.

A rule nothing enforces decays, so
[tests/test_card_settings_are_saved.py](tests/test_card_settings_are_saved.py)
pins both halves:

- it parses every `saveTrack2PullConfig` call in `web/src`, walks braces
  to take only TOP-level keys, and asserts each is declared server-side.
  **This is what found `golden_set`** -- it was not found by reading the
  code.
- it fails any `Track2*Panel` holding 8+ pieces of state that writes none
  of them, with an allowlist where every exemption carries its reason
  (read-only displays, controlled components, cards saved through their
  own endpoints).

Suite: 1222 pass.

**Verified on the running container**, not on the exit code: all three
keys accepted by `ConfigRequest` with their values intact, all three in
`DEFAULTS`, and a full write/read round trip through the live config --
`{}` before, the probe stored, `PERSISTED: True`, then restored to `{}`.

---

## The PPE judge is retired (2026-10-14)

This repo is now only Track2. The judge, its screens, its tests and the
private package it depended on are gone -- removed, not disabled.

**What went.**

| | Removed |
|---|---|
| `src/ppe_pipeline/` | 52 modules -- the whole package |
| `tests/` | 42 judge-only files |
| `src/ppe_pipeline_api/` | `paths.py`, `dataset_registry.py` (nothing imported either) |
| `web/src/` | `VerdictStamp.tsx` + `verdictColors.ts` -- a dead pair that imported only each other |
| `web/src/lib/api.ts` | 6354 -> 3147 lines |
| `pyproject.toml` | the `aiops-pipeline` dependency, its git source pin, `[project.scripts]` |

**The build no longer needs a credential.** `aiops-pipeline` came from a
private `git+https` URL, which is why the Dockerfile installed `git` and
threaded a GitHub token in as a BuildKit secret, configuring and
unsetting a credential helper inside each `RUN` so it could not survive
into a layer. Every package now resolves from an index, so all of that
is gone. `uv.lock` dropped to 72 packages -- sagemaker-core,
scikit-learn and scipy among them; the app only ever used boto3's
SageMaker client, never the SDK.

The buildspec in `terraform/app/main.tf` still reads
`TRACK_REPOS_TOKEN` from the parameter store and writes `/tmp/token`.
That is **harmless** -- an unused BuildKit secret is not an error -- and
was left alone deliberately so this change needed no `terraform apply`
during a deploy. Removing it is a one-resource targeted apply whenever
it is worth doing.

**The tests got better, not just smaller.** 1201 pass, 0 fail, 0 skip,
and the two `--ignore` flags are gone -- both named files were judge
tests that failed at *collection* and aborted the whole run without
them. The three permanent failures and two quiet skips were judge tests
needing binary assets that live in S3.

**`CONTRACT.md` was regenerated from the running app** (1291 -> 271
lines). It had been documenting an API that mostly 404s -- except it
does not 404, which is the point: the SPA catch-all returns
`index.html` with a 200 for any unmatched path, so a dead endpoint
answers *successfully*. Shapes now live in `/openapi.json`, which cannot
drift; the file keeps the part a schema cannot say.

**`web/README.md` was rewritten.** It described six judge screens that
no longer exist and told you to set `PPE_STORAGE_BACKEND=local` before
running the backend. That switch no longer exists anywhere in the repo,
so the advice was not merely stale -- it implied a local sandbox. There
is none: a locally-run backend writes to the real tables and the real
bucket.

### The one thing nothing local could catch

The first deploy failed, and `deploy.sh` still **exited 0**. The build
log is the only place it said so:

```
Building ppe-pipeline @ file:///app
Failed to build `ppe-pipeline @ file:///app`
```

`[build-system]` uses `uv_build`, which looks for the module at
`src/<normalised project name>`. The project was called `ppe-pipeline`,
so the backend went looking for `src/ppe_pipeline` -- the directory that
had just been deleted.

Nothing on this machine could have caught it. The tests import
`ppe_pipeline_api` from source, the app imports fine, `tsc -b` compiled
the whole frontend, and `uv sync` locally was already satisfied. **The
Docker build is the only place this project is BUILT rather than
imported**, which is exactly what `uv sync --locked` does at
`COPY . .`. The lesson is narrower than "deploy to find out": after
moving or deleting anything under `src/`, run `uv build --wheel` -- it
takes seconds and reproduces the failure exactly.

The project is now `track2-mlops`, the description says what it is
instead of describing a Bedrock judge, and
`[tool.uv.build-backend] module-name = "ppe_pipeline_api"` says plainly
where the package lives. The package was NOT renamed: that would churn
every import, the Dockerfile ENTRYPOINT and both Lambda handlers for
nothing.

**Four judge-only dependencies went too** -- `anthropic`, `openai`,
`pandas`, `filelock`: declared, imported by nothing. Four others look
removable and are not, so they are now commented in `pyproject.toml`:

| Kept | Why it is not dead |
|---|---|
| `uvicorn` | the Dockerfile ENTRYPOINT runs it; never imported |
| `python-multipart` | `POST /track2/models/base-models/upload` takes `File`/`Form`; FastAPI raises at request time, not import |
| `torchvision`, `pillow` | ultralytics needs them, and `[tool.uv.sources]` pins them to the CPU wheel index -- the default Linux wheel is the CUDA build |

`config.toml` at the repo root was deleted: judge tuning parameters whose
own header pointed at `src/ppe_pipeline/config.py`, read by nothing.

**Verified on the box**, not on the exit code -- `track2-app-dev` on
`20261014-054041`, `src/` holding only `ppe_pipeline_api`, no
`ppe_pipeline`, **no `git` in the image**, and none of anthropic,
openai, pandas, sagemaker or sklearn installed.

One useful thing fell out of it. On the running container:

```
/openapi.json  ->  200  application/json
/track1/nope   ->  200  text/html
```

That is the catch-all, live. Every retired judge path still answers
*successfully*.

### Outstanding: 15 Track2 client functions were removed with the judge

Sweeping dead code out of `api.ts` also took 15 **Track2** wrappers that
had no caller anywhere in `web/src`:

`addTrack2PullWeek`, `assembleTrack2Dataset`, `buildTrack2Retention`,
`deleteTrack2PullWeek`, `discoverTrack2Units`, `exportTrack2Metrics`,
`getTrack2BaseModelPackages`, `getTrack2BaseModelUploadUrl`,
`getTrack2ClassesFromYaml`, `getTrack2ScratchCheck`,
`getTrack2WeekContents`, `runTrack2Pull`, `saveTrack2Retrain`,
`track2RetentionCurve`, `uploadTrack2BaseModel`.

They were uncommitted work and there is no snapshot of them. **Nothing
imported any of them**, so no screen lost a call and all 283 imports in
`web/src` still resolve -- but the endpoints behind them are all still
live and in `app.openapi()`. Some were clearly superseded (the Assemble
card moved to `writeTrack2DatasetYaml` and left `assembleTrack2Dataset`
behind); others may have been the client half of a card not yet wired.

Each is a thin wrapper over a route whose request model is in the
matching router, so they can be regenerated exactly rather than guessed
at. Say the word and they come back.

Declarations that *were* in the last commit were restored and are marked
with a banner in `api.ts` explaining why they are kept without callers.

Separately, `web/src/components/Track2ModelProvenance.tsx` is imported
by nothing. Left in place -- it is Track2, not judge.

## Two queues, both ticking without the app (2026-10-14)

**The problem.** `lambda_handler` swept with no check at all while the
manual route answered 409 -- the guard was on the door nobody uses
unattended. An arrival burst of 500 files was 500 concurrent sweeps. And
the training queue, though durable, only advanced on the app's 25-second
poll loop, so it stopped the moment the container did.

**Now.** A trigger arriving during a sweep joins a queue instead of
racing it. Both queues are drained by `track2-app-queue-tick`
(EventBridge, `rate(2 minutes)`), which has two targets:

| Target | Function | Drains |
|---|---|---|
| `tick` | `track2-app-pull-control-router` | the pull queue |
| `runner` | `track2-app-queue-runner` | the training queue |

The app still ticks both on its own loop. Both paths call the same
`advance_pull_queue` / `try_advance_queue`, so the tickers cannot race.

**A pull queue entry carries its settings**, snapshotted when queued --
so you can line up several differently-configured sweeps and walk away,
and changing the card afterwards does not change what a queued entry
does.

**The queue runner imports the real module.** `ppe_pipeline_api` plus
PyYAML, assembled by `deploy.sh` into `terraform/app/.build/queue_runner`
and zipped by Terraform. It calls the same `try_advance_queue` the
container calls: a second implementation of "is the GPU free?" is how an
account with one training instance gets two runs submitted against it.
`import yaml` in `track2_shared` is now lazy (`_yaml()`) so the module
loads without it; the advance still needs it, because validating a
dataset reads its `data.yaml`.

### What had to be fixed to make any of it safe

**The submission guards were in-memory.** `_pending_submission` and
`_custom_submission_at` are module globals, so only the process that set
them could see them. CodeBuild accepts a build and returns; SageMaker
does not report Executing for minutes. In that window a second process
saw an idle GPU and submitted onto the one slot. This already applied to
dev vs stable, which share these tables and both run the poll loop.

There is now a durable marker row (`_training_submission`, 900s,
`record_type: training_submission` so it shows up as neither an audit row
nor a queue entry). `training_already_running_reason` reads it first.

**The pull queue's FIFO order was a lie, twice.** The key was
`added_at#entry_id`; the clock has ~15ms granularity, so a burst shared a
timestamp and the tiebreak fell to the random hex. The first fix,
`itertools.count()`, was per-process -- useless for Lambda bursts, where
every invocation starts at zero (five cold starts came back
`second, fifth, fourth, third, first`). It is now an atomic DynamoDB
`ADD`, which needed `dynamodb:UpdateItem` added to both the router and
app policies -- without it the counter failed closed to the broken path
**only in the Lambda**, which is where bursts happen.

### Still open

- Pull rules `track2-app-pull-control-schedule` and `-arrival` remain
  **DISABLED**. The tick rule is enabled; it only drains what is queued.
- The custom-class training path still starts pipeline executions from
  the container only -- the runner's SageMaker grant is read-only on
  purpose.

---

## Training parameters: the full Ultralytics set (2026-10-14)

The card at the foot of step 1 now carries everything Ultralytics takes
-- optimiser, learning rate, schedule, loss weights, augmentation --
under *Show the rest of the training parameters*.

**Overrides only.** A field left blank is absent from what is sent, so a
run that touches none of it trains exactly as it did before. The card
says how many are changed, and lists them when collapsed.

**One blob, not a parameter per knob.** `Hyperparams` is a single
base64 pipeline parameter; `train.py` decodes it and validates every key
against the Ultralytics it actually has installed. An unknown key fails
the job in seconds rather than being dropped in silence while the audit
records the operator's intent. Keys the pipeline owns -- `epochs`,
`data`, `seed`, `patience` -- are refused.

The point of the blob: adding another dial now changes nothing in the
pipeline, the coordinator, or the training image. Without it, every new
parameter was three releases across two stacks.

### The chain, and where it can die quietly

    card -> default_training.hyperparams -> TrainingParams
         -> Hyperparams pipeline parameter -> train.py --hyperparams
         -> model.train(**extra)

**The pipeline pins the training image by DIGEST, not by tag.** Building
`track2-yolo-train:train-latest` therefore changes nothing on its own --
`pipeline_open.json` has to be repointed at the new digest and the
pipeline re-registered, or runs keep using the old image while every
build reports success. That caught me; it will catch the next person.

To ship a trainer change:

```bash
# 1. zip model_training/training-container/* FLAT, back up, upload
aws s3 cp training-container.zip s3://track2-s3/codebuild-source/training-container.zip
# 2. build
aws codebuild start-build --project-name track2-yolo-train-images --region ap-southeast-1
# 3. take the NEW digest and repoint pipeline_open.json, then update_pipeline
aws ecr describe-images --repository-name track2-yolo-train --image-ids imageTag=train-latest
```

### Two things that were wrong in the handover before this

**Patience was never inert.** An earlier note here said the control did
nothing; that was traced on the full-frame replay path
(`track2-yolo-training-pipeline`), which is NOT what Track2 uses. The
open pipeline has always declared `Patience` and
`start_custom_training_run` has always sent it.

**`pipeline_open.json` is the source of truth**, and it matches the live
pipeline exactly. The `aiops-pipeline` package pinned in `pyproject.toml`
is imported only by `ppe_pipeline/track2_status.py`, which nothing
imports -- it has no part in training.

---

## Right now

Two containers on instance `i-0eee645bac49b482a`, sharing everything
below the container -- same tables, same bucket, same pipelines:

| | Port | Container | Role | Image |
|---|---|---|---|---|
| Stable | 8501 | `ppe-pipeline-track2-test` | `stable` | `20260929-133333` |
| Development | 8090 | `track2-app-dev` | `dev` | `20260930-105317` |

**All work goes to development.** Production is promoted by redeploying
the tag dev is already running (`IMAGE_TAG=...`), never rebuilt. See
`CLAUDE.md`, which is the authority on how to deploy; this file is the
handover of *what state things are in*.

The **build badge** under the wordmark shows the role and image tag of
whichever container you are looking at, coloured when it is not stable.
`GET /build` returns the same, unauthenticated -- so "which build am I
looking at" never has to be guessed.

**The host no longer shuts down** (2026-09-24), so the public IP is
stable -- only a stop releases it. It was `18.140.196.76` on 2026-09-25.
Check rather than assume after any restart:

```powershell
aws ec2 describe-instances --instance-ids i-0eee645bac49b482a --region ap-southeast-1 --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

**Builds run in CodeBuild, never locally.** Do not start Docker Desktop;
`deploy.sh` ships the source to CodeBuild, which runs `tsc -b && vite
build` inside the image build, so a TypeScript error fails the build
before the container is swapped. Base images are mirrored into this
account's ECR, which is what keeps a deploy at ~4 minutes instead of up
to 18 -- `terraform/app/base_images.tf` and `mirror.yml` have the
reasoning, and the mirror does **not** auto-refresh.

```bash
.venv/Scripts/python.exe -m pytest tests/ -q   --ignore=tests/test_sync_onedrive_to_s3.py --ignore=tests/test_sync_s3_state.py
```

**1994 pass, 0 fail, 0 skip** (2026-10-14). Every test in the suite
runs. A failure is a real failure; a skip would be a new fact.

Five tests needed binary assets that are not in the repo -- the few-shot
library, a helmet `.pt`, a sample snapshot. They were made to skip, then
removed: all five covered the PPE judge, and nothing in the training path
references `few_shot`, `helmet_detector`, `helmet_cv` or
`snapshot_boxes`. Their files stayed -- the 61 other tests in them run.

`gate.py` carries no allow-list. It requires zero failures and treats a
collection error as a failure too, since a collection error produces no
FAILED lines at all.

The two `--ignore`d files still fail at *collection*: they import a
`scripts/` directory that did not survive the repo split.

The inference endpoint is **deleted** (its normal resting state -- it is
recreated on demand and torn down when idle). Nothing is billing for it.

### Pull control -- where it got to (2026-09-25)

The card is one flow: pick folders -> roots -> cut-off -> weeks -> pull.

**Which week a file goes in is arithmetic from the cut-off**, and
`week_at` is the only implementation of it. Week 1 is the backlog and has
no start: everything up to the cut-off lands there however old it is.
Week 2 begins at the cut-off and runs for `week_length_days`, week 3
after that, and so on without further input. `project_week`,
`plan_routing`, `route_pending` and the forecast all route through
`week_at`, so a preview and a sweep cannot disagree.

**`week_by` now defaults to `file_time`**, not `scan`. Each file is dated
by its own S3 `LastModified`. This is what makes a pull interval longer
than a week work: pulling fortnightly into 7-day weeks under `scan` put
two weeks of images into one folder and left the other empty. `scan` is
still there and still right for a source with no per-file dating, and
`schedule_plan()` flags the combination that misfiles
(`misfiling_risk`).

**A pull time is not a week boundary.** `schedule_start` +
`schedule_every_days` say when the system goes and *looks*; the cut-off
says which folder what it finds goes in. They are deliberately unrelated
-- pulling twice a day into weekly folders, or fortnightly into weekly
ones, are both normal. `test_the_pull_interval_does_not_change_the_week_boundaries`
pins that.

The timer is an **EventBridge Scheduler** schedule
(`track2-app-pull-control`), not the old `pull-control-schedule` rule: a
classic rule cannot express "first pull at this moment, then every N
days", because its `rate()` anchors to whenever the rule was last written
and its cron is UTC with no start date. The old rule still exists and is
**disabled by `set_mode` on every call** so it cannot fire alongside the
new one; it can be deleted once no environment carries an enabled copy.

The schedule is created `DISABLED` and terraform keeps it that way
(`ignore_changes` on expression, state and start date). Arming it is a
decision made in the card.

**A start date in the past rolls forward by whole intervals**, keeping
the time of day -- so "09:00 every fortnight" stays at 09:00 rather than
drifting to whenever the setting was applied. Scheduler rejects a past
start anyway, and the first pull is the next one due, not a missed one.

**A skipped week is shown, not written.** A late file no longer creates
the weeks it jumped over as empty S3 folders; the week *number* is
arithmetic now, so a gap renumbers nothing. The forecast fills the gap in
instead, as a row reading "nothing arrived" -- a jump from week_2 to
week_4 with no week_3 row would read as a numbering bug.

The weeks table polls every 20 seconds while the card is open and marks
what grew, comparing a `stamp` so a poll that found nothing redraws
nothing. Every row states when it closes; week 1's closing moment **is**
the cut-off, and is labelled as such.

**The weeks table is a rolling window of 8.** Once week 9 exists, week 1
stops being drawn -- trimmed, not deleted, with a count of what is out of
view and a Show all beside it. The files stay on S3 and stay counted in
the totals; hiding a row from a table is not a retention decision and
must not read like one.

**The table re-dates live under unapplied settings.** Changing "new week
every N days" passes the pending length to `GET
/track2/pull-control/forecast?week_length_days=N`, which is the same
`week_forecast` the saved view uses -- so the preview cannot drift from
what Apply then does (`test_the_preview_is_what_applying_it_would_give`).
The counts never move, because changing the week length moves no files.

That last point is the trap, and the card warns about it: a folder named
week_3 gathered under seven-day weeks keeps every file it holds and
simply gains a fourteen-day label. Re-dating is relabelling, not
refiling.

**Week folders are flat (2026-09-25).** The layout is:

```
retraining_training/week_1/images/      <- training images
retraining_training/week_1/labels/      <- their labels
retraining_evaluation/week_1/images/    <- evaluation images
retraining_evaluation/week_1/labels/    <- their labels
```

No `train/` or `val/` level, because the ROOT already says which split
it holds and `images/train/` under `retraining_training` says it twice.

**The exception is load-bearing.** When a single root receives BOTH
splits -- a source with no declared split and no evaluation root --
flattening would merge train and val into one folder. That is train/val
leakage: the held-out set stops being held out and every metric measured
against it is quietly wrong while still looking fine. `splits_per_root`
answers "which splits does this root receive" across EVERY source, and
`placement` keeps the split level exactly when the answer is more than
one. The four pickers in the card never produce that shape, but the API
accepts a hand-written source that does.

`placement` takes `splits` as an argument, computed once per sweep. That
is not only for speed: it makes visible that where a file lands depends
on the whole configuration, not only on the source it came from.

**`create_week` creates no folders.** It used to write eight empty
marker keys per week -- both roots x both splits x images and labels --
half of them for splits that week would never receive. Nothing needs
them: the weeks table fills in every week between 1 and the current one
whether or not a folder exists. A folder now means something is in it.

**`week_by` was still `scan` in the live config (corrected 2026-09-25).**
The default changed to `file_time` in code, but `read_config` merges the
stored row OVER the defaults and that row held `scan` explicitly -- so
the deployed system was still filing whole sweeps into one week. A
read-only `plan_routing` caught it: files dated March 2026 were all
routed to `week_2` instead of the `week_1` backlog. The stored value was
rewritten to `file_time`; no files moved. Worth remembering as a shape:
**changing a DEFAULT does not change a saved config**, and this table's
row is long-lived.

Applied by hand on 2026-09-25 because the sandbox blocks
`terraform apply`, with `terraform/app/pull_control.tf` written to match
and the resources imported -- a plan there is now `0 to add, 0 to
destroy`:
`aws_iam_role.pull_control_scheduler`,
`aws_iam_policy.pull_control_scheduler`, its attachment, and
`aws_scheduler_schedule.pull_control`; plus two statements on
`track2-app-pull-control` (`scheduler:GetSchedule`/`UpdateSchedule`, and
`iam:PassRole` scoped to the scheduler role). That policy was at the
5-version cap, so **v2 was deleted** to make room; it is now on v7.

**Live state to be aware of:** the stored config says `mode: schedule`,
but the EventBridge Scheduler schedule is `DISABLED` -- it was created
off and `set_mode` has not been called since. Pressing **Apply settings**
in the card is what reconciles them, and doing so ARMS a real pull every
7 days from the cut-off (2026-09-23 09:00 SGT). Left unarmed
deliberately; arming it moves real files in shared production data.

---

## Two fixes worth knowing about (2026-09-30)

**The retention lists were written where Ultralytics could not read
them.** They went to `<root>/_retained/train.txt` with lines like
`./week_1/images/train/x.jpg`. `get_img_files` expands a leading `./`
against THE LIST FILE'S OWN DIRECTORY, not against `path:` -- so every
line resolved to `<root>/_retained/week_1/...`, which does not exist,
and a real run died with "No valid images found in
.../_retained/week_1/labels/train.cache".

`retained_lists.FOLDER` is now `""`: the lists sit at the root the paths
are relative to. The two are only correct together and either alone
looks reasonable, so `test_the_lists_sit_at_the_root_the_paths_are_relative_to`
encodes the pairing rather than either half.

Anything saved before 2026-09-30 has its lists in the old place. **Press
Save to S3 once** to rewrite them; the stale `_retained/` objects were
deleted.

**The detector now cascades from step 1.** "Register a model and
Classes" gained a *train as this* radio and a size select per row,
writing `default_detector = {slug, architecture}` into the pull config
-- so it rides in the one settings file with everything else.

`retrain_schedule.settings()` OVERWRITES its own `detector_slug` and
`architecture` from it. One source, not a copy that can drift: a
schedule still naming last month's detector while step 1 says otherwise
is a disagreement nobody would notice until a model landed in the wrong
registry. The training form pre-selects from it too, and may still be
overridden for a single run -- but the SCHEDULE always reads step 1, so
a one-off manual run cannot silently change what the scheduled runs do
months later.

The selection is server-side on purpose. The training form's choice
lives in a browser tab, which is why the unattended retrain had nothing
to read and blocked on "no detector chosen"; the preflight now says
*"Pick one in step 1"* instead.

Verified in the running container, not from the exit code:

```
DEFAULTS has default_detector: True
step1 beats a stale schedule: helmet yolo11m
FOLDER = ''
```


## YOLO26, and the trainer bump it needed (2026-09-30)

`GENERATION_SIZES` now holds `{"11": (s,m,l,x), "26": (s,m,l,x)}`, in
both copies (`track2_shared.py` and `model_training/src/config.py`; the
review app's third copy left with the repo split).

**YOLO26 was not a dropdown entry.** The training image pinned
`ultralytics==8.3.*`, whose `cfg/models` tree contains no yolo26 at all,
so a YOLO26 run would have died at model construction. The full chain
was run:

| Step | What |
|---|---|
| 1 | `training-container/Dockerfile` -> `ultralytics==8.4.*` |
| 2 | repacked the zip, uploaded, **local md5 == S3 ETag** confirmed |
| 3 | `track2-yolo-train-images` build, log shows `ultralytics-8.4.166` |
| 4 | `pipeline_open.json` re-pinned to `sha256:8c54aacd...` |
| 5 | `update-pipeline`, then read the LIVE definition back to confirm |

**YOLO11 runs now train on 8.4 too.** That is the real risk in this
change and it has not been exercised yet -- no YOLO11 run has gone
through the new image. First one to go through is worth watching.

**Two things that would have failed quietly, found while doing it:**

  * `open_registry_group()` stripped `yolo{YOLO_GENERATIONS[0]}` only, so
    `yolo26m` kept its prefix and became a SIZE called `yolo26m` --
    yielding a nonsense group name rather than raising. Now
    `split_architecture()`, longest-prefix-first, with a test.
  * A YOLO26 selection would have been offered YOLO11 production
    checkpoints as seeds. They would load and train perfectly well, and
    register a YOLO11 model under a YOLO26 name. Seeds are now filtered
    to the matching generation.

**The built-in PPE detector stays YOLO11 only, deliberately.**
`yolo26s-detector` is not on the exact-match list the approval rule
watches, so a YOLO26 model registered there would look like production
and never deploy. Better no group than a convincing dead one. Detectors
you define get both generations.

**Widening is safe, not a migration.** Stored detectors have
`group_pattern = "yolo11{size}-..."`; they are widened on read to
`"yolo{generation}{size}-..."`, which produces the IDENTICAL string for
generation 11 -- so every version already registered keeps its group and
simply gains the 26 ones.
`test_widening_a_legacy_pattern_keeps_every_yolo11_group_name` holds it.


## The Models card -- one place for "which model" (2026-09-30)

Step 1 was "Register a model and Classes" and step 6 held a second copy
of the detector list, the size dropdown and the registry table. Two
places to say one thing, with nothing forcing them to agree.

Step 1 is now **"Models and Classes"** and owns all of it: define a
detector, pick the generation and size, **import a checkpoint**, **train
from scratch**, **train on an existing model**. Step 6 keeps the
dataset, epochs and submit, and READS BACK what step 1 chose.

It renders on **both tabs**. Import and scratch were previously
`{!multi && ...}`, so the guided 1-6 flow -- the tab the whole retrain
story lives on -- had no way to bring in a checkpoint at all.

`Track2BaseModelPicker.tsx` is new and deliberately written fresh rather
than moved: the old table carried audit expansion and version deletion
that belong with a registry browser, not with choosing a starting point.


## The Models card, finished (2026-09-30)

**Terminology.** Every user-visible "detector" is now **"Class"** -- 17
strings across seven cards, swept to zero. The REGISTRY GROUP NAMES are
untouched (`yolo26s-helmet-detector`): those are real Model Package Group
names and renaming them would orphan every registered version. So the
word still appears inside identifiers, which is the resource's name
rather than label text.

**Training parameters moved to step 1**, as "How every run trains":
epochs, image size, batch, seed, patience and a default note, saved to
`default_training` in the pull config. `retrain_schedule.settings()`
overwrites its own copies from it, the same way it does for
`default_detector`. Step 6 reads them back and has no inputs of its own.
Bad values fall back to the default rather than failing the schedule.

**The card, as it now stands** (iterated over several rounds; this is
the shape that survived):

    Define a new Class      <- ALWAYS live, no tick, never greyed
      Fill these in from a data.yaml  [Read from a data.yaml]
    Your Classes            <- ALWAYS live
      Class [v] Model [v] Size [v]
      Registers into yolo11s-helmet-detector   [Edit] [Delete]
    What are you doing?
      [ ] Train on an existing model          [ ] Train from scratch              > exclusive, tick on the TITLE row
      [ ] Import a pretrained checkpoint /
    How every run trains    <- ALWAYS live

**Defining a Class is NOT one of the exclusive options.** It was, briefly.
It is not a way of GETTING a model -- it is what all three ways need to
exist first -- and in the group it meant ticking a box before you could
type a name, with the card greyed for most of the time it was on screen.

**The Class list sits ABOVE the three ways.** "Which Class" is what every
one of them asks about, and the base-model picker was asking for a
registered version of a Class it had not yet shown.

**The ticks are on each card's title row**, not its first field: whether
a card is in play is a fact about the card, not a setting inside it.

**Three dropdowns, not a row per Class.** There used to be a list of
Class rows, each carrying a radio plus a model and size select that
rendered only on the chosen one -- so the same three facts were spread
across as many rows as there were Classes, and two of them were
invisible until you ticked something. The rows are gone entirely; Edit
and Delete act on whatever the dropdowns name.

**Delete offers the force follow-up.** The server refuses while the
Class's registries hold versions -- correct, since deleting must not
orphan registered models -- so without "Delete it and its versions"
behind a second confirm the button simply errors for every Class that
has ever been trained. The force target is pinned to the Class that was
refused, so changing the dropdown mid-flow cannot redirect it.

Exclusive CHECKBOXES, not radios: `onChange` always selects and never
clears, so ticking one unticks the others and unticking the last is
ignored. A box that could reach "none selected" leaves every section grey
and every button dead, which reads as a broken page.

New / Edit / Save move the selection with them -- otherwise New opens a
form that is already greyed out, which looks like a bug. A successful
import moves it to train-on-existing, because the version that just
appeared would otherwise sit in a greyed-out section.

The auto-queue setting sits OUTSIDE that fieldset deliberately -- it is a
standing setting for the unattended path, not part of configuring one
run, so picking "scratch" must not grey it out.

**Two defects in the merge, found from a screenshot and fixed:**

  * The analysis named the WRONG REGISTRY for YOLO26. `groups[size]` is
    the flat generation-11 map, so "helmet . yolo26s" reported
    "registers into yolo11s-helmet-detector" while the server correctly
    used `yolo26s-helmet-detector`. The card and the run disagreed about
    where a model was about to be filed. Now reads
    `groups_by_generation`.
  * YOLO26 could not be submitted at all. The base picker offered "start
    from scratch -- no existing weights", which sets `basePackageArn`
    empty -- the exact field the form requires -- so Add to Queue stayed
    disabled with nothing saying why, and for YOLO26 it was the only
    option on offer. That radio is gone; scratch is its own section, and
    the empty state explains why there is nothing to seed from.


## Queue a retrain after a pull (2026-09-30)

`queue_after_pull.py`, off by default. On the app's 25s tick it looks for
a sweep that finished since the last one it acted on and, if there is
one, runs `retrain_schedule.run_chain(submit="queue")` -- rebuild the
lists, rebuild the yaml, freeze a snapshot, add ONE queue entry. It does
not start anything; a person presses Start Queue.

**Watched, not called.** A scheduled pull runs in a Lambda whose process
cannot reach this one's in-memory queue, so a hook there would have
worked for manual pulls only -- the ones least in need of it. Every sweep
writes an `event` row to DynamoDB, so watching covers both triggers with
one path.

Three things it refuses, each for a reason:

  * a sweep still running -- the week folders are still filling
  * a stopped or failed sweep -- what landed is not what was meant to
  * a sweep that moved nothing -- the corpus is unchanged, so the run
    would reproduce the model that already exists and bill for it
    (overridable)

The run id is recorded EVEN WHEN THE CHAIN FAILED. Otherwise one
misconfiguration retries every 25 seconds forever. Switching the option
on marks the newest finished pull as already handled, so enabling it
cannot queue work for last week's sweep. 8 tests.


## A brace check cannot see a missing `>` (2026-09-30)

A reorder of the Models card sliced JSX by character offset and swallowed
the `>` that closed `<Track2DetectorsPanel ...>`, turning its three child
sections into attributes. The build failed on six TS1003/TS17002 errors
and nothing swapped, which is the gate working.

The pre-flight brace/paren balance check passed it, and always would
have: **a missing `>` leaves braces perfectly balanced.** The check that
catches it reads multi-line opening tags and asserts each terminates in
`>` or `/>` before the next tag starts --
`scratchpad/jsxcheck.py`, worth keeping next to any bulk JSX edit.

Two lessons, both paid for once each:

  * Slicing JSX by offset is not safe. Anchor on unique full lines.
  * **Minified bundle offsets say nothing about render order.** A check
    that "scratch appears before import in the bundle" reported False for
    a correctly ordered page -- the minifier hoists strings across
    modules. Order is verified from the SOURCE.


## One submit button, not two (2026-09-30)

The scratch card used to carry its own Analysis, its own Add to Queue and
its own `params()`. Step 6 carried another set. Two submit paths on one
page meant two sets of preconditions to keep in step, and they had
already drifted -- step 6 hardcoded `base_model_scratch: false`, so the
two could not even agree on what kind of run they were building.

The scratch card is now settings-only. **Step 6 submits every mode:** it
takes a `scratch` flag, sends `base_model_scratch` accordingly, and drops
the "a starting version is required" guard in that mode, because random
weights do not have one. Its read-back says which kind of run it is.

Removing the card's submit took `busy`/`error`/`message`, `params()`,
`handleAddToQueue`, `handleStartAllOver` and the `onSubmitted` prop with
it.


## Every Multi-Auto Training control, made to work (2026-09-30)

Found by seven read-only auditor agents run over all twelve pages, then
each claim verified against the code before fixing.

| Control | Was | Now |
|---|---|---|
| Train from scratch -> submit | **impossible** | analysis passes |
| "Run after every scheduled pull" | wrote a field nothing reads | writes the real gate |
| Scratch card's 8 inputs | silently discarded | removed; 166->90 lines |
| "Queue even when nothing moved" | unreachable for scheduled pulls | reads `run_state` |
| `pull_first` / `skip_if_unchanged` | settable, dead | removed |
| Retention badge "step 5" | a card that does not exist | `4b`, nested in 4 |
| Import architecture | stripped literal `yolo11` | parses the generation |
| "press train as this" x3 | control removed two deploys earlier | gone |

**The scratch regression was mine, same day.** Moving the submit into
step 6 added the scratch exemption to `baseSelectionIncomplete` and NOT
to `analysis.blocking`, and the button reads both -- so from that deploy
until this one, training from scratch could not be started from anywhere
in the product. The card said "Starting from random weights" while the
Analysis two inches below said "No starting version selected".

**Two checkboxes claimed to run the retrain after a pull.** Retention's
wrote `retrain_schedule.enabled`, which nothing reads; step 6's wrote
`queue_after_pull.enabled`, which is the gate. Both now write the second.
`retrain_schedule.enabled` remains in DEFAULTS as the schedule's own flag
but is no longer bound to a control that claims otherwise.

**The scratch card was a form with no way out.** Model size, Class,
epochs, image size, batch, seed, patience and notes were all local state
with no callback; its props are `active/onActivate/datasetFolder/
evalFolder`. Typing 300 epochs changed nothing and said nothing.


## `event` rows vs `run_state` rows -- read the right one (2026-09-30)

The pull table holds two row kinds for one sweep, and they do NOT cover
the same sweeps:

| | `event` | `run_state` |
|---|---|---|
| Written by | `_record_run` | `write_run_state` |
| Carries `finished_at` / `status` | **no** | **yes** |
| Empty SCHEDULED sweep | **skipped** ("a wall of empty runs") | written |
| TTL | `EVENT_TTL_DAYS` | 2 days |

`queue_after_pull` got this wrong twice in one day:

1. read `event`, checked `finished_at` -> nothing ever fired, and the
   symptom was identical to "no pull has happened yet";
2. read `event` without the check -> could not see an empty scheduled
   sweep, so `even_if_nothing_moved` was unreachable for the one trigger
   it exists for.

It now reads `run_state` via the new `pc.recent_run_states()`, which is
both the honest completion signal and the only source that covers every
sweep. The tests pin WHICH ROW TYPE it reads and why, because the first
set of tests invented a fixture shape and passed against data the system
never produces.


## The unattended retrain, working end to end (2026-09-30)

`0a. NEXT UP` is DONE, by a different route than the one planned. There
is no new Lambda: the chain runs inside the app, on the existing 25s poll
loop, watching for a finished sweep.

    pull fires  ->  rebuild the file lists under the retention policy
                ->  build a new data.yaml
                ->  freeze a snapshot
                ->  publish it to the training card
                ->  queue one run
                ->  START the queue

**It starts the run.** By explicit decision (2026-09-30): "do it
automatically" promised "rebuild, freeze, then train" and stopped one
step short, so an unattended schedule produced a to-do item rather than a
model. Safe unattended because the queue serialises on the account's one
GPU slot -- `try_advance_queue` refuses to pop while anything is
Executing, so a schedule firing mid-run waits behind it. `start` is its
own recorded step, so a chain that queued but failed to start says which.

**What "tracked" means.** `snapshot_id` is a content hash of the yaml
plus the exact file selection, so an unchanged corpus reuses its id
rather than piling up duplicates. Each snapshot archives its own
`data.yaml`, `manifest.json` and `train.txt`/`val.txt` under
`_dataset_snapshots/<root>/<id>/`. The queue entry carries the id, and
`default_dataset` in the pull config carries {folder, snapshot_id, files,
set_by, at} so the training card opens on what the schedule built --
without it the unattended path was invisible and the card still read
"Nothing chosen yet".

First real run: snapshot `f458acb0db7fcb7b`, 1,496 files, 2026-09-30.

**Three bugs it hit on the way, all in the unattended path only:**

  * the yaml entry was rebuilt as `f"{rl.FOLDER}/{rl.TRAIN_NAME}"`.
    `FOLDER` became `""` when the lists moved to the root that morning,
    so this produced **`/train.txt`** -- a leading slash the trainer
    cannot resolve. Fixed to take `written["train_entry"]`, which is the
    only thing that knows where the file actually went. Two other
    callers were fixed that morning; this third was missed.
  * the chain passed neither `class_sources` nor `names` to
    `dataset_yaml.build`, both of which the UI passes -- so it failed
    with "No data.yaml was found to read class names from" on a root the
    UI builds happily. The UI gets `names` from an editable field; at
    18:33 there is nobody to type into it.
  * `build()` now falls back to the root's OWN `data.yaml` when nothing
    is found beside the sources. It is the record of what the label
    indices in those week folders mean.

**Still to set, and inert until then: RETENTION IS OFF.** The decay
policy does not apply, so the yaml names the week folders whole and a run
trains on every image. Epochs is 1.


## Export control exports the checkpoint now, not a spreadsheet (2026-10-08)

On dev as `20261008-095339`. The card was a metrics CSV builder; the
brief was "select the class and the model and the size, then select which
criteria determines which is the best version of the file for export".
The file is the **.pt**. Four questions were asked before building, and
the answers shape everything:

| Question | Answer |
|---|---|
| What ships | the model weights, not a CSV |
| What fires the auto export | a new version registering in the group |
| Guards on the winner | **none** -- top score wins, whatever it was measured on |
| S3 naming | timestamped, never overwritten |

The flow is Class -> Model -> Size (one registry group, named on screen
rather than implied) -> criterion -> This computer | An S3 bucket ->
Export now, with an **Automatic export** block underneath.

**No guards has teeth here, so the cost is stated rather than hidden.**
`yolo11s-helmet-detector` v2, v3 and v4 all score 0.995 mAP50 on FIVE
validation images; v9 scores 0.000 on 149. Ranked on mAP50 with no floor,
five images win every time -- which is what was asked for. So every pick
carries its validation count and approval status, the count goes red
below 50 with words beside it, and a non-Approved winner says so in a
line of its own. Not gating is a decision; hiding what it costs would be
a defect. `test_nothing_is_excluded_for_being_rejected_or_thinly_validated`
pins the behaviour so it cannot drift back by accident.

**A missing metric is not a zero.** An imported checkpoint has not been
measured; ranked as 0.0 it would sit at the bottom of a list it does not
belong in -- and would WIN a group where nothing else was measured
either. A group with nothing measured refuses rather than picking one.
Ties go to the higher version number, because something has to break them
and version order is the one ordering stable between two identical calls.

**The watcher triggers on a new version and writes on a new WINNER.**
Those are different conditions. A run finishing registers a version; if
it does not beat the current best, nothing is written -- otherwise every
run would add a byte-identical timestamped copy of one checkpoint. The
trigger marker still moves, or the decision is retaken every 25 seconds.

`metrics_export.py`, on the existing 25s tick beside `queue_after_pull`
and for the same reason: a model package appears because a PIPELINE
registered it, so there is nothing to hook, only something to watch.
Steady-state cost is one `list_model_packages` per tick; the expensive
half runs only on the tick after a version appears.

**Armed state is NOT in the preset.** Arming is its own call
(`/export/auto/arm`), so loading a preset named after something else
cannot quietly start writing to a shared bucket on a watch -- the same
reasoning as the pull schedule, which is armed in its card. The watcher's
file is `_auto_export.json` in the same folder; the preset name rule
requires a letter or digit first, so it can never be reached through Save
or appear in Load. Arming marks the group's current newest version as
already seen, or arming on a group holding fourteen versions would
immediately export v14 -- a version registered weeks ago is not "a new
version registered".

**Automatic export is S3 only**, said rather than silently refused: there
is no browser attached when it fires, so "this computer" has nowhere to
put the file. Armed per container, because the settings folder is
`/opt/track2-app/settings/<container>` -- arming on dev does not arm
stable, which the card states.

**The checkpoint never passes through the API container.** Both
destinations go through `publish_versioned_weights` (idempotent, extracts
model.pt to a stable key), and then S3 is a server-side `copy_object` and
a download is a presigned link the browser follows itself. A 19MB-110MB
.pt streamed through this single-worker process would stall every request
for its duration. `test_the_checkpoint_never_travels_through_this_container`
asserts the absence of `download_file`/`upload_file`.

**The pick is the server's, not the card's**, although the page holds
every row and could rank them itself. Two implementations of "best" --
one deciding what the card says, one deciding what the button does -- is
how a card comes to promise v4 and ship v9. `/export/preview` is that one
implementation, read-only.

`registry_packages` paginates, unlike `model_registry_summary`, which
passes no `MaxResults` and takes whatever the service returns. Tolerable
where it shows the top few; not where "the best version" would quietly
become "the best of whichever came back". **`model_registry_summary` is
still unpaginated** -- worth a look, since it backs `scan_all_model_metrics`
and therefore the comparison chart.

The metrics CSV endpoint (`POST /track2/metrics/export`) and
`build_metrics_csv` are **kept and still tested, but nothing in the UI
calls them now**. Easy to re-wire if a spreadsheet is wanted beside the
checkpoint.

**The dropdowns were built from what exists, which is the wrong set
(fixed same day, `20261008-100933`).** Reported as "why does it only say
YOLO11 and S or M -- I want to be able to program all the sizes". The
three selects were derived from the metrics rows, so a Class that has
only ever trained at YOLO11-S offered YOLO11 and S and nothing else. The
group a YOLO26-X run will land in next week was unreachable -- which is
exactly when you would want it set up, because armed in advance the first
version to register there goes out the moment it does.

`exportable_groups()` now enumerates every group the platform can
produce, and `scan_all_model_metrics` builds its own scan list from the
same function -- two lists of "every group" is two chances for a group to
be exportable and unchartable, or the reverse. Live, it offers 29:
`helmet` and `segment` at 11 and 26 across s/m/l/x, `custom` the same plus
the legacy group, and `production` at 11 only (`yolo26{size}-detector`
does not exist in AWS and is not on the approval rule's exact-match
list -- a group that looks like production and can never deploy is worse
than no group).

Names come from the SERVER, never composed in the browser: rebuilding
`yolo{gen}{size}-{slug}-detector` client-side is how the Models card once
told somebody a YOLO26 run would register into `yolo11s-helmet-detector`
while the run used the right one. The legacy custom group keeps an empty
generation and size rather than being given ones it never had, so the
selects test MEMBERSHIP rather than truthiness -- falsy-testing them put
that group permanently out of reach while looking like a default.

An empty group now says so (`nothing registered yet`) and explains that
the automatic export can still be armed on it. `arm` on an empty group
records `last_version = 0`, so v1 counts as new.

**Folder housekeeping, and three smaller changes (`20261008-103017`).**

`custom` is **no longer offered as a Class**. It is where a run with no
declared Class lands, so it read as a fifth Class in a dropdown labelled
Class while being a different kind of thing. Removed from the PICKER, not
from the platform: those groups stay on the comparison chart, which
enumerates from the same `exportable_groups()`. A registered model that
appears on no page is how a YOLO26 open run was once trained, Approved
and invisible. Live: offered `helmet, production, segment`; enumerated
`custom, helmet, production, segment`.

The **settings toolbar is sticky**. `<main>` is the real scroll container
(`overflow-y-auto` plus a fixed `md:h-svh`), so `top-0` is the top of the
content area rather than of a page that is itself scrolling away; `-mx-4
px-4` pulls the background to the full content width so what scrolls
underneath is covered; z-20 keeps it below the stale-build banner at
z-40, which has to stay the thing you see first.

**"Once the new one lands"** sits under the S3 folder browser -- three
mutually exclusive modes for what happens to the EARLIER exports:

| Mode | What it does |
|---|---|
| `keep_all` | nothing. The folder is the history. (default) |
| `archive_oldest` | moves them to `<folder>/archive/`. Nothing is lost. |
| `keep_latest` | deletes them. The only irreversible one. |

**The blast radius is deliberately narrow, and this is the decision worth
knowing about.** "Delete everything in the folder keeping only the
latest" is the obvious reading and the wrong thing to implement: an
export folder is a normal S3 prefix that may hold a README, last
quarter's reports, another Class's checkpoints. `export_siblings` requires
ALL of: ends in `.pt`; starts with this group's own export stem; sits
DIRECTLY in the folder (which also keeps `archive/` out of its own reach,
or archiving would feed itself every run); and is not the object just
written. Everything else is left alone. The card lists exactly what would
go before you press, from the SAME `export_siblings` the deletion calls --
a preview scoped differently from the deletion is worse than none.

Widening it to the whole folder is a one-line change if that is what was
meant, but it is not a default to take on somebody's behalf.

**Tidying runs after the new file lands**, never before. The other order
leaves a window in which the folder holds nothing, and a consumer reading
it then gets an empty answer rather than an old one -- the worse failure,
because an old checkpoint is still a checkpoint.
`test_tidying_happens_after_the_new_file_lands` pins the ordering.
Archiving copies then deletes, and only deletes if the copy returned.

`keep_latest` is behind a confirm naming every file, on the manual export
AND on arming the watcher -- armed, it deletes with nobody present, which
is agreed to once rather than inherited from a radio. The watcher carries
its **own** retention mode, taken at arming time; the card says so when
the radio has since been changed, because a mode changed on screen must
not silently change what an already-armed watcher does to a folder. A
watcher armed before this existed reads as `keep_all`.

**"Once the new one lands" is always on screen (`20261008-104316`).** It
was inside the S3 branch, so choosing *This computer* hid it. It only
ACTS on an S3 folder -- a download leaves nothing behind to tidy -- but
hiding it made a standing retention policy look like a property of one
destination. It now says where it applies instead of disappearing.

**The filename preview named a key that never existed.** The card
composed the stem as `group_name.replace(/-detector$/, "")`, giving
`yolo11s-helmet`, while an export lands as `yolo11-s-helmet` -- so the
one line whose job is saying where a file goes named something that never
appears in the bucket. `exportable_groups()` now hands over
`export_stem`, from the same `export_stem()` the exporter uses, and a
test asserts every group's stem is the prefix of its own filename.
Verified across all 29 live.

**Automatic export got real On/Off buttons and a settings panel
(`20261008-110201`).** Reported as "what is this automatic export for and
watch this group?" -- asked twice, which is the answer: the block said
what it would DO and never what it was, or what it was set to.

A single toggle reading *Watch this group* never states the current
state: you have to know that a button offering to start something means
it is stopped. And the same button turning into *Switch it off* is the
one you press by muscle memory to do the opposite of what you meant. It
is now two buttons, On and Off, with the active one filled.

Below them is the panel of what the watcher is actually running on,
written as a numbered sequence rather than described -- "watches for new
versions" is a sentence somebody has to translate into "so when my 3am
run finishes, what lands where", which is the only question being asked.
It names the group, the criterion, the exact `s3://` destination, the
retention mode, the highest version seen, the version last written, and
when it was switched on.

**Off, the panel is greyed and shows what On WOULD do** -- read from the
controls above. Armed, it reads the SERVER's copy, taken when On was
pressed, so a radio changed on screen cannot silently change what is
already running against somebody's folder. When the two disagree the card
says so in a line of its own, outside the greyed panel because it is
about the controls, not the watcher.

**One CodeBuild failure on the way**, TS1002 at
`Track2MetricsExportCard.tsx(347,29)`. A `
` inside a Bash heredoc
collapsed into a real newline inside a double-quoted string. Nothing
deployed -- the gate working. There is now a scan for exactly that
(a newline inside a quoted string, comments and template literals
excluded) worth running before any bulk TSX edit; a brace check cannot
see it, because the quotes stay balanced.

Verified in the running container, not from the exit code: all five new
routes in `app.openapi()["paths"]`, the watcher importable and `armed
False`, the tick hooking it, and `Best version decided by` present in the
bundle; then the groups route and the full 29-group matrix, the Class
list, the retention modes and the four card markers read back off the
running container. 1,805 tests pass; the 3 known environmental failures
remain.

---

## Review & Metrics audited, and Help rebuilt (2026-10-08)

On dev as `20261008-112148`.

### What the audit found

**The chart offered a dead metric by default and hid the live one.**
Read off the running registry: of the 14 measured versions, `f1` has a
value on **14 of 14** and `accuracy` on **0 of 14**. The current
evaluator reports F1; `accuracy` is the legacy evaluator's equivalent,
kept so older versions still read. The comparison card's metric list
contained `accuracy`, ticked, and **did not contain `f1` at all** -- so
the page opened on an empty Accuracy cluster for every version, with its
own comparability note firing to explain the blank bars, while the metric
with data on everything could not be charted anywhere. Export control
would happily rank by F1 with no way to see it.

F1 is now charted and ticked; `accuracy` stays selectable (a version old
enough to have one is still comparable) but starts unticked and is
labelled legacy. The picker's detail panel had the same five-column
mistake and now shows F1, with accuracy appearing only when the version
actually has one.

**`model_registry_summary` was unpaginated.** It passed no `MaxResults`
and took whatever one call returned -- enough today, the helmet group
comes back at 14 of 14, and silently not enough at the service cap. It
feeds `scan_all_model_metrics`, so the failure is versions missing from
the chart and from every count with nothing erroring. Same shape as
`list_s3_images` scoring a 4,000-image dataset on the first 1,000 keys
(A2). Now follows `NextToken`.

**The preview cost 1.7s and ~28 AWS calls per dropdown flick.**
`group_versions` is a describe plus an S3 read per version and the card
calls it on every criterion change. It now takes an opt-in `max_age`,
which the route passes as 20s (cold 1.69s, warm 0.000s) and **the watcher
deliberately does not**: it runs because a new version just appeared, so
a cached list is the one answer it cannot use. A cached read hands back
copies, so a caller mutating a row cannot poison later reads.

**Clear reset the metric checkboxes.** The button sits next to
*Comparing* and says it clears the comparison; it also silently re-ticked
every metric, so having unticked three to look at one, Clear put them all
back. It now clears the selection only.

**There was no way to refresh.** A run finishing while the page was open
changed nothing on it and the only remedy was a browser reload, which
also loses the comparison you had built. There is now a Refresh button
and an "as of" line with the version count.

**A failed fetch looked like a slow one.** `getTrack2ExportGroups` and
`getTrack2Buckets` both swallowed their errors, leaving the Class
dropdown reading "loading..." for ever with the whole card dead behind
it. Both now say what happened.

**A preset naming a Class no longer offered retargeted silently.** Load a
preset saved against `custom`, and the normalising effect moved the
selection to the first Class that does exist -- the right recovery and
the wrong silence, since an export would then go out under a Class nobody
chose. It now says so in red until dismissed.

### Track2 Help

**The old page was four cards of reference and the reference was wrong.**
It documented `train/images` + `val/images`, which is the wrong way round
and does not resolve -- Ultralytics finds a label by swapping `/images/`
for `/labels/`, so the split sits UNDER images. It documented `nc:`,
which nothing in this codebase reads or writes (E3). And it explained the
Evaluation queue, a tab removed from the product. Stale reference is
worse than none: it reads authoritative and costs a day.

It is now **three tabbed walkthroughs** -- Multi-Auto Training (10
steps), Review & Metrics (10), Notifications (6) -- each a numbered
sequence with a tick box, remembered per browser in `localStorage` and
wrapped so blocked site data cannot break the page. Each tab ends with
what to watch out for; the Notifications tab states the two known gaps
(D2, and manual pulls being one IAM grant short) rather than leaving
somebody waiting for email that will not arrive.

**Open in a window** pops out at `/track2/help/popout`, registered
OUTSIDE the Layout route so it renders without the sidebar, wordmark and
build badge -- a second window carrying all of that is a second copy of
the app, not a reference you can keep beside the page you are working in.
Still behind `RequireAuth`. The tab rides in the URL so the pop-out opens
on what you were reading.

The dataset-layout reference survives, corrected, on the main page only:
the right `images/{train,val}` shape, the `0: helmet` names mapping, and
the list-file form with the note that a leading `./` resolves against the
list file's own directory. `nc:` is left out rather than explained.

1,808 tests pass; the 3 known environmental failures remain. Verified in
the running container: pagination follows `NextToken`, the cache is warm
for the preview and absent for the watcher, every new Help string is in
the bundle and all three stale ones are gone.

---

## Training from scratch removed from the page (2026-10-12)

On dev as `20261012-081504`. Removed on request -- a from-scratch run
needs far more data and many more epochs than fine-tuning, so it takes
correspondingly longer on the single GPU slot everything queues behind.

**Removed from the page, not from the platform.** `resolve_scratch_seed_uri`,
`scratch_refusal`, `base_model_scratch` and `GET /track2/models/scratch-check`
are all still there and still work; the sixteen seed objects under
`external_models/scratch/` are untouched. The seeds are real artefacts and
the refusal is a registry fact -- a v4 from random weights continues
nothing while wearing a number that says it continues v3 -- and neither
stops being true because a card went. Putting the card back is a render
line and a piece of state.

What went with it: the `modelSource` state machine ("scratch" | "existing"),
the `default_source` save and restore, the per-Class `scratchRefusal`
lookup, the effect that unstuck scratch on a Class that could not use it,
and `TrainingForm`'s `scratch` prop with its three readers.

**The base model picker lost its tick box.** It was one of two exclusive
modes; it is now the only way a run starts, and a checkbox that can never
be unticked is a control that does nothing -- one sitting on a card's
title row reads as if the card could be switched off.

**Three things this nearly broke, each caught before it shipped:**

  * **The saved starting version stopped being restored.** `default_version`
    was restored by the same effect that restored `default_source`, so
    deleting the block took it with it. That is the C2 bug again -- a
    settings file coming back with the Class right and the version empty
    -- and it was re-added as its own effect.
  * **A CodeBuild failure on `setModelSource` still being called** from
    the import panel's `onImported`. Nothing deployed. Worth noting WHY
    the pre-flight grep missed it: `grep "modelSource"` does not match
    `setModelSource`, because the capital M breaks the substring. Sweep
    case-insensitively, or for the stem.
  * **An empty state that named a route no longer in the product.** The
    base picker's "nothing registered for this architecture yet" panel
    read "Two ways forward: Import a pretrained checkpoint, or Training
    from scratch". That is the one message whose whole job is getting
    somebody unstuck. It now names the one real way.

**The consequence worth knowing:** importing a checkpoint is now the ONLY
way to give a Class its first model at a given generation and size.
Before, scratch was the other. The Models card's caption and the Help
walkthrough both say so.

`test_base_model_exclusivity.py` was rewritten -- its first test pinned
the card's scratch conditional, which no longer exists. It now pins that
the page sends one source and a literal `base_model_scratch: false`, with
no `modelSource` left to disagree; the server-side exclusivity checks are
unchanged, and a new test asserts the scratch path still resolves on the
server.

1,809 tests pass; the 3 known environmental failures remain.

---

## The folder browser built paths that never existed (2026-10-12)

On dev as `20261012-083738`. Reported as: open
`latest_approved_model_weights`, click it again, and the breadcrumb reads
`bucket root / latest_approved_model_weights / latest_approved_model_weights`
with "Nothing here." underneath.

**A race, not a double-click.** `folders` is a PROP, and every one of the
twelve callers does the same two things at two different moments:
`onNavigate` sets a prefix, an effect notices and fetches the listing for
it. Between them the browser re-renders with the NEW prefix and the OLD
rows -- and a row's destination is `prefix + "/" + name` computed at
click time:

    click 1   prefix ""                   -> latest_approved_model_weights
    click 2   prefix latest_approved...   -> latest_approved.../latest_approved...

Any click in that window is wrong, not only a second click on the same
row: a DIFFERENT row gives an equally impossible path. So a double-click
guard would have fixed the report and not the bug.

Fixed once, in `Track2S3FolderBrowser`, so all twelve callers get it.
While a navigation is in flight the rows are **hidden** and the browser
says `Opening <name>...`. Hidden rather than disabled, because a tick box
beside a stale row selects `currentPrefix + "/" + oldName` -- the same
wrong path, quieter and harder to notice.

Three things worth keeping:

  * **Every navigation goes through one `navigate()`.** The breadcrumb
    and the bucket-root button change the prefix too and leave the same
    stale listing behind; guarding the rows alone would have covered one
    third of the ways in. A test asserts `onNavigate(` appears exactly
    once in the file.
  * **The listing is compared by VALUE, not identity.** Two callers hand
    over a brand new array on every render (`folders={listing[key] ?? []}`),
    and an identity check would read that as the new listing arriving
    before the request had left.
  * **A 1.5s backstop.** A caller whose effect does not refetch, or a
    folder holding exactly what its parent held so the listing never
    changes, would otherwise sit behind the guard for ever. A short pause
    is recoverable; an empty browser is not.

`tests/test_folder_browser_navigation.py` pins the shape -- one
navigation path, rows hidden, value comparison, backstop -- plus that
nothing else in the product draws its own prefix breadcrumb, since the
fix is only system-wide while there is one browser.

**A backslash-u-0000 escape in a Bash heredoc became a real NUL byte in the .tsx**, which
git reported as a binary file. Same family as the `
` collapse that
failed a CodeBuild earlier. The separator is plain text now, and the
pre-flight check gained a control-character scan.

1,816 tests pass; the 3 known environmental failures remain.

---

## One Name field instead of Slug and Name (2026-10-12)

On dev as `20261012-091621`. Both cards that create a Class asked for a
Slug and a Name side by side -- the same question twice, where one of
the two silently decided the registry group every version of that Class
would be filed under for the rest of its life, and the other was a
label. Nothing on screen said which was which.

There is one **Name** field now. `web/src/lib/slug.ts` derives the
identifier, and the card shows the registry groups it produces
underneath, so the consequence stays visible without being a second
thing to fill in.

**One derivation, in its own module, on purpose.** Two cards create
Classes -- "Define a new Class", and the import card when a checkpoint's
class names match nothing that exists. Two derivations that drifted
would file the same typed name under two different group names depending
on which card was used. A test asserts both import the helper and
neither rolls its own.

**It has to satisfy the server's rule**, which it did not have to when
it was free text: `upsert_detector` requires `[a-z0-9]([a-z0-9-]{0,30})`
because the result is interpolated into
`yolo{generation}{size}-{slug}-detector`. Anything the browser derives
that fails it is a 422 naming a field nobody can see any more. The
truncation at 31 also strips a trailing hyphen, which lands often enough
to matter. `tests/test_class_name_to_slug.py` re-implements the
TypeScript in Python and asserts the two rules agree, because the
AGREEMENT is what would silently rot.

**Three things the collapse made possible that had to be handled:**

  * **Editing a Class must not move its identifier.** The slug is in the
    registry group name, so renaming the group would strand every
    version already filed under the old one -- and the server upserts,
    so it would create a SECOND Class rather than rename the first. The
    slug is now derived only while defining a new one; editing keeps the
    original and the card says so.
  * **Two names can derive to one identifier.** "Vehicle detection" and
    "vehicle-detection" both give `vehicle-detection`, and
    `upsert_detector` overwrites -- so the second silently replaces the
    first's class list. With the slug on screen that was at least
    visible; the card now names the Class that would be replaced.
  * **A name with nothing usable in it derives to nothing.** "!!!" has
    no identifier in it. Empty stays a real answer and the card refuses,
    rather than inventing something like `class-1` and filing a model
    under a name nobody chose and nobody could find again.

The production groups are unreachable from this, which is worth stating
since `yolo11{s,m,l,x}-detector` is the exact-match list that fires
`dump_approved_weights`: a slug is always inserted between the size and
`-detector`, so producing one of those four names needs an EMPTY slug,
which both sides refuse. Pinned by a test.

Existing Classes are untouched -- `ppe`, `helmet`, `segment` keep their
slugs and labels.

1,826 tests pass; the 3 known environmental failures remain.

---

## The import card says what it is doing (2026-10-12)

On dev as `20261012-111434`. Four relabels and one new control.

| Was | Is |
|---|---|
| This checkpoint detects | Available classes in the .pt file |
| No Class has these class names, so one will be created from them when you register. | Input a name for the model with these classes -- for example Helmet or Forklift. |
| Create detector and register | Register New Model/Class |
| Give the new Class a name -- a version has to belong to one. | Name the model above -- every registered version belongs to a Class, and this is what creates it. |

**New: `Use the classes in this file`**, a tick box under the class
list, and it is an EITHER/OR with "Define a new Class" below.

  * **Ticked** (the default, and what the card did before): the Class is
    the checkpoint's own class names, named in the box below it, and
    "Define a new Class" is dimmed with a line saying why.
  * **Unticked**: "Define a new Class" comes alive, the import card's own
    name box stands down and points at it, and Register refuses to create
    a Class from the file -- doing it anyway would be the card doing the
    thing the operator just switched off.

They are two answers to one question -- what does this model detect --
and the registry records only one, so both live at once is the state
that files a version under a taxonomy nobody chose.

**The dim is reported, not inferred.** Only the import card knows whether
a checkpoint has been inspected, so it raises `onClassSourceChange` and
the PAGE holds the one flag both controls read. Two copies, one per
component, is a pair that can disagree about which half is live.

**And it only dims DURING an import.** `Track2DetectorsPanel` spent a
release greyed out for most of the time it was on screen because it had
been put in an exclusive group it did not belong in -- "defining a Class
is not a way of getting a model, it is what the other ways all need to
exist first". Dimming it whenever the import card merely EXISTS would be
that bug again, so the flag is
`Boolean(inspection) && useFileClasses`. A test pins the `Boolean(inspection) &&`.

The unticked branch states the constraint rather than letting it be
discovered: the hand-written Class's names have to match what the file
detects, in the same order, because a label row carries `2` and never
"vehicle".

1,830 tests pass; the 3 known environmental failures remain.

---

## Register a checkpoint into a Class it does not detect (2026-10-12)

On dev as `20261012-113227`. "Define a new Class" gained a **Register New
Model/Class** button, which appears only while the import card above is
holding an inspected checkpoint with *Use the classes in this file*
unticked. It saves the Class and files the checkpoint into it as v1, in
one action.

**This required relaxing a real guard, so the guard became an opt-in
rather than disappearing.** `register_uploaded_base_model` re-read the
checkpoint and refused outright when its class names differed from the
Class's -- correct, and the reason a ship checkpoint cannot be filed in
the helmet registry. But it also blocked the import card's whole purpose:
a COCO `yolo11s.pt` detects eighty things, so under the strict rule it
could only ever CREATE an eighty-class Class and could never be the first
version of a three-class one.

`allow_class_mismatch=False` is still the default and still refuses. Only
the new button passes it.

**What does not follow is that the record may be silent.** Training
rebuilds the head for whatever the dataset declares, so the checkpoint's
own names genuinely do not survive the first run -- but until that run
the weights detect one thing and the registry says another, and Export
Control's `newest` criterion will hand that .pt to somebody. So a
mismatched registration is marked, three ways:

  * the description leads with `SEED -- NOT A DETECTOR FOR THIS CLASS
    YET.` and names both class lists. The description is the field the
    metrics page, the base-model picker and the registry console all
    already show.
  * `CustomerMetadataProperties` gains `checkpoint_classes` beside
    `classes` -- the latter was the DETECTOR's list, which on a seed is
    exactly the half that is not true of the weights -- plus a
    `class_mismatch` flag.
  * it registers **PendingManualApproval, not Approved**. It is not an
    approved detector for this Class; it is waiting for the run that
    makes it one. Pending keeps it out of everything filtering on
    Approved (`export_weights_to_folder` refuses outright) while leaving
    it usable as a starting point, which the base-model picker allows
    precisely because a new taxonomy's first run has nothing approved to
    start from.

The card says the same thing before the press: what the weights currently
detect, that the head is rebuilt on the first training run, and that it
must not be deployed as it stands.

**The Class is saved before the registration, not after.** There is
nothing to register into until it exists, and a failure between the two
leaves a saved Class and no version -- visible and recoverable, unlike a
version filed under a Class that was never written. `handleSave` now
returns what it saved and takes `keepForm`, so a failed registration
leaves the form intact.

**The button appears only when a checkpoint is waiting.** With the box
ticked the import card registers it itself; a second button doing the
same thing is two ways to perform one action, which is how they drift.

1,837 tests pass; the 3 known environmental failures remain.

---

## "Registers into" named eight groups for a one-group press (2026-10-12)

On dev as `20261012-122029`. A `yolo11s` checkpoint waiting in "Define a
new Class" sat under a list of all eight groups the Class would own --
YOLO11 and YOLO26 at s/m/l/x -- which reads as if the press filed it into
all eight. It files it into exactly one: the group its own architecture
belongs to.

The line now leads with that one group, starred in the list below so the
others still read as what the Class owns for later runs at a different
model or size. The architecture is PARSED (`/^yolo(\d+)([smlx])$/`), not
produced by stripping a literal `yolo11` -- that is a no-op on yolo26 and
has yielded a size of `yolo26m` here before.

**Also caught from the same screenshot: the name derived to `ppe`.** That
is the built-in detector's slug. `upsert_detector` refuses it outright
and `register_uploaded_base_model` refuses a builtin detector, so the
press was a 422 -- about an identifier that is no longer on screen to
explain why "PPE" collided with anything. The clash notice now
distinguishes the two cases: an ordinary collision warns that saving
replaces that Class's class list, the built-in says it cannot be
redefined at all, and both buttons are disabled for it rather than
running into the refusal.

1,840 tests pass; the 3 known environmental failures remain.

---

## Training from an existing version, and finding what you just imported (2026-10-12)

On dev as `20261012-124210`. Three changes.

**The built-in refusal is reworded.** `'ppe' already exists -- unable to
comply. It is the built-in detector and owns the production registries.`
Server and card say the same thing, so hitting it through the API reads
the same as being stopped before it.

**"Train on an existing model" has a tick, and it dims the IMPORT card.**
The mirror of the one already there: importing dims "Define a new Class",
and this dims the import card. The tick does not switch its own card off
-- that card is the only way a run starts, so a box that could disable it
would break the page -- it says "I am training, not importing" and puts
the other out of the way. Opt-in: nothing is dimmed until something is
chosen, and the two dimmings cannot both fire
(`defineDimmed={fileClassesInUse && !trainingFromExisting}`), or both
halves of the page go grey at once.

**A registered version now lands where it can be trained.** This was a
real hole: the base-model picker filters on step 1's Class, model and
size, and an import left step 1 pointing wherever it already was -- so
the version that had just been created appeared in a list nobody was
looking at. To the operator that is indistinguishable from the import
having failed.

The import card now reports WHAT it registered (slug, architecture, ARN).
The page keeps it, hands it to `Track2DetectorsPanel`, which moves all
three dropdowns onto it and preselects the version as the one the run
starts from -- and ticks "Train on an existing model", because a
checkpoint is imported precisely so it can be trained on. `default_version`
is saved server-side like any other choice, so the settings file carries
it. From there it is step 6 and Add to Queue.

Two details that would have broken it:

  * The landing marks itself `restored`. `onChoose(choice, restored)`
    clears the version when somebody PICKS a different Class -- right,
    because a version belongs to one group -- and this is not that: the
    version was created for exactly this Class a moment ago, so clearing
    it would undo the whole point. That is the C2 bug's own mechanism,
    reused correctly.
  * A DELETE reports no payload. There is no new version to select after
    one is removed, and landing on a deleted ARN would be worse than
    landing on nothing.

1,845 tests pass; the 3 known environmental failures remain.

---

## A seed says "Seed", and the dropdowns say Model (2026-10-12)

On dev as `20261012-130917`.

**A seed version read `PendingManualApproval`.** It IS Pending, and
pending BECAUSE it is a seed -- one fact, and "Seed" is the half that
explains the other. It now reads `Seed` in all three places a version's
status appears: the base-model picker, the Review & Metrics picker and
Export control's chosen-version panel. The raw status stays on hover and
in the tone of the badge, so nothing is hidden.

**Keyed on the description the server wrote, never on the version
number.** "Always have this as Seed for version 1" is the right answer
for this version and the wrong rule: a v1 produced by a training run is a
real model, and calling it a seed would be a lie in the one column whose
job is saying what a version is. `is_seed_description` tests for the
`SEED --` marker, and a test asserts nothing anywhere keys the label on
`version === "1"`.

`SEED_MARKER` is one string shared by both sides -- `track2_shared.py`
writes it, `web/src/lib/slug.ts` reads it -- because a longer sentence
drifting on one side would silently stop matching on the other and the
label would quietly revert. A test pins the two equal.

Export control's pick now carries `is_seed` rather than leaving the fact
in a description the card does not show: `criterion: "newest"` will pick
a seed, since nothing filters on status by choice, and exporting one
hands over a checkpoint that does not do what the Class says.

**The dropdown vocabulary.** The row that asks "what is it called, which
network, which size" appears on three cards and read `Class / Model /
Size` on all of them -- where "Class" also means the YOLO class names in
a data.yaml, and "Model" meant the architecture. It is now
`Model Name / Model type / Size`, in all three, because two vocabularies
for one question is the confusion the rename was for. The deeper
terminology (card titles, prose, the `Class` concept itself) is
unchanged -- that is a larger sweep and was not asked for.

**Deleting asks about a Model by name:** `Would you like to delete the
Model PPE2?`

1,851 tests pass; the 3 known environmental failures remain. Verified in
the registry: `yolo11s-ppe2-detector v1` is the one seed on the estate
and reads as one.

---

## A stock weights library, and two relabels (2026-10-13)

On dev as `20261013-023137`.

**`Approved` now reads `Ready`.** A display decision only: `Approved` is
the value SageMaker stores and the one the EventBridge rule that fires
`dump_approved_weights` matches on, and `export_weights_to_folder`
refuses anything else -- so the stored value is untouched. What it MEANS
here is "ready to deploy or train from", so that is what it says. Decided
in one `statusLabel` in `web/src/lib/slug.ts`, beside the seed rule,
rather than in each of the three cards that show a status; the Review &
Metrics picker's own copy was deleted.

**The week prefix shows `week_` as a placeholder, not a value.** Every
reader on the server is `config.get("week_prefix") or "week_"`, so an
empty field already behaves as the default -- and a box pre-filled with
it reads as something somebody chose, which is the thing you hesitate to
clear. A prefix that really was chosen still loads as a value; only the
default blanks.

### The third import source

`s3://track2-s3/track2-yolo-weights/` now holds the official COCO
releases, and the import card has a third radio -- **Use a stock YOLO
release** -- beside upload and browse. Picking one sets the size dropdown
from the file, because `yolo11m.pt` IS a yolo11m and that was the one way
this source could be got wrong.

Six files, s/m/l for both generations (asked and answered; x was
offered and declined):

    yolo11s 18.4MB   yolo11m 38.8MB   yolo11l 49.0MB
    yolo26s 19.5MB   yolo26m 42.2MB   yolo26l 50.7MB

**A folder in the pipeline bucket, not a bucket of its own.** The
instance role already reads and writes `track2-s3`, the folder browser
already walks it, and a new bucket would need an IAM grant plus a change
to the platform stack -- which has drift, so a targeted AWS change rather
than an apply. Nothing about these files needs their own bucket.

**Mirrored once, not fetched on demand**, for the reason the base images
are mirrored: a run that reaches the public internet depends on GitHub
being up and on whatever rate limit applies that day, and that was the
only thing that varied between a four-minute build and an eighteen-minute
one here. `scripts/mirror_yolo_weights.py` does it, `--apply` required,
and already-present files are skipped so a re-run is free.

**The release tag is `v8.4.0`, and that was worth checking.** `v8.3.0`
carries yolo11 and NOT yolo26 -- every yolo26 asset 404s against it --
and `latest` is not a resolvable tag on `ultralytics/assets` at all.
Verified with a HEAD before running anything; otherwise half the library
would have mirrored and the other half "skipped" with a 404 nobody reads
as a wrong tag. A test pins the tag and the file list.

**Detection weights only.** `-seg`, `-pose`, `-cls` and `-obb` are
different tasks, and one offered beside the others would train and
register as if it were a detector.

`list_yolo_weights_library` lists what is ACTUALLY in the folder rather
than what ought to be: a card promising `yolo26l.pt` when its upload
failed sends somebody looking for a file that does not exist.

1,857 tests pass; the 3 known environmental failures remain.

---

## The checkpoint's classes are a table (2026-10-13)

On dev as `20261013-025607`. `Available classes in the .pt file` rendered
them as one comma-separated run. Fine for a four-class detector,
unreadable for a COCO checkpoint: eighty entries wrapping over six lines,
in exactly the place the index matters most. A label row carries the
NUMBER and never the name, so "index 2 means car" is the fact being
checked here.

Now a two-column table, `No` and `Class`, sorted NUMERICALLY -- as text
10 sorts before 2 and the list goes silently wrong from the eleventh
class on. It scrolls past about ten rows with a sticky header rather than
pushing the rest of the card off screen; the usual case of one to six
classes fits without scrolling. The count is stated underneath.

Left alone: the short class lists elsewhere (the chosen-version panel in
Export control, the Review & Metrics detail panel). Those show a Class's
own taxonomy, which is one to six names, and a comma list reads better
than a table for that.

1,858 tests pass; the 3 known environmental failures remain.

---

## One button while a checkpoint is waiting (2026-10-13)

On dev as `20261013-031051`. "Save classes" and "Register New
Model/Class" sat side by side whenever a checkpoint was waiting, and
Register already saves the Class before filing the checkpoint into it --
so the second button did a worse version of what the first does, and the
obvious mistake was pressing it and then wondering where the version
went.

Now one button: **Register New Model/Class** with a checkpoint waiting
("Saving and registering..." while it runs), **Save classes** without
one, which is still the only way to define a Class on its own.

Also shortened the same eighty-name wall the import card lost a deploy
earlier. The waiting box now reads `80 classes (person, bicycle, car,
motorcycle, ...)` rather than listing them all: it is a warning box, not
the place to read a taxonomy, and the full list is in the table above it.
Eighty names wrapping over six lines buried the sentence underneath them,
which is the part that matters.

1,860 tests pass; the 3 known environmental failures remain.

---

## The import card stays editable after a file is read (2026-10-13)

On dev as `20261013-032422`. Everything above the reading froze the
moment a checkpoint was inspected -- the source radios, the model size,
the file input, the release dropdown -- so changing any of them meant
pressing **Start over** and doing the whole thing again. All four are
live now and Start over is gone.

**A different FILE clears the reading**: the source radios, the file
input, the release dropdown and the bucket browser all set
`setInspection(null)`. A panel of eighty class names belonging to a file
you are no longer pointing at is the worst of the three states this could
be in.

**The model size does NOT clear it.** It is a declaration about the file,
not a choice of file; clearing would throw away a correct reading because
the question it is compared against moved.

**So the mismatch is recomputed in the browser.**
`architecture_matches` was baked server-side against the size as it stood
when the file was read -- with the dropdown now editable, that stale
boolean would keep saying "matches" about a comparison nobody is making.
`detected_architecture` is a fact about the file and still comes from the
server; the comparison is live.

**And the Register button now refuses a mismatch.** This is the part
worth knowing: freezing the dropdown was the ONLY thing preventing a
wrong-size registration. Read `yolo11s.pt`, switch the size to
`yolo11m`, press Register, and the checkpoint lands in the yolo11m group
-- a mislabelling that follows every model fine-tuned from it, with
nothing erroring. Unfreezing the control without this would have turned
that into a one-click mistake, so the guard replaces what the freeze was
doing. A file that does not RECORD its architecture is still not a
mismatch, so a fork or an older Ultralytics still imports.

The notice says what to do rather than describing the problem: *"This
file is a yolo11s, but Model size says yolo11m ... set the size to
yolo11s and it will register."*

1,865 tests pass; the 3 known environmental failures remain.

---

## "Define a new Class" starts shut (2026-10-13)

On dev as `20261013-035209`. Reported twice in a row: first that the card
should be out of the way at the start, then that the tick added to do
that should not be there either. Both are right, and the card already had
the answer -- `Card` has a **Show / Hide** of its own.

So: `defaultOpen={false}` and no extra control. Show/Hide already says
whether a card is in play, and a tick beside the title saying the same
thing was one more thing to read. `defineActive` and
`onDefineActiveChange` are gone from both the panel and the page.

A shut card is a title and nothing else, so it uses `summary`, which is
what that prop is for. Normally "Open this to define a Class"; when a
checkpoint is waiting for a Class that has not been named yet, a red "A
checkpoint is waiting for a Class. Press *Show* to name one and register
it." -- that is the one moment a closed card would hide something that
matters.

The body is still dimmed while the import card above is supplying the
classes. That is a different thing from being shut: shut means "not
looking at this", dimmed means "this question is already answered
upstairs".

**A duplicated JSX prop, caught before it shipped.** The card briefly
carried `defaultOpen={false}` AND a bare `defaultOpen` -- the last one
wins, so it would have rendered open and the change would have looked
like it had not worked. The pre-flight now scans each `<Card>` for
repeated props, which is cheap and catches a class of edit that reads
fine in a diff.

1,868 tests pass; the 3 known environmental failures remain.

---

## Superseded: the tick that did this (2026-10-13)

On dev as `20261013-033602`. It gained a tick on its own title row --
**Define a Class myself** -- and is greyed until that is ticked.

**This reverses a documented decision, deliberately and on request.**
The card was pulled OUT of an exclusive group earlier precisely because
defining a Class is not a way of getting a model -- it is what the other
ways all need to exist first -- and being in the group meant ticking a
box before you could type a name, with the card greyed for most of the
time it was on screen. Reported now as the opposite problem: a card
sitting live beside the import card reads as a second thing to FILL IN
rather than an alternative to it.

**The cost is one extra click, and only when defining a Class from
nothing.** The import flow does not pay it: unticking *Use the classes in
this file* turns the card on by itself, because a checkpoint read with
its own classes declined IS the request to define them here, and making
that a second click in the middle of one decision is the thing the
earlier change was avoiding. Ticking *Use the classes in this file* turns
it back off.

The tick sits OUTSIDE the dimmed wrapper, or turning the card on would
need a control the card had already disabled.

**If this proves worse than before**, the lever is one line:
`useState(false)` -> `useState(true)` for `defineActive` in
`Track2ModelsPage`, which restores the old behaviour while keeping the
tick as a way to put the card away.

1,868 tests pass; the 3 known environmental failures remain.

---

## The Define card stands down as soon as a .pt is chosen (2026-10-13)

On dev as `20261013-040249`.

**It used to wait for the READ.** `fileClassesInUse` was
`Boolean(inspection) && useFileClasses`, so "Define a new Class" stayed
live from the moment a file was picked until the inspection came back --
a window with a spinner in one card and a fully usable Class form in the
other, inviting somebody to start typing a taxonomy the checkpoint was
about to supply. It is now `Boolean(inspection || file || bucketKey)`:
picking the .pt is the decision, the read is just the wait.

**And the button goes while it is dimmed.** Dimming already stops the
press, but a greyed-out *Save classes* still reads as the thing to reach
for, and the answer in that state is upstairs rather than here. The row
renders only when the card is live.

Both halves of the exclusivity still hold: untick *Use the classes in
this file* and the card comes back, button and all.

1,870 tests pass; the 3 known environmental failures remain.

---

## "Define a new Class" has exactly one live state (2026-10-13)

On dev as `20261013-041042`. The rule is now
`defineDimmed={!pendingImport}` -- dimmed in every state except a
checkpoint read with its own class names **declined**. That is the one
moment defining a Class by hand is the next thing to do; every other
state has its answer somewhere else. No checkpoint means there is
nothing for the Class to hold, and a checkpoint whose classes are
accepted has already named them.

**One exception, and it is load-bearing: editing.** `Edit` on a Class row
loads it into this same form, so a card dimmed in every state but one
would make that button appear to do nothing at all. The panel computes
`bodyDimmed = defineDimmed && !editingSlug`, so Edit always opens.

**Worth knowing what this narrows:** a Class can no longer be defined
from nothing. Every Class now begins with an imported checkpoint. That is
coherent with the platform as it stands -- training from scratch was
removed, so importing is the only way to get a first version, and a
Class with no versions could not be trained anyway -- but it is a real
constraint rather than a cosmetic one, and the lever back is the single
expression above.

The notice says all three ways out rather than only the one that applied
before: read a checkpoint and untick *Use the classes in this file*, or
press *Edit* on an existing Class.

**Dead state removed with it.** `fileClassesInUse` on the page,
`onClassSourceChange` on the import card and `checkpointChosen` inside it
all existed to say "the import card is supplying the classes" -- one of
the ways this card is not the place to be, and now not the one the rule
reads. `pendingImport` carries the whole signal.

1,869 tests pass; the 3 known environmental failures remain.

---

## The "Train on an existing model" tick is gone (2026-10-13)

On dev as `20261013-042504`. Removed on request. It meant "I am
training, not importing" and dimmed the import card above; the card
itself never switched off, because it is the only way a run gets its
starting weights and a box that could disable it would break the page.

Removed with it: `onlyThis` / `onOnlyThisChange` on the picker,
`importDimmed` on the panel and the wrapper it drove, and
`trainingFromExisting` on the page -- including the line that set it true
after a successful import. Nothing dims the import card any more; zero
references to either name remain in both files, checked on the running
container.

The card is a plain heading again, which is what it was two deploys ago
before the tick arrived.

1,869 tests pass; the 3 known environmental failures remain.

---

## The data.yaml picker works like the checkpoint picker (2026-10-13)

On dev as `20261013-044636`. The dataset browser offered FOLDERS with
tick boxes and read `<folder>/data.yaml` from whichever was ticked. It
now lists the `.yaml` files at the level being browsed and selects one,
exactly as the checkpoint picker lists `.pt` files -- one browser, one
behaviour, rather than the same component used two ways on one page.

Two things that fixes, and the second is the one that bites:

  * a dataset whose file is `dataset.yaml`, or which carries several, was
    unreachable;
  * **a folder tick box says nothing about which file it is about to
    read.** "Read data.yaml" on a folder holding three of them was a
    guess presented as a choice.

`list_s3_folders_and_yaml_files` mirrors the weights one, from a single
`list_objects_v2`. `classes_from_yaml_key` reads the EXACT object, so
picking `dataset.yaml` reads that and not the `data.yaml` beside it.
Both `.yaml` and `.yml` count -- Ultralytics writes the first, plenty of
exports write the second.

`GET /track2/models/dataset-classes` now takes `folder` OR `key`.
`folder` keeps its old meaning (`<folder>/data.yaml`) because the
training form and the assemble picker genuinely have only a folder in
hand, and both still call it. `dataset_classes` in `track2_shared` is
untouched for the same reason.

**A test caught a real break, and it was mine.** `/track2/models` builds
`dataset_browser`, so swapping `list_s3_folders` for the new function
made that endpoint reach S3 in a test environment with no bucket
configured -- `test_queue_add_remove_start_stop` fails on the GET, not on
the queue. The test stubs each lister by name; the new one is stubbed
beside `list_s3_folders_and_weight_files`.

**Also worth recording: `git stash` was the wrong way to answer "is this
failure mine".** It took the untracked files with it, so the
"before" run failed for a different reason and proved nothing. Reading
the traceback -- which named `client.get("/track2/models")` -- answered
it in one step.

Live: `retraining` lists `data.yaml` beside its week folders, and reading
it returns the six classes.

1,876 tests pass; the 3 known environmental failures remain.

---

## "Fill these in from a data.yaml" takes a local file too (2026-10-13)

On dev as `20261013-060931`. Two sources now: **From my computer** and
**From the bucket**. The bucket half is the file picker from the last
change; the computer half is a file input filtered to `.yaml` and `.yml`,
which is the whole set this card reads.

A dataset exported from Roboflow arrives in Downloads. Uploading it to S3
to read six class names off it was a detour, and left a file in the
bucket that nobody asked for.

**The file is parsed server-side, and that is the decision worth
recording.** The browser reads the file and posts its TEXT;
`POST /track2/models/dataset-classes/from-text` runs it through the same
`classes_from_yaml_text` the bucket path uses. The alternative was a YAML
reader in TypeScript -- which would be the THIRD hand-rolled reader of
this exact key in the codebase, after `yaml.safe_load` here and
`pull_control._classes_from_yaml` (which cannot import PyYAML because it
IS the Lambda handler Terraform zips). A third is the one that drifts,
and the failure mode is a silently shifted class index: every box trained
under the wrong name, nothing erroring.

**Nothing is uploaded.** A data.yaml is a few hundred bytes, so it rides
in the request; empty and >1MB are refused, because anything that size is
not a data.yaml and parsing it would be somebody else's problem arriving
here.

Both sources land in one `applyReadClasses`, so a yaml off a laptop and
one in the bucket cannot be interpreted differently -- a test pins the
call count.

Live: the route is registered, and the same parser returns the six
classes from `retraining/data.yaml` and two from posted text.

1,882 tests pass; the 3 known environmental failures remain.

---

## The data.yaml browser can explore any bucket (2026-10-13)

On dev as `20261013-062428`. "From the bucket" meant "from the one bucket
this app happens to write to" -- the browser rode on `/track2/models`,
which is bound to `BUCKET`. The datasets are as often somewhere else:
the account also holds `safety-model-dataset` and
`safety-model-dataset-dev`, both readable, both previously unreachable
from this card.

There is a bucket dropdown now, the same one Export Control uses, and two
new routes beside the existing `/folders`:

| Route | What |
|---|---|
| `POST /track2/pull-control/yaml-files` | folders AND the `.yaml`/`.yml` beside them, any readable bucket |
| `POST /track2/pull-control/yaml-classes` | the class names one yaml declares, any readable bucket |

`/folders` returns folders only, which is right for picking a dataset
ROOT; reading class names needs the file.

**Three sources, one parser.** A yaml off a laptop, one in this app's
bucket and one in somebody else's all go through
`t2.classes_from_yaml_text`, and a test asserts all three call sites do.
Three readers of a class index is how one of them starts disagreeing,
and the failure is silent: every box trained under the wrong name.

Three details worth keeping:

  * **The extension is checked BEFORE the fetch.** This is handed a key
    the browser chose, and "read whatever key you are given" is a wider
    door than reading class names needs. Verified live: a `.txt` is
    refused without an S3 call.
  * **Only the current level.** A nested key would render as a name with
    slashes in it and select something the browser is not showing.
  * **Changing bucket clears the path.** Keeping the old prefix would
    walk to somewhere that does not exist in the new bucket and read as
    an empty bucket. A bucket that cannot be read lists as empty rather
    than leaving the previous one's folders under the new name.

**One CodeBuild failure on the way**, TS2345: `applyReadClasses` was
typed `Track2DatasetClasses` while the new route returns a narrower
shape. Narrowed the parameter to `{ classes }`, which is all it ever
read -- the wider type was claiming a contract the function did not need.
Nothing deployed; the gate working.

1,887 tests pass; the 3 known environmental failures remain.

---

## "Define a new Class" is two titled parts (2026-10-13)

On dev as `20261013-065021`. The card ran Name, then a collapsed "Read
from a data.yaml" button, then the class list, as one undifferentiated
form -- so "name the Class" and "where do its class names come from"
read as one decision rather than two, and the second was behind a toggle.

    1 . Name the Class
        Name [                    ]
    2 . Read its classes from a data.yaml
        ( ) From my computer   (o) From the bucket
        [ bucket v ] [ browser, .yaml only ]
        [Read this file]
        Classes, one per line [ ... ]

The picker is no longer behind a button. Reading the names IS the way to
fill this in -- a label file carries `0 0.51 0.22 ...` and no name at
all -- and typing them is the fallback; a collapsed toggle made the
correct route the one you had to go looking for.

**Four things broke on the way, and the order they were found in is the
point.**

  * **A CodeBuild failure**, TS2552: the "Use these N" button still
    called `setReadOpen(false)` to close a toggle that no longer exists.
    Caught by the gate; nothing deployed.
  * **A read confirmation that stopped rendering.** `readNote` lived
    inside the block the toggle opened, so a successful read said
    nothing. Re-rendered under the button.
  * **A note that lied about its source.** It said
    `from ${picked}/data.yaml` -- appending `/data.yaml` to a value that
    is already a full key, and saying it about a local file too. The
    label is named once now (`yamlSourceLabel`) and used in all four
    places it appears.
  * **The card's own `?` help still named the removed button.** tsc
    cannot see prose, and a test did: it asserts no USER-VISIBLE text
    names it, stripping comments so the history can still be explained.
    A sweep over every page for removed control names found no others.

**The listings follow the card being in play.** They gated on `readOpen`;
with it gone they would have run on every page load, including while the
card is dimmed and nothing can be chosen. They gate on `bodyDimmed`,
which had to move above them -- a dependency array is evaluated during
render, so a const declared after would be in its temporal dead zone.

### A deploy went out on a red suite

Worth recording as process, not code. The command was
`pytest ... | tail -5 && deploy.sh`: `&&` chains on **`tail`**, which
always succeeds, and the pipe threw pytest's exit code away. The suite
had one failing test and shipped anyway.

`pytest && deploy` cannot work here either, because the three
environmental failures mean pytest always exits non-zero. So there is now
a gate that runs the suite and exits 0 only when the failures are exactly
those three -- and says so loudly if one of them starts PASSING, since a
stale allow-list is how a real failure gets waved through later.

1,891 tests pass; the 3 known environmental failures remain.

---

## A seed could not be registered into an existing Class (2026-10-13)

On dev as `20261013-070612`. Reported as "I am not able to register a
seed weight after importing", and it was a real dead end rather than a
confusing control.

**The Class dropdown listed only Classes whose class names MATCHED the
file**, captioned "only those whose class names match". A COCO checkpoint
detects eighty things and matches nothing on this estate, so the dropdown
was always empty and the only offer was *create a new Class from this
file* -- which makes an eighty-class Class called "Vehicle".

Giving an EXISTING Class its first checkpoint to train from -- which is
exactly what a seed IS, and the reason the seed machinery was built --
could not be done from this card at all. The only route to it was
unticking *Use the classes in this file* and defining a Class by hand,
which creates a NEW one. There was no path to `helmet`.

**Every Class is offered now**, with the mismatch stated in the option
itself rather than discovered afterwards:

    Create a new Class from this file
    helmet    -- SEED, it detects 1 other class
    segment   -- SEED, it detects 4 other classes
    unversal  -- SEED, it detects 1 other class

Choosing a mismatching one shows the seed panel before the press, and
passes `allow_class_mismatch` so the server marks the version a seed and
registers it **Pending rather than Ready**. Choosing a matching one, or
creating a new Class, registers normally -- a new Class is built FROM the
file's classes, so it never mismatches and is not waved through as
though it might.

The built-in `ppe` is left out: `register_uploaded_base_model` refuses it
outright, since it owns the production groups the replay pipeline feeds.

**Nothing is preselected any more.** The card used to select the first
matching Class on inspection. With every Class now listed, that same slip
would file a checkpoint into one with entirely different classes, so the
dropdown opens on *Create a new Class from this file*.

1,896 tests pass; the 3 known environmental failures remain.

---

## Your Classes says what the chosen one detects (2026-10-13)

On dev as `20261013-072049`. The row read `PPE2 (1 classes)` and where
its versions are filed, and said nothing about the taxonomy -- which is
the thing every dataset is checked against and the thing a training run
inherits. A line under the dropdown now names it:

    Detects  0:horizontal_plane, 1:vertical_plane, 2:hatch_wall, 3:dangerous_plane

**With the index, in index order, sorted numerically.** The index is what
a label file actually carries -- a row reads `2 0.51 0.22 ...` and never
"vehicle" -- so "which name is index 2" is the question this answers.
Numerically because as text 10 sorts before 2 and the list would go
wrong from the eleventh class on.

Also `1 classes` -> `1 class` in the dropdown itself.

1,898 tests pass; the 3 known environmental failures remain.

---

## The page went blank -- a temporal dead zone tsc could not see (2026-10-13)

Fixed on dev as `20261013-074117`. `20261013-072049` -- the "Detects
0:helmet, 1:gloves" line added a few minutes earlier -- rendered nothing
at all.

**The server was never involved.** `/build`, `/health` and `/` all
answered 200, and the container logs showed the browser fetching every
endpoint successfully: `/track2/models`, `/track2/pull-control`,
`/track2/models/detectors`, all 200. The app loaded, fetched, and then
threw while rendering. Checking the server first is what made that
quick -- it ruled out half the search space in one call.

The cause:

```
161:  const chosenClasses = Object.entries(
162:    detectors.find((d) => d.slug === chosenSlug)?.classes ?? {},
163:  ).sort(...);
...
190:  const [chosenSlug, setChosenSlug] = useState("");
```

Line 161 runs at render while `chosenSlug` is still in its temporal dead
zone, so every render threw `Cannot access 'chosenSlug' before
initialization`. It was placed beside `bodyDimmed`, which had been moved
UP so the effects could gate on it -- and the new const was dropped in
next to it without checking what it read.

**Why the whole page went, not one card.** The throw is in
`Track2DetectorsPanel`'s component body. `CardBoundary` catches a crash
inside a card's children; a crash while the panel itself renders happens
above every boundary, so React unmounts the tree.

**Why `tsc` passed it.** TS2448 flags a DIRECT use before declaration.
Here the reference sits inside the `find` callback, which TypeScript
cannot prove runs immediately, so it compiled clean. The TypeScript gate
is real and it is not total -- this is the shape that gets through it.

`scratchpad/tdzcheck.py` now finds it: for every `const` in a component
body it checks whether the statement reads a `useState`/`useRef`/
`useMemo` binding declared further down. Verified both ways -- clean
against the fix, and it reproduces the exact line and both line numbers
against the reconstructed bug. Swept every component and route: clean.

1,898 tests pass; the 3 known environmental failures remain.

---

## The week prefix is fixed, and shown (2026-10-13)

On dev as `20261013-075828`. It is a read-only field displaying the live
value -- `week_` -- with a line saying why it cannot be changed.

**The reason is not style, it is the data.** `plan_routing` and `week_at`
skip any folder whose name does not start with the configured prefix.
Changing it renames nothing: it makes every week folder already in the
bucket invisible to the system, all at once. There are seven
(`week_1`..`week_7`) under `retraining/`.

**It shows the real stored value, not a hardcoded string.** A read-only
field that hid what is configured would just be a field that lies --
`week_3` on screen and `week_3` in the bucket have to be the same
string. Two deploys ago it was blanked so a placeholder could speak,
which was right while it was editable and wrong the moment it was not.

`readOnly`, not `disabled`: a disabled input is skipped by the keyboard
and reads to a screen reader as unavailable, when the value is the thing
worth reading. `aria-describedby` ties it to the explanation.

**The gate earned its keep.** Two tests failed and the deploy stopped:
one was the old assertion about the placeholder, superseded; the other
was my new one tripping over my own comment, which explains why
`readOnly` was chosen over `disabled` and therefore contains the word
"disabled". It strips comments now -- the same fix the "Read from a
data.yaml" test needed, which says that check belongs in any assertion
about markup.

1,900 tests pass; the 3 known environmental failures remain.

---

## Outstanding, in order

Rewritten 2026-09-30 after seven read-only auditor agents covered all
twelve pages. Every claim below was verified against the code before
being written down; where a fix is obvious it is named.

**Done since the last version of this list:** `0a` the unattended
retrain (see above -- it runs in the app, not a Lambda); `0c` YOLO26 and
the generation work; the Multi-Auto Training control sweep.

**Done 2026-10-01:** the pull schedule is now editable in exactly one
place (the pinned toolbar) rather than two; **C**, the `yolo11` hardcode
sweep, is closed apart from two named items; **D1**, the Notifications
Remove that could never work and the mock that said it could; **E3**, the
`data.yaml` parser that accepted one of the four shapes Ultralytics
allows.

Also 2026-10-01: Import a pretrained checkpoint moved above "Define a new
Class" and is never greyed (it files a .pt, it does not train, so as an
exclusive option it greyed out both modes that do); Auto Training's root
now follows the pull control's; the card re-discovers after a pull; a
finished pull emails the subscribers (**C3** -- manual pulls still do
not, one IAM grant short).

**Not touched, and deliberately:** **D2** (widening the
`training_pipeline_completed` rule so the retrain chain and custom runs
notify at all) starts sending email to real subscribers for runs that
currently send none. It is a one-line `pipelineArn` change in
`monitoring.tf` plus the matching targeted AWS change -- say the word and
it is a few minutes -- but it is outward-facing, so it is yours to
approve rather than mine to assume.

---

### A. Wrong numbers presented as right -- do these first

**A1. Evaluation can score the wrong images against the wrong labels.**
`track2_shared.py` `queue_eval_run` validates only that the ARGUMENTS
are present -- no taxonomy check at all -- and the scoring loop compares
`int(parts[0])` from the label against `d["class"]` from the prediction.
A name never enters the chain. Point a person/helmet model at a
`{0: forklift, 1: person}` dataset and you get a completed run with
plausible numbers attributed to the wrong labels.

The rest of the codebase already prevents this: `evaluate_fullframe.py`
raises if `model.names` differs, `evaluate_any.py` calls it "a
consistency check", and `validate_dataset_folder(used_for_evaluation=True)`
sits UNUSED in the same module. This page reimplemented evaluation from
scratch and skipped every guard.

**A2. It silently scores at most 1,000 images.** `list_s3_images` makes
one `list_objects_v2` call with no continuation token -- S3's hard cap.
A 4,000-image dataset is scored on whichever ~1,000 keys come back
first, alphabetically, so likely one camera or one date, and reported as
`num_images: 1000`. `list_eval_runs` twenty lines below DOES paginate,
so this is an oversight, not a convention.

**A3. And `Delimiter="/"` means flat folders only**, so a proper
`images/val/` dataset finds nothing and fails -- while the layouts that
DO work are the flat training folders. The worst pairing: the datasets
it succeeds on are the ones where a model is scored on its own training
images, and nothing says so. `_evaluates_on_its_training_images` exists
in the same module and is not called here.

**Treat every number this page has produced as unverified.**

**A4. Deploy can run one model and record another.** `_deployed_matches`
decides whether to skip the endpoint reload by comparing
`deployed["active_size"]` and `deployed["source"]` -- and both label
strings are built as `f"YOLO11-{size.upper()} v{version}"`, carrying NO
model package group. Versions are per-group, so `yolo11m-detector v3`,
`yolo11m-acme-detector v3` and `yolo26m-acme-detector v3` are
indistinguishable. It skips the redeploy, runs the previously loaded
weights over the new job's images, and writes lineage naming the model
that was intended. Fix: tag the Model with the `model_package_arn` and
compare that.

**A5. Lineage is empty for every run anyone actually starts.** It reads
`track2-dynamoDB`, written only by the legacy CodeBuild path. Detector
runs and the scheduled retrain go through `start_custom_training_run`,
which writes a training-audit row instead. Weeks of runs, and the page
says "No lineage records yet". Fix: back it with
`scan_training_audits()` and merge the legacy rows for history.

---

### B. Safety

**B1. Approve swaps live production with no confirmation.** On the
production tab the group is `yolo11{s,m,l,x}-detector`, the exact-match
list on the EventBridge rule that fires `dump_approved_weights` --
overwriting `latest_approved_model_weights/<size>/model.pt` and swapping
the live inference component. The only `window.confirm` on that page
guards DELETE. Fix: label it "Approve & deploy" in the production tab
and gate it behind an explicit confirm naming what it replaces.

**B2. The Schedule page rewrites your schedule if you only look at it.**
`load()` never parses `schedule.expression`; any enabled rule renders as
"weekly, SUN, 09:00" from the `useState` seeds. Press Save without
touching anything -- the natural thing after checking the page -- and a
daily rule is overwritten. Fix: parse the expression back, or return the
structured fields; until then disable Save until something changes.

**B3. Monthly on day 1 before 08:00 SGT fires on the 29th.** The UTC
day-shift substitutes day 28 of the same month rather than the last day
of the previous one, while the summary string still says "day 1".
EventBridge supports `L` for last-day-of-month. Also `ScheduleRequest`
has no bounds, so a typed 31 silently skips five months.

---

### C. The `yolo11` hardcode sweep -- mostly closed (2026-10-01)

Each item was a literal `yolo11` or a `removeprefix("yolo11")` that is a
no-op for any other generation. **Done**, on dev as of `20261001-073627`
and the deploy that follows it:

  * `export_weights_to_folder` -- was exporting `yolo26s-open-detector`
    as `yolo11-yolo26s-open-v3-<date>.pt`. Now `yolo26-s-open-v3-...`.
    Every generation-11 name is byte-identical to before, proven by a
    test that keeps the old implementation beside the new one, so
    nothing already in a folder changed name or stopped matching the
    archive-on-collision check.
  * `open_registry_group(f"yolo11{size}")` in the registry scan and the
    base-model picker -- replaced with one `builtin_open_groups()`.
    YOLO26 open-taxonomy models were trained, registered Approved, and
    on no page anywhere, with nothing erroring to say so.
  * the deploy label and the Model tag (`source_desc`), plus the two
    queue labels (eval and inference) -- four independent copies of
    `f"YOLO11-{size.upper()} v{version}"`, now one
    `registry_source_desc()` that reads the generation off the group.
    Guarded by a test that reads each caller's source with comments
    stripped, so a fifth copy fails rather than merely reading wrong.
  * `scan_all_model_metrics` rows now carry `generation`
  * the metrics UI's five places -- chart legend, bar tooltip, x-axis,
    the generation line, the production block header -- now read it.
    The page-level line lists the generations actually registered
    instead of asserting one.
  * the Export page rendered four hardcoded `yolo11{size}-detector`
    keys while the route had always passed `include_open=True`. Every
    other group was fetched and silently discarded -- no empty state,
    no error. It now renders every group it receives.

**Still open:**

  * the eval page's `const GENERATION = "11"` and its group string.
    Left alone deliberately: Eval reads `serialize_registries_brief()`
    without `include_open`, so it only ever sees the four production
    groups, which really are YOLO11-only. Making it generation-aware
    means first deciding whether evaluating an open-taxonomy model
    against a production-taxonomy golden set is meaningful -- a
    taxonomy question, not a naming one. See A1.
  * `Track2DeployPage.tsx:187` renders lineage as `YOLO11-${size}`
    unconditionally. Deploy only offers production ARNs today, so it is
    not reachable from the UI -- but `model_package_arn` is free-form on
    the route, so the server will deploy and tag whatever it is given.
    The tag itself is now correct; only this one display is not.

The production groups `yolo11{s,m,l,x}-detector` are deliberately
YOLO11-only and must stay: `terraform/platform/sagemaker.tf` creates
exactly those four from `yolo_size_group_names`, and the same four are
an exact-match list in the EventBridge rule that fires
`dump_approved_weights`. `yolo26{size}-detector` does not exist in AWS;
YOLO26 lives in the open and per-detector groups, which match the
existing `yolo*-detector/*` IAM grant and are correctly inert there.

---

### C2. Settings files restored the Class but not the version (fixed 2026-10-01)

Reported as "I selected version 1 of the helmet Class at yolo11s, saved
it in meow, and on reload it was not recorded".

`meow.json` on the host was CORRECT -- it held
`default_detector = {slug: helmet, architecture: yolo11s}` and
`default_version = .../yolo11s-helmet-detector/1`. The save was never the
problem; the restore was.

The Models page loads the config in **three** places at once: the page
itself (restores `default_version`), `Track2DetectorsPanel` (restores
`default_detector`), and `TrainingForm` (preselects the detector). The
panel's restore calls `onChoose`, and the page's `onChoose` cleared
`basePackageArn`. Clearing is right when somebody PICKS a different
Class -- a version belongs to one registry group and cannot follow it --
but wrong on a restore, where the version being put back belongs to
exactly that Class. The two were indistinguishable, so whichever fetch
resolved last won: roughly half the time the panel wiped the version the
page had just restored.

`onChoose` now takes a `restored` flag and only invalidates the version
on a real change.

Second gap in the same file: `default_training` saved as `{}` unless a
field had been edited and blurred, so a file restored "whatever the page
was showing" rather than what was saved. The panel now writes the shape
once when none is recorded.

**Still worth doing:** collapse those three concurrent
`getTrack2PullControl()` calls into one. The flag makes the ordering
harmless, but three components independently restoring from three
responses is the shape this bug came out of, and the next field added
will have the same problem.

---

### C3. Pull -> Auto Training -> notification (2026-10-01)

**The card now re-discovers itself after a pull.** It listed whatever was
there when it mounted and said nothing when that changed: run a pull,
come back, and it still read "No folders directly under retraining" until
somebody pressed Reload -- worse on a schedule, where nobody is on the
page to press it. It polls `GET /track2/pull-control/last-pull` every 20s
and re-discovers when `finished_at` moves.

That endpoint is deliberately tiny -- one DynamoDB query -- because the
full status response lists S3 to build `weeks` and is far too heavy to
poll. It reads `run_state`, not the event rows: `_record_run` skips a
scheduled sweep that moved nothing, and "the schedule fired and found
nothing" is exactly what somebody watching this wants to know. It is NOT
under `/run/`, because `GET /run/{run_id}` is registered first and would
have received it as a run id.

The card also states what it is as of ("Last pull ... -- 12 files
moved"). Without it, "no folders" reads the same whether the pull has
never run or ran and found nothing.

**Auto Training's root now follows the pull control's.** It used to seed
the root once and then let its own saved value win forever; change the
pull's root afterwards and the card showed the old one while `run_chain`,
which reads `config["root"]` directly, used the new one. `follow_pull`
defaults true (a missing flag reads as follow, so old configs are not
stranded), an override is explicit, and the card says in red when it is
overridden -- because the automated run still uses the pull's root.

**A finished pull emails the subscribers.** The publish sits in
`route_pending`, the one line BOTH triggers reach: a manual Pull now runs
in the app container, the schedule and arrival rules run the same
function in the Lambda. In the router it would have notified for exactly
the pulls somebody was already watching and stayed silent for every
unattended one.

Two deliberate behaviours, both tested:

  * **A quiet sweep says nothing.** Files moved, errors, a failure or a
    stop get an email; a schedule that fires and finds nothing does not.
    A daily "0 files" is how an address gets filtered, and then the one
    that mattered goes unread too.
  * **A failed publish cannot fail a pull.** AccessDenied is logged, not
    raised. A pull that moved files has done its job, and failing it
    because SNS was unreachable turns a notification problem into a data
    problem.

`PULL_NOTIFY_TOPIC_ARN` is named in `terraform/app/pull_control.tf`
rather than derived, because that file IS the Lambda handler Terraform
zips and may import nothing beyond boto3 and the standard library --
`track2_shared`, where every other caller gets this ARN, does not resolve
there. Empty means "do not notify".

**STILL OPEN -- manual pulls do not notify.** `sns:Publish` was granted
to `track2-app-pull-control-router` (in `terraform/app`, applied
2026-10-01), so scheduled and arrival pulls notify. A manual Pull now
runs in the app container under **`track2-yolo-review-host`**, which
`aws iam simulate-principal-policy` still reports as `implicitDeny`. That
role lives in `terraform/platform`, the drifted stack, so it wants a
targeted AWS change plus a matching `.tf` edit rather than an apply.

Check it with, rather than by sending a test email to a real subscriber:

```
aws iam simulate-principal-policy   --policy-source-arn arn:aws:iam::<account>:role/track2-yolo-review-host   --action-names sns:Publish   --resource-arns arn:aws:sns:ap-southeast-1:<account>:track2-yolo-reviewer-notifications
```

**Nothing has fired yet either way**: `picked` is empty and `sources` is
`[]`, so arrival mode is armed but watching nothing. Delivery has been
verified as far as permission, topic ARN, code and call site -- not as a
received email, which needs a pull that moves real files.

---

### D. Controls that do nothing, or lie

**D1. DONE 2026-10-01.** Remove is no longer offered for an unconfirmed
subscriber -- there is no ARN to unsubscribe until they click the link,
and no AWS call that could delete one, so the row now says it expires on
its own in three days. Posting one anyway gets a 422 explaining why
rather than a bare 500. `FakeSnsClient.unsubscribe` now rejects a non-ARN
the way SNS does, which immediately failed the test that had asserted
Remove works for a pending subscriber -- that test now asserts the 422
and a separate one pins the mock itself, so a double that is more
permissive than the service cannot certify an impossible call again. The
React key is the email, which is unique per topic.

**D2. Half this platform's training runs send no email.** The
`training_pipeline_completed` rule filters on
`track2-yolo-training-pipeline` only. The retrain chain and every custom
run use `track2-yolo-open-training-pipeline`, match no rule, and
therefore never notify -- while the page promises "training pipeline
status". Fix: widen the `pipelineArn` list in `monitoring.tf`.

**D3. Nothing proves an email can be delivered.** No test-send endpoint
exists. "Confirmed" means only that SNS issued an ARN. Subscribing an
already-confirmed address is a no-op that still says "Confirmation email
sent". Terraform-managed subscriptions appear in the list, can be
removed here, and come back on the next apply.

**D4. Two retrain schedules, on two pages, neither mentioning the
other.** The Schedule page calls itself "the single global retrain
schedule"; it drives the full-frame REPLAY coordinator on production
lineage. The Retention/Models pages drive the OPEN pipeline on a
detector's own taxonomy. Setting both gives two unrelated retrains, and
the single-GPU guard refuses whichever loses, naming a run the user did
not start.

---

### E. Unfinished, presented as finished

**E1. Synthetic Image Generation is a stub** -- static JSX, no state, no
API import, no router, no S3 write. It holds a permanent nav slot beside
eleven working pages. Fix: drop it from `NAV_ITEMS` and keep the route,
the pattern already used for retired paths.

**E2. There is no way to mark an image as synthetic, anywhere.** Not in
the snapshot manifest (a flat list of keys), not in the generated
data.yaml, and the UI field named `provenance` answers a different
question ("how the dataset was pinned"). Meanwhile
`seed_training_data.py` has ALREADY written 275 fabricated labels
(`"{class_id} 0.5 0.5 0.5 0.5"`) into the pipeline bucket with no
metadata, no marker, and a warning that lives only in a docstring.
Specify the marking before a generator ships: object metadata
`x-amz-meta-origin: synthetic` survives `copy_object` with
`MetadataDirective="COPY"`, which the snapshot code already uses.

**E3. Parser DONE 2026-10-01; the Help page copy is not.**
`pull_control._classes_from_yaml` now reads all four shapes Ultralytics
allows -- block mapping, block sequence, inline list, inline mapping --
and a test asserts it agrees with `yaml.safe_load` on ten documents, so
the guided flow and the standard page can no longer disagree about the
same file. It stays hand-rolled on purpose: this module IS the Lambda
handler file that Terraform zips, so it may not import PyYAML. It still
refuses to renumber a non-numeric index rather than guess, because a
shifted index is a silently mislabelled class.

`deploy.sh` pushes this file to the Lambda as well as the image, so both
copies move together.

**Still open:** the Help page documents `nc:`, which nothing reads and
nothing writes.

**E4. Help has no orientation at all** -- it never names a page, a card
or a step, for an audience with no ML background. The single most
valuable addition is a "Where to start" card naming the six steps and
which page they live on.

---

### F. Carried over, still true

  * **Golden Set is a dead end** -- it splits a folder and nothing
    carries the result; Retention needs the same path typed again. Its
    own settings are not persisted either.
  * **Nothing says what is blocking you on the MANUAL path.**
    `preflight()` does exactly this for the scheduled one.
  * **A run does not record `snapshot_id` on its audit row**, so
    snapshot -> run -> registered version is still not closed. The queue
    entry now carries it; the audit row does not.
  * **Retention is four steps nested two cards deep.**
  * **`derive_filtered_dataset` cannot read a week-root.** Only matters
    for a detector wanting SEVERAL of a dataset's classes; one class
    goes through `classes=` + `single_cls` with no copy. Honestly
    reported, so a real limit rather than a bug.
  * **The review host is at 81% disk.** `deploy.sh` prunes each time so
    it self-corrects, but the margin is thin.
  * **Two `GENERATION_SIZES` copies with no test binding them** --
    `track2_shared.py` and `model_training/src/config.py`. Both were
    updated for YOLO26 by hand.

---

### G. Cost and speed, verified

  * **Deploy polls every 3 seconds** and `GET /track2/deploy` rebuilds
    `registries` + `folders` + `weight_files` on each -- ~30 serial
    SageMaker calls -- and the page destructures NONE of them. Worst
    during the multi-minute window a user sits watching a deploy.
  * **One Metrics page load runs the registry scan three times over**,
    and a folder click in the export browser runs a fourth.
  * **Cost is real Cost Explorer data**, not estimates -- but it is
    account-wide SageMaker presented as "Track2", with no tag filter.
    Anyone else's notebook lands in the total.
  * **Cost returns one more day than asked** (`start = end - (days+1)`)
    and labels it from the row count, so "7d" reads "last 8 days" and an
    empty account reads "last 0 days".

---

### H. Operational -- yours to set, nothing to build

  * **RETENTION IS OFF.** The decay policy does not apply; the yaml
    names the week folders whole and a run trains on every image. The
    entire retention design is inert until the box is ticked.
  * **Epochs is 1.** Trains, but will not produce a usable model.
  * **YOLO26 has nothing to seed from.** Production weights are all
    YOLO11 and seeds are filtered to the matching generation, so a
    YOLO26 run needs an imported `yolo26s.pt` or scratch.
  * **The root `data.yaml` has five blank class names** (`0: helmet`,
    `1..5: ''`). If the labels genuinely use 1-5, every model in that
    group is mislabelled and nothing errors.
  * **Production (8501) is on `20260929-133333`** -- a day behind, none
    of this promoted. Promotion is `IMAGE_TAG=<dev's tag>`, never a
    rebuild.
  * **No training run has succeeded since 2026-09-24.** The two after it
    failed on the `_retained/` path bug. ultralytics 8.4 has never been
    exercised on a YOLO11 run. The next scheduled run is the first test
    of both.

---

## What was done

### One settings file for every card

**And ONE save button.** The card briefly had two, both called Save
settings: the old one above the bars wrote only the decay policy, the new
one below them wrote the file. Two buttons with the same name on one
card, one of which saves a quarter of it, is how a policy gets
half-written. The retention-only one is gone.

The remaining one does both halves IN ORDER: it applies this card's
retention controls to the live config first, then writes the named file.
That order is the whole point -- the file is a copy of the live config,
so writing it while the controls were still only on screen would save a
policy nobody had set.

`Preview` stays and needs nothing saved; it sends the controls as they
read.

Load makes every setting live, but only THIS card rereads itself -- the
pull control and Auto Training hold their own state and cannot be told
from here. The note says so and asks for a page reload, rather than
leaving those two showing the previous values.


`save_settings_file` writes `read_config()` -- the LIVE config in
DynamoDB -- so it already carried pull control AND retention, which keeps
its settings under the `retention` key of that same blob. Two cards were
missing: Auto Training and the Golden Set.

Auto Training now stores `auto_training: {root, excluded}` in the config,
so its selection rides along. It is written debounced and NOT until the
saved value has been read back -- writing before that would persist the
card's empty starting state over whatever was saved, which loses it
silently. The Golden Set was deliberately left out (asked and answered).

**Saved settings moved to the Retention card**, just above Repeatable
runs, and came off Pull Control. It works from there precisely because
the file is written from the live server-side config: this card saves
four cards' choices without knowing what any of them contain.

The limit is stated in the card rather than left to be discovered: it
saves what is IN EFFECT. A change typed into Pull Control and not applied
is not in the file.

Pull Control keeps `settingsFiles` -- the weeks preview's "viewing" list
reads it -- but `settingsName`, `ticked`, `confirmDelete` and
`settingsDir` went with the block.

The two old files were moved to
`/opt/track2-app/settings/.superseded-20260929/` rather than deleted, and
the name box now defaults to `track2`.

### Auto Training is a folder view now, not an assembler

Removed: the images/labels subfolder inputs, Classes, Destination, "What
will be combined", and "Assemble from all folders". 570 lines to 468.

The subfolder names come from the pull control instead of being typed
again. Two places to set them was two places for them to be wrong, and a
mismatch reads as a root that holds nothing.

Dropping the assemble path left `setError` dead, which was worth noticing
rather than deleting: discovery failures had been swallowed, so a root
that could not be READ looked exactly like a root that was EMPTY. It now
says which.

### Training a one-class detector from a six-class dataset, without a copy

`retraining` declares six classes; the helmet detector declares one. The
old answer was `derive_filtered_dataset`, which rewrites every label file
and copies the images -- and which reads `<source>/images/train`
directly, so it cannot read a root of week folders at all.

Ultralytics can do it itself, verified in the installed source:

    data/build.py     classes=cfg.classes, single_cls=cfg.single_cls
    data/base.py      update_labels(include_class) drops other boxes;
                      single_cls renumbers what survives to 0
    engine/trainer.py single_cls -> names={0:"item"}, nc=1

So `classes=[0]` + `single_cls=True` filters the boxes, renumbers them
and sizes the head to one output, touching nothing in S3. The week
folders stay the record.

`class_filter_for()` decides which of three applies, and it is a fact
about the two taxonomies rather than a preference:

  * identical -> nothing to do
  * ONE wanted class, present in the dataset -> the flags, no copy
  * SEVERAL wanted classes -> still a copy. `classes=` keeps their
    original indices and `single_cls` collapses them all into one;
    neither is an n-class model, and Ultralytics has no renumber-to-0..n
    mechanism.

Validation is skipped on the flags path: it checks the yaml equals the
detector's list, which is true after a copy and false when the filtering
happens at load time -- it would reject exactly the case this serves.

Verified in a real job: `classes 0`, `single-cls true`, image
`sha256:cbbcfe2f`, `train-data-s3-uri s3://track2-s3/retraining/`.

### The data.yaml's classes are editable before it is written

Index -> name rows above Build it, seeded from what the build derived.
NAMES only: a label line is `2 0.51 ...` and means class 2, so the index
is what every label file carries. Renaming index 2 renames every box
numbered 2 -- the useful edit. Adding or removing entries changes what
the numbers refer to, which is how a dataset silently starts training
the wrong thing, so it is not offered. Training on a subset is what the
detector plus the class filter does.

### Repeatable runs shows what was actually sent

A snapshot says what COULD be trained on; the runs list says what WAS.
Each row carries its dataset, count, status and its frozen copy under
`_runs/<id>/`, plus **where the names came from** -- "from the file
lists" or "whatever the folders held". Only the first is reproducible
exactly, and that is invisible from the counts alone.

**Load** and **Train on this** are deliberately separate: Load pulls a
snapshot's classes into the editor and touches nothing; Train on this
restores it to the live slot and points the form at it.

### The retention policy now reaches the run

It did not before, and that is worth stating plainly: after *Build
dataset* was removed, the only training path was the `data.yaml`, which
names week FOLDERS. A run trained on every image in them whatever the
decay said. The card could report "keep 301 of 310" while the run used
all 310, and nothing in the system disagreed.

**Ultralytics resolves a split entry two ways** (`data/base.py`,
`get_img_files`): a directory is globbed whole, and a FILE is read as a
list of image paths, one per line, with a leading `./` resolved against
the list's own parent. Labels are then found per image by swapping
`/images/` for `/labels/` -- so a list may cross week folders freely and
each entry still finds its own label. That is the hook, and it needs
nothing added to the trainer.

`retained_lists.py` writes `retraining/_retained/train.txt` and
`val.txt`; the yaml points at them. An "Apply the retention policy" tick
in the data.yaml card, on by default.

**The train side is decayed; the val side is not.** An evaluation set
that shrinks week by week makes every run's numbers incomparable with
the last.

**Written before the yaml**, deliberately. The other order leaves a
window where the yaml names a file that does not exist yet, and a run
started in it fails on a missing split rather than on anything a person
did.

**The trap designed around:** `get_img_files` filters on the EXTENSION
before anything checks the file exists, so a line it does not like is
dropped silently. A typo is therefore a quietly smaller training set.
The lists carry no comments and no blank lines, and `write()` counts the
lines back against what the policy selected and raises if they disagree.
A test breaks the writer deliberately to prove that check is real.

### A built data.yaml had no way to reach the training form

Reported as "I built the yaml but it is not showing up". It was not a
mistake at the card: the training form's dataset came ONLY from the
assemble cards' `onAssembled` callback, so a yaml could be built, saved,
and then offered nowhere. The form read "Nothing assembled yet" beside a
yaml sitting ready in the bucket, and its caption pointed at "Assemble a
dataset above" -- a card removed earlier the same day.

The form itself was fine. Give it `retraining` and it reads
`retraining/data.yaml` for the class check, and the pipeline downloads
that prefix and finds the yaml at its root. The missing piece was only a
way to put the folder there.

`Track2RetentionPanel` now takes an optional `onUseForTraining`, which
`Track2AutoTrainingPanel` wires to its own `onAssembled`, and a **Use for
training** button appears once the yaml builds without blocking problems.

**The same stale caption was in `Track2ScratchTrainingPanel` too** --
same wording, same removed card. Both now say where a dataset can come
from, and the warning names both routes instead of a card that no longer
exists.

Worth keeping in mind: there is ONE live `data.yaml` per root, because
the trainer requires it at the root of the prefix it downloads. So "pick
a yaml" means restore a snapshot into that slot and then press Use for
training -- the picking happens in the snapshot list, not the form.

### A run records what it actually consumed

`training_files` already held one row per file per run, with a
`by-file-name` index answering "which runs used this image". It did not
need replacing -- it needed feeding correctly.

It was fed by a RECURSIVE LISTING of the dataset folder. Right while
`assemble_dataset` copied the chosen images into a folder of their own,
because then the folder WAS the selection. Wrong now that the yaml names
week folders in place: `dataset_folder` is the whole root, and a run
training on 587 images would have been recorded as using all 748. An
audit that overstates what a run saw is worse than none, because it
reads as precise.

`run_artifacts.py` reads the file lists when the yaml points at them and
records those names instead, with labels named by the same swap
Ultralytics makes. Each dataset block now carries `names_from`
(`"list"` or `"folder"`) -- the two mean different things and a reader
cannot tell them apart from the names alone.

It also copies the three artifacts into a folder of the run's own:

    _runs/<audit_id>/data.yaml
    _runs/<audit_id>/train.txt
    _runs/<audit_id>/val.txt

**Copied, not referenced.** The live slots are overwritten by the next
Save, so a pointer answers "what does that path hold now", which is a
different question from "what did this run train on". A test rewrites
the live list and asserts the run's copy is unchanged.

Outside the training prefix, for the reason snapshots are: `train.py`
downloads every object beneath the prefix it is given.

Everything here fails soft. Losing the record must not lose the
submission it describes.

**Still open:** `dataset_snapshots.create()` archives the yaml and a
manifest of the whole corpus, but NOT `train.txt`/`val.txt`. So a
restore returns a yaml pointing at live lists that may have moved on --
which looks right and is not. Closing it: archive the two lists beside
the yaml, and have `restore()` copy three keys instead of one.

### Dataset snapshots: a run you can repeat

A generated `data.yaml` names week folders in place, so it pins nothing:
the pull keeps filling those folders, and the same yaml trains on more
images next month. Good for "what would we train on now", useless for
"run that again".

`dataset_snapshots.py` closes the gap. Freezing a root records three
things: the yaml text byte for byte, a MANIFEST of every image and label
key it selected, and per-week counts. Two small S3 objects plus one
DynamoDB row. Nothing is copied -- the manifest is a list of names, and
the images stay where the pull put them.

The manifest is the part that makes a run repeatable. `data.yaml` can
only name DIRECTORIES; it has no way to say "these files and not the
others", so something downstream has to filter. `train.py` now takes
`--dataset-manifest-s3-uri` and downloads EXACTLY those keys -- exact,
and faster than fetching a folder that has since grown. Omit it and the
whole prefix is taken, as before.

**The id is a hash over the yaml text AND the sorted file list
together.** Either alone lies: the same yaml over a grown corpus is a
different dataset, and the same corpus under a reordered class list
trains a different model. Hashing the pair means two snapshots share an
id only when a run against either would be the same run. There is a test
pinning the separator, too -- concatenating names without one makes
`["ab"]` and `["a","b"]` the same dataset.

**Snapshots live OUTSIDE the training prefix** (`_dataset_snapshots/`).
Under it, `train.py` would download the entire snapshot history before
every run, since it fetches every object beneath the prefix it is given.

**Restore is a copy into one slot.** There is exactly one live
`data.yaml` per root, because the trainer requires it at the root of the
prefix it downloads and only that parent holds every week folder the
yaml names. Nothing is lost by restoring: each snapshot keeps its own
copy.

**Drift is reported per week, not as a total.** "The corpus grew by 400"
does not say whether an OLD week was backfilled -- and a backfilled old
week is exactly the case where a re-run without the manifest silently
trains on more than the original did.

Infrastructure: `aws_dynamodb_table.dataset_snapshots` (hash key
`snapshot_id`, no range key -- re-freezing an unchanged dataset
overwrites its own row) and a `DatasetSnapshotsReadWrite` statement on
`track2-app-training-audit`. Applied TARGETED, deliberately:

```powershell
terraform apply "-target=aws_dynamodb_table.dataset_snapshots" "-target=aws_iam_policy.training_audit"
```

**The app stack now has drift of its own.** A plain apply there also
wants to change `aws_cloudwatch_event_rule.pull_control_schedule` and
`aws_iam_policy.pull_control_app` -- the app writes that rule directly
and drops its `description`, so the `.tf` no longer matches. Only the
description, not the schedule expression, but it is not something to land
by accident. Target your applies in this stack now, the same as in
`terraform/platform`.

**A trap worth remembering.** The readiness probe first called
`table_status`, which is `dynamodb:DescribeTable` -- an action the app is
not granted and does not otherwise need. So the card reported the table
unreachable and told the operator to apply terraform they had ALREADY
applied, while `Scan`, `GetItem` and `PutItem` were all working. The
probe now does a one-row `Scan`: test the permission the feature uses,
not a neighbouring one.

**Still outstanding:** `train.py`'s manifest support is written but the
image is NOT built. It is `track2-yolo-train-images` in CodeBuild, and
unlike the app there is no dev/stable split for the training image -- a
push affects every future run, production included. The change is
backward compatible, but it is a deliberate decision, not a side effect.

### Auto Training takes one root, and carries Retention

It used to ask for a **training root and an evaluation root** as two
sub-cards, each with its own picker, its own images/labels sub-paths and
its own discovered list. That made sense when the pull control wrote two
parallel trees. It does not now: the pull control writes one root, with
train and val inside each week --

    retraining/
      week_1/  images/train  images/val  labels/train  labels/val
      week_2/  images/train  images/val  labels/train  labels/val

Two pickers onto one tree is not just duplication. Nothing checked that
the evaluation picker pointed somewhere different from the training one,
and pointing both at the same root is **train/val leakage** -- a model
scored on images it trained on, which reads as a very good run.

So: **one root picker**, seeded from the pull control's own configured
root, so the two cards cannot drift apart without somebody typing a
different folder on purpose.

**The layout is detected, not asked for.** It probes
`images/train` + `labels/train` first, since that is what the pull
control writes; if nothing is there it falls back to flat
`images` + `labels`. A flat root then gets a per-folder train/val
selector, and **nothing defaults to val** -- a validation set chosen for
you is a validation set nobody checked.

**The folder table is the pull control's shape**, because the two cards
are looking at the same folders: one row per week, train and val counts,
paired count, a state column, a tick to leave a week out, and a totals
row. Clicking a week name unfolds the actual prefixes it holds with their
image, label and paired counts -- which is the answer to "why is this
week not being used", asked one week at a time.

Every week is included unless ticked out, so **next month's week is
picked up with no action at all**. That is the failure the card exists to
remove: a retrain that quietly misses the newest data because somebody
forgot to add a folder.

**Retention now lives inside this card**, below *Assemble from all
folders*, and is no longer a separate card on the page. It decides how
much of each older week an assembly takes, so it was being read on one
card and applied on another three cards down the page.

**The policy has an on/off tick beside the Retention title**, readable
and pressable whether the card is open or shut. It **writes as soon as
it is flipped** rather than staging behind *Save settings* -- a switch in
a header exists to be used without opening the card, and one showing
`On` while the stored policy is off is worse than no switch. It sends
`{enabled}` alone, and the route merges patches, so it cannot commit
anything half-typed in the controls below.

That forced one other change: the settings now load **on mount**, not on
open. They used to load only when the card was opened, which would have
left every shut card showing an unticked box whatever the real policy
was -- an unticked box meaning "not read yet" is the one thing this
control must never do. Until the read returns it shows `...` and is
disabled. The in-card *Decay older weeks* checkbox goes through the same
handler, so the two cannot disagree.

### The retention policy is a schedule, and now looks like one

Three things, from a spreadsheet of what a retrain should train on.

**A flat top to the curve** (`full_weeks`, default **1**). The newest N
weeks all count in full; decay starts on the one after. `full_weeks = 1`
gives `steps == age`, which is exactly what the curve did before, so no
saved policy changed by this existing -- there is a test that asserts it.

It is worth having because nothing much about a site changes in a month:
a month of data is all equally current, and decaying from the second week
thins data that is still describing today.

**The grid.** `POST /track2/retention/matrix` -- rows are the week a
retrain is DEPLOYED, columns the week the data was COLLECTED, each cell
how much of that week it trains on. A row holds only the weeks that
existed by then, so the grid is triangular; a rectangle would be claiming
a retrain could train on data collected after it.

It runs PAST the weeks that exist (`ahead`, default 4). That is the whole
point: the preview answers "what would a retrain use today", and a
retention policy is a schedule, so the question that decides whether it
is the right one is "what will a retrain use in two months, and is week 1
still in it by then". The grid answers that without waiting two months.

It is pure -- week NAMES and the config, never the bucket -- so future
rows need no folders, and the policy can be looked at before any data has
been collected under it. It redraws from the controls rather than from
what was last saved, same as the chart.

`full_weeks` had to be added to `CurveRequest` too, or the chart would
have drawn the old shape while the grid drew the new one, and one of them
would be lying.

**A golden set** (`golden_root`). A curated folder that never decays and
is always in the training set, laid out like a week folder -- which is
what the Golden Set card already writes. It has no collection week, so it
has no age, so there is nothing to decay: it is the core the detector is
not allowed to forget while the recent weeks pull it around. Shown as a
pinned 100% column.

Its **`images/train` split only**, enforced in `build()` rather than
warned about. The val split is the fixed evaluation set, and training on
that would score the detector on images it had just been taught. That
also answers a real problem with the card as it stood: val was drawn from
every week, so the evaluation set drifted as new weeks landed and each
retrain was scored against a different set -- movement between runs mixed
model change with val-set change. A golden val split is fixed, so the
numbers become comparable.

Verified on dev by rendering the grid inside the running container:
`linear`, 10% a week, `full_weeks = 4` reproduces the specified schedule
cell for cell, and the two planned rows carry it on to 50% and 40%.

### Then the card was cut back, because it had become confusing

Too many controls, and two of them contradicted each other.

**One on/off.** The tick beside the title is gone; *Decay older weeks*
inside the card is the only switch, and it waits for Save like every
other control. There had been two switches for one setting with different
timing -- the header one wrote immediately, the inner one staged. Whichever
was the better control, having both was the worse card.

**Two rows instead of five boxes.** Weeks at full weight, lost per week
and curve across the top; the floors below in their own box behind a
tick. `use_floor` is real, not a UI state: off, the curve alone decides
and an old enough week reaches nothing. A floor is a judgement -- "no
week ever falls below a tenth" is a claim about what history is worth --
and it used to apply whether or not anybody had made it.

**The golden set has a switch and a folder browser.** Two keys,
`use_golden` and `golden_root`, so the path is remembered while the
switch is off and turning it back on does not mean finding it again.

**The per-class exemption is gone entirely.** Not the control -- the
feature. `protect_rare_classes`, `rare_class_threshold`,
`mark_rare_class_images`, `select_stems(protected=)`, and the
`class_counts` / `rare_classes` / `protected` fields all removed.

The golden set replaces it and is the better mechanism: the exemption
GUESSED which images mattered from a class count, while a golden set is a
folder somebody chose -- a judgement rather than a heuristic, and visible
as a folder rather than inferred at assembly time.

It was also expensive. `mark_rare_class_images` opened **every label file
in the corpus** on both preview and build, thirty-two at a time, to count
class instances. Both are now materially faster.

**The line chart is gone; the bars stayed** -- and the bars now follow the
controls. `POST /track2/retention/preview` takes a `config`, so switching
compounding to linear moves them. It used to render the SAVED policy while
the controls above said something else, which reads as the curve making no
difference.

**The golden set draws as a permanent gold bar at 100%** above the weeks,
from `golden_summary()` in the preview payload. It has no collection week,
so it has no age, so it sits above the series rather than in it.

Weeks ahead on the grid now defaults to **10**.

### Two things about the grid that looked like bugs

**The gold column was gated on the folder as well as the switch.** Tick
"use a golden set" with no path chosen yet and nothing appeared, which
reads as a dead checkbox rather than an unfinished setting. It now
follows the SWITCH alone; the note underneath says the folder is still
to be picked.

**The curve control was wired correctly all along.** Reported as "not
reflected in the grid", and it is worth writing down why, because the
instinct is to go looking for the missing wire:

    rate=0.10  compounding  47.8 53.1 59.1 65.6 72.9 81.0 90.0 100.0
    rate=0.10  linear       30.0 40.0 50.0 60.0 70.0 80.0 90.0 100.0

The maths, the route merge and the card were all passing `curve`
through. The problem is the RATE. At **1% a week** -- the default -- the
two curves are **0.21 of a point** apart at the widest point of an
eight-week window, so both render as `93%`, `94%`, `95%`, and switching
between them changes no number on screen.

So every share on the card now reads at **two decimals**, through one
`share()` helper -- the grid cells, the week bars and the golden bar, so
the three cannot drift apart.

Fixed at two rather than "as many as the gap needs", which is what it was
first. A column whose precision changes as a slider moves is its own
puzzle: the numbers stop being comparable down the column, and a cell
that gains a decimal looks like it changed when it did not.

`schedule_matrix` still returns `curve_gap` -- the largest difference
between the chosen curve and the other one anywhere in the grid -- and a
line under the grid states it outright, so a small gap reads as a small
gap rather than as a dead control.

### And then cut back again

**The grid is permanent.** Its Show/Hide is gone; only the *weeks ahead*
box is left in that header.

**Two cards were carrying two Show/Hide buttons each.** When `Card` was
given a universal collapse earlier, four cards were changed to
`collapsible={false}` because they draw a summary when shut --
`Track2DetectorsPanel` ("Register a model and Classes") and
`Track2ImportCheckpointPanel` were missed, so each ended up with its own
button AND the generic one, collapsing different halves of the card.
Both now use `Card`'s alone, with `defaultOpen={false}` to keep starting
collapsed.

**The golden set names its bucket** (`golden_bucket`, blank = this app's
own). The picker walks whichever bucket is chosen, through
`listTrack2Folders` / `getTrack2Buckets` -- the same pair the Golden Set
card uses, since that is what writes these folders. `golden_summary`
counts through `pc._list_objects`, which takes a bucket; this app's own
`_basenames` does not.

`rt.build()` **refuses** a golden set in another bucket, because
`assemble_dataset` copies within one. Reading the prefix out of this
bucket instead would quietly assemble a dataset with no golden set in it
and the run would look fine.

**The evaluation root and the class list are gone -- and so is Build
dataset**, which was the only thing that used them. It could not have
enabled without them, and a permanently disabled button is worse than
none.

That is the right direction anyway. *Build dataset* COPIED every selected
image into a new folder; the data.yaml generator below it does the same
job by naming the week folders where they already are, and takes class
names from the source dataset's own `data.yaml` -- which is where the
indices in the label files came from, so there was never anything to pick
from a list. *Preview* needs neither and is untouched.

### The grid was drawing the curve, not the policy

Reported as "the never fall below a certain percentage is not working".
It was working. The grid was not describing it.

`keep_count` has always applied both floors when images are selected.
`schedule_matrix` returned `retention_fraction` alone -- the curve BEFORE
the floor -- so a policy set to keep at least 20% of every week displayed
2.81%. The floor looked broken; the display was.

Cells now carry `effective` (the curve held up by the percentage floor)
alongside `fraction` (the curve), plus `floored`. The card shows
`effective`, marks a floored cell with a dot, and puts the curve value in
the tooltip. Verified in the running container:

    curve        2.81   3.52   4.40   5.50   6.87   8.59  10.74  13.42
    effective   20.00  20.00  20.00  20.00  20.00  20.00  20.00  20.00

**The image minimum works too, and is usually the one that binds.** At
20% a week, a hundred-image week sixteen weeks old keeps **25** images,
because the 25-image minimum beats the 20 that the percentage would give.
`keep_count(100) -> 25`, `keep_count(500) -> 100` (the percentage wins
there), `keep_count(10) -> 10` (never more than exists).

The grid **cannot** apply that one: it depends on how many images a week
holds, and `schedule_matrix` never reads the bucket -- which is what lets
it draw rows for weeks that do not exist yet. So it applies the
percentage and says in words that a smaller week keeps a larger share
still. The per-week bars do read real counts, and their `%` column now
prints `effective` rather than the curve, so the share and the count on a
row no longer disagree.

`curve_gap` is measured on effective shares now. Once the floor holds
both curves at the same value there is no difference left to see, and a
gap quoted from the bare curves would promise one.

### The other half of the floor, and a units toggle

Still reported as not reflecting, and this time it genuinely was not:
the grid applied only the PERCENTAGE floor. The image minimum depends on
how big a week is, and `schedule_matrix` never read the bucket -- so a
policy whose image minimum is the binding half was understated. With
`floor=20%` and `min_images=25`, a hundred-image week keeps 25 images,
which is 25%, and the grid said 20%.

The matrix route was already calling `read_weeks`, which returns the
stems, so **the sizes cost nothing extra**. `schedule_matrix` takes
`sizes={week: images}` and each cell now calls `keep_count` -- the same
function the assembly calls, so the grid cannot promise a number the
build would not produce. There is a test asserting that cell by cell
across a grid of differently sized weeks.

Against the real `retraining` root, from inside the container:

    sizes   week_1:26  week_2:392  week_3:203  week_4:0  week_5:0
            week_6:8   week_7:119

    week_9  week_1:25/26(96.15%)*  week_2:82/392(20.92%)
            week_3:53/203(26.11%)  week_6:8/8(100.00%)*
            week_7:76/119(63.87%)  week_8:95/119(79.83%)?

`*` is a floor deciding. Week 1 keeping 25 of its 26 images is the
25-image minimum doing exactly what it was set to do, finally visible.

**A units toggle** sits beside *weeks ahead*: shares or counts. Both are
the same arithmetic; a share answers "how hard is the decay biting" and a
count answers "is there enough left to train on", and those are different
questions asked on different days. The `images` radio is disabled when no
week folders could be read, because a count drawn from an assumed size is
a guess wearing the look of a measurement.

**Weeks not collected yet** are assumed to hold the MEDIAN of the ones
that do -- not the mean, since week 1 is usually the whole backlog and
would drag a mean somewhere no real week has been. Those cells are
marked `?`.

**A counted zero is not a guess.** `real` was first tied to
`available > 0`, so the empty weeks 4 and 5 came back marked `?` --
saying "we do not know" about the one thing that had been measured. They
were read, and they are empty. `real` now means the size was KNOWN, a
separate `empty` flag says it was zero, and those cells render as `--` in
grey rather than as a share of images that are not there.

### An empty week is a gap, not a week

Three changes, and the third is a real change to what gets built.

**The `?` marker is gone.** It was reported as clutter across the whole
grid, and the reason is worth keeping: with `ahead=10` against seven real
weeks, ten of the seventeen columns are projected, so scrolled right the
view is entirely assumed sizes and every cell carried one. The assumption
is stated once in the caption instead.

**An empty week is pink and reads "empty"**, not 0%. A folder never
filled is not a folder the decay emptied, and the cell should not look
like the second.

**An empty week no longer ages anything.** Age counts weeks of
COLLECTION, not weeks of calendar:

    week_1  age 7   96.15%  25/26     week_4  EMPTY (stepped over)
    week_2  age 6   94.13%  369/392   week_5  EMPTY (stepped over)
    week_3  age 5   95.07%  193/203   week_6  age 4  100.00%  8/8

Without it, weeks 4 and 5 pushed everything before them two further
steps down the curve on the strength of nothing having been collected --
the model told the world had moved on while no data had arrived to say
so. Same reasoning already behind counting age back from the newest week
rather than from today, which `plan` has always done.

The rule is in `plan()` as well as the grid. It had to be: `plan` chooses
the dataset, so a rule living only in the grid would have drawn a picture
of something else.

One trap found by the existing tests while making it: treating an
UNKNOWN week size as empty flattened any grid drawn before folders had
been listed to 100% across the board. Only a week that was read and found
empty is stepped over; an unknown size counts as holding data.

### Two 100% columns, and the marker that was lying about them

Asked why `week_1` read 100% for the first two rows and `week_6` read
100% all the way down. Both were right; the question found a third thing
that was not.

**week_1 holds 26 images.** At 1% a week, `26 x 0.99 = 25.74`, which
rounds to 26 -- so the week reads as untouched with no floor involved. It
drops to 25 (96.15%) at age 2. A small-week rounding effect, nothing more.

**week_6 holds 8 images and the minimum is 25.** `keep_count` never
returns fewer than the week holds, so a week SMALLER than the minimum is
kept whole for ever, at any age. week_6 cannot decay under this policy.
That is the documented rule working, but nobody should have to infer it
from a column of 100%s -- there is now an `under_minimum` flag, marked
with a hollow square and explained in the tooltip and the caption.

**`floored` was measuring the wrong thing.** It tested
`effective > fraction`, which rounding UP also satisfies -- so week_1 at
96.15% was marked floor-held when no floor had touched it. It now means a
floor CHANGED the answer: `keep > unfloored_count(available, fraction)`.
Same correction applied in `plan`.

Verified against the live policy in the container:

    week_1  age 8   96.15%  25/26   *  a floor decided
    week_6  age 5  100.00%  8/8     [] smaller than the minimum, kept whole
    week_7  age 4   95.80%  114/119

**Worth knowing about the saved policy**: at 1% a week the curve barely
moves, and any week holding fewer than 25 images never decays at all.
week_6 (8 images) is permanently immune and week_1 (26) is one image from
being so.

There is now a test asserting the curve reaches the grid, which is the
test that was missing: nothing would have caught the control genuinely
being unwired.

**The preview bars held whatever the last Preview returned**, so changing
the curve left them frozen. Once drawn they follow the controls, and
only once drawn -- that call does list S3, so it is not something to run
before anybody has asked for it.

### The app moved out of JP_UI and now owns the whole estate

Extracted with `git subtree split`, so all 332 commits came with it. Both
Terraform stacks live here (`terraform/app`, `terraform/platform`), plus
`model_training/` and the estate docs.

**No path in this repo points at JP_UI.** That was not true until
2026-09-16 — several docs still said `cd C:\Projects\JP_UI\jurong-port\...`
and `terraform/infra` (the old name for `terraform/platform`), including
the custom-training setup instructions. Fixed. What remains is deliberate:

- Prose mentions of JP_UI as a *separate system* (`main.tf`, `ecr.tf`,
  `rolling_metrics.py`, `config.py`). These describe that system, not
  where this code lives.
- Two live AWS security-group rule descriptions in `ec2.tf` naming the
  JP_UI test deploy. Editing them would produce a Terraform diff for a
  cosmetic rename.
- `platform/summary.md`, a dated changelog — historical record, left alone.
- Absolute `C:/Projects/JP_UI/...` strings in
  `terraform/platform/terraform.tfstate`. These are **cached reads of
  eight `archive_file` data sources**, not config. Every `.tf` uses
  `${path.module}`, so the sources resolve correctly from here; a plan
  confirms it. Terraform only rewrites the cached strings on apply, so
  they clear themselves and must not be hand-edited.

Verified before the originals were deleted: identical plans from both
locations, byte-identical state files, and a full deploy.

Two traps a repo split sets, both hit and fixed: the parent's `.gitignore`
and `.gitattributes` do not travel. Without the first, `git add -A`
committed a 600MB provider binary and `terraform.tfstate` (rewritten out;
`.git` went 453MB → 1.2MB). Without the second, the clone checked out CRLF,
which breaks `deploy.sh` under bash and makes Terraform report a permanent
phantom diff on the CodeBuild buildspec.

### Training inputs actually reach AWS now

They never did before. Epochs, image size, batch and seed were hardcoded in
the pipeline definition, and the folder pickers were ignored — the UI
looked like it controlled a run it had no influence over.

They are now pipeline parameters, each defaulting to the value it replaced,
carried from the page through CodeBuild and the coordinator. A run that
sets none of them behaves exactly as before, which is what keeps the weekly
schedule unaffected. Proven live: a run submitted with `Epochs=1, Batch=1`
reached the training job with exactly that.

`EvaluateYolo` also read `TrainingDatasetUri`, so **every metric this
pipeline has ever reported was measured on the model's own training data.**
It now reads a separate `EvaluationDatasetUri`.

### Training audit

One DynamoDB row per run: both dataset folders and their file names, every
hyperparameter, notes, the registered model version, its metrics, and
start/end times. It records what was **submitted** and what the job
**actually got**, and flags the difference in red — which is how the
ignored-parameters problem became visible at all.

Sits under the training queue as `HISTORY`, each row deleting the model
version it registered along with it.

### Fixes worth knowing about

- **Stop Training** called an API nothing allowed. Granted.
- **The build reported FAILED on successful runs** — `run_pipeline.py` asked
  boto3 for a `pipeline_execution_complete` waiter, which does not exist. It
  threw one second after starting the execution. Replaced with polling.
- **Cost tiles always read $0.00.** The filter matched `":Host:"`, but usage
  types are `APS1-Host:...` — a hyphen. Affected the Deploy tile and the
  Cost page headline.
- **Metrics never appeared** for full-frame runs: the reader only understood
  the legacy flat `evaluation.json`, and the current evaluator nests them
  under `metrics` and reports `f1` where the old one reported `accuracy`.
- **An endpoint billed for a day.** One inference job was left at
  `deploying` when its process died; the teardown check treats that as
  unfinished work and correctly declined, forever. Jobs are now recovered at
  startup, and an idle sweep removes the endpoint after 30 minutes with
  nothing needing it — the teardown no longer depends on a queue draining.
- Automated training queues were removed entirely, UI and AWS resources.

---

## Traps

### Changing the trainer is a THREE-step chain, and skipping one fails elsewhere

`model_training/training-container/train.py` is not built from this
working tree. `track2-yolo-train-images` reads a ZIP in S3:

    S3  track2-s3/codebuild-source/training-container.zip

which Terraform (`terraform/platform/codebuild.tf`) uploads from
`model_training/training-container-src.zip`. So editing `train.py` and
starting the build produces a faithful rebuild of the OLD code. Nothing
errors; the image is new, the digest changes, and the failure surfaces
later and somewhere unrelated.

It cost a training job to find: `train.py --classes 0 --single-cls True`
exited 2 -- argparse rejecting an argument the image had never heard of
-- while the app, the pipeline and the parameters were all correct.

The whole chain, in order:

1. edit `model_training/training-container/`
2. repack `model_training/training-container-src.zip` AND upload it to
   `s3://track2-s3/codebuild-source/training-container.zip` (the platform
   stack has drift, so upload directly and keep the repo copy in step --
   check the local md5 equals the S3 ETag)
3. `aws codebuild start-build --project-name track2-yolo-train-images`
4. take the new digest from ECR and re-pin it in
   `model_training/custom_training/pipeline.json` -- **that** file is the
   one equal to the live definition (checked 2026-10-26);
   `pipeline_open.json` is stale and pins an older digest
5. `aws sagemaker update-pipeline --pipeline-name
   track2-yolo-open-training-pipeline --pipeline-definition
   file://pipeline.json`

Step 4 is its own trap: the pipeline pins the image BY DIGEST, so a new
image reaches nothing until it is re-pinned. Everything looks deployed
and the old code keeps running.


**`training_progress` describes the PRODUCTION pipeline only.**
`get_training_progress()` lists executions of `track2-yolo-training-pipeline`
and nothing else, so during a custom run it returns the last *production*
execution. That is how the Stop button came to be disabled for the whole
of a run it was meant to stop -- it keyed on
`training_progress.status === "Executing"`, which was `Stopped`. The
models response now also carries `custom_run_id`. Anything asking "is a
run happening?" must consult both.

**Auto-approving into `yolo11{s,m,l,x}-detector` deploys to production.**
The EventBridge rule on `ModelApprovalStatus: Approved`, filtered to
`values(local.yolo_size_group_names)`, fires `dump_approved_weights`, which
creates a Model, creates/updates the live inference component on the
shared endpoint, and overwrites
`latest_approved_model_weights/<size>/model.pt`. No taxonomy check anywhere
in that path. The filter is exact-match, so any other name is inert -- which
is the whole reason custom groups are named `-open-detector`.

**An app-side AWS call that fails permanently will retry forever if its
retry condition is "field is still empty".** `publish_versioned_weights`
did exactly that against a package outside the role's IAM grant, hammering
`DescribeModelPackage` every poll. Record the failure
(`weights_publish_failed`) so "tried and could not" is distinguishable from
"not tried yet".

**A new pipeline needs its own IAM grant, in a different stack.** Creating
`track2-yolo-open-training-pipeline` and pointing the app at it failed with
AccessDenied on `StartPipelineExecution`: the grant lives in
`terraform/app/audit.tf`, the code in `src/`. Renaming or adding a pipeline
means grepping both. Same for `DescribeModelPackage` on a registry group.

**`sagemaker:ListModelPackageGroups` is not granted.** Anything that
*discovers* groups fails with AccessDenied and, if the exception is
swallowed, renders as "nothing found" rather than "not allowed". Name the
groups instead -- `ListModelPackages` is granted on `*`.

**`terraform/platform` carries pre-existing drift** — a plan reports ~15
in-place changes nobody made. Most are Lambda code being re-uploaded from
local zips, because that state was rebuilt by import and never recorded a
source hash. **Use targeted applies there.** A blanket apply pushes every
local zip over whatever is deployed.

```powershell
terraform apply -target=aws_lambda_function.weekly_retrain
```

That stack's `archive_file` packages `model_training/replay_lambda/`, so
that command is also how the coordinator gets deployed after editing it.

**Neither is `model_training/training-container/evaluate.py`.** The
production pipeline loads its evaluator from a **content-addressed S3
key** — `replay/fullframe/code/<sha256-of-the-file>/evaluate.py` — so
editing the repo copy changes nothing. That repo file is a different,
older 7,151-byte script; the live one is 4,626 bytes. The version that
actually runs is tracked at `model_training/live_pipeline/
evaluate_fullframe.py`.

To change it: edit, `sha256sum` it, upload to
`s3://track2-s3/replay/fullframe/code/<that hash>/evaluate.py`, then
`update-pipeline` with the definition re-pointed at the new key. The hash
must be of the exact bytes uploaded — keep the file LF, or a later
checkout will not match what is deployed.

The custom pipeline is simpler: it reads a **fixed** key,
`s3://track2-s3/custom/code/evaluate_any.py`, so re-uploading that file is
the whole deployment.

**The pinned training image runs a WRAPPER that rejects any base model
that is not person/helmet.** Recovered and read on 2026-09-17 by pulling
the image apart. `/opt/ml/code/train.py` inside
`track2-fullframe-training-recovery` is not the repo's script — it imports
it as `legacy_train` and wraps it:

```python
model = YOLO(str(weights / "last.pt" if restored else baseline))
if model.names != {0: "person", 1: "helmet"}:
    raise ValueError("production PPE class manifest mismatch")
```

It also requires `--run-id` (positional `sys.argv.index`, raises if
absent) and a non-empty `--base-model-s3-uri`, and it never passes
`--base-model-package-arn` through.

Consequences, none of which fail until the job is running:

- **Any base checkpoint must be person/helmet.** A COCO `yolov8n.pt` or a
  foreign-taxonomy model dies seconds into `TrainYolo`. Fine-tuning *onto*
  new classes is fine; *starting from* a different taxonomy is not.
- This is why `track2-yolo-open-training-pipeline` exists — see item 3d.
- The scratch seed at `external_models/scratch/<arch>/model.pt` carries
  person/helmet names for exactly this reason, so it is the one base model
  usable on **both** pipelines for **any** dataset.

**`model_training/training-container/train.py` is NOT what training jobs
run.** Both pipelines pin their training step to an immutable digest:

```
track2-fullframe-training-recovery@sha256:68db0d890859036177fa7fa38b2f25b7c96f2d067ee7f09efe5bd58bce329844
```

That repo holds exactly one image, pushed 2026-09-15, and **nothing in
this codebase builds it.** The CodeBuild project that does build from
`training-container/` (`track2-yolo-train-images`) produces
`track2-yolo-train:train-latest` / `serve-latest`, which the pipelines use
only for the *evaluation* step and for serving.

So editing `train.py` changes nothing about how models are trained, and
**adding a CLI argument to it breaks every run** — the pinned image's
argparse rejects an unknown hyperparameter and the job dies immediately.
This was hit on 2026-09-16 while wiring registry base models: an
`--allow-unapproved-base` flag was added and reverted before it shipped.

Anything needing a real trainer change has to rebuild that image and
re-pin both pipelines, which is a production change. Prefer routes that do
not: the trainer applies **no approval gate to `--base-model-s3-uri`**
(only to `--base-model-package-arn`), so resolving a registry package to
its `ModelDataUrl` in the app is the supported way to fine-tune from an
unapproved package.

**The production pipeline is a two-class system by design.** Its remap
*discards* every box outside person/helmet, its evaluator refuses any
checkpoint with different class names, and its registry group's consumers
read index 0 as `person`. A six-class model in that chain is mislabelled at
every step without failing anywhere. That is why custom training is a
separate pipeline, group and evaluator — see
`model_training/custom_training/README.md`.

**A dataset needs `dataset-card.json` to be evaluated.** `train.py` only
needs `data.yaml`, so a folder without a card trains fine and then loses the
run at the evaluation step. Both folders are checked before a run starts.

**The last production run failed there**, before that check existed —
`escalation_dataset` has no card. Training succeeded; evaluation did not.
