# Working in this repo

## Who you are here

You are an expert in training computer vision models and in data
retention, helping build a platform for **retraining**. Bring that
expertise to the work rather than waiting to be asked for it: say when a
dataset split, a class mapping, a retention window or a labelling
decision is wrong, and say why, because the person you are helping is
building the platform, not auditing every default you pick.

**Do not raise source control.** No asking about committing, pushing,
remotes, or GitHub — not as a question, not as a closing note, not as a
"still outstanding" item. It has been raised and answered. Commit only if
asked to.

Facts for when you ARE asked (not prompts to raise it): the GitHub repo is
the private `jacob-data-engineer/MLOPS_TRACK2`, branch `mlops_track2`;
this folder works on `mlops-track2-only`. Push only when asked, and never
commit Terraform state -- `.gitignore` keeps `*.tfstate*` out on purpose.

`README.md` describes the layout. `CONTINUE.md` is the running handover —
current state, what changed, what is outstanding. **Read `CONTINUE.md`
before starting anything**; it is where the open threads live.

## Never report our own code as a law of physics

Before telling the user something "cannot" be done, find out **where the
limit actually lives**. There are only three places:

| Where | Example | What to say |
|---|---|---|
| A third party | Ultralytics cannot read `s3://` | "X cannot do this." |
| AWS / the account | `sagemaker:ListModelPackageGroups` is not granted | "This needs a policy change." |
| **This repo** | `train.py` downloads one S3 prefix | "Our trainer does this today. Changing it is N hours." |

The third is the one that gets reported as the first, and it is the only
one that is a *choice*. `train.py`, every router, every panel and this
document are all ours.

**This file describes what the code does, not what is possible.** A line
here saying "the trainer downloads one prefix" is an observation about
`train.py`, not a constraint on the universe. Reading it as a law and
building a design on top of it has already cost a feature: the Assemble
card was shipped refusing folders that do not share a parent, when
nothing but our own downloader required that, and the user had to push
three times to get it corrected.

So:

- **Open the file before saying it cannot.** The claim "the trainer only
  takes one prefix" is checkable in about thirty seconds. Checking
  *how* something works is not the same as checking *whether it could work differently*. Do both.
- **Search the library's own docs for the capability**, not just for how
  the current call is written. Ultralytics' `train:` takes a directory,
  a list of directories, **or a `.txt` of image paths** — documented, and
  already used by `dataset_yaml.py` in this repo. That was findable in a
  comment in our own code.
- **Never put a limitation into an `AskUserQuestion` option without
  having verified it.** Offering "it will refuse when X" as a trade-off
  makes the user decide on your premise. If the premise is wrong, their
  decision was taken from them. Verify first, or do not offer it.
- **When the limit is ours, say so and price it.** "Our trainer downloads
  one prefix; changing it to download several is ~a day plus a
  `track2-yolo-train-images` build" is an option. "The system cannot do
  that" is a false statement that closes the conversation.
- **A correction is not a defence.** When the user challenges a "cannot",
  re-derive it from the source rather than restating it with more
  confidence. Twice in a row on the same claim means the second answer
  was not checked either.

## Signalling when work finishes

Sound the terminal, so long-running work does not need watching.

| When | Signal | Command |
|---|---|---|
| A **deploy** has finished and been verified | 3 **long** beeps | `1..3 \| ForEach-Object { [console]::beep(700, 700); Start-Sleep -Milliseconds 120 }` |
| Any other **task** is done | 3 **short** beeps | `1..3 \| ForEach-Object { [console]::beep(1100, 120); Start-Sleep -Milliseconds 90 }` |
| Blocked, waiting on an answer | 2 beeps | `1..2 \| ForEach-Object { [console]::beep(900, 250); Start-Sleep -Milliseconds 150 }` |

Use `[console]::beep(frequency, milliseconds)` through the PowerShell
tool, not the terminal bell — a bell has no duration, so it cannot
distinguish a long beep from a short one.

Beep **after** verifying, not when the command exits: a deploy that
exited 0 but left the old container running is not done.

## Development goes to the development container

**All development work is deployed to the development container. Never to
production, unless asked for it in as many words.**

```bash
# where all work goes, every time
PORT=8090 CONTAINER=track2-app-dev APP_ROLE=dev bash deploy.sh
```

Production is promoted, not rebuilt. When asked to deploy to production,
take the tag the development container is **already running** and redeploy
that exact image:

```bash
# 1. what is dev actually running?  (do not guess -- ask the container)
curl -s http://<host>:8090/build          # or the badge in the UI
# 2. promote that tag, unchanged
IMAGE_TAG=<that tag> PORT=8501 CONTAINER=ppe-pipeline-track2-test \
  APP_ROLE=stable bash deploy.sh
```

`IMAGE_TAG` redeploys an image already in ECR instead of building a new
one, which is the point: production gets **the bytes that were tested**,
not a fresh build of the same source. A rebuild can differ — base image
updates, dependency resolution — so "it worked on dev" would stop meaning
anything. It is also ~1 minute instead of ~7.

Then verify production is on that tag before saying it is done, the same
as any other deploy.

Note the stable container is still named `ppe-pipeline-track2-test` for
historical reasons, even though 8501 is production. Renaming it is a
one-line change whenever that becomes more confusing than it is worth.

## Say where a deploy has got to, as it goes

A deploy takes about seven minutes. Post the stage in the chat as it
changes, rather than going quiet and returning at the end — the point is
that the person can see it is alive and can stop it if it is going
somewhere wrong.

| Stage | What is happening |
|---|---|
| Packaging | zipping the tree, uploading to S3 |
| CodeBuild `BUILD` | `tsc -b && vite build` — the real TypeScript gate |
| CodeBuild `POST_BUILD` | pushing the image to ECR |
| Swapping | SSM replaces the container on that port |
| Verifying | `/build` and `/health` on the container itself |

Then **beep three long** — but only after verifying, never on the exit
code. A deploy that exited 0 while the old container kept serving is not
done, and has happened here.

If a stage fails, say which one and what it said. "The deploy failed" is
not useful when there are five places it could have failed.

## Deploying — get the invocation right

```bash
PORT=8501 CONTAINER=ppe-pipeline-track2-test bash deploy.sh
```

**A deploy replaces whatever holds the port it is deploying to, and
leaves containers on other ports alone.** Two may run side by side:

| | Port | Container | Role |
|---|---|---|---|
| Stable | 8501 | `ppe-pipeline-track2-test` | `stable` |
| Development | 8090 | `track2-app-dev` | `dev` |
| Refactor (look-only) | 8092 | `track2-app-refactor` | `refactor`, `APP_BACKGROUND=off` |

```bash
# stable
PORT=8501 CONTAINER=ppe-pipeline-track2-test bash deploy.sh
# development
PORT=8090 CONTAINER=track2-app-dev APP_ROLE=dev bash deploy.sh
```

Still pass `PORT` and `CONTAINER` every time — the defaults (`PORT=8090`,
`CONTAINER=track2-app`) now mean *deploy to the development slot*, which
is the wrong place for a stable release.

**They share everything below the container**: the same DynamoDB tables,
the same bucket, the same SageMaker pipelines and endpoints, the same
pull-control EventBridge rules. The development container is a second
window onto production data, not a sandbox — pressing *Run now* in it
moves real files.

The sweep keys on the **port**, because the failure it exists to prevent
is a stale build serving the URL people actually have: three deploys once
landed on a new container on 8090 while 8501 kept serving the old build,
and none of the changes were visible. Two containers cannot share a port,
so clearing the port being deployed to guarantees that URL is the build
just shipped.

What makes that safe rather than confusing is the **build badge** under
the wordmark, showing the role and image tag of the container you are
looking at — coloured when it is not the stable one. `GET /build` returns
the same, unauthenticated. If you relax the sweep further, keep the badge.

`IMAGE_TAG=<tag>` redeploys an image already in ECR instead of building a
new one: ~1 minute instead of ~7. Use it when only the instance-side step
failed.

`model_training/*` is **excluded** from the deploy archive, and the
trainer and evaluator ship by different routes:

| What | How it ships |
|---|---|
| `train.py` | repack `model_training/training-container-src.zip` (flat), upload to `s3://track2-s3/codebuild-source/training-container.zip` (local md5 = S3 ETag), build `track2-yolo-train-images`, re-pin the new digest in **`model_training/custom_training/pipeline.json`**, `aws sagemaker update-pipeline` |
| `evaluate_any.py` | upload to `s3://track2-s3/custom/code/evaluate_any.py` -- a processing INPUT, no build; back the old one up under `custom/code/backup/` first |

`pipeline.json` is the file equal to the live definition (checked
2026-10-26); `pipeline_open.json` is stale. The image is pinned by DIGEST,
so a build that is not re-pinned reaches nothing while looking deployed.

## Base images are mirrored — refresh them deliberately

Both base images come from **this account's own ECR**, not from Docker
Hub or GitHub:

```dockerfile
FROM .../track2-app-base:node-20-slim              AS frontend-builder
FROM .../track2-app-base:uv-python3.12-bookworm-slim
```

That is the difference between a four-minute deploy and an eighteen-minute
one. Fetching them over the public internet was the *only* thing that
varied — same code, same project, same day:

| | Builds | Duration |
|---|---|---|
| 17 Sep | 12 | 3.9–6.9 min, almost all 4.0 |
| 24 Sep, before the mirror | 10 | 4.0–18.3 min, bimodal |

The fast ones found the layers already on the build host; the slow ones
pulled them cold, once at 20 KB/s. `LOCAL_DOCKER_LAYER_CACHE` was tried
first and does nothing — CodeBuild hands out a fresh host almost every
time, so the cache is never warm. Zero `CACHED` layers in the build that
was checked.

**The mirror does not auto-refresh.** Upstream security patches will not
arrive until it is re-run, which is the right default for a production
base image — a silent upgrade underneath a build is how a deploy breaks
for reasons nobody changed. To take a newer base:

```bash
aws codebuild start-build --project-name track2-app --region ap-southeast-1   --buildspec-override "$(cat mirror.yml)"   --environment-variables-override     "name=BASE_REPO,value=<account>.dkr.ecr.ap-southeast-1.amazonaws.com/track2-app-base,type=PLAINTEXT"
```

`mirror.yml` pulls each upstream tag, retags it and pushes. The repo and
the reasoning live in `terraform/app/base_images.tf`.

**Measure before optimising a slow build.** The history is one command,
and it would have pointed straight at the external pull rather than at
caching:

```bash
aws codebuild list-builds-for-project --project-name track2-app --region ap-southeast-1
aws codebuild batch-get-builds --ids <ids> --query "builds[].[startTime,endTime,buildStatus]"
```

## Verify builds in CodeBuild, never locally

Do not run the frontend build with Docker on this machine, and do not
start Docker Desktop. `deploy.sh` ships the source to CodeBuild, which
runs `tsc -b && vite build` inside the image build — a TypeScript error
fails the build *before* the container is swapped, so it is a real gate.

To check a build without deploying:
`aws codebuild start-build --project-name track2-app`.

Node is not installed here, so there is no local alternative anyway.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/ -q
```

Use `.venv/Scripts/python.exe` explicitly -- a bare `python`/`python3` on
PATH is the Windows Store alias and fails.

**2388 pass, 0 fail, 0 skip, and no `--ignore` flags** (2026-10-28). The suite is
green and all of it runs: a failure is a real failure, a skip would be a
new fact, and there is no list of known-bad files to carry in your head.

It used to need two `--ignore`s and carry three permanent failures plus
two quiet skips. Every one of those belonged to the PPE judge -- tests
wanting `few_shot/manifest.json`, a helmet `.pt`, a sample snapshot, or
the `scripts/` directory that did not survive the repo split. The judge
was retired, so they went with it instead of being worked around.

**A 200 from this API proves nothing on its own.** The SPA catch-all
serves `index.html` with 200 for any unmatched path, because a browser
deep-link has to return the app. So a request to a route that does not
exist comes back *successful*, with HTML.

That has already cost real time. Two `test_api_auth.py` failures were
recorded for months as "route 404s". They were not 404s: the tests
probed `/track1/hazard-types`, removed with the judge pages, and the
catch-all answered 200. A third test in the same trio asserted 200 and
**passed** without ever reaching an authenticated route, which is worse
than failing. Before believing a route answered, check the content-type,
or that the path is in `app.openapi()["paths"]`.

## Verify against the live system, not the exit code

A green deploy does not prove the container came back on the new image.
Check the box over SSM (instance `i-0eee645bac49b482a`):

```bash
aws ssm send-command --region ap-southeast-1 --instance-ids i-0eee645bac49b482a \
  --document-name AWS-RunShellScript --parameters file://cmds.json
```

Put the commands in a JSON file — nested quoting through `--parameters`
inline breaks in ways that look like the command failing.

Useful checks: `docker ps --format '{{.Image}} {{.Status}}'`, and
`docker exec ppe-pipeline-track2-test grep -c '<marker>' <file>` to prove
a specific change is really in the running image.

**DynamoDB `describe-table` ItemCount is stale** — it refreshes roughly
every 6 hours. It reported 0 on a table holding 4,490 rows. Query the
table; do not trust the count.

## The review host

Instance `i-0eee645bac49b482a`, stable app on **8501**, development on
**8090**.

**It no longer shuts down.** The 22:00 stop schedule
(`track2-yolo-review-host-stop`) was disabled on 2026-09-24; the 09:00
weekday start is still enabled as a way back if the host is ever stopped
by hand. So **the public IP is now stable** — only a stop releases it —
but check it after any restart rather than assuming:

```bash
aws ec2 describe-instances --instance-ids i-0eee645bac49b482a --region ap-southeast-1 \
  --query "Reservations[0].Instances[0].PublicIpAddress" --output text
```

## Two Terraform stacks

`terraform/app` is this app's own stack (ECR, CodeBuild, the audit
tables) and is what `deploy.sh` reads. `terraform/platform` owns the
shared estate — the instance, roles, bucket, schedule.

**The platform stack has drift.** A plan there wants to change ~18
resources and delete two outputs, almost none of it intended. Do not
`terraform apply` it to land a small change; make the targeted AWS change
and edit the `.tf` to match so a future apply agrees.

## A new card's settings go in the settings file — always

**The settings file is meant to hold every card's choices.** One file,
loaded on one machine, restores the whole page: the folders, the
detector, the shape of a run, the split, the retention curve. That is
what it is *for*.

So when you add a card, or add a control to one:

1. Add the key to `DEFAULTS` in
   [pull_control.py](src/ppe_pipeline_api/pull_control.py), with a
   comment saying what it is and why it is a setting.
2. Add it to `ConfigRequest` in
   [track2_pull_control.py](src/ppe_pipeline_api/routers/track2_pull_control.py),
   or the POST silently drops it — Pydantic discards fields the model
   does not declare, and nothing anywhere says so.
3. Load it on mount and write it on change, the way
   `Track2DetectorsPanel` does with `default_training`.

**A control whose value lives only in `useState` is not saved.** It
looks saved, because it is still on screen. It is gone on reload, and a
settings file written while it was on screen does not contain it.

This has already happened twice. The Assemble card and the Import
Checkpoint card were both added without any of the three steps, so
every folder, class list and destination chosen in them vanished on
reload — while the file reported a successful save, because the save
is honest: it snapshots the live config, and those cards never put
anything in it.

**How to check, in one command.** The settings file is a faithful copy
of the live config, so the question is only ever "is it in the config?":

```bash
# every panel, and whether it persists anything at all
for f in web/src/components/Track2*.tsx; do
  echo "$(basename "$f") -> $(grep -o 'saveTrack2[A-Za-z]*' "$f" | sort -u | tr '\n' ' ')"
done
```

A panel with a dozen `useState` lines and nothing after the arrow is a
card whose settings are not being saved.

`tests/test_card_settings_are_saved.py` enforces all of this: it parses
every `saveTrack2PullConfig` call in `web/src` and asserts each
top-level key is declared server-side, and it fails any panel holding
8+ pieces of state that writes none of them. **That test found
`golden_set`**, which had been silently dropped for as long as the card
had existed.

## How the cards are meant to behave

Decided on 2026-10-14. These are standing answers — do not re-litigate
them, and do not quietly build something that contradicts one.

### What a card saves, and what it must not

**Save the inputs and whether the card is open.** Folders, classes,
detector, numbers, ticks — and the open/closed state, so a loaded file
puts the page back the way it was left rather than collapsed.

**Do not save results.** Previews, discovered weeks, folder listings and
run summaries are re-fetched live on load. A saved preview that no
longer matches the bucket is a number somebody acts on, and nothing on
screen would say it was stale.

Card open/closed goes through `useCardOpen` in
[cardState.ts](web/src/lib/cardState.ts) — **one shared record**, not a
save per card. `write_config` merges at the *top* level, so `card_state`
is replaced wholesale on every write; seven cards each writing their own
key means six get dropped by whichever wrote last. That hook is a
drop-in for `useState`, updater form included (`setOpen((v) => !v)`), and
it has to stay that way: every call site is written that way.

### The class and model registry travels in the file

A settings file carries the full detector definitions in a `registry`
section **beside** `config`, never inside it — `load_settings_file`
writes `config` straight into the live configuration, so a copy in there
would be written into live config on every load where nothing reads it.

**On load: create missing only. Never update, never delete.** This
machine's detectors always win. A load must not change the classes of a
detector a run is training against, nor remove one that registered model
packages are filed under — the registry group name is derived from the
detector and the approval rule matches those names exactly. The file can
add, and that is all.

### The unattended chain runs all the way to a model

`pull → rebuild lists → rebuild data.yaml → freeze snapshot → queue →
start`. `submit="queue"` already does the whole thing; it is not
queue-and-wait. A chain that stopped at a queued entry would produce a
to-do item, not a model.

**Which pulls set it off is a setting**, saved with everything else:

| | Default | |
|---|---|---|
| `triggers` | `["schedule"]` | `arrival` and `manual` are selectable |
| `min_files` | `1` | what counts as a batch worth training on |
| `cooldown_minutes` | `0` | caps a burst into one run |

The defaults reproduce the old hardcoded behaviour exactly, so an
existing settings file cannot start chaining on arrival by surprise.

**`arrival` needs both guards before it is ticked.** It fires once per
upload burst. On 2026-10-06 three arrival pulls fired in ten minutes,
one of them moving a single file, and each would have been a full GPU
run. `min_files` is the difference between "somebody uploaded a photo"
and "a batch landed"; `cooldown_minutes` is what stops twenty uploads
becoming twenty runs. Never offer `arrival` without them.

## Make the screen do the work, not the person

The platform is used by somebody who has a job to do and did not help
design this. Every one of these came from a real complaint.

### The answer belongs where the question was asked

A control at the bottom of a card must not be answered by a message at
the top of it. Deleting a Model Name did exactly that: the button sat in
a row near the bottom, the refusal appeared as a red line at the top,
and the follow-up — **Delete it and its versions** — was a separate
button up there too. Three hundred pixels and a scroll away from the
thing that was clicked.

If an action can fail, can escalate, or needs a decision, all of that
goes **in one place, where the action is**.

### Destructive actions get a dialog, and it carries everything

Use [ConfirmDialog](web/src/components/ConfirmDialog.tsx). Not
`window.confirm`: it cannot carry a sentence of consequence, cannot show
what the server just answered, cannot offer a second and harsher option,
and looks like a browser error rather than part of the page.

The dialog says:

- **what goes**, named — and what does *not*. "Datasets and trained
  weights are untouched" is the sentence that lets somebody press it.
- **what the server said**, verbatim, when it refused. That text names
  the registries and version counts, which is the part that decides
  whether this is safe.
- **every choice**, including the harsher one. A refusal turns the same
  dialog into the escalation rather than sending the operator somewhere
  else to find it.

**Default to the safe answer.** Cancel takes focus, Escape closes,
clicking the backdrop closes. Enter on a freshly opened dialog must
never delete anything. The destructive button is never focused and never
the default.

### Do not make somebody finish an action somewhere else

A flow that starts at a button and ends at a different control the
person has to find is not one flow. If pressing the thing cannot
complete the job, the dialog that opens completes it.

### Say the consequence in their words

"This cannot be undone" earns its place. So does naming what survives.
What does not help is restating the button: a dialog that says "Delete
this? [Delete] [Cancel]" has told them nothing they did not already
know.

## Registry groups — the one that bites

Custom runs register as **Approved with no human in the loop**. The four
names `yolo11{s,m,l,x}-detector` are an exact-match list on the rule that
fires `dump_approved_weights`, which swaps the live inference component
and overwrites production weights. Any group name containing `-open-` or
a detector slug is inert there.

So: never name a new group `yolo11<size>-detector`. Keep new names
starting `yolo` and ending `-detector` to stay inside the existing
`model-package/yolo*-detector/*` IAM grant.

A run's registry group comes from the **detector** chosen; its class
names come from the **dataset's `data.yaml`**. Nothing forces those to
agree when no detector is selected.

## Datasets

An Ultralytics dataset is `data.yaml` + `images/{train,val}` +
`labels/{train,val}`. Labels are found by swapping `/images/` → `/labels/`
in each path, so that layout is mandatory, and a `data.yaml` can never
point at S3 — Ultralytics reads local files only.

**Be precise about which of those is a law and which is our code.**
`train:` and `val:` each take *a directory, a list of directories, or a
`.txt` file listing image paths one per line* — Ultralytics globs a
directory and reads a file as a path list (`get_img_files`;
`dataset_yaml.py` already uses the file-list form for retention). So the
yaml can name folders anywhere, in any number.

`train.py` takes a comma-separated list of prefixes and lays files out by
ONE rule: one prefix is stripped; several in one bucket keep full keys;
several buckets keep `<bucket>/<key>`. `dataset_yaml.local_layout` is that
rule on the app side -- the yaml, the remap block and snapshot paths all
use it. Never decide a layout from "is this folder in another bucket".

**A queued run trains on a dataset version, never a shared slot.** At
queue time the yaml is resolved to the exact images and written once
under `_datasets/<dv-id>/` (`dataset_versions.py`); the run gets that
yaml, the version's `train/` list folder as an extra download prefix, and
`_datasets/<id>/eval` (a KEY, not an `s3://` URI -- the start-up check
reads keys) as its evaluation folder. `<folder>/data.yaml`,
`_generated/<hash>/` and the evaluation yaml written into a val folder are
overwritten by every build, which is how a golden-set-only run once
trained on another run's evaluation yaml. Do not point a queued run at
them.

**A source's own `data.yaml` is the only record of its label numbering.**
Assemble keeps a verbatim copy as `data.source.yaml` the first time it
replaces one; a changed class list with no recoverable original is
refused, not guessed.

`assemble_dataset` copies scattered folders into that shape and writes
`sources.json` beside `data.yaml`, mapping each copied file back to the
key it came from. That mapping is the only record of where a dataset came
from; the assembled folder is deleted once the run finishes.

**Class names exist only in `data.yaml`.** A label line is
`0 0.51 0.22 0.26 0.08` — scanning labels gives you indices and counts,
never names.

## Ultralytics carries `single_cls` over from the starting checkpoint

`Model._reset_ckpt_args` keeps `imgsz`, `data`, `task` and `single_cls`
from a loaded `.pt`. One single-class ancestor (v14 of
`yolo11s-helmet-detector`) made every continuation single-class, named
the class `item`, and the evaluator refused it. So state every training
argument that matters explicitly -- `train.py` passes `single_cls=False`
now -- and suspect the base checkpoint when a run does something its
parameters did not ask for.

## Permission gaps that are known, not bugs to rediscover

- The app's role has Scan/Put/Get/Delete on `track2-yolo-dataset-snapshots`
  but **no UpdateItem** -- `dataset_versions.mark` reads, merges, puts.
- The queue-runner Lambda (`track2-app-queue-runner`) can only read S3 and
  has no snapshot-table access. It starts queued runs fine; a scheduled
  chain that has to BUILD (yaml, dataset version) cannot write from it.
- IAM changes from this session are blocked by the permission classifier
  unless the user approves them; prepare the policy and ask.

## Windows / Git Bash

Git Bash rewrites paths that look POSIX, which breaks `docker run -v`,
`-w`, and CloudWatch log group names like `/aws/sagemaker/TrainingJobs`.
Prefix with `MSYS_NO_PATHCONV=1`, or use the PowerShell tool.

Watch `MAX_PATH`: this repo's S3 keys are ~90 characters, and writing them
under a long scratch path exceeds 260 and fails.
