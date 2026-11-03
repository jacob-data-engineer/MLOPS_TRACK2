# Track2 Retraining Platform — User Guide

*Preliminary — 2 November 2026. Describes the development app (port 8090).
Production (8501) runs an older build until it is promoted, so some of what
is described here is not there yet. For what each file does and the AWS
infrastructure, see [technical_documentation.md](technical_documentation.md).*

---

## 1. What this platform is for

It retrains the site's object-detection models (YOLO) on new images. The
work is split into two halves, each with its own tab, its own settings file
and its own schedule:

| Half | Tab | Question it answers |
|---|---|---|
| **Data** | Track2 Data Preparation | *What does a model learn from?* — which folders, which weeks, how much of the older weeks, which images are held back to test on |
| **Training** | Track2 Multi-Auto Training | *Which model, trained how, and when?* |

Around them: a tab for **finding images in the review database** (SQL
Database Query) and a tab for **looking back at what was trained on what**
(Data Catalog).

## 2. Where to find it

| | Address |
|---|---|
| Development | http://18.140.196.76:8090 |
| Production | http://18.140.196.76:8501 |

Log in with the app password. The **badge under the logo** says which one you
are looking at: coloured means it is *not* production.

> **Development is not a sandbox.** Both apps share the same data, buckets,
> training pipelines and schedules. Pressing *Run now* or *Pull* in
> development moves real files and can start real training runs.

## 3. The tabs, in the sidebar

| Tab | Use it to | Status |
|---|---|---|
| Track2 SQL Database Query | Pick human-verified frames from the review database, preview them, pull them into a training folder | new |
| Track2 Data Preparation | Choose and prepare the training data; the pull schedule | ready (green) |
| Track2 Data Catalog | See every run and dataset version, their files; export them | new |
| Track2 Multi-Auto Training | Choose the model, start or queue training; the training schedule | ready (green) |
| Track2 Scheduled | See what the clocks will start, and when (read only) | prototype (yellow) |
| Track2 Review & Metrics | Compare trained models | prototype (yellow) |
| Track2 Notifications | Who is emailed about what | — |
| Track2 Help | Built-in help | — |

---

## 4. Settings files — one per tab

Each of the two main tabs has a toolbar pinned at the top with its own
settings file.

| Tab | Saves as | Holds |
|---|---|---|
| Data Preparation | `<name>_DP` | folders, weeks, retention, golden set, Assemble, the pull schedule, the SQL tab's choices, the central data store |
| Multi-Auto Training | `<name>_AT` | the model and classes, how a run is shaped, which version it continues from, the training schedule |

- **Save**: type a name (e.g. `helmet`) and press *Save as* — the ending
  `_DP` or `_AT` is added for you. Leave the name blank to save over the file
  you are in.
- **Load**: pick a file and press *Load*. Only that tab's settings change;
  the other tab is untouched. The page reloads to show them.
- **New**: puts *this tab's* settings back to defaults. The other tab is
  untouched. Saved files are never touched by New.
- **Delete**: removes the file; asks first.

A file saves what is **in force**. A card with an *Apply* button must be
applied before its change is in the file — the toolbar tells you if one was
not.

## 5. Schedules — one per tab

**Pull schedule** (Data Preparation toolbar) — when new images are collected:
*Off*, *When files arrive*, or *On a schedule* (every N days at a time, SGT).
Press **Apply** after changing it.

**Training schedule** (Multi-Auto Training toolbar) — when a model is trained:

| Choice | What happens |
|---|---|
| Off | Nothing trains by itself |
| After each data pull | Every scheduled pull also builds the dataset, freezes it, queues and trains |
| On its own schedule | Trains every N days at a time, from the data as it stands; pulls only bring data |

With *When files arrive* plus *After each data pull*, always set **at least N
files** and **at most one run every N minutes** — otherwise every single
upload starts a GPU run.

---

## 6. Data Preparation — preparing what a model learns from

Work top to bottom.

### Assemble a dataset
Point at image and label folders anywhere in S3; the card checks they pair up
and shows the `data.yaml` it will build. Tick **Use these folders** to include
them.

**Run: Manual / Scheduled**
- *Manual* — built when you ask.
- *Scheduled* — rebuilt and frozen by itself after each **scheduled** pull
  (the PULL SCHEDULE above must be *On a schedule*). Nothing is trained. If
  the training schedule is *After each data pull*, that chain does the build
  instead, so this stands down.

### Pick Golden Set / Edge Cases
A hand-picked set split into train and val. **Its val side is the fixed test
set** every run is scored on — keep it the same between runs so scores can
be compared, and never train on it.

### Pull control
Collects new files from the source folders into **week folders**
(`week_1`, `week_2` …), dated by when each frame was captured.

### Auto Training (with Retention inside)
The week folders and how much of each older week a run takes (the decay
curve).

### Pull queue
Pulls waiting their turn, so two never run at once.

### Central data store
The one bucket and folder where all data is meant to be kept. Below it, the
**Catalog** line shows what is recorded (copied from DynamoDB every 5
minutes), a **Sync now** button, and a link to the Data Catalog.

---

## 7. SQL Database Query — finding human-verified frames

1. **Database** — pick one. If it says *stopped*, press **Start the
   database** (5–10 minutes; the page notices by itself). Press *Stop it
   when you are done* to save cost. *(The Start/Stop permission is awaiting
   approval — until then the button says so.)*
2. **Class** — e.g. *no helmet*. The list and counts come from the database.
3. **Which frames** — *All human-resolved*, *Ready for training (helmet box
   present)*, or *Test-grade only (box drawn by a person)*. The last two
   apply to the helmet class only.

The frames appear as thumbnails straight away (60 at a time, *Show more*).
Only frames where **every person** was answered by a human are included;
VLM-only and pre-filled answers are left out.

- **Show labels** draws the boxes: blue = person, green = helmet. A red
  border means a helmet box is missing.
- **Pull these frames into a training folder** — choose a bucket, a folder
  and the % that goes to train (the rest to val), tick **Include labels**,
  and press *Pull*. The folder gets `images/`, `labels/`, a `data.yaml`, and
  `sources.json` / `query.json` recording where everything came from. Nothing
  is overwritten; pull again to add more. Then point **Assemble** at it.

> **Check the overlay before training.** The boxes cover only the people the
> review app judged. Anyone else in the frame is unlabelled, and a model
> trained on that learns those workers are background. Complete such frames
> (e.g. in Label Studio) first.

*Advanced: write your own SQL* at the bottom gives the raw editor and table
list. Queries are read-only and cannot change the database.

---

## 8. Multi-Auto Training — training a model

1. **Model types and Model Names** — choose the Model Name (its classes) and
   size, e.g. `helmet` / `yolo11s`. *Define a new Model Name* adds one.
2. Choose the training data — the dataset prepared on Data Preparation, or a
   frozen snapshot.
3. Set how the run is shaped (epochs, image size, batch …).
4. **Start the training run**, or add it to the **Training queue** — runs
   wait their turn for the GPU.

Every queued run trains on a **frozen dataset version**: the exact images are
fixed at queue time, so later changes cannot alter what it trains on.

*Import a pretrained COCO or an already-trained model* files a `.pt` as a
model version without training.

---

## 9. Data Catalog — looking back, and exporting

**Training runs** — every run, numbered from Run 1 (oldest): when, status,
model, the dataset version it trained on, images, and the model it
registered.

**Dataset versions** — every frozen dataset: *version* (frozen when a run was
queued) or *snapshot* (frozen by hand), train/val counts, and which runs used
it. A version marked *deleted* was removed in the app but is kept here.

**Click a run or version** to see its files: filter by split (and for runs,
training or evaluation), *Show thumbnails*, page through 100 at a time.

- **File paths only (CSV)** — an inventory of paths and splits, not the files.
- **Local PC (download ZIP)** — press **Prepare ZIP for PC** to package the
  actual images, labels, and metadata in the background. On **Exports**, press
  **Download ZIP** when ready (link valid one hour). Your browser controls the
  local save location.
- **S3 bucket** — select an available bucket and browse its folders.
  Choose an existing folder, or enter a new name and press **Create & use**.
  Choose **ZIP archive** or **Folder (individual images and labels)**, then
  press **Export ZIP to S3** or **Copy files to S3 folder**. New folders are
  created as files are written. Exports are stored
  under a new `<run-or-version>-<export-id>/` subfolder; its address appears in
  the Exports table. The app's AWS role needs read access to the sources and
  write access to the destination; choosing a bucket does not grant access.
- Both destinations preserve `images/`, `labels/`, `manifest.csv`, and
  `data.yaml` when class names are known. The Exports tab also lets you export
  an existing entry again to either destination.

An export with missing source files is marked **Incomplete**; `manifest.csv`
names them. Its available files can still be downloaded or inspected in S3.
Permission or transfer errors mark the export **failed**. An interrupted S3
copy can leave some files in its unique subfolder; retrying uses a new folder.

---

## 10. Good practice

- **Keep the golden set's val side fixed** and score every run on it.
- **Do not train on frames whose labels are incomplete** (check *Show labels*).
- **Keep near-identical frames from one alarm on the same side** of the
  train/val split, or scores look better than the model is.
- **Save a settings file** after a setup you want to come back to — one per
  tab.
- **Stop the preview database** when finished with it.

## 11. Known limitations (at the time of writing)

- The SQL tab's **Start/Stop** buttons wait on a permission approval.
- Labels pulled from the review database **cover only judged people**.
- **Worker on edge** finds more frames (461) than the readiness report (351);
  not yet traced.
- Older dataset versions may export with **missing files** until images are
  copied into the central store (planned).
- The app still **reads** records from DynamoDB; the catalog is a copy until
  production is promoted.
