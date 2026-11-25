# Experiments

Open **Multi-Auto Training**, then select **Experiment: baseline + variants**.
Use **Standard training** for the existing manual and scheduled workflow. Only
the selected mode's cards are visible. Switching modes preserves unfinished form
edits while the page remains open; the URL preserves the selected mode on reload.
Old Experiments links redirect to this mode, and saved experiments remain available.
Select a model name, architecture,
exact registered baseline version, and a saved SQL dataset. Older datasets without
a permanent central-store recipe are rejected rather than silently rebuilt from
today's source folders. Dataset and model class names must match.

Set shared training parameters, then duplicate or edit up to 20 variants. Every
variant starts from the same preserved baseline checkpoint and recreates the same
recorded train/validation file selection. Defaults are explicitly captured; choose
a fixed optimizer when comparing learning rates (Ultralytics `auto` may choose its
own learning rate). Review shows the effective settings before **Run experiment**
starts billed AWS work.

The SQL-backed worker prepares and submits variants sequentially. Queue publication
and experiment state are committed together. Deterministic submission identifiers
prevent a lost response or worker restart from creating a second training attempt.
Preparation progress, run numbers, failures and resource waits appear in saved
experiments. The backend must remain running; the browser and AI chat need not.

After all variants finish, a separate processing job evaluates the original
baseline and every successful variant on the same saved validation files with one
set of evaluation parameters. This does not overwrite original training scores.
The table and charts include ranking, baseline deltas and per-class results. These
are validation comparisons, not an independent test benchmark.

Failed variants can be retried after current work finishes; successes remain intact
and retry history is retained. A completed retry triggers a fresh final comparison.
Use **Resume / retry comparison** after resolving a preparation or comparison error.

**Export experiment models** reuses the existing package exporter: select baseline
and/or successful variants, choose `.pt`, images/labels and metadata, then review
sizes and export as ZIP or directly to an S3 folder. The complete experiment
configuration and comparison report have a separate JSON download.

No schema migration or new paid training run is required to deploy this feature.
