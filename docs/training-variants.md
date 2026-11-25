# Saved training variants

In Multi-Auto Training, select a Model Name, YOLO type/size and registered starting
version. Choose **Use a saved dataset** in the training card. This fixes the
recorded train/validation split for the comparison.

Adjust **How every run trains**, including advanced training parameters. Under
**Save training setups as variants**, enter a name and click **Save as variant**.
Saving writes a full settings snapshot to SQL; it does not start training. All
displayed defaults are captured, even when the normal training form uses defaults.

Change the settings and save another named variant. Earlier variants do not change.
Each model/baseline/dataset combination has its own saved table, which reappears when
you select that combination again. No variant dropdown or experiment mode is used.

The table highlights differences from the first saved variant. Edit loads that
variant into the normal training parameter controls; **Update variant** replaces
only that row, while **Save as new variant** creates another snapshot. Duplicate
and Remove operate on saved configurations, never existing trained models. The
normal training controls also supply scheduled training, as before; review those
settings before leaving them saved for a schedule.

Tick the variants to train and click **Review selected variants**, then **Run selected
variants**. Only this final action starts paid background work. Every selected
variant starts independently from the same preserved baseline and recreates the
same saved images/labels. The SQL-backed worker continues without a browser or AI
session, provided the backend remains running. Failed variants remain retryable.

After training, one common evaluation compares the original baseline against all
successful variants. View the batch rows, run numbers, settings differences,
rankings, bar charts and per-class results below the saved table. Original training
metrics are not overwritten. These are validation results, not an independent test.

Export the baseline and/or trained variants using the package controls below the
results: model `.pt`, images/labels, metadata, ZIP or S3 folder output. The complete
batch configuration and comparison results also have a JSON report download.

Historical datasets lacking a permanent central-store recipe are refused at launch;
the system never substitutes current source folders. No SQL schema migration is
needed. Existing run numbers and data are retained.
