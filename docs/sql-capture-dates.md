# SQL capture dates in weekly pulls

SQL image pulls preserve the Track 1 capture timestamp as S3 user metadata
`captured_at` on both images and labels, including labels generated from SQL.
The timestamp is normalized to UTC. Explicit capture columns take priority;
the timestamp in `case_key` is the fallback. Review timestamps are not used.

Per-file week assignment reads this metadata before filename or upload dates,
including saved configurations using `file_time`. Explicit scan mode still
assigns the whole sweep to one week. Retention continues to operate on those
week folders. S3 LastModified is not changed to a historical date.

Server-side copies retain the metadata, and class-ID rewrites retain label
metadata. An explicit repeat SQL pull fills missing capture metadata on
existing destination files without replacing their image/label contents.
Scheduled checks may skip files that are already present. Previously filed
week folders are not migrated or renumbered automatically.

Files without capture metadata retain the configured filename/upload fallback;
invalid capture metadata fails visibly rather than silently changing the date.
Metadata reads use bounded concurrency and an object-revision cache.
