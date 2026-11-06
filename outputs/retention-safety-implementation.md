# Retention and golden-set implementation

Date: 6 November 2026.

## Changes

- Golden validation assignments persist across rebuilds. Newly added pairs join training. A changed source or split policy requires a new target folder.
- Legacy split manifests are adopted without moving their existing members; content validation begins with the first upgraded build.
- Relative-path pairing preserves images with the same basename under different cameras. Ambiguous pair identities fail explicitly.
- Validation ETags are checked before reuse and conditional source copies detect changes during copying.
- A conditional S3 target lock prevents overlapping builds. Initial membership is reserved before copying, and successful split revisions have immutable history objects and benchmark IDs.
- Copy and cleanup errors prevent success reporting. Interrupted builds can retry the same assignments.
- Retention reports actual recent/history/golden contributions and names its collection-age semantics.
- An explicit label-coverage scan reports before/after class image counts and incomplete labels. Its saved warning threshold does not change the policy.
- Dataset versions expose validation selection IDs for comparing membership/taxonomy across runs.

## Scope

No new calendar-age mode, sampling budget, automatic class exemption, separate SageMaker test stage, or automatic model promotion was introduced. These alter training policy or pipeline behavior and require their own data/model experiments. The existing sampling policy remains intact; exclusions are now also reflected in the preview's selected counts.

Coverage inspection is scoped to weekly sources and golden training images, before class remapping. Picked folders, content duplicates, scene grouping, and final detector filtering remain outside that report. A reported class count measures images, not boxes.

## Recovery

A crash can leave `<golden target>/.split-build-lock.json`. Verify that the writer has stopped before removing that exact lock. Do not remove a live writer's lock. Retry the same configuration to complete a reserved initial split. A legacy target without reliable membership history cannot establish past evaluation independence.

## Validation

- Final repository suite: **2,596 passed**, one existing Starlette/httpx deprecation warning, 29.02 seconds.
- CodeBuild `track2-app:b33407f3-20d6-40b7-9ad8-68f6af305fee`: succeeded, including TypeScript/frontend build and app import.
- Development build: `20261106-003029`, port 8090, role `dev`, PostgreSQL backend. Image digest: `sha256:8e156546b142c682a5d778f385153627ab35a1b2317552ea0e2675754fd083d2`.
- Deployment SSM command: `844634ef-7aa8-4a19-a728-3714855541c1`. Production and shared Lambdas were not deployed.
- Post-deployment verification SSM command: `1db6f739-2814-4550-90d7-d56648af6c2c`. Health, exact build identity, authenticated retention preview/coverage, OpenAPI route, and compiled UI markers passed. An earlier check ran before replacement and correctly rejected the old build; it was superseded by this completed check.
- Live read-only inspection: 7 weekly inventory rows, 5 inspected sources, 1,189 selected training images: 100 recent and 1,089 historical (91.6%), no golden training contribution in that report. Labels inspected were readable/valid under the detection-label scan; week 6 had an observed class below the default 5-image warning threshold.
- No training or golden rebuild was submitted for live validation. Golden writes/retries/concurrency were exercised against local S3 doubles, not the live bucket.
- Visual browser QA could not run: the computer-use browser host/native pipe were unavailable. Compiled assets and HTTP behavior were verified; rendered layout and interactive browser behavior remain unverified.
- AWS authentication initially appeared absent because the sandbox could not read the user's `.aws` directory. The existing default profile worked outside the sandbox; no credentials were changed.
