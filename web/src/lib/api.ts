/**
 * Thin fetch wrapper for the ppe_pipeline_api FastAPI backend (see
 * src/ppe_pipeline_api/CONTRACT.md at the repo root -- this file should
 * stay in step with that document, not guess at shapes).
 *
 * The base URL points at the local FastAPI dev server by default; override
 * with VITE_API_BASE_URL for any other deployment.
 */

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

let authToken: string | null = null;

/** Called by AuthContext whenever the session token changes (login/logout). */
export function setAuthToken(token: string | null): void {
  authToken = token;
}

let onUnauthorized: (() => void) | null = null;

/** Registered by AuthContext so a DEAD SESSION drops to the sign-in screen.
 *
 * Sessions live in the server's memory, so every deploy -- and every restart
 * -- invalidates every token that browsers are holding. Before this, the 401
 * was thrown like any other error and nothing cleared the token: the whole app
 * showed "Invalid or expired session token." on every panel, on every poll,
 * with no way out except knowing to press Sign out. That is not an error the
 * user can act on; it is a session that ended. */
export function setUnauthorizedHandler(handler: (() => void) | null): void {
  onUnauthorized = handler;
}

const SESSION_PROBE_PATH = "/auth/session";
/** At most one probe in flight: a page mid-refresh fires a dozen calls at
 * once, and every one of them would otherwise probe separately. */
let sessionProbe: Promise<void> | null = null;

/** Asks the server whether the session is actually dead before signing
 * anyone out.
 *
 * A 401 on one endpoint is not proof: it can come from a request that
 * raced a token change, or from a route failing for its own reasons. The
 * old code treated every 401 as a dead session and cleared the token
 * immediately, so a single unlucky background poll dropped the operator
 * to the sign-in screen with their work in progress lost.
 *
 * Only a 401 from the probe itself clears the session. Anything else --
 * including the probe failing to reach the server at all -- leaves the
 * session alone, because a network blip is not a logout.
 */
function confirmSessionIsDead(): Promise<void> {
  if (sessionProbe) return sessionProbe;
  sessionProbe = (async () => {
    try {
      const headers = new Headers();
      if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
      const res = await fetch(`${API_BASE_URL}${SESSION_PROBE_PATH}`, { headers });
      if (res.status === 401 && onUnauthorized) onUnauthorized();
    } catch {
      // Unreachable server: keep the session and let the call that
      // triggered this surface its own error.
    } finally {
      sessionProbe = null;
    }
  })();
  return sessionProbe;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const headers = new Headers(init.headers);
  if (!(init.body instanceof FormData) && init.body !== undefined) {
    headers.set("Content-Type", "application/json");
  }
  if (authToken) {
    headers.set("Authorization", `Bearer ${authToken}`);
  }

  const res = await fetch(`${API_BASE_URL}${path}`, { ...init, headers });

  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // non-JSON error body -- fall back to statusText
    }
    // A 401 USUALLY means the session ended -- but not always, and acting
    // on it immediately meant one failing background poll signed the
    // operator out mid-task. Confirm against /auth/session first, and
    // only then fall back to sign-in.
    //
    // NOT for the login request itself: a wrong password is a 401 the
    // login form must show, and logging out mid-login would wipe the
    // error. Nor for the probe, which would recurse.
    if (res.status === 401 && path !== "/auth/login" && path !== SESSION_PROBE_PATH) {
      void confirmSessionIsDead();
    }
    throw new ApiError(res.status, detail);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

/** GETs already in flight, by path.
 *
 * One page mount fans out far more requests than it looks like it does:
 * five separate panels on Multi-Auto Training each call
 * `/track2/pull/control` for themselves, four each call `/track2/models`
 * and `/track2/detectors`. That is the same answer fetched five times,
 * concurrently, every time the page opens -- slower to settle, and five
 * chances for one of them to be the one that gets throttled.
 *
 * Coalescing them here rather than lifting the state into a provider
 * keeps every panel self-contained, which is what makes them movable
 * between pages. It is NOT a cache: the entry is dropped the moment the
 * request settles, so the next call is a fresh request and nothing can
 * go stale. It only merges the ones that overlap.
 */
const inFlightGets = new Map<string, Promise<unknown>>();

/** A private copy for each extra subscriber.
 *
 * The first caller gets the response object itself; everyone joining an
 * in-flight request gets a clone. Without this a panel that sorted or
 * spliced its result in place would be reaching into another panel's
 * state, and that bug would appear only when the two happened to mount
 * close enough together to share -- which is the worst kind to chase.
 */
function privateCopy<T>(value: T): T {
  try {
    return structuredClone(value);
  } catch {
    // Non-cloneable (or no structuredClone): the responses are JSON, so
    // this round-trip is equivalent for everything that actually lands
    // here.
    return JSON.parse(JSON.stringify(value)) as T;
  }
}

export function apiGet<T>(path: string): Promise<T> {
  const existing = inFlightGets.get(path) as Promise<T> | undefined;
  if (existing) return existing.then(privateCopy);

  const started = request<T>(path, { method: "GET" }).finally(() => {
    inFlightGets.delete(path);
  });
  inFlightGets.set(path, started);
  return started;
}

export function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "POST",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}


export function apiDelete<T>(path: string): Promise<T> {
  return request<T>(path, { method: "DELETE" });
}

/** POST with a multipart body (file upload) -- `request` already skips the
 * JSON Content-Type header for FormData bodies, so this just names the
 * intent at call sites instead of building RequestInit by hand each time. */
export function apiPostForm<T>(path: string, formData: FormData): Promise<T> {
  return request<T>(path, { method: "POST", body: formData });
}


// ---------------------------------------------------------------------------
// Auth
// ---------------------------------------------------------------------------

export interface LoginResponse {
  token: string;
}

export function login(password: string): Promise<LoginResponse> {
  return request<LoginResponse>("/auth/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
}

// ---------------------------------------------------------------------------
// Helmet CV vs judge -- production CV's own violation decision on a frame,
// set against the judge's (the VLM verification stage the expansion spec
// routes CV violations to), both scored by a reviewer who has seen neither.
//
// TWO FACTS ABOUT THIS DATA DRIVE THE WHOLE SURFACE.
//
// 1. THE SAMPLER IS GATE-BLIND. Candidates are drawn without reference to
//    what CV called them, so the audit set holds frames CV cleared as well as
//    frames it flagged. A reviewer calling "violation" on a cleared frame is
//    a genuine false negative, which is what makes recall, F1 and accuracy
//    computable here at all. An earlier cut of this file asserted they were
//    not measurable; that was wrong. The fix was not to hide the figures but
//    to name the population they cover -- see `HelmetCvSummary.population`,
//    which the server writes and this app renders verbatim.
//
// 2. ONLY `no_helmet` ROUTES TO THE JUDGE. Per the expansion spec, section 4,
//    only a CV VIOLATION goes on to the VLM stage. `unknown` is CV declining
//    to answer, which is not a violation claim, so it does not route -- and
//    nothing on this screen may leave a reader thinking it did.
// ---------------------------------------------------------------------------

/**
 * What the ledger filters on: CV's own STATE, not what routed.
 *
 * It filtered on the gate reason until the spec settled that only a violation
 * routes, at which point a "gate reason" filter had exactly one reason left
 * in it and its second segment named something that never routes at all. The
 * state is the honest axis, and routing is now a column of its own rather
 * than a property conflated with the filter.
 *
 * `helmet_worn` folds CV's three colour states into the one claim they share.
 * The five states carry three distinct claims -- a helmet is worn, no helmet
 * is worn, no answer -- and the claim is what a reviewer picks work along;
 * the colour is a property of the helmet, not a different finding.
 */
/** THERE IS NO ROLE FILTER, and there cannot be one.
 *
 * The ledger filtered on CV's gate, then CV's colour state, then briefly on
 * frame role. `cv_verdict` on the server is literally
 * `return "TP" if frame_role == "alarm"`, so frame role IS the CV arm's
 * verdict — a filter over it would disclose CV's claim on exactly the frames
 * the CV precision measurement rests on, before the reviewer decided. It was
 * also worthless: every alarm holds one alarm frame plus its clip frames by
 * construction, so an unrestricted role set is `["alarm","clip"]` on every
 * row. Which frame a row is stays visible per frame, post-review, inside the
 * withheld `cv` block. */


// ---------------------------------------------------------------------------
// Track2 (YOLO training pipeline) -- /track2/review, /track2/models,
// /track2/schedule, /track2/lineage, /track2/metrics, /track2/deploy,
// /track2/inference, /track2/notifications, /track2/export
//
// Ported from the standalone terraform/review-app/review_app.py app (no
// auth, its own EC2 instance) -- every one of these now sits behind the
// same session auth as the rest of this API. See CONTRACT.md's "Track2"
// section for the authoritative shapes.
// ---------------------------------------------------------------------------

export interface Track2Lineage {
  id: string;
  record_type: "baseline" | "training_run";
  timestamp_utc?: string;
  [key: string]: unknown;
}


export interface Track2RegistryPackageBrief {
  model_package_arn: string;
  version: string;
  approval_status: string;
}

export interface Track2RegistrySummaryBrief {
  total: number;
  approved: number;
  packages: Track2RegistryPackageBrief[];
}

export type Track2RegistriesBrief = Record<string, Track2RegistrySummaryBrief>;


export function deleteTrack2Model(modelPackageArn: string): Promise<{ model_package_arn: string; deleted: boolean }> {
  return apiPost(`/track2/review/model/${encodeURIComponent(modelPackageArn)}/delete`);
}

// --- Training progress / GPU / queues ---------------------------------------

export interface Track2TrainingProgress {
  execution_arn: string;
  status: string;
  start_time: string | null;
  steps: Record<"TrainYolo" | "EvaluateYolo" | "RegisterYoloModel", string>;
  epoch_progress: { current: number; total: number } | null;
  download_progress: { current: number; total: number } | null;
}

export interface Track2GpuJob {
  name: string;
  start_time: string | null;
  own_pipeline: boolean;
  /** Managed spot. It draws on a DIFFERENT quota, so it cannot block an
   * on-demand run -- see SPOT_USES_A_DIFFERENT_QUOTA server-side. */
  spot?: boolean;
  /** Whether an instance was ever allocated. `start_time` is null until
   * one is, and a job in a capacity retry loop holds no hardware and is
   * billed nothing -- "InProgress" on its own reads as "running". */
  started?: boolean;
}

export interface Track2QueueEntry {
  /** The folders the images actually came from, resolved at queue time.
   * Absent when the dataset was not assembled -- then dataset_folder is
   * already the origin. */
  dataset_source_folders?: string[];
  eval_source_folders?: string[];
  /** Images used from each folder, keyed as the two lists above. */
  dataset_source_counts?: Record<string, number>;
  eval_source_counts?: Record<string, number>;
  /** The frozen dataset this entry trains on (dataset_versions.py). */
  dataset_version_id?: string;
  dataset_version_train_images?: number;
  dataset_version_val_images?: number;
  id: string;
  generation: string;
  size: string;
  epochs: number;
  img_size: number;
  batch: number;
  seed: number;
  dataset_folder: string;
  eval_dataset_folder?: string;
  base_model_folder: string;
  notes: string;
  added_at: string;
  build_id?: string;
  started_at?: string;
  /** Set on entries that run through the open (custom-class) pipeline.
   * Those have no build_id -- they are started as a pipeline execution
   * directly, so `run_id` is what identifies them. */
  custom_classes?: boolean;
  run_id?: string;
  /** The Model Name this run registers under, and the architecture it
   * trains as. Both have always been stored on the entry server-side;
   * they were simply never declared here, so nothing could show them.
   *
   * Worth showing: the registry group is derived from this pair, and
   * the CLASS NAMES come from the dataset's own data.yaml -- nothing
   * forces the two to agree. A run queued against the wrong Model Name
   * trains happily, reports metrics and files them under a taxonomy
   * that does not describe it. */
  detector_slug?: string;
  architecture?: string;
  /** The frozen definition this entry trains on, captured when it was
   * queued. An entry can sit for hours while the live slot is restored
   * to something else, so reading it at run time would name whatever
   * happened to be current then rather than what was chosen. */
  snapshot_id?: string;
  /** When that snapshot was FROZEN -- not when it was queued. */
  snapshot_created_at?: string;
  snapshot_files?: number;
  /** True when the run reads a pinned file list rather than the week
   * folders, which is the difference between repeatable and not. */
  snapshot_pinned?: boolean;
  snapshot_note?: string;
}

export interface Track2QueueSnapshot {
  pending: Track2QueueEntry[];
  running: Track2QueueEntry | null;
  started: boolean;
  completed_count: number;
  last_error: string;
  max_size: number;
}

export interface Track2ModelsResponse {
  registries: Track2RegistriesBrief;
  latest_approved: Record<string, Record<string, { group: string; version: string } | null>>;
  generation_sizes: Record<string, string[]>;
  generation_labels: Record<string, string>;
  size_labels: Record<string, string>;
  default_dataset: string;
  training_progress: Track2TrainingProgress | null;
  /** The open pipeline's executing run id, or null. `training_progress`
   * now describes whichever pipeline ran most recently, so this remains
   * the signal for "a custom-class run is on the GPU right now". */
  custom_run_id: string | null;
  pending_message: string | null;
  gpu_jobs: Track2GpuJob[];
  cpu_jobs: Track2CpuJob[];
  queue: Track2QueueSnapshot;
  /** `yaml_files` are the .yaml/.yml names at this exact level, so the
   * data.yaml picker can offer the FILE rather than a folder tick box
   * that says nothing about which file it is about to read. */
  dataset_browser: { prefix: string; folders: string[]; yaml_files: string[] };
  /** `weight_files` are names at this exact level, so a base model can be
   * an arbitrary .pt key rather than a folder holding a `model.pt`. */
  weights_browser: { prefix: string; folders: string[]; weight_files: string[] };
  eval_browser: { prefix: string; folders: string[] };
}

/** One audit row per submitted run. Written at submit time with what the
 * operator chose, then completed by the server once the run finishes --
 * so `ended_at`, `model` and `metrics` are null until then. */
/** The parts of the Models page that go stale while you look at it.
 *
 * Its own endpoint because the full payload lists registries, S3 folders
 * and the dataset browser -- far too heavy to poll. The page used to
 * fetch all of it once and then sit on it, which is why the GPU card
 * could read "Available" in green while a training job was downloading
 * its dataset. */
export interface Track2ModelsLive {
  training_progress: Track2TrainingProgress | null;
  custom_run_id: string | null;
  pending_message: string | null;
  gpu_jobs: Track2GpuJob[];
  cpu_jobs: Track2CpuJob[];
  queue: Track2QueueSnapshot;
}

/** An InProgress SageMaker processing job -- evaluation, in practice. */
export interface Track2CpuJob {
  name: string;
  step: string;
  instance_type: string;
  own_pipeline: boolean;
}


export function getTrack2ModelsLive(): Promise<Track2ModelsLive> {
  return apiGet<Track2ModelsLive>("/track2/models/live");
}

export interface Track2TrainingAuditDataset {
  /** Where the run's file names came from. "list" -- the yaml's file
   * lists, which is exactly what the run read. "folder" -- a recursive
   * listing of the dataset folder, which is everything that was there.
   * Only the first can be reproduced exactly. */
  names_from?: "list" | "folder";
  /** The data.yaml those names were read through, if there was one. */
  yaml_key?: string;
  /** The folder the trainer actually read. */
  folder: string;
  s3_uri: string;
  file_count: number;
  /** Fully qualified, and resolved to the ORIGIN where the assemble step
   * recorded one -- the folder someone picked, not the dated copy. */
  file_names: string[];
  /** The distinct folders those images were copied FROM. Absent for a
   * dataset not built by the assemble step. */
  source_folders?: string[];
  /** How many label files were recorded alongside the images. */
  label_count?: number;
  /** file_names is a capped sample, not the whole set -- show file_count. */
  truncated: boolean;
  /** How many per-file rows were recorded for this dataset. The row
   * itself cannot hold every name -- a DynamoDB item is capped at 400KB
   * -- so they live in their own table, and Export CSV reads them from
   * there. 0 for runs audited before that table existed. */
  file_rows?: number;
}

export interface Track2TrainingAudit {
  id: string;
  /** Which Class the run was for. The registry group implies it, but
   * only to somebody who knows how those names are built -- and a run
   * that never reached registration has no group at all. */
  detector_slug?: string;
  /** The frozen dataset, when there was one. It reached the history
   * only inside the free-text notes before, which made it a habit
   * rather than a record. */
  snapshot?: {
    snapshot_id?: string;
    created_at?: string;
    files?: number;
    pinned?: boolean;
    note?: string;
  };
  build_id: string;
  source: string;
  status: string;
  submitted_at: string;
  started_at: string | null;
  ended_at: string | null;
  generation: string;
  size: string;
  parameters: { epochs: number; img_size: number; batch: number; seed: number };
  notes: string;
  /** This run's own copy of the yaml and file lists it was sent, under
   * `_runs/<id>/`. Copied at submit time, so it survives the live slots
   * being overwritten by a later save. */
  artifacts?: {
    folder?: string;
    s3_uri?: string;
    yaml_key?: string;
    train_list_key?: string;
    val_list_key?: string;
    train_list_lines?: number;
    val_list_lines?: number;
  };
  train_dataset: Track2TrainingAuditDataset;
  eval_dataset: Track2TrainingAuditDataset;
  /** True when no separate evaluation set was chosen, so the run was
   * evaluated on its own training data. */
  eval_is_train_dataset: boolean;
  base_model: { folder: string; source: string };
  /** What the training job was actually given, read off the job itself --
   * which is not always what was submitted, because the deployed
   * run_pipeline.py substitutes its own dataset and hyperparameters. Null
   * until the job exists. */
  actual: {
    training_job_name: string;
    instance_type: string;
    epochs: number | null;
    img_size: number | null;
    batch: number | null;
    seed: number | null;
    train_data_s3_uri: string;
    base_model_s3_uri: string;
    run_id: string;
  } | null;
  model: { group: string; version: string; model_package_arn: string; approval_status: string } | null;
  metrics: Record<string, number | null> | null;
  execution_arn: string;
}

export function getTrack2TrainingAudit(limit = 50): Promise<{ audits: Track2TrainingAudit[] }> {
  return apiGet(`/track2/models/audit?limit=${limit}`);
}

/** Removes the run from the history and deletes the model version it
 * produced -- `deleted_model` names what went, or is null if the run never
 * registered one. */
export function deleteTrack2TrainingAudit(
  auditId: string,
): Promise<{ deleted: boolean; deleted_model: string | null }> {
  return apiPost(`/track2/models/audit/${encodeURIComponent(auditId)}/delete`);
}

// `undefined` means "not specified, use the server's nice landing default";
// "" (explicitly passed) means "the bucket root" -- these are NOT the same
// thing, and conflating them (e.g. `if (browsePrefix) q.set(...)`, which
// treats an empty string as absent) is exactly what made navigating to the
// bucket root silently snap back to the default subfolder instead.
export function getTrack2Models(
  browsePrefix?: string,
  weightsBrowsePrefix?: string,
  evalBrowsePrefix?: string,
): Promise<Track2ModelsResponse> {
  const q = new URLSearchParams();
  if (browsePrefix !== undefined) q.set("browse_prefix", browsePrefix);
  if (weightsBrowsePrefix !== undefined) q.set("weights_browse_prefix", weightsBrowsePrefix);
  if (evalBrowsePrefix !== undefined) q.set("eval_browse_prefix", evalBrowsePrefix);
  return apiGet<Track2ModelsResponse>(`/track2/models?${q}`);
}

export interface Track2TrainingParams {
  generation: string;
  size: string;
  epochs?: number;
  img_size?: number;
  batch?: number;
  seed?: number;
  dataset_folder: string;
  /** Omitted or "", the run is evaluated on its training dataset. */
  eval_dataset_folder?: string;
  base_model_folder?: string;
  notes?: string;
  /** Custom-class runs: fine-tune from a registered package instead of a
   * browsed/uploaded folder. Exactly one of the two may be set. */
  base_model_package_arn?: string;
  /** Which YOLO architecture the base checkpoint is. */
  architecture?: string;
  /** Train on the dataset's own classes, via the custom pipeline. Sent on
   * queued entries too, so a queued custom run still runs as one. */
  custom_classes?: boolean;
  /** An exact .pt/.tar.gz key anywhere in the bucket. */
  base_model_key?: string;
  /** Start from random weights (resolved to the seed for `architecture`). */
  base_model_scratch?: boolean;
  /** Which detector to train. Picks the taxonomy the dataset is checked
   * against and the per-size registry group the version lands in. */
  detector_slug?: string;
  /** Early stopping: epochs without improvement before the run stops
   * itself. 0 leaves it to Ultralytics (whose own default is 100). */
  patience?: number;
  /** Everything else Ultralytics takes -- optimiser, learning rate,
   * schedule, loss weights, augmentation.
   *
   * OVERRIDES ONLY: a key absent means "whatever Ultralytics does by
   * default", so a run that sets none of this trains exactly as it did
   * before the field existed. The trainer validates each key against the
   * Ultralytics it actually has and refuses one it does not know, rather
   * than dropping it in silence.
   *
   * TYPED: numbers and booleans as JSON numbers and booleans, `freeze`
   * as a count or a list of layer indices. Ultralytics refuses a string
   * `lr0` and ignores a string `freeze`; the API checks every value
   * against src/ppe_pipeline_api/hyperparams.py before a job starts. */
  hyperparams?: Track2Hyperparams;
}

/** One extra Ultralytics training argument, as the run receives it. */
export type Track2HyperparamValue = string | number | boolean | number[];
/** The extra Ultralytics training arguments a run is sent. */
export type Track2Hyperparams = Record<string, Track2HyperparamValue>;


export interface Track2CheckpointInspection {
  key: string;
  s3_uri: string;
  folder: string;
  declared_architecture: string;
  detected_architecture: string;
  architecture_matches: boolean;
  architecture_known: boolean;
  classes: Record<string, string>;
  size_bytes: number;
}


export interface Track2DatasetClasses {
  folder: string;
  classes: Record<string, string>;
  matches_production: boolean;
  production_classes: Record<string, string>;
}

/** Advisory only -- used to preselect the pipeline and explain why, never
 * to route the run. A person/helmet dataset can legitimately want either
 * pipeline, so the choice stays the operator's. */
export interface Track2DatasetLayoutSplit {
  split: string;
  /** Which source within the split. A split can now be several folders
   * merged, and one total would hide an empty folder among full ones. */
  source_index?: number;
  images_prefix: string;
  labels_prefix: string;
  image_count: number;
  label_count: number;
  paired: number;
  /** Images with no label file. A few are wanted as background examples;
   * all of them usually means the wrong label folder. */
  unlabelled: number;
  orphan_labels: number;
  sample: string[];
}

export interface Track2DatasetLayout {
  splits: Track2DatasetLayoutSplit[];
  warnings: string[];
  total_paired: number;
}

/** The four single folders. Deliberately all-string: the single-source
 * panel indexes this by `keyof`, so adding a non-string field here breaks
 * every one of those lookups. */
export interface Track2DatasetFolders {
  train_images: string;
  train_labels: string;
  eval_images: string;
  eval_labels: string;
}

export interface Track2DatasetSource {
  images: string;
  labels: string;
}

/** What actually goes over the wire. Several folders may be merged into
 * one split; when present these take precedence over the single fields,
 * which the one-source panel still sends. */
export interface Track2DatasetRequest extends Track2DatasetFolders {
  train_sources?: Track2DatasetSource[];
  eval_sources?: Track2DatasetSource[];
}

/** Read-only preview: pairs the chosen image and label folders on
 * filename stems and reports what it found, before anything is written. */
/** The whole run as CSV, every file name included.
 *
 * Fetched rather than linked: the endpoint needs the bearer token, and a
 * plain anchor would send none and download a 401 page instead. */
export async function exportTrack2TrainingAuditCsv(auditId: string): Promise<Blob> {
  const headers = new Headers();
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
  const res = await fetch(
    `${API_BASE_URL}/track2/models/audit/${encodeURIComponent(auditId)}/export.csv`,
    { headers },
  );
  if (!res.ok) {
    // Raw fetch, so it misses the 401 handling in request(). Without this
    // a dead session showed "Export failed (401)" beside a page that
    // still looked signed in -- every deploy ends every session, so that
    // is the most likely reason this ever fails.
    if (res.status === 401) {
      void confirmSessionIsDead();
      throw new ApiError(401, "Your session ended. Sign in again and retry.");
    }
    throw new ApiError(res.status, `Export failed (${res.status})`);
  }
  return res.blob();
}

/** A short-lived link to the .pt this run produced. The browser then
 * downloads straight from S3, so a 19MB checkpoint never passes through
 * the API container. */
export function getTrack2TrainingAuditWeightsUrl(
  auditId: string,
): Promise<{ url: string; uri: string }> {
  return apiGet(`/track2/models/audit/${encodeURIComponent(auditId)}/weights-url`);
}

export interface Track2DiscoveredChild {
  name: string;
  images_prefix: string;
  labels_prefix: string;
  image_count: number;
  label_count: number;
  paired: number;
}

export interface Track2Discovery {
  root: string;
  images_subpath: string;
  labels_subpath: string;
  children: Track2DiscoveredChild[];
  /** Those actually holding images -- a partially uploaded week shows up
   * here as missing rather than as a silent zero. */
  usable: string[];
}

/** Every child folder of a root, expanded into image/label pairs. Show it
 * one week's shape and it finds the rest, so a week added later needs no
 * new configuration. */
export function discoverTrack2Dataset(
  root: string,
  imagesSubpath: string,
  labelsSubpath: string,
): Promise<Track2Discovery> {
  return apiPost("/track2/models/dataset-discover", {
    root, images_subpath: imagesSubpath, labels_subpath: labelsSubpath,
  });
}

export function getTrack2DatasetLayout(
  folders: Track2DatasetRequest,
): Promise<Track2DatasetLayout> {
  return apiPost("/track2/models/dataset-layout", folders);
}


/** One data.yaml naming the folders where they already are. Copies nothing.
 *
 * The trainer cannot read s3://: it downloads ONE prefix to local disk and
 * only then opens the yaml. So every folder named has to sit under one
 * common parent, and that parent is what the run downloads -- all of it,
 * not just the folders named. `download.extra_objects` is everything under
 * the prefix that is NOT in a chosen folder, which is what the run pays
 * for and does not train on. */
export interface Track2DatasetYamlInPlace {
  folder: string;
  key: string;
  bucket: string;
  uri: string;
  download_prefix: string;
  train: string[];
  val: string[];
  names: Record<string, string>;
  text: string;
  existing: { bytes?: number; ours?: boolean };
  /** A data.yaml is already there and this card did not write it. */
  needs_overwrite: boolean;
  /** Sources whose label files use a class the yaml does not name. The
   * preview reports these; the write refuses them unless
   * `drop_undeclared` was sent. */
  undeclared?: { entry: string; indices: number[]; sampled: number; total: number; text: string }[];
  /** Images per folder the yaml names, per split. `images` null when a
   * folder could not be listed. */
  image_counts?: Record<"train" | "val", {
    /** `retained_of`: on the retention list's row, how many images the
     * week folders hold -- `images` is how many the decay keeps. */
    folders: {
      folder: string; images: number | null; retained_of?: number | null;
      /** On the retention list's row: each week folder it draws from. */
      weeks?: { folder: string; keep: number; available: number }[];
    }[];
    total: number | null;
  }>;
  drop_undeclared?: boolean;
  /** True when the yaml names folders under several prefixes. */
  many_roots?: boolean;
  /** The SECOND download, when validation lives under another prefix.
   *
   * Training and validation are separate downloads -- the training job
   * takes TRAIN_DATA_S3_URI, the evaluation step takes EVAL_DATA_S3_URI
   * and runs in a different container -- so the validation set can live
   * anywhere. Null when one prefix reaches both. */
  eval: null | {
    folder: string; key: string; uri: string;
    train: string[]; val: string[]; text: string;
    download: Track2DatasetYamlInPlace["download"];
  };
  download: {
    prefix: string;
    in_named_folders: number;
    in_named_bytes: number;
    extra_objects: number;
    extra_bytes: number;
  };
  written?: boolean;
  /** Every file written -- one per download. */
  written_uris?: string[];
}

export function previewTrack2DatasetYaml(
  folders: Track2DatasetRequest,
  classes: Record<string, string>,
  overwrite = false,
  dropUndeclared = false,
  /** Add Pull Control's week folders, as the run does when it is ticked. */
  includePull = false,
): Promise<Track2DatasetYamlInPlace> {
  return apiPost("/track2/models/dataset-yaml-preview",
    { ...folders, classes, overwrite, drop_undeclared: dropUndeclared,
      include_pull: includePull });
}

export function writeTrack2DatasetYaml(
  folders: Track2DatasetRequest,
  classes: Record<string, string>,
  overwrite = false,
  dropUndeclared = false,
): Promise<Track2DatasetYamlInPlace> {
  return apiPost("/track2/models/dataset-yaml-write",
    { ...folders, classes, overwrite, drop_undeclared: dropUndeclared });
}

export function getTrack2DatasetClasses(folder: string): Promise<Track2DatasetClasses> {
  return apiGet(`/track2/models/dataset-classes?folder=${encodeURIComponent(folder)}`);
}

/** Folders AND the .yaml files beside them, in ANY readable bucket.
 *
 * `listTrack2Folders` returns folders only, which is right for picking a
 * dataset root. Reading class names needs the file itself -- and the
 * datasets are often in a bucket that is not this app's. */
export function listTrack2YamlFiles(
  bucket: string,
  prefix: string,
): Promise<{ bucket: string; prefix: string; folders: string[]; yaml_files: string[] }> {
  return apiPost("/track2/pull-control/yaml-files", { bucket, prefix });
}

/** The class names one yaml declares, from any readable bucket. */
export function readTrack2YamlClasses(
  bucket: string,
  key: string,
): Promise<{ bucket: string; key: string; classes: Record<string, string> }> {
  return apiPost("/track2/pull-control/yaml-classes", { bucket, key });
}

/** Class names out of a data.yaml the operator has on their machine.
 *
 * The file's TEXT goes to the server, which parses it. Not a YAML reader
 * in the browser: this codebase already carries two hand-rolled readers
 * of this exact key, and a third is the one that drifts -- with a
 * silently shifted class index as the failure. Nothing is uploaded to
 * S3; reading class names is not a reason to leave a file in the
 * bucket. */
export function readTrack2ClassesFromYamlText(
  text: string,
  filename: string,
): Promise<Track2DatasetClasses> {
  return apiPost("/track2/models/dataset-classes/from-text", { text, filename });
}


export interface Track2Detector {
  slug: string;
  label: string;
  /** The built-in PPE detector owns the legacy `yolo11{size}-detector`
   * groups every existing version is registered under, so it cannot be
   * edited or deleted. */
  builtin: boolean;
  classes: Record<string, string>;
  /** size -> registry group name, generation 11. Kept flat because every
   * existing caller reads it that way; `groups_by_generation` is the one
   * to use when the generation is in play. */
  groups: Record<string, string>;
  /** generation -> size -> registry group name. */
  groups_by_generation?: Record<string, Record<string, string>>;
  /** Which generations this detector has registries for. The built-in is
   * YOLO11 only, deliberately -- see detector_generations server-side. */
  generations?: string[];
}

/** One YOLO generation a new model may be created as. */
export interface Track2Generation {
  generation: string;
  label: string;
  sizes: string[];
  size_labels: Record<string, string>;
}

export interface Track2QueueAfterPull {
  settings: {
    enabled: boolean;
    even_if_nothing_moved: boolean;
    /** Which pull triggers may chain. `schedule` only, by default.
     *
     * `arrival` fires per upload burst, so it is paired with the two
     * guards below: on 2026-10-06 three arrival pulls fired in ten
     * minutes, one of them moving a single file, and each would have
     * been a GPU training run. */
    triggers?: string[];
    /** Files a sweep must have moved to count as a batch worth training. */
    min_files?: number;
    /** Shortest gap between two chains, in minutes. 0 means no wait. */
    cooldown_minutes?: number;
    last_run_id: string;
    last_result: {
      run_id?: string; at?: string; ok?: boolean; detail?: string; moved?: number;
    };
  };
  next: {
    act: boolean;
    reason: string;
    run: { run_id?: string; moved?: number; finished_at?: string; trigger?: string } | null;
  };
}

export function getTrack2QueueAfterPull(): Promise<Track2QueueAfterPull> {
  return apiGet("/track2/retention/queue-after-pull");
}

export function saveTrack2QueueAfterPull(
  patch: {
    enabled?: boolean; even_if_nothing_moved?: boolean;
    triggers?: string[]; min_files?: number; cooldown_minutes?: number;
  },
): Promise<Track2QueueAfterPull> {
  return apiPost("/track2/retention/queue-after-pull", patch);
}

export interface Track2WeekPurgePreview {
  bucket: string;
  /** `<root>/<week>/` -> object count. A week lives in every root a
   * source files it into, so this is not one folder per week. */
  prefixes: Record<string, number>;
  objects: number;
  bytes: number;
}

export function previewTrack2WeekPurge(week = ""): Promise<Track2WeekPurgePreview> {
  // `week` narrows it to one. The per-row purge has to send the exact
  // object count, and it cannot send a number nobody ever counted.
  return apiGet("/track2/pull-control/weeks/purge-preview"
    + (week ? `?name=${encodeURIComponent(week)}` : ""));
}

/** PERMANENTLY deletes every object under the week folders. There is no
 * second copy -- the pull MOVES files out of the drop folders. The
 * server re-counts and refuses unless `confirmObjects` still matches, so
 * a sweep landing between the preview and the press cannot widen what
 * gets destroyed. */
export function purgeTrack2Weeks(
  confirmObjects: number,
  names: string[] = [],
): Promise<{
  deleted: number; errors: string[]; message: string;
  weeks_forgotten: number;
}> {
  // `/weeks/purge-all`, NOT `/weeks/purge` -- that path already belongs
  // to the single-week delete, which sends {name} and no confirm. Two
  // handlers on one path means the first registered wins, and the row
  // button would 422.
  return apiPost("/track2/pull-control/weeks/purge-all", {
    confirm_objects: confirmObjects, names,
  });
}

export function getTrack2Detectors(): Promise<{
  detectors: Track2Detector[];
  sizes: string[];
  generations?: Track2Generation[];
  default_generation?: string;
}> {
  return apiGet("/track2/models/detectors");
}

export function saveTrack2Detector(
  slug: string,
  label: string,
  classes: Record<string, string>,
): Promise<{ slug: string; saved: boolean }> {
  return apiPost("/track2/models/detectors", { slug, label, classes });
}

/**
 * Give a Model Name a new identifier, and point the workflow at it.
 *
 * The identifier derives the registry group name, and a model package
 * group CANNOT be renamed -- SageMaker has create, delete, describe and
 * list, and no update. So future runs register into the new groups and
 * the old identifier is remembered, which keeps the versions already
 * registered findable rather than moving them.
 *
 * `cascaded` lists what now follows the new name -- the training form's
 * selection, the Assemble and Import cards, the unattended retrain, and
 * any queued runs. `kept_groups` are the old registries, named so nobody
 * goes looking for versions that are exactly where they always were.
 */
export function renameTrack2Detector(
  slug: string,
  newSlug: string,
  label: string,
): Promise<{
  detector: {
    slug: string; label: string;
    classes: Record<string, string>; previous_slugs: string[];
  };
  cascaded: string[];
  groups: string[];
  kept_groups: string[];
}> {
  return apiPost(`/track2/models/detectors/${encodeURIComponent(slug)}/rename`,
                 { new_slug: newSlug, label });
}

/** `force` takes the detector's registered versions with it. Off by
 * default: that is not recoverable. */
export function deleteTrack2Detector(
  slug: string,
  force = false,
): Promise<{ slug: string; deleted: boolean; versions_removed: number }> {
  return apiPost(
    `/track2/models/detectors/${encodeURIComponent(slug)}/delete?force=${force}`,
  );
}

export function getTrack2Architectures(): Promise<{ architectures: string[] }> {
  return apiGet("/track2/models/base-models/architectures");
}

/** Uploads the checkpoint THROUGH this app and returns what it actually
 * is. Same-origin, so the bucket's CORS policy never applies -- the
 * presigned PUT below cannot be used from this UI because track2-s3
 * allows only *.sagemaker.aws origins. */
export function uploadTrack2BaseModelViaApi(
  file: File,
  architecture: string,
): Promise<Track2CheckpointInspection> {
  const body = new FormData();
  body.append("architecture", architecture);
  body.append("file", file);
  return apiPostForm("/track2/models/base-models/upload", body);
}


/** Reads the uploaded checkpoint back and reports what it actually is, so
 * a wrong declaration is caught at upload rather than following the model
 * through the registry. */
export function verifyTrack2BaseModel(
  key: string,
  architecture: string,
): Promise<Track2CheckpointInspection> {
  return apiPost("/track2/models/base-models/verify", { key, architecture });
}

export interface Track2ImportedModel {
  model_package_arn: string;
  group_name: string;
  version: string;
  artifact_uri: string;
  detected_architecture: string;
  classes: Record<string, string>;
}

/** Registers an already-uploaded checkpoint as a version in a detector's
 * registry, so an imported baseline is tracked and reusable the same way
 * a trained one is rather than living as a loose S3 key. */
/** A checkpoint inspected in the import card and waiting to be filed,
 * while the Class it will be filed under is being defined by hand.
 *
 * Carried through the page rather than duplicated: the import card is
 * the only thing that knows what was inspected, and "Define a new Class"
 * is the only thing that knows the Class about to exist. */
export interface Track2PendingImport {
  key: string;
  architecture: string;
  description: string;
  /** What the WEIGHTS detect, in index order. Shown beside the classes
   * being typed, because the two differing is the normal case here and
   * the reason the version is registered as a seed. */
  checkpointClasses: string[];
}

/** An official YOLO release mirrored into the pipeline bucket. */
export interface Track2LibraryWeight {
  architecture: string;
  key: string;
  size_bytes: number;
}

/** What the import card's third source offers.
 *
 * Reads what is ACTUALLY in the folder rather than listing what ought to
 * be there: a card promising `yolo26l.pt` when its upload failed sends
 * somebody looking for a file that does not exist. */
export function getTrack2WeightsLibrary(): Promise<{
  weights: Track2LibraryWeight[];
  prefix: string;
}> {
  return apiGet("/track2/models/base-models/library");
}

export function registerTrack2BaseModel(
  key: string,
  detectorSlug: string,
  architecture: string,
  notes: string,
  /** File it under a Class whose class names it does NOT detect.
   *
   * The deliberate exception, not the default. Training rebuilds the head
   * for whatever the dataset declares, so a COCO checkpoint really can be
   * the first version of a three-class Class -- but until that run the
   * weights and the registry disagree, so the server marks the version a
   * seed in its description and registers it Pending rather than
   * Approved. */
  allowClassMismatch = false,
  /** Register only these of the checkpoint's classes, by their ORIGINAL
   * index in the file.
   *
   * A strict subset makes the server cut the head down to them first, so
   * the version detects exactly what the Model Name declares and is
   * deployable immediately rather than a seed waiting on a training run.
   * The kept classes predict bit-for-bit what they predicted before --
   * nothing is learned or lost, outputs are simply removed. */
  keepClasses: number[] = [],
): Promise<Track2ImportedModel> {
  return apiPost("/track2/models/base-models/register", {
    key, detector_slug: detectorSlug, architecture, notes,
    keep_classes: keepClasses,
    allow_class_mismatch: allowClassMismatch,
  });
}


export function stopTrack2Training(
  includeForeign = false,
): Promise<{ message: string; stopped?: string[]; problems?: string[] }> {
  // OFF unless asked for, and asked for in a dialog that names each job:
  // those belong to somebody else, and "Stop Training" meaning "stop
  // everything on the account's GPU" is not what the button says.
  return apiPost("/track2/models/train/stop", { include_foreign: includeForeign });
}

export function addTrack2Queue(params: Track2TrainingParams): Promise<{ message: string; entry: Track2QueueEntry }> {
  return apiPost("/track2/models/queue/add", params);
}

export function removeTrack2Queue(entryId: string): Promise<{ removed: boolean }> {
  return apiPost(`/track2/models/queue/remove/${encodeURIComponent(entryId)}`);
}

export function startTrack2Queue(): Promise<{ message: string }> {
  return apiPost("/track2/models/queue/start");
}

export function stopTrack2Queue(): Promise<{ message: string }> {
  return apiPost("/track2/models/queue/stop");
}

// --- Schedule ----------------------------------------------------------------

export interface Track2ScheduleResponse {
  schedule: { enabled: boolean; expression: string };
  training_progress: Track2TrainingProgress | null;
}

export function getTrack2Schedule(): Promise<Track2ScheduleResponse> {
  return apiGet<Track2ScheduleResponse>("/track2/schedule");
}

export function trainTrack2Now(): Promise<{ message: string }> {
  return apiPost("/track2/schedule/train-now");
}

export interface Track2ScheduleRequest {
  mode: "manual" | "daily" | "weekly" | "monthly";
  daily_hour?: number;
  weekly_day?: string;
  weekly_hour?: number;
  monthly_day?: number;
  monthly_hour?: number;
}

export function setTrack2Schedule(payload: Track2ScheduleRequest): Promise<{ message: string; expression?: string }> {
  return apiPost("/track2/schedule", payload);
}

// --- Lineage / Metrics ---------------------------------------------------------

export function getTrack2Lineage(): Promise<{ records: Track2Lineage[] }> {
  return apiGet("/track2/lineage");
}

export interface Track2MetricsRow {
  group_name: string;
  /** Which registry this version lives in: "production" for the per-size
   * production groups, "custom" for the shared open-taxonomy ones, or a
   * DETECTOR SLUG for a version in that detector's own per-size group.
   * Not a closed union -- a detector is created at runtime, so any slug
   * can appear here. */
  registry: string;
  /** "11", "26", ... -- read off the group name server-side, "" for a
   * group whose name does not state one (the legacy custom group). The
   * page used to print the literal "YOLO11" in five places, so a YOLO26
   * version charted, labelled and tooltipped as a YOLO11 one. */
  generation: string;
  size: string;
  version: string;
  model_package_arn: string;
  status: string;
  metrics: {
    mAP50: number | null;
    "mAP50-95": number | null;
    precision: number | null;
    recall: number | null;
    /** Reported by the current evaluators; `accuracy` is the legacy
     * evaluator's equivalent and is still read for older versions. */
    f1: number | null;
    accuracy: number | null;
    /** The class names this version actually detects, in index order.
     * Empty for older versions whose evaluation report carried neither a
     * `classes` map nor `per_class`. */
    classes?: string[];
    /** Where this version's training images came from. `source_uri` is the
     * production pipeline's pinned snapshot or the open pipeline's chosen
     * folder; empty for legacy reports that recorded neither. */
    dataset?: {
      source_uri: string;
      dataset_version: string;
      provenance: string;
      validation_count: number | null;
      baseline_uri: string;
    };
  } | null;
  description: string;
  lineage: Track2Lineage | null;
}

export function getTrack2Metrics(): Promise<{ rows: Track2MetricsRow[] }> {
  return apiGet("/track2/metrics");
}


/** An export preset, saved on the EC2 host.
 *
 * Its own file list, NOT the pull-control settings. That one holds a
 * whole retraining configuration and loading it rearms a schedule; this
 * holds which Class, which criterion and where the checkpoint goes. One
 * dropdown for both would make Load a button whose consequences depend
 * on which name you happened to pick.
 *
 * The automatic export's armed state is deliberately NOT in here. Arming
 * is its own call, so loading a preset named after something else cannot
 * quietly start writing to a shared bucket on a watch. */
export interface Track2ExportSettings {
  /** The three dropdowns, stored so a preset restores the SELECTION it
   * was saved against and not merely the group that selection resolved
   * to. "Class helmet, YOLO26, size M" is what somebody chose; the group
   * name is what it means. */
  registry: string;
  generation: string;
  size: string;
  /** What those three resolve to, and the only field the server acts on. */
  group_name: string;
  /** Which metric decides the best version, or "newest". Part of what
   * the export IS -- the same Class exported by mAP50 and by recall are
   * two different files. */
  criterion: Track2ExportCriterion;
  /** Where it goes, mutually exclusive by construction. The same
   * checkpoint downloaded to a laptop and written to a shared bucket are
   * two different jobs. */
  destination: "local" | "s3";
  bucket: string;
  prefix: string;
  /** What happens to EARLIER exports of this Class in that folder once
   * the new one lands. A retention decision, so it rides in the preset:
   * restoring where an export goes without restoring what it does to
   * what is already there would reproduce half the job. */
  retention: Track2ExportRetention;
}

/** `keep_all` leaves them -- the folder is the history. `archive_oldest`
 * moves them into `archive/` under the same folder, so a consumer
 * listing it sees one file and nothing is lost. `keep_latest` deletes
 * them, which is the only irreversible one.
 *
 * All three are scoped to THIS Class's own exports in that folder, never
 * to the folder's other contents -- see `getTrack2ExportFolder`, which
 * lists exactly what they would act on. */
export type Track2ExportRetention = "keep_all" | "archive_oldest" | "keep_latest";

/** The earlier exports of one Class already sitting in a folder.
 *
 * The blast radius of the two housekeeping modes, listed before either
 * runs. A button whose reach you have to infer from its label is one
 * nobody should press. */
export function getTrack2ExportFolder(body: {
  group_name: string;
  bucket: string;
  prefix: string;
}): Promise<{ folder: string; files: string[] }> {
  return apiPost("/track2/metrics/export/folder", body);
}

/** Highest value wins; `newest` ranks on the version number instead --
 * a real answer for a Class trained on a growing corpus, where the newest
 * model saw the most data whatever it scored against a validation set
 * that also moved. */
export type Track2ExportCriterion =
  | "mAP50" | "mAP50-95" | "precision" | "recall" | "f1" | "accuracy" | "newest";

/** The version a criterion picks, stated in full.
 *
 * `validation_images` and `status` are here because nothing EXCLUDES a
 * version on them -- that was a deliberate choice -- so they have to be
 * visible instead. On this estate three versions score 0.995 mAP50 on
 * five validation images; ranked on mAP50 with no floor, five images win
 * every time. */
export interface Track2ExportPick {
  version: string;
  model_package_arn: string;
  group_name: string;
  status: string;
  criterion: string;
  value: number | null;
  validation_images: number | null;
  classes: string[];
  thin_validation: boolean;
  /** Its weights detect something other than this Class -- a starting
   * point awaiting its first training run, not a model of this Class.
   * `criterion: "newest"` will pick one, so the fact travels with the
   * pick rather than being left in the description. */
  is_seed: boolean;
  label: string;
}

/** One registry group a model could land in, whether or not one has.
 *
 * The export card's dropdowns come from THESE, not from the versions
 * that exist -- otherwise a Class that has only ever trained at YOLO11-S
 * offers YOLO11 and S and nothing else, and the group a YOLO26-X run
 * will land in next week is unreachable. Which is exactly when you would
 * want it set up: armed in advance, the first version to appear there is
 * exported the moment it registers.
 *
 * Enumerated by the server, never composed here. Rebuilding
 * `yolo{gen}{size}-{slug}-detector` in the browser is how the Models
 * card once named the wrong registry for a YOLO26 run. */
export interface Track2ExportGroup {
  /** "production", "custom", or a Class slug. */
  registry: string;
  /** "" for the legacy custom group, whose name states neither. */
  generation: string;
  size: string;
  group_name: string;
  /** What an export of this group is named before the version and
   * timestamp: `yolo11-s-helmet`. From the server, because composing it
   * here produced `yolo11s-helmet` -- a key that never appears in the
   * bucket, printed on the line that says where the file will go. */
  export_stem: string;
}

export function getTrack2ExportGroups(): Promise<{ groups: Track2ExportGroup[] }> {
  return apiGet("/track2/metrics/export/groups");
}

/** Which version this criterion picks, without exporting it.
 *
 * Asked of the SERVER although the page holds every row already. Two
 * implementations of "best" -- one deciding what the card says, one
 * deciding what the button does -- is how a card comes to promise v4 and
 * ship v9. */
export function previewTrack2BestVersion(body: {
  group_name: string;
  criterion: string;
}): Promise<{
  group_name: string;
  criterion: string;
  considered: number;
  best: Track2ExportPick | null;
  reason: string;
}> {
  return apiPost("/track2/metrics/export/preview", body);
}

/** Export the best version's .pt, now.
 *
 * The bytes never pass through the API container: S3 is a server-side
 * copy, and a download comes back as a presigned link the browser follows
 * straight to S3. */
export function exportTrack2BestWeights(body: {
  group_name: string;
  criterion: string;
  destination: "local" | "s3";
  bucket?: string;
  prefix?: string;
  retention?: Track2ExportRetention;
}): Promise<{
  picked: Track2ExportPick;
  considered: number;
  destination: string;
  /** `s3://bucket/key`, when that is where it went. */
  written_to: string;
  filename: string;
  /** A 15-minute link straight to S3, when the destination was local. */
  url: string;
  /** What it then did about the earlier exports. */
  retention: { mode: string; archived: string[]; deleted: string[] };
}> {
  return apiPost("/track2/metrics/export/weights", body);
}

export interface Track2AutoExport {
  enabled: boolean;
  group_name: string;
  criterion: string;
  bucket: string;
  prefix: string;
  retention: Track2ExportRetention;
  /** The highest version already seen. Set to the group's newest at
   * ARMING time, so switching it on does not export a version that
   * registered weeks ago. */
  last_version: number;
  /** The version it last actually wrote out. A new version registering
   * does not mean the winner changed, so this and `last_version` differ
   * whenever a run produced something that did not beat the old best. */
  last_exported_version: string;
  last_result: {
    at?: string;
    ok?: boolean;
    detail?: string;
    written_to?: string;
    /** A new version registered but did not beat the current best, so
     * nothing was written. A real outcome, not a silence. */
    skipped?: boolean;
    version?: string;
    status?: string;
    validation_images?: number | null;
    thin_validation?: boolean;
    trigger_version?: number;
  };
  armed_at: string;
}

/** One version as the export card states it: its score on the measure
 * the watcher ranks by, and how many images that score rests on. */
export interface Track2ExportScore {
  version: string;
  value?: number | null;
  validation_images?: number | null;
  thin_validation?: boolean;
}

export interface Track2ExportCounts {
  in_folder: number;
  archived: number;
  /** The newest .pt in the folder -- the version a consumer gets now. */
  latest?: (Track2ExportScore & { file: string; stamp: string }) | null;
  /** The group's best by the watched measure, right now. */
  best?: Track2ExportScore | null;
  criterion?: string;
}

export function getTrack2AutoExport(): Promise<{
  settings: Track2AutoExport;
  pending: { act: boolean; reason: string; version: number };
  /** This Model Name's .pt files in the folder, and in its archive/. */
  counts?: Track2ExportCounts | null;
}> {
  return apiGet("/track2/metrics/export/auto");
}

export function armTrack2AutoExport(body: {
  group_name: string;
  criterion: string;
  bucket: string;
  prefix: string;
  retention: Track2ExportRetention;
}): Promise<{ settings: Track2AutoExport }> {
  return apiPost("/track2/metrics/export/auto/arm", body);
}

export function disarmTrack2AutoExport(): Promise<{ settings: Track2AutoExport }> {
  return apiPost("/track2/metrics/export/auto/disarm", {});
}

export interface Track2ExportSettingsFile {
  name: string;
  bytes: number;
  modified: string;
}

export function listTrack2ExportSettings(): Promise<{
  directory: string;
  files: Track2ExportSettingsFile[];
  writable: boolean;
}> {
  return apiGet("/track2/metrics/export/settings");
}

export function saveTrack2ExportSettings(
  name: string,
  settings: Track2ExportSettings,
): Promise<{ name: string; path: string; bytes: number }> {
  return apiPost("/track2/metrics/export/settings/save", { name, settings });
}

export function loadTrack2ExportSettings(name: string): Promise<{
  name: string;
  saved_at: string;
  settings: Track2ExportSettings;
}> {
  return apiPost("/track2/metrics/export/settings/load", { name });
}

export function deleteTrack2ExportSettings(
  name: string,
): Promise<{ name: string; deleted: boolean }> {
  return apiPost("/track2/metrics/export/settings/delete", { name });
}

/** The 8-color CVD-safe palette the old app used for the metrics
 * comparison chart -- kept exact for continuity with historical charts. */
/** The comparison series colours, in FIXED ORDER -- slot 1 is always
 * blue, never "whatever is first after a filter".
 *
 * FOUR, not eight, and these four. The previous list was the dark-mode
 * column of the same ramps used on a light surface, and in a grouped bar
 * chart -- where every pair sits side by side, so every pair must be
 * separable -- it failed outright:
 *
 *   #199e70 vs #d55181   dE 1.6 under protan/deutan   (floor 6, target 8)
 *   #d95926 vs #c98500   dE 10.6 for normal vision    (hard floor 15)
 *
 * Green and pink were the same colour to a red-green colourblind reader,
 * and orange and amber were near-identical to everyone. Stepping to the
 * light column fixed the first and not the second; five slots could not
 * be made to pass at all, so the count came down to four:
 *
 *   CVD separation   worst dE 9.2  (>= 8)   PASS
 *   Normal vision    worst dE 16.3 (>= 15)  PASS
 *   Contrast         #eb6834 2.6:1, #1baf7a 2.29:1 vs this app's
 *                    #efe7d4 surface -- a WARN, which obliges visible
 *                    labels. Every bar carries its value above it and
 *                    the picker lists the exact numbers, so it is met.
 *
 * Computed, not judged: scripts/validate_palette.js in the dataviz
 * skill, run against surface #efe7d4 with --pairs all. Re-run it before
 * changing any value here.
 */
export const TRACK2_COMPARE_COLORS = [
  "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7",
];

// --- Deploy / Inference / Export ---------------------------------------------

export type Track2DeployWeightsKind = "registry" | "s3";

export interface Track2DeployedComponent {
  /** Which size (s/m/l/x) is currently deployed -- read back off the
   * deployed Model resource's own name. */
  active_size: string | null;
  status: string;
  model_name: string;
  last_modified: string | null;
  /** Friendly label ("YOLO11-S v3" or "S3: weights/s/2026-09-07") for
   * whatever's actually deployed here, read back off a tag on the
   * SageMaker Model resource -- durable across an app restart. `null`
   * only for a component deployed before this tagging existed. */
  source: string | null;
  /** Bare registry version number ("12"), shown distinctly from `source`
   * next to the raw model_name. `null` for an S3-sourced deploy (there's
   * no registry version to point at) or a pre-tagging deploy. */
  version: string | null;
  /** 0 means scaled to zero (idle 30+ min) -- queuing (and starting) a
   * new deploy is the only way back up, since every deploy requests
   * CopyCount: 1 regardless of the count beforehand. */
  current_copy_count: number | null;
  desired_copy_count: number | null;
  /** True while this app itself has an invoke_inference call in flight
   * (in practice, mid-way through an eval run) -- it can't see
   * inference traffic from anywhere else. */
  running_inference: boolean;
}

export type Track2InferenceQueueItemStatus = "queued" | "deploying" | "running" | "done" | "failed";

export interface Track2InferenceQueueItem {
  id: string;
  status: Track2InferenceQueueItemStatus;
  size: string;
  weights_kind: Track2DeployWeightsKind;
  weights_label: string;
  model_package_arn: string | null;
  weights_folder: string | null;
  weights_key: string | null;
  /** Only ever set for weights_kind="s3" -- a free-text description of
   * the checkpoint, since it has no metadata of its own the way a
   * registered version does. */
  notes: string | null;
  /** Applies regardless of weights_kind -- a free-text note about this
   * particular run, not about the checkpoint. */
  comment: string | null;
  images_folder: string;
  output_folder: string;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  result: Track2BatchInferenceResult | null;
  error: string | null;
}

export interface Track2EndpointStatus {
  /** "InService" | "Creating" | "Deleting" | "Updating" | "Failed" ... */
  status: string;
  instance_type: string;
  instance_count: number | null;
  created_at: string | null;
}

export interface Track2DeployResponse {
  registries: Track2RegistriesBrief;
  /** Only one model is ever live at a time -- deploying any size
   * replaces whatever was here before. `null` if nothing's deployed yet. */
  deployed: Track2DeployedComponent | null;
  /** null when the endpoint has been torn down -- the normal resting
   * state between runs, since it bills per instance-hour while it exists. */
  endpoint: Track2EndpointStatus | null;
  folders: string[];
  /** .pt/.tar.gz files directly under browse_prefix -- lets a picker
   * select one specific checkpoint file instead of assuming a folder
   * holds a fixed "model.pt". */
  weight_files: string[];
  inference_queue: Track2InferenceQueueItem[];
  inference_queue_status: { started: boolean; worker_alive: boolean };
}

export function getTrack2Deploy(browsePrefix = ""): Promise<Track2DeployResponse> {
  const q = new URLSearchParams();
  if (browsePrefix) q.set("browse_prefix", browsePrefix);
  return apiGet(`/track2/deploy?${q}`);
}

export interface Track2CostItem {
  usage_type: string;
  amount: number;
}

export interface Track2CostDay {
  /** Day the cost was incurred, YYYY-MM-DD. */
  date: string;
  total: number;
  items: Track2CostItem[];
}

export interface Track2CostResponse {
  currency: string;
  start: string;
  end: string;
  total: number;
  days: Track2CostDay[];
  by_usage_type: Track2CostItem[];
  fetched_at: string;
}

/** Cost Explorer charges $0.01 per request, so the backend caches this and
 * nothing here polls it. Pass refresh to force a fresh read -- that is what
 * the Cost tab's Refresh button does, and the only path that spends. */
export function getTrack2Cost(days?: number, refresh = false): Promise<Track2CostResponse> {
  const q = new URLSearchParams();
  if (days) q.set("days", String(days));
  if (refresh) q.set("refresh", "true");
  return apiGet(`/track2/cost?${q}`);
}

/** Cost Explorer usage types read "<region>-<category>:<detail>", e.g.
 * "APS1-Host:ml.m5.2xlarge", "APS1-Train:ml.g4dn.xlarge",
 * "APS1-Processing:ml.m5.xlarge". The separator before the category is a
 * HYPHEN -- an earlier filter here looked for ":Host:", which never occurs
 * in any usage type, so every hosting figure in the UI summed to exactly
 * zero. Parse the category out rather than substring-matching it. */
function usageCategory(usageType: string): string {
  const prefix = usageType.split(":")[0];
  const dash = prefix.lastIndexOf("-");
  return (dash >= 0 ? prefix.slice(dash + 1) : prefix).toLowerCase();
}

// "Hst" is how Cost Explorer labels endpoint data transfer
// ("APS1-Hst:Data-Bytes-Out") -- same endpoint, same bill.
const HOSTING_CATEGORIES = new Set(["host", "hst"]);
// A training run is the GPU training job plus the Processing job that
// evaluates it; both are the cost of running the pipeline once.
const TRAINING_CATEGORIES = new Set(["train", "processing"]);

function sumByCategory(items: Track2CostItem[], categories: Set<string>): number {
  return items
    .filter((i) => categories.has(usageCategory(i.usage_type)))
    .reduce((sum, i) => sum + i.amount, 0);
}

/** The standing cost of the inference endpoint for a cost day. */
export function track2HostingTotal(items: Track2CostItem[]): number {
  return sumByCategory(items, HOSTING_CATEGORIES);
}

/** What training cost for a cost day -- GPU training jobs plus the
 * evaluation Processing jobs that run with them. */
/** Today's training spend measured from the JOBS rather than from Cost
 * Explorer, which lags roughly a day and so answers $0.00 on the
 * afternoon of a day spent training. */
export interface Track2CostToday {
  date: string;
  currency: string;
  estimated: number;
  /** False when a rate or a job could not be read, so the figure is a
   * floor rather than a total. */
  complete: boolean;
  training_seconds: number;
  processing_seconds: number;
  /** The same total, split by whose job it was. A day's spend that is
   * mostly somebody else's is a different thing to know than a day's
   * spend that is all ours, and one number cannot say it. */
  estimated_ours?: number;
  estimated_other?: number;
  jobs: {
    name: string; kind: "training" | "processing"; instance_type: string;
    seconds: number; hours: number; rate: number; cost: number; status: string;
    /** Created by one of this app's pipeline steps. The panel lists
     * everything on the account, because the bill charges for
     * everything -- but the row has to say which is which. */
    own_pipeline?: boolean;
    /** When it ran, ISO with the offset so the browser renders it in the
     * reader's own zone. Null while a job is still waiting on capacity:
     * billing starts at `started_at`, and a job that never got an
     * instance has neither. */
    started_at?: string | null;
    ended_at?: string | null;
  }[];
}

export function getTrack2CostToday(): Promise<Track2CostToday> {
  return apiGet("/track2/cost/today");
}

export function track2TrainingTotal(items: Track2CostItem[]): number {
  return sumByCategory(items, TRAINING_CATEGORIES);
}

export interface Track2RunLineage {
  bucket: string;
  /** Folder the images were read from. */
  images_folder: string;
  output_folder: string;
  /** Exact file names that went through the model on this run. */
  files: string[];
  model: {
    /** YOLO size: s | m | l | x. */
    size: string | null;
    weights_kind: string | null;
    weights_label: string | null;
    /** Registry version number, null for S3-sourced weights. */
    version: string | null;
    model_package_arn: string | null;
    weights_folder: string | null;
    weights_key: string | null;
    notes: string | null;
  };
  inference_started_at: string | null;
  inference_completed_at: string | null;
  generated_at: string;
}

/** Lineage for one run: DynamoDB by run id, falling back to the output
 * folder's own lineage.json. 404s for runs that predate lineage being
 * recorded, or that failed before inference started. */
export function getTrack2RunLineage(outputFolder: string, runId?: string): Promise<Track2RunLineage> {
  const q = new URLSearchParams({ output_folder: outputFolder });
  if (runId) q.set("run_id", runId);
  return apiGet(`/track2/deploy/lineage?${q}`);
}

export function deleteTrack2Endpoint(): Promise<{ deleted: boolean; reason: string | null }> {
  return apiPost("/track2/deploy/endpoint/delete");
}

export interface Track2DeployBrowseResponse {
  folders: string[];
  weight_files: string[];
}

/** Folder-browser navigation only. Deliberately NOT getTrack2Deploy: that
 * one also rebuilds the registry + deployed-component state (~30 serial
 * AWS calls), which made every click into a folder take over a second. */
export function getTrack2DeployBrowse(browsePrefix = ""): Promise<Track2DeployBrowseResponse> {
  const q = new URLSearchParams();
  if (browsePrefix) q.set("prefix", browsePrefix);
  return apiGet(`/track2/deploy/browse?${q}`);
}

export interface Track2BatchInferenceResult {
  processed: number;
  total: number;
  output_folder: string;
  errors: string[];
}

export interface QueueInferenceRunParams {
  size: string;
  weights_kind: Track2DeployWeightsKind;
  images_folder: string;
  output_folder: string;
  model_package_arn?: string;
  /** For weights_kind="s3": exactly one of these two. */
  weights_folder?: string;
  weights_key?: string;
  /** For weights_kind="s3" only -- a free-text description of the checkpoint. */
  notes?: string;
  /** Applies regardless of weights_kind -- a free-text note about this run. */
  comment?: string;
}

export function addTrack2InferenceQueueItem(
  params: QueueInferenceRunParams,
): Promise<{ message: string; item: Track2InferenceQueueItem }> {
  return apiPost("/track2/deploy/inference-queue/add", params);
}

export function removeTrack2InferenceQueueItem(itemId: string): Promise<{ removed: boolean }> {
  return apiPost(`/track2/deploy/inference-queue/remove/${encodeURIComponent(itemId)}`);
}

export function startTrack2InferenceQueue(): Promise<{ message: string }> {
  return apiPost("/track2/deploy/inference-queue/start");
}

export function stopTrack2InferenceQueue(): Promise<{ message: string }> {
  return apiPost("/track2/deploy/inference-queue/stop");
}

export interface Track2NotificationSubscription {
  subscription_arn: string;
  email: string;
  confirmed: boolean;
  /** Which kinds this subscriber receives. Null when it cannot be read --
   * an unconfirmed subscriber, whose choices were set when added. */
  events: string[] | null;
}

/** One kind of email a subscriber can choose. `kiv`: offered, not sent yet. */
export interface Track2NotificationEvent {
  key: string;
  label: string;
  kiv: boolean;
}

export function getTrack2Notifications(): Promise<{
  subscriptions: Track2NotificationSubscription[];
  events: Track2NotificationEvent[];
}> {
  return apiGet("/track2/notifications");
}

export function addTrack2Notification(
  email: string, events?: string[],
): Promise<{ email: string; message: string }> {
  return apiPost("/track2/notifications/add", { email, events });
}

export function setTrack2NotificationEvents(
  subscriptionArn: string, events: string[],
): Promise<{ subscription_arn: string; events: string[] }> {
  return apiPost("/track2/notifications/events", { subscription_arn: subscriptionArn, events });
}

export function removeTrack2Notification(subscriptionArn: string): Promise<{ removed: boolean }> {
  return apiPost("/track2/notifications/remove", { subscription_arn: subscriptionArn });
}

export interface Track2ExportResponse {
  registries: Track2RegistriesBrief;
  folders: string[];
}

/** When a sweep last finished. Polled by Auto Training so its week
 * folders appear after a pull without anybody pressing Reload -- for a
 * scheduled pull as much as a manual one, since nobody is watching the
 * page when the schedule fires. */
export interface Track2LastPull {
  finished_at: string;
  run_id: string;
  moved: number;
  status: string;
}

export function getTrack2LastPull(): Promise<Track2LastPull> {
  return apiGet("/track2/pull-control/last-pull");
}

export function getTrack2Export(browsePrefix = ""): Promise<Track2ExportResponse> {
  const q = new URLSearchParams();
  if (browsePrefix) q.set("browse_prefix", browsePrefix);
  return apiGet(`/track2/export?${q}`);
}

export function exportTrack2Weights(
  modelPackageArn: string,
  destinationFolder: string,
): Promise<{ message: string; new_key: string; archived_key: string | null }> {
  return apiPost("/track2/export", { model_package_arn: modelPackageArn, destination_folder: destinationFolder });
}

// ---------------------------------------------------------------------------
// Track2 evaluation queue (/track2/eval) -- run a registry version or your
// own S3 .pt weights against a chosen dataset, get back real detection
// metrics, persisted in DynamoDB so results survive a restart.
// ---------------------------------------------------------------------------

export type Track2EvalRunStatus = "queued" | "deploying" | "running" | "completed" | "failed";
export type Track2EvalWeightsKind = "registry" | "s3";

export interface Track2EvalMetrics {
  mAP50: number;
  "mAP50-95": number;
  precision: number;
  recall: number;
  accuracy: number;
  num_images: number;
  num_classes: number;
}

export interface Track2EvalRun {
  id: string;
  status: Track2EvalRunStatus;
  size: string;
  data_folder: string;
  weights_kind: Track2EvalWeightsKind;
  weights_label: string;
  model_package_arn: string | null;
  s3_folder: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  metrics: Track2EvalMetrics | null;
  error: string | null;
}

export interface Track2EvalResponse {
  runs: Track2EvalRun[];
  queue: { started: boolean; worker_alive: boolean };
  registries: Track2RegistriesBrief;
  generation_sizes: Record<string, string[]>;
  /** "11" -> "YOLO11". Sent rather than derived, so a new generation
   * carries whatever name it is actually given. */
  generation_labels?: Record<string, string>;
  size_labels: Record<string, string>;
}

export function getTrack2Eval(): Promise<Track2EvalResponse> {
  return apiGet("/track2/eval");
}

export interface QueueEvalRunParams {
  size: string;
  data_folder: string;
  weights_kind: Track2EvalWeightsKind;
  model_package_arn?: string;
  s3_folder?: string;
}

export function addTrack2EvalRun(params: QueueEvalRunParams): Promise<{ message: string; run: Track2EvalRun }> {
  return apiPost("/track2/eval/add", params);
}

export function removeTrack2EvalRun(runId: string): Promise<{ removed: boolean }> {
  return apiPost(`/track2/eval/remove/${encodeURIComponent(runId)}`);
}

export function startTrack2EvalQueue(): Promise<{ message: string }> {
  return apiPost("/track2/eval/start");
}

export function stopTrack2EvalQueue(): Promise<{ message: string }> {
  return apiPost("/track2/eval/stop");
}

// ---------------------------------------------------------------------------
// Pull control
//
// Files land in a flat drop folder; a Lambda files each one into the week
// folder its own modified time says it belongs to. These types mirror
// pull_control.py, which is also the Lambda's handler -- so the card
// previews the same rule that runs unattended.
// ---------------------------------------------------------------------------

export interface Track2PullSource {
  /** The watched drop folder. */
  prefix: string;
  /** Which root its files are filed under. */
  destination: string;
  /** The bucket it is READ from. Blank means the app's own bucket. The
   * escalation datasets are in safety-model-dataset-dev, read-only. */
  bucket?: string;
  /** Where the `val` side goes, if the source is already split. Blank
   * sends everything to `destination`. */
  eval_destination?: string;
  /** "copy" leaves the originals alone; "move" empties the drop folder. */
  action?: "copy" | "move";
  /** Which split this folder feeds. Blank infers it from the path; set
   * explicitly when the folder was picked by hand. */
  split?: "train" | "val" | "";
}

export interface Track2PullWeek {
  name: string;
  starts_at: string;
}

export interface Track2PullRule {
  name: string;
  exists: boolean;
  enabled: boolean;
  schedule: string;
  /** When it next fires, ISO. The cadence alone does not say WHICH day:
   * the grid is counted from the week cut-off, so a time of 15:43 set on
   * the 5th can arm the 7th at 15:43 -- and "armed: rate(7 days)" reads
   * as "today" to anyone who has just pressed Apply. Schedule only. */
  next_run?: string;
  error?: string;
}

export interface Track2PullConfig {
  /** The Auto Training card's own selection: the root it is pointed at
   * and the weeks ticked out of it. Server-side so it rides along in the
   * settings file -- a selection that only exists in a tab is one nobody
   * can restore. */
  /** Auto Training's own selection.
   *
   * `follow_pull` defaults TRUE: the root cascades from the pull
   * control's `root`, which is the folder the pull actually writes week
   * folders into, and is what the unattended chain reads. `root` is only
   * consulted once somebody has deliberately pointed the card elsewhere,
   * and the card says so on screen when they have -- a card quietly
   * looking at a different folder from the one the automated run uses is
   * not a disagreement anybody would think to check for. */
  auto_training?: { root?: string; excluded?: string[]; follow_pull?: boolean };
  /** Which cards are open. Saved with everything else so a loaded file
   * puts the page back the way it was left.
   *
   * One record for every card, because the server merges at the TOP
   * level: each card writing its own key would mean the rest being
   * dropped by whichever wrote last. See lib/cardState.ts. */
  card_state?: Record<string, boolean>;
  /** The Assemble card's own choices: the folders it merges, the
   * detector it writes classes for, whether it may replace a data.yaml
   * it did not write, and whether it is the ONLY source of training
   * data -- the tick that switches Pull Control, Auto Training and
   * Retention off.
   *
   * Server-side, like every other card's settings, so the one settings
   * file restores the whole page. This card shipped without it, so the
   * folders picked in it lived in a browser tab and a saved file did
   * not contain them. */
  /** The retention card's settings, which ride in this config because
   * `pull_control.DEFAULTS` holds them.
   *
   * READ ONLY from here. `ConfigRequest` does not declare it, so a
   * `saveTrack2PullConfig` carrying it would be dropped in silence --
   * `/track2/retention/...` is the route that writes it. Declared
   * because the training card needs `use_golden` to say which sources
   * a run is composed from, and one fetch it already makes is better
   * than a second round trip. */
  retention?: { use_golden?: boolean };
  dataset_assemble?: {
    only?: boolean;
    /** Whether this card's folders are one of the sources for the final
     * data.yaml. Absent means true: every configuration written before
     * the three source ticks existed was using its folders. */
    use?: boolean;
    /** What the LABEL FILES mean, every class in index order -- as
     * distinct from the Model Name, which is what to train. The yaml
     * declares these; the run filters to the Model Name's class at load
     * time. Absent means the Model Name's list is also the labels'. */
    label_classes?: Record<string, string>;
    /** The training container drops boxes whose class the yaml does not
     * name, from its downloaded copy only. S3 is never touched. */
    drop_undeclared_labels?: boolean;
    train_sources?: { images: string; labels: string }[];
    eval_sources?: { images: string; labels: string }[];
    folders?: Track2DatasetFolders;
    detector_slug?: string;
    overwrite?: boolean;
    /** Runs this card's build as the unattended chain's first step, in
     * place of the root-and-week-folders one. Read server-side by
     * `retrain_schedule.assemble_plan`; absent or false means the old
     * path, so a settings file written before this cannot start
     * assembling in place by surprise. */
    scheduled?: boolean;
    /** What this card last wrote a data.yaml for. The training card
     * offers it as "use the assembled dataset"; without it the folder
     * lived in a browser tab and that option was empty on every
     * reload. */
    built?: { folder: string; eval_folder?: string; at?: string } | null;
  };
  /** The Import Checkpoint card's STANDING choices -- not the file it
   * last read or what that registration returned, which belong to one
   * action rather than to the configuration. */
  import_checkpoint?: {
    architecture?: string;
    source?: string;
    bucket_key?: string;
    prefix?: string;
    detector_slug?: string;
    use_file_classes?: boolean;
    /** The key the last successful read used. The card re-reads it on
     * load rather than storing what it said -- a .pt's classes are a
     * property of the file, so a re-read cannot be stale, and the card
     * comes back without anybody pressing Read classes.
     *
     * Not `bucket_key`: after an upload the file is in the bucket, so
     * this restores all three sources including the one whose original
     * lives on somebody's laptop. */
    read_key?: string;
    /** Which classes were ticked. A CHOICE, not a result: "these four of
     * the eighty" is what the picker is for. */
    picked_classes?: string[];
    new_label?: string;
    description?: string;
  };
  /** Step 2's own choices. Server-side so the settings file holds them --
   * they used to be browser-tab state only, so a saved file recorded the
   * golden set's on/off switch and none of the folders it pointed at. */
  /** The SQL tab's choices, saved so a settings file brings them back. */
  sql_explorer?: {
    database?: string;
    bucket?: string;
    folder?: string;
    sql?: string;
  };
  golden_set?: {
    bucket?: string;
    source?: string;
    /** Where the labels are, when they are not under `source`. Empty for
     * the two layouts this card always read. */
    labels_source?: string;
    target_bucket?: string;
    target_folder?: string;
    split?: boolean;
    train_percent?: number;
  };
  /** What everything trains as, chosen once in "Register a model and
   * Classes". The unattended retrain always uses this; the training form
   * pre-selects from it but can be overridden for a single run. */
  default_detector?: { slug?: string; architecture?: string };
  /** The shape of every run, chosen in step 1: epochs and the rest.
   * Read by the training form AND by the scheduled retrain, so there is
   * one set of numbers rather than a copy in each. */
  /** The dataset the last automatic run built and froze -- folder,
   * snapshot id, file count and who set it. Read by the training card so
   * it opens pointing at what the schedule produced. */
  /** Which KIND of training: "existing" | "scratch" | "import". Saved
   * beside the model, because the two together are what defines a run. */
  default_source?: string;
  /** The model package ARN a run continues from -- which VERSION, as
   * opposed to which Class. Both are needed to define a run. */
  default_version?: string;
  /** "live" -- the week folders as they stand. "snapshot" -- a frozen
   * definition, the same images every time whatever has been pulled
   * since. Saved with the rest: two runs with identical folders,
   * detector and epochs are still different runs if one is pinned and
   * the other follows the live corpus. */
  /** Which of the three sources step 6 trains from.
   *
   * `"live"` is the old name, from when there were two choices and the
   * first meant "the week folders as they stand". Still accepted so a
   * settings file written before the third option existed loads; the
   * page maps it onto `"assembled"`. */
  data_source?: "assembled" | "snapshot" | "pull" | "live";
  /** Whether every run freezes a snapshot first. Absent means true. */
  freeze_on_run?: boolean;
  /** What the pull chain last froze.
   *
   * Separate from `default_dataset`, which is the live slot and which
   * ANY pick overwrites -- including loading a snapshot by hand. The
   * training card offers "the snapshot from the last pull" as its own
   * choice, so that answer needs a slot only the chain writes. */
  last_pull_dataset?: {
    folder?: string; snapshot_id?: string; files?: number;
    set_by?: string; at?: string;
  };
  /** Which settings file these settings came from, or were last written
   * to. Shown on every card that offers Save. */
  settings_file?: string;
  default_dataset?: {
    folder?: string; snapshot_id?: string; files?: number;
    set_by?: string;
    /** When this definition was PUT BACK. */
    at?: string;
    /** When the snapshot was FROZEN -- a different date entirely, and
     * the one that tells one frozen dataset from another. */
    snapshot_created_at?: string;
    snapshot_note?: string;
    /** True when a run will read a pinned file list rather than the week
     * folders. Persisted rather than inferred from the click that caused
     * it: "which images will this train on" is not a question that
     * should expire with a React state variable. */
    pinned?: boolean;
  };
  default_training?: {
    epochs?: number; img_size?: number; batch?: number;
    seed?: number; patience?: number; notes?: string;
    /** The rest of what Ultralytics takes, as overrides only -- a key is
     * absent unless it was changed. Saved here with the other numbers so
     * the training form, "Train on this" on a snapshot, and the scheduled
     * retrain all read ONE set rather than keeping copies that can
     * differ without anything saying so. */
    hyperparams?: Record<string, string>;
    /** Whether those reach the run at all.
     *
     * OFF (the default, and what every file written before this has):
     * nothing is sent and Ultralytics applies its own defaults.
     *
     * ON: every field is sent, the untouched ones at their shown
     * default -- so the run is PINNED to those numbers and the audit
     * records them, rather than inheriting whatever a later Ultralytics
     * decides. */
    hyperparams_enabled?: boolean;
  };
  /** Derived from `picked`; still what the sweep walks. */
  sources: Track2PullSource[];
  /** The four folder lists, which are the real configuration. A source
   * records a prefix and a split but not whether the folder holds images
   * or labels, so sources alone cannot be turned back into these. */
  picked?: {
    train_images: string[];
    train_labels: string[];
    val_images: string[];
    val_labels: string[];
  };
  source_bucket?: string;
  /** Where week folders are written. Blank means the app's own bucket. */
  destination_bucket?: string;
  /** ISO timestamp for when week 1 ENDS. Week 1 is the open-ended
   * backlog; week 2 begins here. Blank means there is only week 1. */
  week_cutoff?: string;
  /** The one root holding every week. Train and val sit inside each
   * week under images/ and labels/. */
  root?: string;
  /** Superseded by `root`; still read from older saved settings. */
  training_root?: string;
  eval_root?: string;
  images_subpath: string;
  labels_subpath: string;
  week_prefix: string;
  week_length_days: number;
  auto_create_weeks: boolean;
  /** "scan" gives one week per sweep -- what a folder edited in place
   * needs, since it has no per-batch event to date anything by.
   * "file_time" dates each file by its own S3 timestamp. */
  /** Which clock decides a file's week.
   *  capture_time -- the moment written into the filename
   *  file_time    -- S3 LastModified, i.e. when it was uploaded
   *  scan         -- one week for the whole sweep */
  week_by?: "scan" | "file_time" | "capture_time";
  /** Week 1 collects everything already in the drop folder. */
  first_week_takes_all: boolean;
  overwrite: boolean;
  mode: string;
  schedule_expression: string;
  /** Days between scheduled pulls. The series is counted from
   * `week_cutoff`, so there is no separate start date. May be longer
   * than a week. */
  schedule_every_days?: number;
  /** "HH:MM" Singapore time a pull lands on, or "" for the cut-off's
   * own time. Moves the clock only -- the dates still come from the
   * cut-off, so this cannot bring a fortnightly pull forward. */
  schedule_time?: string;
  /** "today": every N days from the next occurrence of the time; "date":
   * every N days from `schedule_start_date` (YYYY-MM-DD, SGT). */
  schedule_from?: "today" | "date";
  schedule_start_date?: string;
}

/** What the configured pull interval means, in weeks rather than cron. */
export interface Track2SchedulePlan {
  every_days: number;
  week_length_days: number;
  /** The cut-off the series is counted from. */
  anchor: string;
  /** "" when pulls land on the cut-off's own time. */
  time_of_day: string;
  /** The next run, or "" when no cut-off has been set. */
  start: string;
  /** False when there is no cut-off, so nothing can be armed. */
  ready: boolean;
  next_runs: string[];
  weeks_per_pull: number;
  covers: { week: string; index: number; starts_at: string; ends_at: string;
            open_start?: boolean }[];
  week_by: "scan" | "file_time";
  spans_multiple_weeks: boolean;
  /** One pull covers several weeks AND files are not dated individually
   * -- so they would all land in one folder. */
  misfiling_risk: boolean;
}

export interface Track2ScheduleState {
  exists: boolean;
  enabled: boolean;
  expression?: string;
  timezone?: string;
  start?: string;
  error?: string;
}

export interface Track2PullRun {
  moved: number;
  skipped: number;
  errors: number;
  trigger: string;
  at: string;
  sample_moved?: string[];
}

export interface Track2PullStatus {
  config: Track2PullConfig;
  weeks: Track2PullWeek[];
  next_week_name: string;
  runs: Track2PullRun[];
  triggers: { arrival: Track2PullRule; schedule: Track2PullRule };
  /** False until the stack that owns the table and the rules is applied. */
  ready: boolean;
  error: string;
  /** Whether the unattended router runs the same routing code as this
   * app. They share one source file but deploy separately, so they can
   * drift silently -- each half stays self-consistent and files simply
   * start landing in a shape the card does not describe. */
  router?: {
    state: "current" | "stale" | "unknown";
    detail?: string;
    app_sha?: string;
    router_sha?: string;
    router_modified?: string;
  };
}

export interface Track2PullPlan {
  planned: {
    key: string;
    destination: string;
    week: string;
    new_week: boolean;
    modified: string;
  }[];
  skipped: { key: string; reason: string }[];
  /** Weeks a sweep would have to create first. */
  would_create: string[];
}

export function getTrack2PullControl(): Promise<Track2PullStatus> {
  return apiGet("/track2/pull-control");
}

export function saveTrack2PullConfig(
  patch: Partial<Track2PullConfig>,
): Promise<{ config: Track2PullConfig }> {
  return apiPost("/track2/pull-control/config", patch);
}


export function previewTrack2Pull(): Promise<Track2PullPlan> {
  return apiPost("/track2/pull-control/preview");
}

// ---------------------------------------------------------------------------
// Watching a sweep
//
// A pull is minutes of work, so it runs as a job: start it, poll it, stop
// it. The old synchronous call could report neither progress nor be
// interrupted.
// ---------------------------------------------------------------------------

export interface Track2PullRunState {
  run_id: string;
  /** running | done | stopped | failed | lost. `lost` means it went quiet
   * -- its container was replaced or its Lambda timed out. */
  status: string;
  trigger: string;
  /** Files it will walk. 0 until the counting pass finishes. */
  total: number;
  done: number;
  moved: number;
  skipped: number;
  errors: number;
  started_at?: string;
  updated_at?: string;
  finished_at?: string;
  aborted?: string;
  stop_requested?: boolean;
  quiet_seconds?: number;
  /** Present after a stop: what the undo removed, or why it refused. */
  undone?: {
    created: number;
    removed: number;
    failed: { key: string; error: string }[];
    refused?: string;
  } | null;
}

export interface Track2PendingSummary {
  /** Files that would be copied by the next pull. */
  waiting: number;
  /** Which clock the count was made with. */
  dated_by?: string;
  /** Files whose name carries no timestamp, dated by upload time
   * instead. Non-zero means the source does not use the convention. */
  fell_back_to_upload_time?: number;
  /** Already in the destination, so the pull will skip them. */
  already_filed: number;
  /** Not dataset content, or no week covers them. */
  ignored: number;
  scanned: number;
  /** True when the scan hit its cap, so `waiting` is a lower bound. */
  capped: boolean;
  weeks: { week: string; files: number; bytes: number; exists: boolean }[];
  checked_at: string;
}

export function startTrack2Pull(limit = 20000): Promise<{ run_id: string; status: string }> {
  return apiPost("/track2/pull-control/run", { limit });
}

export function getTrack2PullRun(runId: string): Promise<Track2PullRunState> {
  return apiGet(`/track2/pull-control/run/${encodeURIComponent(runId)}`);
}

export function getTrack2CurrentRun(): Promise<{ run: Track2PullRunState | null }> {
  return apiGet("/track2/pull-control/run");
}

export function stopTrack2Pull(runId: string): Promise<Track2PullRunState> {
  return apiPost(`/track2/pull-control/run/${encodeURIComponent(runId)}/stop`, {});
}

export function getTrack2Pending(): Promise<Track2PendingSummary> {
  return apiGet("/track2/pull-control/pending");
}


export function setTrack2PullMode(
  mode: "off" | "arrival" | "schedule",
  opts: {
    everyDays?: number; expression?: string; timeOfDay?: string;
    from?: "today" | "date"; startDate?: string;
  } = {},
): Promise<{
  mode: string;
  triggers: Track2PullStatus["triggers"];
  schedule: Track2ScheduleState;
  plan: Track2SchedulePlan;
  problems: string[];
}> {
  return apiPost("/track2/pull-control/mode", {
    mode,
    schedule_expression: opts.expression ?? "",
    schedule_every_days: opts.everyDays,
    schedule_time: opts.timeOfDay,
    schedule_from: opts.from,
    schedule_start_date: opts.startDate,
  });
}

/** The PULL timer -- not `getTrack2Schedule`, which is the retraining
 * schedule. Two different clocks, so two different names. */
export function getTrack2PullSchedule(): Promise<{
  schedule: Track2ScheduleState;
  plan: Track2SchedulePlan;
}> {
  return apiGet("/track2/pull-control/schedule");
}

// ---------------------------------------------------------------------------
// Retention -- recency weighting
//
// Older weeks contribute fewer images to an assembled training set. The
// week folders themselves are never touched: this selects a subset at
// assembly time, so any past dataset can be rebuilt exactly.
// ---------------------------------------------------------------------------

export interface Track2RetentionConfig {
  enabled: boolean;
  /** How many of the newest weeks count in full before decay starts.
   * 1 is the original behaviour: only the newest week is untouched. */
  full_weeks: number;
  /** A curated folder that never decays and is always trained on, laid
   * out like a week folder. */
  golden_root: string;
  /** Which bucket that folder is in. Blank is this app's own. */
  golden_bucket: string;
  /** Whether that folder is used. Separate from the path, so the path
   * survives being switched off. */
  use_golden: boolean;
  /** Whether `floor` and `min_images` apply at all. */
  use_floor: boolean;
  /** Fraction lost per week of age. 0.01 is 1%. */
  weekly_rate: number;
  curve: "exponential" | "linear";
  floor: number;
  min_images: number;
  root?: string;
  eval_root?: string;
}

export interface Track2RetentionWeek {
  week: string;
  /** The share that actually survived: floors and rounding included. */
  effective?: number;
  age_weeks: number;
  fraction: number;
  available: number;
  keep: number;
  dropped: number;
  /** This week's validation images. Kept in full, never decayed, and not
   * part of `available` -- which counts the train side only, the same
   * images the trainer's retained list is drawn from. */
  val?: number;
}

export interface Track2RetentionSettings {
  config: Track2RetentionConfig;
  suggested_root: string;
  suggested_eval_root: string;
  /** The decay drawn server-side, so the card cannot disagree with it. */
  curve: Track2RetentionPoint[];
}

export interface Track2RetentionPoint {
  age_weeks: number;
  /** The curve itself. */
  fraction: number;
  /** What actually happens: the curve held up by the floor. */
  effective: number;
}

/** The golden set's contribution, beside the weeks. Always its whole
 * self: it has no collection week, so it has no age, so there is nothing
 * for the decay to act on. */
export interface Track2RetentionGolden {
  enabled: boolean;
  root: string;
  bucket: string;
  available: number;
}

export interface Track2RetentionPlan {
  golden?: Track2RetentionGolden;
  root: string;
  weeks: Track2RetentionWeek[];
  available: number;
  keep: number;
  dropped: number;
  /** Validation images across every week, kept in full. */
  val_available?: number;
  enabled: boolean;
}

export function getTrack2Retention(): Promise<Track2RetentionSettings> {
  return apiGet("/track2/retention");
}

export function saveTrack2Retention(
  patch: Partial<Track2RetentionConfig>,
): Promise<{ config: Track2RetentionConfig }> {
  return apiPost("/track2/retention/config", patch);
}

/** `config` sends the settings as the card currently reads rather than as
 * last saved. Without it the bars answered with the SAVED policy while
 * the controls above them said something else -- switching the curve from
 * compounding to linear left the bars where they were, which reads as the
 * curve making no difference. */
/** A dataset definition frozen hard enough to run again: the yaml, the
 * per-week counts at that moment, and a pointer to an S3 manifest naming
 * every file it selected. The manifest is what makes a run repeatable --
 * a data.yaml can only name DIRECTORIES, and those keep filling. */
export interface Track2DatasetSnapshot {
  snapshot_id: string;
  created_at: string;
  root: string;
  bucket: string;
  note: string;
  yaml_key: string;
  manifest_key: string;
  manifest_uri: string;
  /** Where a run is pointed, and where a restore puts the yaml. */
  download_prefix: string;
  live_key: string;
  weeks: { week: string; images: number; labels: number }[];
  /** The archived file lists. Empty for a folder-list yaml, which trains
   * on whole weeks and has nothing to pin. */
  lists?: Record<string, { key: string; live_key: string; lines: number }>;
  /** How many files a run against this snapshot would read. */
  selected?: number;
  /** "list" -- the selection is pinned by file lists. "folder" -- it is
   * whatever the week folders hold at the time. */
  names_from?: "list" | "folder";
  file_count: number;
  train_images: number;
  val_images: number;
  /** In index order -- what makes an old run explicable after somebody
   * reorders a taxonomy. */
  classes: string[];
  ready: boolean;
  yaml?: string;
}

export interface Track2SnapshotDrift {
  snapshot_id: string;
  root: string;
  file_count_then: number;
  file_count_now: number;
  changes: { week: string; was: number; now: number; new_week: boolean }[];
  /** A re-run WITH the manifest is unaffected either way. This says
   * whether a re-run without it would have differed. */
  unchanged: boolean;
}

export function takeTrack2Snapshot(
  root: string,
  note: string,
): Promise<Track2DatasetSnapshot> {
  return apiPost("/track2/retention/snapshots", { root, note });
}

/**
 * The dataset the auto pull last generated, if there is one.
 *
 * Looks in the config record first, then falls back to the snapshot
 * store -- every chain run froze a snapshot tagged with its trigger, so
 * runs that predate the config field are still found. Without that
 * fallback the option sat greyed while sixteen snapshots from previous
 * pulls were sitting in the table.
 */
export function getTrack2LastPullDataset(): Promise<{
  dataset: {
    folder?: string; snapshot_id?: string; files?: number;
    at?: string; set_by?: string; found_in?: string;
  } | null;
  problem?: string;
}> {
  return apiGet("/track2/retention/last-pull-dataset");
}

export function listTrack2Snapshots(
  root = "",
  limit = 50,
): Promise<{ snapshots: Track2DatasetSnapshot[]; ready: boolean }> {
  const q = new URLSearchParams({ root, limit: String(limit) });
  return apiGet(`/track2/retention/snapshots?${q}`);
}

export function getTrack2Snapshot(id: string): Promise<Track2DatasetSnapshot> {
  return apiGet(`/track2/retention/snapshots/${encodeURIComponent(id)}`);
}

export function getTrack2SnapshotDrift(id: string): Promise<Track2SnapshotDrift> {
  return apiGet(`/track2/retention/snapshots/${encodeURIComponent(id)}/drift`);
}

export function deleteTrack2Snapshot(id: string): Promise<{
  snapshot_id: string;
  root: string;
  /** The artifacts removed. The IMAGES are untouched -- a snapshot is a
   * list of names, never a copy of the corpus. */
  removed: string[];
}> {
  return apiDelete(`/track2/retention/snapshots/${encodeURIComponent(id)}`);
}

/** The unattended retrain. `follow_pull` chains it to the pull's own
 * schedule so it runs AFTER a scheduled sweep finishes, rather than on a
 * clock of its own that could start mid-pull. */
export interface Track2RetrainSchedule {
  enabled: boolean;
  follow_pull: boolean;
  pull_first: boolean;
  every_days: number;
  at: string;
  detector_slug: string;
  architecture: string;
  epochs: number;
  img_size: number;
  batch: number;
  seed: number;
  skip_if_unchanged: boolean;
  /** Which registered version a run continues FROM. READ-ONLY here: it
   * comes from step 1's own choice, so there is one control for it
   * rather than two that can disagree. */
  base_version: string;
}

export interface Track2RetrainStatus {
  settings: Track2RetrainSchedule;
  root: string;
  problems: { level: "blocking" | "warn"; text: string }[];
  /** False when something would stop it firing. Returned whether or not
   * the schedule is on, so "will this work on Sunday" is answerable on
   * Saturday. */
  ready: boolean;
  next_runs: string[];
  expression: string;
  /** Which clock is actually in charge. `follow_pull` is the SETTING;
   * "Use these folders only" overrides it, because that switch turns
   * Pull Control off and a schedule hung off a disabled mechanism never
   * fires. Computed server-side so the rule is not restated here. */
  follows_pull?: boolean;
}

/** One pass of the chain: rebuild, freeze, submit. Each step reports
 * separately, because a chain that stopped half way should say where. */
export interface Track2RetrainRun {
  trigger: string;
  at: string;
  ok: boolean;
  snapshot_id?: string;
  steps: {
    step: string;
    ok: boolean;
    detail: string;
    [key: string]: unknown;
  }[];
}

export function getTrack2Retrain(): Promise<Track2RetrainStatus> {
  return apiGet("/track2/retention/retrain");
}


/** A PATCH -- only the fields sent are changed, so turning the schedule
 * on does not reset the cadence somebody set last week. */
export function saveTrack2Retrain(
  patch: Partial<Pick<Track2RetrainSchedule, "enabled" | "every_days" | "at" | "follow_pull">>,
): Promise<Track2RetrainStatus> {
  return apiPost("/track2/retention/retrain", patch);
}

/** `start` submits directly; `queue` adds to the queue and releases it,
 * the way the schedule does. Defaults to `start` so the Retention card's
 * "Run it now" is unchanged. */
export function runTrack2Retrain(submit: "start" | "queue" | "enqueue" = "start"): Promise<Track2RetrainRun> {
  return apiPost("/track2/retention/retrain/run", { submit });
}

export function restoreTrack2Snapshot(id: string): Promise<{
  snapshot_id: string;
  root: string;
  written_to: string;
  /** Every key put back, lists first and the yaml last. */
  restored: string[];
  download_prefix: string;
  manifest_uri: string;
  selected: number;
  names_from: "list" | "folder";
  /** When the snapshot was FROZEN. Not when it was put back -- those are
   * different dates, and this is the one that tells one frozen dataset
   * from another. */
  created_at?: string;
  note?: string;
  /** True when the restore had to build the two lists from the archived
   * manifest, because the snapshot froze a folder-naming yaml. That is
   * what makes an old snapshot genuinely repeatable rather than "train
   * on whatever those week folders hold today". */
  pinned_from_manifest?: boolean;
}> {
  return apiPost(`/track2/retention/snapshots/${encodeURIComponent(id)}/restore`, {});
}

/** Restore a snapshot AND start a run on it, in one step.
 *
 * Restoring alone only puts the frozen definition back; the run still
 * has to be started from the training card, which is three chances to
 * pick up a different dataset in between. */
export function trainOnTrack2Snapshot(id: string): Promise<{
  restore: { snapshot_id: string; root: string; selected: number;
             names_from: "list" | "folder"; created_at?: string; note?: string;
             pinned_from_manifest?: boolean };
  run: Track2RetrainRun;
}> {
  return apiPost(`/track2/retention/snapshots/${encodeURIComponent(id)}/train`, {});
}

export function previewTrack2Retention(
  root: string,
  config?: Partial<Track2RetentionConfig>,
): Promise<Track2RetentionPlan> {
  return apiPost("/track2/retention/preview", { root, config });
}

/** One retrain's row: how much of each earlier week it trains on. */
export interface Track2RetentionMatrixRow {
  week: string;
  deployed_index: number;
  /** 1 when a golden set is configured, null when there is none. It never
   * decays, so it is the same in every row. */
  golden: number | null;
  /** False for the rows past the weeks that exist -- retrains that have
   * not happened yet. */
  existing: boolean;
  cells: {
    week: string;
    age_weeks: number;
    steps: number;
    /** The curve on its own. */
    fraction: number;
    /** What actually survives -- the curve held up by the percentage
     * floor. This is the number to show: the grid used to print
     * `fraction` and call it the answer, so a policy keeping at least
     * 20% of every week displayed 2.81%. */
    effective: number;
    /** True where a floor CHANGED the answer. Not merely where the
     * share came out above the curve -- rounding does that too, and on a
     * small week it looks identical from the outside. */
    floored: boolean;
    /** Holds fewer images than the minimum, so it is kept whole however
     * old it gets. A week of 8 under a 25-image minimum never decays. */
    under_minimum: boolean;
    /** What that week holds, and what survives, as counts. */
    available: number;
    keep: number;
    /** False where `available` is the assumed size rather than a real
     * one -- a week nobody has collected yet. A week that exists and
     * holds nothing is a counted zero, so it stays true. */
    real: boolean;
    /** Exists, and holds nothing. */
    empty: boolean;
  }[];
}

export interface Track2RetentionMatrix {
  weeks: string[];
  existing: number;
  ahead: number;
  curve: "exponential" | "linear";
  other_curve: "exponential" | "linear";
  /** The largest difference, as a FRACTION, between the chosen curve and
   * the other one anywhere in this grid. At 1% a week the two differ by
   * 0.002 -- two tenths of a point -- which whole-percent cells round
   * away entirely, so switching curve appeared to do nothing. The card
   * uses this to decide how many decimals a cell needs. */
  curve_gap: number;
  /** The percentage floor in force, as a fraction. Zero when switched off. */
  floor: number;
  use_floor: boolean;
  /** The image minimum. Applied whenever `sized` is true; without real
   * week sizes only the percentage floor can be, since this one depends
   * on how big a week is. */
  min_images: number;
  /** Whether any real week sizes were in hand. Without them only the
   * percentage floor is applied, and counts cannot be shown. */
  sized: boolean;
  /** The size assumed for weeks not collected yet: the median of the
   * ones that exist. */
  assumed_size: number;
  golden_available: number;
  golden: boolean;
  golden_root: string;
  golden_bucket: string;
  enabled: boolean;
  full_weeks: number;
  root: string;
  rows: Track2RetentionMatrixRow[];
}

/** Every retrain against every week of data, as one grid.
 *
 * `config` tries the settings as the card currently reads rather than as
 * last saved, so the grid redraws while the controls move -- the point of
 * it is to be looked at BEFORE the policy is committed. */
export function track2RetentionMatrix(
  root: string,
  ahead: number,
  config: Partial<Track2RetentionConfig>,
): Promise<Track2RetentionMatrix> {
  return apiPost("/track2/retention/matrix", { root, ahead, config });
}


/** Which image this container runs, and whether it is the stable one or a
 * development one. Two may run side by side against the same data, so the
 * difference has to be visible rather than remembered. */
export interface Track2Build {
  tag: string;
  role: string;
  port: string;
}

export function getTrack2Build(): Promise<Track2Build> {
  return apiGet("/build");
}

/** A bucket the account owns. `readable` is false when its NAME is
 * visible but its contents are not -- bucket listing is account-wide,
 * object access is not. */
export interface Track2Bucket {
  name: string;
  readable: boolean;
}

export function getTrack2Buckets(): Promise<{
  buckets: Track2Bucket[];
  default: string;
}> {
  return apiGet("/track2/pull-control/buckets");
}

/** One level of folders under a prefix, in any readable bucket. */
/** One queued sweep, with the settings it will run under.
 *
 * The snapshot is what makes an entry a piece of work rather than a
 * repeat: a sweep reads the live config and moves whatever is waiting, so
 * two entries that both meant "sweep now" would be the same instruction
 * twice and the second would find an empty folder. */
export interface Track2PullQueueEntry {
  id: string;
  added_at: string;
  added_by: string;
  note: string;
  /** One line saying what this entry sweeps, built server-side. */
  summary: string;
  config: Record<string, unknown>;
}

export function getTrack2PullQueue(): Promise<{
  entries: Track2PullQueueEntry[];
  running: Track2PullRunState | null;
}> {
  return apiGet("/track2/pull-control/queue");
}

export function queueTrack2Pull(note = ""): Promise<{
  entry: Track2PullQueueEntry; queued: number;
}> {
  // No config sent: the server snapshots the live one at queue time, so
  // there is one place that decides what "the settings as they stand"
  // means rather than the browser sending its own idea of them.
  return apiPost("/track2/pull-control/queue/add", { note });
}

export function removeTrack2PullQueueEntry(entryId: string): Promise<{
  removed: string; queued: number;
}> {
  return apiPost(`/track2/pull-control/queue/remove/${encodeURIComponent(entryId)}`);
}

export function clearTrack2PullQueue(): Promise<{ removed: number }> {
  return apiPost("/track2/pull-control/queue/clear");
}

export function advanceTrack2PullQueue(): Promise<{
  started: boolean; reason?: string; run_id?: string; remaining?: number;
}> {
  return apiPost("/track2/pull-control/queue/advance");
}

/** What deleting a folder would remove, and which settings still use it. */
export type Track2FolderDeleteCheck = {
  bucket: string;
  prefix: string;
  objects: number;
  bytes: number;
  in_use: string[];
};

/** Asked twice: without `confirm` this only counts. A folder a setting
 * still uses is refused with 409 until `force` is sent too. */
export function checkTrack2FolderDelete(
  bucket: string, prefix: string,
): Promise<Track2FolderDeleteCheck> {
  return apiPost("/track2/pull-control/folders/delete", { bucket, prefix });
}

export function deleteTrack2Folder(
  bucket: string, prefix: string, force = false,
): Promise<{ bucket: string; prefix: string; deleted: number; errors: string[] }> {
  return apiPost("/track2/pull-control/folders/delete",
    { bucket, prefix, confirm: true, force });
}

/** One run a clock will start: a scheduled pull, or a training run. */
export interface Track2ScheduledEvent {
  at: string;
  kind: "pull" | "train";
  label: string;
  /** What a training run builds its yaml from: Pull, Golden set, Assembled folders. */
  sources: string[];
  /** The settings file this event belongs to. */
  file: string;
  /** True for the loaded configuration -- the only one whose clocks fire. */
  live: boolean;
}

/** One settings file's schedule summary. */
export interface Track2ScheduledFile {
  name: string;
  live: boolean;
  error: string;
  sources: string[];
  next: Track2ScheduledEvent | null;
  notes: string[];
}

export interface Track2Scheduled {
  settings_file: string;
  now: string;
  until: string;
  next: Track2ScheduledEvent | null;
  next_pull: Track2ScheduledEvent | null;
  sources: string[];
  events: Track2ScheduledEvent[];
  notes: string[];
  /** The live configuration first, then each saved file asked for. */
  files: Track2ScheduledFile[];
  /** Every saved settings file, for the picker. */
  available: string[];
}

/** `files`: saved files to show beside the live one, or "*" for all. */
export function getTrack2Scheduled(days = 42, files: string[] | "*" = []): Promise<Track2Scheduled> {
  const list = files === "*" ? "*" : files.join(",");
  return apiGet(`/track2/retention/scheduled?days=${days}&files=${encodeURIComponent(list)}`);
}

/** A frozen dataset a run was queued or started on (dataset_versions.py). */
export interface Track2DatasetVersion {
  version_id: string;
  created_at: string;
  trigger?: string;
  sources?: string[];
  classes?: string[];
  train_images?: number;
  val_images?: number;
  drop_undeclared?: boolean;
  state?: string;
  entry_id?: string;
  run_id?: string;
  data_yaml_uri?: string;
  entry_template?: Record<string, unknown>;
}

export function listTrack2DatasetVersions(limit = 30): Promise<{ versions: Track2DatasetVersion[] }> {
  return apiGet(`/track2/retention/versions?limit=${limit}`);
}

export function getTrack2DatasetVersion(
  id: string,
): Promise<Track2DatasetVersion & { data_yaml: string; eval_yaml: string }> {
  return apiGet(`/track2/retention/versions/${encodeURIComponent(id)}`);
}

export function queueTrack2DatasetVersion(id: string): Promise<{ queued: string; version_id: string }> {
  return apiPost(`/track2/retention/versions/${encodeURIComponent(id)}/queue`);
}

export function listTrack2Folders(
  bucket: string,
  prefix: string,
): Promise<{ bucket: string; prefix: string; folders: string[] }> {
  return apiPost("/track2/pull-control/folders", { bucket, prefix });
}


/** Deletes a week's folders AND forgets the week. Removes data.
 *
 * `confirmObjects` must be the count the preview reported for THIS week.
 * The server re-counts and refuses with a 409 if it no longer matches --
 * the same rule the all-weeks purge has always followed. Before this,
 * the API took a name and nothing else, so one request deleted a week. */
export function purgeTrack2Week(name: string, confirmObjects: number): Promise<{
  week: string; deleted: number; roots: string[];
}> {
  return apiPost("/track2/pull-control/weeks/purge",
    { name, confirm_objects: confirmObjects });
}

export interface Track2SettingsFile {
  name: string;
  bytes: number;
  modified: string;
}

/** Blank template: puts the LIVE settings back to the shipped defaults.
 *
 * Not a form reset. Every card reads the live config, so clearing only
 * what is on screen would leave the old settings running underneath.
 * Saved files are untouched -- this starts a new one, it does not throw
 * the old ones away. */
export function newTrack2Settings(): Promise<{
  config: Record<string, unknown>;
  problems: string[];
}> {
  return apiPost("/track2/pull-control/settings/new");
}

/** One stage of the unattended chain, as the status bar draws it.
 *
 * `pending` stages are included deliberately: a progress display that
 * only lists steps already taken cannot show what is still to come,
 * which is most of what somebody watching a twenty-minute chain wants. */
export interface Track2ChainStage {
  step: string;
  label: string;
  state: "pending" | "running" | "ok" | "failed";
  detail: string;
}

export interface Track2ChainStatus {
  running: boolean;
  trigger: string;
  started_at: string;
  finished_at: string;
  ok: boolean | null;
  stages: Track2ChainStage[];
  /** The sweep running now, or the last one that finished. Carried on
   * this response rather than polled separately because a pull can run
   * with NO chain behind it, and that case had nowhere on screen to
   * appear. `chains` says whether anything follows it. */
  pull: {
    run_id: string;
    status: string;
    trigger: string;
    done: number;
    total: number;
    moved: number;
    skipped: number;
    errors: number;
    started_at: string;
    finished_at: string;
    chains: boolean;
  };
  gpu: {
    known: boolean;
    busy: boolean;
    /** True when the job holding the slot is one this app started. The
     * account has a single training slot, so "somebody else is using it"
     * and "our run is going" need different answers. */
    ours?: boolean;
    /** The SAME shape `get_gpu_resource_usage` returns, not a narrower
     * copy of it. The copy left out `spot`, so anything reading this
     * list could not tell a managed-spot job -- which draws on a
     * different quota and blocks nothing -- from one actually holding
     * the slot. */
    jobs: Track2GpuJob[];
    error: string;
  };
  /** The PROCESSING half of a run. Evaluation runs on an ml.m5.xlarge
   * after training releases the GPU, so for the seven or eight minutes
   * it takes, "a run is going" and "the GPU is free" are both true. This
   * is what makes that legible instead of contradictory. */
  cpu: {
    known: boolean;
    busy: boolean;
    ours?: boolean;
    /** The pipeline step, read off the job name: "EvaluateYolo". */
    step: string;
    jobs: { name: string; step: string; instance_type: string; own_pipeline: boolean }[];
    error: string;
  };
}

export function getTrack2ChainStatus(): Promise<Track2ChainStatus> {
  return apiGet("/track2/retention/chain-status");
}

/** What a saved settings file already trains out of.
 *
 * Two files on one root is not an error -- the same week folders CAN
 * feed two Classes -- but it is almost never meant: the pull writes into
 * that root, retention prunes it and the chain trains from it, so the
 * second file inherits the first one's data and the first one's pulls
 * reshape the second one's corpus. */
export interface Track2SettingsClaim {
  name: string;
  root: string;
  model: string;
  assembled: string;
}

export function getTrack2SettingsFiles(): Promise<{
  directory: string;
  files: Track2SettingsFile[];
  writable: boolean;
  claims?: Track2SettingsClaim[];
}> {
  return apiGet("/track2/pull-control/settings");
}

/** Writes the live settings to a file on the EC2 host, which survives the
 * container being replaced by a deploy. */
/** `allowClash` saves past the "another file already writes there"
 * refusal, which comes back as a 409. It is a warning worth reading, not
 * a rule the format depends on -- two configs filing into one set of week
 * folders is usually a mistake and occasionally the point. */
export function saveTrack2SettingsFile(name: string, allowClash = false): Promise<{
  name: string; path: string; bytes: number;
}> {
  return apiPost("/track2/pull-control/settings/save", { name, allow_clash: allowClash });
}

/** `apply: false` reads the file WITHOUT making it live -- the card fills
 * its controls from it and nothing changes until Apply settings. */
export function loadTrack2SettingsFile(name: string, apply = false): Promise<{
  config: Track2PullConfig;
  name: string; saved_at: string; applied: boolean;
  /** What was actually restored: how many sources the picks rebuilt to,
   * which trigger was re-applied, and anything that did not stick. */
  mode: string; sources: number; problems: string[];
  /** What the class and model registry restore did. CREATE MISSING
   * ONLY -- `kept` is every detector the file also named that this
   * machine already had and that was therefore left untouched. */
  registry?: { created: string[]; kept: string[]; problems: string[] };
}> {
  return apiPost("/track2/pull-control/settings/load", { name, apply });
}

/** How well a set of image folders lines up with a set of label folders.
 * Matched by stem, which is how Ultralytics finds a label for an image. */
export interface Track2Pairing {
  images: number;
  labels: number;
  paired: number;
  images_without_labels: number;
  labels_without_images: number;
  sample_images_without_labels: string[];
  sample_labels_without_images: string[];
  checked: boolean;
}

export function checkTrack2Pairing(
  bucket: string,
  imagePrefixes: string[],
  labelPrefixes: string[],
): Promise<Track2Pairing> {
  return apiPost("/track2/pull-control/pairing", {
    bucket, image_prefixes: imagePrefixes, label_prefixes: labelPrefixes,
  });
}

/** Removes a saved settings file from the EC2 host. */
export function deleteTrack2SettingsFile(name: string): Promise<{
  name: string; deleted: boolean;
}> {
  return apiPost("/track2/pull-control/settings/delete", { name });
}

/** One week in the forecast: filled, being filled, or still to come. */
export interface Track2ForecastWeek {
  week: string;
  images: number;
  labels: number;
  bytes: number;
  roots: string[];
  /** False for a week that has not been created yet. */
  exists: boolean;
  starts_at: string;
  ends_at: string;
  /** Week 1 reaches back forever, so it has no start. */
  open_start?: boolean;
  /** When this week stops accepting files. For week 1 this IS the
   * cut-off; for every later week it is its own boundary. */
  closes_at: string;
  /** True only for week 1, whose closing moment is the cut-off. */
  is_cutoff: boolean;
  /** filled  -- the folder holds files
   *  empty   -- the folder exists and holds none
   *  missing -- a past week whose folder was never created
   *  current -- being filled now
   *  future  -- has not happened yet */
  state: "filled" | "empty" | "missing" | "current" | "future";
}

/** Every week, dated.
 *
 * `opts` re-dates the table under settings chosen but not applied, so a
 * change to the week length shows its effect before it is saved. Only
 * the dates move -- the counts are whatever is in the folders, since
 * changing the week length moves no files. */
export function getTrack2Forecast(opts: {
  weekLengthDays?: number;
  weekCutoff?: string;
  weekPrefix?: string;
  trainingRoot?: string;
  evalRoot?: string;
  /** A saved settings file to VIEW. Applies nothing. */
  settings?: string;
} = {}): Promise<{
  weeks: Track2ForecastWeek[];
  /** The week the clock is in. */
  current: string;
  /** The week the next pull will actually file into, which is not the
   * same thing while the week 1 backlog is still uncollected. */
  active: string;
  cutoff: string;
  week_length_days: number;
  roots: string[];
  now: string;
  /** Changes only when a count changes, so the card can poll cheaply
   * and redraw only when something has actually landed. */
  stamp: string;
  total_images: number;
  total_labels: number;
  schedule: Track2SchedulePlan;
  /** True when the dates are for unapplied settings. */
  previewing: boolean;
  /** Week folders that already exist -- what a length change re-dates. */
  existing_weeks: number;
}> {
  const query = new URLSearchParams();
  if (opts.weekLengthDays !== undefined) {
    query.set("week_length_days", String(opts.weekLengthDays));
  }
  if (opts.weekCutoff !== undefined) query.set("week_cutoff", opts.weekCutoff);
  if (opts.weekPrefix) query.set("week_prefix", opts.weekPrefix);
  if (opts.trainingRoot) query.set("training_root", opts.trainingRoot);
  if (opts.evalRoot) query.set("eval_root", opts.evalRoot);
  if (opts.settings) query.set("settings", opts.settings);
  const tail = query.toString();
  return apiGet(`/track2/pull-control/forecast${tail ? `?${tail}` : ""}`);
}


// ---------------------------------------------------------------------------
// data.yaml over the week folders
//
// Points a trainer at the weeks in place. No folder is created and no
// image is copied -- an Ultralytics `train:` takes a list of directories.
// ---------------------------------------------------------------------------

export interface Track2YamlWeek {
  week: string;
  images: number;
  labels: number;
  bytes: number;
  images_path: string;
  labels_path: string;
  /** Images with no labels beside them: every one would train as a
   * background example. */
  unlabelled: boolean;
}

export interface Track2DatasetYaml {
  /** The file, as it would be written. */
  yaml: string;
  document: Record<string, unknown>;
  /** The S3 prefix a trainer has to download for these paths to resolve. */
  download_prefix: string;
  /** Where the file goes, relative to the bucket. */
  write_to: string;
  bucket: string;
  train_weeks: Track2YamlWeek[];
  val_weeks: Track2YamlWeek[];
  train_images: number;
  val_images: number;
  classes: { names: Record<string, string>; source: string; looked_in: string[] };
  problems: { level: "blocking" | "warn"; text: string }[];
  /** Present when the retention policy was applied as a FILE LIST.
   * Without it a folder list trains on every image in each week whatever
   * the decay says, so the policy reaches nothing. */
  retained?: {
    train: number;
    val: number;
    train_available: number;
    val_available: number;
    weeks: {
      week: string; available: number; keep: number;
      age_weeks: number; fraction: number; val: number;
    }[];
    written?: Record<string, { key: string; lines: number }>;
  };
  /** False when something would stop it training at all. */
  ready: boolean;
  written?: { bucket: string; key: string; uri: string };
}

export interface Track2DatasetYamlRequest {
  /** Write `train:`/`val:` as file lists holding exactly what the
   * retention policy chose, rather than as the week folders whole.
   * Ultralytics reads a split entry that is a FILE as a list of image
   * paths, which is what lets the policy reach the run at all. */
  apply_retention?: boolean;
  /** The one root holding every week, with train/ and val/ inside. */
  root?: string;
  /** Superseded by `root`; two separate roots, one per split. */
  training_root?: string;
  eval_root?: string;
  bucket?: string;
  images_subpath?: string;
  labels_subpath?: string;
  class_sources?: string[];
  names?: Record<string, string>;
}

export function buildTrack2DatasetYaml(
  body: Track2DatasetYamlRequest = {},
): Promise<Track2DatasetYaml> {
  return apiPost("/track2/retention/dataset-yaml", body);
}

export function saveTrack2DatasetYaml(
  body: Track2DatasetYamlRequest & { key?: string } = {},
): Promise<Track2DatasetYaml> {
  return apiPost("/track2/retention/dataset-yaml/save", body);
}


// ---------------------------------------------------------------------------
// Splitting a curated folder into train and val
//
// The folder is one somebody chose by hand. Nothing here decides what
// belongs in it -- only how it is divided, and that each image stays with
// its own label.
// ---------------------------------------------------------------------------

export interface Track2SplitPlan {
  source: string;
  bucket: string;
  target_bucket: string;
  target_folder: string;
  /** "images/ + labels/" or "flat" -- which shape was found. */
  layout: string;
  split: boolean;
  train_percent: number;
  /** Image/label pairs. The slider divides THESE, not files. */
  pairs: number;
  train: number;
  val: number;
  images: number;
  labels: number;
  /** Images with no label: valid background examples, ignored here. */
  unpaired_images: number;
  /** Labels with no image: always a mistake. */
  orphan_labels: number;
  files_to_copy: number;
  sample: { stem: string; side: string; image: string; label: string }[];
  problems: { level: "blocking" | "warn"; text: string }[];
  ready: boolean;
}

/** What a BUILD returns: a plan, plus what it actually did.
 *
 * Its own type rather than optional fields on the plan. A preview has no
 * `copied` because it copied nothing, and a build always does -- making
 * them optional on one shared type meant every read needed a guard for a
 * case that cannot happen.
 */
export interface Track2SplitResult extends Track2SplitPlan {
  copied: number;
  errors: { key: string; error: string }[];
  /** Where the record of which stems went where was written. */
  manifest: string;
  uri: string;
}

export interface Track2SplitRequest {
  source: string;
  /** Where the LABELS are, when they do not live under `source`.
   *
   * Omitted keeps the two shapes the split always accepted -- a folder
   * holding `images/` and `labels/`, and a flat folder with both side by
   * side -- so an existing settings file splits exactly as it did. */
  labels_source?: string;
  bucket?: string;
  target_bucket?: string;
  target_folder: string;
  split?: boolean;
  train_percent?: number;
}

export function previewTrack2Split(body: Track2SplitRequest): Promise<Track2SplitPlan> {
  return apiPost("/track2/pull-control/split/preview", body);
}

export function buildTrack2Split(body: Track2SplitRequest): Promise<Track2SplitResult> {
  return apiPost("/track2/pull-control/split/build", body);
}


/** Creates a folder for each empty week between weeks that hold files. */
export function fillTrack2WeekGaps(): Promise<{ created: string[]; keys: number }> {
  return apiPost("/track2/pull-control/weeks/fill-gaps", {});
}


// ---------------------------------------------------------------------------
// Track2 client functions with no caller today.
//
// Every one of these hits a route that is still live and still in
// app.openapi(). They lost their callers when a card was rebuilt, not
// when the judge was retired -- the Assemble card, for instance, moved
// to writeTrack2DatasetYaml and left assembleTrack2Dataset behind. Kept
// deliberately: deleting a working client for a working endpoint is a
// Track2 decision, and this pass was about the judge.
// ---------------------------------------------------------------------------

export function apiPut<T>(path: string, body?: unknown): Promise<T> {
  return request<T>(path, {
    method: "PUT",
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
}

/**
 * Fetches a path that returns raw image bytes (every `.../image` endpoint)
 * as a Blob. Plain `<img src=...>` can't carry the `Authorization` header
 * these endpoints require, so any caller needing the bytes (to display,
 * or to re-upload as part of a later POST) goes through this.
 */
export async function apiGetImageBlob(path: string): Promise<Blob> {
  const headers = new Headers();
  if (authToken) headers.set("Authorization", `Bearer ${authToken}`);
  const res = await fetch(`${API_BASE_URL}${path}`, { headers });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body?.detail ?? detail;
    } catch {
      // non-JSON error body -- fall back to statusText
    }
    throw new ApiError(res.status, detail);
  }
  return await res.blob();
}

/** Same as `apiGetImageBlob`, wrapped as an object URL for `<img src>` --
 * see `useApiImage`, which also owns revoking it. */
export async function apiGetImageUrl(path: string): Promise<string> {
  const blob = await apiGetImageBlob(path);
  return URL.createObjectURL(blob);
}

export interface Track2RegistryPackage {
  model_package_arn: string;
  version: string;
  approval_status: string;
  description?: string;
  lineage?: Track2Lineage | null;
  creation_time?: string | null;
}

export interface Track2RegistrySummary {
  total: number;
  approved: number;
  packages: Track2RegistryPackage[];
}

export type Track2Registries = Record<string, Track2RegistrySummary>;

export interface Track2ReviewResponse {
  counts: { pending: number; approved: number; rejected: number };
  registries: Track2Registries;
}

export function getTrack2Review(): Promise<Track2ReviewResponse> {
  return apiGet<Track2ReviewResponse>("/track2/review");
}

export function approveTrack2Review(reviewId: string): Promise<{ review_id: string; status: string }> {
  return apiPost(`/track2/review/${encodeURIComponent(reviewId)}/approve`);
}

export function rejectTrack2Review(reviewId: string): Promise<{ review_id: string; status: string }> {
  return apiPost(`/track2/review/${encodeURIComponent(reviewId)}/reject`);
}

export function approveTrack2Model(modelPackageArn: string): Promise<{ model_package_arn: string; approval_status: string }> {
  return apiPost(`/track2/review/model/${encodeURIComponent(modelPackageArn)}/approve`);
}

export function rejectTrack2Model(modelPackageArn: string): Promise<{ model_package_arn: string; approval_status: string }> {
  return apiPost(`/track2/review/model/${encodeURIComponent(modelPackageArn)}/reject`);
}

export function startTrack2Training(params: Track2TrainingParams): Promise<{ message: string; build_id: string }> {
  return apiPost("/track2/models/train/dataset", params);
}

/** Trains on the dataset's own classes, through a separate pipeline and
 * registry group -- a model whose classes are not the production
 * person/helmet pair must not reach the production lineage. Requires
 * `base_model_folder`: the trainer fine-tunes from a checkpoint. */
export function startTrack2CustomTraining(
  params: Track2TrainingParams,
): Promise<{ message: string; run_id: string; execution_arn: string; audit_id: string }> {
  return apiPost("/track2/models/train/custom", params);
}


// --- the SQL tab (2026-10-28) -------------------------------------------

/** A Postgres database the SQL tab can query, and whether it is up. */
export interface Track2SqlDatabase {
  key: string;
  label: string;
  status: string;
  available: boolean;
}

export interface Track2SqlTable {
  name: string;
  columns: { name: string; type: string }[];
}

/** One image found in the result, narrowed by the chosen bucket and folder. */
export interface Track2SqlImage {
  uri: string;
  bucket: string;
  key: string;
  /** The column it was found in, and the row. */
  column: string;
  row: number;
}

export interface Track2SqlResult {
  columns: string[];
  rows: unknown[][];
  /** Cut short near the 6 MB a query function may return. */
  truncated: boolean;
  images: Track2SqlImage[];
}

export function getTrack2SqlDatabases(): Promise<{ databases: Track2SqlDatabase[] }> {
  return apiGet("/track2/sql/databases");
}

export function getTrack2SqlSchema(database: string): Promise<{ tables: Track2SqlTable[] }> {
  return apiPost("/track2/sql/schema", { database });
}

export function runTrack2Sql(
  database: string,
  sql: string,
  bucket: string,
  folder: string,
): Promise<Track2SqlResult> {
  return apiPost("/track2/sql/query", { database, sql, bucket, folder });
}
